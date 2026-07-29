# Rollout 模块知识笔记

本文档记录对 `slime/ray/rollout.py` 中 rollout 子系统的源码解读，涵盖：

1. `RolloutManager` 的整体作用
2. `start_rollout_servers` 的详细工作流程
3. 为什么 rollout 引擎的通信走 HTTP，而不是让 Ray 直接调用引擎
4. `rollout_engine_lock` 的工作机制
5. model / server_group / engine 三层结构详解（是否支持多 model、`SGLangEngine` 对应哪一层）
6. sglang server 是如何与 Ray placement group（GPU 资源）绑定起来的
7. slime 里出现的两种"router"的区别（sgl-model-gateway 网关 vs 每个 engine 自己的 HTTP server）

---

## 1. `RolloutManager` 的作用

`RolloutManager`（`slime/ray/rollout.py:427`）是一个 **Ray remote actor**，是 slime 训练框架中负责**执行 rollout（推理采样）并将结果转换为训练数据**的核心调度类。

### 1.1 `__init__` 主要做的事

```python
@ray.remote
class RolloutManager:
    def __init__(self, args, pg):
        ...
```

1. **启动/管理 rollout 推理服务器**
   - 若 `args.debug_train_only`（只调试训练，不需要真实推理），`self.servers` 为空字典。
   - 否则调用 `init_http_client(args)` 初始化 HTTP 客户端，并通过 `start_rollout_servers(args, pg)` 在给定的 Ray placement group `pg` 上拉起 SGLang 推理引擎集群。`self.servers` 是 `model_name -> RolloutServer` 的映射；每个 `RolloutServer` 内部可能包含多个 `ServerGroup`（对应 prefill/decode 等不同角色的引擎组）。此步骤只发起初始化（非阻塞），返回的 `rollout_init_handles` 稍后统一 `ray.get` 等待。

2. **加载数据源与用户自定义函数（插件式扩展点）**
   - `self.data_source`：通过 `load_function` 动态加载数据集类并实例化。
   - `self.generate_rollout` / `self.eval_generate_rollout`：训练时/评测时用于生成 rollout 的函数。
   - `self.custom_reward_post_process_func`、`self.custom_convert_samples_to_train_data_func`：可选的自定义奖励后处理、样本转训练数据函数。

3. **等待引擎初始化完成 & 初始化监控**
   - `ray.get(rollout_init_handles)` 阻塞等待所有推理引擎真正初始化完毕。
   - `init_tracking(args, primary=False)` 初始化日志/监控（wandb 等）。
   - `self.rollout_engine_lock` 创建一个 Ray actor 锁，用于协调权重更新时对推理引擎的并发访问（详见第 4 节）。
   - `self.rollout_id = -1` 记录当前 rollout 步数。

4. **启动容错健康监控**
   - 若 `args.use_fault_tolerance`，为每个 server 的每个 `server_group` 创建并启动 `RolloutHealthMonitor`，用于检测引擎是否存活（死掉后可通过 `recover_updatable_engines` 重启）。
   - `_ci_fault_injection_pending`：CI 测试专用标志，用于模拟引擎故障以验证容错逻辑。

### 1.2 整体角色

结合类中其他方法：

- `generate(rollout_id)` / `eval(rollout_id)`：驱动一次 rollout 生成/评测，调用 `self.generate_rollout`，再通过 `_convert_samples_to_train_data` 和 `_split_train_data_by_dp` 把生成的样本转换为按 DP rank 切分好的训练 batch。
- `offload/onload/onload_weights/onload_kv`：管理推理引擎显存的释放与恢复（训练与推理共享 GPU 时的资源调度）。
- `recover_updatable_engines`：故障后重启需要更新权重的推理引擎。
- `get_updatable_engines_and_lock`：暴露可更新权重的引擎和锁，供权重同步（training → rollout）逻辑使用。

**一句话总结**：`RolloutManager` 是训练循环中"生成侧"的总控 actor，负责拉起并维护 SGLang 推理集群、驱动 rollout 采样、计算/处理 reward、把采样结果打包成可供 Megatron 训练侧消费的数据，并处理推理引擎的显存 offload 和故障恢复。

---

## 2. `start_rollout_servers` 详解

```python
def start_rollout_servers(args, pg) -> tuple[dict[str, Any], list[Any]]:
    ...
    for model_idx, model_cfg in enumerate(config.models):
        ...
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))
        ...
        for group_cfg in model_cfg.server_groups:
            group = _make_group(group_cfg, router_ip, router_port)
            handles, port_cursors = group.start_engines(port_cursors)
            ...
        servers[model_cfg.name] = RolloutServer(server_groups=server_groups, router_ip=router_ip, ...)
    return servers, pending_init_handles
```

它是"拉起整套推理服务"的编排函数，按 **model → server_group → engine** 三层展开：

1. **解析配置**（`_resolve_sglang_config`）：把 `args`（或 `--sglang-config` YAML）转成 `SglangConfig`，得到每个 model 需要多少 `ServerGroupConfig`（regular / prefill / decode / encoder / placeholder），各自的 TP size、GPU 数。
2. **为每个 model 启动一个 router**（`_start_router`）：router 是一个独立的 HTTP 网关进程，每个 model 各自一个 router 实例（`router_ip:router_port`）。
3. **为每个 server_group 拉起 Ray actor 池**（`ServerGroup.start_engines` → `RolloutRayActor = ray.remote(SGLangEngine)`）：每个 `SGLangEngine` actor 对应一份 GPU 资源，`.remote(...)` 后立即返回，`init` 调用也是 `.remote()` 非阻塞的，返回值收集到 `pending_init_handles`，交给调用方（`RolloutManager.__init__`）稍后统一 `ray.get()` 等待。这样可以和"加载 data_source / 用户自定义函数"等其他初始化步骤重叠执行，缩短启动耗时。
4. **端口分配**（`_allocate_rollout_engine_addr_and_ports_normal`）：为每个引擎分配 server port / nccl port / dist_init_addr port，用 `node_port_cursor` 避免同节点多个 server_group 抢同一批端口。
5. **EPD（Encoder-Prefill-Decode）两阶段**：如果开启了 encoder 分离，先同步拉起并等待 encoder 组，拿到它们的 URL，再把这些 URL 作为 `encoder_urls` overrides 注入到 prefill/regular 组里，最后才异步拉起非-encoder 组。
6. **收尾**：把 `router_ip/router_port` 写回 `args.sglang_model_routers`，供自定义 rollout 函数使用；返回 `servers: dict[model_name, RolloutServer]` 和 `pending_init_handles`。

---

## 3. 为什么用 HTTP 通信，而不是让 Ray 直接调用 rollout engine

### 3.1 每个 `SGLangEngine` 到底是什么

关键点：**`SGLangEngine`（Ray actor）本身不是推理引擎，它只是一个"进程启动器 + HTTP 客户端适配器"**。

```python
def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    ...
    from sglang.srt.entrypoints.http_server import launch_server
    multiprocessing.set_start_method("spawn", force=True)
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()
    ...
```

它用 `multiprocessing.Process` 启动了一个**独立的操作系统进程**，运行的是 sglang 自己的 `launch_server`（`sglang/python/sglang/srt/entrypoints/http_server.py`）。该函数文档字符串写得很清楚：

> The SRT server consists of an HTTP server and an SRT engine.
> - HTTP server: A FastAPI server that routes requests to the engine.
> - The engine consists of three components:
>   1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
>   2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
>   3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.
>
> Note: Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.

也就是说，真正跑推理的 Scheduler / DetokenizerManager 是**另外的子进程**，进程间用 **ZMQ IPC** 通信，Ray 完全不参与这条链路。sglang 对外暴露的唯一稳定"接口契约"就是 FastAPI 起的这个 HTTP 服务（`/generate`、`/health_generate`、`/update_weights_from_tensor`、`/release_memory_occupation` 等几十个 endpoint）。

所以 `SGLangEngine._make_request` / `health_generate` / `update_weights_from_disk` 等方法本质上都是 `requests.post/get(f"http://{host}:{port}/{endpoint}", ...)`——**不是"不想"直接调引擎，而是"没有别的方式"直接调**：sglang 引擎本身就没有暴露"进程内 Python 方法"这种接口，它的架构就是 HTTP-server-in-front-of-subprocess-engine，Ray actor 也无法穿透到它内部的 ZMQ IPC 去做函数调用。

### 3.2 为什么真正的生成请求（海量并发采样）走 HTTP router，而不走 Ray

`RolloutManager` 里 `self.servers` 只是 engine 的"管理句柄"（用于 offload/onload/权重更新/健康检查），**真正的 `generate` 采样请求根本不经过这些 Ray actor**，而是 rollout 函数直接对 router 的 HTTP 地址发请求（`args.sglang_model_routers`），由 `slime/utils/http_utils.py` 的 `post()`/`get()`（基于 `httpx.AsyncClient` 连接池）发出。

Router 在 sglang 里叫 `sgl-model-gateway`（即 `sglang_router` 这个 pip 包背后的实现），是一个用 **Rust** 编写的独立 HTTP 网关，功能非常完整：

```rust
// sgl-model-gateway/src/server.rs
.route("/generate", post(generate))
.route("/v1/chat/completions", post(v1_chat_completions))
...
.route("/workers", post(create_worker).get(list_workers_rest))
...
.route("/ha/status", get(get_cluster_status))
.route("/ha/health", get(get_mesh_health))
```

配套还有 `policies/`（`cache_aware.rs`、`power_of_two.rs`、`consistent_hashing.rs`、`prefix_hash.rs`、`round_robin.rs`、`bucket.rs`）、`core/circuit_breaker.rs`、`core/retry.rs`、`core/token_bucket.rs`（限流）、`core/worker_registry.rs`（worker 动态注册表）、`service_discovery.rs` 等——这是一整套生产级负载均衡/容错基础设施。

`SGLangEngine._register_to_router` 就是把每个新起的引擎注册成这个网关的一个 worker：

```python
def _register_to_router(self, server_args_dict):
    ...
    response = requests.post(
        f"http://{self.router_ip}:{self.router_port}/workers",
        json=payload,
    )
```

**为什么用 HTTP router 而不是让 Ray 直接调 engine 来做生成/负载均衡：**

1. **一个"engine"可能跨多个节点，Ray 看不到内部拓扑**：当 `tp_size` 超过单机 GPU 数时，一个 engine 会跨 `nnodes` 个节点，只有 `node_rank == 0` 的进程对外暴露 HTTP 端口（代码里到处是 `if self.node_rank != 0: return`），其它节点的 `SGLangEngine` actor 基本是空壳。真正的跨 rank 协同（TP/PP all-reduce 等）发生在 sglang 内部通过 NCCL 完成，Ray 完全介入不了这层，因此天然只能"从外面"以整体一个 HTTP endpoint 的方式去调用它。
2. **负载均衡/容错/PD-disaggregation 路由是重活，没必要在 Ray 里重造轮子**：多个引擎背后要做请求分发、prefix-cache 感知路由、熔断重试、PD 场景下 prefill/decode 的 bootstrap room 路由——sgl-model-gateway 已经用 Rust 实现好了这一整套，直接复用它的 HTTP 接口就能拿到工业级的性能和稳定性；用 Ray actor 调用去重新实现同等能力代价极高。
3. **支持"外部引擎"（`args.rollout_external`）**：`external.py` 里 `discover_external_engines` 可以把完全独立于 Ray、已经在别处跑起来的 sglang server 地址接进同一个 router。对上层 rollout 代码来说，内部/外部引擎没有任何区别——都只是 router 背后的一个 HTTP worker url。如果依赖 Ray RPC 做生成，就无法兼容这种"外部引擎"模式。
4. **性能/并发模型更适合**：一次 rollout 通常是成千上万个并发采样请求，`init_http_client` 用 `httpx.AsyncClient` 连接池 + 重试（`_post`），比每次都走 Ray 的 GCS 调度 + cloudpickle 序列化 + object store 更轻量；而且 HTTP 天然支持流式（SSE，`stream=True`）返回 token，Ray remote call 的返回值语义上是"一次性"的，不方便做流式增量输出。
5. **职责分离**：Ray 在这里只承担"控制面"（申请 GPU、起停进程、健康检测重启、权重更新时的 NCCL 握手协调）——这些是低频、非并发热点的管理操作，天然适合 Ray remote call；"数据面"（海量 generate 请求）完全走 HTTP router，两者解耦得很干净，也是 sglang 自身单机部署（不依赖 Ray）时的标准用法，slime 只是把这条路径原样复用。

**一句话总结**：`SGLangEngine` 这个 Ray actor 管的是"进程生死"和"控制类操作"（health check、权重更新、显存 offload），而"推理请求本身"从设计上就是 sglang 的 HTTP 协议 + 独立 Rust 网关的地盘，Ray 不适合也没必要插手这条热路径。

---

## 4. `rollout_engine_lock` 的工作机制

### 4.1 它是什么

```python
# slime/ray/utils.py
@ray.remote
class Lock(RayActor):
    def __init__(self):
        self._locked = False  # False: unlocked, True: locked

    def acquire(self):
        if not self._locked:
            self._locked = True
            return True
        return False

    def release(self):
        assert self._locked, "Lock is not acquired, cannot release."
        self._locked = False
```

就是一个极简的**全局互斥锁**，本质是一个只有一个布尔状态的 Ray actor。因为 Ray actor 默认单线程串行处理传入的方法调用，`acquire()` 里"检查 + 置位"这两步天然是原子的，所以即便多个训练 rank 同时 `acquire.remote()`，也不会出现竞态。

`acquire()` 是**非阻塞、立即返回 True/False**，调用方需要自己轮询（自旋锁）：

```python
while not ray.get(self.rollout_engine_lock.acquire.remote()):
    time.sleep(0.1)
```

### 4.2 它是怎么被创建和传递的

`RolloutManager.__init__` 里创建了**唯一一个**锁 actor，供该 `RolloutManager` 管理的所有可更新（`update_weights=True`）引擎共用：

```python
self.rollout_engine_lock = Lock.options(
    num_cpus=1,
    num_gpus=0,
    runtime_env={"env_vars": add_default_ray_env_vars()},
).remote()
```

训练端（Megatron actor）每次 `update_weights()` 时通过 `get_updatable_engines_and_lock()` 拿到这同一个锁的 handle，再传给权重更新器：

```python
def get_updatable_engines_and_lock(self):
    srv = self._get_updatable_server()
    engines = srv.engines if srv else []
    ...
    return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets
```

→ `slime/backends/megatron_utils/actor.py::update_weights()` 拿到后传给 `weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock, ...)`。

### 4.3 它到底在防什么

只有 **NCCL 分布式广播路径（`UpdateWeightFromDistributed`）** 真正用到这把锁，用在"每个权重 bucket 广播"的临界区：

```python
# slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py
def _update_bucket_weights_from_distributed(self, converted_named_tensors, pbar=None, load_format=None):
    """
    Lock → broadcast → clear → unlock → pbar++. Lock prevents NCCL deadlock.
    """
    while not ray.get(self.rollout_engine_lock.acquire.remote()):
        time.sleep(0.1)

    refs = update_weights_from_distributed(
        self._group_name, self._model_update_groups, self.weight_version,
        self.rollout_engines, converted_named_tensors, load_format=load_format,
    )
    ray.get(refs)
    converted_named_tensors.clear()
    ray.get(self.rollout_engine_lock.release.remote())
    pbar.update(1)
```

**原因**：`connect_rollout_engines_from_distributed` 会为**每个 PP rank**建一个独立 NCCL group（`"slime-pp_{pp_rank}"`），group 里包含"该 PP 源 rank + 所有 rollout engine 的对应 GPU"。`update_weights_from_distributed` 内部对每个权重张量都执行 `dist.broadcast(param.data, 0, group=group, async_op=True)`，这属于 **NCCL 集合通信**：同一个 group 里的所有参与方必须**以完全相同的顺序**发出对应的集合操作，否则会出现"张量 A 的广播在训练侧发出、但引擎侧先收到了本该属于张量 B 广播的调用"这种错位，导致 NCCL 通信卡死（deadlock/hang）。

如果没有这把锁，多个 PP rank（甚至同一 rank 内多个 bucket）可能会**并发地**对同一批 rollout engine 发起 `engine.update_weights_from_distributed.remote(...)`（Ray 侧）+ `dist.broadcast(...)`（NCCL 侧），Ray 端调用到达 engine 的顺序和 NCCL 端广播发起的顺序可能不一致，就会踩上面说的死锁问题。加这把全局锁后，**任意时刻只允许一个"bucket 广播"在跟 rollout engine 群的 NCCL group 打交道**，把所有这类广播强制串行化——代码注释写得很直白："Lock prevents NCCL deadlock."

### 4.4 补充说明：不是唯一的同步机制，也不是所有更新路径都用

- `UpdateWeightFromTensor`（colocated 场景，通过 Ray/CUDA-IPC 或 nixl 传输权重）虽然 `connect_rollout_engines` 也接收了 `rollout_engine_lock` 参数，但**根本没有调用它的 `acquire/release`**——因为这条路径每个 bucket 是一次性的 Ray remote 调用搭配 GPU 直接传输，没有长期存活、需要按顺序参与集合通信的 NCCL group，不存在"顺序错位死锁"的风险。
- `UpdateWeightFromDisk` / `UpdateWeightFromDiskDelta` 同理不使用这把锁；`update_weight_from_disk_delta.py` 里注释直接说明：*"The rollout_engine_lock the NCCL path uses isn't needed — the engine-side apply is serialized by a per-host flock."*（改用文件锁保证同机串行）。
- 真正保证"权重替换期间不会用半新半旧的权重去生成"的，是**另一套完全独立的机制**——`pause_generation` / `continue_generation`（对每个 sglang engine 的 HTTP `/pause_generation`、`/continue_generation` 调用），几乎所有更新路径（tensor / distributed / disk-delta）在替换权重前后都会调用它俩，让 sglang 的 scheduler 暂停/恢复接单。这与 `rollout_engine_lock` 是两个维度的问题：
  - `pause_generation` / `continue_generation` → 保证"推理正确性"（不读到半更新的权重），几乎每种更新方式都用。
  - `rollout_engine_lock` → 只保证"NCCL 广播调用顺序不冲突/不死锁"，只在真正用持久 NCCL process group 做权重广播的 `UpdateWeightFromDistributed` 路径里才需要。

---

## 5. model / server_group / engine 三层结构详解

### 5.1 结论：支持多个不同的 model

`SglangConfig` 的文档字符串（`slime/backends/sglang_utils/sglang_config.py`）就是最好的证明：

```yaml
sglang:
  - name: actor
    model_path: /path/to/actor
    update_weights: true          # receives training weight updates (default)
    num_gpus_per_engine: 2
    server_groups:
      - worker_type: prefill
        num_gpus: 4
        num_gpus_per_engine: 2
      - worker_type: decode
        num_gpus: 8
        num_gpus_per_engine: 4
  - name: ref
    model_path: /path/to/ref
    update_weights: false          # frozen, no weight updates
    server_groups:
      - worker_type: regular
        num_gpus: 4

# Each model gets its own router.
```

`--sglang-config` 这个 YAML 的顶层 `sglang:` 是一个**列表**，每一项就是一个"model"。这在 RL 训练里很常用：比如同时部署

- `actor`：接收训练侧权重更新的模型（生成 rollout 用）
- `ref`：参考模型（算 KL），权重永远不更新（`update_weights: false`）
- 也可以再加一个 `reward`：奖励模型

它们是**互相独立**的一整套 SGLang 部署（各自有各自的 router 进程、各自的 engine 进程），只是共用同一个 `RolloutManager`/同一个 Ray placement group 来编排。

### 5.2 三层结构

```
SglangConfig                                    (整个 --sglang-config YAML)
 └─ models: list[ModelConfig]                   (第 1 层：model，如 actor / ref / reward)
     └─ server_groups: list[ServerGroupConfig]  (第 2 层：server_group)
         └─ 展开为运行时的 ServerGroup(dataclass)
             └─ all_engines: list[SGLangEngine(Ray actor)]  (第 3 层：engine)
```

**第 1 层：model（`ModelConfig`）** —— "一个独立部署的模型服务"，有自己的 `name`、`model_path`、`update_weights` 标志、自己的一整套 `server_groups`。运行时对应关系：

```python
# slime/ray/rollout.py: start_rollout_servers()
for model_idx, model_cfg in enumerate(config.models):
    router_ip, router_port = _start_router(args, ...)   # 每个 model 自己的 router 进程
    ...
    servers[model_cfg.name] = RolloutServer(server_groups=server_groups, router_ip=router_ip, router_port=router_port, ...)
```

最终 `RolloutManager.servers` 是一个 `dict[model_name -> RolloutServer]`。

**第 2 层：server_group（`ServerGroupConfig` / 运行时的 `ServerGroup`）** —— 同一个 model 内部，可能需要不同"角色"的引擎组，最典型的场景是 **PD 分离**（Prefill-Decode disaggregation）。上面 YAML 例子里 `actor` model 就有两个 server_group：

- `worker_type=prefill`，`num_gpus=4`，`num_gpus_per_engine=2` → TP=2，用 4 张卡跑出 **2 个** prefill engine
- `worker_type=decode`，`num_gpus=8`，`num_gpus_per_engine=4` → TP=4，用 8 张卡跑出 **2 个** decode engine

`worker_type` 还可以是 `regular`（普通，不做 PD 分离）、`encoder`（多模态场景的 encoder-only 引擎，EPD 分离用）、`placeholder`（只占 GPU 位置不真正起引擎）。**一个 `ServerGroup` 内部所有 engine 必须同构**（同样的 TP size / 同样的 worker_type），这也是为什么 PD 分离要拆成两个 group。

**第 3 层：engine（`SGLangEngine` 这个 Ray actor）** —— **一个 `SGLangEngine` Ray actor ≈ 一个逻辑上的 sglang server 实例（一个 TP 副本）**：

```python
# ServerGroup.start_engines()
num_engines = group_cfg.num_gpus // num_gpus_per_engine_on_node
for i in range(len(self.all_engines)):
    rollout_engine = RolloutRayActor.options(...).remote(
        self.args, rank=global_rank, worker_type=self.worker_type,
        base_gpu_id=base_gpu_id, sglang_overrides=..., num_gpus_per_engine=self.num_gpus_per_engine,
    )
```

即：`ServerGroup.num_gpus / ServerGroup.num_gpus_per_engine = 该 group 内的 engine 数`。每个 engine 会在自己独占的一批 GPU 上，通过 `multiprocessing.Process` 启动一份独立的 sglang server 进程。

**跨节点 TP 的细节：**

```python
@property
def nodes_per_engine(self):
    return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

@property
def engines(self):
    """Node-0 engines only (for multi-node serving)."""
    return self.all_engines[:: self.nodes_per_engine]
```

如果 `num_gpus_per_engine`（TP size）超过单机 GPU 数（比如 TP=16，每台机器 8 卡），一个"逻辑 engine"要跨 2 台节点部署：

- `all_engines`（原始列表）：每个"node-in-engine"都对应一个 `SGLangEngine` Ray actor，如 `[engine0_node0, engine0_node1, engine1_node0, engine1_node1, ...]`。
- `engines`（对外可见的）：每隔 `nodes_per_engine` 取一个，只保留每个逻辑 engine 里 `node_rank==0` 那个——只有它对外暴露 HTTP 端口，其余节点的 actor 代码里到处是 `if self.node_rank != 0: return`，节点间协同全靠 sglang 内部的 NCCL/`dist_init_addr`。

### 5.3 完整例子

假设 `actor` model 配置了 `worker_type=regular, num_gpus=8, num_gpus_per_engine=4`（TP=4），每台机器 8 卡：

- `num_engines = 8 // 4 = 2` → 2 个 `SGLangEngine` Ray actor（`nodes_per_engine=1`，单机放下一个 TP=4 引擎）
- 这 2 个 engine 各自在自己的 4 张 GPU 上起一个 sglang server 进程（各自监听不同 host:port）
- 它俩分别通过 `_register_to_router` 把自己的 URL 注册到 `actor` model 唯一的那个 router 上
- rollout 代码调用 `http://{actor_router_ip}:{actor_router_port}/generate` 时，router 在这 2 个 engine 之间做负载均衡

如果同时还配置了 `ref` model（`worker_type=regular, num_gpus=4, num_gpus_per_engine=4`，1 个 engine，`update_weights=False`），`RolloutManager.servers` 就是：

```python
{
    "actor": RolloutServer(server_groups=[ServerGroup(worker_type="regular", all_engines=[engine_a0, engine_a1])], router_ip=..., router_port=..., update_weights=True),
    "ref":   RolloutServer(server_groups=[ServerGroup(worker_type="regular", all_engines=[engine_r0])], router_ip=..., router_port=..., update_weights=False),
}
```

`ref` 有独立的 router 端口、独立的 engine 进程；训练侧权重更新只会找 `update_weights=True` 的那个（即 `_get_updatable_server()`）。

---

## 6. sglang server 是如何与 Ray placement group（GPU 资源）绑定起来的

完整调用链：

```
train.py / train_async.py
  └─ create_placement_groups(args)              # slime/ray/placement_group.py
       └─ _create_placement_group(num_gpus)     # 创建 Ray placement group + 探测每个 bundle 的真实物理 GPU
  └─ create_rollout_manager(args, pgs["rollout"])
       └─ RolloutManager.__init__(args, pg)      # slime/ray/rollout.py
            └─ start_rollout_servers(args, pg)
                 └─ ServerGroup(pg=pg, gpu_offset=..., rank_offset=...)
                      └─ ServerGroup.start_engines()
                           └─ PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=...)
                           └─ RolloutRayActor.options(num_gpus=0.2, scheduling_strategy=..., runtime_env={env_vars: NOSET_*}).remote(base_gpu_id=...)
                                └─ SGLangEngine.init() → ServerArgs(base_gpu_id=..., ...) → 真正的 sglang 子进程
```

### 6.1 第一步：创建"物理探测过"的 placement group

```python
# slime/ray/placement_group.py
def _create_placement_group(num_gpus):
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    # 用一个临时的 InfoActor 逐个 bundle 探测：这个 bundle 实际落在哪个节点、对应哪个物理 GPU id
    info_actors = [InfoActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=i)).remote() for i in range(num_bundles)]
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    # 按 (节点IP, GPU id) 排序，得到"逻辑顺序 -> 实际 bundle index / 实际物理 GPU id"的映射
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]
    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids
```

**为什么要"探测+重排"？** Ray 的 placement group 只保证"申请到 N 个 GPU bundle"，但 `bundle index 0,1,2...` 分配到哪个物理节点/哪张卡是不确定的（比如 bundle 0 在节点 B 的 GPU 3、bundle 1 在节点 A 的 GPU 0）。slime 需要**确定性的、按节点+GPU编号排序好的顺序**，才能安全地说"前 `actor_num_gpus` 个逻辑位置分给训练，后面 `rollout_num_gpus` 个逻辑位置分给推理"，而不会出现训练和推理的卡在物理上"犬牙交错"分布导致资源规划混乱。做法是先起一批一次性的 `InfoActor`（每个绑定到一个 bundle）问它"你在哪个节点、拿到了哪张物理 GPU"，问完就 `ray.kill` 掉，再按 `(node_ip, gpu_id)` 排序，产出：

- `pg`：原始的 Ray placement group 对象（真正的资源句柄）
- `pg_reordered_bundle_indices`：`逻辑顺序 i -> 该逻辑位置对应的真实 bundle index`
- `pg_reordered_gpu_ids`：`逻辑顺序 i -> 该逻辑位置对应的真实物理 GPU id`

### 6.2 第二步：从同一个 pg 里"切"出 actor 和 rollout 各自的一段

```python
# create_placement_groups()
num_gpus, rollout_offset = _get_placement_group_layout(args)   # 例如非colocate: actor_num_gpus + rollout_num_gpus, 偏移=actor_num_gpus
pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)
rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]

result = {
    "actor":   (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
    "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
}
```

**举例**：假设 `actor_num_gpus=8`（训练用），`rollout_num_gpus=8`（推理用），非 colocate 模式：`num_gpus=16`，`rollout_offset=8`。整个大 placement group 有 16 个 bundle，按物理顺序排好后：

- `actor` 拿到重排数组的 `[0:8]` 这一段（物理上排好序的前 8 张卡）
- `rollout` 拿到重排数组的 `[8:16]` 这一段（"后 8 个"）

**注意：`pg`（Ray placement group 对象本身）是同一个！** 只是切出来的"重排索引数组"不同段。如果是 `colocate` 模式（训练和推理共享同一批卡，训练时推理 offload、推理时训练 offload），`rollout_offset=0`，`actor` 和 `rollout` 会拿到**完全重叠**的索引段，即共享同一批物理 GPU。

这个 `("rollout")` 三元组就是 `RolloutManager.__init__(self, args, pg)` 收到的 `pg` 参数，存成 `self.pg`，再传给每个 `ServerGroup(pg=pg, ...)`。

### 6.3 第三步：`ServerGroup` 内部再切出"某个 server_group 应该用哪几张卡"

```python
# ServerGroup.start_engines()
pg, reordered_bundle_indices, reordered_gpu_ids = self.pg
gpu_index = self.gpu_offset + i * num_gpus_per_engine_on_node   # 在"rollout 专属"的重排数组里的下标
base_gpu_id = int(reordered_gpu_ids[gpu_index])                 # 这个 engine 第 0 张卡对应的真实物理 GPU id

scheduling_strategy = PlacementGroupSchedulingStrategy(
    placement_group=pg,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=reordered_bundle_indices[gpu_index],   # 告诉 Ray：把这个 actor 调度到这个具体 bundle 上
)
```

- `gpu_offset`：这个 server_group 在"整个 rollout GPU 池"里的起始偏移（前面已经有多少张卡被别的 model/别的 server_group 占用了，逐个 group 累加，见 `_make_group` 里的 `gpu_offset += group_cfg.num_gpus`，是**跨 model 全局累加**的）。
- `gpu_index = gpu_offset + i * num_gpus_per_engine_on_node`：第 `i` 个 engine 在该节点上占用的第一张卡，在"rollout 专属重排数组"里的下标。
- `reordered_bundle_indices[gpu_index]` → 交给 `PlacementGroupSchedulingStrategy`，这是 Ray 真正用来"把这个 Ray actor 钉死在哪个物理 bundle（哪台机器的哪个资源槛）"的依据。
- `reordered_gpu_ids[gpu_index]` → "真实物理 GPU 编号"（比如 nvidia-smi 看到的卡 3），作为 `base_gpu_id` 一路传进 `SGLangEngine.init()` → `_compute_server_args(..., base_gpu_id=base)` → 最终变成 sglang 自己的 `--base-gpu-id` 启动参数，**告诉 sglang 子进程该用哪些物理 GPU**。

**举例**（接上例）：假如 `actor` model 只有一个 `server_group`（`regular`, `num_gpus=8, num_gpus_per_engine=4`，`gpu_offset=0`）：

- engine0：`gpu_index = 0 + 0*4 = 0` → `base_gpu_id = reordered_gpu_ids[0]`（rollout 池里第 1 张物理卡）
- engine1：`gpu_index = 0 + 1*4 = 4` → `base_gpu_id = reordered_gpu_ids[4]`（rollout 池里第 5 张物理卡）

如果 `ref` model 紧跟在 `actor` 后面被处理，它的 `gpu_offset` 就会从 `8` 开始，从而占用 rollout 池里第 9 张卡开始的资源，不会跟 `actor` 冲突。

### 6.4 关键细节：`num_gpus=0.2` 与 `CUDA_VISIBLE_DEVICES` 的解耦

```python
num_gpus = 0.2
num_cpus = num_gpus
env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {...}
rollout_engine = RolloutRayActor.options(
    num_cpus=num_cpus,
    num_gpus=num_gpus,          # 只向 Ray 申报 0.2 张卡！
    scheduling_strategy=scheduling_strategy,
    runtime_env={"env_vars": add_default_ray_env_vars(env_vars)},
).remote(self.args, ..., base_gpu_id=base_gpu_id, ...)
```

正常情况下 Ray actor 申请 `num_gpus=N` 会让 Ray 自动帮它设置 `CUDA_VISIBLE_DEVICES`，把进程"锁"在分配的那 N 张卡上，且进程内看到的卡编号会被 Ray 重映射成 `0..N-1`。但这里：

1. `num_gpus=0.2`——故意申报一个远小于真实需求（TP=4 需要 4 张卡）的"零头"值。真正的资源隔离由 `placement_group_bundle_index` 精确指定的 bundle 保证，`num_gpus=0.2` 只是个"意思一下"的记账数字，避免跟 placement group 里"每个 bundle=1 GPU"的真实资源预留产生冲突。
2. `NOSET_VISIBLE_DEVICES_ENV_VARS_LIST`（如 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`）——告诉 Ray **不要**自动帮这个 actor 设置/重映射 `CUDA_VISIBLE_DEVICES`。因为 sglang 是通过 `multiprocessing.Process` 另起一个子进程去真正用 GPU 的，且 sglang 自己接受一个显式的 `--base-gpu-id` 参数来精确指定要用哪几张卡——如果让 Ray 再自动重映射一层，两边的"GPU 编号"体系就会打架、错位。所以这里完全绕开 Ray 的自动 GPU 分配机制，资源隔离全靠"申请到了这个 bundle（对应这台机器）"+"用 `base_gpu_id` 精确指定用哪张卡"这两步手动保证。

**一句话总结**：`pg`（Ray placement group）只负责"预定哪些物理机器上的哪些 GPU 资源槛"，`bundle_index` 决定 Ray actor（`SGLangEngine`）被调度到哪台物理机；而"这台机器上具体用第几号 GPU 卡"是通过 `base_gpu_id` 显式传给 sglang 自己的 `ServerArgs`，绕开 Ray 的自动 GPU 分配/重映射机制，两条腿分别负责"选机器"和"选卡"。

---

## 7. slime 里两种"router"的区别

这里有一个容易混淆的地方：**"router"这个词出现在两个完全不同的层面**，而且**slime 并没有自己实现一个 router**。

### 7.1 先澄清：slime 没有"自己的" router 实现

```python
# slime/ray/rollout.py
def _start_router(args, ...):
    from sglang_router.launch_router import RouterArgs
    from slime.utils.http_utils import run_router
    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    process = multiprocessing.Process(target=run_router, args=(router_args,))
    process.start()

# slime/utils/http_utils.py
def run_router(args):
    from sglang_router.launch_router import launch_router
    router = launch_router(args)
```

`sglang_router` 这个包**不是 slime 写的**，它是 sglang 项目自己的子项目（源码在 `sglang/sgl-model-gateway/`，Rust 实现，Python 侧只是薄封装 `sgl-model-gateway/bindings/python/src/sglang_router/`）。`_start_router` 做的事情就是：**用 `multiprocessing.Process` 把 sglang 自带的这个 router 组件当成子进程拉起来**。所以严格来说不存在"slime 的 router"和"sglang 的 router"两套东西——**只有一套 router（sglang 项目提供的 sgl-model-gateway），slime 只是负责启动它、管理生命周期、往里注册/摘除 worker**。

### 7.2 真正需要区分的两层："router"（网关）vs 每个 engine 自己的 HTTP server

| | 每个 engine 自己的 HTTP server | Router（sgl-model-gateway） |
|---|---|---|
| 代码位置 | `sglang/python/sglang/srt/entrypoints/http_server.py`（FastAPI） | `sglang/sgl-model-gateway/`（Rust） |
| 由谁启动 | 每个 `SGLangEngine.init()` → `launch_server_process` → `multiprocessing.Process` | 每个 model 一份，由 `_start_router` 启动 |
| 数量 | **每个 engine 一份** | **每个 model 一份**（1 个 router 管这个 model 下所有 engine） |
| 知道谁存在 | 只知道自己 | 维护一份 worker 注册表，知道这个 model 下**所有** engine 的地址（`worker_registry.rs`） |
| 对外职责 | 真正跑推理请求、处理该 engine 自己的控制类请求（`/health_generate`、`/update_weights_from_tensor`、`/release_memory_occupation`...） | 接收统一入口请求（`/generate`、`/v1/chat/completions`...），决定转发给哪个 engine（负载均衡），支持 worker 动态加入/摘除、熔断、重试、限流、缓存感知路由、PD-disaggregation 的 bootstrap room 路由等 |
| slime 里谁调用它 | `SGLangEngine._make_request` / `health_generate` / `update_weights_from_disk` 等——直接拿这个 engine 自己的 `server_host:server_port` 发请求 | rollout 代码（`slime/rollout/sglang_rollout.py::generate()`）——拿 `args.sglang_router_ip:args.sglang_router_port` 发请求 |

**为什么两者都要存在：**

1. **只留每个 engine 自己的 HTTP server 不行**：rollout 需要发起成千上万个并发采样请求，如果直接打给某一个固定 engine，既没有负载均衡（其它 engine 空闲），也没法在某个 engine 挂掉时自动切换（`SGLangEngine` 本身不知道"同伴"的存在）。
2. **只留 router 不行**：router 只是转发层，它自己不跑模型、不管理具体某个 engine 的生命周期。像 `release_memory_occupation`、`update_weights_from_tensor`、`health_generate` 这些**必须对每一个 engine 都单独调一遍**的控制操作，slime 是绕过 router、直接拿每个 `SGLangEngine` 存的 `server_host:server_port` 发请求的：

```python
# slime/backends/sglang_utils/sglang_engine.py
def _make_request(self, endpoint: str, payload: dict | None = None):
    if self.node_rank != 0:
        return
    url = f"http://{self.server_host}:{self.server_port}/{endpoint}"   # 直接打这个 engine 自己的地址，不经过 router
    response = requests.post(url, json=payload or {})
```

而真正的采样生成请求，则是打 router 地址，让 router 去挑一个 engine：

```python
# slime/rollout/sglang_rollout.py
async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"   # 打 router，不关心具体哪个 engine 处理
    output = await post(url, payload, headers=headers)
```

**调用链对比（一次生成 vs 一次权重更新）**：

```
【生成一条样本】                                    【给某个 engine 做健康检查/更新权重】
rollout function                                    Megatron actor / RolloutManager
  └─ POST http://router_ip:router_port/generate         SGLangEngine.health_generate() / update_weights_from_tensor()
       └─ [router 内部按策略选中 engine_k]                   └─ 直接 POST http://engine_k自己的host:port/xxx
            └─ 转发给 engine_k 自己的 http_server.py             （完全不经过 router）
                 └─ TokenizerManager → Scheduler(NCCL/ZMQ) → 生成结果
                      └─ router 把结果原样转发回调用方
```

3. **顺带一提**：`http_server.py` 里有一句 `app.include_router(v1_loads_router)`，这是 FastAPI 框架自己的"路由表"概念（把一组 endpoint 打包挂载到 app 上，即"URL 路径 → 处理函数"的映射），跟上面说的"负载均衡网关"完全不是一回事，只是命名撞车，不用管它。

**一句话总结**：slime 没有自研 router，`_start_router` 只是把 sglang 项目自带的独立 Rust 网关进程（`sgl-model-gateway`，pip 包名 `sglang_router`）拉起来管理；每个 sglang engine 自己也有一份 FastAPI HTTP server，但那只服务它自己、不做负载均衡。控制类操作（健康检查、权重更新、显存 offload）绕开 router 直连每个 engine；真正的生成请求走 router，由 router 在多个 engine 间做负载均衡/容错。

---

## QA
### Server Group
在当前如 **SGLang** 等现代大语言模型（LLM）推理引擎以及围绕它们构建的分布式训练/后训练框架（如 THUDM 的 `slime`、字节跳动的 `verl`、NVIDIA 的 `NeMo-RL` 等）中，**Server Group（服务器组）** 是一个非常核心的**逻辑部署与路由单元**。

简单来说，**Server Group 是指一组具有相同或相似配置、共同承担特定推理任务或模型角色的推理引擎实例（Worker/Engine）的集合**。它通常在统一的 API 网关或路由器（如 `sglang_router`）的管理下运行。

这一概念的引入主要是为了应对 LLM 规模扩大、多轮对话、以及强化学习（RLHF）训练中复杂的分布式推理需求。以下是 Server Group 的核心设计意义与应用场景：

---

#### 1. 核心设计意义与应用场景

##### ① Prefill-Decode Disaggregation (PD 分离 / 预填充与解码分离)
在 LLM 推理中，**Prefill**（处理 Prompt 并计算 KV Cache，属于计算密集型）和 **Decode**（逐字生成 Token，属于显存带宽受限型）阶段的物理特性截然不同。
通过 Server Group，可以将计算资源划分为不同的组：
* **Prefill 组**（`worker_type: prefill`）：分配适合大算力的张量并行（Tensor Parallelism, TP）大小，并设置较大的分块预填充尺寸（如 `chunked_prefill_size: 8192`）。
* **Decode 组**（`worker_type: decode`）：配置更大的显存比例（`mem_fraction_static`）以存放高并发的 KV Cache，降低生成延迟。
* 路由器（Router）会根据请求处于哪一个阶段，动态将请求分发给对应的 Server Group。

##### ② 异构扩容与参数覆盖（Heterogeneous Configs）
在同一个模型（如 Llama-3-70B）的集群中，不同的 Server Group 可以使用不同的硬件规格和软件参数。例如：
* 组 A 使用 4 张 GPU（TP=4），组 B 显存受限仅使用 2 张 GPU（TP=2）。
* 不同的组可以针对不同的客户端或场景，通过 `overrides` 单独重写 SGLang 启动参数。

##### ③ 强化学习（RLHF）训练中的多模型角色隔离
在强化学习（如 PPO、GRPO 算法）的训练过程中，通常需要同时运行和调用多个模型推理服务（例如 **Actor 策略模型**、**Reference 参考模型**、**Reward 奖励模型**、**Critic 价值模型**）。
在框架中，每个模型拥有自己的 Router，每个 Router 下管理着不同角色的 Server Group：
* **Actor 组**：需要在训练迭代时频繁接收训练端同步过来的最新权重（`update_weights: true`）。
* **Reference 组**：权重保持冻结，不需要进行任何权重更新（`update_weights: false`）。

##### ④ 占位符组（Placeholder）
在“联合训练与推理（Co-located Training and Inference）”的集群配置中，可以使用 `worker_type: placeholder` 的 Server Group。这类组并不实际启动推理引擎，而是仅仅在物理 GPU 上“占位”（预留显存和卡位），以便在训练与生成阶段之间精细协调、避免显存冲突。

---

#### 2. 配置结构示例

以集成 SGLang 的大模型训练框架中的 `server_groups` 典型 YAML 配置为例，可以更直观地理解其结构：

```yaml
sglang:
  - name: actor              # 这是一个名为 actor 的模型
    update_weights: true     # 允许训练中途同步权重
    server_groups:
      - worker_type: prefill  # 组1：专门负责 Prefill
        num_gpus: 4           # 总共给该组分配 4 张 GPU
        num_gpus_per_engine: 2 # 两个实例，每个实例 TP=2
        overrides:            # 专门针对预填充优化参数
          chunked_prefill_size: 8192

      - worker_type: decode   # 组2：专门负责 Decode
        num_gpus: 12          # 分配 12 张 GPU 应对大并发
        num_gpus_per_engine: 4 # 每个实例 TP=4 
        overrides:            # 专门针对解码优化参数
          mem_fraction_static: 0.88
```

---

#### 3. Server Group 如何与 Router 协同工作？

Server Group 的高效运转高度依赖前端的**路由器层**（如 `sglang_router`）：
1. **统一入口**：客户端无需感知底层有多少个 GPU、多少个组，只需将请求发送给 Router 暴露的 OpenAI 兼容接口。
2. **状态感知**：Router 能够感知不同 Server Group 中各个 Worker 的负载情况、显存占用，从而进行智能的负载均衡。
3. **缓存感知路由（Cache-Aware Routing）**：结合 SGLang 核心的 Radix Cache（前缀缓存树）机制，Router 能够分析 Prompt。如果发现某个请求含有与刚才某组 Worker 处理过的一模一样的 System Prompt（系统提示词）或多轮历史，Router 会**优先将请求分发到同一个 Server Group 的特定节点**，极大提高前缀缓存命中率，省去 Prefill 阶段重复计算的时间。

#### 总结
在现代 LLM 推理生态中，**Server Group** 的概念标志着推理服务从“单机单卡/单机多卡单一实例”的粗放模式，演进到了**多节点、异构、按需解耦（如 PD 分离、多模型并存）**的微服务化集群管理模式，是实现超大规模高吞吐、低延迟推理的基础设施。

## 参考源码位置

- `slime/ray/rollout.py`：`RolloutManager`、`ServerGroup`、`RolloutServer`、`start_rollout_servers`、`_start_router`、`_allocate_rollout_engine_addr_and_ports_normal`
- `slime/backends/sglang_utils/sglang_engine.py`：`SGLangEngine`（Ray actor，HTTP 客户端适配器）
- `slime/backends/sglang_utils/sglang_config.py`：`SglangConfig`、`ModelConfig`、`ServerGroupConfig`（model/server_group 配置层）
- `slime/backends/sglang_utils/external.py`：外部引擎发现与接入（`rollout_external`）
- `slime/ray/rollout_validation.py`：`validate_server_group_gpu_indices`（GPU 索引合法性校验）
- `slime/ray/placement_group.py`：`_create_placement_group`、`create_placement_groups`（placement group 创建与物理 GPU 探测重排）
- `slime/utils/http_utils.py`：`init_http_client`、`post`/`get`（异步 HTTP 客户端池）、`run_router`
- `slime/rollout/sglang_rollout.py`：`generate()`、`get_model_url()`（真正的生成请求走 router）
- `slime/ray/utils.py`：`Lock`（`rollout_engine_lock` 的实现）
- `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py`：`rollout_engine_lock` 的实际使用（NCCL 广播互斥）
- `sglang/python/sglang/srt/entrypoints/http_server.py`：sglang 引擎的 FastAPI HTTP 服务实现
- `sglang/sgl-model-gateway/`：Rust 实现的 router / 网关（`sglang_router` pip 包背后的实现）
