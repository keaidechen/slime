# 02 Rollout 子系统：Server 模式的 SGLang 推理是如何生成训练数据的

> 对应综述（`00_rl_infra_survey.md`）§2.2「Rollout 架构：Engine 模式 vs Server 模式」。
> slime 是纯 server 模式框架：推理是独立的 HTTP 服务，训练侧通过 router 访问。本篇逐层解读这条链路。

---

## 1. 三个层次

```
编排层  slime/ray/rollout.py        RolloutManager / RolloutServer / ServerGroup
        ─ 起停 server、起 router、取数据、打包训练 batch、容错
函数层  slime/rollout/sglang_rollout.py
        ─ 默认 rollout 函数：异步生成循环、RM 打分、过滤、abort 回收
引擎层  slime/backends/sglang_utils/
        ─ SGLang server 进程封装、server_control（abort/显存）、引擎配置
```

数据流向：`RolloutManager.generate()` → （函数层）对 router 发 HTTP `/generate` → （引擎层）SGLang server 采样 → 带 token 级 logprob 的响应 → 填回 `Sample` → 攒够 `rollout_batch_size` 组 → 转换成训练数据。

---

## 2. 编排层：server 与 router 的启动

### 2.1 启动总入口

`RolloutManager.__init__`（`slime/ray/rollout.py:430` 起）里调用：

- `start_rollout_servers(args, pg)`（rollout.py:1090）：解析 `--sglang-config`（`_resolve_sglang_config`，rollout.py:1232，支持异构 server group、multi-model serving 的 YAML 配置），为每个模型启动 router，为每个 ServerGroup 创建引擎 actor；
- `_start_router(args, ...)`（rollout.py:1019）：用 `multiprocessing.Process` 启动 **sglang_router** 进程。router 是所有推理 server 前的负载均衡入口——训练侧只认 `sglang_router_ip:sglang_router_port` 一个地址；
- `ServerGroup.start_engines()`（rollout.py:150-259）：创建 `SGLangEngine` Ray actor 并 `engine.init.remote(...)`，每个 actor 内部用 `launch_server_process` 拉起 `sglang.http_server` 子进程。

**server 模式的关键收益**在这里体现：server 崩溃只影响一个 actor（有 `recover()`，rollout.py:346）；server 数量可以独立于训练规模扩缩；router 让多实例对训练侧透明。

### 2.2 RolloutManager 的远程接口

driver 与训练侧看到的是这组方法（rollout.py）：

| 方法 | 行号 | 用途 |
|---|---|---|
| `generate(rollout_id)` | 552 | 产一批训练数据（核心） |
| `eval(rollout_id)` | 567 | 评测集生成 |
| `save/load` | 578-583 | 数据源游标持久化 |
| `offload/onload/onload_weights/onload_kv` | 584-600 | 显存错峰（04 篇） |
| `get_updatable_engines_and_lock` | 533 | 权重同步时取引擎列表 + 互斥锁 |
| `recover_updatable_engines` | 601 | 容错恢复 |
| `check_weights` | 628 | 权重一致性校验（09 篇） |

### 2.3 深入拆解：为什么用 HTTP 通信，而不是让 Ray 直接调用 rollout engine

**关键点：`SGLangEngine`（Ray actor）本身不是推理引擎，它只是一个"进程启动器 + HTTP 客户端适配器"**：

```python
def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    from sglang.srt.entrypoints.http_server import launch_server
    multiprocessing.set_start_method("spawn", force=True)
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()
```

它用 `multiprocessing.Process` 启动了一个**独立的操作系统进程**，运行 sglang 自己的 `launch_server`。真正跑推理的 Scheduler / DetokenizerManager 是**另外的子进程**，进程间用 **ZMQ IPC** 通信，Ray 完全不参与这条链路；sglang 对外暴露的唯一稳定"接口契约"就是 FastAPI 起的这个 HTTP 服务（`/generate`、`/health_generate`、`/update_weights_from_tensor` 等几十个 endpoint，11 篇详解）。所以 `SGLangEngine._make_request` 等方法本质上都是 `requests.post/get(...)`——**不是"不想"直接调引擎，而是"没有别的方式"直接调**：sglang 引擎架构就是"HTTP server 前置于子进程引擎"，Ray actor 无法穿透到它内部的 ZMQ IPC 做函数调用。

**为什么真正的生成请求（海量并发采样）走 HTTP router，而不走 Ray**：

1. **一个"engine"可能跨多个节点，Ray 看不到内部拓扑**：当 `tp_size` 超过单机 GPU 数时，一个 engine 跨 `nnodes` 个节点，只有 `node_rank == 0` 的进程对外暴露 HTTP 端口，真正的跨 rank 协同（TP/PP all-reduce）发生在 sglang 内部靠 NCCL，Ray 完全介入不了这层；
2. **负载均衡/容错/PD-disaggregation 路由是重活，没必要在 Ray 里重造轮子**：sgl-model-gateway（Rust 实现）已经提供了 prefix-cache 感知路由、熔断重试、PD bootstrap room 路由等工业级能力（11 篇 §7）；
3. **支持"外部引擎"（`args.rollout_external`）**：完全独立于 Ray、已经在别处跑起来的 sglang server 可以直接接入同一个 router，对上层 rollout 代码没有任何区别；
4. **性能/并发模型更适合**：一次 rollout 是成千上万个并发采样请求，`httpx.AsyncClient` 连接池 + 流式 SSE 输出比 Ray RPC（GCS 调度 + cloudpickle 序列化 + object store）更轻量，也更适合流式增量输出；
5. **职责分离**：Ray 只承担"控制面"（申请 GPU、起停进程、健康检测重启、权重更新时的 NCCL 握手协调）——低频、非并发热点；"数据面"（海量 generate 请求）完全走 HTTP router，两者解耦干净，也是 sglang 自身单机部署（不依赖 Ray）时的标准用法，slime 只是原样复用。

### 2.4 深入拆解：`rollout_engine_lock` 只为一件事而存在

`RolloutManager.__init__` 创建的这把锁（`slime/ray/utils.py` 的 `Lock`，本质是一个只有布尔状态的 Ray actor，`acquire()` 非阻塞立即返回 True/False，调用方自旋轮询）**只在 NCCL 分布式广播路径（`UpdateWeightFromDistributed`）里用到**：

```python
# update_weight_from_distributed.py
def _update_bucket_weights_from_distributed(self, converted_named_tensors, pbar=None, load_format=None):
    """Lock → broadcast → clear → unlock → pbar++. Lock prevents NCCL deadlock."""
    while not ray.get(self.rollout_engine_lock.acquire.remote()):
        time.sleep(0.1)
    refs = update_weights_from_distributed(...)
    ray.get(refs)
    ray.get(self.rollout_engine_lock.release.remote())
```

**原因**：`connect_rollout_engines_from_distributed` 为**每个 PP rank**建一个独立 NCCL group（04 篇 §5.5 的 `slime-pp_{n}`）。NCCL 集合通信要求同一 group 内所有参与方**以完全相同的顺序**发出对应操作，否则"张量 A 的广播在训练侧发出、但引擎侧先收到了本该属于张量 B 广播的调用"这种错位会导致 NCCL 死锁（hang）。没有这把锁，多个 PP rank（甚至同一 rank 内多个 bucket）可能并发对同一批 rollout engine 发起广播，Ray 端调用到达顺序和 NCCL 端发起顺序不一致就会踩坑。加了这把全局锁后，任意时刻只允许一个"bucket 广播"跟 rollout engine 群的 NCCL group 打交道——代码注释直白写着"Lock prevents NCCL deadlock"。

**它不是唯一的同步机制，也不是所有更新路径都用**：`UpdateWeightFromTensor`（colocated，CUDA-IPC/nixl）接收了这个参数但根本不调用它——每个 bucket 是一次性 Ray remote 调用 + GPU 直接传输，不存在长期存活的 NCCL group，没有"顺序错位死锁"的风险；`UpdateWeightFromDisk*` 同理不用，改用同机 flock 文件锁保证串行。真正保证"权重替换期间不会用半新半旧权重生成"的，是另一套完全独立的机制——`pause_generation`/`continue_generation`（几乎所有更新路径都用，保证推理正确性），`rollout_engine_lock` 只管 NCCL 调用顺序不冲突，两者是完全不同维度的问题，不要混淆。

---

## 3. 函数层：默认 rollout 主循环

`RolloutManager.generate` 最终调用 `args.rollout_function_path` 指定的函数，默认是：

```617:639:slime/rollout/sglang_rollout.py
def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    ...
    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output
```

注意两个设计：(1) `data_source.get_samples` 作为**函数**传入，rollout 函数决定何时取多少数据；(2) `aborted_samples`（本轮被截断的半成品）**回写**进数据源——这是 partial rollout 的关键一环（03 篇详解）。

### 3.1 异步生成主循环 `generate_rollout_async`

```408:448:slime/rollout/sglang_rollout.py
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()
            ...
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                ...
                continue
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)
    ...
    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)
```

机制解读：

- **over-sampling**：每次取 `over_sampling_batch_size` 组提交（sglang_rollout.py:411），在飞任务可以多于目标数 `rollout_batch_size`——先完成先收，收够即停，多出来的请求直接 abort。这就是综述 §2.4 讲的"主动超投抗长尾"（APRIL 同款思路的轻量版）。
- **`asyncio.wait(..., FIRST_COMPLETED)`**：乱序完成，不被最慢样本阻塞。
- **dynamic filter**：每组完成后过 `--dynamic-sampling-filter-path`（如 DAPO 的"全对全错组丢弃"），被丢弃的组会扣减 `remaining_batch_size` 触发补采（sglang_rollout.py:429-433）。
- 收够目标后调用 `abort` 清尾（见 §3.4）。

### 3.2 全局状态单例 `GenerateState`

```84:118:slime/rollout/sglang_rollout.py
class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        ...
        self.semaphore = asyncio.Semaphore(args.sglang_server_concurrency * get_rollout_num_engines(args))
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            ...
        )
```

三个值得注意的点：

1. **并发上限**：`sglang_server_concurrency × 引擎数` 的信号量，防止把 server 打爆；
2. **top-p 回放**：`rollout_top_p != 1.0` 时自动加 `custom_params={"return_top_p_token_ids": True}`（sglang_rollout.py:107-108）——让 SGLang 返回每步采样时的候选核（nucleus），训练侧算 logprob 时只在相同候选集内计算，**保证训推 logprob 可比**（05 篇的 `_build_topp_keep_mask`）；
3. **`dp_rank_context`**（sglang_rollout.py:120-130）：SGLang data-parallel 多 DP rank 时的简单均衡——记录每个 DP rank 的在飞请求数，随机挑一个最空闲的。

### 3.3 单样本生成链路

`generate_and_rm_group`（294-333）→ `generate_and_rm`（224-286）→ `generate`（153-220），层层分工：

- `generate_and_rm_group`：一个 prompt 的 N 个采样（GRPO 的 group）并发执行；若 `--group-rm`（如需要整组比较的比赛型 RM），在这里统一对整组打分；
- `generate_and_rm`：处理 partial rollout 续写样本的 loss mask（231-232 行，见 03 篇）；已完成的样本直接短路返回；分发到自定义生成函数或默认 `generate`；生成后调 RM（`async_rm` / `batched_async_rm`）；
- `generate`：拼 HTTP payload 发给 router：

```174:202:slime/rollout/sglang_rollout.py
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    ...
        payload["input_ids"] = prompt_ids
    ...
    with trace_span(sample, "sglang_generate", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}) as span:
        output = await post(url, payload, headers=headers)
```

- `return_logprob: True` 是 RL 与普通 serving 的核心区别：SGLang 会在 `meta_info["output_token_logprobs"]` 里返回**每个生成 token 的 logprob**，训练侧用它计算重要性采样比（05 篇）；
- 多模态样本改发 `image_data + text`（183-188 行）；
- **session affinity**：`sample.session_id` 存在且 router 策略为 `consistent_hashing` 时，请求头加 `X-SMG-Routing-Key`（195-199 行）——multi-turn agent 的多轮请求因此落到同一台 server，命中本地 KV cache。

响应回填：

```205:218:slime/rollout/sglang_rollout.py
    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    ...
    sample.append_response_tokens(
        args,
        tokens=new_response_tokens,
        log_probs=new_response_log_probs,
        trainable=True,
        meta_info=output["meta_info"],
        text=output["text"],
    )
```

`append_response_tokens`（`slime/utils/types.py:253-314`）会同步维护 `tokens / loss_mask / rollout_log_probs / top-p replay / 状态`，并做严格的长度校验——token 级数据对齐是 RL 正确性的生命线，03 篇详解 `Sample`。

### 3.4 中断：`abort`

```336:372:slime/rollout/sglang_rollout.py
async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []
    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True
    ...
    await abort_servers_until_idle(urls)
    ...
    if args.partial_rollout:
        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            ...
            aborted_samples.append(group)
```

三步：(1) 置全局 `aborted` 标志，新的 `generate_and_rm` 进入时直接标记 ABORTED 返回（245-247 行）；(2) `abort_servers_until_idle`（`slime/backends/sglang_utils/server_control.py`）对所有 server 调 `/abort_request` 并等待排空——SGLang 会中断在飞请求并**返回已生成前缀**；(3) 若开了 `--partial-rollout`，把这些半成品打上 `start_rollout_id` 元数据收集起来，由 `generate_rollout` 回写 buffer，下一轮用新权重**续写**（03 篇）。

---

## 4. 引擎层：训练侧看不见的 SGLang

`slime/backends/sglang_utils/` 中：

- `sglang_engine.py`：`SGLangEngine` Ray actor。`init` 里拼 SGLang 命令行（把 `--sglang-*` 参数透传）拉起 server 子进程；并提供 RL 专用远程方法：`update_weights_from_distributed`（加入 NCCL 组收权重）、`update_weights_from_tensor`（CUDA IPC）、`pause_generation/continue_generation`、`release_memory_occupation/resume_memory_occupation`（offload）、`flush_cache`、`abort_request` 转发等。这些端点是 SGLang 为 RL 系统专门提供的（官方文档「SGLang for RL Systems」）。
- `server_control.py`：跨引擎的批量控制，如 `abort_servers_until_idle`。
- `sglang_config.py`：`--sglang-config` YAML 的解析（异构 server group、per-group 参数覆盖、PD 分离部署）。
- `external.py`：外部托管引擎（`--rollout-external`）的适配。

**为什么 slime 敢只做 server 模式？** 因为 SGLang 把 RL 需要的一切控制面（热更新、显存释放、中断、logprob 回传、top-p/routed-experts 回放）都经 HTTP 暴露了，engine 模式的内存直传优势被抹平，剩下全是 server 模式的结构性收益（隔离、弹性、负载均衡、多模型）。

### 4.1 深入拆解：model / server_group / engine 三层结构

`--sglang-config` YAML 的顶层是一个**列表**，每一项是一个"model"——这在 RL 训练里很常用：可以同时部署 `actor`（接收权重更新，生成 rollout）、`ref`（参考模型，`update_weights: false` 永不更新，算 KL 用）、甚至 `reward`（奖励模型）。它们是**互相独立**的一整套 SGLang 部署（各自的 router 进程、各自的 engine 进程），只是共用同一个 `RolloutManager`/同一个 Ray placement group 编排：

```
SglangConfig                                    (整个 --sglang-config YAML)
 └─ models: list[ModelConfig]                   (第 1 层：model，如 actor / ref / reward)
     └─ server_groups: list[ServerGroupConfig]  (第 2 层：server_group)
         └─ 展开为运行时的 ServerGroup(dataclass)
             └─ all_engines: list[SGLangEngine(Ray actor)]  (第 3 层：engine)
```

**第 2 层 server_group** 是同一个 model 内部不同"角色"的引擎组，最典型场景是 **PD 分离**：`worker_type=prefill, num_gpus=4, num_gpus_per_engine=2` → TP=2，4 卡跑出 2 个 prefill engine；`worker_type=decode, num_gpus=8, num_gpus_per_engine=4` → TP=4，8 卡跑出 2 个 decode engine。`worker_type` 还可以是 `regular`（不做 PD 分离）、`encoder`（多模态 EPD 分离用）、`placeholder`（只占 GPU 位置不真正起引擎）。**一个 `ServerGroup` 内部所有 engine 必须同构**。

**第 3 层 engine** 是 `SGLangEngine` 这个 Ray actor，**一个 engine ≈ 一个逻辑上的 sglang server 实例（一个 TP 副本）**：`num_engines = group_cfg.num_gpus // num_gpus_per_engine`。如果 `num_gpus_per_engine`（TP size）超过单机 GPU 数，一个"逻辑 engine"要跨多台节点部署，`all_engines`（原始列表）包含每个"node-in-engine"的 actor，而 `engines`（对外可见的）每隔 `nodes_per_engine` 取一个——只保留每个逻辑 engine 里 `node_rank==0` 那个，只有它对外暴露 HTTP 端口，节点间协同全靠 sglang 内部的 NCCL。

**一个完整例子**：`actor` model 配置 `worker_type=regular, num_gpus=8, num_gpus_per_engine=4`（TP=4），单机 8 卡：`num_engines=2`，各自在 4 张 GPU 上起 sglang server，都注册到 `actor` model 唯一的那个 router 上，rollout 代码打 `http://{actor_router_ip}:{actor_router_port}/generate` 时 router 在这 2 个 engine 间做负载均衡。如果同时配置了 `ref` model（`update_weights: false`），`RolloutManager.servers` 就是 `{"actor": RolloutServer(...update_weights=True), "ref": RolloutServer(...update_weights=False)}`——训练侧权重更新只会找 `update_weights=True` 的那个。

### 4.2 深入拆解：GPU 分配如何从 placement group 精确落到 `base_gpu_id`

链路：`create_placement_groups`（探测重排，01 篇 §7.1）→ 按 `actor`/`rollout` 切出各自的重排索引段（01 篇 §7.3 的例子：非 colocate 时 rollout 拿后半段，colocate 时训练/推理拿完全重叠的段）→ `ServerGroup.start_engines()` 再从"rollout 专属"的重排数组里，按 `gpu_offset`（跨 model/跨 group 全局累加，避免不同 model 抢同一批卡）+ `i * num_gpus_per_engine_on_node` 算出这个 engine 在数组里的下标 → 取出该下标对应的**真实物理 GPU id**作为 `base_gpu_id`，一路传进 `SGLangEngine.init()` → `ServerArgs(base_gpu_id=...)` → sglang 自己的 `--base-gpu-id` 启动参数。

同 01 篇 §7.2 描述的训练侧机制一样，`RolloutRayActor.options(num_gpus=0.2, ...)` 这里的 `num_gpus=0.2` 也**故意申报一个远小于真实需求的零头值**——真正的资源隔离由 `placement_group_bundle_index` 精确指定的 bundle 保证，`num_gpus` 只是记账数字；同时设置 `NOSET_VISIBLE_DEVICES_ENV_VARS_LIST`，让 Ray **不要**自动重映射 `CUDA_VISIBLE_DEVICES`，因为 sglang 通过 `multiprocessing.Process` 另起子进程，且自己接受显式的 `--base-gpu-id` 精确指定要用哪几张卡——如果 Ray 再自动重映射一层，两套"GPU 编号"体系会打架错位。**一句话**：`pg` 负责"预定哪些物理机器上的哪些 GPU 资源槛"、`bundle_index` 决定 Ray actor 被调度到哪台物理机，而"这台机器上具体用第几号卡"由 `base_gpu_id` 显式传给 sglang，绕开 Ray 的自动分配机制。



## 5. 评测路径

`eval_rollout` / `eval_rollout_single_dataset`（sglang_rollout.py:473-614）与训练路径共用 `generate_and_rm`，差异：数据来自 `--eval-*` 配置的 `EvalDatasetConfig`，每个数据集可有自己的采样参数与 per-sample `generate_function_path`；用 `EVAL_PROMPT_DATASET` 全局缓存避免每个 rollout 重复加载；结果按数据集聚合 rewards/truncated，由 `_log_eval_rollout_data` 记录。**评测数据集配置的完整字段、reward 打分的分发机制、dynamic filter 的完整实现，见 [08_rm_hub_and_eval.md](08_rm_hub_and_eval.md)。**

---

## 6. 深入拆解：prompt 是怎么"复用"的、abort 精确做了什么

### 6.1 `_prepare_prompt_ids`：partial rollout 续写的真正入口

02 篇正文提到"`generate` 里 `sample.tokens` 非空时直接把已有 token 作为输入续写"，具体实现是 `_prepare_prompt_ids`（`sglang_rollout.py:43-62`）：

```python
def _prepare_prompt_ids(sample: Sample, tokenizer, processor: Any) -> list[int]:
    raw_multimodal_inputs = sample.multimodal_inputs or {}
    has_multimodal_inputs = any(value is not None for value in raw_multimodal_inputs.values())
    reuse_existing_input_ids = bool(sample.tokens) and (
        sample.multimodal_train_inputs is not None or not has_multimodal_inputs
    )
    if processor and has_multimodal_inputs and not reuse_existing_input_ids:
        processor_output = processor(text=sample.prompt, **build_processor_kwargs(raw_multimodal_inputs))
        ...
        return prompt_ids
    if reuse_existing_input_ids:
        return sample.tokens          # ← partial rollout 续写：直接把"prompt+已生成前缀"整段送回去当输入
    return tokenizer.encode(sample.prompt, add_special_tokens=False)
```

三个分支的选择逻辑：`sample.tokens` 非空（即这是一条从 buffer 里取出的半成品）且（非多模态，或多模态但已经缓存过 `multimodal_train_inputs`）→ 直接复用已有 token 序列；否则如果有多模态输入且还没 tokenize 过，走 `processor` 编码一次（顺带把非 `input_ids/attention_mask` 的字段，如 `pixel_values`，缓存进 `sample.multimodal_train_inputs`，避免每次续写都重新跑一次视觉 encoder 的预处理）；纯文本首次生成则用 tokenizer 直接编码 prompt。**这是"续写"与"首次生成"共用同一套代码路径的关键分支点**——不是两套逻辑，而是一个函数根据 `Sample` 当前状态自动决定行为。

### 6.2 `abort()` 的精确时序（`sglang_rollout.py:336-372`）

```python
async def abort(args, rollout_id):
    aborted_samples = []
    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True                                  # ① 置全局标志：新请求短路返回 ABORTED

    response = await get(f"http://{router_ip}:{router_port}/workers")   # ② 问 router 要在线 worker 列表
    urls = [w["url"] for w in response["workers"]]
    await abort_servers_until_idle(urls)                    # ③ 对每个 server 发 /abort_request 并等排空

    count = 0
    while state.pendings:                                   # ④ 等所有本地挂起的 asyncio task 真正 return
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        if not args.partial_rollout:
            continue
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id   # ⑤ 打上"从哪一轮开始半成品"的标记
            aborted_samples.append(group)
            count += len(group)
    return aborted_samples
```

容易忽略的两点：

1. **`assert not state.aborted`**：`abort()` 只能在一次 rollout 里调用一次（`GenerateState` 是单例，跨轮由 `reset()` 清空），如果被误调两次会直接触发断言——这是"防御式编程"的例子，宁可显式崩溃也不要一个已经在收尾的批次被再次打断产生数据错位。
2. **`start_rollout_id` 只在样本第一次被 abort 时打**（`"start_rollout_id" not in sample.metadata`）：一条样本可能在 partial rollout 下被打断、续写、又被打断多次，`start_rollout_id` 始终记录**它第一次开始半成品生命周期时的 rollout 步数**，而不是最近一次——这个字段与 `weight_versions` 字段配合，能诊断"一条最终样本经历了从第几轮到第几轮的权重跨度"，是排查 partial rollout 训练异常时最先要看的两个字段。

### 6.3 一个数字例子：`over_sampling_batch_size` 如何驱动 abort 的触发时机

假设 `rollout_batch_size=16`（要 16 组）、`n_samples_per_prompt=4`、`over_sampling_batch_size=20`（每次多要 20 组）：

1. 循环第一次进入 `while state.remaining_batch_size < target_data_size(16)`：`remaining_batch_size` 初值 0 < 16，取 20 组样本提交（`remaining_batch_size` 变为 20，超过 16 也没关系，循环条件已经满足退出内层 while）；
2. 进入 `asyncio.wait(FIRST_COMPLETED)`，样本陆续完成，`data` 列表逐渐填满（每完成一组检查 dynamic filter，未被丢弃则 `data.append(group)`）；
3. 因为一开始就提交了 20 组（比目标 16 组多 4 组"富余"），大概率不需要二次补采就能凑够 16 组——但一旦有组被 dynamic filter 丢弃（比如判定"组内 reward 方差为 0"），`remaining_batch_size` 会减 1，若减到低于 16 又会触发内层 while 再取样本补齐；
4. 一旦 `len(data) >= 16`，外层 `while len(data) < target_data_size` 退出主循环，调用 `abort(args, rollout_id)` ——此时可能还有 20-16=4 组左右（受 dynamic filter 影响，实际数字会波动）尚在生成中，它们的中间态会被 `abort` 收集成 `aborted_samples` 回写 buffer。

**直觉**：`over_sampling_batch_size` 越大于 `rollout_batch_size`，"抢跑"的富余度越大，长尾样本被提前甩出主路径的概率越高，但同时被 abort 掉的半成品也越多（如果不开 partial rollout，这些半成品直接丢弃，等价于浪费的算力）。

---

## 7. 小结

> 本篇讲的是客户端视角；服务端（SGLang 内部）如何实现这些端点，见 [11_engine_internals_sglang.md](11_engine_internals_sglang.md)。

- server 模式 = 训练侧只面对一个 router URL + 一组 HTTP 控制端点；
- model/server_group/engine 三层结构支持多模型部署（actor/ref/reward 各自独立一整套 SGLang），GPU 分配从 placement group 探测重排到 `base_gpu_id` 全程可追踪；
- `rollout_engine_lock` 只为防止 NCCL 广播路径的调用顺序错位死锁而存在，与保证"不读到半更新权重"的 `pause_generation` 是完全不同维度的两把锁；
- 生成主循环 = over-sampling 提交 + 乱序完成 + 动态过滤 + abort 清尾 + 半成品回 buffer；
- RL 相对普通 serving 的三个增量：`return_logprob`、top-p/routed-experts 回放、会话亲和与中断续写；
- `_prepare_prompt_ids` 的三分支选择是"续写 vs 首次生成"这两种看似不同的路径实际共用一套代码的关键；`abort()` 的幂等断言与 `start_rollout_id` 首次打标语义是排查 partial rollout 问题的入口。
- 下一篇（03）看这些数据如何被组织成 `Sample`、如何在 buffer 中流转、以及 fully async 如何进一步解耦。
