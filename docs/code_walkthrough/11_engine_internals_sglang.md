# 11 引擎内部实现（SGLang 篇）：RL 专用端点在服务端发生了什么

> 衔接 [02_rollout_sglang_server_mode.md](02_rollout_sglang_server_mode.md) 与 [04_weight_sync_and_memory.md](04_weight_sync_and_memory.md)。
> 02/04 篇讲的是 slime（客户端）如何调用；本篇深入仓库根目录 vendored 的 `sglang/` 源码，讲服务端收到请求后做了什么。读完你会理解"server 模式为什么可行"——这些端点是 SGLang 专门为 RL 系统造的。

---

## 1. 三层调用链总览

SGLang server 内部是多进程架构，一个 RL 端点请求的路径是：

```
HTTP (FastAPI)                    entrypoints/http_server.py
  → TokenizerManager (ZMQ IPC)    managers/tokenizer_control_mixin.py
    → Scheduler                   managers/scheduler_components/weight_updater.py
      → TpModelWorker             managers/tp_worker.py
        → ModelRunner.WeightUpdater  model_executor/model_runner_components/weight_updater.py
```

- TokenizerManager 与 Scheduler 之间用 ZMQ IPC 传 `io_struct.py` 里定义的 msgspec 消息（`*ReqInput`/`*ReqOutput`）；
- 多 DP 时用 `FanOutCommunicator` 扇出到各 DP rank 并合并结果；
- 权重相关的"重活"最终都落到 `ModelRunner.WeightUpdater`——它是理解本篇的核心类。

---

## 2. `update_weights_from_distributed`：NCCL 广播的服务端

slime 侧流程见 04 篇 §2；这里看对称的另一端。

### 2.1 HTTP 路由与请求体

```1331:1345:sglang/python/sglang/srt/entrypoints/http_server.py
async def update_weights_from_distributed(
```

请求体 `UpdateWeightsFromDistributedReqInput`（`managers/io_struct.py:1565-1580`）字段：`names / dtypes / shapes / group_name / flush_cache / abort_all_requests / weight_version / load_format / torch_empty_cache`。注意——**HTTP 里只传元数据（名字/dtype/shape），张量本体走 NCCL**，这与 04 篇 slime 侧"元数据走 Ray、张量走 NCCL"严格对称。

### 2.2 TokenizerManager：并发闸门

`tokenizer_control_mixin.py:418-446` 做两件保命的事：

1. `abort_all_requests=True` 时先中断全部在飞请求（428-429 行）；
2. **读写锁** `model_update_lock.writer_lock`（431-439 行）：如果引擎已被 `/pause_generation` 暂停就直接放行；否则获取写锁，**阻止新请求进入生成循环**后才发 IPC。这保证"权重换到一半被采样"在服务端也被结构性地杜绝（slime 侧的 pause/flush/continue 是第一道，这是第二道）。

成功后记录 `weight_version`（442-444 行）——04 篇 disk 模式的版本对账依赖它。

### 2.3 Scheduler / TpWorker

- `scheduler_components/weight_updater.py:138-151`：包一层 Prometheus 指标 `weight_load_duration_seconds`，成功后按 `recv_req.flush_cache` 调 `flush_cache_after_weight_update`（103-108 行）——**旧权重算出的 radix cache 必须作废**；
- `tp_worker.py:162-174`：透传给 `model_runner.weight_updater`。

### 2.4 ModelRunner.WeightUpdater：建组与接收

**建组**（`model_runner_components/weight_updater.py:67-111`）：

- `_model_update_group` 是 `dict[group_name → ProcessGroup]`——**多个命名 NCCL 组并存**（对应 slime 每 PP rank 一个 `slime-pp_{n}` 组，04 篇 §2.1）；
- rank 拓扑约定：**rank 0 = 训练引擎，其余 = 推理各 TP rank**（`rank = rank_offset + self.tp_rank`）；
- `init_custom_process_group`（`utils/common.py:2463` 起）：基于 torch `_new_process_group_helper` 封装（借鉴 OpenRLHF），用 TCPStore rendezvous + `PrefixStore` 建一个**独立于推理 TP 组**的新 NCCL 组——推理自身的 TP 通信与权重接收互不干扰。

**接收**（`weight_updater.py:222-283`）：按 `names/dtypes/shapes` 逐个 `torch.empty` 分配 → `dist.broadcast(..., src=0)` 从训练源接收 → 按名字写回模型参数。还有一条 **bucket 路径**：训练侧把多张量 flatten 进一个大 buffer 一次广播，服务端按元数据切片 reshape 落位——即 `FlattenedTensorBucket`（`weight_sync/tensor_bucket.py:19`），与 slime 侧 `update_weight_buffer_size` 分桶（04 篇 §2.2）配套。

**对照记忆**：slime 侧 `update_weight_from_distributed.py:240-355`（pause→分桶→广播）与本节（锁→empty→broadcast→flush_cache）合起来就是一次完整的权重同步。

### 2.5 深入拆解：`FlattenedTensorBucket`（`weight_sync/tensor_bucket.py:19-89`）——为什么要把多个张量拼成一个再传

不分桶的朴素做法是"每个参数发一次 NCCL broadcast"——一个几百层的模型有几千个参数张量，每次 broadcast 都有固定的 kernel launch + 同步开销，几千次这样的小广播会让延迟被"次数"而不是"总字节数"主导。`FlattenedTensorBucket` 的做法是先在发送侧把一组张量拼成一条：

```python
for i, (name, tensor) in enumerate(named_tensors):
    flattened = tensor.flatten().view(torch.uint8)      # 按字节视图摊平（不同 dtype 也能拼在一起）
    metadata_obj = FlattenedTensorMetadata(
        name=name, shape=tensor.shape, dtype=tensor.dtype,
        start_idx=current_idx, end_idx=current_idx+numel, numel=numel,
    )
    current_idx += numel
self.flattened_tensor = torch.cat(flattened_tensors, dim=0)   # 一次 cat，一次 broadcast
```

`.view(torch.uint8)` 是这里的关键技巧：把张量按**原始字节**重新解释（不做数值转换），这样即使一个桶里混有 `bf16` 的权重和 `fp8` 的另一个权重，也能被拼进同一个 `uint8` 一维数组一起传输——**通信层完全不关心 dtype，只搬字节**；接收端按 `FlattenedTensorMetadata` 里的 `start_idx/end_idx/shape/dtype` 切片、`view(dtype)`、`reshape(shape)` 还原出每个原始张量。这与 slime 侧 `update_weight_buffer_size` 分桶（04 篇 §2.2）是同一机制的两端：训练侧按桶大小把参数分组拼包，引擎侧收到后按同一份 metadata 拆开——**metadata 本身很小（只有 name/shape/dtype/offset），走 HTTP／Ray 控制面传递即可，真正的大字节流走一次 NCCL**。

**直觉**：如果模型有 3000 个参数张量、单次 broadcast 固定开销 0.5ms，不分桶要 1.5 秒纯开销；分桶成 30 个大 bucket（每桶 100 个参数拼一起）就只需要 30 次广播的固定开销（15ms），网络传输时间由总字节数决定不受影响——分桶省的是"次数税"，不是"流量税"。

---

## 3. `update_weights_from_tensor`：CUDA IPC 的服务端

colocate 模式的端点（http_server.py:1309；`tokenizer_control_mixin.py:476`；`weight_updater.py:319`）。客户端把张量经 `MultiprocessingSerializer` 序列化（本质是 CUDA IPC handle——显存指针的跨进程"钥匙"）传来，服务端反序列化还原出**同一块物理显存**，直接拷进模型参数。零网络、零序列化大 buffer——这是 colocate 下 7 秒级同步的来源（04 篇 §3）。

---

## 4. `release/resume_memory_occupation`：显存错峰的服务端

`tokenizer_control_mixin.py:757`（release）、`scheduler_components/weight_updater.py:202`。带 `tags` 参数（如 `["weights"]` / `["kv_cache"]`）：

- **weights**：把模型权重从显存卸下（经 torch_memory_saver 挂起到 CPU 或直接释放）；
- **kv_cache**：释放 radix cache 的 KV 池显存；
- resume 时按 tag 恢复。

slime 的 `onload_weights/onload_kv` 拆成两步（04 篇 §4.1）正是为了利用 tags 的粒度：权重先恢复就能开始收新权重，KV 池最后重建。SGLang 内部同样基于 **torch_memory_saver** 实现挂起——slime 训练侧 `LD_PRELOAD` 注入的（04 篇 §4.2）与它同源，两边一个 hook CUDA 分配、一个显式调用，构成 colocate 显存管理的两端。

---

## 5. `abort_request` 与已生成前缀的保留

slime 的 partial rollout 依赖它（02 篇 §3.4）。服务端行为：

1. scheduler 把目标请求从运行队列移除，释放其 KV slot；
2. **已生成的 token 不丢**：响应照常返回，`meta_info.finish_reason = "abort"`；
3. 客户端（`slime/utils/types.py:410-416`）把 `"abort"` 映射为 `Sample.Status.ABORTED`，带着已有 tokens 回 buffer 续写（03 篇 §3）。

值得注意的对称设计：`update_weights_from_distributed` 请求体里也有 `abort_all_requests` 字段——权重更新与 abort 可以合并成一次原子操作。

---

## 6. logprob 与回放数据的产生

02 篇讲了客户端传 `return_logprob: True`；服务端链路：

- **output_token_logprobs**：采样阶段每个被采 token 记录 `(logprob, token_id, text)`，随响应放进 `meta_info["output_token_logprobs"]`——RL 的 TIS 与重要性比值的原料（05 篇）；
- **`return_top_p_token_ids`**：top-p 采样时返回每步的候选核（nucleus）token 集合——训练侧据此把 logprob 限制在相同候选集内（slime `loss.py` 的 `_build_topp_keep_mask`，05 篇 §2.2）；
- **`return_routed_experts`**：MoE 模型返回每 token 的专家路由决策——slime 的 routing replay（06 篇 §5）消费它，强制训练侧走相同路由，消除训推 router 数值差。

这三个"回放"端点是 true on-policy 的基石：推理引擎把采样时的**完整随机性上下文**（核集合、路由决策、logprob）都交还给训练侧。

---

## 7. Router：会话亲和的实现

slime 启动的是 **sgl-model-gateway**（Rust crate，`sglang/sgl-model-gateway/`），02 篇的 `X-SMG-Routing-Key` 头在这里消费：

- 策略工厂（`src/policies/factory.rs:85`）注册 `consistent_hashing` 策略（`policies/consistent_hashing.rs`）；
- 优先级（consistent_hashing.rs:128 注释）：有 `X-SMG-Routing-Key` 时按 key 做一致性哈希选 worker（O(log n)），否则退化到常规负载均衡；
- `header_utils.rs` 负责提取/透传该头。

一致性哈希的好处：worker 增减（扩缩容、故障恢复）时只有少量 session 的映射变化——multi-turn agent 的 KV 亲和最大化保留。这与 slime 侧 `sample.session_id`（02 篇 §3.3）是同一枚硬币的两面。

---

## 8. 其余 RL 端点速查

| 端点 | HTTP 路由 | 作用 |
|---|---|---|
| `/init_weights_update_group` | http_server.py:1277 | 预建 NCCL 组（也可由 update 请求隐式建） |
| `/destroy_weights_update_group` | 同上附近 | 销毁组 |
| `/update_weights_from_disk` | — | 从磁盘 checkpoint 重载（04 篇 disk 模式） |
| `/pause_generation` / `/continue_generation` | — | 生成闸门（配合权重更新的第一道锁） |
| `/flush_cache` | — | 清空 radix cache |
| `get_weight_version` | — | 版本对账（04 篇 §3、09 篇 §2.1） |

Engine 类（`entrypoints/engine.py:1244-1443`）还提供同名的进程内 API（非 HTTP 模式时用），与 HTTP 路由共享同一套 TokenizerManager 实现。

---

## 9. 小结

- 服务端 = 路由（FastAPI）→ 并发闸门（TokenizerManager 读写锁）→ 指标/缓存善后（Scheduler）→ 执行（ModelRunner.WeightUpdater）；
- 三个 RL 专属能力：命名 NCCL 组并存、IPC handle 还原、torch_memory_saver 显存挂起；
- 三个回放数据：logprob、top-p 核、MoE 路由——true on-policy 的数据基础；
- 客户端（02/04 篇）与服务端（本篇）的每一对调用都是严格对称的协议，建议对照着读。
