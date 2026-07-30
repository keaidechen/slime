# 02. Scheduler、请求状态机与连续批处理

## 1. 三个核心对象

- `Req`：`srt/managers/schedule_batch.py:713`，单请求的 token、采样、KV、状态；
- `ScheduleBatch`：同文件约 1822 行，一次 GPU forward 的请求集合与 tensor metadata；
- `Scheduler`：`srt/managers/scheduler.py:326`，队列、内存、batch 和执行循环。

不要把 API request、`Req` 和 batch tensor 混为一体。一个 `Req` 会经历许多 `ScheduleBatch`；一个 batch 每轮可合并不同生命周期的请求。

## 2. Prefill/extend 与 decode

- extend/prefill：为一个请求处理多个尚未有 KV 的 input token；
- decode：通常每请求每轮处理一个新 token；
- chunked prefill：长 prompt 分成多轮 extend，避免独占 GPU/延迟尖峰。

prefill 的大 GEMM 算术强度高；decode 的 batch×1 GEMM 和 KV 读取更依赖带宽/launch latency。混在同一 batch 的策略决定 TTFT 与 ITL 的权衡。

## 3. 正常 scheduler loop

`event_loop_normal()` 在 `scheduler.py:1552`，主干非常清晰：

```python
while True:
    recv_reqs = recv_requests()
    process_input_requests(recv_reqs)
    plan = get_next_batch_to_run(running_batch, last_batch)
    batch = plan.batch_to_run
    if batch:
        result = run_batch(batch)
        process_batch_result(batch, result)
    else:
        on_idle()
```

复杂度集中在 `get_next_batch_to_run` 与 result processing：它们必须维护 request、KV allocator、prefix cache、grammar/spec 状态的一致性。

## 4. overlap scheduler

`event_loop_overlap()` 让 GPU 执行当前 batch 时，CPU 处理上一 batch 的结果并准备下一轮。关键是“结果落后一拍”：

```text
iteration t:
  GPU run batch[t]
  CPU process result[t-1]
  prepare/schedule batch[t+1]
```

代码用 `result_queue` 保存 `(batch.copy(), result)`，因为原 batch 会继续变化。若直接保存可变 batch 引用，处理 result 时可能已指向下一状态。

### WAR barrier

overlap 下 CPU 可能写 scheduler/共享 buffer，而 GPU forward 仍在读，形成 write-after-read hazard。源码 `_apply_war_barrier()` 优先等待 forward 发布的 read-done event，fallback 才等待整个 forward stream。

这是罕见但关键的并发点：正确性不能靠“CUDA 通常很快”或 Python GIL。修改共享 metadata 时要知道哪个 stream、哪个 iteration 在读。

### 何时禁用 overlap

源码会在某些 batch 边界禁用：

- 连续 prefill 为优先完成前一个 TTFT；
- grammar/speculative 的状态依赖需要同步；
- DP attention 所有 ranks 必须做一致决策，否则 collective deadlock；
- 需要安全 flush/compact 的边界。

## 5. 构建 prefill batch

`PrefillAdder` 位于 `srt/managers/schedule_policy.py:441`。它本质上做受约束装箱：

```text
约束：
  可用 KV token 数
  本轮 max prefill tokens
  max running requests
  已运行 decode 的保留预算
  chunk 限制 / grammar / multimodal 等
目标：
  选出 waiting req，兼顾策略、缓存命中、TTFT 与吞吐
```

对候选请求不能只看 input length，要减掉 prefix cache hit 后真正需要 extend 的 token 数。

## 6. running batch 与请求推进

一次 generation result 后，每个请求可能：

- 接受一个或多个 token；
- 因 EOS/stop/max length/abort 完成；
- grammar FSM 前进；
- speculative 状态更新；
- KV length 增长；
- streaming 输出；
- 留在 running batch 进入下轮 decode。

finish 顺序很重要：先确定哪些 KV 可纳入 prefix cache，再释放 request slot 和 lock；先完成 detokenize 所需 token metadata，再清理输出状态。

## 7. 内存不足与 retract

当 KV 空间不足，scheduler 可能：

- 驱逐未锁定 prefix cache；
- 减少本轮 prefill；
- retract/preempt 部分 running request，使其之后重算；
- 拒绝超长请求。

retract 不是“暂停 GPU kernel”，而是释放其部分资源并把请求放回可恢复状态。需要保证 output token、已缓存前缀、sampling RNG 与 grammar 状态可重建，否则会产生不一致输出。

## 8. 调度策略评审

任何新策略都要评估：

- 无共享前缀的公平性；
- 高共享前缀的 cache locality；
- 大请求是否饿死小请求，反之亦然；
- decode ITL 是否被长 prefill 破坏；
- KV 满载时是否稳定；
- DP ranks 是否决策一致；
- abort/timeout 是否及时回收；
- p99 而非只看均值。

## 9. `Req` 源码中的不变量

`schedule_batch.py:713`：

```python
self.origin_input_ids = origin_input_ids
self.output_ids = array("q")
self.full_untruncated_fill_ids = array("q")
self.extend_range = None
self.kv_committed_len = 0
self.kv = None
self.extend_batch_idx = 0
self.decode_batch_idx = 0
```

源码注释特别强调 `output_ids` **append-only**。`_refresh_fill_ids` 通过长度推断已有多少 output 已进入 full sequence；若原地改写一个 token 而长度不变，fill ids 会静默错误。

字段语义：

- `origin_input_ids`：用户/processor 产生的逻辑输入；
- `output_ids`：已经提交的生成 token；
- `full_untruncated_fill_ids`：调度/模型真正需要的完整序列视图；
- `extend_range`：本轮需要计算 KV 的区间；
- `kv_committed_len`：已经安全提交的 KV 边界；
- `extend/decode_batch_idx`：跨轮追踪执行次数。

speculative、chunked prefill、SWA 都依赖“committed”和“临时”边界不混淆。

## 10. normal 与 overlap 源码对照

normal：

```python
result = self.run_batch(batch)
self.process_batch_result(batch, result)
```

overlap：

```python
batch_result = self.run_batch(batch)
self._apply_war_barrier()
self.result_queue.append((batch.copy(), batch_result))

if self.last_batch:
    pop_and_process()
```

为什么 `batch.copy()`：Scheduler 下一轮会修改 running batch、seq lens、request 列表；result 必须和 launch 时的 snapshot 对齐。

为什么 sample 在处理上一批之后：

```python
self.launch_batch_sample_if_needed(batch_result, batch)
```

grammar/future token 等状态可能依赖上一批实际结果。调换这两步会让下一轮 mask 或 token 错一拍。

## 11. future token 的直觉

CPU 要在 GPU result 真正回到 host 前准备下一批，但下一 token 尚未知。overlap scheduler 可先为它预留位置/构造 future placeholder，等 GPU 结果到达后填值。

需要区分：

```text
位置已分配 ≠ token 值已提交 ≠ KV 已提交
```

任何读取 token 值的逻辑（grammar、stop、hash key）都必须等对应 event/result；只需要地址的 attention metadata 可以提前准备。

## 12. PrefillAdder 的预算例

假设：

```text
max_prefill_tokens=4096
可用 KV=6000
running decode 100 requests，需保留约 100 token/轮及安全余量
waiting:
  A input=3000, prefix_hit=2000 → extend=1000
  B input=2500, prefix_hit=0    → extend=2500
  C input=4000, prefix_hit=3500 → extend=500
```

cache-aware 策略可选 A+C（1500 真正计算 token），仍有容量；按原始 input length 排序则可能误认为 A/C 很贵。若再加入 priority/fairness，不能让不断到来的高命中请求让 B 永久饥饿。

## 13. retract 的状态回滚

retract 某请求至少要处理：

- running batch 移除；
- 未提交 output/draft token 丢弃；
- 临时 KV 释放；
- 已共享 prefix 保留并正确解锁/重锁；
- request pool slot 是否保留取决于重算策略；
- grammar/RNG 回到 committed token 边界；
- 重新进入 waiting queue 的优先级与等待时间。

最危险的是只释放 KV，但 `kv_committed_len` 未回退，重入时 scheduler 认为 KV 已存在。

## 14. 调度日志最小字段

每 iteration 建议低频采样：

```text
iteration
forward_mode
num_waiting / num_running
batch_size
extend_tokens / decode_tokens
available_kv
prefix_hit_tokens
retracted_reqs
cuda_graph_replay_size
schedule_ms / forward_ms / result_process_ms
```

有了这些字段，TTFT spike 才能区分排队、长 prefill、cache miss 和 CPU scheduler。

## 15. 本章延伸阅读

- [SGLang v0.4 Zero-Overhead Scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)：包含 overlap 时间线和 Nsight 图。
- [Mini-SGLang](https://www.lmsys.org/blog/2025-12-17-minisgl/)：用更短代码理解 chunked prefill 与 overlap。
- [Orca：Iteration-level Scheduling](https://www.usenix.org/conference/osdi22/presentation/yu)：continuous batching 的经典系统背景。
- [SGLang Bench Serving Guide](https://docs.sglang.io/docs/developer_guide/bench_serving)：构造 arrival rate、并发和长度 workload。
