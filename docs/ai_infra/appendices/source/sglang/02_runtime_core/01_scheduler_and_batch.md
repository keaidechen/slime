# 2.1 Scheduler、请求状态机与连续批处理

相关基础：[Transformer 与 KV](<../../../../05_Inference/19_生成执行与采样.md#s1>)。


如果让每个请求独占 GPU 直到生成完，短请求会被长请求挡住；如果把一批请求固定到全部完成，又会让先完成的位置空转。连续批处理把调度单位缩成“一次模型迭代”，用更频繁的准入、退出与资源核算换取设备利用率。下面先用普通自回归、无推测解码的路径，随后再加入异步重叠。

## 1. 三个核心对象

- `Req`：`srt/managers/schedule_batch.py:713`，单请求的 token、采样、KV、状态；
- `ScheduleBatch`：同文件约 1822 行，一次 GPU forward 的请求集合与 tensor metadata；
- `Scheduler`：`srt/managers/scheduler.py:326`，队列、内存、batch 和执行循环。

不要把 API request、`Req` 和 batch tensor 混为一体。一个 `Req` 会经历许多 `ScheduleBatch`；一个 batch 每轮可合并不同生命周期的请求。

## 2. Prefill/extend 与 decode

- extend/prefill：为一个请求处理多个尚未有 KV 的 input token；
- decode：通常每请求每轮处理一个新 token；
- chunked prefill：长 prompt 分成多轮 extend，避免独占 GPU/延迟尖峰。

TTFT（Time To First Token，首 token 延迟）从约定的请求起点计到首个输出 token，通常包含排队、输入处理和 Prefill。客户端与服务端的起点可能不同，报告时应明确。

prefill 的大 GEMM 算术强度高；decode 的 batch×1 GEMM 和 KV 读取更依赖带宽/launch latency。混在同一 batch 的策略决定 TTFT 与 ITL 的权衡。

GEMM 是 General Matrix Multiplication（通用矩阵乘法）。若线性层输入 `X` 为 `[N,d]`、权重 `W` 为 `[d,h]`，输出 `XW` 为 `[N,h]`；`N` 是本轮 token 数，`d/h` 是输入/输出特征维度。每个输出元素需要 `d` 次乘法及约 `d` 次加法，总工作约 `2Ndh` FLOP（浮点运算次数）。权重的 `dh` 个数可供 N 行复用，N 较大时更容易摊薄权重读取成本。

例如两个请求 A/B 分别还需 2/4 个输出，忽略 prefill：固定 batch 的四轮有效请求数是 `2,2,1,1`，最后两轮留有空位；连续批处理允许 A 完成后把新请求 C 加进来。它减少空转，但每轮都要维护请求顺序、长度、KV 和采样状态。若新请求需要很长 prefill，插入它反而可能拉长 B 的下一个 token 间隔，所以“及时准入”仍受预算限制。

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

用假设时长估计收益：CPU 调度与结果处理每轮 `C=2 ms`，GPU forward 每轮 `G=6 ms`。完全串行三轮需 `3(C+G)=24 ms`；若各轮可理想流水且依赖就绪，三轮约需 `C+G+2×max(C,G)=20 ms`，长序列稳态轮距从 8 ms 降到 6 ms。`max` 来自 CPU/GPU 同时工作时较慢的一方决定推进速度。若 grammar 每轮必须等 GPU 结果回到 CPU 才能准备下一轮，或 barrier 覆盖整段 forward，这个理想重叠就无法完全实现。

GIL（Global Interpreter Lock，Python 全局解释器锁）约束 Python 线程执行字节码，不负责同步 CUDA stream。Stream 是设备命令的有序队列；同一队列保持顺序，不同队列之间的依赖通常要用 event（事件）显式表达。

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

#### 把准入条件写成两本预算

令请求 `i` 的完整待处理长度为 `L_i`、可复用长度为 `H_i`，则本轮未命中工作量是 `E_i=L_i-H_i`，单位均为 token。简化为 page size=1、无额外预分配：选中集合 `S` 必须满足 `sum(E_i)≤P`，其中 P 是本轮 prefill token 预算；还需 `sum(E_i)+R≤F`，其中 F 是当前可分配 KV slot 数，R 是留给已运行请求的预算。两条不等式分别保护单轮工作量和长期内存，不能互相代替。

小例：`P=5`、`F=7`、`R=3`，候选 A 的 `(L,H)=(4,2)`、B 为 `(3,0)`、C 为 `(3,2)`。A+B 的工作量是 5，符合 P，却需要 `5+3=8` 个 slot，超过 F；A+C 只需 `2+1+3=6`，两项都通过。即便还有空闲 request slot，也不能接纳 A+B。

这只是预算直觉，实际 `PrefillAdder` 还估计未来输出、页取整、可驱逐空间和特殊模型状态。命中的 H 个 KV 已占据池空间；它们不能既算作“已复用”又算作 free。反过来，满足 token 预算也不意味着两批耗时相同：历史越长，新 Query 需要读取的 KV 越多。

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

RNG（Random Number Generator，随机数生成器）决定随机采样。回撤恢复时要区分“重建已输出 token 的 KV”和“为下一 token 新采样”，不能因为重算而额外消耗同一输出位置的随机数。

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
- `kv_committed_len`：请求侧 KV 长度账本；部分准备路径会提前推进本轮目标长度，设备可读性仍由执行依赖与结果边界保证；
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

例如某请求已输出 `[20,21]`，设备容量不足时回撤。恢复时输入应是原 prompt 加这些已提交 token，用来重建缺失的 KV；不能再把重算所得的 `20,21` 发给客户端，也不能把它们当作两次新的随机采样。grammar（语法状态机）也应处在已提交前缀之后，而非再消费一遍这些 token。回撤省下的是占用时间和当下容量，增加的是未来重算、排队以及恢复状态的复杂度。

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
