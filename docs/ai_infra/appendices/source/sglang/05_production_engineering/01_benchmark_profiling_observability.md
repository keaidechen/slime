# 5.1 Benchmark、Profiling 与可观测性

相关基础：[Transformer 与 KV](<../../../../05_Inference/19_生成执行与采样.md#s1>)。


## 入门例子：一条容量曲线比一个峰值更有用

固定模型、输入/输出长度、采样与缓存设置，从低并发开始逐点提高。每点记录请求数、测量时间、完成/失败、吞吐、TTFT/ITL 分布和 KV/queue。达到饱和后，吞吐可能基本不变而尾延迟迅速变坏。

连续前缀重复的压测与随机前缀压测是不同 workload；要标明缓存热度。客户端发送不足也可能让服务看起来很空闲，不能据此判断 GPU 算力有余。

吞吐和延迟曲线共同定义容量；负载与稳定测量区间决定曲线之间能否比较。工具的观测模型在性能专题中解释，本篇连接 SGLang 的状态与因果关系。

## 机制与实现

## 1. Benchmark 先描述 workload

SLO（Service Level Objective，服务等级目标）是服务承诺的可量化目标，例如“至少 99% 的请求首 token 延迟小于 1 秒”；阈值、统计窗口、失败请求和分母都属于定义。

至少记录模型/精度、硬件/互联、SGLang commit、并行配置、输入/输出长度分布、并发或到达过程、cache 命中率、streaming 和 SLO。缺少这些条件的 tokens/s 无法复现。

性能指标同时报告：

goodput（有效吞吐）只计入同时满足约定质量、延迟和成功条件的产出，再除以时间。提高总吞吐却增加超时，未必提高 goodput。

- TTFT、ITL/TPOT、E2E 的 p50/p95/p99；
- request/s、input/output/total token/s；
- 满足 SLO 的 goodput；
- GPU/CPU/网络利用率和峰值显存；
- 错误、超时、abort 和 retract 数量。

### 用四个时间戳区分指标

令请求在客户端发出时间为 t0，三个输出 token 到达时间依次为 t1、t2、t3，采用同一单调时钟、单位秒。假设 `t0=0,t1=0.100,t2=0.130,t3=0.190`：

| 指标 | 公式 | 本例 |
|---|---|---|
| TTFT，首 token 延迟 | `t1-t0` | 100 ms |
| ITL，相邻 token 间隔 | `t2-t1` 与 `t3-t2` | 30 ms、60 ms |
| TPOT，首 token 后平均每 token 时间 | `(t3-t1)/(3-1)` | 45 ms/token |
| E2E，端到端时间 | `t3-t0`，本例忽略最后协议尾部 | 190 ms |

分母为 2，是因为三个 token 只有两个间隔；只有一个输出 token 时 TPOT 无此定义，不应记作 0 后混入平均。实际 SSE 消息可能合并 token，客户端只能测到 chunk 到达时间时应标注口径。服务端 GPU 首 token 时间又不包含最后网络传输，不能拿它与客户端 TTFT 直接互换。

吞吐是窗口总产出除以窗口长度。例如 10 s 内完成 20 条请求、共产生 100 个输出 token，得到 2 req/s 和 10 output token/s。若只有 15 条同时成功且满足约定 TTFT、TPOT、质量条件，请求 goodput 为 1.5 req/s；若计算 token goodput，则还需只统计这 15 条里的 token。不能把失败请求从延迟样本删除后又宣称全部请求满足 SLO。

p95 表示约 95% 的样本不超过该值，不是“最慢 5% 的平均值”。十条样本不足以稳定估计 p99；重复试验、样本量与统计方法也应进入报告。

## 2. 找容量拐点

逐步提高并发或 arrival rate，观察 throughput 何时趋于饱和、waiting queue 何时持续增长、TTFT/ITL 何时越过 SLO。拐点前保留安全裕量，而不是把峰值点当生产容量。

固定并发是 closed-loop（闭环）压测：一个请求结束才补下一个，服务变慢后发出速率会自动降低。固定到达过程是 open-loop（开环）：按预定时间发请求，更容易暴露真实突发与排队，两者不应放在同一横轴上直接比较。

假设服务每秒稳定完成 μ=10 条请求，外部每秒到 λ=12 条，且请求工作量相近，则积压约按 `λ-μ=2 req/s` 增长；30 s 后约多 60 条。若把这 60 条都留在队列，它们即使不立即占 KV，也会占请求状态并增加等待。现实 μ 会随 batch、长度和缓存变化，这个算例只说明“排队不能创造容量”。即使 λ 略低于平均 μ，随机突发也可能使尾延迟越线。

冷缓存与热缓存应分别实验：先跑不同前缀测重算成本，再跑明确共享前缀测复用收益；不能优化前全冷、优化后全热。固定并发若还受客户端 CPU 限制，会把服务容量低估，因此需同时记录实际发送率。

## 3. 四层观测

| 层 | 关键观测 |
|---|---|
| API/路由 | QPS、错误码、排队、重试、TTFT/E2E |
| Scheduler/KV | waiting/running、batch token、retract、cache hit/eviction |
| Model/通信 | forward、kernel、collective、graph hit |
| 系统 | 显存、GPU 利用率、CPU、网络、page fault |

指标必须能够沿 request id/trace id 关联，否则只能看到相关性，不能定位因果。

Metric（指标）是聚合计数或分布，适合发现趋势；trace（追踪）记录一条请求各阶段及依赖，适合定位单次慢请求；log（日志）保存事件与上下文，适合解释状态变化。不要把每个 rid 都作为长期指标标签，否则标签数量会随请求量无界增长；rid 适合进入采样 trace 或日志。

## 4. Profiling 顺序

```text
端到端指标确认症状
 -> Scheduler/KV 计数器定位阶段
 -> CPU/GPU timeline 判断空洞与 overlap
 -> operator/kernel profile 定位热点
 -> 区分竞争原因的证据
```

先问 GPU 是忙、等 CPU、等通信还是等 IO，再深入 kernel。首次编译、graph capture、模型加载和 warmup 要与稳态结果分开。

### 为什么 Python 计时不一定是 GPU 耗时

设备执行通常异步。CPU 在 1 ms 内提交一轮，并不表示 GPU 在 1 ms 内完成。用设备 event 测 GPU 区间，或在明确同步点读取时间；端到端则保留真实异步流水。为了计时在每个算子后都同步，会破坏本来想测量的 overlap，所以精细 profile 与常规吞吐实验应分开。

读 trace 时先画依赖：若 CPU 从 0–2 ms 准备批次，GPU 从 2–8 ms 执行，CPU 同时在 2–4 ms 处理旧结果，那么该区间墙钟耗时为 8 ms，不能把三个区间加成 10 ms。跨节点时间还需时钟对齐；否则“接收早于发送”可能只是时钟偏差。

若 forward 只占总延迟 40%，即便把它加速两倍，端到端加速也仅为 `1/(0.6+0.4/2)=1.25` 倍。这里将原始总耗时归一化为 1，不变部分为 0.6，变快部分为 0.4/2。这说明先找长阶段，再优化 kernel 的原因。

## 5. 常见因果链

| 症状 | 先看 | 常见原因 |
|---|---|---|
| TTFT 高 | waiting/prefill/prefix hit | 排队、长 prompt、cache miss |
| ITL 抖动 | decode batch、混合 prefill | 大 batch、chunk 配置、通信抖动 |
| 吞吐低且 GPU 低 | CPU timeline | tokenizer、grammar、调度或网络 |
| OOM | 静态/KV/graph 分账 | mem fraction、长上下文、碎片 |
| 多卡不扩展 | collective trace | 小 batch、慢链路、负载不均 |

## 6. 合格性能报告

报告包含环境、workload 生成方式、所有启动参数、warmup、原始 JSONL、分位数、容量曲线、资源曲线、误差范围和已知限制。优化前后使用同一请求集合、seed 和正确性门槛。

## 7. 官方入口

- `sglang/docs_new/docs/developer_guide/bench_serving.mdx`
- `sglang/docs_new/docs/developer_guide/benchmark_and_profiling.mdx`
- `sglang/docs_new/docs/advanced_features/observability.mdx`
- `sglang/docs_new/docs/references/production_metrics.mdx`
- `sglang/docs_new/docs/references/production_request_trace.mdx`
