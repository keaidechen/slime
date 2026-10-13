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

## 2. 找容量拐点

逐步提高并发或 arrival rate，观察 throughput 何时趋于饱和、waiting queue 何时持续增长、TTFT/ITL 何时越过 SLO。拐点前保留安全裕量，而不是把峰值点当生产容量。

## 3. 四层观测

| 层 | 关键观测 |
|---|---|
| API/路由 | QPS、错误码、排队、重试、TTFT/E2E |
| Scheduler/KV | waiting/running、batch token、retract、cache hit/eviction |
| Model/通信 | forward、kernel、collective、graph hit |
| 系统 | 显存、GPU 利用率、CPU、网络、page fault |

指标必须能够沿 request id/trace id 关联，否则只能看到相关性，不能定位因果。

## 4. Profiling 顺序

```text
端到端指标确认症状
 -> Scheduler/KV 计数器定位阶段
 -> CPU/GPU timeline 判断空洞与 overlap
 -> operator/kernel profile 定位热点
 -> 区分竞争原因的证据
```

先问 GPU 是忙、等 CPU、等通信还是等 IO，再深入 kernel。首次编译、graph capture、模型加载和 warmup 要与稳态结果分开。

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
