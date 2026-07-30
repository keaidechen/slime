# 00. Infra 工程师的 SGLang 学习地图

## 1. 推理与训练的系统差异

训练的 shape 和 step 相对稳定；在线推理面对：

- 请求随机到达；
- prompt/output 长度未知且长尾；
- prefill 和 decode 算术强度不同；
- 每个请求独立采样、停止、超时、abort；
- KV cache 生命周期跨越许多 scheduler iteration；
- 延迟 SLO 与吞吐相互制约。

因此 runtime 的核心不是一次 forward，而是持续做资源分配和在线决策。

## 2. 性能指标

| 指标 | 含义 | 常见影响因素 |
|---|---|---|
| TTFT | 到首 token 时间 | 排队、tokenize、prefill、prefix hit |
| ITL | 相邻输出 token 间隔 | decode batch、调度抖动、网络/stream |
| TPOT | 首 token 后每 token 平均处理时间 | decode kernel/通信 |
| E2E | 请求总延迟 | TTFT + 所有 decode |
| throughput | req/s 或 tok/s | batching、模型、并行、长度分布 |
| goodput | 满足 SLO 的有效吞吐 | 尾延迟、过载策略 |

报告平均值远远不够，要给 p50/p95/p99 和输入/输出长度分桶。

## 3. 必须掌握的知识点

### A. 模型推理

- autoregressive generation；
- prefill/extend 与 decode；
- MHA/GQA/MLA、RoPE、logits、sampling；
- KV cache 大小：

```text
MHA KV bytes/token
≈ 2 × layers × num_kv_heads × head_dim × bytes_per_element
```

还需乘并发 token 数，并考虑 TP shard、page metadata、alignment。

### B. 动态调度

- continuous batching；
- waiting/running batch；
- token budget、memory budget；
- chunked prefill；
- preemption/retract；
- FCFS、cache-aware、priority；
- 请求 finish/abort/timeout 状态机；
- CPU scheduling 与 GPU execution overlap。

### C. 内存系统

- request slot pool；
- token→KV pool；
- paged KV；
- prefix cache/RadixAttention；
- lock reference 与 LRU/priority eviction；
- fragmentation 与 page size；
- HiCache 的 GPU/host/远端层级。

### D. 执行层

- `ForwardBatch` metadata；
- `ModelRunner` 与 `TpModelWorker`；
- attention backend；
- CUDA Graph capture/replay 与 shape padding；
- torch.compile/JIT/custom kernels；
- logits processor 与 sampler。

### E. 分布式

- TP、PP、DP attention、EP；
- MoE all-to-all 与 expert load balance；
- router/load balancer；
- prefill/decode disaggregation；
- KV transfer 的 RDMA、bootstrap、ownership 和失败清理。

### F. 高级生成

- speculative decoding：draft、verify、accept/reject；
- structured output：regex/JSON grammar FSM；
- multi-LoRA batching；
- quantized weights/KV；
- deterministic inference 的限制；
- RL rollout 的 weight update、abort、token consistency。

### G. 生产工程

- benchmark workload 建模；
- admission control/backpressure；
- autoscaling 与 cache affinity；
- health/readiness；
- Prometheus/OpenTelemetry/profiling；
- correctness/accuracy regression；
- canary、故障注入、容量规划。

## 4. 源码地图

```text
sglang/python/sglang/srt/
  entrypoints/       HTTP、OpenAI protocol、Engine API
  managers/          tokenizer、scheduler、batch、detokenizer、TP worker
  mem_cache/         request/token pool、radix/HiCache、allocator
  model_executor/    ModelRunner、ForwardBatch、CUDA graph
  layers/            attention、logits、quantization、MoE 等
  sampling/          sampling metadata 与算子
  speculative/       EAGLE/MTP/ngram 等
  constrained/       grammar backend
  disaggregation/    prefill/decode 与 KV transfer
  observability/     metrics、tracing、request time
```

## 5. 建议实验

1. 单 GPU 发一个请求，记录每个进程与 IPC message。
2. 分别跑纯 prefill、纯 decode、混合 workload。
3. 构造相同 system prompt，比较 radix cache 开关的 TTFT。
4. 将 KV pool 压满，观察 eviction/retract，不允许 silent corruption。
5. 开启 overlap scheduler，对齐相邻 iteration 的 batch/result。
6. TP=2 抓 NCCL trace；再测试 DP attention/MoE EP。
7. speculative decoding 改变 draft steps，画 acceptance 与速度曲线。
8. 制造 abort、worker crash、KV transfer timeout，检查资源是否回收。

## 6. 先掌握三个计算模型

### Prefill FLOPs

对 dense Transformer，prefill 的大头可粗看成：

```text
线性层 FLOPs ∝ tokens × parameter-related dimensions
attention FLOPs ∝ batch × heads × sequence² × head_dim
```

长 prompt 既有大 GEMM，也有二次 attention。FlashAttention 降低中间矩阵 IO/显存，但没有改变 dense attention 的理论二次计算量。

### Decode bandwidth

每次只产生少数 token，却要读取大部分模型权重与历史 KV。batch=1 时 GEMM M 维很小，算术强度低。增加 decode batch 能复用权重读取、提高吞吐，但让单请求等待同批并影响 ITL。

### KV 容量

Llama 类 MHA/GQA：

```text
KV bytes/token =
  layers × 2 × local_kv_heads × head_dim × dtype_bytes
```

例：32 层、8 KV heads、head_dim=128、BF16：

```text
32×2×8×128×2 = 131072 bytes ≈ 128 KiB/token
100k cached tokens ≈ 12.5 GiB
```

TP 若切 KV heads，按 local heads 计算；额外还有 page table、allocator、workspace。

## 7. Request/Batch/KV 三个状态机

### Request

```text
received → waiting → running
                   ↘ finished
                   ↘ aborted
          ↖ retracted
```

### Batch

```text
schedule plan → prepare metadata → GPU forward
              → sample/result process → next plan
```

overlap 模式下这些阶段属于相邻 iteration，不能把“当前 batch”和“当前处理的 result”默认视为同一个。

### KV

```text
free slot → allocated to req → committed
          → radix cached/protected → evictable → freed
```

silent corruption 往往来自三个状态机推进不同步。

## 8. 一条 8 周路线

| 周 | 主题 | 交付物 |
|---|---|---|
| 1 | 进程/IPC | 单请求时序与 message schema |
| 2 | scheduler | prefill/decode 状态表 |
| 3 | KV/Radix | pool 与引用锁不变量 |
| 4 | ModelRunner | ForwardBatch shape/metadata 表 |
| 5 | CUDA graph/kernel | graph hit 与 eager trace |
| 6 | TP/EP/PD | 通信和 ownership 图 |
| 7 | speculative/grammar | token/KV 回滚测试 |
| 8 | 生产调优 | latency-throughput/SLO 报告 |

## 9. 本章延伸阅读

- [SGLang 论文](https://arxiv.org/abs/2312.07104)：先读 runtime 与 RadixAttention。
- [RadixAttention 项目团队博客](https://www.lmsys.org/blog/2024-01-17-sglang/)：图示比直接读树代码更适合入门。
- [SGLang v0.4 调度器博客](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)：理解 overlap scheduler、cache-aware router 与 DP attention。
- [Mini-SGLang 教学项目博客](https://www.lmsys.org/blog/2025-12-17-minisgl/)：用较小实现建立完整推理引擎心智模型，再回主仓库。
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)：理解 paged KV 与连续批处理的内存动机。
