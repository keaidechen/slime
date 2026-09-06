# 并行策略与 Bubble 分析

Bubble（气泡）是本可用于推进关键路径、却因依赖或负载不均而空置的资源时间。不同并行维度生成不同 bubble，必须用其 process group 与 phase 语义解释。

DDP = Distributed Data Parallel（分布式数据并行）；FSDP = Fully Sharded Data Parallel（全分片数据并行）；TP/PP/CP/EP 分别是 Tensor/Pipeline/Context/Expert Parallelism（张量/流水线/上下文/专家并行）。

## 1. 并行维度的时间线指纹

| 并行 | 常见等待形态 | 关键对照量 |
|---|---|---|
| DDP | backward 中 bucket AllReduce，step 尾部最后 bucket exposed | bucket ready time、grad compute、rank skew |
| FSDP | layer 前 parameter AllGather，backward 后 ReduceScatter | prefetch 距离、full-param peak、compute overlap |
| TP | 每层多次 collective，频率高、消息中等 | GEMM 粒度、跨节点边界、stream dependency |
| PP | stage 间 P2P 与 warm-up/drain 空洞 | micro-batch 数、最慢 stage、schedule |
| CP | attention 内 KV exchange/All-to-All | sequence length、causal imbalance、ring step |
| EP | dispatch/combine AllToAllV + expert compute 长尾 | per-expert token、max/mean、capacity/drop |

## 2. Pipeline bubble

对简单 1F1B（One-Forward-One-Backward，一前向一反向）流水，stage 数 $p$、micro-batch 数 $m$ 时，一个粗略 bubble fraction 是：

$$bubble \approx \frac{p-1}{m+p-1}$$

它假设 stage 等长、通信理想，真实系统还要加 stage imbalance 和 P2P exposed time。增大 $m$ 可摊薄 warm-up/drain，却可能增加 in-flight activation、调度开销与 batch semantics 约束。

### 怎么从 trace 量

1. 给每个 virtual/physical stage 标注 forward/backward micro-batch ID；
2. 对齐同一 global step；
3. 计算每个 stage 的 active union，而非 kernel duration 求和；
4. 找 bottleneck stage 与其他 stage 等待它的区间；
5. 区分 schedule 固有 bubble、stage imbalance 和通信延迟。

## 3. DDP/FSDP：最后一个 bucket 最重要

大量 gradient communication 可以被 backward 覆盖，但最后生成的 bucket 常落在尾部。优化方向是：

- bucket 划分与 parameter order，使关键梯度更早 ready；
- ReduceScatter/AllReduce 在独立 stream 及时发起；
- 避免过小 bucket 的 latency 开销，也避免过大 bucket 延迟启动；
- 检查 optimizer 是否在全局 barrier 后串行执行；
- FSDP 调整 forward/backward prefetch，但同时检查显存峰值。

## 4. TP：算子变小与通信变频繁

增大 TP degree 会缩小 per-rank GEMM。若从大 GEMM 变成多个低 occupancy 小 GEMM，即使参数能放下，MFU 也会下降。trace 上通常表现为 GEMM duration 缩短、collective 比例和 kernel count 上升、空洞更敏感。

所以 TP 优先留在高速节点内；跨节点是否值得必须用真实 message size 和 GEMM shape 实测。

## 5. CP：平均 token 数不够

causal attention 的早期 query 可见 KV 更少，简单连续切 sequence 会造成各 rank 工作不均。长短样本混合时，固定 CP degree 还会让短样本过度通信。

检查：每 rank query/KV token、有效 attention pair、ring step duration、padding/packing、动态 CP degree。只看总 sequence length 会掩盖 causal 与 packed-sequence imbalance。

## 6. EP：通信与 expert straggler 耦合

Mixture of Experts（MoE，混合专家）中，router 令 token 分布动态变化。记录：

$$imbalance=\frac{\max_e(tokens_e)}{\operatorname{mean}_e(tokens_e)}$$

同时画 dispatch AllToAllV、expert grouped GEMM、combine AllToAllV。一个 hot expert 会让对应 rank compute 变长，并让其他 rank 的 combine 等待；这不是单纯网络问题。

## 7. 组合并行的归因方法

为每个 event 附加：`global_step / micro_batch / layer / parallel_dimension / process_group / bytes / tokens`。多维 process group 不标明时，所有 NCCL kernel 名看起来相似。

建议生成矩阵：行是 rank，列是 phase，颜色是 duration；再按 node、TP group、PP stage、EP group 聚合。这样能一眼区分“同一节点慢”“同一 stage 慢”与“同一 expert group 慢”。

## 8. 优化顺序

1. 修正 rank/stage/expert 负载不均；
2. 确保拓扑 mapping 正确；
3. 调通信颗粒度与发起时机；
4. 建立 overlap；
5. 再调整 parallel degree；
6. 重新核算显存峰值、correctness 与容错。

## 资料

- [Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [并行专题总览](../03_Parallelism/00_专题总览.md)
- [通信计算重叠](../02_Distributed_Communication_Memory/09_通信计算重叠.md)
