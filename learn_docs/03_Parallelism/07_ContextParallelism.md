# Context Parallelism：长上下文 Attention 如何跨 GPU

Context Parallelism（CP，上下文并行）沿 sequence/context 维切 activation，让每个 rank 只常驻部分 token。困难在于 self-attention：每个 query 必须看到允许范围内的全部 Key/Value（KV，键值）。

## Ring Attention

设 CP degree 为 $p$。每 rank 保留本地 query block，KV block 在 ring 上轮转；每轮计算本地 Q 对当前 KV 的局部 attention，再通过 online softmax 合并局部最大值、归一化和 output。

```mermaid
flowchart LR
    K0["KV block 0"] --> K1["rank 1"]
    K1 --> K2["rank 2"]
    K2 --> K3["rank 3"]
    K3 --> K0
```

计算与下一块 KV P2P 可以流水重叠。每 rank 最终处理完整 context 的交互，因此总 attention FLOPs 没消失，只被分摊；通信量随 KV block 和轮数增长。

## Causal mask 的负载不均

若按连续 token 块分配，序列前部 query 只能看较少 KV，后部 query 看更多，rank 工作量不同。Megatron 常用双端/zig-zag 重排：把前段与后段 token 配到同一 rank，使 causal triangle 面积更均衡。

## Ulysses：Sequence shard ↔ Head shard

DeepSpeed-Ulysses 在 attention 前执行 All-to-All：

```mermaid
flowchart LR
    S["每 rank：部分 sequence、全部 heads"] --> A["All-to-All"]
    A --> H["每 rank：完整 sequence、部分 heads"]
    H --> F["Local FlashAttention"]
    F --> B["All-to-All 回 sequence shard"]
```

优势是可直接运行完整 sequence 的高效 attention kernel；限制是并行度受 attention head/KV head 数与 All-to-All topology 影响。Grouped-Query Attention（GQA，分组查询注意力）的 KV heads 少时，Ulysses degree 可能受限。

## Megatron CP 的通信类型

Megatron Core 支持多种 CP communication style，包括 P2P、AllGather、All-to-All 以及 hierarchical 组合。它们在 latency、memory、拓扑利用和 kernel integration 上不同。不要把 `context_parallel_size` 当成唯一决策；应结合 `cp_comm_type` 与 attention backend。

## Dynamic Context Parallelism

固定 CP degree 按最大 sequence 配置时，短样本也承担多 rank 通信，变长 batch 还会不均衡。Dynamic Context Parallelism（Dynamic CP，动态上下文并行）按 micro-batch 的实际长度选择 CP degree/分组。

Megatron 团队在 2026 年公开的结果显示，针对真实变长数据可获得最高约 1.48× speedup；这是特定 workload 报告，不是普遍保证。它揭示了重要方向：并行配置可以成为 runtime schedule，而非 job 启动时的常数。

## CP 与 TP/EP 的冲突

- TP 和 Ulysses 都可能切 attention heads，组合时 degree 受 head 数乘积约束；
- CP 的 sequence layout 进入 MoE 前，token All-to-All 需要保留 global token metadata；
- Ring P2P 与 EP All-to-All 若共享 NIC，会产生 traffic contention；
- packed sequence 的 causal boundary、position 与 loss mask 必须在重排后保持正确。

## 何时选择哪类方法

| 条件 | 初始倾向 |
|---|---|
| head 数充足、All-to-All fabric 强 | Ulysses |
| head 数少/GQA、可良好 overlap P2P | Ring-style CP |
| 跨节点与节点内带宽差异大 | hierarchical CP / USP |
| sequence 长度高度变化 | Dynamic CP / fine-grained block assignment |
| attention 本身稀疏或线性 | 需要架构专用 sharding rule，不能默认 dense CP |

## 资料

- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509)
- [USP](https://arxiv.org/abs/2405.07719)
- [PyTorch Context Parallel Tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)
- [Megatron Dynamic CP](https://forums.developer.nvidia.com/t/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/358971)

