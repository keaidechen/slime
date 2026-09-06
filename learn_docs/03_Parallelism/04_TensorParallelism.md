# Tensor Parallelism：在算子内部切矩阵

<!-- learning-position -->
> **学习定位**：A3 · 必修。
> **前置**：[通信与 tensor 基础](<../00_Foundations/06_两卡通信与torchrun.md>)。
> **首读/二读**：Column/Row Linear 的 local shape、forward/backward 通信。
> **进度与实验**：[学习清单](<../学习清单.md>) · [总入口](<../README.md>)。
<!-- /learning-position -->

Tensor Parallelism（TP，张量并行）把一个 layer 的参数和计算分到多张 GPU。它解决“单层太大/单卡算不下”，但每层通常都要通信，因此优先放在 NVLink/NVSwitch 高速域内。

## Column Parallel Linear

设 $Y=XW$，$W\in\mathbb{R}^{H_{in}\times H_{out}}$。沿输出维切列：

$$
W=[W_0,W_1,\ldots,W_{p-1}],\qquad Y_i=XW_i
$$

每张卡需要完整 X，得到 Y 的 feature shard。若下一个算子能直接消费这个 shard，无需立即 AllGather。

## Row Parallel Linear

沿输入维切行，并把输入按 feature 对应切分：

$$
W=\begin{bmatrix}W_0\\W_1\\\vdots\\W_{p-1}\end{bmatrix},\quad
X=[X_0,X_1,\ldots,X_{p-1}]
$$

每卡计算 partial output $Z_i=X_iW_i$，完整输出为：

$$
Y=\sum_i Z_i
$$

所以 row parallel 的输出是 `Partial(sum)`，通常需要 AllReduce，或用 ReduceScatter 直接产生 sequence shard。

## MLP 为什么天然一列一行

Multi-Layer Perceptron（MLP，多层感知机）可写成：

$$
Y=\phi(XW_1)W_2
$$

把扩张层 $W_1$ 做 column parallel，激活后保持 hidden shard；把回投影 $W_2$ 做 row parallel，最后只需一次 reduction。

```mermaid
flowchart LR
    X["X replicated"] --> C["W1 column parallel"]
    C --> H["Intermediate sharded"]
    H --> R["W2 row parallel"]
    R --> A["AllReduce / ReduceScatter"]
```

如果两层之间先 AllGather，就浪费了布局可组合性。

## Attention 的 TP

Multi-Head Attention（MHA，多头注意力）通常沿 attention heads 切分 Query/Key/Value（Q/K/V）投影；每 rank 计算一部分 heads，再由 output projection 的 row parallel reduction 合并。Grouped-Query Attention（GQA，分组查询注意力）会带来限制：KV heads 数必须能被 TP degree 合理切分，或采用复制 KV heads/fine-grained mapping。

Multi-head Latent Attention（MLA，多头潜在注意力）与 DeepSeek Sparse Attention（DSA，DeepSeek 稀疏注意力）等结构改变可切 tensor 与 kernel 支持，不能套用 MHA 的 head-shard 假设。

## Forward 与 backward 的通信对偶

Column parallel forward 产生 output shard；其 backward 对 input gradient 通常需 reduction。Row parallel forward 需 reduction；其 backward 对 input naturally sharded。Megatron 通过相邻 column/row layer 配对，让一个 Transformer block 中大 collective 数量受控。

## TP degree 的边界

增大 TP：

- 参数/activation shard 更小；
- collective group 更大且每 layer 同步；
- local GEMM 的 M/N/K 维度变小，Tensor Core 利用率可能下降；
- divisibility、head 数、expert hidden size 和 vocab size 限制增多。

所以“模型能放下后继续加 TP”不一定加速。推理低延迟常用 TP 缩短单请求 compute，但跨节点 TP 往往付出较大同步代价。

## PyTorch 的表达

PyTorch `torch.distributed.tensor.parallel` 用 `ColwiseParallel`、`RowwiseParallel`、`SequenceParallel` 等 ParallelStyle 描述 module plan，并基于 DeviceMesh/DTensor 传播 layout。API 仍标注 experimental，复杂自定义 module 要验证 input/output layout。

## 资料

- [Megatron-LM 原始 TP 论文](https://arxiv.org/abs/1909.08053)
- [PyTorch Tensor Parallel API](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [PyTorch Large-Scale TP Tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [Megatron Core Tensor Parallel API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.html)



<a id="notebook-11"></a>

## 学习问答：原笔记 §11

### 11｜Transformer TP 通信量如何估算：为什么是 21 GiB / microbatch / GPU？

问题：假设一个 48 层 decoder-only Transformer，TP=8，hidden size=8192，sequence length=4096，micro batch size=1，使用 BF16。按“每层训练阶段发生 4 次 Ring All-Reduce”做粗略估算，为什么会得到每个 microbatch 每卡约 21 GiB 的通信量，以及 42～84 GiB/s 的通信带宽需求？

#### 核心结论

- 一次参与 TP collective 的 activation-sized tensor：1 × 4096 × 8192 × 2 Byte = 64 MiB。

- TP=8 的 Ring All-Reduce 中，每个 rank 实际发送的数据量约为 2 × (8−1)/8 × 64 MiB = 112 MiB。

- 每层按 4 次 collective、共 48 层：112 MiB × 4 × 48 = 21 GiB / microbatch / GPU。

- 如果纯计算只需要 0.25～0.5 s，要让通信处在同一时间尺度，需要约 21/0.5～21/0.25 = 42～84 GiB/s 的有效吞吐。

- 若有效带宽只有 23～29 GB/s，在“通信与计算完全不重叠”的简化假设下，通信约耗时 0.78～0.98 s，通信占总时间约 61%～80%。

#### 参数表

| 参数                | 取值          | 在估算中的作用                               |
|---------------------|---------------|----------------------------------------------|
| Layers              | 48            | collective 次数随层数线性增长                |
| TP size             | 8             | 决定 Ring All-Reduce 的 (N−1)/N 系数         |
| Hidden size H       | 8192          | 决定 activation tensor 的最后一维            |
| Sequence length S   | 4096          | 决定 token 维度                              |
| Micro batch B       | 1             | 决定一次 microbatch 的 activation 规模       |
| dtype               | BF16 = 2 Byte | 决定每个元素的字节数                         |
| Collectives / layer | 4             | 本例的粗略假设：forward 2 次 + backward 2 次 |

#### 第一步｜一次 TP collective 的逻辑 tensor 有多大？

把一次需要在 TP ranks 间聚合的 activation 简化成形状 $$B, S, H$$。本例中元素数为：

1 × 4096 × 8192 = 33,554,432 elements

BF16 每个元素占 2 Byte，所以：

33,554,432 × 2 Byte = 67,108,864 Byte = 64 MiB

因此，一次 collective 对应的“逻辑 tensor 大小”是 64 MiB。

#### 为什么这里不是再除以 TP=8？

TP 的确把矩阵乘法拆到 8 张 GPU 上，但某些 Row Parallel Linear 的局部输出仍然具有完整的 $$B, S, H$$ 形状；每个 rank 只计算最终结果的一部分贡献。可以写成：

Y = X₁W₁ + X₂W₂ + … + X₈W₈

每个 rank 得到一个与最终 Y 同形状的 partial result，最后需要把 8 个 partial result 做求和聚合。因此这里参加 All-Reduce 的逻辑对象仍然可以是完整的 64 MiB，而不是简单地把 64 MiB 除以 8。

注意：这是基于经典 Megatron-style TP 的概念级估算。启用 Sequence Parallel、不同 fused kernel 或不同 collective 实现后，实际通信算子可能变为 ReduceScatter / AllGather 等，不能机械地把“每层 4 次纯 All-Reduce”套到所有现代实现。

#### 第二步｜为什么 Ring All-Reduce 的每卡传输量是 112 MiB？

Ring All-Reduce 可以粗略理解成两个阶段：Reduce-Scatter + All-Gather。设参与 GPU 数 N=8，逻辑 tensor 大小 M=64 MiB。

- Reduce-Scatter：每张卡在 ring 中传 N−1=7 轮，每轮发送 M/N=8 MiB，因此每卡发送 7 × 8 = 56 MiB。

- All-Gather：再走 7 轮，每卡又发送 56 MiB。

所以 Ring All-Reduce 的经典每-rank 通信量公式为：

V_AR = 2 × (N−1)/N × M

= 2 × 7/8 × 64 MiB = 112 MiB

这里的系数 2 来自 Reduce-Scatter 和 All-Gather 两个阶段，不是 BF16，也不是 forward/backward。

#### 第三步｜从一次 collective 推到 48 层：21 GiB 是怎么来的？

本例假设每个 Transformer layer 在一次完整训练（forward + backward）过程中合计发生 4 次这样的 TP collective，因此：

每层：112 MiB × 4 = 448 MiB / GPU

48 层：448 MiB × 48 = 21,504 MiB = 21 GiB / GPU

所以“21 GiB”不是模型参数量，也不是单次消息大小，而是一个 microbatch 穿过 48 层 forward + backward 后，按该 Ring All-Reduce 模型累计得到的每卡网络传输量。

#### 第四步｜为什么对应 42～84 GiB/s 的通信需求？

如果一个 microbatch 的纯计算耗时约 0.25～0.5 s，而它同时需要完成约 21 GiB 的 TP 通信，那么为了让通信时间不比计算时间更长，对应的平均有效通信吞吐约为：

21 GiB / 0.5 s = 42 GiB/s

21 GiB / 0.25 s = 84 GiB/s

一个很重要的 Infra 直觉：GPU 算得越快，留给通信的时间窗口越短，因此对互联带宽和通信/计算 overlap 的要求反而越高。

#### 第五步｜23～29 GB/s 为什么会让 60%～80% 的时间花在通信？

这里要先区分 GB/s 和 GiB/s：1 GiB ≈ 1.074 GB，所以 23～29 GB/s ≈ 21.4～27.0 GiB/s。

传 21 GiB 所需通信时间约为：

T_comm ≈ 21 / 27.0 ～ 21 / 21.4 ≈ 0.78～0.98 s

若假设通信与计算完全串行，则 T_total = T_compute + T_comm。

- 最好组合：T_compute=0.5 s、T_comm≈0.78 s，通信占比 ≈ 0.78/(0.78+0.5) ≈ 61%。

- 最差组合：T_compute=0.25 s、T_comm≈0.98 s，通信占比 ≈ 0.98/(0.98+0.25) ≈ 80%。

所以文章中的“60%～80% 时间花在通信上”来自这个串行化的上界式粗估。实际系统会通过 overlap、不同 collective 算法和拓扑优化降低暴露出来的通信时间。

#### 一条通用估算式

在上述简化假设下，可以把每卡、每 microbatch 的累计通信量记成：

V ≈ L × C × (B × S × H × bytes) × 2(TP−1)/TP

其中 L=层数，C=每层 collective 次数。这个式子适合做数量级 intuition，而不是替代 NCCL trace / profiler 的真实测量。

#### 最值得记住的 Insight

- TP 是一种高频通信并行：通信发生在 Transformer 层内部，而不是只在一个 training step 的末尾。因此 TP 对 GPU↔GPU 的低延迟、高带宽互联特别敏感。

- 21 GiB 这个例子直接解释了为什么训练系统会尽量把一个 TP group 放在 NVLink / NVSwitch 的 scale-up 域内；一旦跨节点，HCA、InfiniBand/RoCE、GPUDirect RDMA 和 GPU↔NIC 拓扑就会成为关键。

- 分析通信时要同时区分三件事：逻辑 tensor 大小、collective 算法导致的实际链路字节数、以及真正暴露在 critical path 上的通信时间。

这也正好引出下一节：为什么大模型训练网络会从 PCIe，逐步发展出 NVLink / NVSwitch，并在跨机侧依赖 InfiniBand HCA 与 GPUDirect RDMA。
