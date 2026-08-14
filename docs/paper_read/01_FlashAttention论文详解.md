# 01｜FlashAttention 论文详解：为什么少算不一定更快，而少搬数据可以更快

> **标题缩写与首次术语说明**：I/O = **Input/Output（输入/输出）**，本文主要指 GPU 显存读写和数据搬运；HBM = **High Bandwidth Memory（高带宽内存，GPU 主显存）**；SRAM = **Static Random-Access Memory（静态随机存取存储器，本文主要指片上高速存储）**；GPU = **Graphics Processing Unit（图形处理器）**；GEMM = **General Matrix-Matrix Multiplication（通用矩阵-矩阵乘法）**；FLOP = **Floating-Point Operation（一次浮点运算）**，FLOPs 表示浮点运算次数。本文中的 **kernel** 是 GPU 核函数，**tiling** 是分块计算，**online softmax** 是“在线/增量 Softmax”，即扫描分块时维护全局归一化统计量，**IO-aware** 指“显式把数据搬运成本纳入算法设计”。 另外：LLM = **Large Language Model（大语言模型）**；AI = **Artificial Intelligence（人工智能）**；CPU = **Central Processing Unit（中央处理器）**；CUDA = **Compute Unified Device Architecture（NVIDIA GPU 并行计算平台与编程模型）**；Q/K/V = **Query/Key/Value（查询/键/值向量）**；KV = **Key/Value（键/值）**；FA1/FA2 = **FlashAttention-1/2**；GQA = **Grouped-Query Attention（分组查询注意力）**；MQA = **Multi-Query Attention（多查询注意力）**；TMA = **Tensor Memory Accelerator（张量内存加速器）**。 会议缩写：NeurIPS = **Conference on Neural Information Processing Systems（神经信息处理系统大会）**；ICLR = **International Conference on Learning Representations（国际学习表征会议）**。

> 论文：**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
>
> 作者：Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
>
> 发表：NeurIPS 2022，arXiv:2205.14135
>
> **前置阅读**：建议先读同目录的 `00_共享基础_GPU与LLM推理硬件基础.md`。本文不再重复解释 HBM、SRAM、kernel、tiling、memory-bound 等基础概念。

---

# 1. 先给出整篇论文的一句话结论

FlashAttention 没有发明新的 Attention，也没有近似 Attention。

它计算的仍然是：

\[
O=softmax\left(\frac{QK^T}{\sqrt d}\right)V
\]

真正改变的是：

> **Attention 在 GPU 上执行时，数据以什么顺序从 HBM 搬到片上 SRAM，又以什么顺序完成 softmax 和矩阵乘法。**

所以 FlashAttention 是一个典型的：

> **IO-aware algorithm（I/O 感知算法：设计时显式考虑数据搬运成本）。**

它告诉 AI Infra 一个非常重要的事实：

> **算法复杂度相同，甚至 FLOPs 更多，也可能因为 HBM IO 大幅减少而更快。**

---

# 2. 为什么 2022 年需要 FlashAttention？

Transformer 自 2017 年提出后，标准 Attention 的一个显著问题是序列长度 \(N\) 增大时：

\[
QK^T\in\mathbb{R}^{N\times N}
\]

因此 Attention score matrix 大小是：

\[
O(N^2)
\]

过去很自然的研究方向是：

> 能不能不计算完整 \(N\times N\) Attention？

于是出现大量：

- sparse attention；
- low-rank attention；
- linear attention；
- approximate attention。

这些方法试图降低：

\[
FLOPs
\]

甚至把理论复杂度从 \(O(N^2)\) 降下来。

但是 FlashAttention 论文指出一个关键问题：

> **FLOPs 少，不代表真实 GPU wall-clock time（真实经过时间） 一定少。**

因为现代 GPU 可能并不是“算不动”，而是：

> **大量时间花在 HBM 和片上计算单元之间搬数据。**

因此作者重新问了一个问题：

> 如果不改变 Attention 数学结果，而只是把 HBM IO 降下来，会怎么样？

这就是 FlashAttention。

---

# 3. 标准 Attention 在 GPU 上到底做了什么？

先忽略 scaling 和 mask：

\[
S=QK^T
\]

\[
P=softmax(S)
\]

\[
O=PV
\]

其中：

\[
Q,K,V\in\mathbb{R}^{N\times d}
\]

而：

\[
S,P\in\mathbb{R}^{N\times N}
\]

传统实现可以粗略理解成三个阶段：

```text
Kernel / GEMM 1
Q, K
 ↓
S = QK^T
 ↓
把 S 写到 HBM

Kernel 2
从 HBM 读 S
 ↓
softmax
 ↓
得到 P
 ↓
把 P 写到 HBM

Kernel / GEMM 3
从 HBM 读 P, V
 ↓
O = PV
 ↓
写回 HBM
```

问题最严重的地方不是 Q/K/V，而是：

```text
S: N × N
P: N × N
```

例如 \(N=8192\)：

\[
8192^2\approx67\text{ million elements}
\]

一个 head 就已经产生非常大的中间矩阵。

而这些中间矩阵往往被：

```text
写入 HBM
→ 再读出来
→ 再写
→ 再读
```

这正是论文认为真正昂贵的部分。

---

# 4. 一个反直觉问题：为什么“多算一点”反而可能更快？

假设有两个算法。

## 算法 A

```text
100 次计算
1000 次 HBM 数据搬运
```

## 算法 B

```text
130 次计算
200 次 HBM 数据搬运
```

如果 GPU 的计算单元非常强，而内存 traffic 才是瓶颈：

> 算法 B 完全可能更快。

FlashAttention 在 backward 中会选择 **recomputation（重计算）**：

```text
少存中间结果
↓
需要时重新算
```

从传统 CPU 程序员直觉看可能觉得浪费。

但在 GPU 上：

> 重算一个局部矩阵，有时比把一个巨大矩阵写到 HBM、以后再读回来便宜。

这是整篇论文非常重要的系统思想。

---

# 5. FlashAttention 的核心：不要 materialize 完整 S 和 P

所谓 materialize，就是：

> 真正在内存里生成并保存完整 tensor。

标准 Attention：

```text
QK^T
 ↓
完整 S[N,N] 存进 HBM
 ↓
softmax
 ↓
完整 P[N,N] 存进 HBM
```

FlashAttention：

```text
Q block
K block
V block
   ↓
搬到片上 SRAM
   ↓
只计算当前 tile 的 attention
   ↓
更新部分 output 和 softmax statistics
   ↓
继续下一块
```

最终：

> **完整的 \(N\times N\) S/P 从来不需要驻留在 HBM。**

这就是最大变化。

---

# 6. Tiling 到底怎样应用到 Attention？

假设：

```text
Q = [Q1, Q2, Q3, ...]
K = [K1, K2, K3, ...]
V = [V1, V2, V3, ...]
```

每个 \(Q_i,K_j,V_j\) 都是能放进 SRAM 的 block。

计算一个 tile：

\[
S_{ij}=Q_iK_j^T
\]

然后在片上完成：

```text
Q_i
K_j
 ↓
S_ij
 ↓
局部 softmax 信息
 ↓
与 V_j 相乘
 ↓
更新 O_i
```

但马上遇到一个数学困难：

> **softmax 是按整行归一化的。**

如果只看到 K 的一部分，怎么知道最终 denominator？

也就是说：

\[
softmax(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
\]

分母需要整行所有元素。

所以 Attention 不是简单把矩阵切块算完再相加就可以。

FlashAttention 能成立的数学关键，就是 **Online Softmax**。

---

# 7. 先理解数值稳定 Softmax

通常不会直接计算：

\[
softmax(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
\]

因为 \(e^{x_i}\) 可能 overflow。

更稳定的形式是：

\[
m=\max_i x_i
\]

\[
softmax(x_i)=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
\]

因此，要描述一行 softmax 的状态，我们可以记录：

1. 当前最大值 \(m\)；
2. 当前归一化和 \(l\)。

其中：

\[
l=\sum_j e^{x_j-m}
\]

---

# 8. Online Softmax：为什么可以一块一块算？

假设一整行 score 被分成两块：

```text
x = [x^(1), x^(2)]
```

第一块的最大值：

\[
m_1=\max(x^{(1)})
\]

第二块到来后：

\[
m_2=\max(m_1,\max(x^{(2)}))
\]

旧的 exponential sum 原本基于 \(m_1\)：

\[
l_1=\sum e^{x^{(1)}-m_1}
\]

现在最大值变成 \(m_2\)，旧状态可以重新缩放：

\[
l_2=e^{m_1-m_2}l_1+
\sum e^{x^{(2)}-m_2}
\]

于是每处理一个新 block，只要保留：

```text
running max m
running sum l
running output O
```

就可以逐步得到与完整 softmax 完全相同的结果。

所以 FlashAttention 不是 approximate attention。

它是在改变：

> **计算顺序，而不是数学定义。**

---

# 9. Output 为什么也可以在线更新？

Attention 最终输出：

\[
O=PV
\]

当 softmax normalization 随着新 block 更新时，之前算过的 partial output 也需要重新 rescale。

直觉可以理解成：

```text
处理 K/V block 1
→ 得到当前 O

发现 K/V block 2 中出现更大的 attention logits
→ 更新 row max
→ 旧 O 的权重要按照新 normalization 缩放
→ 加入 block 2 的贡献
```

所以一个 Q block 可以不断维护：

```text
m_i : 当前行最大值
l_i : 当前 softmax normalization
O_i : 当前累计输出
```

直到全部 K/V blocks 扫完。

此时：

\[
O_i
\]

就是精确 Attention 的输出。

---

# 10. 为什么 FlashAttention 可以只用 O(N) 额外内存？

标准实现要保存：

\[
S,P\in\mathbb{R}^{N\times N}
\]

所以额外内存：

\[
O(N^2)
\]

FlashAttention 主要保存：

```text
O: N × d
m: N
l: N
```

而不会保存完整 \(N\times N\) score / probability matrix。

因此论文证明算法额外内存可以做到：

\[
O(N)
\]

这里“线性内存”并不意味着 Attention FLOPs 变成线性。

它的计算复杂度仍然主要是：

\[
O(N^2d)
\]

这是非常容易混淆的一点：

> **FlashAttention 没有把 dense Attention 的理论计算复杂度从 \(N^2\) 变掉；它主要改变的是 memory complexity / IO complexity 和实际 wall-clock。**

---

# 11. Backward 为什么要 Recomputation？

训练时 forward 以后还要 backward。

标准 Attention 为了反向传播，通常需要中间矩阵：

```text
S = QK^T
P = softmax(S)
```

最简单的方法：

> forward 时保存 S/P，backward 直接读。

但这意味着巨大的：

\[
O(N^2)
\]

显存占用和 HBM traffic。

FlashAttention 选择：

```text
forward:
不保存完整 S/P
只保存 O 和 softmax normalization statistics

backward:
把局部 Q/K/V 重新搬进 SRAM
重新计算当前需要的 S/P tile
立即完成梯度计算
```

这类似：

> **Selective Gradient Checkpointing。**

系统 trade-off：

```text
增加一些 FLOPs
        ↓
省掉巨大中间 tensor 的 HBM write/read
        ↓
反而更快 + 更省显存
```

---

# 12. FlashAttention 真正优化的是 IO Complexity

标准 Attention 对 HBM 的读写包含大型：

\[
N\times N
\]

中间矩阵。

论文分析标准 Attention 的 HBM accesses 为：

\[
\Theta(Nd+N^2)
\]

而 FlashAttention 在 SRAM 大小为 \(M\) 时，可以将 HBM access 降到大致：

\[
\Theta\left(\frac{N^2d^2}{M}\right)
\]

这里最重要的不是背公式，而是理解变量：

- \(N\)：sequence length；
- \(d\)：head dimension；
- \(M\)：可利用的片上 SRAM 大小。

SRAM 越能容纳合适 block：

> 同一批 HBM 数据就越能被片上 reuse。

论文在典型设置中展示了 FlashAttention 可以把 HBM R/W 大幅降低，并由此显著减少 attention runtime。

---

# 13. 为什么一个 fused kernel 很重要？

标准 Attention 从程序层面像：

```text
matmul
→ mask
→ softmax
→ dropout
→ matmul
```

如果每一步都是独立 kernel：

```text
HBM ↔ GPU
HBM ↔ GPU
HBM ↔ GPU
...
```

FlashAttention 把这些核心步骤组织到一个 fused CUDA kernel 中。

中间 tile 可以留在：

- registers；
- shared memory / SRAM；

而不必完整落回 HBM。

所以 FlashAttention 可以看成两个思想叠加：

\[
\boxed{Tiling + Online\ Softmax}
\]

让数学上可以分块；

再加上：

\[
\boxed{Kernel\ Fusion + Recomputation}
\]

让硬件数据流真正高效。

---

# 14. 一个很容易误解的问题：FlashAttention 是不是“稀疏 Attention”？

不是。

FlashAttention 主体是：

> **Exact Dense Attention。**

它仍然计算全部需要的 attention interaction。

论文另外还展示了 Block-Sparse FlashAttention：

```text
如果已知某些 attention blocks 本来就不需要计算
↓
在 FlashAttention 的 tiled framework 中跳过这些 zero blocks
```

这是论文的 extension，而不是 FlashAttention 本身的定义。

---

# 15. 为什么这篇论文在 AI Infra 历史上特别重要？

在它之前，一个常见思考方式是：

```text
Attention 太慢
↓
减少 FLOPs
↓
设计 approximate attention
```

FlashAttention 强化了另一种思路：

```text
Attention 太慢
↓
分析 GPU memory hierarchy
↓
算清 HBM traffic
↓
重新排列计算顺序
↓
让 exact algorithm 也能大幅提速
```

这就是所谓：

> **Hardware-aware / IO-aware algorithm design。**

这个思想后来在 LLM Infra 里越来越重要。

---

# 16. FlashAttention → FlashAttention-2：下一步瓶颈发生了什么变化？

FlashAttention v1 解决了最大问题：

> HBM IO 太多。

但当 IO 降下来以后，新的瓶颈开始显现：

- thread block 如何划分工作；
- warp 之间如何协作；
- GPU occupancy；
- shared memory communication；
- non-matmul FLOPs。

FlashAttention-2 因此进一步做三类优化：

1. 减少 non-matmul FLOPs；
2. 除 batch/head 外，也沿 sequence length 增加 thread-block 级并行；
3. 重新设计同一 block 内 warp 的 work partition，减少 shared memory 读写和同步。

论文报告 FA2 相比 FA1 在多种设置中大约又获得约 2× 的 attention kernel 提升，并明显提高 A100 上理论计算吞吐的利用率。

因此技术演化可以记成：

```text
FlashAttention v1
主要解决 HBM IO
      ↓
FlashAttention-2
进一步解决 GPU work partition / occupancy
      ↓
FlashAttention-3
进一步针对 Hopper 的异步 Tensor Core、TMA 等新硬件能力做 pipeline
```

这也是 AI Infra 中非常典型的现象：

> **当你解决一层 bottleneck 后，下一层 bottleneck 才会暴露出来。**

---

# 17. FlashAttention 与训练 / 推理是什么关系？

原始 FlashAttention 论文非常重视训练，因为 backward 的显存问题和 recomputation 是其重要贡献。

但是它的核心 IO-aware attention kernel 思想同样影响推理。

不过 LLM serving 又带来了新问题：

```text
Prefill:
Q 很长，K/V 很长

Decode:
Q 通常只有 1 个 token
K/V 却可能很长
```

Decode 的形状和训练中的大矩阵 attention 差别非常大。

再加上：

- 请求长度不一致；
- KV Cache 不连续；
- prefix sharing；
- speculative decoding；

于是后来才进一步出现：

> **FlashInfer：针对 inference serving 的 heterogeneous attention workload 建立更通用的 kernel/runtime abstraction。**

---

# 18. FlashAttention 与 PagedAttention 有什么区别？

这是非常重要的层次区分。

## FlashAttention 问

> 已经给我 Q/K/V 了，一个 Attention kernel 怎样减少 HBM IO？

重点是：

```text
GPU kernel execution
```

## PagedAttention 问

> 服务很多请求时，不断增长的 KV Cache 应该怎样分配、存储和索引？

重点是：

```text
KV cache memory management
```

因此：

```text
PagedAttention 可以决定 KV 在哪几个 physical blocks
                ↓
Attention kernel 再按照这个 layout 读取 KV 并计算
```

两者是相邻但不同的系统层次。

---

# 19. FlashAttention 与 FlashInfer 的关系

可以这样记：

```text
FlashAttention
解决“一个 Attention 如何 IO-efficient”
        ↓
成为重要 kernel algorithm 基础
        ↓
FlashInfer
解决 serving 中各种 KV layout / query length /
attention variant 如何统一、高效执行
```

FlashInfer 并不是简单“比 FlashAttention 更快的下一代”。

它处理的问题空间更接近 inference engine/backend：

- paged KV；
- sparse KV loading；
- variable query lengths；
- customized attention variants；
- dynamic scheduling；
- CUDA Graph compatibility。

---

# 20. 论文实验应该怎样读？

原始论文报告：

- 在 GPT-2 Attention microbenchmark 中，相比 PyTorch baseline 的 attention computation 可获得很大的 kernel-level speedup；
- BERT-large、GPT-2 和 Long Range Arena 的训练 wall-clock 均有明显改善；
- 由于不保存 \(N\times N\) 中间矩阵，memory footprint 从关于 sequence length 的 quadratic 降到 linear 量级；
- 能够训练更长 sequence length。

但是系统论文的数字不要背成永久常数。

正确理解是：

> 在论文时代的 GPU、baseline 和 kernel 实现下，作者证明了“减少 HBM IO”不是理论上的小优化，而可以真正转化成巨大的 wall-clock 收益。

这才是实验最重要的意义。

---

# 21. FlashAttention 的局限是什么？

## 1. 没有消除 dense Attention 的 O(N²) FLOPs

长上下文无限增长时，计算量仍然存在。

## 2. 高性能实现高度依赖硬件

tile size、register、shared memory、Tensor Core、warp partition 都与 GPU 架构强相关。

## 3. 不同 workload 的最优 kernel 不一样

训练、prefill、decode、GQA、MQA、sparse mask 等形状不同。

这正是后来 FlashAttention-2/3、FlashInfer 等工作继续发展的原因。

---

# 22. 最重要的 7 个 Insight

1. **Attention 慢不只因为 \(O(N^2)\) FLOPs，还因为 \(N\times N\) 中间矩阵导致巨大 HBM IO。**
2. **FLOPs 少不等于 wall-clock 快。**
3. **FlashAttention 是 exact attention，不改变模型数学语义。**
4. **Tiling + Online Softmax 是“能正确分块计算”的数学基础。**
5. **Kernel Fusion + SRAM reuse 是减少 HBM traffic 的执行基础。**
6. **Backward 中“多算换少存”在 GPU 上可能同时更快、更省显存。**
7. **FlashAttention 把硬件 memory hierarchy 从“实现细节”提升成了算法设计的一部分。**

---

# 23. 一句话串到下一篇论文

FlashAttention 解决以后，Attention kernel 本身已经高效很多。

但是 LLM 从“训练”走向“在线自回归 serving”以后，系统出现了新的巨大对象：

\[
\boxed{KV\ Cache}
\]

它会：

- 随 token 动态增长；
- 每个请求长度不同；
- 消耗大量显存；
- 限制 batch size。

于是下一个核心问题从：

> “Attention 怎么少搬数据？”

演变成：

> **“几百个动态请求的 KV Cache 到底该怎么管理？”**

这就是下一篇 **PagedAttention / vLLM**。

---

## 主要参考资料

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022, arXiv:2205.14135.
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, arXiv:2307.08691, ICLR 2024.
