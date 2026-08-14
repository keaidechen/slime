# 04｜FlashAttention-2 论文详解：当“减少 HBM IO”之后，为什么还要重新设计并行方式？

> **标题缩写与首次术语说明**：FA1/FA2 分别指 **FlashAttention-1 / FlashAttention-2**；HBM = **High Bandwidth Memory（高带宽内存）**；I/O = **Input/Output（输入/输出，这里主要指显存数据搬运）**；GPU = **Graphics Processing Unit（图形处理器）**；SM = **Streaming Multiprocessor（流式多处理器）**；CTA = **Cooperative Thread Array（协作线程阵列，通常对应 CUDA thread block）**；GEMM = **General Matrix-Matrix Multiplication（通用矩阵-矩阵乘法）**；FLOP = **Floating-Point Operation（浮点运算）**；TMA = **Tensor Memory Accelerator（张量内存加速器）**；WGMMA = **Warpgroup Matrix Multiply-Accumulate（warp group 级矩阵乘加）**；FP8 = **8-bit Floating Point（8 位浮点格式）**。本文中的 **warp** 是 GPU 线程束，**work partitioning（工作划分：决定任务如何分给 thread/warp/CTA）** 指把一项计算工作拆给不同线程/warp/CTA 的方式。 另外：Q/K/V = **Query/Key/Value（查询/键/值向量）**，QKᵀ 表示 Query 与 Key 的矩阵乘，PV 表示注意力概率矩阵与 Value 的矩阵乘；AI = **Artificial Intelligence（人工智能）**。 另外：LLM = **Large Language Model（大语言模型）**。会议缩写：ICLR = **International Conference on Learning Representations（国际学习表征会议）**；NeurIPS = **Conference on Neural Information Processing Systems（神经信息处理系统大会）**。

> 论文：Tri Dao, **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**，2023，后发表于 ICLR 2024。
>
> 推荐前置：
> - `00_共享基础_GPU与LLM推理硬件基础.md`
> - `01_FlashAttention论文详解.md`
>
> 本文重点不是重复 FlashAttention-1，而是回答：**FA1 已经把 Attention 从 IO 角度优化得很漂亮了，为什么仍只能利用 GPU 峰值算力的一部分？FA2 到底把剩下的性能浪费在哪里找了回来？**

---

# 1. 一句话先记住 FlashAttention-2

如果 FlashAttention-1 的核心是：

> **不要把完整 Attention Matrix 写回 HBM；通过 tiling + online softmax，把大量 HBM IO 消掉。**

那么 FlashAttention-2 的核心是：

> **算法已经 IO-aware 了，接下来要让 GPU 的并行度、warp 分工和 Tensor Core 利用率也更加合理。**

所以：

```text
FlashAttention-1
关注：数据怎么少搬

        ↓

FlashAttention-2
关注：工作怎么更合理地分给 GPU
```

它不是推翻 FA1，而是在 FA1 的 IO-aware 框架里继续做三件事：

1. 减少 Tensor Core 不擅长的 non-matmul FLOPs；
2. 增加 thread block 层面的并行度；
3. 改进一个 thread block 内 warp 之间的工作分配。

---

# 2. 为什么 FA1 之后还有很大优化空间？

很多人第一次看 FlashAttention 会产生一个错觉：

> “已经避免了 N×N Attention Matrix 的 HBM 读写，那是不是已经接近最优？”

并不是。

一个 GPU kernel 的性能至少同时取决于：

```text
数据搬运是否高效
        +
Tensor Core 是否吃满
        +
SM 上有没有足够多 CTA
        +
warp 之间有没有额外同步/通信
        +
非矩阵运算占了多少执行时间
```

FA1 主要解决了第一项。

FA2 的作者观察到，FA1 在 A100 上虽然已经远快于普通 Attention，但距离理论峰值仍有明显差距。

因此问题从：

> “能不能减少 IO？”

变成了：

> “已经把数据搬得比较合理以后，为什么 Tensor Core 还没有一直忙？”

---

# 3. 先重新看 Attention 的计算组成

Attention：

\[
O = \operatorname{softmax}(QK^T)V
\]

可以粗略拆成：

```text
矩阵乘 1：QKᵀ
       ↓
非矩阵操作：max / exp / sum / rescale
       ↓
矩阵乘 2：PV
```

GPU 对这些工作的处理能力不是一样的。

## 3.1 Matmul FLOPs

例如：

```text
QKᵀ
PV
```

可以高度利用 Tensor Core。

## 3.2 Non-matmul FLOPs

例如：

```text
exp
max
除法
rescale
类型转换
```

往往不能以 Tensor Core GEMM 那样恐怖的吞吐运行。

所以“1 FLOP 就是 1 FLOP”的想法在硬件上是错误的。

对现代 GPU 来说，常常是：

> **一个非矩阵 FLOP 的机会成本，比一个 Tensor Core 矩阵乘 FLOP 高得多。**

这就是 FA2 第一个优化方向的来源。

---

# 4. 改进一：减少 non-matmul FLOPs

FA1 已经使用 online softmax，让每个 KV tile 可以流式处理。

处理新 block 时，需要维护当前行的：

```text
最大值 m
归一化统计量 l
输出 accumulator O
```

随着新的 K/V block 加入：

```text
旧状态
  +
新 block
  ↓
更新 max
更新 exp/sum
重新缩放旧输出
```

FA2 重新组织了这些更新公式，使得中间过程可以少做一些重复 rescale / normalization。

直觉上可以理解成：

```text
FA1：
每一小步都尽早把状态整理成“归一化后的最终形式”

FA2：
允许 accumulator 暂时保持一种未完全归一化的形式
最后需要时再统一整理
```

这样减少了一些：

```text
乘法
除法
rescale
```

尤其是那些不走 Tensor Core 的操作。

这里的重要系统 insight 是：

> **当一个硬件对 GEMM 极其擅长时，优化算法不能只数总 FLOPs，而应该区分“什么类型的 FLOPs”。**

---

# 5. 改进二：Sequence Length 维度上的并行

这是 FA2 非常重要的一步。

假设：

```text
Batch Size = 1
Number of Heads = 32
```

如果只按照：

```text
batch × head
```

分配 thread block，那么最多只有约 32 个大任务。

但一张 A100 有很多 SM。

当：

- batch 很小；
- head 数有限；
- sequence 很长；

就可能出现：

```text
SM0  busy
SM1  busy
...
SM31 busy
SM32 idle
SM33 idle
...
```

有算力，但是没有足够多 CTA 可以并行。

---

# 6. FA2 怎么增加并行度？

Attention Matrix 可以想成：

```text
                 K sequence
        ┌────────────────────────┐
Q seq   │                        │
        │     Attention Matrix   │
        │                        │
        └────────────────────────┘
```

FA2 进一步沿 sequence length 把矩阵切开。

比如 forward：

```text
Q rows
│
├── rows 0~127     → CTA 0
├── rows 128~255   → CTA 1
├── rows 256~383   → CTA 2
└── ...
```

这样即使：

```text
batch = 1
```

也可以因为 sequence 很长而产生大量 CTA。

于是并行度从：

\[
O(B\times H)
\]

扩展为大致还包含：

\[
\text{sequence tiles}
\]

这一维。

这对长序列训练尤其重要。

---

# 7. 为什么 Forward 和 Backward 的切法还不一样？

Forward 的输出自然按 Q row 来分：

```text
每一组 Query row
→ 扫过所有 K/V block
→ 得到这一组 Query 的 Output row
```

所以 forward 让不同 CTA 负责不同 Q block 很自然。

Backward 则涉及：

```text
dQ
dK
dV
```

不同变量的数据依赖和 reduction 模式不同。

FA2 重新设计 backward 的 block parallelization，使工作可以更好地沿 K/V 方向切分。

这里不要急着背具体 tile。

真正该记的是：

> **forward/backward 虽然来自同一个数学公式，但最佳 GPU work partition 不一定相同。**

这也是系统 kernel 优化里非常常见的思维。

---

# 8. 改进三：Thread Block 内 Warp 怎么分工？

现在进入 FA2 最“CUDA 味”的部分。

假设一个 CTA 中有：

```text
warp 0
warp 1
warp 2
warp 3
```

FA1 的一种工作划分方式，可以粗略理解成不同 warp 分别处理不同 K/V slice。

例如：

```text
warp 0 → K0,V0
warp 1 → K1,V1
warp 2 → K2,V2
warp 3 → K3,V3
```

但这些 warp 最后都对同一组 Q 对应的输出做贡献。

于是会出现：

```text
warp 0 partial O ┐
warp 1 partial O ├─→ shared memory → reduction
warp 2 partial O │
warp 3 partial O ┘
```

这意味着：

- 更多 shared memory 写；
- 更多 shared memory 读；
- 更多 warp 间 reduction / synchronization。

---

# 9. FA2 的思路：不要 Split-K，更多采用 Split-Q

FA2 的 forward work partition 可以简化理解成：

```text
旧思路：
多个 warp 共享一组 Q，但各自处理不同 K/V
→ 最后要合并 partial outputs

FA2：
多个 warp 负责不同 Q row
但共享相同 K/V tile
→ 每个 warp 更独立地产生自己的 output
```

图示：

```text
               shared K/V tile
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        warp0     warp1     warp2
          │         │         │
         Q0        Q1        Q2
          │         │         │
         O0        O1        O2
```

而不是：

```text
Q same
 │
 ├→ warp0 + KV0 → partial O0
 ├→ warp1 + KV1 → partial O1
 ├→ warp2 + KV2 → partial O2
 │
 └──────────────→ reduction
```

于是大幅减少 warp 之间为了输出 accumulator 进行的 shared-memory 通信。

---

# 10. 为什么 Shared Memory 读写也是成本？

初学时很容易产生：

> “Shared Memory 已经很快了，为什么还优化？”

因为“很快”不代表“免费”。

当 Tensor Core 的吞吐极高时：

```text
Tensor Core compute
```

本身可能结束得非常快。

这时：

```text
shared memory traffic
warp sync
non-matmul instruction
```

都可能成为新的限制。

这和 FlashAttention-1 的逻辑其实一脉相承：

```text
FA1：HBM 很贵，所以减少 HBM traffic
FA2：HBM traffic 已经改善后，SM 内部的数据交换也开始值得优化
FA3：Tensor Core 再变快后，异步流水和 softmax overlap 继续成为重点
```

你可以看到一个很清楚的性能工程演化规律：

> **上一代优化消灭的瓶颈，会暴露下一层瓶颈。**

---

# 11. 为什么“更多并行”不是永远越多越好？

如果把工作切得过碎：

```text
一个 Attention
→ 10000 个特别小 CTA
```

会产生新的成本：

- kernel scheduling overhead；
- 每个 tile 数据复用变差；
- 更多 partial result；
- 更多 reduction；
- shared memory / register 的使用模式可能变差。

所以 FA2 做的并不是：

> “尽可能多切。”

而是：

> **在 tile reuse、occupancy、并行度和 reduction overhead 之间找到更好的平衡。**

---

# 12. 为什么长序列尤其受益？

sequence 越长：

```text
Attention Matrix 越大
```

FA2 可以沿序列产生更多独立 tile：

```text
长 Q sequence
↓
更多 Q blocks
↓
更多 CTA
↓
更多 SM 同时工作
```

尤其在：

```text
batch size 小
```

时，这个优势很重要。

这也是为什么 FlashAttention 系列和 long-context Transformer 的发展高度绑定。

---

# 13. FA2 的性能结果该怎么读？

论文报告在 A100 上，FlashAttention-2 相比 FlashAttention 有大约 2× 左右的速度提升，在不同设置下可达到约 50%–73% 的理论峰值 FLOPs，并显著提升端到端 Transformer 训练吞吐。

不要把某个百分比死记成普遍结论。

应该读成：

> **仅仅减少 HBM IO 还不足以吃满 GPU；把 sequence-level parallelism 和 warp-level work partition 一起重新设计，可以继续释放大量性能。**

---

# 14. FA1 → FA2 到底变化了什么？

可以用下面这张表记忆：

| 问题 | FlashAttention-1 | FlashAttention-2 |
|---|---|---|
| 最大关注点 | HBM IO | GPU 并行与 work partition |
| 核心数学 | tiling + online softmax | 延续 online softmax，简化状态更新 |
| CTA 并行 | 相对受 batch/head 限制 | 更充分沿 sequence 切分 |
| Warp 分工 | 更多 partial-output 合并 | 更偏 Split-Q，减少 warp 通信 |
| 主要收益 | IO-aware exact attention | 更高 Tensor Core/SM 利用率 |

二者并不是互斥方案：

```text
FA2 = FA1 的 IO-aware 基础
      + 更好的并行化
      + 更好的 warp 分工
```

---

# 15. 一个工厂比喻

FlashAttention-1 发现：

> 工厂效率低，是因为工人一直跑到远处仓库搬原料。

于是建立了 SM 附近的小工作台：

```text
HBM 仓库
   ↓ 一次搬一块
Shared Memory 工作台
   ↓
反复加工
```

FlashAttention-2 进来以后发现：

> 仓库问题缓解了，但工人的排班仍然不好。

有时：

```text
一部分车间有活
另一部分车间闲着
```

有时几个工人：

```text
各做一部分
最后又要一起合并
```

所以 FA2 做的是：

```text
更多车间同时开工
+
每组工人的职责重新划分
+
减少内部交接
```

这就是：

> **Better Parallelism and Work Partitioning。**

---

# 16. 从 FA2 到 FA3：下一层瓶颈是什么？

到了 H100/Hopper：

```text
Tensor Core 更强
TMA 出现
异步 WGMMA 出现
FP8 吞吐大幅提高
```

结果是：

> **FA2 在 A100 上很优秀的 schedule，并不能自动把 H100 新硬件能力吃满。**

尤其：

- 数据搬运和 Tensor Core 能否重叠？
- softmax 能否和 GEMM 重叠？
- 能否使用 FP8？

成为下一代问题。

这就是 FlashAttention-3。

---

# 17. AI Infra 视角最值得记住的 7 个 Insight

## Insight 1：IO-aware 不等于 hardware-saturating

消除 HBM IO 只是第一层。

之后还有：

```text
parallelism
occupancy
warp communication
instruction mix
```

## Insight 2：不同 FLOP 的硬件成本不同

Tensor Core matmul FLOPs 和 exp/div/reduction 等 FLOPs 不应简单等价看待。

## Insight 3：并行维度是算法设计的一部分

当 batch/head 不够产生足够 CTA 时，可以从 sequence tile 中挖并行。

## Insight 4：Thread block 内的 warp 分工会直接影响性能

不是“数学公式一样，CUDA 随便实现都行”。

## Insight 5：Shared Memory 很快，但不是免费

当主计算越来越快时，shared-memory traffic 和 synchronization 会浮现成瓶颈。

## Insight 6：性能优化是逐层暴露瓶颈

```text
HBM IO
 ↓ 优化
parallelism / warp communication
 ↓ 优化
async pipeline / softmax overlap
 ↓
...
```

## Insight 7：FlashAttention 系列本质是算法与 GPU 架构共同演化

它不是静态的“一个 attention trick”，而是一系列针对不同硬件代际重新设计执行方式的工作。

---

# 18. 读完后你应该能回答

1. 为什么 FA1 已经 IO-aware，FA2 还能快很多？
2. 为什么 non-matmul FLOPs 特别值得减少？
3. 为什么 batch size=1 时 sequence parallelism 很重要？
4. Split-Q 相比 Split-K 为什么能减少 warp 间通信？
5. 为什么 Shared Memory 访问也值得优化？
6. FA2 为什么在 H100 上又不够了？

如果这些问题可以用自己的话回答，就已经真正抓住 FA2。

---

## 主要参考资料

- Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, arXiv:2307.08691 / ICLR 2024.
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022.
- Dao-AILab FlashAttention 官方实现与说明。
