# 05｜FlashAttention-3 论文详解：为什么 H100 需要“异步 Attention”？

> **标题缩写与首次术语说明**：FA1/FA2/FA3 分别指 **FlashAttention-1/2/3**；GPU = **Graphics Processing Unit（图形处理器）**；TMA = **Tensor Memory Accelerator（张量内存加速器，用于异步搬运大块张量）**；WGMMA = **Warpgroup Matrix Multiply-Accumulate（warp group 级异步矩阵乘加）**；GEMM = **General Matrix-Matrix Multiplication（通用矩阵-矩阵乘法）**；HBM = **High Bandwidth Memory（高带宽内存）**；FP8 = **8-bit Floating Point（8 位浮点格式）**；FP16 = **16-bit Floating Point（16 位浮点格式）**。本文中的 **pipeline（流水线）**指把数据搬运、矩阵乘法和 Softmax 等阶段重叠执行，**warp specialization（warp 专职化）**指让不同 warp group 分别长期承担 producer/consumer 等不同职责。 另外：LLM = **Large Language Model（大语言模型）**；CUDA = **Compute Unified Device Architecture（NVIDIA GPU 并行计算平台与编程模型）**；Q/K/V = **Query/Key/Value（查询/键/值向量）**，QKᵀ 与 PV 分别对应 Attention 的两次主要矩阵乘；AI = **Artificial Intelligence（人工智能）**；CUTLASS = **CUDA Templates for Linear Algebra Subroutines（NVIDIA 高性能线性代数 CUDA 模板库）**。H100/A100/B200 是 NVIDIA GPU 产品型号，不属于需要展开的缩写。 会议缩写：NeurIPS = **Conference on Neural Information Processing Systems（神经信息处理系统大会）**。

> 论文：Jay Shah et al., **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision**，2024，NeurIPS 2024。
>
> 推荐前置：
> - `00_共享基础_GPU与LLM推理硬件基础.md`，尤其 Hopper/TMA/WGMMA 部分
> - `01_FlashAttention论文详解.md`
> - `04_FlashAttention-2论文详解.md`
>
> 这篇论文标志着 FlashAttention 从“IO-aware algorithm（I/O 感知算法：设计时显式考虑数据搬运成本）”进一步进入“**与 GPU 微架构协同设计 software pipeline**”的阶段。

---

# 1. 一句话先记住 FlashAttention-3

FlashAttention-3 的问题是：

> **H100 的 Tensor Core 已经非常快，如果 kernel 仍然按照“搬数据 → 算 GEMM → 做 softmax → 再算 GEMM”的串行思路执行，那么很多硬件单元会轮流等待。**

它的核心回答是：

```text
异步数据搬运
+
Warp Specialization（warp 专职化：不同 warp group 承担不同流水线职责）
+
GEMM / Softmax overlap
+
FP8
```

让不同硬件流水线尽量同时工作。

所以最适合记忆成：

\[
\boxed{\text{FA3 = Asynchrony + Pipelining + Low Precision}}
\]

---

# 2. 为什么 FA2 在 H100 上不够？

FA2 的重点是：

```text
better parallelism
better warp work partition
```

在 A100 上已经能达到很高的有效算力。

但 H100/Hopper 又加入了：

- 更强 Tensor Core；
- TMA；
- WGMMA；
- 更强 FP8 Tensor Core；
- 更适合 warp-specialized pipeline 的机制。

论文指出，FA2 在 H100 上并没有充分利用这些能力，FP16 attention 的硬件利用率仍有很大提升空间。

这是一条非常重要的 Infra 规律：

> **kernel 在上一代 GPU 上“已经优化得很好”，并不意味着换一代 GPU 只会自动等比例加速。**

新硬件通常需要新的 schedule。

---

# 3. Attention 为什么特别适合做流水线重叠？

Attention tile 的主循环可以简化成：

```text
for each KV tile:
    load K/V
    S = Q Kᵀ
    P = softmax update(S)
    O += P V
```

从硬件资源看：

```text
load K/V
→ Memory / TMA pipeline

QKᵀ, PV
→ Tensor Core pipeline

max/exp/sum/rescale
→ CUDA core / special-function / reduction pipeline
```

三类工作不完全使用相同执行资源。

所以如果总是串行：

```text
load
 ↓
GEMM
 ↓
softmax
 ↓
GEMM
```

会造成大量“你干活的时候我在等”。

FA3 的目标就是把时间线重排。

---

# 4. 第一层异步：数据搬运与计算 overlap

传统简化时间线：

```text
        tile 0                      tile 1
Memory [load K0/V0]               [load K1/V1]
Compute             [compute 0]               [compute 1]
```

理想时间线：

```text
Memory [load 0][load 1][load 2][load 3]
Compute        [comp0 ][comp1 ][comp2 ][comp3 ]
```

关键就是：

> consumer 计算当前 tile 时，producer 已经开始准备下一 tile。

Hopper 的 TMA 让这件事更自然。

---

# 5. Producer-Consumer Warp Specialization

FA3 把一个 thread block 内不同 warp group 分成不同角色。

可以先理解为：

```text
┌──────────────────── Thread Block ────────────────────┐
│                                                     │
│ Producer Warp Group                                 │
│   └── TMA: HBM → Shared Memory                     │
│                                                     │
│ Consumer Warp Group                                 │
│   └── WGMMA: Shared Memory → Tensor Core compute   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

传统思路比较像：

```text
每个 warp 都：
load → compute → load → compute
```

而 warp specialization 变成：

```text
一批 warp 专门喂原料
另一批 warp 专门做计算
```

为什么会更好？

因为角色固定以后：

- producer 可以更专注地维护 TMA pipeline；
- consumer 可以连续发 WGMMA；
- 两边通过 barrier / shared-memory stage 同步；
- 数据移动和计算更容易 overlap。

---

# 6. Shared Memory 在 FA3 里像“传送带”

可以把 Shared Memory 看成 producer 和 consumer 之间的缓冲区：

```text
HBM
 │
 │ TMA
 ▼
[Shared Buffer A] ← producer 填
[Shared Buffer B] ← producer 填
 │
 │ consumer 取
 ▼
Tensor Core
```

如果只有一个 buffer：

```text
producer 填
↓
consumer 用
↓
consumer 用完
↓
producer 才能覆盖
```

就不能充分 overlap。

所以要做多阶段 buffering。

---

# 7. Ping-Pong Scheduling 是什么？

FA3 使用一种 ping-pong 风格的调度，让不同 consumer warp group 交替推进工作。

可以用两组工人来类比：

```text
Consumer Group A
Consumer Group B
```

大致希望形成：

```text
A 发起一阶段矩阵乘
B 准备/执行另一阶段
A 处理后续
B 再接上
```

像乒乓球一样来回切换。

重点不在“必须背具体哪一拍是谁做”，而是理解：

> **通过显式软件调度，让硬件上异步执行的 Tensor Core 工作彼此交错，减少 pipeline bubble。**

---

# 8. 第二层异步：GEMM 与 Softmax overlap

这是 FA3 非常漂亮的一点。

Attention：

```text
QKᵀ → Softmax → PV
```

直觉上看似有严格依赖：

```text
必须先算完 QKᵀ
才能 softmax
再才能 PV
```

但注意 FlashAttention 是 **block-wise** 处理。

假设：

```text
block 0
block 1
block 2
```

当：

```text
block 1 的 QKᵀ
```

正在 Tensor Core 里算时，另一些执行资源可能可以处理：

```text
block 0 的 softmax update
```

于是时间线从：

```text
GEMM0 → softmax0 → GEMM1 → softmax1
```

变成更像：

```text
Tensor Core: [GEMM0][GEMM1][GEMM2]
Softmax:             [soft0][soft1][soft2]
```

这就是 intra-warpgroup GEMM-softmax overlapping 的直觉。

---

# 9. 为什么 H100 上 Softmax 更“碍眼”了？

假设上一代 GPU：

```text
QK GEMM     8 ms
softmax     2 ms
PV GEMM     8 ms
```

softmax 只占：

```text
2 / 18
```

但新一代 Tensor Core 把 GEMM 大幅加速：

```text
QK GEMM     3 ms
softmax     2 ms
PV GEMM     3 ms
```

softmax 的相对占比突然变大。

所以：

> **Tensor Core 越快，非 Tensor Core 操作越容易成为瓶颈。**

这其实是 FA2 “减少 non-matmul FLOPs”思想的进一步延伸。

FA3 更进一步：

> 不只减少它，还尝试把它藏在 GEMM 执行时间后面。

---

# 10. “Overlap”到底是不是把计算量减少了？

不是。

例如：

```text
GEMM = 5 ms
softmax = 2 ms
```

串行：

```text
5 + 2 = 7 ms
```

如果可以完全 overlap：

```text
max(5,2) ≈ 5 ms
```

softmax 还是算了 2ms 的工作。

只是它和 GEMM 同时发生。

所以：

> **Latency 优化不只有“少做工作”这一条路，也可以通过并行和隐藏延迟。**

---

# 11. 第三部分：为什么 FA3 强调 FP8？

H100 对 FP8 Tensor Core 提供非常高的理论吞吐。

如果 Attention 可以使用 FP8：

```text
每个元素更小
→ HBM / Shared Memory 搬运压力下降

Tensor Core FP8 throughput 更高
→ QKᵀ / PV 更快
```

理论上非常诱人。

但问题是：

> Attention 对数值范围很敏感。

尤其：

\[
\operatorname{softmax}(QK^T)
\]

如果 Q/K 量化误差把 logits 扭曲，softmax 后的概率分布可能发生明显变化。

---

# 12. FP8 的问题不是只剩“精度少一点”

低精度量化必须决定：

```text
用什么 scale？
scale 的粒度多大？
异常值怎么办？
不同 block 的动态范围差很多怎么办？
```

如果全 tensor 共用一个很大的 scale：

```text
极端 outlier
   ↓
迫使整体 scale 变大
   ↓
大部分普通值的有效分辨率变差
```

因此 FA3 采用更细粒度的 block quantization 思路，并研究如何降低 FP8 Attention 的数值误差。

---

# 13. Incoherent Processing 的直觉

论文中的 incoherent processing 是一个数值技巧。

不要先被名字吓到。

核心直觉可以理解成：

> **在量化前对数据做一种不会改变最终 Attention 数学意义、但会让数值能量更均匀分布的变换，从而减少少量异常维度对 FP8 scale 的破坏。**

可以类比：

原始向量：

```text
[0.1, 0.2, 0.1, 18.0]
```

有一个超大 outlier。

量化 scale 被 18 主导。

如果经过一种保持内积结构的正交/符号混合变换后，能量更均匀：

```text
[8.8, -8.7, 9.1, -9.0]
```

就可能更适合有限动态范围表示。

真实算法比这个例子严谨得多，但理解它的系统目的即可：

> **让 FP8 更好量化，而不是单纯追求低 bit。**

---

# 14. 为什么这种变换不能改变 Attention？

Attention 的关键分数是：

\[
qk^T
\]

如果对 Q 和 K 施加一个适当的正交变换：

\[
q' = qR,
\qquad
k' = kR
\]

且：

\[
RR^T=I
\]

那么：

\[
q'k'^T
= qRR^Tk^T
= qk^T
\]

所以数学内积保持不变。

这就给了工程优化空间：

> 可以改变数值表示分布，但保持理论 Attention score。

---

# 15. FA3 的三个核心技术终于可以放在一起了

```text
FlashAttention-3
│
├── 1. Producer/Consumer Warp Specialization
│      └── TMA load 与 Tensor Core compute overlap
│
├── 2. GEMM / Softmax Interleaving
│      └── 非矩阵运算藏进 Tensor Core 执行窗口
│
└── 3. FP8 Attention
       ├── block quantization
       └── incoherent processing
```

注意这三件事分别针对不同瓶颈：

```text
内存/计算串行
→ async data movement

Tensor Core / softmax 串行
→ execution overlap

数值精度过高导致吞吐受限
→ FP8
```

---

# 16. FA3 的性能数字怎么读？

论文在 H100 上报告：

- FP16 FlashAttention-3 相比 FA2 通常约 1.5–2.0× 加速；
- FP16 峰值可达到约 740 TFLOPs/s，约为 H100 理论峰值的 75%；
- FP8 可接近 1.2 PFLOPs/s；
- 其 FP8 方案相对一个简单 FP8 baseline 有更低数值误差。

真正应该记的不是具体数字，而是：

> **在新一代 GPU 上，性能的关键从“减少 HBM IO”进一步转向“把多个硬件 pipeline 重叠起来”。**

---

# 17. FlashAttention 三代到底分别解决什么？

```text
FA1
│
├── 问题：Attention Matrix 产生巨大 HBM IO
└── 解法：Tiling + Online Softmax

FA2
│
├── 问题：GPU parallelism / warp partition 不够好
└── 解法：Sequence parallelism + better work partition

FA3
│
├── 问题：Hopper 异步能力和 FP8 没吃满
└── 解法：TMA/WGMMA pipeline + softmax overlap + FP8
```

最浓缩的记忆：

```text
FA1 = IO
FA2 = Parallelism
FA3 = Asynchrony
```

---

# 18. FA3 与 FlashInfer 的关系

不要把两者混成竞争方案。

FlashAttention-3 更关注：

> **一个 dense/causal Attention kernel 在 Hopper 上如何高效执行。**

FlashInfer 更关注：

> **真实 LLM serving 中 paged / ragged / shared-prefix / sparse / decode 等多种 Attention workload 如何统一和动态调度。**

两者的层次可以画成：

```text
Serving Runtime
vLLM / SGLang / TensorRT-LLM
        │
        ▼
Attention Engine / Backend
FlashInfer / FA kernels / custom kernels
        │
        ▼
GPU Hardware
A100 / H100 / B200 ...
```

---

# 19. 一个厨房比喻

FA1：

> 厨师老跑远处仓库拿菜，所以把菜分块搬到灶台边。

FA2：

> 菜已经在灶台边了，但厨师分工不好；有人忙死，有人闲着，所以重新排班。

FA3：

> 厨师和传菜员已经很快了，现在要让：
>
> - 传菜员在厨师炒上一盘时准备下一盘；
> - 一组厨师炒菜时，另一组处理调味；
> - 使用更轻、更高吞吐的原料表示（FP8）；
>
> 让整个厨房形成真正的流水线。

---

# 20. AI Infra 最值得记住的 8 个 Insight

1. **硬件换代会改变最佳 kernel schedule。**
2. **高性能不只是减少数据搬运，也包括把数据搬运隐藏在计算后面。**
3. **异步不是“更快的单条指令”，而是创造 overlap 的能力。**
4. **Warp specialization 是 GPU kernel 中的 producer-consumer 架构。**
5. **Tensor Core 越快，softmax 等 non-matmul 工作相对越重要。**
6. **低精度性能与数值算法必须共同设计。**
7. **FP8 并不是直接 cast 就结束，scale/outlier/误差控制才是难点。**
8. **FlashAttention 系列本质上是一条 algorithm-hardware co-design 的演化线。**

---

# 21. 读完后你应该能回答

1. TMA 和普通线程 load 有什么概念上的区别？
2. 为什么 producer / consumer warp 可以提升性能？
3. “GEMM 和 softmax overlap”为什么数学上可行？
4. 为什么 Tensor Core 越快，softmax 越容易成为瓶颈？
5. FP8 Attention 为什么比普通 FP8 GEMM 更难？
6. FA1、FA2、FA3 各自的核心关键词是什么？

---

## 主要参考资料

- Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, arXiv:2407.08608 / NeurIPS 2024.
- Tri Dao, *FlashAttention-2*, arXiv:2307.08691.
- NVIDIA Hopper Tuning Guide.
- NVIDIA CUDA Programming Guide：Tensor Memory Accelerator / asynchronous copies.
- NVIDIA CUTLASS documentation：Hopper warp-specialized GEMM.
