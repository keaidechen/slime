# Hopper 异步硬件：TMA、WGMMA 与流水线

<details>
<summary>本篇分段导航：按首读范围进入，其余二读</summary>

- [26. Hopper/H100 时代的新硬件概念：为 FlashAttention-3 做准备](#read-01)

</details>

<!-- learning-position -->
> **学习定位**：A8 · 专项。
> **前置**：[GPU、tensor 与通信基础](<../00_Foundations/README.md>)。
> **首读/二读**：TMA/WGMMA/warp specialization；先具备 tiling 与 Stream 基础，适用硬件与软件另行核验。
> **进度与实验**：[学习清单](<../学习清单.md>) · [总入口](<../README.md>)。
<!-- /learning-position -->

> 专项前置：GPU 执行、Stream/Event、GEMM tiling。原材料讨论 Hopper/H100 的设计；H20 实验须核对实际编译目标、backend 与软件支持，不套用其他设备的性能数字。


<a id="shared-26"></a>


<a id="read-01"></a>

## 26. Hopper/H100 时代的新硬件概念：为 FlashAttention-3 做准备


前面介绍的 `SM / warp / shared memory / Tensor Core / HBM / tiling` 足以帮助我们理解 FlashAttention-1/2。

但到了 FlashAttention-3，论文已经不只是“重新安排 tile”，而是在主动利用 Hopper（H100）引入的新异步执行能力。

因此这里补充一组之后会反复遇到的概念：

```text
H100 / Hopper
│
├── Tensor Core 更快
│
├── TMA
│   └── 专门负责 Global Memory ↔ Shared Memory 的大块异步搬运
│
├── WGMMA
│   └── 以 warp group 为粒度发起异步矩阵乘
│
├── warp specialization
│   ├── producer warps：搬数据
│   └── consumer warps：做矩阵乘
│
└── software pipeline
    └── 把“搬下一块数据”和“计算当前块”重叠起来
```

理解这一组概念之后，FlashAttention-3 的设计会从“很多 CUDA 黑话”变成一个非常直观的流水线问题。

#### 26.1 同步执行和异步执行有什么区别？

先假设一个非常简化的 kernel：

```text
1. 从 HBM 搬 tile A
2. 等 A 搬完
3. 算 A
4. 从 HBM 搬 tile B
5. 等 B 搬完
6. 算 B
```

时间线可能是：

```text
Memory: [load A]          [load B]
Compute:        [compute A]       [compute B]

时间 --->
```

你会发现：

- 搬数据时，Tensor Core 可能在等；
- Tensor Core 计算时，内存搬运单元也可能没有充分工作。

理想情况是：

```text
Memory: [load A][load B][load C][load D]
Compute:        [comp A][comp B][comp C][comp D]

时间 --->
```

也就是：

> **计算当前 tile 的同时，把下一个 tile 提前搬进来。**

这就是 pipeline / overlap 最重要的直觉。

---

#### 26.2 TMA 是什么？

TMA = **Tensor Memory Accelerator（张量内存加速器）**。

它是 Hopper GPU 上专门帮助大块 tensor 在：

```text
Global Memory / HBM
        ↕
Shared Memory
```

之间搬运的硬件机制。

在旧的思路里，线程往往需要参与很多地址计算和 load/store 指令：

```text
thread 0 load x0
thread 1 load x1
thread 2 load x2
...
```

TMA 更接近：

```text
“这里有一个二维/多维 tensor tile，
请硬件帮我异步搬到 shared memory 的这个位置。”
```

之后发起搬运的线程不必一直站在那里等待。

所以 TMA 的系统意义不是单纯“带宽更高”，而是：

> **把大量数据搬运工作从普通 CUDA 执行流水线中解耦出来，让计算与数据移动更容易并行。**

这也是 FlashAttention-3 能做 producer-consumer pipeline 的硬件基础之一。

---

#### 26.3 Warp Group 是什么？

前面说过：

```text
1 warp = 32 threads
```

Hopper 的一些 Tensor Core 指令进一步以多个 warp 组成的 **warp group** 为协作单位。

可以先粗略理解成：

```text
warp 0 ┐
warp 1 ├── 一个 warp group
warp 2 │
warp 3 ┘
```

即 4 个 warp、128 个线程共同参与某些矩阵乘工作。

这里最重要的不是背“128”，而是理解：

> GPU 的高性能矩阵运算已经不只是“每个 thread 自己算什么”，而越来越强调一组线程如何协作搬数据、发 Tensor Core 指令、同步和流水化。

---

#### 26.4 WGMMA 是什么？

WGMMA 可以粗略展开理解成：

> **Warp Group Matrix Multiply-Accumulate**。

你可以把它当成 Hopper 面向 Tensor Core 矩阵乘的一种更高级、异步的执行机制。

普通直觉：

```text
发起矩阵乘
↓
等矩阵乘结束
↓
做下一件事
```

而异步 WGMMA 给软件更多机会：

```text
发起 Tensor Core GEMM A
↓
GEMM A 在硬件中运行
↓
当前 warp group 可以安排其他独立工作
↓
真正需要结果时再等待
```

这给 FlashAttention-3 一个非常关键的机会：

> **把 Tensor Core 的矩阵乘和 softmax 等非矩阵运算交错执行。**

---

#### 26.5 什么是 Warp Specialization（warp 专职化：不同 warp group 承担不同流水线职责）？

传统 kernel 中，多个 warp 经常执行相似的工作流程：

```text
warp 0: load → compute → load → compute
warp 1: load → compute → load → compute
warp 2: load → compute → load → compute
warp 3: load → compute → load → compute
```

Warp Specialization 则让不同 warp 扮演不同角色。

例如：

```text
Producer Warp Group
        │
        │ TMA
        ▼
负责把 Q/K/V tile 搬进 Shared Memory

Consumer Warp Group
        │
        │ WGMMA
        ▼
负责 Tensor Core 矩阵乘
```

于是整个 block 更像一家流水线工厂：

```text
搬运工 ─────→ 工作台 ─────→ 计算工
Producer       Shared Mem     Consumer
```

当 consumer 在计算 tile 0 时：

```text
Producer: 搬 tile 1
Consumer: 算 tile 0
```

就能够产生 overlap。

---

#### 26.6 Double Buffer / Multi-stage Pipeline 是什么？

假设 shared memory 只有一个 buffer：

```text
Buffer A
```

consumer 还在用 A 时，producer 不能覆盖它。

所以常见方法是准备两组甚至多组 buffer：

```text
Buffer A
Buffer B
```

时间线：

```text
阶段 1:
Producer → Buffer A

阶段 2:
Consumer 算 A
Producer → Buffer B

阶段 3:
Consumer 算 B
Producer → Buffer A
```

A/B 来回切换：

```text
A → B → A → B → ...
```

这就是 double buffering 的直觉。

FlashAttention-3 中的 ping-pong scheduling 可以看成更复杂的类似思想：

> **让不同 warp group / pipeline stage 交替工作，尽量不让 Tensor Core 或数据搬运流水线空下来。**

---

#### 26.7 为什么 Tensor Core 变得越快，Softmax 反而越值得优化？

这是一个很容易反直觉的地方。

Attention 大体是：

```text
QKᵀ
 ↓
softmax
 ↓
P V
```

其中：

```text
QKᵀ 和 PV
```

主要是矩阵乘，可以使用 Tensor Core。

但 softmax 包含：

- max reduction；
- exp；
- sum reduction；
- normalization；

它并不是 Tensor Core 最擅长的 GEMM。

当 Tensor Core 还没那么快时：

```text
GEMM █████████████████
softmax ███
```

softmax 占比不大。

但 Tensor Core 变得越来越强以后：

```text
GEMM ██████
softmax ███
```

softmax 相对占比反而上升。

这就是典型的：

> **优化掉旧瓶颈之后，新的瓶颈浮现出来。**

FlashAttention-3 因此不仅关注 HBM IO，还重点研究：

```text
Tensor Core GEMM
        ↕ overlap
Softmax
```

---

#### 26.8 FP16、BF16、FP8 为什么会影响 kernel 设计？

低精度的基本意义是：

```text
更少 bit / element
→ 同样 HBM 带宽可以搬更多元素
→ Tensor Core 通常也能以更高吞吐计算
```

所以从：

```text
FP32 → FP16/BF16 → FP8
```

理论上性能越来越高。

但数值范围和精度也越来越有限。

Attention 又包含：

```text
exp(QKᵀ)
```

这类对数值范围敏感的操作，因此不能简单把所有东西粗暴改成 FP8。

后面的 FlashAttention-3 会介绍：

- block quantization；
- scaling；
- incoherent processing；

它们的共同目标是：

> **利用 FP8 Tensor Core 的高吞吐，同时控制 Attention 的数值误差。**

---

#### 26.9 把 Hopper 硬件能力映射到 FlashAttention-3

最终你只需要先记住这张关系图：

```text
Hopper / H100 新能力
│
├── TMA
│   └── 更容易异步搬 Q/K/V tile
│
├── WGMMA
│   └── 更容易异步执行 Tensor Core GEMM
│
├── Warp Specialization
│   └── Producer 搬运 / Consumer 计算
│
└── FP8 Tensor Core
    └── 更高低精度吞吐

             ↓

FlashAttention-3
│
├── 数据搬运和计算 overlap
├── GEMM 和 softmax overlap
└── FP8 Attention
```

如果 FlashAttention-1 的关键词是：

```text
IO-aware
```

FlashAttention-2 的关键词是：

```text
parallelism + work partitioning（工作划分：决定任务如何分给 thread/warp/CTA）
```

那么 FlashAttention-3 最适合记成：

```text
asynchrony + hardware-aware pipelining + low precision
```

---

#### Hopper 部分参考资料

- NVIDIA Hopper Tuning Guide（Tensor Memory Accelerator / Warp Specialization）
- NVIDIA CUDA Programming Guide（Asynchronous Data Copies / TMA）
- NVIDIA CUTLASS Documentation（Hopper Warp-Specialized GEMM）
- Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, NeurIPS 2024 / arXiv:2407.08608
