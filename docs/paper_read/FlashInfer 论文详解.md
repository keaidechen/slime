# FlashInfer 论文详解：从 Attention、KV Cache 到 LLM 推理 Runtime

> **标题缩写与首次术语说明**：FlashInfer 是面向大语言模型推理的高性能算子与 Attention runtime（运行时）库；LLM = **Large Language Model（大语言模型）**；MLSys = **Conference on Machine Learning and Systems（机器学习与系统会议）**；AI Infra = **Artificial Intelligence Infrastructure（人工智能基础设施）**；KV Cache = **Key-Value Cache（键值缓存）**；JIT = **Just-In-Time（即时编译）**；GPU = **Graphics Processing Unit（图形处理器）**；MHA = **Multi-Head Attention（多头注意力）**；GQA = **Grouped-Query Attention（分组查询注意力）**；MQA = **Multi-Query Attention（多查询注意力）**；MLA = **Multi-head Latent Attention（多头潜在注意力）**；IR = **Intermediate Representation（中间表示）**；DSL = **Domain-Specific Language（领域专用语言）**；OI = **Operational Intensity（运算强度，单位数据搬运对应的计算量）**；LSE = **Log-Sum-Exp（对指数和取对数的数值稳定统计量）**；BSR = **Block Sparse Row（块稀疏行存储格式）**；TTFT = **Time To First Token（首 Token 延迟）**；ITL = **Inter-Token Latency（相邻 Token 延迟）**。本文中的 **kernel** 是 GPU 核函数，**scheduler** 是调度器，**block sparse** 是块稀疏表示，**composable format** 指可组合的数据布局抽象。 另外：CUDA = **Compute Unified Device Architecture（NVIDIA GPU 并行计算平台与编程模型）**；HBM = **High Bandwidth Memory（高带宽内存）**；SM = **Streaming Multiprocessor（流式多处理器）**；CTA = **Cooperative Thread Array（协作线程阵列）**；CPU = **Central Processing Unit（中央处理器）**；FP16 = **16-bit Floating Point（16 位浮点格式）**；I/O = **Input/Output（输入/输出）**；API = **Application Programming Interface（应用程序编程接口）**；LLVM 是现代编译器基础设施项目（名称历史上源于 **Low Level Virtual Machine**）；ARM 是主流精简指令集 CPU 架构；MLC = **Machine Learning Compilation（机器学习编译）**。

> 论文：**FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving**  
> 会议：**MLSys 2025**  
> 阅读目标：从零 AI Infra 背景理解 FlashInfer 为什么出现、它解决什么问题，以及它与 FlashAttention、PagedAttention、vLLM、SGLang 的关系。

---

## 0. 一句话理解 FlashInfer

FlashInfer 不是一种新的 Attention 数学公式，也不是新的模型结构。

它更像：

> **Attention Runtime + Kernel Library + JIT Compiler**

它要解决的问题是：真实 LLM 在线推理里，Attention workload 已经不再是一个固定形状、固定存储方式的矩阵计算，而是伴随 Paged KV Cache、Prefix Cache、Sliding Window、Speculative Decoding、GQA 等机制，变得高度动态、碎片化和异构。

FlashInfer 希望给这些复杂 Attention workload 建立一个统一的执行层。

---

# 1. 先把 FlashInfer 放进历史技术演化里

如果不理解前面的技术为什么出现，FlashInfer 的很多设计会显得很突兀。

```text
2017 Transformer
      ↓
Attention 成为核心计算，但 O(N²) 且 GPU 内存访问昂贵
      ↓
2022 FlashAttention
      ↓
解决：单次 Attention 怎么在 GPU 上算得更快
      ↓
2023 vLLM / PagedAttention
      ↓
解决：推理过程中不断增长的 KV Cache 怎么高效管理
      ↓
2023 SGLang / RadixAttention
      ↓
解决：不同请求之间重复的 Prefix KV Cache 怎么复用
      ↓
Speculative Decoding / Prefix Cache / Tree Attention
Long Context / GQA / MQA / Sliding Window ...
      ↓
Attention workload 越来越不规则
      ↓
2025 FlashInfer
      ↓
统一：
KV Cache 数据结构
+ Attention Kernel
+ Kernel 定制
+ Runtime 动态调度
```

可以把问题演化概括成两句话：

- FlashAttention 时代的问题是：**一个 Attention 怎么算快？**
- FlashInfer 时代的问题是：**在线同时服务大量不同长度、不同 KV 布局、不同 Attention 变体的请求时，怎么仍然算快？**

---

# 2. LLM 推理为什么天然分成 Prefill 和 Decode？

假设用户输入：

> 中国最大的城市是什么？

LLM 推理基本分成两个阶段。

## 2.1 Prefill

第一次把整个 prompt 喂给模型。

假设 prompt 有 1000 个 token：

```text
Q: 1000 tokens
K: 1000 tokens
V: 1000 tokens
```

Attention 中的大致矩阵乘法是：

\[
Q_{1000\times d}K^T_{d\times1000}
\]

这是比较大的矩阵乘法。

GPU 很擅长这种大规模并行计算，因此 Prefill 往往更加 **compute-bound（算力受限）**。

所谓 compute-bound，可以粗略理解为：

> 数据已经给到了 GPU，但算术单元本身不够快，计算吞吐成为瓶颈。

---

## 2.2 Decode

Prefill 完成以后，模型开始逐 token 生成。

例如当前只有一个新的 Query token：

```text
Q: 1 token
KV: 1001 tokens
```

再生成一个：

```text
Q: 1
KV: 1002
```

于是核心计算更像：

\[
1\times d \quad \times \quad d\times N
\]

这里每次只处理 1 个 Query token，但为了算 Attention，需要读取之前很长的一整段 K/V。

于是 GPU 的问题变成：

> 计算量不算很大，但需要从显存里不断搬运大量 KV 数据。

因此 Decode 往往是 **memory-bandwidth-bound（显存带宽受限）**。

这是理解 FlashInfer 非常重要的背景：

> **Prefill Attention 和 Decode Attention 数学形式很像，但从 GPU workload 的角度，它们是两种完全不同的任务。**

---

# 3. KV Cache 是什么？

Transformer Attention 的经典公式：

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt d}\right)V
\]

假设模型已经生成：

```text
A B C D E
```

现在要生成 F。

A～E 对应的 K/V 在之前已经计算过了。

如果每生成一个新 token，都重新计算：

```text
K_A K_B K_C K_D K_E
V_A V_B V_C V_D V_E
```

显然非常浪费。

于是推理系统会把以前算过的 K/V 存起来：

```text
GPU Memory

token A → K_A V_A
token B → K_B V_B
token C → K_C V_C
token D → K_D V_D
token E → K_E V_E
```

这就是 **KV Cache**。

下一次只需要计算新 token F 对应的 K/V：

```text
K_F V_F
```

然后让 Q_F 去和过去所有 K 做 Attention。

所以 KV Cache 本质上是在用显存换计算。

---

# 4. 为什么 KV Cache 最终会变成一个 AI Infra 问题？

单请求时 KV Cache 很简单。

但线上 serving 不是一个请求，而是很多请求同时存在：

```text
用户 A：context = 300
用户 B：context = 15,000
用户 C：context = 2,000
用户 D：刚结束
用户 E：刚加入
```

而且：

- 请求不断加入；
- 请求不断结束；
- 每个请求长度不同；
- KV Cache 每 decode 一步还会继续增长。

如果每个请求都预留一整块连续显存，就会产生大量浪费和内存碎片。

因此 KV Cache 不再只是一个 Tensor，而逐渐变成了一个：

> **Memory Management System**

这直接推动了 PagedAttention 的出现。

---

# 5. PagedAttention：把操作系统分页思想搬到 KV Cache

vLLM 的 PagedAttention 借鉴了操作系统 Virtual Memory / Paging。

操作系统不会要求：

```text
Virtual Page 0
Virtual Page 1
Virtual Page 2
```

在物理内存里也连续存放。

而是允许：

```text
Virtual Page 0 → Physical Page 7
Virtual Page 1 → Physical Page 103
Virtual Page 2 → Physical Page 22
```

中间通过 Page Table 建立映射。

PagedAttention 对 KV Cache 做了类似的事情。

逻辑 token：

```text
token 0
token 1
token 2
token 3
```

物理上可以在：

```text
GPU KV block 7
GPU KV block 31
GPU KV block 4
GPU KV block 99
```

这意味着一个关键变化：

> **KV Cache 从此不一定连续。**

这对 GPU kernel 影响很大，因为 kernel 不能再默认“沿着一个连续 Tensor 顺序读取 KV”。

---

# 6. RadixAttention：不只是分页，还要复用 Prefix

很多真实请求会共享长 Prefix。

例如 100 个用户都有同一个 system prompt：

```text
You are ChatGPT ...
[3000 tokens]
```

如果每个请求都重新计算并保存这 3000 token 的 KV Cache，就很浪费。

SGLang 的 RadixAttention 会把公共 prefix 放进 radix tree：

```text
             Shared System Prompt
                  3000 tokens
                       │
          ┌────────────┼────────────┐
          │            │            │
        User A       User B       User C
```

这样公共 Prefix KV Cache 可以只保留一次，多个请求共享。

到这里，KV Cache 的逻辑结构已经从简单连续数组，逐渐变成：

- Page Table
- Radix Tree
- Shared Prefix
- Sliding Window
- Sparse Mask
- Tree Attention

这正是 FlashInfer 出场的背景。

---

# 7. FlashInfer 的第一个核心问题：这些 KV Layout 能不能统一？

FlashInfer 给出的答案是：

> **Block Sparse Matrix Representation**

论文把 Page Table、Radix Tree、Sparse Mask 等访问关系，统一看成 Query 和 KV Block 之间的连接关系。

例如规定：

- 行：Query
- 列：KV Block

如果某个 Query 需要访问某个 KV Block，就记为非零；否则记为零。

```text
          KV Block
        0 1 2 3 4 5

Query A █ █ █ . . .
Query B █ █ █ █ . .
Query C █ █ █ . █ █
```

这就是一个 Sparse Matrix。

它的关键意义不是“矩阵里有很多 0”这么简单，而是：

> **把不同 KV Cache 数据结构统一成同一种 Query × KV connectivity abstraction。**

于是 Page Table、Radix Tree、Sparse Mask 等不再必须分别对应完全不同的 kernel 实现。

---

# 8. 什么是 Block Sparse？

普通 Sparse Matrix 可能按单个元素记录：

\[
A=\begin{bmatrix}
1&0&0&3\\
0&0&4&0\\
0&5&0&0
\end{bmatrix}
\]

只记录非零元素的位置。

Block Sparse 则不是一个元素一个元素处理，而是一块一块：

```text
██ .. .. ██
██ .. .. ██

.. .. ██ ..
.. .. ██ ..
```

这更适合 GPU，因为 GPU 天然喜欢 tile/block 粒度的数据和计算。

---

# 9. 为什么这是 FlashInfer 很重要的“大统一”思想？

以前很容易变成：

```text
PagedAttention → 一个 kernel
RadixAttention → 一个 kernel
Tree Attention  → 一个 kernel
Sparse Attention → 再写一个 kernel
```

FlashInfer 希望变成：

```text
Page Table ─────┐
Radix Tree ─────┼──→ Block Sparse Representation
Sparse Mask ────┤
Tree Attention ─┘
                        ↓
                 Attention Kernel
```

这种思路很像编译器：

```text
C / C++ / Rust
      ↓
    LLVM IR
      ↓
x86 / ARM / GPU
```

所以可以把 FlashInfer 的 block-sparse abstraction 理解成一种 Attention 世界里的“中间表示”。

---

# 10. Composable Formats：共享 Prefix 为什么可以减少 HBM Traffic？

假设有三个请求：

```text
Q1 ── shared prefix ── A
Q2 ── shared prefix ── B
Q3 ── shared prefix ── C
```

它们有一大段相同 KV：

```text
shared KV
████████████████
```

如果每个 query 都独立读取：

```text
Q1 load shared KV
Q2 load shared KV
Q3 load shared KV
```

同一份 KV 就可能被从 HBM 反复读取。

而 shared prefix 在 block sparse matrix 中天然形成 dense submatrix：

```text
Q1 ███████
Q2 ███████
Q3 ███████
```

于是 FlashInfer 可以让多个 Query 一起处理这一块共享 KV。

例如：

```text
HBM
 ↓
Shared Memory
 ↓
Q1 ┐
Q2 ├─ reuse same KV
Q3 ┘
```

这样共享 KV 可以在更快的片上存储中被多个 query 复用。

这就是论文中的 **Composable Formats** 思路。

而每个 request 自己独有的 suffix，又可以用更细的 block format 单独处理。

关键点：

> FlashInfer 不一定需要真正移动或重排 KV Cache 数据本身，而可以通过重建 indices / indptr 等 metadata，建立不同 sparse view 指向同一份底层 KV。

---

# 11. GPU Memory Hierarchy：为什么“少读 HBM”这么重要？

可以先用一个粗略层级理解：

```text
                 快 / 小
                   ↑
               Register
                   │
            Shared Memory
                   │
                L2 Cache
                   │
          HBM / Global Memory
                   ↓
                 慢 / 大
```

HBM 本身已经很快，但 GPU 算术单元更快。

因此 GPU kernel 经常不是“算得太慢”，而是：

> **数据搬不过来。**

于是 GPU kernel 优化最常见的目标之一就是：

> 从 HBM 读取一次，在 Register / Shared Memory 中尽量多 reuse。

FlashAttention 是这个思想。

FlashInfer 的 Composable Format 进一步把问题扩展到：

> 不只是一个 Query 内部能不能 reuse，同一个 Prefix KV 能不能跨多个 Query reuse？

---

# 12. FlashAttention 和 FlashInfer 到底是什么关系？

最容易混淆的地方就在这里。

## FlashAttention

核心问题：

> **一个 Attention 怎么高效计算？**

主要思想包括：

- tiling；
- online softmax；
- 减少 HBM IO；
- kernel fusion。

## FlashInfer

核心问题：

> **真实 LLM serving 中，各种异构、不规则 Attention workload 怎么统一、高效执行？**

所以可以理解成：

```text
FlashAttention
      │
      │ 底层 Attention 高效计算思想
      ↓
FlashInfer
      ├── KV Format
      ├── Sparse Loading
      ├── JIT
      ├── Runtime Scheduling
      └── Serving Integration
```

FlashInfer 并不是推翻 FlashAttention，而是在 serving 场景继续向上扩展问题边界。

---

# 13. Operational Intensity：为什么 Decode 特别难优化？

论文给出：

\[
OI=O\left(\frac{1}{1/l_{qo}+1/l_{kv}}\right)
\]

其中：

- \(l_{qo}\)：Query / Output sequence length
- \(l_{kv}\)：KV sequence length

Operational Intensity 可以粗略理解成：

\[
OI=\frac{\text{计算量 FLOPs}}{\text{内存访问 Bytes}}
\]

即：

> 每搬 1 Byte 数据，可以做多少计算。

如果 OI 很低：

```text
搬很多数据
只做一点计算
```

通常更容易 memory-bound。

如果 OI 很高：

```text
搬一份数据
做大量计算
```

更容易 compute-bound。

LLM serving 中通常：

\[
l_{qo}\le l_{kv}
\]

所以可近似为：

\[
OI=O(l_{qo})
\]

Prefill 时：

\[
l_{qo}\gg1
\]

OI 高。

Decode 时：

\[
l_{qo}=1
\]

OI 很低。

因此：

> **Decode 是典型的 KV memory-traffic dominated workload。**

---

# 14. GQA 为什么不仅能减少 KV Cache，还能提高推理效率？

普通 MHA 可能是：

```text
32 Query Heads
32 Key Heads
32 Value Heads
```

GQA 可能是：

```text
32 Query Heads
8 Key Heads
8 Value Heads
```

即多个 Q Head 共用同一份 K/V。

定义：

\[
g=\frac{H_q}{H_{kv}}
\]

那么一份 KV 可以被更多 Query head reuse。

因此 GQA 不只是：

> KV Cache 更小。

从 Infra 角度还有另一层价值：

> **相同 KV 数据被更多计算复用，提高 Operational Intensity，降低相对 memory traffic。**

---

# 15. FlashInfer 的第二个统一：统一 Attention 变体

今天的 Attention 已经不是只有：

\[
softmax(QK^T)V
\]

而是存在很多变体：

- RoPE
- Sliding Window
- Custom Mask
- Logits Soft Cap
- FlashSigmoid
- MLA
- 各种模型特定 Attention Variant

如果每出现一个变体就单独维护一个 CUDA kernel：

```text
attention_v1.cu
attention_v2.cu
attention_v3.cu
attention_v4.cu
```

长期维护成本会非常高。

FlashInfer 因此提供 **Customizable Attention Template**。

大致把 Attention 拆成：

```text
QueryTransform
KeyTransform
ValueTransform
        ↓
      QKᵀ
        ↓
LogitsTransform
LogitsMask
        ↓
      Softmax
        ↓
       × V
        ↓
OutputTransform
```

主体 FlashAttention skeleton 可以复用，只把变体逻辑作为定制点插进去。

---

# 16. 例子：RoPE Fusion 为什么有效？

普通实现可能是：

```text
Kernel 1:
Q → RoPE(Q)
K → RoPE(K)

write HBM
      ↓
Kernel 2:
read Q/K
Attention
```

中间 RoPE 输出需要：

```text
write HBM
read HBM
```

FlashInfer 可以把它融合：

```text
┌────────────────────────┐
│    fused CUDA kernel   │
│                        │
│ Q/K → RoPE → Attention │
└────────────────────────┘
```

这样中间结果可以保留在 Register / Shared Memory 中，不需要额外 HBM round trip。

这就是 kernel fusion 对 serving 延迟有价值的典型例子。

---

# 17. 可定制性和高性能天然有冲突

越通用的代码通常越容易出现：

```python
for ...:
    if ...:
        ...
```

性能往往不够极致。

而最快的 CUDA kernel 通常非常 specialized：

```text
head_dim = 128
dtype = FP16
tile = 64 × 128
mask = causal
GPU = Hopper
```

因此系统设计中经常存在：

```text
Flexibility
    ↑
    │ tension
    ↓
Performance
```

FlashInfer 的解决方式之一是 **JIT Compilation**。

---

# 18. JIT 是什么？

JIT = Just-In-Time Compilation。

不是提前把所有可能 kernel 都编译好，而是在知道具体 workload 后：

```text
GPU = H100
head_dim = 128
dtype = FP16
Attention = sliding window
Q tile = 16
KV tile = 64
```

再生成对应的 specialized kernel：

```text
Attention Specification
        ↓
FlashInfer Template
        ↓
      CUDA
        ↓
    JIT Compile
        ↓
Specialized Kernel
```

于是达到：

```text
上层：灵活
      ↓
JIT specialization
      ↓
底层：高度 specialized
```

这是很典型的 compiler/runtime 思维。

---

# 19. 为什么 FlashInfer 要支持不同 Tile Size？

假设一个 kernel 的 Query tile 是：

```text
128 rows
```

Prefill 时：

```text
Q length = 4096
```

很好，可以充分利用大 tile。

但 Decode：

```text
Q length = 1
```

如果还使用 128-row tile：

```text
█ valid query
............................
............................
```

大量资源没有实际工作。

所以 FlashInfer 根据 workload 选择不同的 Query/KV tile size。

直觉上：

```text
Decode
Q ≈ 1
↓
Small Q Tile
```

```text
Prefill
Q ≫ 1
↓
Large Q Tile
```

这也是为什么 serving kernel 很难只有一个“万能配置”。

---

# 20. FlashInfer 的第三个核心问题：Dynamic Workload

真实 serving 中不同请求的 KV 长度差异非常大：

```text
Request A: KV = 100
Request B: KV = 200
Request C: KV = 10000
Request D: KV = 80
```

如果简单地：

```text
CTA0 → A
CTA1 → B
CTA2 → C
CTA3 → D
```

很快会出现：

```text
CTA0 done
CTA1 done
CTA3 done

CTA2 █████████████████████████████████
```

大量 GPU 计算资源空闲，只剩一个超长任务还没结束。

这就是 **Load Imbalance**。

---

# 21. SM 和 CTA 是什么？

GPU 可以粗略理解为有很多计算工作站：

```text
GPU
├── SM0
├── SM1
├── SM2
├── SM3
└── ...
```

SM = Streaming Multiprocessor。

CUDA 会把工作拆成 Thread Block，也常称 CTA。

```text
CTA0
CTA1
CTA2
CTA3
...
```

再由 GPU 把 CTA 调度到 SM。

如果 CTA 工作量差别很大：

```text
CTA0 ███
CTA1 █████
CTA2 █████████████████████████
CTA3 ██
```

最终就会出现大量 SM idle。

---

# 22. FlashInfer 怎么做动态负载均衡？

核心思想：

> **把超长 KV 切成多个更小的 chunk。**

例如：

```text
Request C: KV = 10000
```

切成：

```text
C1 = 2000
C2 = 2000
C3 = 2000
C4 = 2000
C5 = 2000
```

于是 workload 从：

```text
A B C D
```

变成：

```text
A B C1 C2 C3 C4 C5 D
```

再把这些工作尽可能平均分给 CTA/SM。

论文使用类似：

\[
cost(l_q,l_{kv})=\alpha l_q+\beta l_{kv}
\]

的 cost model 估算任务大小，再做动态调度。

目标从：

```text
SM0 ██
SM1 ████
SM2 █████████████████████
SM3 █
```

变成更接近：

```text
SM0 ██████████
SM1 ██████████
SM2 ██████████
SM3 ██████████
```

---

# 23. Attention 可以随便把 KV 切开吗？

这是一个非常关键的问题。

假设：

```text
KV
████████████████
```

切成：

```text
KV1 = ████████
KV2 = ████████
```

分别计算：

\[
Attention(Q,KV_1)
\]

\[
Attention(Q,KV_2)
\]

最终输出不能简单做：

\[
O=O_1+O_2
\]

因为 Attention 中间存在 Softmax normalization。

这就引出 FlashInfer 非常重要的数学基础：**Attention Composition**。

---

# 24. Attention Composition：为什么拆开以后还能正确合并？

对于一部分 KV 集合 \(I\)，定义：

\[
Z_I=\sum_{i\in I}e^{qk_i}
\]

以及：

\[
O_I=\frac{\sum_{i\in I}e^{qk_i}v_i}{Z_I}
\]

类似地得到：

\[
O_J,Z_J
\]

那么完整 Attention 可以写成：

\[
O_{I\cup J}=\frac{Z_IO_I+Z_JO_J}{Z_I+Z_J}
\]

因此每一个 KV chunk 不只返回局部 Output，还返回 normalization information。

论文用：

\[
LSE=\log Z
\]

并定义：

\[
AttentionState=(O,LSE)
\]

然后定义合并操作：

\[
State(I\cup J)=State(I)\oplus State(J)
\]

这个合并操作具有：

- Associativity（结合律）
- Commutativity（交换律）

即：

\[
(A\oplus B)\oplus C=A\oplus(B\oplus C)
\]

以及：

\[
A\oplus B=B\oplus A
\]

这意味着 Attention 可以像 Reduction 一样被拆分和重新组合。

---

# 25. 为什么 Attention State 是 FlashInfer 很深的一个 Insight？

因为从此长 KV 可以：

```text
                Long KV
                   │
      ┌────────────┼────────────┐
      ↓            ↓            ↓
   Chunk 1      Chunk 2      Chunk 3
      ↓            ↓            ↓
   State 1      State 2      State 3
      └────────────┼────────────┘
                   ↓
                Reduce
                   ↓
              Final Output
```

于是系统可以自由地：

```text
Split
  ↓
Parallel Compute
  ↓
Schedule
  ↓
Load Balance
  ↓
Reduce
```

所以 **Attention State 的可组合性，是 FlashInfer dynamic load balancing 的数学基础。**

---

# 26. 又出现一个系统矛盾：Dynamic Scheduling vs CUDA Graph

正常 GPU kernel launch 大概是：

```text
CPU
 ↓ launch
GPU Kernel 1

CPU
 ↓ launch
GPU Kernel 2

CPU
 ↓ launch
GPU Kernel 3
```

每次 launch 都有 CPU overhead。

LLM decode 每生成一个 token，都要跑很多 Transformer layer 和大量 kernel，因此 kernel launch overhead 会越来越明显。

NVIDIA 提供 CUDA Graph：

```text
第一次：
record kernel A → B → C → D

以后：
replay()
```

这样可以减少 CPU 反复 launch 的开销。

但是 CUDA Graph 偏好相对静态的执行结构：

```text
fixed grid
fixed pointers
fixed graph structure
```

而 serving workload 偏偏极度动态：

```text
Step 1:
A=500 B=1000 C=1200

Step 2:
A=501 B=1001 C finished D joined
```

于是形成：

```text
Dynamic Scheduling
        ↑
        │ conflict
        ↓
Static CUDA Graph
```

---

# 27. FlashInfer 的解决方式：Plan / Run 分离

FlashInfer 把运行过程拆成：

```text
            CPU
             │
           plan()
             │
根据 request / KV length
生成 scheduling metadata
             │
             ↓
      Workspace Buffer
             │
             ↓
            GPU
           run()
             │
      CUDA Graph replay
```

## Plan

动态计算：

```text
CTA0 做什么
CTA1 做什么
CTA2 做什么
...
```

## Run

尽量保持：

```text
fixed grid size
fixed workspace pointer
fixed kernel structure
```

真正变化的只是 workspace 里的 scheduling metadata。

于是可以同时保留：

- workload 动态性；
- CUDA Graph replay 能力。

可以把它记成一个非常经典的系统设计模式：

\[
\boxed{Dynamic\ Plan + Static\ Run}
\]

---

# 28. FlashInfer 的整体架构

到这里可以把论文整体抽象成三层：

```text
               FlashInfer
┌────────────────────────────────────┐
│            KV Storage              │
│                                    │
│ Page Table ───────┐                │
│ Radix Tree ───────┼→ Block Sparse  │
│ Sparse Mask ──────┘                │
├────────────────────────────────────┤
│            Compiler                │
│                                    │
│ Attention Variant                  │
│ Hardware                           │
│ KV Layout                          │
│       ↓                            │
│      JIT                           │
│       ↓                            │
│ Specialized Kernel                 │
├────────────────────────────────────┤
│             Runtime                │
│                                    │
│ Sequence Lengths                   │
│       ↓                            │
│ Load-Balancing Scheduler           │
│       ↓                            │
│ Scheduling Plan                    │
│       ↓                            │
│ CUDA-Graph-Compatible Run          │
└────────────────────────────────────┘
```

---

# 29. 用“三个统一”记住整篇 FlashInfer

## 统一一：统一数据表示

```text
Page Table
Radix Tree
Sparse Mask
Tree Attention
```

统一成：

\[
\boxed{Block\ Sparse\ Representation}
\]

## 统一二：统一 Attention 计算变体

```text
RoPE Attention
Sliding Window
Custom Mask
FlashSigmoid
...
```

统一到：

\[
\boxed{Attention\ Template + JIT}
\]

## 统一三：统一 Runtime 调度

```text
Request A: 100
Request B: 500
Request C: 10000
```

通过：

\[
\boxed{Dynamic\ Load\ Balanced\ Scheduling}
\]

同时保持：

\[
\boxed{CUDA\ Graph\ Compatibility}
\]

这三个“统一”基本就是整篇论文的主线。

---

# 30. FlashInfer 和 vLLM / SGLang 不是同一层

可以粗略把 LLM serving stack 画成：

```text
┌───────────────────────────────┐
│        Application            │
│ Agent / Chat / Search         │
├───────────────────────────────┤
│        Serving Engine         │
│                               │
│ vLLM / SGLang / MLC           │
│                               │
│ batching                      │
│ request scheduler             │
│ KV cache manager              │
│ prefix cache                  │
├───────────────────────────────┤
│        Kernel Backend         │
│                               │
│ FlashInfer                    │
│                               │
│ Attention kernel              │
│ GPU-level scheduling          │
│ JIT                           │
├───────────────────────────────┤
│          CUDA                 │
├───────────────────────────────┤
│           GPU                 │
└───────────────────────────────┘
```

所以完全可以：

```text
SGLang + FlashInfer
```

或者：

```text
vLLM + FlashInfer
```

Serving Engine 更关心请求层面的：

- batching；
- request scheduling；
- KV cache lifecycle；
- prefix caching。

FlashInfer 更靠近 GPU kernel/runtime 层。

---

# 31. FlashInfer 和 Triton 又是什么关系？

可以先这样理解：

## CUDA

- 很底层；
- 开发难；
- 可以做非常强的 hardware-specific optimization。

## Triton

- 更高级的 GPU DSL；
- 更容易写高性能 kernel；
- 编译器自动完成部分底层工作。

## FlashInfer

不是一门 GPU 编程语言，而是：

> **面向 LLM inference 的 specialized kernel/runtime library。**

它可以使用高度特化的 CUDA implementation，同时配合 template、JIT 和 runtime scheduler。

---

# 32. 怎么正确看 FlashInfer 论文实验？

系统论文中的性能数字不能简单记成：

> “FlashInfer 永远快 X%。”

正确的阅读方式是：

> 在作者测试的 GPU、模型、batching 配置、sequence length distribution、serving engine、kernel backend 和 FlashInfer 版本下，这些设计是否真正转化成了端到端收益？

FlashInfer 的实验价值主要是证明：

- 动态 load balancing 对不均匀 sequence length workload 确实有价值；
- fused attention variant 能降低额外 memory traffic；
- serving engine 使用 specialized attention backend 可以改善 TTFT / ITL；
- irregular production workload 下统一 abstraction 不只是代码优雅，也可以带来实际性能收益。

---

# 33. 为什么 Skewed Workload 尤其重要？

漂亮 benchmark 可能是：

```text
1024 1024 1024 1024
```

但线上 request 更可能是：

```text
100 200 1000 50000
```

这种 sequence length distribution 非常 skewed。

于是固定的 workload partition 很容易导致 GPU load imbalance。

这告诉我们一个很重要的 MLSys 思维：

> **真实生产系统的困难往往不在规则矩形 Tensor，而在 dynamic、irregular、skewed workload。**

FlashInfer 很多设计就是围绕这个现实展开的。

---

# 34. FlashInfer 最深层的思想：把 Attention 重新定义成系统问题

在模型论文中，我们可能只写：

\[
Y=f(Q,K,V)
\]

但在真实 serving 系统里，它更像：

\[
Y=f(
Q,
KV,
KV\ Layout,
Mask,
Sequence\ Length,
Attention\ Variant,
Hardware,
Schedule
)
\]

FlashInfer 分别为这些复杂性建立 abstraction：

```text
KV Layout
   ↓
Block Sparse

Attention Variant
   ↓
JIT Template

Sequence-Length Dynamics
   ↓
Runtime Scheduler

GPU Launch Overhead
   ↓
CUDA Graph
```

这是非常典型、也非常值得学习的系统设计方法：

> **先找到稳定 abstraction，再在 abstraction 之下进行 specialization。**

---

# 35. AI Infra 视角最值得记住的 6 个 Insight

## Insight 1：Training Attention 与 Serving Attention 是不同 workload

特别是 Decode：

\[
Q=1,\quad KV\gg1
\]

高度 memory-bound。

---

## Insight 2：KV Cache 已经演化成 Memory Management System

PagedAttention、RadixAttention 本质都在改变：

```text
KV 怎么存
KV 怎么索引
KV 怎么共享
KV 怎么复用
```

---

## Insight 3：FlashAttention 和 FlashInfer 解决的是不同层次的问题

```text
FlashAttention:
How to compute one Attention efficiently

FlashInfer:
How to execute heterogeneous serving Attention efficiently
```

---

## Insight 4：Block Sparse 是强大的统一 abstraction

把：

```text
Page Table
Radix Tree
Sparse Mask
Tree Attention
```

统一为：

```text
Query × KV Connectivity
```

---

## Insight 5：Attention State 让 Attention 可以像 Reduction 一样拆分

\[
AttentionState=(O,LSE)
\]

支持：

```text
Split
→ Parallel Compute
→ Reduce
```

这是 dynamic load balancing 的数学基础。

---

## Insight 6：一个经典系统矛盾

\[
Dynamic\ Workload
\]

vs

\[
Static\ CUDA\ Graph
\]

FlashInfer 通过：

\[
\boxed{Dynamic\ Plan + Static\ Run}
\]

解决。

以后阅读 compiler、runtime、sparse computation 等系统论文时，还会经常遇到类似思想。

---

# 36. 用“GPU 超级工厂”比喻整篇论文

假设 GPU 是一家超级工厂。

Transformer 说：

> 我要生产 Attention。

FlashAttention 说：

> 你的工厂花太多时间在仓库和车间之间搬原材料了。一次搬进来以后，在车间里多加工几步再搬出去。

PagedAttention 说：

> 你的 KV 仓库很容易碎片化。不要让每个订单预占一整条连续货架，把仓库切成 Page。

RadixAttention 说：

> 这 100 个订单的前半部分原材料完全一样，不要存 100 份，公共部分共享。

于是仓库里逐渐出现：

```text
Page
Shared Prefix
Radix Tree
Sparse Mask
Sliding Window
Tree Attention
...
```

FlashInfer 进来说：

> 第一，我把各种仓库布局统一抽象成 Block Sparse Map。

> 第二，我根据订单类型和硬件即时生成最合适的生产线。

> 第三，超长订单不要霸占一个工作站，把它拆成小任务分给多个工作站。

> 第四，工厂流水线结构尽量保持固定，从而可以用 CUDA Graph 不断 replay；每天变化的只是任务表。

这基本就是 FlashInfer。

---

# 37. 后续学习建议

建议按照下面顺序继续深入：

1. **Figure 2：Page Table 为什么可以转换成 BSR / Block Sparse？**
2. **Figure 3：Composable Format 为什么真的能降低 HBM Traffic？**
3. **Attention State \((O,LSE)\) 为什么能够数学上正确 merge？**
4. **FlashInfer Scheduler 怎么把 KV Chunk 映射到 CTA / SM？**
5. **Plan / Run 为什么能同时兼容 Dynamic Scheduling 和 CUDA Graph？**
6. **再进入 FlashInfer 源码，看 BatchDecodeWithPagedKVCacheWrapper 等 API 如何映射到论文 abstraction。**

把前五个问题吃透以后，再进入源码，就不会只看到 CUDA template、workspace、indptr、indices、scheduler metadata，而能够知道每个数据结构背后到底在解决什么系统问题。

---

# 38. 最终总结

如果只允许记住一句话：

> **FlashInfer 的核心不是发明新的 Attention，而是为真实 LLM Serving 中高度异构和动态的 Attention workload，建立统一的数据表示、可定制的高性能 kernel 生成机制，以及动态 GPU runtime 调度体系。**

如果进一步压缩成三个关键词：

\[
\boxed{Block\ Sparse}
\]

\[
\boxed{JIT\ Attention\ Template}
\]

\[
\boxed{Dynamic\ Load\ Balancing}
\]

这三件事串起来，就是理解 FlashInfer 这篇论文最重要的主线。
