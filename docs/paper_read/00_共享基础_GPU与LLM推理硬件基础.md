# 00｜共享基础：GPU、显存与 LLM 推理系统硬件基础

> **标题缩写与首次术语说明**：GPU = **Graphics Processing Unit（图形处理器）**；LLM = **Large Language Model（大语言模型）**；CPU = **Central Processing Unit（中央处理器）**；HBM = **High Bandwidth Memory（高带宽内存，通常指 GPU 主显存）**；SRAM = **Static Random-Access Memory（静态随机存取存储器，本文主要指片上高速存储）**；DRAM = **Dynamic Random-Access Memory（动态随机存取存储器）**；SM = **Streaming Multiprocessor（流式多处理器，GPU 的基本计算执行单元）**；CTA = **Cooperative Thread Array（协作线程阵列，CUDA 中通常对应一个 thread block）**；GEMM = **General Matrix-Matrix Multiplication（通用矩阵-矩阵乘法）**；CUDA = **Compute Unified Device Architecture（NVIDIA 的通用 GPU 并行计算平台与编程模型）**；KV Cache = **Key-Value Cache（键值缓存）**。本文中的 **warp** 指 NVIDIA GPU 通常以 32 个线程组成的线程束，**kernel** 指一次在 GPU 上执行的并行核函数，**memory-bound** 指受内存/显存带宽限制，**compute-bound** 指受计算吞吐限制，**tiling** 指把大矩阵切成适合片上高速存储的小块。 另外：AI = **Artificial Intelligence（人工智能）**；I/O = **Input/Output（输入/输出）**；PCIe = **Peripheral Component Interconnect Express（高速外设互连总线）**；C2C = **Chip-to-Chip（芯片间互连）**；MHA = **Multi-Head Attention（多头注意力）**；GQA = **Grouped-Query Attention（分组查询注意力）**；MQA = **Multi-Query Attention（多查询注意力）**；TMA = **Tensor Memory Accelerator（张量内存加速器）**；WGMMA = **Warpgroup Matrix Multiply-Accumulate（warp group 级矩阵乘加）**；FP16/FP32/FP8 分别表示 **16/32/8 位浮点格式**；BF16 = **bfloat16（16 位 Brain Floating Point 格式）**；CUTLASS = **CUDA Templates for Linear Algebra Subroutines（NVIDIA 高性能线性代数 CUDA 模板库）**。 会议缩写：NeurIPS = **Conference on Neural Information Processing Systems（神经信息处理系统大会）**；ICLR = **International Conference on Learning Representations（国际学习表征会议）**；SOSP = **ACM Symposium on Operating Systems Principles（ACM 操作系统原理大会）**；OS = **Operating System（操作系统）**。

> **适用范围**：FlashAttention、PagedAttention/vLLM、SGLang/RadixAttention、FlashInfer 等论文的共同前置知识。
>
> 目标不是把你培养成 CUDA 工程师，而是让你读系统论文时，看到 `HBM / SRAM / SM / warp / kernel / memory-bound / page / KV cache` 这些词时知道作者到底在解决哪一层问题。

---

# 1. 先建立一张总地图：LLM 最终是在一台什么机器上跑？

一台典型的 AI 服务器，可以先粗略理解成：

```text
CPU
├── CPU cores
├── CPU cache
└── CPU DRAM
      │
      │ PCIe / NVLink-C2C 等连接
      ▼
GPU
├── 很多 SM（Streaming Multiprocessor）
│   ├── CUDA Cores
│   ├── Tensor Cores
│   ├── Registers
│   └── Shared Memory / L1
├── L2 Cache
└── HBM（GPU 显存）
```

对于这组论文，最重要的不是某一代 GPU 有多少个核心，而是理解三个层次：

1. **计算单元**：Tensor Core、CUDA Core；
2. **执行组织**：thread、warp、thread block/CTA、SM；
3. **存储层次**：register → shared memory/L1 → L2 → HBM → CPU DRAM。

系统优化的本质经常是：

> **让数据尽可能待在更靠近计算单元的地方，并尽量少在慢层级之间搬来搬去。**

这条原则就是 FlashAttention、FlashInfer 等工作的硬件出发点。

---

# 2. CPU 和 GPU 为什么不一样？

CPU 的设计目标通常是：

> 用少量非常强的核心，把复杂、分支很多、延迟敏感的任务快速完成。

GPU 的设计目标更接近：

> 用大量相对简单的执行单元，同时处理海量相似计算。

例如矩阵乘法：

\[
C = AB
\]

其中每个输出元素：

\[
C_{ij}=\sum_k A_{ik}B_{kj}
\]

不同的 \(i,j\) 可以高度并行，因此特别适合 GPU。

所以 Transformer 中的大量 GEMM（General Matrix-Matrix Multiplication，通用矩阵-矩阵乘法）非常适合 GPU。

但是：

> **GPU 算得快，不代表整个程序一定快。**

如果 GPU 每做一点计算，都要等大量数据从 HBM 搬进来，那么计算单元会处于“等数据”的状态。

这就是为什么 AI Infra 必须同时关心：

- FLOPs；
- 显存容量；
- 显存带宽；
- Cache / Shared Memory；
- 数据布局；
- kernel 调度。

---

# 3. 什么是 HBM？为什么论文天天讨论 HBM？

HBM = **High Bandwidth Memory（高带宽内存）**。

你可以把它近似理解为：

> GPU 自己的大容量主内存，也就是平时 `nvidia-smi` 看到的那几十 GB / 上百 GB“显存”的主体。

例如模型参数、activation、KV Cache 等大量数据，主要都放在 HBM 中。

HBM 的“High Bandwidth”已经比普通 CPU DRAM 快很多，但 GPU 的计算能力增长得更快，因此在很多 workload 里：

> **HBM 仍然相对太慢。**

一个很重要的观念是：

```text
数据在 HBM 里
    ↓ 读取
送到 SM 附近
    ↓
GPU 做计算
    ↓
结果写回 HBM
```

如果一个算法不停：

```text
HBM → compute → HBM → compute → HBM
```

那么即使 FLOPs 并不多，也可能很慢。

FlashAttention 就是从这里开始重新审视 Attention。

---

# 4. GPU Memory Hierarchy：显存并不是只有一层

可以先记住下面这张简化图：

```text
              更快、更小
                  ↑
             Register
                  │
      Shared Memory / L1
                  │
                L2
                  │
                HBM
                  │
             CPU DRAM
                  ↓
              更慢、更大
```

## 4.1 Register

Register 是线程直接使用的极小、极快存储。

例如一个线程正在累加：

```text
sum += a * b
```

`sum` 很可能就在 register 中。

特点：

- 极快；
- 每个线程可用数量有限；
- register 用得太多，会影响 GPU 同时容纳多少线程。

这会与后面讲的 **occupancy** 联系起来。

## 4.2 Shared Memory / SRAM

Shared Memory 是一个 thread block 内线程可以共享的片上存储。

可以理解成：

> 一个 SM 上非常快的小工作台。

典型优化方式：

```text
HBM 中的一块数据
      ↓
一次搬进 Shared Memory
      ↓
很多线程反复使用
```

而不是：

```text
线程 1 从 HBM 读
线程 2 又从 HBM 读
线程 3 又从 HBM 读
...
```

**Tiling（分块）** 的核心意义之一就是制造这种数据复用。

FlashAttention 论文经常把这类片上快速内存统称为 SRAM。

## 4.3 L2 Cache

L2 是 GPU 上更大、但比 shared memory 更远的一层缓存。

它对多个 SM 可见。

通常你不需要像 shared memory 那样显式地控制所有 L2 行为，但数据是否能命中 L2，会影响实际显存流量。

## 4.4 HBM

容量大，带宽高，但相对于片上内存仍然远得多。

模型参数和 KV Cache 主要驻留于此。

---

# 5. “带宽”和“延迟”到底是什么？

这两个词不要混淆。

## 带宽 Bandwidth

表示单位时间最多能搬多少数据，例如：

```text
TB/s
GB/s
```

类比高速公路：

> 一秒钟能通过多少辆车。

## 延迟 Latency

表示一次访问从发出请求到拿到数据需要多久。

类比：

> 一辆车从北京开到上海要多久。

AI kernel 经常通过：

- 并发；
- cache；
- prefetch；
- pipeline；

隐藏单次访问延迟，而最终受到总带宽限制。

所以 LLM decode 中经常说：

> **memory bandwidth bound**。

意思不是“显存很慢”，而是：

> 每生成一个 token 需要搬的数据太多，相对于计算量而言，HBM 总吞吐成为上限。

---

# 6. Thread、Warp、Block/CTA、SM：CUDA 的执行层次

这是以后读 FlashAttention-2 / FlashInfer 必须认识的一组词。

可以从大到小理解：

```text
GPU
└── 很多 SM
    └── 执行很多 Thread Block / CTA
        └── 一个 block 包含很多 threads
            └── threads 以 warp 为基本执行群组
```

## 6.1 Thread

CUDA 中最小的编程线程。

每个 thread 通常处理矩阵中的某几个元素。

## 6.2 Warp

NVIDIA GPU 中线程不是完全各自独立执行，而是以 **warp** 为重要执行单位。

经典 NVIDIA 架构中，一个 warp 包含 32 个线程。

可以粗略理解成：

```text
32 个线程
   ↓
一起执行一条指令
```

所以如果同一个 warp 中：

```text
16 个线程走 if 分支 A
16 个线程走 if 分支 B
```

硬件通常不能真正同时执行两条不同路径，而要分开执行，导致效率下降。

这就是 **warp divergence** 的直觉。

## 6.3 Thread Block / CTA

一组 threads 构成一个 block。

CTA（Cooperative Thread Array）在很多 NVIDIA 文档/论文中基本可以视作 thread block 这一层概念。

一个 block 的线程可以：

- 使用同一块 shared memory；
- 做 block 内同步。

## 6.4 SM

SM = Streaming Multiprocessor。

可以把 GPU 想成有很多“计算车间”：

```text
SM0
SM1
SM2
...
```

CUDA runtime 会把 thread blocks 调度到不同 SM 上执行。

如果一些 block 工作很快结束，而另一些特别慢，就会出现：

```text
SM0: idle
SM1: idle
SM2: █████████████ still working
SM3: idle
```

这就是 **load imbalance**。

FlashInfer 的动态 scheduler 就是在处理这种问题。

---

# 7. Tensor Core 是什么？为什么矩阵乘法特别快？

现代 NVIDIA GPU 不仅有普通 CUDA Core，还拥有专门加速矩阵运算的 **Tensor Core**。

它非常擅长类似：

\[
D=A\times B+C
\]

这样的矩阵乘加。

因此：

> 在 AI GPU 上，一个 FLOP 并不总是“成本一样”。

矩阵乘法 FLOP 可以被 Tensor Core 极高效执行，而：

- exp；
- max；
- divide；
- index manipulation；

这类非矩阵运算可能相对昂贵。

FlashAttention-2 的一个重要改进就是：

> 尽量减少 non-matmul FLOPs，把更多时间留给 Tensor Core 擅长的矩阵计算。

---

# 8. Kernel 是什么？

Kernel 可以理解为：

> CPU 发给 GPU 的一个“并行计算任务程序”。

例如标准 Attention 可能被拆成：

```text
Kernel 1: QK^T
Kernel 2: mask
Kernel 3: softmax
Kernel 4: dropout
Kernel 5: P @ V
```

每个 kernel 往往需要：

```text
从 HBM 读输入
→ 计算
→ 把结果写回 HBM
```

因此多个 kernel 之间的大中间 tensor 会反复进出 HBM。

---

# 9. Kernel Fusion 为什么有效？

假设有：

```text
A → operation1 → B → operation2 → C
```

分两个 kernel：

```text
Kernel 1:
read A
compute B
write B to HBM

Kernel 2:
read B from HBM
compute C
write C
```

融合以后：

```text
Kernel:
read A
compute B
B 留在片上
compute C
write C
```

省掉：

```text
write B
read B
```

因此 kernel fusion 往往不是为了减少数学计算，而是为了：

> **减少 HBM traffic 和 kernel launch overhead。**

FlashAttention 把多个 Attention 操作融合进一个 kernel，就是经典案例。

---

# 10. 什么是 Tiling？

假设你需要做一个巨大矩阵运算，但 shared memory 装不下整个矩阵。

最自然的方法是切块：

```text
完整矩阵
┌──────────────┐
│ □ □ □ □ □  │
│ □ □ □ □ □  │
│ □ □ □ □ □  │
└──────────────┘
```

每次只处理：

```text
┌─────┐
│ tile│
└─────┘
```

流程：

```text
HBM
 ↓
加载 tile 到 shared memory/register
 ↓
尽量多做计算
 ↓
必要结果写回 HBM
```

Tiling 的关键不是“把矩阵切小”本身，而是：

> **让同一批从 HBM 搬来的数据，在片上被重复使用很多次。**

FlashAttention 的整个核心算法就是围绕 Attention 设计一个正确的 tiling 方式。

---

# 11. Arithmetic Intensity / Operational Intensity

这是读系统论文非常重要的指标。

可以粗略定义：

\[
AI=\frac{\text{做了多少计算}}{\text{从内存搬了多少 Byte}}
\]

例如：

### 程序 A

```text
读 1 GB 数据
只做少量加法
```

Arithmetic Intensity 很低。

### 程序 B

```text
读 1 GB 数据
做大量矩阵乘法
```

Arithmetic Intensity 很高。

由此产生两个经典瓶颈。

---

# 12. Compute-bound 与 Memory-bound

## Compute-bound

程序速度主要受 GPU 计算能力限制。

也就是：

```text
数据已经喂得足够快
Tensor Core 一直满负荷计算
```

此时优化方向通常是：

- 更高 Tensor Core 利用率；
- 更好的 work partition；
- 更少无效 FLOPs。

## Memory-bound

程序速度主要受数据搬运速度限制。

此时可能出现：

```text
Tensor Core：我算完了，数据呢？
HBM：还在搬。
```

优化方向通常是：

- 少读 HBM；
- 少写 HBM；
- reuse；
- fusion；
- quantization；
- 更紧凑的数据结构。

FlashAttention 和 LLM decode 都非常强调 memory traffic。

---

# 13. Roofline Model 的直觉

Roofline Model 可以帮助判断一个 workload 应该优化什么。

横轴可以想象成 Arithmetic Intensity，纵轴是实际计算吞吐。

```text
性能
 ↑
 |              __________ 计算上限
 |            /
 |          /
 |        /
 |      /
 |_____/______________________→ Arithmetic Intensity
      内存带宽限制
```

左边：

> memory-bound。

右边：

> compute-bound。

重要 insight：

> **减少 FLOPs 不一定会更快。**

如果程序本来就在左边受内存带宽限制，FLOPs 再少一点可能没有意义；反而减少 HBM IO 更有价值。

这正是 FlashAttention 论文批评很多早期 approximate/sparse attention 工作的出发点之一。

---

# 14. Occupancy 是什么？

GPU 要隐藏内存延迟，需要同时准备很多可以执行的 warp。

如果一个 warp 在等数据：

```text
warp A: 等 HBM
```

SM 可以切换去执行：

```text
warp B
warp C
warp D
```

所以通常希望 SM 上同时有足够多 active warps。

这种资源占用程度常被称为 **occupancy**。

但 occupancy 受到：

- register 数量；
- shared memory 使用量；
- block 大小；

限制。

因此 tile 越大并不总越好：

```text
大 tile
→ reuse 可能更好
→ 但 shared memory/register 用量变大
→ 同时驻留 block 变少
→ occupancy 可能下降
```

这就是 kernel tuning 中典型的 trade-off。

---

# 15. 为什么 LLM 推理必须理解 Prefill 与 Decode？

一个请求：

```text
Prompt: 1000 tokens
```

推理通常分为两阶段。

## Prefill

一次性处理整个 prompt。

此时：

```text
Q length ≈ 1000
K/V length ≈ 1000
```

有大量大矩阵乘法，GPU 并行度较高。

## Decode

之后每一步只生成一个或少数 token：

```text
Q length = 1
K/V length = 1001
```

下一步：

```text
Q length = 1
K/V length = 1002
```

此时每一步计算量相对小，但必须读取历史 KV Cache。

所以 decode 常常表现为：

> **memory-bandwidth-bound。**

这也是为什么 KV Cache 管理会直接决定 serving throughput。

---

# 16. KV Cache：为什么一个“缓存”能成为系统核心？

Attention：

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt d}\right)V
\]

自回归生成第 \(t\) 个 token 时，前面 token 的 K/V 已经计算过。

如果每一步重新计算全部历史 K/V，非常浪费。

所以保存：

```text
token 1 → K1,V1
token 2 → K2,V2
...
token t → Kt,Vt
```

这就是 KV Cache。

一个粗略的每 token KV Cache 大小公式：

\[
\text{bytes/token}
=
2\times L\times H_{kv}\times d_h\times \text{bytes(dtype)}
\]

其中：

- `2`：K 与 V；
- \(L\)：Transformer 层数；
- \(H_{kv}\)：KV heads 数；
- \(d_h\)：head dimension。

例如一个仅作直觉演示的 MHA 配置：

```text
32 layers
32 KV heads
head_dim = 128
FP16 = 2 bytes
```

则：

\[
2\times32\times32\times128\times2
=524288\text{ bytes}
\]

约等于：

> **512 KiB / token**。

2000 token 就接近 1 GiB KV Cache。

现代 GQA/MQA 会明显降低这个数字，但它仍然可能成为 serving 中最大的动态显存消费者之一。

---

# 17. 为什么 KV Cache 会产生“内存碎片”？

真实 serving 中请求长度不同：

```text
A: 700 tokens
B: 14000 tokens
C: 1200 tokens
D: finished
E: just arrived
```

输出长度在开始时通常又不知道。

如果提前给每个请求预留最大长度：

```text
Request A: [used][unused..............]
Request B: [used........][unused......]
```

会浪费很多显存。

如果不断 malloc/free 不同大小区域，又可能出现：

```text
used | hole | used | small hole | used
```

虽然空闲总量够，但找不到足够大的连续区域。

这就是碎片问题。

PagedAttention 的核心就是把这个问题类比成操作系统的 virtual memory / paging。

---

# 18. Virtual Memory 与 Paging：为什么 vLLM 会借鉴操作系统？

操作系统希望进程看到：

```text
Virtual address
0 1 2 3 4 5 6 ...
```

但物理内存可以是：

```text
Physical page 7
Physical page 2
Physical page 91
...
```

中间通过 page table 映射：

```text
Virtual Page 0 → Physical Page 7
Virtual Page 1 → Physical Page 2
Virtual Page 2 → Physical Page 91
```

因此：

> **逻辑连续 ≠ 物理连续。**

PagedAttention 把 KV Cache 也拆成固定大小 block：

```text
Logical KV Block
       ↓ block table
Physical KV Block
```

这让请求可以按需增长，而不需要拥有一块连续的巨大显存。

---

# 19. Internal Fragmentation 与 External Fragmentation

这是读 PagedAttention 必须区分的两个词。

## Internal Fragmentation

已经分配了一块内存，但里面没有完全使用。

例如 block 容量 16 token：

```text
[10 tokens used][6 slots empty]
```

这 6 个 slot 是块内部浪费。

## External Fragmentation

空闲内存存在，但被分散在很多小洞中：

```text
used | free | used | free | used
```

没有一块足够大的连续区域。

固定大小 page/block 可以显著缓解外部碎片。

---

# 20. Copy-on-Write 为什么会出现在 LLM Serving？

假设同一个 prompt 要采样 4 个回答：

```text
Prompt
  ├── Answer A
  ├── Answer B
  ├── Answer C
  └── Answer D
```

Prompt 的 KV Cache 完全相同。

没必要复制 4 份：

```text
共享 prompt blocks
```

当某个 branch 要写入自己的新 token 时，再复制需要修改的最后一个共享 block。

这就是：

> **Copy-on-Write（写时复制）**。

它同样源自操作系统进程 `fork()` 的经典设计思想。

---

# 21. Prefix Cache 与 Radix Tree 的直觉

假设很多请求都包含：

```text
You are a helpful assistant...
```

如果这段 prefix 的 KV 已经算过，就可以直接复用。

但真实请求共享关系可能是多层的：

```text
                System Prompt
                /           \
          Conversation A   Few-shot Prompt
             /    \              /   \
           A1     A2            Q1    Q2
```

简单哈希表只能表达“整个 prefix 是否命中”。

Radix Tree 可以表达：

> **任意长度的最长公共前缀和树状共享关系。**

这就是 RadixAttention 的核心数据结构基础。

---

# 22. 四篇论文分别在硬件/系统哪一层？

可以用下面这张图记住：

```text
Transformer Attention 数学
        │
        ▼
FlashAttention
减少 HBM IO，优化单次 Attention kernel
        │
        ▼
PagedAttention / vLLM
把动态 KV Cache 做成 paged memory
        │
        ▼
RadixAttention / SGLang
在 paged KV 之上管理跨请求 prefix reuse
        │
        ▼
FlashInfer
把越来越多不规则 KV layout / Attention pattern
统一成可编译、可动态调度的 inference kernel/runtime
```

注意：

这并不是“后者淘汰前者”。

它们经常可以叠加：

```text
SGLang
  ↓ runtime / cache policy
Paged KV Cache
  ↓ physical storage
FlashInfer / FlashAttention-like kernels
  ↓ GPU execution
GPU HBM / SM / Tensor Core
```

---

# 23. 读论文时遇到这些词，可以直接这样翻译

| 术语 | 先这样理解 |
|---|---|
| HBM | GPU 大容量显存 |
| SRAM / Shared Memory | SM 上很快但很小的片上工作区 |
| Register | 单线程极快临时变量存储 |
| SM | GPU 的一个计算车间 |
| Warp | NVIDIA GPU 中一组协同执行的线程 |
| Thread Block / CTA | 被调度到 SM 上的一组线程 |
| Kernel | CPU 发给 GPU 的并行计算程序 |
| Kernel Fusion | 多个操作合成一个 kernel，减少中间 HBM IO |
| Tiling | 把大问题切块，在片上反复 reuse |
| GEMM | 通用矩阵乘法 |
| Tensor Core | 专门做矩阵乘加的硬件单元 |
| Compute-bound | 算力先到上限 |
| Memory-bound | 内存搬运先到上限 |
| Arithmetic Intensity | 每搬 1 Byte 数据做多少计算 |
| Occupancy | SM 上能维持多少活跃线程/warp 的程度 |
| KV Cache | 自回归推理缓存历史 token 的 K/V |
| Page / Block | 固定粒度的内存分配单位 |
| Block Table | logical block → physical block 的映射 |
| Prefix Cache | 复用已有 prompt prefix 的 KV |
| Radix Tree | 压缩前缀树，用于管理多层 prefix sharing |

---

# 24. 推荐阅读顺序

在 `AI Infra 论文讲解/` 中，建议按照：

```text
00｜共享基础：GPU、显存与 LLM 推理系统硬件基础
        ↓
FlashAttention
        ↓
PagedAttention / vLLM
        ↓
SGLang / RadixAttention
        ↓
FlashInfer
```

原因是这四篇恰好构成了一条非常好的 AI Infra 学习路径：

> **从 GPU IO → 动态显存管理 → KV Cache 复用 → 统一 inference kernel/runtime。**

---

# 25. 最值得先记住的 8 个硬件 Insight

1. **GPU 很快，但“搬数据”可能比“算数据”更贵。**
2. **HBM 是大仓库；Shared Memory/Register 是计算单元身边的小工作台。**
3. **Tiling 的本质是用一次 HBM 搬运换更多片上 reuse。**
4. **Kernel Fusion 的本质通常不是减少数学计算，而是减少中间结果反复读写 HBM。**
5. **FLOPs 少不等于 wall-clock 更快；必须看 workload 到底 compute-bound 还是 memory-bound。**
6. **Prefill 与 Decode 虽然都叫 Attention，但 GPU workload 特性差别很大。**
7. **KV Cache 是推理中的动态显存对象，因此内存管理会直接影响 batch size 和 throughput。**
8. **AI Infra 的很多经典创新，本质上是在硬件约束之上重新设计数据布局、执行顺序和调度方式。**

---

## 主要参考资料

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022, arXiv:2205.14135.
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023/ICLR 2024, arXiv:2307.08691.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023, arXiv:2309.06180.
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, arXiv:2312.07104 / NeurIPS 2024.


---

# 26. Hopper/H100 时代的新硬件概念：为 FlashAttention-3 做准备

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

## 26.1 同步执行和异步执行有什么区别？

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

## 26.2 TMA 是什么？

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

## 26.3 Warp Group 是什么？

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

## 26.4 WGMMA 是什么？

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

## 26.5 什么是 Warp Specialization（warp 专职化：不同 warp group 承担不同流水线职责）？

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

## 26.6 Double Buffer / Multi-stage Pipeline 是什么？

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

## 26.7 为什么 Tensor Core 变得越快，Softmax 反而越值得优化？

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

## 26.8 FP16、BF16、FP8 为什么会影响 kernel 设计？

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

## 26.9 把 Hopper 硬件能力映射到 FlashAttention-3

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

## Hopper 部分参考资料

- NVIDIA Hopper Tuning Guide（Tensor Memory Accelerator / Warp Specialization）
- NVIDIA CUDA Programming Guide（Asynchronous Data Copies / TMA）
- NVIDIA CUTLASS Documentation（Hopper Warp-Specialized GEMM）
- Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, NeurIPS 2024 / arXiv:2407.08608
