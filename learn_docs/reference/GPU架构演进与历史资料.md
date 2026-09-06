# GPU 架构演进与历史资料

<details>
<summary>本篇分段导航：按首读范围进入，其余二读</summary>

- [学习问答：原笔记 §02](#read-01)

</details>

<!-- learning-position -->
> **学习定位**：A8 · 参考。
> **前置**：[GPU、tensor 与通信基础](<../00_Foundations/README.md>)。
> **首读/二读**：保留硬件演进资料与出处，按具体模型和机器问题查阅；不作为当前 H20 规格表。
> **进度与实验**：[学习清单](<../学习清单.md>) · [总入口](<../README.md>)。
<!-- /learning-position -->

> 参考层：保留原笔记的架构演进资料与出处。具体型号、发布日期、版本号和性能结论属于原资料记录，本轮只做内容归位，未逐项重新核验；不作为当前 H20 环境的规格承诺。先学基础课，再按需要回查。


<a id="notebook-02"></a>


<a id="read-01"></a>

## 学习问答：原笔记 §02

### 02｜从 A100 → H100 → B200 → Rubin：NVIDIA GPU 架构为什么不断这样变化？

问题：前面已经认识了 SM、Tensor Core、Warp Scheduler、Register、Shared Memory、TMA、NVLink 等组件，但这些组件不是一开始就同时以今天的形态存在。沿着 A100 → H100 → B200 → Blackwell Ultra → Rubin 看，它们为什么不断变化？每一代到底是在解决上一代的什么瓶颈？

#### 先给结论：NVIDIA GPU 的主线不是“单纯堆更多 CUDA Core”

如果只看 SM 数量，会得到一个很不完整的故事：

```text
A100 (Ampere)       108 SM
        ↓
H100 (Hopper)       132 SM
        ↓
B200 (Blackwell)    148 SM
        ↓
B300 / Blackwell Ultra 约 160 SM
        ↓
Rubin               224 SM
```

SM 确实越来越多，但真正推动 AI 性能跨代增长的更重要因素是下面五条线同时演化：

1. **计算粒度越来越“大”**：CUDA Core → Tensor Core → Warp-level MMA → Warp-group MMA → CTA-pair / 更大协作域。
2. **数值精度越来越“低而聪明”**：FP32/FP16 → TF32/BF16 → FP8 → FP4 / microscaling → 更灵活的低比特与压缩表示。
3. **数据搬运越来越“异步、专用化”**：普通 LD/ST → Ampere async copy → Hopper TMA → Blackwell TMEM + 更强 TMA → Rubin enhanced TMA。
4. **存储层级越来越靠近计算**：Register / Shared Memory / L2 / HBM 之外，又出现专门服务 Tensor Core 的 Tensor Memory 等结构，减少中间结果反复进出 Register/HBM。
5. **“一张 GPU”不再是性能优化的终点**：NVLink / NVSwitch 带宽持续翻倍，优化单位从单 SM、单 GPU，扩展到多 GPU、整机、整 rack。

因此更准确的历史主线是：

```text
更多算力
   ↓
算力越来越难喂饱
   ↓
强化 HBM / L2 / Shared Memory / async copy
   ↓
Tensor Core 又更快
   ↓
数据搬运、同步、Kernel 边界成为瓶颈
   ↓
TMA / Warp Specialization / Thread Block Cluster
   ↓
单 die 到达 reticle / 功耗边界
   ↓
双 die 组成一个逻辑 GPU + 更低精度 + TMEM
   ↓
单卡继续变快后，多卡通信与 decode memory wall 更突出
   ↓
HBM4 + NVLink 6 + 更细粒度 kernel 协作 + rack-scale co-design
```

#### 一张表先看懂跨代变化

| 代际 | 代表 GPU | SM | 关键 Tensor 计算 | HBM / 带宽（代表配置） | 片上/搬运关键变化 | GPU-GPU 互联 | 这一代主要在解决什么 |
|---|---:|---:|---|---|---|---|---|
| Ampere | A100 | 108 | 3rd-gen Tensor Core；TF32/BF16；结构化稀疏 | 80 GB HBM2e，约 2 TB/s | async global→shared copy、async barrier、L2 residency control | NVLink 3，约 600 GB/s/GPU | 让 DL/HPC 大规模进入 Tensor Core + mixed precision，同时开始认真解决“算得快、搬不动” |
| Hopper | H100 SXM | 132 | 4th-gen Tensor Core；FP8；Transformer Engine；WGMMA | 80 GB HBM3，约 3.35 TB/s | TMA、Thread Block Cluster、Distributed Shared Memory、更大 Shared Memory | NVLink 4，约 900 GB/s/GPU | Transformer 成为核心 workload；需要把计算和数据搬运流水化，并扩大线程块协作粒度 |
| Hopper memory refresh | H200 | 与 H100 同属 Hopper | 计算架构基本延续 H100 | 141 GB HBM3e，约 4.8 TB/s | 重点不是改 SM，而是加容量/带宽 | NVLink 4 | 非常重要的“证据”：很多 LLM 已经不是缺 Tensor FLOPS，而是缺容量和带宽 |
| Blackwell | B200 | 148 | 5th-gen Tensor Core；`tcgen05.mma`；FP4/microscaling | 180 GB HBM3e，最高约 8 TB/s | 双 reticle die 统一成一颗 GPU；Tensor Memory (TMEM)；CTA-pair MMA；更大的 cluster 能力 | NVLink 5，约 1.8 TB/s/GPU | 单 die 尺寸、低精度推理、MoE、多卡扩展同时成为瓶颈；“一颗 GPU”开始本身就是多 die 系统 |
| Blackwell Ultra | B300 | 160（full implementation） | 5th-gen Tensor Core 强化；NVFP4；attention 加速 | 288 GB HBM3e，约 8 TB/s | 每 SM 256 KB TMEM；dual-thread-block MMA；更偏向大 batch / RL / inference | NVLink 5 | Blackwell 的方向继续加强：容量、attention、低精度与中间结果驻留 |
| Rubin | Rubin GPU | 224 | 3rd-gen Transformer Engine；更宽 K 维 Tensor Core；NVFP4；3-bit LUT 等 | 288 GB HBM4，最高约 22 TB/s | enhanced TMA、activation sparsity/adaptive compression、更细粒度 dependent-kernel triggering | NVLink 6，约 3.6 TB/s/GPU | agentic inference、长上下文、MoE、decode memory wall、kernel-to-kernel latency 和 rack-scale 通信 |

> 注：不同板卡/SXM/PCIe SKU 会有不同 SM、频率、显存和带宽，表里选的是数据中心主线中最具代表性的配置。理解架构时不要把某个 SKU 的数字误当成整代架构的硬上限。

---

#### 第一阶段｜A100：Tensor Core 已经很快，于是开始认真解决“数据怎么喂进去”

A100（Ampere）可以看成现代 AI GPU 的一个重要起点。它有 108 个 SM，第三代 Tensor Core，每个 SM 4 个 Tensor Core；同时加入 TF32、BF16、FP64 Tensor Core 与结构化稀疏支持。

如果只理解成“Tensor Core 比上一代快”会漏掉一个非常关键的变化：**A100 开始把数据搬运本身当成一等公民。**

Ampere 引入了从 global memory 直接异步复制到 shared memory 的指令路径，可以不再让数据先经过普通寄存器中转；同时配合异步 barrier，把“搬下一块 tile”和“算当前 tile”重叠起来。

```text
更早的直觉：
HBM → Register → Shared Memory → Compute

Ampere 开始强调：
HBM ── async copy ──→ Shared Memory
             │
             └── 与当前 tile 的计算 overlap
```

为什么要这样？因为 Tensor Core 吞吐越来越高后，计算单元真正怕的不是“没有乘法器”，而是：

```text
Tensor Core ready
↓
下一块 A/B tile 还没到
↓
等待
↓
昂贵的计算单元空转
```

所以从 A100 开始，一个非常重要的 GPU 优化思想越来越明显：

> **不是只提高 compute throughput，而是让数据移动、同步和计算形成 pipeline。**

A100 的另一条重要线是 **TF32 / BF16 / structured sparsity**。其本质是：AI 并不总需要 FP32 的每一位精度，如果允许使用更低精度或稀疏结构，就能用更少带宽、更少存储和更高 Tensor Core throughput 完成同样的训练/推理任务。

这为后面的 FP8、FP4 埋下了路线。

---

#### 第二阶段｜H100：Transformer 变成中心 workload，GPU 开始围绕“异步流水线”重新设计

到了 Hopper/H100，变化就不再只是“更快 Tensor Core”。H100 SXM 有 132 个 SM；第四代 Tensor Core 在相同数据类型下每 SM 的 MMA throughput 相比 A100 进一步提升，并引入 FP8 和 Transformer Engine。

但是 Hopper 最值得理解的并不是 FP8，而是三件互相关联的东西：

```text
TMA
+
Warp Specialization / WGMMA
+
Thread Block Cluster / Distributed Shared Memory
```

##### 1. TMA：把“搬 tensor”从普通线程指令里剥离出来

A100 的 async copy 已经能异步搬数据，但地址计算、tile 组织等仍然需要较多软件工作。Hopper 的 TMA（Tensor Memory Accelerator）进一步把多维 tensor 的搬运变成专用硬件能力。

它可以让很少的线程发起大块 1D～5D tensor transfer：

```text
HBM
 │
 │ TMA
 ↓
Shared Memory
 │
 ↓
Tensor Core
```

数据在飞的时候，其他 Warp 可以继续算。

于是一个 SM 内部非常自然地出现：

```text
Producer Warp
   │
   │ TMA：负责下一块数据
   ↓
Shared Memory
   │
   ↓
Consumer Warp Group
   │
   │ WGMMA
   ↓
Tensor Core
```

这就是前面讨论的 Warp Specialization 为什么在 Hopper/FlashAttention-3 里变得重要。

##### 2. WGMMA：矩阵计算的协作粒度从 Warp 扩大到 Warp Group

普通 CUDA 心智模型里：

```text
32 Threads = 1 Warp
```

但大矩阵 MMA 越来越适合让更多线程一起完成。Hopper 的 Warp-Group MMA（WGMMA）把典型协作粒度提升到 4 Warps = 128 Threads。

这反映出一个趋势：

> **GPU 的“高性能编程粒度”正在从 Thread/Warp，逐渐向 Warp Group / CTA / Cluster 变大。**

Thread 仍然存在，CUDA 也没有消失，但真正决定 Tensor Core 是否吃满的，越来越是更大的 tile 和协作单元。

##### 3. Thread Block Cluster：Block 不再完全是“孤岛”

传统 CUDA 最强的边界是：

```text
一个 Block 内：
Shared Memory + __syncthreads()

不同 Block：
通常通过 Global Memory / Kernel boundary 协作
```

Hopper 增加 Thread Block Cluster，使多个 Block 可以被安排成一个 cluster，并通过 Distributed Shared Memory 访问同 cluster 中其他 Block 的 Shared Memory。

于是硬件层级出现了一个新的“中间协作域”：

```text
单 Block Shared Memory
        ↓
Thread Block Cluster / Distributed Shared Memory
        ↓
L2
        ↓
HBM
```

为什么需要它？因为有些 tile/工作集已经大到一个 SM 的 Shared Memory 装不下，但如果每次都落到 HBM/L2，又太贵。

---

#### 一个值得单独记住的过渡：H200 为什么几乎没改 SM，却仍然很重要？

H200 仍然属于 Hopper，核心计算架构并不是全新一代；但显存提升到 141 GB HBM3e，带宽进一步提升到约 4.8 TB/s。

这件事本身非常说明问题：

> **如果 AI 性能只由 Tensor FLOPS 决定，那么 NVIDIA 没必要做这样一个“主要升级显存容量和带宽”的产品。**

LLM 尤其是 inference/decode 会反复读取权重和 KV Cache，往往是 memory-bound。模型越大、context 越长，capacity 也变成硬约束。

因此从 H100 → H200 可以把 GPU 的一个根本矛盾看得非常清楚：

```text
Tensor Core throughput 增长很快
            ↓
HBM bandwidth / capacity 增长跟不上
            ↓
Memory Wall 越来越突出
```

---

#### 第三阶段｜B200：单 die 也碰到物理极限，于是“一颗 GPU”本身开始变成多 die 系统

Blackwell/B200 的第一个巨大变化，甚至不在 SM 内部，而是在芯片封装层。

Blackwell GPU 由两个接近光刻 reticle 上限的 compute die 组成，并通过约 10 TB/s 的 die-to-die link 连接，对 CUDA 软件呈现为一个统一 GPU。

```text
以前：
┌───────────────┐
│  单个大 GPU die │
└───────────────┘

Blackwell：
┌──────────┐   10 TB/s   ┌──────────┐
│ GPU Die A│ ═══════════ │ GPU Die B│
└──────────┘             └──────────┘
          \               /
           \             /
             一个逻辑 GPU
```

为什么这么做？因为单 die 已经越来越接近：

- 光刻 reticle size 上限；
- 良率与成本压力；
- 晶体管数量增长；
- 功耗与布线压力。

所以 GPU 从“单 monolithic die”逐渐走向 package-level system。

这个变化非常关键，因为未来“GPU”这个词会越来越像：

> **一个封装内的计算系统，而不一定等于一块完整单 die。**

##### Blackwell 的第二个关键变化：Tensor Memory（TMEM）

在 Hopper 里，Tensor Core 的 accumulator 等中间状态会大量占用 Register File。Tensor Core 越来越快、tile 越来越大后，Register pressure 会反过来限制 occupancy 和数据复用。

Blackwell SM100 的 `tcgen05.mma` 引入 Tensor Memory（TMEM），专门服务 Tensor Core accumulator 等数据。

可以把数据路径粗略理解为：

```text
             TMA
HBM ─────────────────→ Shared Memory
                          │
                          │ A/B operands
                          ↓
                     Tensor Core
                          │
                          │ accumulator
                          ↓
                       TMEM
                          │
                          ↓
                       output
```

这解决的是一个很具体的问题：

> **Tensor Core 变得太快以后，Register File 既要保存线程状态又要承受巨大的矩阵 accumulator 压力。**

把一部分 Tensor 专用状态搬到 TMEM，可以释放 Register File、提高数据复用，也让 Tensor Core 的编程模型进一步专用化。

##### Blackwell 的第三个变化：CTA-pair cooperation

`tcgen05.mma` 可以让两个相邻 CTA/Block 协同执行一个 MMA。

注意这个趋势：

```text
早期：Thread / Warp
       ↓
Ampere：Warp-level MMA
       ↓
Hopper：Warp Group (4 warps)
       ↓
Blackwell：CTA pair / Cluster cooperation
```

**高性能矩阵计算正在不断扩大“协作粒度”。**

##### Blackwell 的第四个变化：FP4 / microscaling

Hopper 把 FP8 推进主流 AI；Blackwell 进一步推动 4-bit floating point 与 fine-grained scaling。

为什么不是一直用 FP16？因为低精度同时改善三件事：

```text
更少 bits
  ├─→ Tensor Core 每周期可算更多元素
  ├─→ HBM 需要搬的数据更少
  └─→ 同样显存能装更大模型 / KV cache
```

所以低精度并不只是“算力技巧”，它同时缓解 **compute wall + memory bandwidth wall + memory capacity wall**。

代价是：scale factor、dynamic range、quantization error 的管理越来越复杂，因此 Transformer Engine 也从一个简单“数据类型支持”逐渐演化成数值精度管理系统。

---

#### Blackwell Ultra / B300：为什么继续加 TMEM、容量和 Attention 能力，而不是只加 CUDA Core？

Blackwell Ultra 的 full GPU implementation 提升到 160 个 SM，并进一步强化第五代 Tensor Core；每个 SM 配置 256 KB Tensor Memory，同时把显存容量提高到 288 GB HBM3e。

NVIDIA 特别强调 attention-layer acceleration、NVFP4、dual-thread-block MMA 等，而不是单纯宣传 FP32 CUDA Core 数量。

这说明现代 AI workload 已经把 GPU 的瓶颈从“有没有足够多 ALU”推向：

```text
长上下文 Attention
MoE
RL / post-training
大 KV cache
低 batch / latency-sensitive inference
中间 tensor 的驻留与搬运
```

换句话说，**SM 内的“数据路径设计”开始和“算术单元数量”同等重要。**

---

#### 第四阶段｜Rubin：瓶颈进一步从单 Kernel 转向“Kernel 之间、GPU 之间、整 rack”

到 2026 年公布的 Rubin，方向更加明显。Rubin GPU 有 224 个 SM、896 个 Tensor Core、288 GB HBM4，峰值 HBM 带宽达到约 22 TB/s；NVLink 6 的 scale-up 带宽达到约 3.6 TB/s/GPU。

如果只把它看成“B200 的更大号版本”，会错过真正有意思的变化。

##### 1. HBM3e → HBM4：decode 已经明确成为 memory subsystem 问题

NVIDIA 对 Rubin 的描述非常直接：token-by-token decode fundamentally memory-system bound。

为什么？

训练中的大 GEMM 往往 arithmetic intensity 很高，Tensor Core 很容易成为主角；而 autoregressive decode 中，每生成一个 token，都需要不断读取：

- 模型权重；
- KV Cache；
- activation / intermediate state；
- MoE expert state。

batch 小时，每次搬来的权重不能被足够多 token 复用，于是：

```text
Tensor Core 峰值很高
        ↓
但大部分时间在等数据
        ↓
实际 tokens/s 由 memory bandwidth 主导
```

所以 Rubin 把 HBM bandwidth 从 Blackwell 的约 8 TB/s 推到最高约 22 TB/s，不是“附属升级”，而是在修正整个系统的 compute : memory balance。

##### 2. enhanced TMA：MoE 让“tensor descriptor 管理”本身都成为成本

MoE 会把 token 动态路由到不同 expert。不同 expert 的 tensor layout 可能相似，但地址不同。如果每次都重新构造/更新 descriptor，会出现 metadata 与控制开销。

Rubin 的 enhanced TMA 支持 inline descriptor update，使 kernel 能复用 tensor layout 描述，只在指令中动态覆盖 pointer/stride 等字段。

这说明 GPU 优化已经进入非常细的层次：

> **不只是“搬数据很贵”，连“告诉搬运引擎数据在哪里、长什么样”的 metadata 管理都值得做硬件优化。**

##### 3. activation sparsity + adaptive compression：不再只稀疏权重，也开始稀疏中间激活

A100 已经支持 structured sparsity，但 Rubin 把稀疏/压缩更深地推进到 attention / activation 路径。

例如 attention 中间结果可以在 Tensor Memory 中被压缩成结构化 sparse representation，使后续 softmax 和第二次 attention GEMM 处理更少数据。

这条线的本质是：

```text
如果搬数据很贵，
最好的数据搬运优化之一就是——根本不要搬那么多数据。
```

##### 4. 更细粒度 dependent-kernel triggering：Kernel boundary 本身开始成为瓶颈

传统同 stream 依赖：

```text
Kernel A 完整结束
        ↓
Kernel B 才开始
```

但 A 的某些 tile 可能早已计算完成，B 实际已经可以消费这些结果。Hopper/Blackwell 有 Programmatic Dependent Launch，可以提前 overlap 一部分 producer / consumer kernel；Rubin 又进一步强调更细粒度、数据驱动的 dependent work triggering。

于是优化粒度再次扩大：

```text
以前：优化 Kernel 内部
现在：Kernel A 和 Kernel B 之间也要 pipeline
```

这对 agentic inference 尤其重要，因为每个 token 由大量小 kernel / dependent stage 串起来，kernel 间的 bubble 会直接伤害 per-user latency。

##### 5. NVLink 6：GPU-GPU communication 越来越像“GPU 内部数据通路”的延伸

Rubin NVLink 6 达到约 3.6 TB/s/GPU，并强化 device-initiated NVLink、counted writes 等机制。

这背后的趋势不是“网卡更快”这么简单，而是：

```text
单 GPU
↓
多 GPU Tensor Parallel / Expert Parallel
↓
通信进入每层、每个 token 的 critical path
↓
GPU-GPU communication 必须被 kernel 直接驱动、直接 overlap
↓
整 rack 越来越像一个巨型 accelerator
```

---

#### 把四代放在一起：真正的“瓶颈迁移”

可以把 A100 → H100 → B200 → Rubin 看成瓶颈不断转移的过程：

```text
A100
问题：Tensor 计算需要更高 throughput，数据搬运开始跟不上
解法：Tensor Core + TF32/BF16 + async copy + sparsity

        ↓ Tensor Core 更快

H100
问题：Transformer pipeline 中 data movement / synchronization 开销突出
解法：FP8 + Transformer Engine + TMA + WGMMA + Block Cluster

        ↓ 单 SM/单 GPU 更快

B200
问题：single-die scaling、Register pressure、显存/通信、超大模型
解法：dual-die unified GPU + TMEM + CTA-pair + FP4 + HBM3e + NVLink 5

        ↓ GPU 已非常强，模型和 inference 又继续放大

Rubin
问题：decode memory wall、MoE routing、长 context、kernel bubbles、rack communication、tokens/watt
解法：HBM4 + enhanced TMA + activation sparsity/compression + finer kernel dependency + NVLink 6
```

最值得记住的是：

> **每当某一层被加速，瓶颈就会向下一层移动。**
>
> Tensor Core 快了 → Memory 成为瓶颈；Memory 快了 → synchronization / scheduling 成为瓶颈；单卡快了 → GPU-GPU communication 成为瓶颈；rack 快了 → power / cooling / reliability 又成为瓶颈。

这就是现代 GPU 架构为什么看起来越来越复杂。

---

#### 软件执行模型也在悄悄变化：从 Thread-centric 到 Tile / Pipeline-centric

CUDA 为兼容性一直保留：

```text
Grid → Block → Warp → Thread
```

但如果观察真正高性能 kernel 的编程方式，会看到另一条演进：

```text
早期 CUDA：
“每个 Thread 算哪个元素？”

        ↓
Tensor Core 时代：
“每个 Warp 算哪个 matrix tile？”

        ↓
Hopper：
“哪个 Warp 是 Producer？哪个 Warp Group 做 WGMMA？”

        ↓
Blackwell：
“CTA pair / Cluster 如何共同处理更大的 tile？
Accumulator 放 TMEM 还是 Register？”

        ↓
Rubin：
“多个 Kernel / GPU / rack 如何围绕 tensor flow 形成持续 pipeline？”
```

所以未来学习 FlashAttention、CUTLASS、Triton、cuTile 时，不要只盯着 `threadIdx.x`。现代 AI Kernel 的核心对象越来越是：

**Tile、Pipeline、Producer/Consumer、Data Movement、Collective、Dependency。**

---

#### 现在还存在哪些问题？

##### 问题 1｜Memory Wall 没有消失，而且 inference 时代更严重

HBM 从 A100 约 2 TB/s → H100 3.35 TB/s → B200 约 8 TB/s → Rubin 最高约 22 TB/s，已经增长巨大，但 Tensor Core throughput 与模型规模也同时增长。

尤其 decode：

```text
小 batch
+
每 token 反复读 weights / KV cache
+
长 context
=
低 arithmetic intensity
```

所以未来不可能只靠“再加 Tensor Core”。

##### 问题 2｜通信墙（Communication Wall）

TP、EP/MoE、长 context parallelism 都让 collective 和 point-to-point communication 进入 critical path。NVLink 从 600 GB/s → 900 GB/s → 1.8 TB/s → 3.6 TB/s 持续翻倍，本身就说明问题没有消失。

当模型扩展到 rack / pod 后：

```text
GPU compute latency
≈
memory latency/bandwidth
+
NVLink collective
+
scale-out network
```

不能再把网络看成“GPU 外面的附属设施”。

##### 问题 3｜Power Wall / Cooling Wall

HGX B200 已允许单 GPU 配置到约 1 kW 级别。随着 GPU 数量和 HBM/NVLink 功耗一起上升，rack 的供电、液冷、功率波动管理正在变成体系结构约束。

未来提升性能时真正要优化的是：

```text
tokens / joule
training progress / joule
useful FLOPs / watt
```

而不只是 peak TFLOPS。

##### 问题 4｜硬件越来越强，但软件越来越难把它用满

A100 时会写好 async copy 就已经很高级；Hopper 要理解 TMA、barrier、WGMMA、Warp Specialization；Blackwell 又增加 TMEM、`tcgen05.mma`、CTA-pair、block scaling；Rubin 再加入更细依赖和压缩机制。

这意味着：

> **Peak performance 与 programmable performance 之间的距离可能越来越大。**

这也是 CUTLASS、Triton、cuTile、compiler auto-tuning 越来越重要的原因——不能要求每个模型开发者手写架构专用 SASS/PTX。

##### 问题 5｜低精度越来越低，数值稳定性越来越难

FP8、FP4、3-bit / LUT 等格式可以显著提升吞吐和降低带宽，但并不是“把 dtype 改一下”即可。需要：

- scale factor 设计；
- block/micro-tensor scaling；
- accumulation precision；
- calibration / dynamic range；
- model-aware quantization；
- training / inference 软件协同。

因此硬件和算法之间会越来越共设计（co-design）。

##### 问题 6｜AI workload 越来越 irregular

传统大 GEMM 很规则，GPU 很喜欢；但未来 workload 越来越包含：

- MoE 动态路由；
- agent tool calls；
- retrieval；
- variable-length context；
- small-batch decode；
- speculative decoding；
- RL rollout；
- 多模态不同 shape。

这些 workload 会带来小 kernel、负载不均、同步与调度 bubble，使“拥有很多 Tensor Core”不等于“Tensor Core 一直有活干”。

---

#### 未来 GPU 很可能往哪里走？——把“已确认”与“推测”分开

Rubin 已经把很多未来方向明确展示出来；Rubin 之后更远的部分需要作为工程推断理解，而不是 NVIDIA 已公布规格。

##### 已经确认的方向：GPU 从 chip 走向 package / rack co-design

Blackwell 已经用双 die 组成统一 GPU；Rubin 继续使用多 compute die，并把 CPU、GPU、NVLink Switch、NIC/DPU、storage platform 作为统一系统设计。

因此“GPU 架构”正在从：

```text
SM microarchitecture
```

扩大成：

```text
SM
+
on-package memory
+
die-to-die interconnect
+
CPU-GPU coherent link
+
NVLink fabric
+
scale-out NIC
+
power/cooling
+
software runtime
```

##### 很可能继续发生 1｜更强的多 die / chiplet 化

单 die 的 reticle、良率与功耗限制不会消失，因此未来一个“GPU”包含更多 compute die / IO die / cache die 是自然方向。

真正的挑战会从“能不能连起来”变成：

> **怎样让软件看起来仍然像一块 GPU，同时尽量隐藏 NUMA / die locality。**

##### 很可能继续发生 2｜更多专用片上内存，而不只是更大的 Register/Shared Memory

TMEM 是非常重要的信号。未来高吞吐单元可能继续配套更专用的 local storage / scratchpad / accumulator memory，让数据尽量不回 HBM，也避免 Register File 成为所有数据的唯一高速落点。

##### 很可能继续发生 3｜Data Movement Engine 会越来越聪明

路线已经很清晰：

```text
普通 LD/ST
→ async copy
→ TMA
→ enhanced TMA + inline descriptor update
```

未来可能进一步把 gather/scatter、layout transform、compression/decompression、collective communication 等数据操作更多下沉到专用硬件，让 CUDA Core/Tensor Core 少做“搬运和 bookkeeping”。

##### 很可能继续发生 4｜调度粒度继续向更大的 cooperative domain 扩展

```text
Warp
→ Warp Group
→ CTA pair
→ Thread Block Cluster
→ Multi-SM cooperative execution
→ Multi-GPU / rack-level execution
```

这并不意味着 Thread/Warp 会消失，而是高性能 kernel 的“主要设计单位”会越来越大。

##### 很可能继续发生 5｜通信与计算进一步融合

未来的目标不会只是：

```text
compute
↓
NCCL
↓
compute
```

而会越来越像：

```text
Tensor Core compute
     ║
     ╠══ NVLink send / reduce
     ║
next tile compute
```

也就是 kernel 内直接通信、collective 与 GEMM/Attention overlap，甚至更多 in-network compute。

##### 很可能继续发生 6｜“精度”从固定 dtype 变成动态资源

从 FP16 → FP8 → FP4 → microscaling → adaptive compression，可以看到 dtype 不再只是 tensor 的静态属性。

更远期很可能是：

```text
不同 layer
不同 tile
不同 token / expert
甚至不同执行阶段
```

使用不同 precision / scale / sparsity，由 hardware + compiler + model runtime 协同决定。

##### 很可能继续发生 7｜最终优化指标从 FLOPS 变成 tokens / watt / dollar

AI 数据中心最大的硬约束越来越是：

```text
Power
Cooling
HBM supply
network
floor space
reliability
```

因此未来架构不会追求“单 GPU 理论 FLOPS 最大化”这么单一，而会越来越围绕：

- 每瓦 token throughput；
- 每美元训练进度；
- 每 rack 可服务的并发 agent 数；
- 故障后的有效集群利用率；
- 通信与内存的实际利用率。

---

#### 把这一章和前面的 SM / Warp 学习串起来

前面学的是一张 GPU 在某一代架构上的“静态剖面”：

```text
GPU
↓
GPC
↓
TPC
↓
SM
├─ Warp Scheduler
├─ CUDA Core
├─ Tensor Core
├─ LSU / TMA
├─ Register
└─ Shared Memory
```

这一章加入的是“时间轴”：这些组件为什么会变成今天这样。

```text
A100：
SM 内计算 + async data movement

H100：
SM 内部 producer/consumer pipeline
+ Warp Group
+ TMA
+ Cluster

B200：
SM 内出现 TMEM
+ CTA pair
+ 一颗 GPU 内变成双 die

Rubin：
SM / Kernel / GPU 之间的边界进一步被打通
+ HBM4
+ 更细 kernel dependency
+ NVLink 6
+ rack-scale co-design
```

> **最终心智模型：现代 GPU 的演化，本质上不是“核心越来越多”，而是在不断把瓶颈从 Compute → Memory → Scheduling → Communication → Power 往外推。GPU 的有效计算单位也从 Thread/SM，逐渐扩展到 Tile/Cluster/Multi-GPU/Rack。**

#### 官方资料建议

- NVIDIA Ampere Architecture In-Depth: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
- NVIDIA Hopper Architecture In-Depth: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- NVIDIA Hopper Tuning Guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/
- NVIDIA Blackwell Architecture: https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/
- NVIDIA Blackwell Tuning Guide: https://docs.nvidia.com/cuda/blackwell-tuning-guide/
- NVIDIA CUTLASS tcgen05 MMA Programming Guide: https://docs.nvidia.com/cutlass/
- Inside NVIDIA Blackwell Ultra: https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
- Inside NVIDIA Rubin GPU Architecture: https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/
