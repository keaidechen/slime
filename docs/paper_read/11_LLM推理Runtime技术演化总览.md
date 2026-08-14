# 11｜LLM 推理 Runtime 技术演化总览：从 FlashAttention 到 vLLM / SGLang / TensorRT-LLM

> **标题缩写与首次术语说明**：LLM = **Large Language Model（大语言模型）**；GPU = **Graphics Processing Unit（图形处理器）**；HBM = **High Bandwidth Memory（高带宽内存）**；I/O = **Input/Output（输入/输出，本文主要指数据搬运）**；TMA = **Tensor Memory Accelerator（张量内存加速器）**；WGMMA = **Warpgroup Matrix Multiply-Accumulate（warp group 级矩阵乘加）**；JIT = **Just-In-Time（即时编译）**；NCCL = **NVIDIA Collective Communications Library（NVIDIA 集合通信库）**；NVSHMEM 是 NVIDIA 的 GPU 对称共享内存通信库；IB = **InfiniBand（高带宽低延迟集群网络）**；IPC = **Inter-Process Communication（进程间通信）**；SLO = **Service Level Objective（服务等级目标）**；NIXL = **NVIDIA Inference Xfer Library（NVIDIA 推理数据传输库）**；RDMA = **Remote Direct Memory Access（远程直接内存访问）**；EAGLE = **Extrapolation Algorithm for Greater Language-model Efficiency（推测解码方法族）**；MTP = **Multi-Token Prediction（多 Token 预测）**；EPLB = **Expert Parallelism Load Balancer（专家并行负载均衡器）**。本文中的 **runtime** 是运行时系统，**kernel** 是 GPU 核函数，**scheduler** 是调度器，**router** 是请求路由层，**disaggregation** 指阶段/资源池解耦。 另外：KV Cache = **Key-Value Cache（键值缓存）**；FP8 = **8-bit Floating Point（8 位浮点格式）**；GEMM = **General Matrix-Matrix Multiplication（通用矩阵-矩阵乘法）**；OS = **Operating System（操作系统）**；CUDA = **Compute Unified Device Architecture（NVIDIA GPU 并行计算平台与编程模型）**；CTA = **Cooperative Thread Array（协作线程阵列）**；SM = **Streaming Multiprocessor（流式多处理器）**；PD = **Prefill-Decode（预填充-解码）**；CPU = **Central Processing Unit（中央处理器）**；FA1/FA2/FA3 = **FlashAttention-1/2/3**。 另外：AI = **Artificial Intelligence（人工智能）**；MoE = **Mixture of Experts（混合专家模型）**。

> 这份文档不是单篇论文，而是 `AI Infra 论文讲解/` 的阶段性总索引。
>
> 目标：当你读完某个概念后，知道它是在 **GPU kernel、KV memory、engine scheduler、cluster scheduler** 中的哪一层，以及下一篇论文为什么会出现。

---

# 1. 整条技术线先看一遍

```text
Transformer Attention
       │
       ▼
FlashAttention-1
减少 HBM IO
       │
       ▼
FlashAttention-2
提升 parallelism / warp work partition
       │
       ▼
FlashAttention-3
利用 Hopper async pipeline / FP8

================================================

传统 Static Batching
       │
       ▼
Orca
Iteration-Level Scheduling
Selective Batching
       │
       ▼
Continuous / In-Flight Batching
       │
       ▼
vLLM / PagedAttention
动态 KV Cache 分页管理
       │
       ▼
SGLang / RadixAttention
跨请求 Prefix KV 复用
       │
       ▼
FlashInfer
统一 irregular serving attention formats / kernels / runtime scheduling

================================================

Prefill + Decode Colocation
       │
       ▼
Chunked Prefill
缓解 Prefill 对 Decode 的 stall
       │
       ▼
DistServe
Prefill-Decode Disaggregation
       │
       ▼
现代分布式 Runtime
Router + P Pool + D Pool + KV Transfer Fabric
```

这其实是三条互相交叉的演化线：

1. **Kernel 线**；
2. **Memory/Scheduler 线**；
3. **Cluster Architecture 线**。

---

# 2. 第一条：Kernel 演化线

## FlashAttention-1

问题：

```text
Attention Matrix N×N
```

产生巨大 HBM traffic。

核心：

```text
Tiling + Online Softmax
```

关键词：

> IO-aware

---

## FlashAttention-2

FA1 之后：

```text
HBM IO 好很多
```

但 GPU：

```text
parallelism 不够
warp 分工不理想
shared-memory communication 偏多
```

核心：

```text
better sequence parallelism
better warp work partition
```

关键词：

> Parallelism

---

## FlashAttention-3

H100 新能力：

```text
TMA
WGMMA
FP8
```

FA2 没有充分利用。

核心：

```text
Data movement ↔ GEMM overlap
GEMM ↔ Softmax overlap
FP8
```

关键词：

> Asynchrony

---

# 3. 第二条：Batching / Scheduler 演化线

## Static Batching

```text
[A B C]
```

整个 batch 一起开始、一起结束。

对自回归 LLM 很差，因为每个请求输出长度不同。

---

## Orca

把 scheduling unit 从：

```text
Request
```

变成：

```text
Iteration
```

每轮：

```text
finished → remove
new → admit
```

关键词：

> Iteration-Level Scheduling

现代叫法常见：

```text
Continuous Batching
In-flight Batching
```

---

# 4. 第三条：KV Memory 演化线

Continuous batching 之后 KV lifecycle 变得高度动态：

```text
request joins
request grows
request finishes
```

如果还是连续预分配 KV：

```text
显存碎片 + 预留浪费
```

于是 vLLM：

```text
PagedAttention
```

把 OS Virtual Memory 思想引入 KV Cache。

关键词：

> Paging

---

# 5. 第四条：KV Reuse 演化线

PagedAttention 主要回答：

> KV 到底放显存哪里？

但如果 100 个请求共享：

```text
同一个 system prompt
```

即使显存管理高效，也不应该重新算 100 次。

于是 SGLang：

```text
RadixAttention
```

回答：

> 哪些 token prefix 对应的 KV 可以跨 request 复用？

关键词：

> Prefix Reuse

---

# 6. 第五条：Irregular Attention 演化线

有了：

```text
Paged KV
Radix Tree
Tree Attention
Sparse Mask
Sliding Window
Speculative Decode
```

Attention workload 越来越不规则。

FlashInfer：

```text
Block Sparse abstraction
Composable formats
JIT customizable attention
Dynamic scheduler
```

回答：

> 这些不同 KV layout / attention variant 怎么用统一、高性能的 kernel/runtime 层执行？

关键词：

> Unified Serving Attention Engine

---

# 7. 第六条：Prefill / Decode 干扰演化线

Continuous batching 后：

```text
Decode requests 正在稳定生成
```

突然进入一个：

```text
超长 Prefill
```

会出现：

```text
Generation Stall
```

第一种解决方向：

```text
Chunked Prefill
```

把长 prefill 切块。

更激进方向：

```text
DistServe
P/D Disaggregation（Prefill/Decode 分离部署）
```

直接让：

```text
Prefill GPU Pool
Decode GPU Pool
```

物理解耦。

---

# 8. 现代 Runtime 到底有几层？

建议以后永远用这张图定位概念：

```text
┌──────────────────────────────────────┐
│ Cluster / Fleet Layer                │
│ Router / Gateway / Load Balancer     │
│ P-D Pool Placement                   │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│ Serving Engine Layer                 │
│ vLLM / SGLang / TensorRT-LLM         │
│ Scheduler / KV Manager / ModelRunner │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│ Kernel / Operator Layer              │
│ FlashInfer / FlashAttention / GEMM   │
│ MoE / Sampling / Communication       │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│ CUDA / Collective Runtime            │
│ CUDA Graph / NCCL / NVSHMEM ...      │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│ Hardware                             │
│ GPU / HBM / NVLink / NVSwitch / IB   │
└──────────────────────────────────────┘
```

当别人说“scheduler”时，一定先问：

> **哪一层的 scheduler？**

---

# 9. 三种 Scheduler 不要混

## Cluster Scheduler / Router

决定：

```text
request 去哪个 engine / 哪个 GPU pool？
```

## Engine Scheduler

决定：

```text
这个 engine 下一 model iteration 算哪些 requests/tokens？
```

## Kernel Scheduler

决定：

```text
这个 Attention workload 如何切 CTA / tile / SM？
```

例如：

```text
SGLang Model Gateway
        ↓
SGLang Scheduler
        ↓
FlashInfer kernel scheduler
```

三个都叫 scheduling，但完全不同。

---

# 10. TensorRT-LLM / vLLM / SGLang 怎么比较？

不要用“一句话谁更快”来比较完整 runtime。

应该分维度。

| 维度 | TensorRT-LLM | vLLM | SGLang |
|---|---|---|---|
| 历史基因 | NVIDIA GPU 深度优化 | PagedAttention / 高吞吐 serving | LM Programs / RadixAttention |
| Kernel 优化 | 很强，紧跟 NVIDIA GPU | 多 backend，生态广 | 多 backend，积极集成高性能 kernel |
| KV 管理 | Paged KV / reuse / runtime cache | Paged KV 是奠基能力 | Radix/prefix cache 是鲜明特色 |
| Continuous batching | In-flight batching | 核心能力 | 核心能力 |
| Prefix caching | 支持 | Automatic Prefix Caching | RadixAttention / cache-aware scheduling（缓存感知调度：调度决策会考虑缓存命中和复用价值） |
| P/D disaggregation | 可融入 NVIDIA 分布式栈 | KV connectors / disagg prefill | 原生 PD mode + transfer backends |
| 使用定位 | NVIDIA-first 高性能部署 | 广泛开源 serving / ecosystem | 高性能 serving / agent/prefix-heavy workloads |

注意：

> 这张表是理解“历史侧重点”，不是固定性能排名。

三个项目都高速演化，具体版本、模型、GPU、batch、context、量化、spec decode 配置都可能改变结果。

---

# 11. 为什么现代 Runtime 越来越像操作系统？

一个 LLM engine 现在要管理：

```text
requests
GPU memory
KV pages
prefix cache
CPU/GPU processes
GPU workers
distributed ranks
network transfer
priorities
preemption
resource budgets
```

这和操作系统的经典问题高度相似：

```text
process scheduling
virtual memory
cache
IPC
resource allocation
I/O
```

这也是为什么学习：

```text
OS / distributed systems / computer architecture
```

会极大帮助理解 AI Infra。

---

# 12. 你现在的论文学习地图

推荐按下面顺序：

```text
00 GPU/LLM 硬件基础
        ↓
01 FlashAttention-1
        ↓
04 FlashAttention-2
        ↓
05 FlashAttention-3
        ↓
06 Orca / Continuous Batching
        ↓
02 PagedAttention / vLLM
        ↓
03 SGLang / RadixAttention
        ↓
FlashInfer
        ↓
07 DistServe / P-D Disaggregation
        ↓
08 TensorRT-LLM Runtime
09 vLLM V1 Runtime
10 SGLang Runtime
```

为什么不是按发表年份绝对排序？

因为学习顺序更应该按：

> **依赖关系和概念递进。**

---

# 13. 下一阶段可以继续读什么？

完成这一组以后，建议分成四个方向。

## A. Serving Scheduler

- Sarathi-Serve：chunked prefill / stall-free scheduling
- FastServe：preemptive scheduling
- DeepSpeed-FastGen / SplitFuse 类工作
- modern SLO-aware schedulers

## B. Distributed Inference

- Splitwise
- Mooncake / KVCache-centric architecture
- Dynamo / disaggregated serving
- KV transfer / NIXL / RDMA

## C. Speculative Decoding

- Speculative Decoding 基础论文
- Medusa
- EAGLE / EAGLE-2 / EAGLE-3
- MTP

## D. MoE Inference

- Expert Parallelism
- All-to-All
- DeepSeek-style MoE serving
- EPLB / expert load balancing
- wide-EP

---

# 14. 最值得形成的系统思维

以后看到一个新 AI Infra 工作，不要先问：

> “它用了什么新名词？”

先问五个问题：

1. **它解决的是哪一层？**
   - Hardware / Kernel / Runtime / Cluster？

2. **上一代系统的瓶颈是什么？**
   - Compute / memory / network / scheduling / fragmentation / tail latency？

3. **它改变了什么 abstraction？**
   - Page？Block Sparse？Iteration？Token Budget？KV Connector？

4. **它把成本转移到哪里？**
   - 少 HBM IO，但多 recompute？
   - 少 interference，但多 KV transfer？
   - 多 cache reuse，但 scheduler 更复杂？

5. **它在哪种 workload 下才成立？**
   - Long context？High concurrency？Prefix-heavy？MoE？H100？

如果能沿着这五个问题读论文，你就不会只停留在“背优化点”。

---

# 15. 最终浓缩成 12 句话

1. **FlashAttention-1：少搬 HBM。**
2. **FlashAttention-2：更合理地分 GPU 工作。**
3. **FlashAttention-3：让 Hopper 的多个硬件流水线并行工作。**
4. **Orca：batch 每一轮都可以重新组成。**
5. **Continuous batching：请求可以动态进入和退出 batch。**
6. **PagedAttention：KV Cache 像虚拟内存一样分页。**
7. **RadixAttention：共享 prefix 的请求可以共享 KV。**
8. **FlashInfer：把复杂 serving attention 统一成高性能执行层。**
9. **Chunked Prefill：不要让一个长 prompt 长时间卡住 decode。**
10. **DistServe：干脆把 Prefill 和 Decode 放到不同 GPU pool。**
11. **vLLM / SGLang / TensorRT-LLM：都是现代完整 LLM runtime，而不是单个算法。**
12. **现代推理系统的核心，已经是 Compute + Memory + Scheduling + Network 的联合优化。**

---

## 参考资料范围

本总览综合本文件夹中各论文笔记所引用的一手论文和官方项目文档；具体算法与实验数字请回到对应单篇文档阅读。
