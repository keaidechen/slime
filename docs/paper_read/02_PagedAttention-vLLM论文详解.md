# 02｜PagedAttention / vLLM 论文详解：把 KV Cache 当成操作系统的虚拟内存

> **标题缩写与首次术语说明**：KV Cache = **Key-Value Cache（键值缓存）**；LLM = **Large Language Model（大语言模型）**；SOSP = **ACM Symposium on Operating Systems Principles（ACM 操作系统原理大会）**；HBM = **High Bandwidth Memory（高带宽内存，GPU 主显存）**；OS = **Operating System（操作系统）**；FCFS = **First-Come, First-Served（先到先服务）**；RAM = **Random-Access Memory（随机存取存储器）**。vLLM 是项目名，本文把它理解为“高吞吐 LLM 推理/服务框架”，不对项目名本身生造字母展开。**Virtual Memory（虚拟内存）**用逻辑地址抽象物理内存，**page（页）**是分页管理的基本单位，**fragmentation（内存碎片）**是分配方式造成的不可有效利用空间，**Copy-on-Write（写时复制）**是在真正修改共享数据时才复制，**preemption（抢占）**是在资源不足时暂停或移出部分请求。 另外：GPU = **Graphics Processing Unit（图形处理器）**；CPU = **Central Processing Unit（中央处理器）**；I/O = **Input/Output（输入/输出）**；CUDA = **Compute Unified Device Architecture（NVIDIA GPU 并行计算平台与编程模型）**。

> 论文：**Efficient Memory Management for Large Language Model Serving with PagedAttention**
>
> 作者：Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
>
> 发表：SOSP 2023，arXiv:2309.06180
>
> **前置阅读**：`00_共享基础_GPU与LLM推理硬件基础.md`。本文默认你已经知道 HBM、KV Cache、prefill/decode、page、internal/external fragmentation 的基本含义。

---

# 1. 一句话理解这篇论文

PagedAttention 的核心不是发明了一种新的 Attention 数学公式，而是：

> **把 LLM 推理中的 KV Cache 从“每个请求一大块连续显存”改造成类似操作系统 Virtual Memory 的 paged memory。**

于是：

```text
逻辑上的连续 token
        ↓
Logical KV Blocks
        ↓ Block Table
分散在 GPU HBM 中的
Physical KV Blocks
```

在这个机制之上，作者构建了：

\[
\boxed{vLLM}
\]

目标是：

> **降低 KV Cache 显存浪费 → 放进更大的 batch → 提高 LLM serving throughput。**

---

# 2. 为什么 FlashAttention 之后仍然需要 PagedAttention？

FlashAttention 解决：

> 一个 Attention kernel 怎样减少 HBM IO。

但 LLM 在线推理还有一个训练阶段没那么突出的系统问题：

> **KV Cache 是动态增长的。**

例如服务器正在处理：

```text
Request A: prompt 500, 已生成 20
Request B: prompt 10000, 已生成 300
Request C: prompt 1800, 已生成 1
Request D: finished
Request E: 刚进入队列
```

每个请求：

- prompt 长度不同；
- output 长度不同；
- output 长度事先不知道；
- 每生成一个 token，KV Cache 都要继续增长。

这已经很像操作系统中：

> 很多动态进程同时申请和释放内存。

因此作者把问题从 Attention kernel 上升到了：

\[
\boxed{LLM\ Serving\ Memory\ Management}
\]

---

# 3. 为什么 KV Cache 对 throughput 影响这么大？

LLM serving 想提高吞吐量，一个关键方式是：

> **把更多 requests 一起 batch 到 GPU。**

假设 GPU 显存剩余 40GB 可用于 KV Cache。

如果平均每个活跃请求需要 2GB：

```text
理论最多 ~20 requests
```

如果内存管理额外浪费一半：

```text
实际可能只能放 ~10 requests
```

batch 直接变小。

因此：

```text
KV Cache 显存利用率
      ↓
能同时驻留多少请求
      ↓
batch size
      ↓
GPU utilization / throughput
```

这就是为什么一个“内存管理”论文最终能带来数倍 serving throughput 改善。

---

# 4. 先理解 LLM serving 的 iteration-level scheduling

自回归生成不是一次把整段 output 算出来，而是：

```text
Iteration 1: 每个 request 生成 1 token
Iteration 2: 每个 request 再生成 1 token
Iteration 3: ...
```

早期简单 batching 可能要求：

> 一个 batch 中所有请求一起开始、一起结束。

如果长度差别很大：

```text
A: ███ done
B: ███████ done
C: ███████████████████████
```

A/B 完成后 GPU batch slot 还可能被浪费。

后来 Orca 等系统推动 **iteration-level scheduling / continuous batching**：

```text
每个 decoding iteration
都允许 finished requests 离开
新 requests 加入
```

这显著提高 GPU 利用率。

但 continuous batching 又让 KV Cache memory management 更动态。

也就是说：

> **更好的 scheduler 暴露了更严重的动态内存问题。**

这就是系统技术演化中常见的“解决一个瓶颈，又暴露下一个瓶颈”。

---

# 5. 旧式 KV Cache 管理为什么浪费？

论文把浪费主要归纳成三类。

## 5.1 Reservation

因为 output 长度未知，系统可能预先为未来 token 保留空间：

```text
当前 KV
█████

为未来预留
.................
```

如果实际很快输出 `<eos>`：

> 大量预留显存从未使用。

## 5.2 Internal Fragmentation

为了方便管理，系统可能按某种较大固定区域分配。

请求只用一部分：

```text
[used][unused inside allocated region]
```

未使用空间属于内部碎片。

## 5.3 External Fragmentation

随着请求不断结束、加入：

```text
used | hole | used | hole | used
```

虽然总 free memory 可能很大，却没有一块足够大的连续区域给新请求。

论文实验中显示，旧方案的 KV cache memory waste 可以非常显著，而 vLLM 的 block-based 设计能让绝大部分显存真正用于 token states。

---

# 6. 为什么不能简单做 Memory Compaction？

传统 heap 出现外部碎片时，可以想：

> 把存活对象都移动到一起，把洞合并。

但 KV Cache 很大。

如果频繁做：

```text
copy several GB KV
```

会产生巨大 memory bandwidth 开销。

而 serving 是 latency-sensitive 的。

所以论文选择类似 OS 的另一条路：

> **不要强迫物理内存连续。**

---

# 7. PagedAttention 最核心的 OS 类比

操作系统：

```text
Process virtual memory
        ↓ page table
Physical pages
```

vLLM：

```text
Request logical KV sequence
        ↓ block table
Physical KV blocks in GPU HBM
```

映射关系可能是：

```text
Logical Block 0 → Physical Block 7
Logical Block 1 → Physical Block 1
Logical Block 2 → Physical Block 3
```

所以从请求角度：

```text
token 0 ... token N
```

依然是连续序列。

但 GPU HBM 中可能：

```text
Block 7
Block 1
Block 3
```

完全不连续。

最重要的抽象是：

\[
\boxed{Logical\ continuity\neq Physical\ continuity}
\]

---

# 8. Logical KV Block 与 Physical KV Block

假设 block size = 4 tokens。

一个 prompt：

```text
Four score and seven years ago our
```

有 7 tokens。

逻辑上：

```text
Logical Block 0:
[t0 t1 t2 t3]

Logical Block 1:
[t4 t5 t6 _]
```

系统只需要两个物理 block：

```text
Logical 0 → Physical 7
Logical 1 → Physical 1
```

当下一个 token 生成：

```text
Logical Block 1:
[t4 t5 t6 t7]
```

只填最后一个 slot。

再生成 token：

```text
Logical Block 2 需要出现
```

这时才申请一个新的 physical block：

```text
Logical 2 → Physical 3
```

所以：

> **KV Cache 按需要增长，而不是一开始预留最大 sequence length。**

---

# 9. Block Table 是什么？

Block Table 就是一个请求的地址翻译表。

例如：

| Logical Block | Physical Block | # Filled |
|---|---:|---:|
| 0 | 7 | 4 |
| 1 | 1 | 4 |
| 2 | 3 | 1 |

Attention kernel 看到 query 后，需要知道历史 K/V 在哪里。

以前可能默认：

```text
KV 是一段连续地址
```

PagedAttention 以后：

```text
先看 block table
↓
找到 physical block 7
读取
↓
找到 physical block 1
读取
↓
找到 physical block 3
读取
```

因此 Attention kernel 必须能够：

> **直接读取 non-contiguous KV blocks。**

这就是名字中 Paged**Attention** 的由来：

它不只是 allocator，还必须有一个与 paged layout 配套的 attention kernel。

---

# 10. PagedAttention kernel 在数学上变了吗？

没有改变 Attention 语义。

原本：

\[
Attention(q,K,V)
\]

其中 K/V 在连续 tensor 中。

现在只是在执行时：

```text
K1,V1 来自 Physical Block 7
K2,V2 来自 Physical Block 1
K3,V3 来自 Physical Block 3
```

kernel 分 block fetch K/V，再计算 attention contribution。

所以真正变化是：

> **memory addressing / data layout。**

---

# 11. 为什么固定大小 block 可以减少碎片？

假设所有 physical blocks 大小一致。

结束一个请求后释放：

```text
Block 7
Block 1
Block 3
```

这些 block 可以被任何其他请求重新使用。

不会出现：

```text
这里有 100MB hole
但新 request 需要连续 500MB
```

因为新请求并不要求 500MB 连续。

它可以拿：

```text
Block 9 + Block 42 + Block 3 + ...
```

因此：

> **外部碎片基本被 page/block abstraction 消掉。**

内部碎片也被限制在：

> 每个 sequence 最后一个 block 的未使用 slot。

block size 越小，最大 internal fragmentation 越小；但 block 太小又会增加 metadata / indexing overhead。

这就是 block size 的典型 trade-off。

---

# 12. PagedAttention 不只是节省空间，还带来了“共享”

一旦逻辑地址和物理 block 解耦，就可以：

```text
Request A logical block 0 ─┐
                           ├→ Physical Block 7
Request B logical block 0 ─┘
```

两个逻辑序列指向同一物理 KV。

这让 vLLM 很自然地支持：

- parallel sampling；
- beam search；
- shared prefix。

这与 OS 中多个进程映射同一物理 page 的思路非常相似。

---

# 13. Parallel Sampling：为什么 Copy-on-Write 非常自然？

假设对同一个 prompt 采样两个答案：

```text
Prompt
  ├── Sample A
  └── Sample B
```

prompt 对应的所有 KV 完全相同。

因此：

```text
A logical prompt blocks ─┐
                         ├→ shared physical blocks
B logical prompt blocks ─┘
```

到真正生成不同 token 时：

```text
A wants to modify last shared block
B may also modify it
```

vLLM 使用：

> **Copy-on-Write at block granularity。**

也就是说直到“写”发生之前都共享。

一旦某个 branch 要修改共享 block：

```text
copy that block
↓
A 写自己的 physical block
B 继续保留原 block
```

这是操作系统 `fork()` 内存优化思想在 LLM serving 中非常漂亮的一次迁移。

---

# 14. Beam Search 为什么收益可能更大？

Beam Search 会保留多个候选序列。

很多 beam candidate 在相当长时间里共享同一个历史 prefix。

传统实现如果不停复制整份 KV：

> 内存和 bandwidth 都很浪费。

PagedAttention 可以让候选之间共享绝大多数历史 blocks，只对发生分叉/写入的部分 Copy-on-Write。

论文实验里，beam search 场景因为可共享部分更多，vLLM 相对于旧 baseline 的收益更加明显。

---

# 15. Shared Prefix：PagedAttention 已经能做 Prefix Cache 了吗？

论文确实展示了一种 shared prefix 场景。

例如服务商预先知道很多用户都会带：

```text
Instruction
+ few-shot examples
```

可以提前保存这段 prefix 的 physical KV blocks。

不同请求：

```text
Request A logical prefix ─┐
Request B logical prefix ─┼→ precomputed physical KV
Request C logical prefix ─┘
```

然后只对各自 task-specific suffix 做 prefill。

但是注意：

> PagedAttention 论文主要解决的是 block-level memory virtualization 和若干可预期的 sharing pattern。

后来 SGLang 的 RadixAttention 继续问：

> 如果 prefix sharing 是动态、多层、树状，而且每次都不知道下一条请求会和谁共享，怎么办？

这就是下一篇论文的核心。

---

# 16. Scheduler 为什么也必须和内存管理一起设计？

假设 GPU physical KV blocks 用完了。

此时系统不只是 allocator 问题，还必须决定：

> 哪些请求继续跑？哪些请求暂时停？

论文 vLLM 使用 FCFS 作为主要公平策略，并设计 preemption。

一个重要观察是：

> 同一个 sequence 的 KV blocks 通常会一起被访问。

所以 eviction/preemption 可以按整个 sequence / sequence group 来处理，而不是随便逐 block 驱逐。

对于同一个 request 中的多个 beam candidates，还需要 gang-scheduling，因为它们可能共享 physical blocks。

---

# 17. 被 Preempt 的 KV 怎么恢复？

论文讨论两种经典选择。

## Swapping

类似 OS swap：

```text
GPU HBM KV
 ↓
copy to CPU RAM
```

以后继续执行：

```text
CPU RAM
 ↓
copy back GPU
```

优点：

> 不需要重新算 KV。

缺点：

> CPU↔GPU 数据传输有成本。

## Recomputation

直接丢掉一部分 KV。

以后需要时：

```text
从 token 重新做 forward
→ 重建 KV
```

本质 trade-off：

```text
communication / memory copy
vs
GPU recomputation
```

这个“重算 vs 搬运”的思想与你在 FlashAttention backward 中看到的 trade-off 是同一个系统哲学。

---

# 18. vLLM 的系统结构怎么理解？

论文中的 vLLM 可以粗略拆成：

```text
                Scheduler
                    │
             KV Cache Manager
                    │
               Block Tables
                    │
      ┌─────────────┼─────────────┐
      ↓             ↓             ↓
   Worker 0      Worker 1       Worker N
      │             │             │
 Model shard     Model shard    Model shard
      │             │             │
 PagedAttention / GPU kernels
```

Scheduler 每个 iteration 决定：

- 哪些 requests 进入 batch；
- 哪些新 logical blocks 需要分配 physical blocks；
- 每个 request 的 block table 是什么。

GPU worker 再根据 block table 读取对应 KV。

可以看到：

> PagedAttention 不是单独一个 CUDA kernel 就构成 vLLM，它需要 scheduler、KV manager、block allocator、kernel 共同 co-design。

---

# 19. 为什么 PagedAttention 和操作系统虚拟内存“像，但不完全一样”？

类比非常有帮助，但不要机械等同。

OS page table 的目标包括：

- address translation；
- process isolation；
- protection；
- virtual address abstraction；
- demand paging。

vLLM block table 的主要目标是：

- 动态 KV allocation；
- non-contiguous storage；
- sharing；
- serving scheduler 配合。

而且 GPU attention kernel 本身要知道如何高效读取这种 block layout。

所以可以认为：

> **PagedAttention 借用了 paging 的抽象思想，但为 LLM KV Cache workload 重新设计。**

---

# 20. 论文最重要的实验结论

论文在多个模型、数据集和 decoding 场景上评估 vLLM，并报告：

> 相比当时的 FasterTransformer / Orca 等 baseline，在相似 latency 条件下，vLLM serving throughput 可提高约 **2–4×**。

收益在以下场景更加明显：

- sequence 更长；
- model 更大；
- decoding method 更复杂；
- KV sharing opportunity 更多。

重要的是不要把“2–4×”理解为今天任何环境中的固定加速比。

实验真正证明的是：

> **KV Cache memory efficiency 可以直接转化成更大的 effective batch 和显著更高 serving throughput。**

---

# 21. PagedAttention 最深层的 Insight 是什么？

很多人第一次看会觉得：

> “不就是把 KV 切成 blocks 吗？”

真正重要的是它引入了一层：

\[
\boxed{Virtualization}
\]

过去：

```text
Sequence
≈
一段 physical contiguous KV memory
```

以后：

```text
Sequence
=
Logical KV address space
        ↓ mapping
Physical KV blocks
```

这层 indirection 一旦建立，就同时解锁：

- 按需增长；
- 减少碎片；
- sharing；
- Copy-on-Write；
- swapping；
- 多种 decoding method 统一映射。

这正是系统设计里“一个好的 abstraction 可以一次解决很多问题”的典型案例。

---

# 22. PagedAttention 和 RadixAttention 最容易混淆的地方

两者都涉及：

> KV Cache。

但关注的问题不同。

## PagedAttention

问：

> **KV Cache 在物理显存里怎么放？**

抽象：

```text
logical blocks → physical blocks
```

主要解决：

- memory fragmentation；
- dynamic allocation；
- block-level sharing。

## RadixAttention

问：

> **来了一个新请求，我应该复用以前哪些 token 的 KV？哪些 KV 应该继续缓存、哪些应该淘汰？**

抽象：

```text
token prefixes → radix tree → KV cache
```

主要解决：

- automatic prefix matching；
- multi-level sharing；
- cache eviction；
- cache-aware scheduling（缓存感知调度：调度决策会考虑缓存命中和复用价值）。

所以完全可以：

```text
RadixAttention
负责“哪些 KV 应该复用”
        ↓
Paged layout
负责“这些 KV 在 GPU memory 哪里”
```

SGLang 论文也明确说明 RadixAttention 与 paged attention 是兼容的。

---

# 23. PagedAttention 与 FlashInfer 的关系

FlashInfer 后来进一步观察：

> Paged KV 本质上让 query 到 KV block 的访问关系变成了不连续、稀疏的结构。

于是它尝试把：

```text
Page Table
Radix Tree
Sparse Mask
...
```

统一转换为更一般的：

> **Block Sparse Attention representation。**

因此历史线可以这样看：

```text
PagedAttention
创造了高效 paged KV storage
        ↓
RadixAttention
在其上形成更复杂的共享结构
        ↓
FlashInfer
尝试给这些不规则 KV layouts 建一个统一 kernel abstraction
```

---

# 24. PagedAttention 的 trade-off / 局限

一个 abstraction 不可能免费。

Paged layout 带来的代价包括：

## 1. 地址 indirection

连续 tensor 可以直接：

```text
base + offset
```

PagedAttention 需要 block table lookup。

## 2. 非连续读取

physical KV 不连续，可能让 memory access pattern 更复杂。

因此需要专门优化的 kernel。

## 3. Block size trade-off

block 小：

- internal fragmentation 小；
- metadata 更多；
- indexing 更复杂。

block 大：

- metadata 少；
- 最后一个 block 浪费更多。

## 4. Scheduler 与 allocator 强耦合

为了真正发挥收益，不能只换一个 kernel，还需要 serving runtime 的 memory manager 配合。

---

# 25. 最值得记住的 8 个 Insight

1. **LLM serving throughput 很大程度受 KV Cache 能容纳多少 concurrent requests 影响。**
2. **Output length 未知使传统连续预分配产生严重 reservation 和 fragmentation。**
3. **PagedAttention 的核心是 logical KV 与 physical KV 解耦。**
4. **逻辑连续不要求物理连续，这是整篇论文最关键的 abstraction。**
5. **固定大小 blocks 能极大缓解 external fragmentation。**
6. **Block mapping 自然支持 KV sharing 和 Copy-on-Write。**
7. **PagedAttention 不是只有一个 Attention kernel，而是 allocator + cache manager + scheduler + kernel 的 co-design。**
8. **这篇论文把操作系统虚拟内存的经典思想成功迁移到了 LLM serving。**

---

# 26. 一句话串到下一篇：为什么还需要 RadixAttention？

PagedAttention 已经让：

```text
A 与 B 可以共享同一个 physical KV block
```

但它还没有系统性回答：

> **面对成千上万个动态请求，系统怎么自动发现“谁和谁共享什么 prefix”？**

真实 LM program 可能出现：

```text
System Prompt
├── Chat Session A
│   ├── Turn 1
│   └── Turn 2
├── Chat Session B
└── Few-shot Examples
    ├── Query 1
    ├── Query 2
    └── Query 3
```

共享关系天然是一棵树。

于是下一步从：

> memory virtualization

演化成：

> **prefix cache data structure + cache policy + scheduler。**

这就是 **SGLang / RadixAttention**。

---

## 主要参考资料

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023, arXiv:2309.06180.
