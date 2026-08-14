# 03｜SGLang / RadixAttention 论文详解：从 Paged KV 到“可自动复用的前缀树缓存”

> **标题缩写与首次术语说明**：SGLang 是项目名，可理解为“面向结构化生成与高性能 LLM serving 的框架”；LLM = **Large Language Model（大语言模型）**；LM = **Language Model（语言模型）**；KV Cache = **Key-Value Cache（键值缓存）**；RAG = **Retrieval-Augmented Generation（检索增强生成）**；JSON = **JavaScript Object Notation（常用结构化数据格式）**；LRU = **Least Recently Used（最近最少使用缓存淘汰策略）**；DFS = **Depth-First Search（深度优先搜索）**；FCFS = **First-Come, First-Served（先到先服务）**；GQA = **Grouped-Query Attention（分组查询注意力）**；MQA = **Multi-Query Attention（多查询注意力）**。本文中的 **runtime** 指运行时系统，**prefix cache** 指复用相同输入前缀已计算出的状态，**cache-aware scheduling（缓存感知调度）** 指调度决策会显式考虑缓存命中与复用价值。 另外：GPU = **Graphics Processing Unit（图形处理器）**；CPU = **Central Processing Unit（中央处理器）**；API = **Application Programming Interface（应用程序编程接口）**；HBM = **High Bandwidth Memory（高带宽内存）**；I/O = **Input/Output（输入/输出）**；CUDA = **Compute Unified Device Architecture（NVIDIA GPU 并行计算平台与编程模型）**；AI = **Artificial Intelligence（人工智能）**；OS = **Operating System（操作系统）**；JIT = **Just-In-Time（即时编译）**；SOSP = **ACM Symposium on Operating Systems Principles（ACM 操作系统原理大会）**。 会议缩写：NeurIPS = **Conference on Neural Information Processing Systems（神经信息处理系统大会）**。

> 论文：**SGLang: Efficient Execution of Structured Language Model Programs**
>
> 作者：Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng
>
> arXiv:2312.07104；后发表于 NeurIPS 2024
>
> **前置阅读**：建议先读 `00_共享基础_GPU与LLM推理硬件基础.md` 和 `02_PagedAttention-vLLM论文详解.md`。本文默认你已经理解 KV Cache、continuous batching 和 paged KV 的基本概念。

---

# 1. 先纠正一个常见理解：SGLang 最初并不只是“一个比 vLLM 快的推理框架”

SGLang 这篇论文想解决的是更大的问题：

> **如何高效编程和执行复杂的 Language Model Programs。**

所谓 LM Program，可以理解成：

```text
调用 LLM
↓
根据结果走控制流
↓
再调用 LLM
↓
并行 fork 多个 generation
↓
调用 tool / API
↓
再生成结构化输出
```

例如：

- Agent；
- multi-turn chat；
- few-shot prompting；
- self-consistency；
- tree-of-thought；
- RAG pipeline；
- JSON constrained decoding。

所以 SGLang 原始设计包含两部分：

```text
Frontend Language
        +
Runtime Backend
```

而 **RadixAttention** 是 runtime 中最重要的优化之一：

> 自动、系统地复用这些多次 LLM 调用之间的 KV Cache。

---

# 2. 为什么 2023/2024 年“多次调用 LLM”成为新问题？

早期 serving 思维比较像：

```text
一个 request
→ 一个 prompt
→ 生成一个 answer
→ 结束
```

但 agent / reasoning workload 变成：

```text
System Prompt
   ↓
Generate plan
   ↓
Tool result
   ↓
Generate next step
   ↓
Fork 5 candidates
   ↓
Judge candidates
```

每次 generation 的 prompt 往往包含前一次的大段内容。

于是一个显著问题出现：

> **大量 prompt prefix 被一遍又一遍重新 prefill。**

而 prefill 的结果中，最重要的可复用状态就是：

\[
\boxed{KV\ Cache}
\]

---

# 3. 为什么相同 Prefix 可以直接复用 KV？

Transformer 是 causal 的。

对于 token \(i\) 的 K/V：

> 它只依赖从 token 0 到 token \(i\) 的 prefix。

假设：

```text
Request A:
[A B C D X]

Request B:
[A B C D Y]
```

对于前四个 token：

```text
A B C D
```

两条请求的 forward computation 完全相同。

因此对应 KV：

```text
KV(A B C D)
```

只需要计算一次。

Request B 到来时可以：

```text
reuse KV(A B C D)
↓
只 prefill Y
```

这就是 prefix caching 的基本理论基础。

---

# 4. 为什么简单 Prefix Cache 不够？

如果只有固定 system prompt：

```text
System Prompt → cached KV
```

一个哈希表就够了。

但真实 LM Programs 的共享关系可能是：

```text
                    System Prompt
                    /           \
              Chat A             Chat B
               /  \                |
          Turn A1 Turn A2         Turn B1
             |
        Tool Result
             |
      Candidate Fork
        /    |    \
       C1    C2    C3
```

新请求可能只匹配：

- system prompt；
- system prompt + chat turn 1；
- system prompt + few-shot examples；
- 某条 generation 结果。

所以缓存不是一个：

```text
key → value
```

而天然是：

> **树状 prefix sharing。**

---

# 5. Trie 是什么？

Trie = Prefix Tree。

假设有字符串：

```text
apple
app
apply
```

普通 trie：

```text
a
└── p
    └── p
        ├── l
        │   ├── e
        │   └── y
        └── end
```

它特别适合：

> 查询“最长公共前缀”。

但是每个节点一个 token / character 时，节点数量很多。

---

# 6. Radix Tree 又是什么？

Radix Tree 可以看作 compressed trie。

如果某一段路径没有分叉：

普通 Trie：

```text
A → B → C → D
```

Radix Tree 可以压缩成：

```text
[A B C D]
```

也就是说：

> **edge label 可以是一串 token，而不只是一个 token。**

这非常适合 LLM prefix，因为很多连续 token 没有必要各建一个独立高层树节点。

---

# 7. RadixAttention 的核心数据结构

SGLang 用 radix tree 保存：

```text
Token Prefix
     ↕
对应的 KV Cache Tensor
```

可以想成：

```text
root
 │
 ├── [You are a helpful assistant...]
 │         │
 │         ├── [User: Hello / Assistant: Hi]
 │         │       ├── [User: Solve this...]
 │         │       └── [User: Write a story...]
 │         │
 │         └── [User: What can you do?]
 │
 └── [Few-shot examples...]
           ├── [Question 1]
           ├── [Question 2]
           └── [Question 3]
```

每条 edge：

```text
[token sequence]
+
对应 KV cache
```

新 request 进入时：

```text
request tokens
    ↓
在 radix tree 做 longest prefix match
    ↓
命中的 prefix KV 直接复用
    ↓
只计算剩余 suffix
```

这就是 RadixAttention 最核心的执行流程。

---

# 8. 为什么叫 Radix“Attention”？它本身是新的 Attention kernel 吗？

不是。

这个名字容易让人误会。

RadixAttention 核心并不是改：

\[
softmax(QK^T)V
\]

而是：

> **管理 Attention 所依赖的 KV Cache，并利用 prefix dependency 自动复用已经算过的 KV。**

可以把它理解为：

```text
Radix Tree
= KV Cache index + cache manager

Attention Kernel
= 真正做 Q × K/V 计算
```

SGLang 论文中 RadixAttention 与 continuous batching、paged attention、tensor parallelism 都是兼容的。

所以它处在比 CUDA kernel 更高的一层。

---

# 9. 一个 Chat Session 是怎样进入 Radix Tree 的？

假设第一次请求：

```text
System: You are a helpful assistant.
User: Hello!
Assistant: Hi!
```

第一次没有缓存，全部 prefill/generate。

执行结束后，这段 token 和 KV 被保存在 tree 中：

```text
root
 └── [System + User Hello + Assistant Hi]
```

第二轮：

```text
System: ...
User: Hello!
Assistant: Hi!
User: Solve this problem...
```

runtime 做 prefix match：

```text
旧部分全部命中
```

因此：

```text
旧 KV reuse
+
只计算 Solve this problem...
```

然后把新 suffix 接到树后面。

---

# 10. 第二个 Chat 为什么会导致“节点 Split”？

假设 tree 当前只有：

```text
[System Prompt + Chat A]
```

突然出现 Chat B：

```text
System Prompt + Chat B
```

两者只共享 System Prompt。

Radix Tree 会把旧 edge 拆开：

```text
原来：
root
 └── [System Prompt + Chat A]

变成：
root
 └── [System Prompt]
       ├── [Chat A]
       └── [Chat B]
```

这个 split 操作非常关键，因为它会把隐藏在长 edge 内部的公共 prefix 显式暴露出来，从而让多个 request 共享 KV。

---

# 11. Few-shot Prompt 为什么特别适合 RadixAttention？

例如你有：

```text
Example 1
Example 2
Example 3
Example 4

Question A
```

然后：

```text
Example 1
Example 2
Example 3
Example 4

Question B
```

前面 examples 很长，但完全一样。

传统 serving：

```text
Question A → examples 全算一次
Question B → examples 再算一次
Question C → examples 再算一次
```

RadixAttention：

```text
[Examples] KV
     ↓ shared
 ┌───┼────┐
 Q A Q B  Q C
```

few-shot 越长、查询越多，prefix reuse 的价值越大。

---

# 12. Self-Consistency / Parallel Generation 又为什么适合？

Self-consistency 常见形式：

```text
同一个 problem
↓
采样多个 reasoning paths
```

例如：

```text
Problem prefix
  ├── Sample 1
  ├── Sample 2
  ├── Sample 3
  └── Sample 4
```

公共 prefix KV 只需要一份。

这和 vLLM 的 parallel sampling sharing 很像，但 RadixAttention 的视角更一般：

> 这些 branch 可以来自 LM program 中任意动态 fork，而不是只有预先定义好的 sampling case。

---

# 13. “请求结束了，为什么 KV 不马上 free？”

这是 RadixAttention 和传统 request-scoped KV management 的一个重大区别。

传统方式：

```text
request finished
↓
free KV cache
```

RadixAttention：

```text
request finished
↓
KV 先留在 radix tree
↓
未来 request 如果共享 prefix
→ cache hit
```

也就是说 KV Cache 从：

> **request-local temporary state**

变成：

> **server-level reusable cache。**

这是一个非常重要的概念升级。

---

# 14. 但 GPU Memory 有限：缓存不能永远留着

假设 GPU KV pool 满了。

必须决定：

> 哪些 cached prefix 应该删除？

SGLang 使用：

\[
\boxed{LRU}
\]

Least Recently Used。

直觉：

> 最久没被使用的缓存，未来再次用到的概率可能更低，优先淘汰。

但树结构里不能随便删内部节点。

例如：

```text
A
└── B
    └── C
```

如果 C 仍需要 A/B，那么直接删 B 会破坏整条 prefix。

所以论文采用：

> **LRU leaf-first eviction。**

先淘汰最久没使用的叶节点。

只有当祖先节点不再被任何 child 依赖、自己成为 leaf 后，才继续成为淘汰候选。

这样能最大限度保留 common prefix。

---

# 15. Reference Counter 是做什么的？

continuous batching 下，有些 radix tree nodes 正在被当前 running requests 使用。

显然不能把它们淘汰。

所以每个节点维护：

```text
ref_count
```

表示：

> 当前有多少 running requests 正在引用这段 KV。

只有：

\[
ref\_count=0
\]

才可以被 eviction。

这和很多系统中的资源生命周期管理非常类似。

---

# 16. Cache 和 Running Requests 为什么共用同一个 Memory Pool？

一种简单设计可以是：

```text
20GB 专门给 active requests
20GB 专门给 prefix cache
```

问题：比例很难提前决定。

高并发时：

```text
active request 需要更多 memory
```

低并发但重复 prompt 多时：

```text
cache 越大越有价值
```

RadixAttention 选择动态共享：

```text
GPU KV Memory Pool
├── currently running KV
└── cached reusable KV
```

当 waiting requests 很多、需要扩大 batch：

```text
逐步 evict cached leaves
↓
把空间让给 active requests
```

所以系统会动态 trade-off：

\[
cache\ hit\ rate
\quad vs \quad
batch\ size
\]

这是非常漂亮的 runtime policy。

---

# 17. Prefix Cache 只有数据结构还不够：Scheduler 顺序也会影响命中率

假设 waiting queue 中有：

```text
A1: prefix A
B1: prefix B
A2: prefix A
B2: prefix B
A3: prefix A
```

如果 FCFS：

```text
A1 → B1 → A2 → B2 → A3
```

在显存紧张时可能不断发生：

```text
A 被缓存
↓
切到 B
↓
A 被 eviction
↓
又回来 A
```

这叫：

> **Cache Thrashing。**

因此 SGLang 不只做 cache，还做：

> **Cache-aware Scheduling。**

---

# 18. Longest-Shared-Prefix-First 是什么意思？

对于 waiting requests，先计算它们与当前 cache 的 matched prefix length。

例如：

```text
Request A: hit 3000 tokens
Request B: hit 100 tokens
Request C: hit 2500 tokens
```

优先：

```text
A → C → B
```

而不是单纯 FCFS。

直觉是：

> 趁某条长 prefix 还在 cache 里，连续处理它附近的 requests。

这样就像访问一棵树时尽量：

```text
深入一个 subtree
↓
把这一支做完
↓
再换另一支
```

因此它与：

> **DFS（Depth-First Search，深度优先搜索）**

密切相关。

---

# 19. 为什么论文会证明 DFS 可以达到最优 Cache Hit Rate？

在论文给定的 offline 条件下，如果 cache capacity 至少能容纳最大 request length：

> 按 radix tree 的 DFS 顺序访问请求，可以让每条 edge 对应的 KV 至少只计算一次。

直觉：

```text
进入 subtree A
↓
A 的公共 prefix 一直是最近使用
↓
处理完 A 的所有 descendants
↓
再离开 A
```

不会：

```text
A → B → A → B
```

来回把缓存冲掉。

论文指出 longest-shared-prefix-first 与这种 DFS 行为相对应。

这是 RadixAttention 很重要的一点：

> **Cache policy 和 request scheduler 必须 co-design。**

---

# 20. 这里为什么会出现公平性问题？

如果永远优先 cache hit 最大的 request：

```text
新来的热门 prefix requests
不断插队
```

某些 cache hit 很低的 request 可能很久得不到执行。

这就是：

> **Starvation。**

论文也承认 cache-aware scheduling 需要在：

```text
throughput / cache hit
        vs
latency / fairness
```

之间权衡。

这是生产 serving 系统里非常真实的问题。

---

# 21. Frontend Hint 为什么是一个有趣的系统设计？

SGLang 不只是 backend runtime，它还有 frontend language。

例如 `fork` primitive 表明：

> 接下来会有几个 generation 共享同一个 prefix。

如果 runtime 完全不知道程序语义，只能等完整 prompts 到来后自己猜 sharing relation。

SGLang frontend 可以提前给 runtime hint：

```text
这个 prefix 接下来会 fork
```

runtime 就可以更容易：

- 提前插入 tree；
- 做 prefix matching；
- 安排 scheduling。

这体现一个系统思想：

> **Application semantics 可以帮助 runtime 做更聪明的优化。**

即所谓 frontend-runtime co-design。

---

# 22. Radix Tree 本身存在哪？KV 又存在哪？

论文中 tree metadata 放在 CPU，维护开销很小。

真正昂贵的：

```text
KV Cache tensors
```

仍然存 GPU memory。

可以粗略理解为：

```text
CPU:
Radix tree nodes
prefix metadata
pointers / references

GPU HBM:
actual K/V tensors
```

CPU tree 决定：

```text
这次 request 命中了哪些 KV
```

GPU 再使用这些 KV 进行 attention/prefill/decode。

---

# 23. RadixAttention 与 PagedAttention 是竞争关系吗？

不是。

这是读这两篇论文最重要的层次区分。

## PagedAttention

```text
问题：KV 在 memory 中怎么分配？

logical blocks
    ↓
physical blocks
```

## RadixAttention

```text
问题：哪些 token prefix 对应的 KV 值得被复用？

request tokens
    ↓
radix prefix tree
    ↓
matched cached KV
```

因此可以组合：

```text
Radix Tree
决定复用哪些 KV
    ↓
Paged KV storage
决定这些 KV physical blocks 在哪里
```

SGLang 原论文明确说明 RadixAttention 与 paged attention 是兼容的。

---

# 24. RadixAttention 与传统 Prefix Cache 的真正区别

简单 prefix cache：

```text
完整 prefix hash
→ cached KV
```

RadixAttention：

```text
compressed prefix tree
+
longest-prefix matching
+
node split / insertion
+
LRU leaf eviction
+
reference counting
+
cache-aware scheduling
+
frontend hints
```

因此它不是“用了一个 radix tree”这么简单。

真正创新是把：

\[
\boxed{data\ structure + cache\ policy + scheduling + LM\ program\ semantics}
\]

组合成完整 runtime optimization。

---

# 25. SGLang 论文里还有哪些东西，但和 RadixAttention 不是一回事？

原论文整体还有：

- frontend language primitives；
- parallelism / `fork`；
- compressed finite state machine，用于更快 structured output decoding；
- API speculative execution；
- frontend/runtime co-design。

论文报告 SGLang 在多种 LM program workloads 上相较当时 baseline **最高可达到 6.4× throughput**。

但要注意：

> **这个 6.4× 是 SGLang 整个系统在特定 workload 上的端到端结果，不能简单说成“RadixAttention 本身固定加速 6.4×”。**

这是读系统论文实验时很重要的严谨性。

---

# 26. RadixAttention 在没有 Cache Hit 时会不会很亏？

这是一个关键问题。

如果每次 request 都完全不同：

```text
prefix hit = 0
```

那么 radix tree lookup 和 maintenance 会不会让系统更慢？

SGLang 论文专门测试了没有 KV reuse opportunity 的情况，并报告 tree management overhead 很小。

这很重要，因为一个通用 serving optimization 必须做到：

> **命中时赚很多，不命中时尽量别亏。**

---

# 27. RadixAttention 的局限

## 1. 主要依赖 exact prefix sharing

如果两段文字语义相同但 token 不同：

```text
What is AI?
Explain artificial intelligence.
```

Radix Tree 不会因为“语义相似”就复用。

## 2. Cache hit 强依赖 workload

如果所有请求完全随机且没有公共 prefix，收益有限。

## 3. Cache-aware scheduling 与 latency/fairness 冲突

为了命中率 reorder 请求，可能增加某些请求等待时间。

## 4. GPU memory 仍然有限

RadixAttention 优化“哪些 KV 留下来”，但并不会让 KV Cache 本身消失。

## 5. Tree/cache policy 是更高层优化

真正 attention kernel 的效率仍然依赖底层实现。

这正是后来 FlashInfer 等 kernel/runtime 工作继续优化的空间。

---

# 28. RadixAttention → FlashInfer：为什么技术又继续演化？

经过 PagedAttention 和 RadixAttention 后，底层 Attention kernel 看到的 KV 已经越来越复杂。

过去：

```text
K/V = 一个连续 tensor
```

后来：

```text
PagedAttention:
K/V = 多个 non-contiguous blocks
```

再后来：

```text
RadixAttention:
多个 queries 共享不同层次的 prefix blocks
```

再加上：

- sliding window；
- tree attention；
- speculative decoding；
- GQA / MQA；
- parallel generation；

Attention 的 data access pattern 开始高度 irregular。

于是 FlashInfer 进一步提出：

> 能不能把 Page Table、Radix Tree、Sparse Mask 等统一成某种 block-sparse representation，再针对不同 workload 动态选择 kernel 和调度？

所以完整技术线是：

```text
FlashAttention
优化 Attention IO
      ↓
PagedAttention
虚拟化 KV physical memory
      ↓
RadixAttention
自动管理跨请求 KV prefix reuse
      ↓
FlashInfer
统一 heterogeneous / irregular inference Attention execution
```

---

# 29. 一个例子把 PagedAttention、RadixAttention、FlashInfer 三层串起来

假设服务器收到：

```text
Request A:
System Prompt + Conversation A + Question A

Request B:
System Prompt + Conversation A + Question B
```

## RadixAttention 层

发现：

```text
System Prompt + Conversation A
```

是公共 prefix。

于是决定：

> 这部分 KV 复用。

## Paged KV 层

公共 KV 可能实际存：

```text
Physical Block 3
Physical Block 19
Physical Block 6
```

A/B 的 logical prefix 都指向这些 blocks。

## FlashInfer / kernel 层

收到 query 与 KV block mapping 后：

> 以适合当前 query length、shared prefix pattern 和 GPU 的 kernel/tile/scheduler 执行 Attention。

因此三者分别回答：

```text
RadixAttention：用哪些 KV？
PagedAttention：KV 在哪里？
FlashInfer：怎样把这些 KV 高效算掉？
```

这是非常值得长期记住的一组层次关系。

---

# 30. 最值得记住的 9 个 Insight

1. **KV Cache 可以从 request-local 临时状态升级为 server-level reusable cache。**
2. **相同 token prefix 对应相同的历史 K/V，因此可以跳过重复 prefill。**
3. **真实 LM Program 的 prefix sharing 往往是树状，而不是一个固定 system prompt。**
4. **Radix Tree 是 compressed prefix tree，非常适合表达 multi-level KV sharing。**
5. **RadixAttention 不只是一个数据结构，还包含 LRU eviction 和 cache-aware scheduling。**
6. **Leaf-first LRU 是为了尽量保留被多个 descendants 共享的 ancestor prefix。**
7. **Scheduler 的 request order 会改变 cache hit rate，所以 scheduling 和 caching 需要 co-design。**
8. **PagedAttention 与 RadixAttention 不冲突：前者是 memory virtualization，后者是 prefix reuse policy/data structure。**
9. **FlashInfer 又进一步把这些复杂 KV access patterns 下沉为统一的 kernel/runtime abstraction。**

---

# 31. 到这里，你已经可以把三篇论文串成一条线

```text
【问题 1】
标准 Attention 大量读写 N×N 中间矩阵
          ↓
FlashAttention
          ↓
用 IO-aware tiling / online softmax
减少 HBM traffic

【问题 2】
在线 Decode 出现巨大的动态 KV Cache
          ↓
PagedAttention / vLLM
          ↓
用 OS paging 思想
logical KV → physical blocks

【问题 3】
大量请求/agent calls 重复相同 prefix
          ↓
RadixAttention / SGLang
          ↓
用 radix tree 自动复用 KV
并做 cache-aware scheduling

【问题 4】
Paged / Radix / Sparse / Tree 等 KV layout
让 Attention kernel 越来越不规则
          ↓
FlashInfer
          ↓
统一 block-sparse format + JIT + runtime scheduler
```

如果把这条线真正理解，你已经不是在记四个项目名，而是在理解：

> **LLM Infra 是如何随着 workload 变化，一层一层把瓶颈从计算、搬运、显存管理、缓存复用，最终推进到统一 runtime abstraction 的。**

---

## 主要参考资料

- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, arXiv:2312.07104, NeurIPS 2024.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023, arXiv:2309.06180.
