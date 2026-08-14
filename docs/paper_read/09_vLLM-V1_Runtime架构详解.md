# 09｜vLLM V1 Runtime 架构详解：Scheduler、KV Cache Manager、Engine Core 与 GPU Worker 如何协作？

> **标题缩写与首次术语说明**：vLLM 是高吞吐大语言模型推理/服务框架，项目名不强行字母展开；V1 指 vLLM 的新一代运行时架构；LLM = **Large Language Model（大语言模型）**；KV Cache = **Key-Value Cache（键值缓存）**；GPU = **Graphics Processing Unit（图形处理器）**；API = **Application Programming Interface（应用程序编程接口）**；P/D = **Prefill/Decode（预填充/解码）**；TP = **Tensor Parallelism（张量并行）**；DP = **Data Parallelism（数据并行）**；NCCL = **NVIDIA Collective Communications Library（NVIDIA 集合通信库）**；HBM = **High Bandwidth Memory（高带宽内存）**；GEMM = **General Matrix-Matrix Multiplication（通用矩阵-矩阵乘法）**；MoE = **Mixture of Experts（混合专家模型）**。本文中的 **Engine Core** 是调度和状态管理核心，**worker** 是执行模型计算的工作进程，**Model Runner** 是组织一次模型前向执行的运行组件。 另外：HTTP = **Hypertext Transfer Protocol（超文本传输协议）**；CUDA = **Compute Unified Device Architecture（NVIDIA GPU 并行计算平台与编程模型）**；AI = **Artificial Intelligence（人工智能）**；SOSP = **ACM Symposium on Operating Systems Principles（ACM 操作系统原理大会）**。

> 本文基于截至 2026-08 的 vLLM V1 官方架构文档整理。
>
> 推荐前置：
> - `02_PagedAttention-vLLM论文详解.md`
> - `06_Orca与Continuous-Batching论文详解.md`
> - `07_DistServe与Prefill-Decode分离论文详解.md`
>
> 目标：从“PagedAttention 论文”进一步进入“今天 vLLM 作为完整 runtime 到底怎么跑一个请求”。

---

# 1. 先区分“vLLM 论文时代”和“今天的 vLLM”

最早认识 vLLM 通常是：

```text
vLLM = PagedAttention
```

这是理解历史贡献的好入口，但今天已经远远不完整。

现代 vLLM 是完整 inference engine：

```text
API Server
Scheduler
KV Cache Manager
Model Executor
GPU Workers
Attention Backends
Distributed Parallelism
Prefix Cache
Speculative Decoding
Structured Output
P/D Disaggregation（Prefill/Decode 分离部署）
...
```

所以现在更准确：

> **PagedAttention 是 vLLM 的奠基技术之一，而 vLLM 本身已经是完整 serving runtime。**

---

# 2. V1 的高层进程结构

官方 V1 architecture 可以先简化为：

```text
                 Client
                   │
                   ▼
              API Server
                   │
                   ▼
              Engine Core
        ┌──────────┼──────────┐
        │          │          │
    Scheduler   KV Cache    coordination
        │        Manager       │
        └──────────┼───────────┘
                   │
                   ▼
            GPU Worker(s)
                   │
                   ▼
              Model Runner
                   │
                   ▼
                  GPU
```

其中最重要的两个层：

```text
Engine Core
→ 决策和状态管理

GPU Worker
→ 真正执行 model forward
```

---

# 3. API Server 做什么？

它不是主要 GPU scheduler。

典型职责包括：

```text
HTTP / OpenAI-compatible protocol
请求解析
tokenization / preprocessing
streaming response
与 Engine Core 通信
```

可以把它理解成：

> **服务入口层。**

而真正每轮决定“谁上 GPU”的逻辑主要在 Engine Core。

---

# 4. Engine Core 是 vLLM V1 的心脏

官方架构中 Engine Core：

- 运行 scheduler；
- 管理 KV cache；
- 协调 GPU workers；
- 不断进行 scheduling → execute → update 的循环。

可以想成：

```text
while server is running:
    scheduler.schedule()
    model_executor.execute()
    scheduler.update_from_output()
```

这就是一个典型 busy loop。

它和 Orca 的思想非常接近：

> **每个 model iteration 都重新做 scheduling decision。**

---

# 5. vLLM V1 Scheduler 的调度单位不是“Request”，而是“Token Budget”

这是理解 V1 很重要的一点。

scheduler 每轮并不只是输出：

```text
[A, B, C]
```

而更接近：

```text
A → 本轮处理 1 token
B → 本轮处理 1 token
C → 本轮处理 512 prompt tokens
D → 本轮处理 128 prompt tokens
```

也就是：

```text
{request_id: num_tokens_to_process}
```

为什么这个抽象很强？

因为它统一了：

```text
Decode
→ num_tokens = 1

Full Prefill
→ num_tokens = prompt length

Chunked Prefill
→ num_tokens = chunk size

Prefix Cache Hit
→ 只处理没缓存的 suffix

Speculative Decode
→ 可能一次验证多个 tokens
```

这是一种比“prefill request / decode request”更统一的 scheduler abstraction。

---

# 6. Continuous Batching 在 vLLM 里怎么体现？

每轮 scheduler 都会检查：

```text
running requests
waiting requests
finished requests
KV capacity
```

于是：

```text
step 1 [A B C]
step 2 [A B C]
A finished
step 3 [D B C]
```

这就是 Orca iteration-level scheduling 继续演化后的 production engine 形态。

---

# 7. KV Cache Manager 是另一个核心心脏

Scheduler 想让 request A 继续跑之前必须知道：

```text
A 下一批 token 需要多少新的 KV blocks？
还有没有 free blocks？
哪些 blocks 已经因为 prefix cache 命中而存在？
request 完成后哪些 blocks 可以释放？
```

所以 scheduler 与 KV Cache Manager 高度耦合。

可以画成：

```text
Scheduler
   │
   │ “我想让 A 再算 128 tokens”
   ▼
KV Cache Manager
   │
   │ “需要 8 blocks，有空间 / 没空间”
   ▼
Scheduler decision
```

这就是 PagedAttention 从论文里的 memory mechanism 变成 runtime control plane 的地方。

---

# 8. Paged KV Cache 到底如何帮助 scheduler？

如果每个 request 必须连续大块内存：

```text
A [████████████]
B [██████████████████]
```

增长、释放都很难。

Paged memory：

```text
A logical blocks
0 1 2 3
│ │ │ │
▼ ▼ ▼ ▼
physical 7, 19, 2, 31
```

新增 token 时：

```text
只需要再 allocate 一个 block
```

request finished：

```text
把对应 blocks 放回 free pool
```

因此：

> **Continuous batching 是“请求调度的动态化”，Paged KV Cache 是“显存生命周期的动态化”。**

两者天然配套。

---

# 9. Prefix Caching 在 V1 中是什么？

假设：

```text
Request A:
System Prompt + User A

Request B:
System Prompt + User B
```

公共：

```text
System Prompt
```

如果 A 已经完成这段 prefill：

```text
KV(System Prompt)
```

B 可以直接 reuse。

vLLM 的 Automatic Prefix Caching 会把已有 KV blocks 与 prefix 内容建立可查找关系。

新 request：

```text
先查 cache
↓
hit blocks
↓
只 prefill 剩余 token
```

因此 Scheduler 的 num_tokens abstraction 又显得很自然。

---

# 10. Prefix Cache 和 SGLang RadixAttention 是不是一样？

目标相似：

> **跳过重复 prefix 的 prefill compute。**

但内部数据结构、cache policy、scheduler 策略和工程实现不完全一样。

可以先这样记：

```text
共同问题：
prefix KV reuse

vLLM：
block/hash-oriented automatic prefix caching

SGLang：
Radix tree 是其经典核心 abstraction
```

不要只因为目标一样，就认为 implementation 完全相同。

---

# 11. GPU Worker 是什么？

官方 V1 架构中通常每个 GPU 有 dedicated worker process。

Worker 的职责包括：

```text
加载 model weights
管理本 GPU runtime state
执行 forward pass
参与 distributed collectives
```

Engine Core 不直接把矩阵乘算出来。

它告诉 worker：

```text
本轮执行这些 tokens / requests
```

worker 再调用 Model Runner 和底层 kernel。

---

# 12. Model Runner 又是什么层？

它更靠近实际模型执行：

```text
SchedulerOutput
      ↓
prepare model inputs
      ↓
attention metadata
      ↓
forward
      ↓
sampling / logits
```

所以：

```text
Scheduler：决定工作
Model Runner：组织本轮模型输入和执行
Kernel：真正做 GPU math
```

---

# 13. 多 GPU 时进程结构怎么理解？

如果：

```text
TP = 4
```

可以粗略理解成：

```text
1 Engine Core
    │
    ├── Worker GPU0
    ├── Worker GPU1
    ├── Worker GPU2
    └── Worker GPU3
```

每个 worker 持有自己那一份模型 shard。

一轮 forward 中：

```text
所有 TP workers 协同执行
```

并通过 NCCL 等通信。

如果再加 Data Parallel：

```text
DP rank 0 → 一个 Engine Core + 一组 workers
DP rank 1 → 一个 Engine Core + 一组 workers
...
```

还需要更上层协调流量。

---

# 14. Chunked Prefill 为什么自然融入 V1 Scheduler？

因为 scheduler 本来就允许：

```text
request → num_tokens
```

所以一个 10K prompt：

```text
不是必须：
num_tokens = 10000
```

而可以：

```text
step1: 1024
step2: 1024
step3: 1024
...
```

中间再安排 running decodes。

这也是为什么 token-budget scheduler 是很强的统一抽象。

---

# 15. Speculative Decoding 为什么也能塞进同一 Runtime？

普通 decode：

```text
每轮确认 1 token
```

Speculative decoding：

```text
draft model 猜多个
↓
target model 一次验证多个
↓
接受若干 tokens
```

scheduler 必须处理：

```text
一次 forward 可能涉及 >1 token / request
```

所以现代 scheduler 不再能简单假设：

```text
decode = exactly 1 token
```

V1 的 token-oriented abstraction对此更灵活。

---

# 16. P/D Disaggregation 在 vLLM 里怎么进入架构？

传统：

```text
一个 vLLM instance
做 Prefill + Decode
```

Disaggregated：

```text
Prefill vLLM instance
       │
       │ KV Connector
       ▼
Decode vLLM instance
```

于是 KV Cache Manager 不再只管理：

```text
本地 HBM blocks
```

还可能需要和 connector 协作：

```text
load remote KV
save KV
等待 transfer 完成
```

这说明 vLLM 正从：

> **single-engine memory manager**

继续向：

> **distributed KV-aware runtime**

演化。

---

# 17. vLLM V1 为什么强调“简化核心架构”？

serving engine 很容易随着功能增长变成：

```text
prefill 一个 path
decode 一个 path
spec decode 一个 path
chunked prefill 一个 path
prefix cache 一个 path
multimodal 又一个 path
```

最后 scheduler 状态极其复杂。

V1 的设计方向之一是尝试用更统一的 request/token scheduling 与 KV management abstractions 支撑不同 feature。

这是很典型的 runtime 重构目标：

> **不是只为了代码更漂亮，而是减少 feature combinations 对 scheduler 的组合爆炸。**

---

# 18. vLLM 与 FlashInfer 的关系

vLLM 负责：

```text
request scheduling
KV management
model execution orchestration
```

Attention backend 负责：

```text
具体 Attention kernel 怎么跑
```

所以在不同环境中可以使用不同 backend。

这再次强调：

```text
Serving Engine ≠ Attention Kernel
```

是两个层次。

---

# 19. 一个请求完整生命周期

```text
Client
  │
  ▼
API Server
  │ tokenize / validate
  ▼
Engine Core
  │
  ├── request enters waiting queue
  │
  ▼
Scheduler
  │
  ├── 查询 prefix hit
  ├── 查询 KV blocks
  ├── 分配 token budget
  └── 生成 SchedulerOutput
  │
  ▼
GPU Workers
  │
  ▼
Model Runner
  │
  ├── attention backend
  ├── GEMM/MoE kernels
  └── sampling
  │
  ▼
new token(s)
  │
  ├── finished → stream to client / cleanup KV
  └── running  → next Engine Core iteration
```

真正把这张图看懂，你就从“会用 vLLM”进入了“理解 vLLM runtime”。

---

# 20. AI Infra 最值得记住的 9 个 Insight

1. **vLLM 今天已经远远不等于 PagedAttention。**
2. **Engine Core 是 scheduler + KV management + worker coordination 的控制中心。**
3. **V1 scheduler 更适合用 token budget 而不是简单 request 类型理解。**
4. **Paged KV Cache 是 scheduler 可以做 fine-grained continuous batching 的基础。**
5. **Prefix caching 会直接改变“本轮还需要算多少 prompt tokens”。**
6. **GPU Worker / Model Runner 属于执行面，Engine Core 更像控制面。**
7. **Chunked prefill 和 speculative decoding 都推动 scheduler 从 request-centric 走向 token-centric。**
8. **KV Connector 把 vLLM 的 KV 管理范围从本地 GPU 扩展到跨 engine。**
9. **现代 LLM runtime 最难的部分往往不是单个 kernel，而是大量动态状态的统一管理。**

---

# 21. 推荐源码阅读顺序

不要直接从 PagedAttention CUDA kernel 开始。

推荐：

```text
1. V1 Architecture Overview
       ↓
2. Engine Core main loop
       ↓
3. Scheduler.schedule()
       ↓
4. KV Cache Manager
       ↓
5. Worker / Model Runner
       ↓
6. Attention Backend
       ↓
7. PagedAttention kernel
       ↓
8. KV Connector / Disaggregated Prefill
```

先控制流，再数据结构，最后 kernel。

---

## 主要参考资料

- vLLM 官方 Documentation：Architecture Overview（V1）。
- vLLM V1 Scheduler API / Scheduler Interface documentation。
- vLLM Automatic Prefix Caching documentation。
- vLLM Paged Attention design documentation。
- vLLM Disaggregated Prefilling / KV Connector documentation（截至 2026-08）。
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023.
