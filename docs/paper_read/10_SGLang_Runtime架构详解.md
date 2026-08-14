# 10｜SGLang Runtime 架构详解：TokenizerManager、Scheduler、ModelRunner、Radix Cache 如何组成一条推理流水线？

> **标题缩写与首次术语说明**：SGLang 是面向结构化生成与高性能大语言模型服务的框架；SRT = **SGLang Runtime（SGLang 运行时）**；LLM = **Large Language Model（大语言模型）**；API = **Application Programming Interface（应用程序编程接口）**；KV Cache = **Key-Value Cache（键值缓存）**；TP = **Tensor Parallelism（张量并行）**；DP = **Data Parallelism（数据并行）**；EP = **Expert Parallelism（专家并行）**；PD = **Prefill-Decode（预填充-解码）**；GEMM = **General Matrix-Matrix Multiplication（通用矩阵-矩阵乘法）**；MoE = **Mixture of Experts（混合专家模型）**；FCFS = **First-Come, First-Served（先到先服务）**；BPE = **Byte Pair Encoding（字节对编码，一种常用分词算法）**；CTA = **Cooperative Thread Array（协作线程阵列）**；SM = **Streaming Multiprocessor（流式多处理器）**。本文中的 **TokenizerManager** 负责 CPU 侧分词/预处理，**Scheduler** 负责请求调度，**ModelRunner** 负责组织 GPU 模型执行，**Router** 负责在更高层选择实例或资源池。 另外：AI = **Artificial Intelligence（人工智能）**；CUDA = **Compute Unified Device Architecture（NVIDIA GPU 并行计算平台与编程模型）**；FA 在本文相关上下文中通常指 **FlashAttention**。

> 本文基于截至 2026-08 的 SGLang 官方文档与项目架构整理。
>
> 推荐前置：
> - `03_SGLang-RadixAttention论文详解.md`
> - `06_Orca与Continuous-Batching论文详解.md`
> - `07_DistServe与Prefill-Decode分离论文详解.md`
> - `FlashInfer 论文详解.md`
>
> 目标：从“RadixAttention 是什么”进一步进入“今天 SGLang Runtime（SRT）到底怎样把一个请求跑起来”。

---

# 1. 先明确：SGLang 不只等于 RadixAttention

原始 SGLang 工作让很多人记住：

```text
SGLang = RadixAttention
```

但今天 SGLang 是完整 inference serving framework。

它包含：

```text
API / Engine
Tokenizer Manager
Scheduler
KV Cache / Radix Cache
Model Runner
Attention Backend
Detokenizer
Router / Model Gateway
TP/DP/EP
Speculative Decoding
PD Disaggregation（Prefill-Decode 分离部署）
Structured Output
...
```

所以：

> **RadixAttention 是它非常重要的核心思想，但不是整个 runtime。**

---

# 2. SRT 的最小请求路径

可以先用这张图记：

```text
Client Request
      │
      ▼
TokenizerManager
      │
      ▼
Scheduler
      │
      ├── request queue
      ├── radix/prefix cache
      ├── KV memory pool
      └── batch scheduling
      │
      ▼
ModelRunner
      │
      ├── Attention Backend
      ├── GEMM / MoE kernels
      └── distributed collectives
      │
      ▼
DetokenizerManager
      │
      ▼
Client Stream
```

这几个组件的进程/线程组织会随启动模式、版本与并行配置变化，但逻辑职责非常稳定。

---

# 3. TokenizerManager 为什么单独存在？

用户传入的是：

```text
“你好，请解释一下 FlashAttention”
```

GPU model 需要：

```text
[token_id_0, token_id_1, ...]
```

TokenizerManager 通常负责：

- request parsing；
- tokenize；
- multimodal / input preprocessing 的一部分；
- 将 request 送给 scheduler；
- 接收生成结果并组织 streaming response。

这样做的一个原因是：

> **CPU-side preprocessing 不应该塞进 GPU scheduler 的核心循环。**

---

# 4. Scheduler 是 SGLang Runtime 的核心控制面

Scheduler 每轮需要管理：

```text
waiting requests
running requests
prefill requests
decode requests
KV Cache usage
prefix cache hits
batch token budget
```

然后决定：

```text
这一轮谁进 batch？
处理多少 token？
是否 prefill？
是否 decode？
是否有 cache hit？
是否要 chunk prefill？
```

它和 vLLM Engine Core 中 scheduler 在系统职责上有大量相似性。

---

# 5. Radix Cache 为什么直接影响 Scheduler？

假设等待队列：

```text
A = [System Prompt][Question A]
B = [System Prompt][Question B]
C = [完全不同 Prompt]
```

如果：

```text
System Prompt KV
```

已经在 cache 中，A/B 的 prefill 成本会明显降低。

因此 scheduler 不能只看：

```text
request arrival time
```

还应该知道：

```text
这个 request 和当前 cache 能共享多少 prefix？
```

这就是 SGLang 非常鲜明的特点：

> **cache state 和 request scheduling 被紧密联系起来。**

---

# 6. Cache-Aware Scheduling 的直觉

假设两个 waiting request：

```text
Request A
可以复用 4000 tokens KV

Request B
0 tokens cache hit
```

如果先调 A：

```text
只算很短 suffix
```

可能更快完成，同时继续保留某些 hot prefix。

所以 scheduling policy 不只是：

```text
FCFS
```

还可以利用：

```text
prefix cache locality
```

这把 serving scheduler 变得有一点像传统计算机系统的：

```text
cache-aware task scheduling
```

---

# 7. ModelRunner 是什么？

Scheduler 决定：

```text
这一轮跑哪些 requests/tokens
```

ModelRunner 负责：

```text
把 scheduling decision 变成真实 GPU tensors
准备 position / attention metadata
调用 model forward
调用 attention backend
执行 sampling
```

可以简单分层：

```text
Scheduler
= policy / control

ModelRunner
= execution preparation

Kernel Backend
= actual GPU math
```

---

# 8. Attention Backend 为什么在 SGLang 中特别值得关注？

Attention kernel 没有一种方案在所有 workload 上都绝对最好。

例如：

```text
Prefill
Q 很长

Decode
Q 很短，KV 很长
```

最佳 kernel 可能不同。

SGLang 提供不同 attention backend，并且现代版本甚至支持：

```text
Prefill backend A
Decode backend B
```

即 hybrid attention backend。

这与 FlashInfer 论文的思想高度一致：

> **Serving Attention 是异构 workload，不应假设一个固定 kernel 解决所有情况。**

---

# 9. FlashInfer 在 SGLang 里处于哪里？

可以画成：

```text
SGLang Runtime
│
├── Scheduler
├── Radix Cache
├── ModelRunner
│
└── Attention Backend
       │
       ├── FlashInfer
       ├── FlashAttention variants
       └── other specialized kernels
```

所以：

```text
SGLang ≠ FlashInfer
```

而是：

```text
SGLang 可以调用 FlashInfer 作为底层 Attention execution backend
```

这就是 serving engine 与 kernel engine 的分层。

---

# 10. DetokenizerManager 为什么还要独立出来？

GPU 输出的是：

```text
token IDs
```

用户要看到：

```text
自然语言字符串
```

并且 streaming decode 还要处理：

- BPE/SentencePiece 边界；
- incomplete byte sequences；
- stop tokens / stop strings；
- streaming text assembly。

这些属于 CPU/string workload。

因此把它从 GPU scheduler 核心路径中分离是很自然的系统设计。

---

# 11. SGLang 如何体现 Continuous Batching？

仍然是：

```text
每个 generation iteration
scheduler 重新形成 batch
```

request 可以：

```text
finish → 离开
new → 加入
```

同时 scheduler 还会结合：

```text
KV memory
prefix cache
prefill/decode 状态
```

所以 SGLang scheduler 是 Orca continuous-batching 思想 + prefix-aware state 的进一步工程化。

---

# 12. Chunked Prefill 在 SGLang 中解决什么？

一个 64K prompt：

```text
如果一次性 prefill
→ 很长 GPU iteration
→ decode requests 被卡住
```

切成：

```text
chunk 1
chunk 2
chunk 3
...
```

就可以在 scheduler 中更灵活地交错：

```text
prefill chunk
+ decode batch
```

这在长 context serving 中非常重要。

SGLang 暴露的 `chunked-prefill-size` 一类参数，就是这种系统 trade-off 的实际入口。

---

# 13. Scheduler 与 KV Memory Pool 的关系

和 vLLM 一样：

```text
想 schedule request
↓
必须保证有 KV memory
```

SGLang 通常会预留大块 GPU memory 作为静态 KV memory pool，再在其中管理 token/KV slots。

参数如：

```text
mem-fraction-static
max-running-requests
```

本质都是在调：

```text
模型权重 / workspace / KV cache / concurrency
```

之间的显存预算。

---

# 14. 为什么 SGLang 的 Prefix Cache 更像 runtime 的一等公民？

因为 RadixAttention 从设计之初就不是：

```text
“偶尔命中一下 prefix”
```

而是把：

```text
token sequence → KV cache
```

组织成 radix tree。

request insertion / matching / eviction / scheduling 都围绕这棵 cache structure 展开。

所以 SGLang 特别适合思考：

> **Agent / multi-turn / few-shot / shared system prompt 这种 prefix-heavy workload。**

---

# 15. SGLang 的 PD Disaggregation 怎么理解？

现代 SGLang 支持：

```text
Prefill-only workers
Decode-only workers
```

高层：

```text
Router
 │
 ├── Prefill Worker Pool
 │       │
 │       │ transfer KV
 │       ▼
 └── Decode Worker Pool
```

Prefill / Decode worker 可以选择不同参数和 backend。

这正是 DistServe 思想进入现代 runtime 的一种工程形态。

---

# 16. KV Transfer Backend 为什么成为新组件？

因为 P worker 算出的 KV 必须给 D worker。

现代 SGLang 的 disaggregation configuration 可以选择不同 transfer backend。

这意味着架构从：

```text
Scheduler
↓
GPU local KV Cache
```

变成：

```text
Scheduler
↓
Local KV
↕
KV Transfer Layer
↕
Remote Engine
```

这层以后会越来越重要。

---

# 17. Router / Model Gateway 是更上一层什么东西？

单个 SGLang engine 解决：

```text
一组 GPU 如何 serve requests
```

但大规模部署可能有：

```text
几十/几百个 model workers
```

于是需要 Router：

```text
Client
  │
  ▼
Router / Gateway
  │
  ├── worker A
  ├── worker B
  ├── worker C
  └── ...
```

Router 可以考虑：

```text
load
cache locality
PD role
health
model
```

所以系统又多一层：

```text
Cluster Scheduler / Router
        ↓
Engine Scheduler
        ↓
GPU Kernel Scheduler
```

注意这三个“scheduler”不是一回事。

---

# 18. 三层调度一定要分清

## 1. Cluster / Router 层

```text
这个 request 发给哪个 engine？
```

## 2. Engine Scheduler 层

```text
这个 engine 下一轮 batch 放哪些 request/token？
```

## 3. GPU Kernel 层

```text
一个 Attention workload 如何分给 CTA/SM？
```

例如：

```text
SGLang Router
       ↓
SGLang Scheduler
       ↓
FlashInfer Scheduler / CUDA Kernel
```

三个层次都可能说 scheduling，但粒度完全不同。

这是读 AI Infra 源码时非常重要的概念区分。

---

# 19. SGLang 和 vLLM Runtime 的相似与不同

## 相似

两者今天都有：

- continuous batching；
- KV cache management；
- prefix caching；
- chunked prefill；
- speculative decoding；
- TP/DP 等并行；
- P/D disaggregation；
- 多种 attention backend；
- OpenAI-compatible serving。

## 历史基因不同

vLLM 最鲜明的起点：

```text
PagedAttention / KV memory efficiency
```

SGLang 最鲜明的起点：

```text
Language Model Programs
+
RadixAttention / prefix reuse
```

所以两者今天功能越来越重叠，但代码组织、scheduler policy、cache abstraction 和优化重心仍有差别。

---

# 20. 请求生命周期完整图

```text
Client
 │
 ▼
TokenizerManager
 │ tokenize / preprocess
 ▼
Scheduler
 │
 ├── 查 Radix Cache
 ├── 管 waiting/running queues
 ├── 预算 KV slots
 ├── 决定 prefill/decode/chunks
 └── build batch
 │
 ▼
ModelRunner
 │
 ├── prepare GPU tensors
 ├── attention metadata
 ├── distributed execution
 └── sampling
 │
 ▼
Attention Backend / Kernels
 │
 ▼
new token IDs
 │
 ▼
DetokenizerManager
 │
 ▼
TokenizerManager / Stream Response
 │
 ├── finished → cleanup / return
 └── running → Scheduler next iteration
```

---

# 21. AI Infra 最值得记住的 10 个 Insight

1. **SGLang 是完整 runtime，RadixAttention 只是核心组件之一。**
2. **Tokenizer/Scheduler/ModelRunner/Detokenizer 分离，本质是 CPU control plane 与 GPU execution plane 解耦。**
3. **Scheduler 不只是 FCFS，它可以感知 KV/prefix cache 状态。**
4. **Radix cache 会直接改变 request 的实际 prefill cost。**
5. **Attention backend 是可替换执行层，FlashInfer 位于这一层。**
6. **Prefill 与 Decode 可以选择不同 backend，这体现 workload heterogeneity。**
7. **Chunked prefill 是控制 long-prefill interference 的关键手段。**
8. **PD disaggregation 让 KV transfer backend 成为 runtime 一等公民。**
9. **Router scheduler、engine scheduler、GPU kernel scheduler 是三种完全不同粒度。**
10. **现代 SGLang/vLLM 的功能越来越重叠，比较时应看具体 workload 和版本，而不是旧标签。**

---

# 22. 推荐源码阅读顺序

```text
1. Engine / launch path
       ↓
2. TokenizerManager
       ↓
3. Scheduler main loop
       ↓
4. Radix Cache / memory pool
       ↓
5. ModelRunner
       ↓
6. Attention backend abstraction
       ↓
7. FlashInfer / FA kernel integration
       ↓
8. PD disaggregation / router
```

依然遵循原则：

> **先看 request lifecycle，再看数据结构，最后看 kernel。**

---

## 主要参考资料

- SGLang 官方 Documentation（截至 2026-08）。
- SGLang Architecture / Runtime documentation：Engine、TokenizerManager、Scheduler、ModelRunner、Detokenizer。
- SGLang PD Disaggregation documentation。
- SGLang Attention Backend documentation。
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, arXiv:2312.07104.
