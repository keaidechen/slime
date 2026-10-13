# SGLang Runtime：请求、执行与 KV 三种状态如何协作

<details>
<summary>文章导航</summary>

- [1. 先明确：SGLang 不只等于 RadixAttention](<#1-先明确sglang-不只等于-radixattention>)
- [2. SRT 的最小请求路径](<#2-srt-的最小请求路径>)
- [3. TokenizerManager 为什么单独存在？](<#3-tokenizermanager-为什么单独存在>)
- [4. Scheduler 是 SGLang Runtime 的核心控制面](<#4-scheduler-是-sglang-runtime-的核心控制面>)
- [5. Radix Cache 为什么直接影响 Scheduler？](<#5-radix-cache-为什么直接影响-scheduler>)
- [6. Cache-Aware Scheduling 的直觉](<#6-cache-aware-scheduling-的直觉>)
- [7. ModelRunner 是什么？](<#7-modelrunner-是什么>)
- [8. Attention Backend 为什么在 SGLang 中特别值得关注？](<#8-attention-backend-为什么在-sglang-中特别值得关注>)
- [9. FlashInfer 在 SGLang 里处于哪里？](<#9-flashinfer-在-sglang-里处于哪里>)
- [10. DetokenizerManager 为什么还要独立出来？](<#10-detokenizermanager-为什么还要独立出来>)
- [11. SGLang 如何体现 Continuous Batching？](<#11-sglang-如何体现-continuous-batching>)
- [12. Chunked Prefill 在 SGLang 中解决什么？](<#12-chunked-prefill-在-sglang-中解决什么>)
- [13. Scheduler 与 KV Memory Pool 的关系](<#13-scheduler-与-kv-memory-pool-的关系>)
- [14. 为什么 SGLang 的 Prefix Cache 更像 runtime 的一等公民？](<#14-为什么-sglang-的-prefix-cache-更像-runtime-的一等公民>)
- [15. SGLang 的 PD Disaggregation 怎么理解？](<#15-sglang-的-pd-disaggregation-怎么理解>)
- [16. KV Transfer Backend 为什么成为新组件？](<#16-kv-transfer-backend-为什么成为新组件>)
- [17. Router / Model Gateway 是更上一层什么东西？](<#17-router--model-gateway-是更上一层什么东西>)
- [18. 三层调度一定要分清](<#18-三层调度一定要分清>)
- [19. SGLang 和 vLLM Runtime 的相似与不同](<#19-sglang-和-vllm-runtime-的相似与不同>)
- [20. 请求生命周期完整图](<#20-请求生命周期完整图>)
- [21. 从概念进入源码](<#21-从概念进入源码>)
- [22. 三种状态怎样衔接](<#22-三种状态怎样衔接>)

</details>

相关基础：[Transformer 与 KV](<../../../../05_Inference/19_生成执行与采样.md#s1>)。


> 阅读位置：SGLang 源码之前的概念课。先读普通文本生成的 Scheduler、KV、ModelRunner 分工，再读高级能力；实际符号与分支以本仓库快照及后续源码章节为准。

本文的架构概览来自原稿截至 2026-08 的资料，具体符号与行为以本系列声明的仓库快照及后续源码章节为准。SRT 指 SGLang Runtime；TokenizerManager 组织输入，Scheduler 推进请求，ModelRunner 组织设备执行，Router 选择实例或资源池。

理解 [KV 内存](<../../../../05_Inference/20_KV容量分页与复用.md#s1>)与[批处理调度](<../../../../05_Inference/21_批处理调度与排队.md#s1>)即可进入；PagedAttention、Orca、DistServe 和 FlashInfer 的完整论文分析是深入资料，不是本篇必须读完的前置。

## 1. 先明确：SGLang 不只等于 RadixAttention

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


## 2. SRT 的最小请求路径

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


## 3. TokenizerManager 为什么单独存在？

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


## 4. Scheduler 是 SGLang Runtime 的核心控制面

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


## 5. Radix Cache 为什么直接影响 Scheduler？

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


## 6. Cache-Aware Scheduling 的直觉

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


## 7. ModelRunner 是什么？

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


## 8. Attention Backend 为什么在 SGLang 中特别值得关注？

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


## 9. FlashInfer 在 SGLang 里处于哪里？

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


## 10. DetokenizerManager 为什么还要独立出来？

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


## 11. SGLang 如何体现 Continuous Batching？

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


## 12. Chunked Prefill 在 SGLang 中解决什么？

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


## 13. Scheduler 与 KV Memory Pool 的关系

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


## 14. 为什么 SGLang 的 Prefix Cache 更像 runtime 的一等公民？

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


## 15. SGLang 的 PD Disaggregation 怎么理解？

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


## 16. KV Transfer Backend 为什么成为新组件？

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


## 17. Router / Model Gateway 是更上一层什么东西？

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


## 18. 三层调度一定要分清


### 1. Cluster / Router 层

```text
这个 request 发给哪个 engine？
```


### 2. Engine Scheduler 层

```text
这个 engine 下一轮 batch 放哪些 request/token？
```


### 3. GPU Kernel 层

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


## 19. SGLang 和 vLLM Runtime 的相似与不同


### 相似

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


### 历史基因不同

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


## 20. 请求生命周期完整图

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


## 21. 从概念进入源码

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


### 主要参考资料

- SGLang 官方 Documentation（截至 2026-08）。
- SGLang Architecture / Runtime documentation：Engine、TokenizerManager、Scheduler、ModelRunner、Detokenizer。
- SGLang PD Disaggregation documentation。
- SGLang Attention Backend documentation。
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, arXiv:2312.07104.


<a id="learning-models"></a>


## 22. 三种状态怎样衔接

### 从容量与工作量理解状态

Prefill 的线性层工作随 token 数增长，dense Attention 随可见 token pair 数增长；decode 中权重和 KV 的读取复用又受 batch 影响。调度器改变的是活跃工作集，不能把“token 数相同”直接当作“计算量相同”。通用推导见[Transformer 执行与 KV](<../../../../05_Inference/19_生成执行与采样.md#s1>)。

KV 容量还要使用本 rank 的 `local_kv_heads`：`layers × 2 × local_kv_heads × head_dim × dtype_bytes`。例如 32 层、8 KV heads、head_dim 128、BF16 得到 128 KiB/token；100,000 个 token 约 12.21 GiB，102,400 个才是 12.5 GiB。页表、allocator 与 workspace 另外计入；请求结束也不代表可复用前缀立刻销毁。完整分配与回收过程见[KV 生命周期算例](<../../../../05_Inference/20_KV容量分页与复用.md#s1>)。

### Request、Batch、KV 状态机

##### Request

```text
received → waiting → running
                   ↘ finished
                   ↘ aborted
          ↖ retracted
```

##### Batch

```text
schedule plan → prepare metadata → GPU forward
              → sample/result process → next plan
```

overlap 模式下这些阶段属于相邻 iteration，不能把“当前 batch”和“当前处理的 result”默认视为同一个。

##### KV

```text
free slot → allocated to req → committed
          → radix cached/protected → evictable → freed
```

silent corruption 往往来自三个状态机推进不同步。


### 请求主线如何连接各专题

进程与 IPC 定义消息边界，scheduler 推进 prefill/decode，KV/Radix 管理 pool 与引用锁。ModelRunner 将 ForwardBatch 的 shape 和 metadata 交给 eager 或 CUDA Graph 路径。TP/EP/PD 再引入跨 rank 数据移动与所有权交接。

Speculative 和 grammar 增加 token、状态回滚与合法输出的约束。最终服务表现由这条完整路径上的 latency-throughput 关系和 SLO 决定；源码模块的边界需要与状态边界一起理解。

### 延伸阅读

- [SGLang 论文](https://arxiv.org/abs/2312.07104)：先读 runtime 与 RadixAttention。
- [RadixAttention 项目团队博客](https://www.lmsys.org/blog/2024-01-17-sglang/)：图示比直接读树代码更适合入门。
- [SGLang v0.4 调度器博客](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)：理解 overlap scheduler、cache-aware router 与 DP attention。
- [Mini-SGLang 教学项目博客](https://www.lmsys.org/blog/2025-12-17-minisgl/)：用较小实现建立完整推理引擎心智模型，再回主仓库。
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)：理解 paged KV 与连续批处理的内存动机。
