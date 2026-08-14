# 08｜TensorRT-LLM Runtime 架构详解：从“高性能 Kernel”到 NVIDIA 推理执行系统

> **标题缩写与首次术语说明**：TensorRT-LLM 是 NVIDIA 面向大语言模型的高性能推理与 runtime（运行时）技术栈，项目名不强行字母展开；LLM = **Large Language Model（大语言模型）**；API = **Application Programming Interface（应用程序编程接口）**；KV Cache = **Key-Value Cache（键值缓存）**；GPU = **Graphics Processing Unit（图形处理器）**；IFB = **In-Flight Batching（在途批处理，即运行中动态更新 batch）**；GEMM = **General Matrix-Matrix Multiplication（通用矩阵-矩阵乘法）**；MoE = **Mixture of Experts（混合专家模型）**；PP = **Pipeline Parallelism（流水线并行）**；EP = **Expert Parallelism（专家并行）**；HTTP = **Hypertext Transfer Protocol（超文本传输协议）**。本文中的 **scheduler** 是请求调度器，**executor/worker** 是实际组织或执行模型计算的组件，**backend** 是某类算子/执行能力的后端实现。 另外：CUDA = **Compute Unified Device Architecture（NVIDIA GPU 并行计算平台与编程模型）**；FP16/BF16/FP8 分别是 **16 位浮点、bfloat16 与 8 位浮点格式**；INT8/INT4 分别是 **8 位/4 位整数格式**；FP4 = **4-bit Floating Point（4 位浮点格式）**；AI = **Artificial Intelligence（人工智能）**。

> 本文不是一篇单独论文摘要，而是截至 2026-08 的 **TensorRT-LLM runtime 架构学习笔记**。
>
> 目标：理解一个请求从 API 进入，到 scheduler、KV Cache、GPU worker、kernel、distributed execution 的完整路径，并把它和 Orca、PagedAttention、FlashAttention、DistServe 串起来。

---

# 1. 先明确 TensorRT-LLM 是什么层

可以先画成：

```text
Application / OpenAI API Client
             │
             ▼
       trtllm-serve / LLM API
             │
             ▼
     TensorRT-LLM Runtime
             │
     ┌───────┼────────┐
     │       │        │
 Scheduler  KV Cache  Executor
     │       │        │
     └───────┼────────┘
             ▼
   Optimized GPU Kernels
             │
             ▼
      NVIDIA GPU(s)
```

TensorRT-LLM 的核心定位可以理解成：

> **NVIDIA 面向 LLM inference 的高性能模型执行与 serving runtime。**

它不仅有 kernel，也包含：

- model execution；
- in-flight batching（在途批处理：运行过程中动态加入/移出请求）；
- paged KV cache；
- request scheduling；
- quantization；
- tensor/pipeline/expert parallelism；
- speculative decoding；
- serving API；
- 多 GPU / 多节点执行。

---

# 2. TensorRT、TensorRT-LLM、Triton Inference Server 不要混

这是初学者最容易混淆的一组名字。

## TensorRT

更通用的 NVIDIA 深度学习推理优化 / execution 技术栈。

## TensorRT-LLM

专门针对：

```text
Transformer / LLM / MoE / multimodal generation
```

进行优化的库和 runtime。

## Triton Inference Server

更上层的 serving server，可以托管不同 backend/model。

今天 TensorRT-LLM 本身也提供：

```text
trtllm-serve
LLM API
```

因此实际部署形态已经比早期“TensorRT-LLM backend + Triton”更加多样。

---

# 3. 为什么 TensorRT-LLM 也必须有 Scheduler？

因为 kernel 再快，也解决不了：

```text
同时来 1000 个用户怎么办？
谁先算？
谁这一轮进 batch？
KV Cache 不够怎么办？
Prefill 和 Decode 怎么混？
```

这就是 Orca 之后所有现代 LLM runtime 都必须面对的问题。

所以 TensorRT-LLM 也有：

> **in-flight batching / request scheduling。**

---

# 4. In-Flight Batching 是什么？

它本质上就是我们在 Orca 文档里讲的 continuous batching 思想。

静态 batch：

```text
[A B C]
一直绑定到所有请求完成
```

In-flight：

```text
step1 [A B C]
step2 [A B C]
step3 [D B C]   A finished
step4 [D B E]   C finished
```

请求可以在 batch 执行过程中动态进入/退出。

这让 GPU decode batch 可以保持较高利用率。

---

# 5. 为什么 In-Flight Batching 与 Paged KV Cache 必须配合？

如果 batch membership 每轮都变化：

```text
A finished
D joins
E joins
B grows
```

那么 KV memory 也必须：

```text
动态 allocate
动态 free
动态 grow
```

如果每个请求预留：

```text
max_context + max_generation
```

会非常浪费显存。

所以现代 TensorRT-LLM runtime 同样使用 paged KV cache 思路：

```text
Logical request KV
        ↓ mapping
Physical KV blocks
```

它和 vLLM PagedAttention 在宏观思想上属于同一代 memory-management 演化。

---

# 6. Runtime 里的 Scheduler 大概在决定什么？

每个 scheduling step 要回答：

```text
哪些 requests 运行？
每个 request 这轮处理多少 tokens？
有没有足够 KV Cache block？
要不要暂停/推迟某些 request？
prefill / decode 怎么组合？
```

最后形成：

```text
scheduled batch
```

交给 executor / model runtime。

所以从系统层次看：

```text
Scheduler
= 决定“算谁”

Executor
= 决定“怎么把它真正跑在 GPU 上”
```

---

# 7. Executor 是什么？

可以把 Executor 理解成 TensorRT-LLM runtime 的核心执行抽象之一。

它负责接收 request / scheduling information，并组织：

- model execution；
- GPU resources；
- KV cache；
- distributed ranks；
- output generation；
- request lifecycle。

简化图：

```text
Requests
  │
  ▼
Scheduler
  │
  ▼
Executor
  │
  ├── rank 0 GPU
  ├── rank 1 GPU
  ├── rank 2 GPU
  └── ...
```

---

# 8. 模型到底怎么跨多 GPU？

TensorRT-LLM 支持多种 parallelism。

最常见先记：

## Tensor Parallelism

把一层大矩阵切给多张 GPU：

```text
Linear Weight
┌──────────────┐
│ GPU0 │ GPU1  │
└──────────────┘
```

每层都需要一定通信。

## Pipeline Parallelism

把不同层切给不同 GPU：

```text
GPU0: layer 0~19
      ↓
GPU1: layer 20~39
      ↓
GPU2: layer 40~59
```

## Expert Parallelism

MoE 中不同 experts 放到不同 GPU。

这些 parallelism 不只是“模型放不下”时使用，也会影响：

```text
latency
throughput
communication
KV placement
```

---

# 9. Runtime 与 Kernel 是什么关系？

TensorRT-LLM 性能强不是因为 scheduler 本身“算得快”。

真正 GPU 计算由各种高度优化 kernel 完成：

```text
Attention
GEMM
MoE
LayerNorm
Quantization
Sampling
Collective Communication
...
```

Runtime 做的是：

```text
什么时候调用什么 kernel
用什么 batch shape
用哪些 GPU
KV 地址在哪里
```

因此可以把系统理解成：

```text
Policy / Scheduling Layer
          ↓
Execution Runtime
          ↓
Kernel Layer
          ↓
CUDA / GPU
```

---

# 10. TensorRT-LLM 为什么高度强调 Quantization？

因为 inference 性能很大程度取决于：

```text
权重搬运量
KV Cache 大小
Tensor Core dtype throughput
```

从：

```text
FP16/BF16
↓
FP8
↓
INT8 / INT4 / FP4 等
```

可以减少：

```text
memory footprint
memory bandwidth demand
```

并使用更高吞吐的低精度 Tensor Core。

但不同 layer / model 对精度敏感度不同，因此 TensorRT-LLM 还必须配合量化策略和专用 kernel。

---

# 11. 为什么 Paged Attention、IFB、Scheduler 是一组功能？

因为三者其实形成闭环：

```text
In-flight Batching
让 requests 动态进入/退出
        ↓
KV Cache 生命周期变动态
        ↓
Paged KV Cache
让显存按 block 动态分配
        ↓
Scheduler
根据 KV 空间决定哪些 requests 可以继续运行
```

这是 LLM serving runtime 最核心的一条 feedback loop。

---

# 12. Chunked Context / Chunked Prefill 在这里做什么？

如果 prompt 很长：

```text
32K tokens
```

一次完整 prefill 可能占用很长 GPU 时间。

Chunking：

```text
32K
↓
4K + 4K + ...
```

让 scheduler 可以在不同 chunk 之间插入其他 request 的 decode。

目标是：

```text
避免一个超长 context 把其他 generation 卡住
```

这正是 Orca 之后 scheduler 演化的重要方向。

---

# 13. TensorRT-LLM 当前为何同时存在“编译引擎”和 PyTorch workflow 的概念？

TensorRT-LLM 历史上很强调：

```text
把模型转换/构建成高度优化的 TensorRT engine
```

然后由 runtime 执行。

随着项目演化，又提供越来越完整的 PyTorch-oriented workflow / LLM API。

因此阅读不同年代文档时可能看到：

```text
Builder / Engine
ModelRunner
Executor
PyTorch workflow
LLM API
```

这些属于项目不同层次和演化阶段。

学习时不要把某个旧版本类名当作永恒架构。

稳定概念是：

```text
Request scheduling
KV cache management
Model execution
Optimized kernels
Distributed parallelism
Serving API
```

---

# 14. TensorRT-LLM 与 vLLM 最大的气质差异是什么？

粗略理解：

## TensorRT-LLM

更强烈地围绕：

```text
NVIDIA GPU architecture
TensorRT / CUDA kernel specialization
quantization
latest hardware features
```

做纵向深度优化。

## vLLM

历史上以：

```text
PagedAttention
easy model serving
flexible open-source engine
scheduler / memory manager
```

迅速形成广泛生态。

今天两者都已经覆盖大量重叠能力，因此不能简单说：

```text
一个只有 kernel
一个只有 scheduler
```

那已经过时。

更准确的是：

> **两者都是完整 LLM inference runtime，只是历史基因、硬件适配策略和生态侧重点不同。**

---

# 15. TensorRT-LLM 与 FlashAttention / FlashInfer 的关系

可以画成：

```text
TensorRT-LLM Runtime
│
├── Scheduler
├── KV Cache Manager
├── Executor
├── Distributed Parallelism
│
└── Kernel Backends / Specialized Kernels
     ├── Attention kernels
     ├── GEMM
     ├── MoE
     └── ...
```

FlashAttention/FlashInfer 更靠近：

```text
Kernel / attention execution layer
```

TensorRT-LLM 更靠近：

```text
end-to-end inference runtime
```

---

# 16. TensorRT-LLM 与 DistServe / P-D 分离的关系

DistServe 是一种 serving architecture：

```text
P pool
  ↓ KV
D pool
```

它和具体 engine 并不冲突。

现代 NVIDIA inference stack 也在向：

```text
router / disaggregated serving / KV transfer
```

方向演化。

所以 DistServe 更像：

> **集群级 architecture idea**

而 TensorRT-LLM 是：

> **可以成为 P worker / D worker 底层执行 engine 的 runtime 之一。**

---

# 17. 一个请求在 TensorRT-LLM 中的概念路径

可以先用这张图记：

```text
HTTP / API Request
       │
       ▼
Input processing / tokenizer
       │
       ▼
Request Queue
       │
       ▼
Scheduler / In-flight Batching
       │
       ├── 查 KV capacity
       ├── 决定 prefill/decode tokens
       └── 形成本轮 batch
       │
       ▼
Executor
       │
       ├── distributed ranks
       ├── model forward
       └── optimized kernels
       │
       ▼
Sampling
       │
       ▼
新 token
       │
       ├── finished → 返回
       └── not finished → 回到下一轮 scheduler
```

你会发现：

> 这就是 Orca iteration-level execution 的现代化、工程化版本之一。

---

# 18. AI Infra 最值得记住的 8 个 Insight

1. **TensorRT-LLM 是完整 inference runtime，不只是一个 Attention kernel 库。**
2. **In-flight batching 是 Orca/continuous batching 思想在生产 runtime 中的体现。**
3. **Paged KV Cache 和 scheduler 是互相依赖的。**
4. **Executor 负责把 scheduling decision 映射到真实 GPU execution。**
5. **多 GPU parallelism 是 runtime architecture 的核心组成，不是附加功能。**
6. **Quantization 与 kernel specialization 是 TensorRT-LLM 的重要硬件优势来源。**
7. **不要把某一版本的 Builder/ModelRunner 类名当作永恒抽象。**
8. **现代 serving 的稳定主线是：request scheduling + KV management + execution + kernels + distributed runtime。**

---

# 19. 推荐进一步读源码/文档的顺序

```text
1. TensorRT-LLM Overview
       ↓
2. In-flight Batching / Request Scheduling
       ↓
3. KV Cache / Paged Attention
       ↓
4. Executor API
       ↓
5. Attention kernel / quantization
       ↓
6. Tensor/PP/EP parallelism
       ↓
7. trtllm-serve / distributed deployment
```

不要一上来钻进 kernel template；先看清 runtime 的请求生命周期。

---

## 主要参考资料

- NVIDIA TensorRT-LLM 官方 Documentation（截至 2026-08）。
- TensorRT-LLM Overview：In-flight Batching、Paged Attention、Parallelism、LLM API。
- TensorRT-LLM Executor API / KV Cache / Memory documentation。
- NVIDIA CUDA / Hopper architecture documents（底层硬件概念）。
