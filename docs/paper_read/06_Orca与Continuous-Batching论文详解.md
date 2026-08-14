# 06｜Orca 与 Continuous Batching：LLM Serving 为什么不能使用传统“静态 Batch”？

> **标题缩写与首次术语说明**：LLM = **Large Language Model（大语言模型）**；OSDI = **USENIX Symposium on Operating Systems Design and Implementation（USENIX 操作系统设计与实现大会）**；KV Cache = **Key-Value Cache（键值缓存）**；GPU = **Graphics Processing Unit（图形处理器）**；GEMM = **General Matrix-Matrix Multiplication（通用矩阵-矩阵乘法）**。**Static batching（静态批处理）**是整批请求绑定到一起直到全部完成；**iteration-level scheduling（迭代级调度）**是在每一轮生成迭代重新决定 batch；**continuous batching（连续批处理）**允许运行过程中动态加入、移出请求；**selective batching（选择性批处理）**是 Orca 对不同算子采用不同 batching 方式的设计。 另外：MLP = **Multi-Layer Perceptron（多层感知机）**；AI = **Artificial Intelligence（人工智能）**。

> 论文：Gyeong-In Yu et al., **Orca: A Distributed Serving System for Transformer-Based Generative Models**，OSDI 2022。
>
> 这一篇在历史上非常重要：现代 LLM serving 里随处可见的 **continuous batching / in-flight batching（在途批处理：运行过程中动态加入/移出请求） / iteration-level scheduling**，都可以沿着这条思想线理解。
>
> 推荐前置：`00_共享基础_GPU与LLM推理硬件基础.md` 中 Prefill / Decode、KV Cache、batch、GPU memory 相关章节。

---

# 1. 先从传统 Batch 说起

假设普通图片分类模型收到 4 张图片：

```text
image A
image B
image C
image D
```

可以拼成一个 batch：

```text
[A, B, C, D]
```

一次 forward：

```text
model(batch)
```

四个请求同时结束。

这种任务很适合传统静态 batching：

```text
收一批请求
↓
一起执行一次
↓
整批返回
```

---

# 2. 但 LLM 是自回归、多 iteration 的

LLM 生成不是一次 forward 完成。

假设请求 A 要生成：

```text
20 tokens
```

请求 B：

```text
200 tokens
```

请求 C：

```text
50 tokens
```

它们会经历：

```text
iteration 1 → 每个请求产生一个 token
iteration 2 → 再产生一个 token
iteration 3 → ...
```

所以一个 request 的生命周期是：

```text
Prefill
  ↓
Decode 1
  ↓
Decode 2
  ↓
Decode 3
  ↓
...
```

而且输出长度事先通常不知道。

这就是 Orca 面对的根本问题。

---

# 3. 静态 batching 会产生什么问题？

假设一开始把：

```text
A = 20 tokens
B = 200 tokens
C = 50 tokens
```

绑成一个 batch。

如果 batch 生命周期固定：

```text
iteration 1
iteration 2
...
iteration 20
```

A 已经结束。

但 B/C 还没结束。

如果系统不能动态改变 batch：

- A 可能还得等 B 完成才统一返回；
- 或者 A 的 slot 后续一直浪费；
- 新来的请求 D 也不能立刻补进来。

时间线：

```text
A: █████ DONE......................................
B: ███████████████████████████████████████████████
C: ███████████ DONE................................

新请求 D:        waiting waiting waiting waiting...
```

GPU batch 越跑越“空”。

---

# 4. Orca 的第一个关键思想：Iteration-Level Scheduling

传统 request-level scheduling：

```text
Scheduler:
“把 A/B/C 交给 engine，直到整个请求结束再说。”
```

Orca：

```text
Scheduler:
“这次只执行一个 iteration。”
```

执行完以后控制权返回 scheduler：

```text
iteration t 完成
      ↓
检查谁 finished
      ↓
移除 finished requests
      ↓
从等待队列加入新 requests
      ↓
形成下一 iteration 的 batch
```

这就是：

> **Iteration-Level Scheduling**。

---

# 5. Continuous Batching 的直觉

今天很多框架更常说：

> **Continuous Batching / In-flight Batching**

最核心的思想就是：

```text
Batch 不是一次建立后固定到结束
```

而是每轮都可能改变：

```text
step 1:
[A B C]

step 2:
[A B C]

A finished

step 3:
[D B C]

C finished

step 4:
[D B E]
```

于是 GPU 的 batch slot 可以持续被新请求填充。

为什么叫 continuous？

因为请求可以：

```text
动态进入
动态退出
```

而不是一批一批“整批清空后才开始下一批”。

---

# 6. 为什么它对 LLM 吞吐特别重要？

Decode 经常是 memory-bound。

每个 request 每轮只有：

```text
1 query token
```

单个请求很难把 GPU 吃满。

所以需要把很多 request 的 decode 放在同一轮：

```text
Q_A
Q_B
Q_C
Q_D
...
```

共同形成更大的有效 batch。

如果静态 batch 里越来越多 request 已经结束：

```text
GPU 有效 batch size ↓
```

吞吐就会掉。

Continuous batching 则不断补充新请求：

```text
尽量维持较高并发
```

这对 decode 尤其关键。

---

# 7. 但 Iteration-Level Scheduling 引入了一个新问题

现在 batch 里可能是：

```text
Request A：已经处理 100 tokens
Request B：已经处理 300 tokens
Request C：已经处理 17 tokens
```

于是不同 request 的 Attention：

```text
A 需要 attend 100 个 KV
B 需要 attend 300 个 KV
C 需要 attend 17 个 KV
```

输入 shape 不一样。

传统 dense batch 喜欢：

```text
所有样本 shape 相同
```

现在 Attention 却天然是 ragged / variable-length。

这就引出了 Orca 第二个关键思想：

> **Selective Batching。**

---

# 8. Selective Batching 是什么？

Transformer 并不是每个算子对 variable-length batch 都同样困难。

例如某些线性层：

```text
hidden state
  ↓
Linear / MLP
```

当前 decode iteration 每个 request 都有一个 token hidden state。

这些数据很容易拼起来：

```text
[h_A,
 h_B,
 h_C]
```

然后一起做 GEMM。

但 Attention 的每个请求拥有不同历史 KV length：

```text
A → KV[0:100]
B → KV[0:300]
C → KV[0:17]
```

很难按传统固定 shape 的方式统一 batch。

Orca 的思路是：

> **只对适合 batching 的算子 batching；对不适合的 Attention 等操作采用不同处理。**

这就是 selective batching。

---

# 9. 为什么 Orca 时代还没有直接用 PagedAttention？

注意历史顺序：

```text
2022 Orca
  ↓
iteration-level scheduling
selective batching

2023 vLLM / PagedAttention
  ↓
更系统地解决动态 KV Cache 内存管理
```

Orca 已经发现：

> LLM serving 的 batch 是动态的，KV 长度也是动态的。

但 vLLM 后来进一步指出：

> 仅仅 scheduler 动态还不够，KV Cache 的显存分配也必须动态、高效。

于是技术演化非常自然：

```text
Orca：
谁这一轮一起算？

vLLM：
这些动态 request 的 KV 到底怎么存？
```

这就是为什么建议先读 Orca，再读 PagedAttention。

---

# 10. Request-Level vs Iteration-Level：画成时间线

## Request-Level

```text
Time ───────────────────────────────>

Batch 1: [A B C] ===================== finish
Batch 2:                              [D E F] ========

D/E/F 必须一直等 Batch 1
```

## Iteration-Level / Continuous

```text
step1 [A B C]
step2 [A B C]
step3 [D B C]   A done, D enters
step4 [D B E]   C done, E enters
step5 [D F E]   B done, F enters
```

系统从“批”为长期生命周期，变成：

> **请求才是生命周期；batch 只是每个 scheduling iteration 临时构造的一组工作。**

这是一个非常深的 serving abstraction 变化。

---

# 11. Continuous Batching 和 Dynamic Batching 是一回事吗？

相关，但不要完全等同。

传统 inference server 的 dynamic batching 常指：

```text
短暂等待几毫秒
把同时到达的一批独立请求拼在一起
执行一次 model forward
```

关键仍是：

```text
每个请求一次 forward 就结束
```

LLM continuous batching 则是：

```text
同一个 request 会经历很多 forward iteration
每一轮 batch membership 都可以改变
```

所以 LLM continuous batching 更强调：

> **跨 autoregressive iterations 动态重组 batch。**

---

# 12. Continuous Batching 和 Chunked Prefill 又不是一回事

这两个也经常混。

Continuous batching：

> batch 中谁参与下一轮？

Chunked Prefill：

> 一个很长的 prefill 是否一次算完，还是切成多个 token chunk？

例如：

```text
Prompt = 8000 tokens
```

普通 prefill：

```text
一次处理 8000
```

Chunked Prefill：

```text
2048
2048
2048
1856
```

这样可以把 decode 工作插在 prefill chunk 中间，控制 generation stall。

它是 Orca 思想之后进一步发展的 scheduler 技术。

---

# 13. 为什么 Prefill 会干扰 Decode？

这在 Orca 之后越来越重要。

Decode batch 正在稳定执行：

```text
step t      20 ms
step t+1    20 ms
step t+2    20 ms
```

突然来一个超长 prompt：

```text
Prefill 30K tokens
```

如果系统把整段 prefill 插进去：

```text
decode step
↓
LONG PREFILL ███████████████████
↓
decode step
```

之前正在生成的用户会感觉：

> “怎么下一个 token 突然等了很久？”

这叫 generation stall / inter-token latency 抖动。

后续的：

- chunked prefill；
- Sarathi-Serve；
- DistServe；
- P/D disaggregation；

都可以看成在继续解决这个问题。

---

# 14. Orca 在历史上的真正价值

不要只记：

```text
Orca 提高了 36.9× throughput
```

更值得记的是它改变了 serving scheduler 的基本单位。

以前：

```text
request
```

Orca：

```text
iteration
```

这让 scheduler 可以观察和控制：

```text
每一轮有哪些 request
谁结束了
谁可以加入
还有多少 KV memory
下一轮 batch 是什么
```

现代 vLLM / TensorRT-LLM / SGLang 的很多 scheduler 设计，都建立在类似的 iteration-level / in-flight 执行观念上。

---

# 15. Orca 论文的 Selective Batching 为什么后来没有成为最常听到的词？

因为后续系统把 variable-length attention 本身做得更强了。

技术继续发展：

```text
Orca
Selective Batching
    ↓
PagedAttention / ragged kernels
    ↓
FlashInfer / FlashAttention variants
    ↓
现代 fused variable-length attention backend
```

所以今天你可能更常听：

```text
continuous batching
paged KV cache
ragged attention
chunked prefill
```

而不是每天说 selective batching。

但历史上它解决的是一个非常真实的问题：

> **不同 generation position 的请求如何同时进入一个 iteration？**

---

# 16. Orca 和 vLLM 的关系

可以这样分工：

```text
Orca
│
└── Scheduling abstraction
    “每 iteration 动态组 batch”

vLLM
│
└── Memory abstraction
    “KV Cache 像 OS virtual memory 一样分页管理”
```

所以：

\[
\boxed{\text{Modern Serving} \approx
\text{Continuous Batching} + \text{Paged KV Management} + \text{Fast Kernels}}
\]

当然现代框架还远不止这些，但这是非常好的最小认知模型。

---

# 17. 一个公交车比喻

传统 static batch 像旅游大巴：

```text
A B C D 上车
↓
车必须完整走完整条路线
↓
大家一起结束
↓
下一批才能上
```

但每个 LLM request 的目的地长度不同：

```text
A 2站下车
B 30站下车
C 5站下车
```

静态 batch 仍把座位锁给已经下车的人，很浪费。

Orca 像城市公交：

```text
每到一站：
有人下车
有人上车
车继续开
```

这就是 Continuous Batching 最直观的感觉。

---

# 18. AI Infra 最值得记住的 8 个 Insight

1. **LLM request 是 multi-iteration workload，不适合传统 request-level batching。**
2. **输出长度未知，使 static batching 天然会产生 slot 浪费和 head-of-line blocking。**
3. **Iteration-level scheduling 把 scheduler 控制权拿回到每一轮 generation。**
4. **Continuous batching 的核心是 request 可以在 batch 中动态进入/退出。**
5. **Decode 的 memory-bound 特性让高并发 batch 特别重要。**
6. **不同 KV length 让 Attention 天然 ragged，不能只依赖传统 dense batching。**
7. **Orca 主要解决调度，vLLM 进一步解决 KV memory management。**
8. **后续 chunked prefill / P-D disaggregation 都是在解决 iteration-level scheduling 暴露出的更细粒度干扰问题。**

---

# 19. 读完后应该能回答

1. 为什么 static batching 对 autoregressive LLM 很差？
2. Request-level scheduling 和 iteration-level scheduling 有什么本质区别？
3. Continuous batching 为什么可以提升 decode throughput？
4. 为什么 variable KV length 会破坏传统 batching？
5. Selective batching 是为了解决什么？
6. Continuous batching、dynamic batching、chunked prefill 三者有什么区别？
7. Orca 和 PagedAttention/vLLM 各自解决哪一层问题？

---

## 主要参考资料

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022.
- vLLM / TensorRT-LLM / SGLang 的现代 serving 文档，用于理解 iteration-level / in-flight batching 思想的后续演化。
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, 2024（用于理解 chunked prefill 的后续问题）。
