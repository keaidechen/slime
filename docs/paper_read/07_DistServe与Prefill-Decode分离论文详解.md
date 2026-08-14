# 07｜DistServe 与 Prefill-Decode Disaggregation：为什么要把一次 LLM 推理拆到两组 GPU？

> **标题缩写与首次术语说明**：LLM = **Large Language Model（大语言模型）**；GPU = **Graphics Processing Unit（图形处理器）**；OSDI = **USENIX Symposium on Operating Systems Design and Implementation（USENIX 操作系统设计与实现大会）**；P/D 或 PD = **Prefill/Decode（预填充/解码）**；TTFT = **Time To First Token（首 Token 延迟）**；TPOT = **Time Per Output Token（平均每个输出 Token 的耗时）**；ITL = **Inter-Token Latency（相邻输出 Token 延迟）**；TBT = **Time Between Tokens（相邻 Token 时间间隔）**；SLO = **Service Level Objective（服务等级目标）**；P2P = **Peer-to-Peer（点到点通信）**；RDMA = **Remote Direct Memory Access（远程直接内存访问）**；NIXL = **NVIDIA Inference Xfer Library（NVIDIA 推理数据传输库，Xfer 即 transfer）**。**goodput（有效吞吐量）**只统计满足 SLO 的请求吞吐；**disaggregation（解耦/分离部署）**指把 Prefill 与 Decode 放到不同资源池。 另外：KV Cache = **Key-Value Cache（键值缓存）**；GEMM = **General Matrix-Matrix Multiplication（通用矩阵-矩阵乘法）**；HBM = **High Bandwidth Memory（高带宽内存）**；AI = **Artificial Intelligence（人工智能）**。

> 论文：Yinmin Zhong et al., **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**，OSDI 2024。
>
> 推荐前置：
> - `00_共享基础_GPU与LLM推理硬件基础.md`
> - `06_Orca与Continuous-Batching论文详解.md`
> - `02_PagedAttention-vLLM论文详解.md`
>
> 这篇论文要理解的核心不是“多了一次 KV 传输”，而是一个更大的系统设计变化：**Prefill 和 Decode 是两种完全不同的 workload，为什么必须强迫它们共享同一组 GPU、同一种并行策略和同一个 scheduler？**

---

# 1. 先把 Prefill / Decode 差异再强调一次

一个请求：

```text
Prompt: 8000 tokens
Output: 500 tokens
```

会经历：

```text
Prefill
一次性处理 8000 tokens
       ↓
产生第一枚 output token
       ↓
Decode
每轮处理 1 个新 token
重复约 499 次
```

它们虽然都执行 Transformer，但硬件行为非常不同。

## Prefill

```text
大量 token 同时算
大 GEMM
高 arithmetic intensity
更偏 compute-bound
```

## Decode

```text
每请求每轮只算一个 query token
要扫描长 KV Cache
GEMM 很瘦
更偏 memory-bandwidth-bound
```

于是：

\[
\boxed{\text{Prefill 和 Decode 是两种不同的 GPU workload}}
\]

这就是 DistServe 的起点。

---

# 2. 传统 serving 为什么把它们放一起？

最自然的实现是：

```text
一个 request 进某个 LLM engine
          ↓
同一组 GPU 做 Prefill
          ↓
同一组 GPU 接着做 Decode
          ↓
直到 request 完成
```

在 continuous batching 中可能变成：

```text
当前 running decodes
        +
新来的 prefills
        ↓
同一个 scheduler
        ↓
同一组 GPU
```

优点是简单：

- 模型只部署一份；
- Prefill 产生的 KV 已经在本 GPU；
- 不需要跨机器搬 KV。

但问题也由此出现。

---

# 3. 问题一：Prefill 会干扰 Decode

假设稳定 decode：

```text
token t      20 ms
token t+1    21 ms
token t+2    20 ms
```

突然新来一个 30K prompt。

如果它进行一个很大的 prefill：

```text
decode
  ↓
████████████ long prefill ████████████
  ↓
decode
```

老用户下一 token 的等待时间突然增加。

这直接影响：

> **TPOT / ITL**

也就是生成过程中 token 与 token 之间的延迟。

---

# 4. 问题二：Decode 也会影响 Prefill

反过来，如果 scheduler 为了保证老用户 token 流畅：

```text
优先 decode
```

新请求的 prefill 就会排队。

于是：

> **TTFT 变差。**

也就是用户提交 prompt 后，等第一枚 token 的时间变长。

所以 colocated engine 天然面临：

```text
保护 TTFT
   ↕ trade-off
保护 TPOT/ITL
```

---

# 5. 两个最重要的 Latency 指标

## TTFT — Time To First Token

```text
用户发请求
    ↓
等待
    ↓
第一枚 token 出现
```

近似主要反映：

- 排队；
- Prefill；
- 第一次 decode/sampling；

对交互体验非常重要。

## TPOT — Time Per Output Token

第一枚 token 之后：

```text
token1 → token2
```

平均需要多久。

也常和：

```text
ITL / Inter-Token Latency（相邻输出 Token 延迟）
TBT / Time Between Tokens（相邻 Token 时间间隔）
```

一起讨论。

用户体感：

```text
TTFT → “多久开始回答”
TPOT → “回答流得顺不顺”
```

DistServe 的重点就是同时满足二者 SLO。

---

# 6. 什么是 SLO？

SLO = **Service Level Objective（服务等级目标）**。

比如一个服务规定：

```text
90% requests:
TTFT < 500 ms
TPOT < 50 ms
```

这不是只看平均吞吐。

即使系统：

```text
tokens/s 很高
```

但大量请求 TTFT/TPOT 超标，生产上仍然不好用。

因此 DistServe 使用 **goodput** 的视角，而不只是 raw throughput。

---

# 7. Throughput 和 Goodput 区别是什么？

Throughput：

> 一秒处理多少请求 / token。

Goodput：

> 在满足服务延迟 SLO 的前提下，一秒真正“合格地”处理多少请求。

例如：

```text
System A
100 req/s
只有 50% 满足 SLO

System B
80 req/s
95% 满足 SLO
```

如果用户关心 SLO，B 可能更有价值。

DistServe 的系统优化目标就是：

> **最大化同时满足 TTFT 与 TPOT 约束的请求率。**

---

# 8. DistServe 的核心思想：Disaggregation

不要再让同一组 GPU 同时做两种 workload。

而是：

```text
Client Request
      │
      ▼
Prefill GPU Pool
      │
      │ 产生 KV Cache
      │
      ├──── KV transfer ────┐
      │                     │
      ▼                     ▼
first token            Decode GPU Pool
                            │
                            │ token by token
                            ▼
                         output
```

即：

```text
P nodes = Prefill only
D nodes = Decode only
```

这就是：

> **Prefill-Decode Disaggregation / PD Disaggregation（Prefill-Decode 分离部署）。**

---

# 9. 为什么分开之后 interference 消失？

Prefill pool 上：

```text
只有 prefills
```

不用为了保护正在 decode 的 request 而切碎工作。

Decode pool 上：

```text
只有 decodes
```

也不会突然插进一个 30K token 的大 prefill。

于是：

```text
TTFT 的资源
```

和：

```text
TPOT 的资源
```

变得更独立。

系统终于可以分别调优。

---

# 10. 更深一层：并行策略也可以分开了

假设一个超大模型需要多 GPU。

Prefill 喜欢：

```text
大 GEMM
高 compute throughput
```

可能更适合某种：

```text
Tensor Parallel / Pipeline Parallel
```

配置。

Decode 的瓶颈不同：

```text
KV Cache
memory bandwidth
per-token latency
```

其最佳 parallelism 配置可能不一样。

如果 colocated：

```text
Prefill 与 Decode 必须共用同一模型部署形态
```

这叫：

> resource allocation / parallelism coupling。

DistServe 分开后：

```text
Prefill Pool:
按 TTFT / compute 优化

Decode Pool:
按 TPOT / memory bandwidth 优化
```

这是它比“单纯避免干扰”更深的一层价值。

---

# 11. 但是最大的代价来了：KV Cache 必须搬

Prefill GPU 算出了：

```text
K/V for every layer
```

Decode GPU 没有这些数据。

所以必须：

```text
Prefill GPU HBM
      ↓
KV transfer
      ↓
Decode GPU HBM
```

这不是一个小 tensor。

长 context、大模型下 KV Cache 可以很大。

于是 P/D 分离制造了一个新的系统瓶颈：

> **KV Transfer。**

---

# 12. KV Transfer 为什么和网络拓扑强相关？

假设：

```text
P GPU 与 D GPU
```

在同一台机器，有 NVLink/NVSwitch：

```text
P GPU ==NVLink/NVSwitch== D GPU
```

可能很快。

但如果跨节点：

```text
Node A P GPU
    │
 InfiniBand / RoCE
    │
Node B D GPU
```

带宽、延迟、网络拥塞都必须考虑。

所以 P/D disaggregation 不是简单地说：

> “Prefill 一半机器，Decode 一半机器。”

而是一个：

```text
compute placement
+
network topology
+
KV transfer
+
request routing
```

的联合系统问题。

---

# 13. DistServe 为什么强调 placement？

因为拆得太远：

```text
Prefill 计算很快
但 KV 网络传 100 ms
```

可能把收益全部吃掉。

所以论文根据集群中的带宽层次，考虑：

- 哪些 P/D workers 放在一起；
- 哪些放跨节点；
- 不同 parallelism strategy 产生多少通信；
- KV transfer 会不会成为瓶颈。

这就是典型的 cluster-level inference optimization。

---

# 14. 为什么 P/D 比“Chunked Prefill”更激进？

Chunked Prefill：

```text
同一组 GPU
```

只是把：

```text
long prefill
```

切成：

```text
chunk 1
chunk 2
chunk 3
```

然后和 decode 交错：

```text
decode
prefill chunk
decode
prefill chunk
```

它在 scheduler 层面缓解 interference。

DistServe：

```text
Prefill 和 Decode 物理上就不在同一 GPU pool
```

直接从资源层面消除 interference。

所以：

```text
Chunked Prefill
= 时间维度上错峰

P/D Disaggregation（Prefill/Decode 分离部署）
= 空间/资源维度上拆开
```

---

# 15. P/D Disaggregation 是不是永远更好？

不是。

它引入了额外成本：

1. KV transfer；
2. P/D 两套 model replicas 可能增加权重占用；
3. routing 更复杂；
4. 两个 pool 的负载可能不平衡；
5. burst workload 下可能一个 pool 排长队，另一个 pool 空闲；
6. cluster topology 不好时网络成为瓶颈。

因此如果：

```text
请求很短
流量不大
单机就能轻松满足 SLO
```

强行 P/D 分离不一定值得。

系统设计永远是 trade-off。

---

# 16. DistServe 的论文结果怎么读？

论文报告，在其模型、工作负载和 SLO 设置下，DistServe 相比当时的 colocated serving baseline，可以服务显著更高的 SLO-satisfying request rate；论文 headline 包括最高约 7.4× 的请求承载能力提升，或在某些比较下支持显著更严格的 SLO。

不要记成：

> “P/D 永远提升 7.4×。”

应该记成：

> **当 TTFT 与 TPOT 都受到严格约束时，消除 Prefill/Decode interference 和 parallelism coupling 可能产生巨大的 goodput 收益。**

---

# 17. 2024 之后为什么 P/D Disaggregation 越来越重要？

模型和 workload 继续向几个方向发展：

```text
更长 context
更大的 KV Cache
更多 agent / reasoning workload
更重 prefill
更长 decode
更大的集群
```

于是：

```text
Prefill/Decode 不对称性
```

越来越明显。

现代 vLLM 和 SGLang 都已经提供 P/D disaggregation 相关能力；KV connector / transfer backend 也成为新的 Infra 抽象。

因此 DistServe 的价值已经不只是“一篇系统论文”，而是帮助理解现代推理集群架构的一条主线。

---

# 18. 现代 P/D 系统多了一层：KV Connector

可以把现代实现抽象为：

```text
Prefill Engine
     │
     │ put KV
     ▼
KV Transfer / KV Store / Connector
     │
     │ get KV
     ▼
Decode Engine
```

Connector 可能背后使用：

```text
GPU P2P
NVLink
RDMA
NIXL
Mooncake
LMCache
远端 KV Store
...
```

所以 KV Cache 已经从：

```text
“某一张 GPU 上的临时 tensor”
```

逐渐演化为：

> **一个可以跨 engine、跨 GPU、跨节点移动和复用的数据对象。**

这是理解 2025–2026 inference infra 很重要的变化。

---

# 19. 从 Orca 到 DistServe 的历史线

```text
Static Batching
      ↓
Orca
Iteration-Level Scheduling
      ↓
Continuous Batching
Prefill + Decode 共存
      ↓
发现长 Prefill 会造成 Decode stall
      ↓
Chunked Prefill / stall-free scheduling
      ↓
仍然共享同一组 GPU，资源/并行策略耦合
      ↓
DistServe
Prefill-Decode Disaggregation
```

这条线体现系统研究的典型方式：

> 每解决一个更粗粒度问题，就会暴露一个更细粒度的新瓶颈。

---

# 20. 一个餐厅比喻

传统 colocated engine 像：

```text
同一批厨师
既负责一次准备 100 道食材（Prefill）
又负责不断给已有客人续一小碟菜（Decode）
```

结果：

```text
大备菜一来
→ 所有续菜客人都等
```

如果只优先续菜：

```text
新客人的大单一直开不了工
```

DistServe：

```text
Prep Kitchen
专门处理大批食材准备

Serving Kitchen
专门快速连续出小碟
```

中间代价：

```text
准备好的食材必须从 Prep Kitchen 搬到 Serving Kitchen
```

这就是 KV Transfer。

---

# 21. AI Infra 最值得记住的 9 个 Insight

1. **Prefill 和 Decode 数学上属于同一个模型，硬件 workload 却完全不同。**
2. **Colocation 会产生 Prefill/Decode interference。**
3. **TTFT 与 TPOT 是两个不同用户体验目标。**
4. **Goodput 比 raw throughput 更适合描述 SLO-sensitive serving。**
5. **P/D 分离不仅消除 interference，也解耦 parallelism/resource allocation。**
6. **Disaggregation 把 KV transfer 变成一等公民。**
7. **网络拓扑开始直接影响 LLM inference scheduler。**
8. **Chunked Prefill 是时间复用；P/D disaggregation 是资源隔离。**
9. **现代 inference infra 正从“单 engine”走向“多 pool + KV fabric + router”。**

---

# 22. 读完后应该能回答

1. Prefill 和 Decode 为什么分别更偏 compute-bound / memory-bound？
2. TTFT 和 TPOT 分别描述什么体验？
3. 什么叫 Prefill-Decode interference？
4. DistServe 为什么不仅是“把两阶段放到两张卡”？
5. 为什么 P/D 分离后 KV transfer 成为新瓶颈？
6. Chunked Prefill 和 P/D disaggregation 有何区别？
7. 什么情况下 P/D 可能不值得？

---

## 主要参考资料

- Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*, OSDI 2024 / arXiv:2401.09670.
- vLLM 官方文档：Disaggregated Prefilling / KV Connectors（用于理解现代实现演化）。
- SGLang 官方文档：PD Disaggregation（用于理解现代实现演化）。
