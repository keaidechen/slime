# 从 AI Infra 视角理解 2024–2026 大模型结构演化
## DeepSeek / Kimi / Qwen / GLM：模型结构为什么这样变，它们具体如何实现，以及瓶颈如何迁移

> 版本：2026-09  
> 目标：不是简单罗列“谁用了什么结构”，而是从 **训练 FLOPs、HBM 容量与带宽、KV Cache、GPU 通信、小 GEMM、长上下文、流水线调度、推理串行性** 等 Infra 约束出发，解释这些结构为什么出现、具体如何工作，以及它们又制造了哪些新的系统瓶颈。

---

# 0. 核心结论：大模型结构演化，本质上是瓶颈不断迁移

如果只看模型论文，很容易得到这样的印象：

- DeepSeek 发明 MLA / DeepSeekMoE / DSA；
- Kimi 使用 KDA / AttnRes；
- Qwen 使用 GDN / QSA / Gated Residual；
- GLM 使用 DSA / IndexShare / Hybrid Attention。

这些看起来像一串相互独立的“算法创新”。

但从 AI Infra 的视角看，它们其实可以连成一条非常清晰的因果链：

```text
Dense Transformer
    │
    │  参数规模扩大 → 每个 token 都经过全部参数
    │  训练/推理 FLOPs 太高
    ▼
Sparse MoE
    │
    │  只激活少量 Expert，FLOPs 降下来了
    │  但 token 要跨 GPU dispatch/combine
    ▼
All-to-All / Load Balance / Small GEMM
    │
    │  → DeepEP / MoonEP / DualPipe / Grouped GEMM
    │
    │
    ├──────────────────────────────────────┐
    │                                      │
    ▼                                      ▼
KV Cache 爆炸                         Ultra-Sparse MoE
    │                                      │
    │  长上下文 decode                  Expert 越来越多
    │  HBM 容量/带宽不够                 每个 Expert token 越少
    ▼                                      ▼
MLA / GQA                           Tiny GEMM / Weight Movement
    │                                      │
    │  每 token KV bytes ↓                 │ FP8 / FP4 / fused MoE
    │                                      │ Grouped GEMM
    ▼                                      ▼
Attention 仍随 L 增长                网络与内存成为主角
    │
    ▼
Sparse Attention
    │
    │  Top-K Attention 本身便宜了
    │  但“找 Top-K”的 Indexer 开始变贵
    ▼
Compressed / Shared Indexer
    │
    ├─ DeepSeek CSA/HCA
    ├─ Qwen QSA
    └─ GLM IndexShare
    │
    ▼
Linear / Recurrent Attention
    │
    │  固定状态，sequence memory 近似 O(1)
    │  但精确 retrieval 变弱
    ▼
Hybrid Attention
Linear State + Sparse/Compressed Exact Retrieval
```

与此同时还有两条独立但正在汇合的路线：

```text
模型越来越深、越来越异构
        │
        ▼
Residual Stream 本身成为瓶颈
        │
        ├─ AttnRes
        ├─ mHC
        └─ Gated Residual

Autoregressive Decode 串行
        │
        ▼
MTP / Speculative Decoding
```

所以这两年的模型结构变化可以浓缩成一句话：

> **过去 Scaling 主要是在增加 FLOPs；现在 Scaling 越来越是在设计“哪些东西值得算、哪些东西值得搬、哪些信息应该留在 HBM、哪些历史应该压缩、哪些参数应该条件激活，以及哪些计算可以被通信隐藏”。**

---

# 1. 版本路线：先把四家的时间线放在同一张图里

下面只列 **Base Architecture 发生明显变化** 的版本；纯 post-training、RL 或能力更新不作为结构主线。

## 1.1 DeepSeek

```text
DeepSeek-V2
    │
    ├─ MLA
    └─ DeepSeekMoE
    │
    ▼
DeepSeek-V3
    │
    ├─ 继续 MLA + DeepSeekMoE
    ├─ Auxiliary-loss-free load balancing
    ├─ MTP
    ├─ FP8
    └─ DualPipe + 通信计算重叠
    │
    ▼
DeepSeek-V3.2
    │
    └─ DSA: DeepSeek Sparse Attention
    │
    ▼
DeepSeek-V4
    │
    ├─ CSA: Compressed Sparse Attention
    ├─ HCA: Heavily Compressed Attention
    ├─ Hybrid CSA/HCA
    ├─ mHC
    ├─ Muon
    ├─ FP4 QAT
    └─ 更激进的 MoE fused kernel / KV hierarchy
```

## 1.2 Kimi

```text
Kimi K2
    │
    ├─ 1T MoE
    ├─ MLA
    └─ MuonClip
    │
    ▼
Kimi Linear
    │
    └─ KDA + MLA Hybrid
    │
    ▼
Attention Residuals
    │
    └─ AttnRes / Block AttnRes
    │
    ▼
Kimi K3
    │
    ├─ 3 KDA : 1 Gated MLA
    ├─ Block AttnRes
    ├─ Stable LatentMoE
    │    └─ 896 routed experts, activate 16
    ├─ Quantile Balancing
    ├─ Per-Head Muon
    ├─ SiTU
    ├─ MXFP4 weights / MXFP8 activations
    └─ MoonEP / FlashKDA 等配套 Infra
```

## 1.3 Qwen

```text
Qwen3
    │
    ├─ Dense + MoE 两条产品线
    │
    ▼
Qwen3-Next
    │
    ├─ Gated DeltaNet
    ├─ Full Attention Hybrid
    ├─ Ultra-Sparse MoE
    └─ MTP
    │
    ▼
Qwen3.5 / 3.6 / 3.7 / 3.8
    │
    └─ Next 架构产品化
    │
    ▼
Qwen3.8-Flash-Next
    │
    ├─ GDN + QSA
    ├─ micro-block sparse index
    ├─ 4-branch Gated Residual
    ├─ N-gram Embedding
    │    └─ Host-memory offload + async prefetch
    └─ Muon
```

## 1.4 GLM

```text
GLM-4.5
    │
    └─ 大规模 MoE
    │
    ▼
GLM-5
    │
    └─ Sparse Attention / DSA 路线
    │
    ▼
GLM-5.2
    │
    ├─ IndexShare
    └─ 更强 MTP
    │
    ▼
GLM-5.3-Flash
    │
    ├─ Linear + Sparse Hybrid Attention
    └─ mHC
```

---

# 2. 第一场革命：从 Dense FFN 到 Sparse MoE

---

## 2.1 Dense Transformer 真正的问题是什么？

标准 Transformer block：

```text
x
│
├─ RMSNorm
│
├─ Attention
│
└─ Residual Add
    │
    ├─ RMSNorm
    │
    ├─ FFN / MLP
    │
    └─ Residual Add
```

对于 SwiGLU FFN，可以粗略写作：

```math
\operatorname{FFN}(x)
=
W_2\left(
\operatorname{SiLU}(W_gx)\odot W_ux
\right)
```

如果 hidden size 为 $d$，intermediate size 为 $d_{ff}$，那么 FFN 中是几个非常大的 GEMM。

Dense 模型的问题是：

> **所有 token 都必须经过全部 FFN 参数。**

如果总参数从：

```math
70B \rightarrow 400B \rightarrow 1T
```

那么在 Dense 架构中，模型容量和 active compute 基本绑在一起：

```math
P_{\text{total}}
\approx
P_{\text{active}}
```

因此：

```text
想增加模型容量
      ↓
增加 FFN width / layer
      ↓
每个 token FLOPs 一起涨
      ↓
训练成本和 decode 成本一起涨
```

---

# 3. MoE 到底怎么工作？

把一个 Dense FFN：

```text
                ┌─────────────┐
token ─────────►│  Dense FFN  │────────► output
                └─────────────┘
```

换成多个专家：

```text
                         ┌── Expert 0
                         ├── Expert 1
token ──► Router ────────┼── Expert 2
                         ├── ...
                         └── Expert N-1
```

Router 对 token $x$ 计算：

```math
s = W_r x
```

然后：

```math
p = \operatorname{softmax}(s)
```

选 Top-K experts：

```math
\mathcal{E}(x)=\operatorname{TopK}(p)
```

最终：

```math
y =
\sum_{e\in \mathcal{E}(x)}
p_e E_e(x)
```

如果：

- 总 expert 数 = 256；
- 每个 token 激活 8 个；

那么：

```text
Total Expert Capacity: 256
Active per token:        8
```

激活比例只有：

```math
8/256=3.125\%
```

这就是 MoE 的核心：

```math
P_{\text{total}}\gg P_{\text{active}}
```

---

# 4. DeepSeekMoE：为什么不是简单 Top-K MoE？

DeepSeek-V2 的另一个关键创新是 **DeepSeekMoE**。

它不是只做：

```text
N 个大 Expert → Top-K
```

而是采用：

1. **Fine-grained Expert Segmentation**
2. **Shared Expert Isolation**

---

## 4.1 Fine-grained Expert Segmentation

假设原本有一个 FFN：

```text
Expert A:
d → 8192 → d
```

可以进一步拆成多个小 expert：

```text
Expert A0: d → 2048 → d
Expert A1: d → 2048 → d
Expert A2: d → 2048 → d
Expert A3: d → 2048 → d
```

为什么要拆细？

因为 coarse expert 很容易出现：

```text
Expert 0 = 数学 + 一点代码 + 一点英语
Expert 1 = 代码 + 一点数学
```

多个 expert 学到大量重复知识。

Fine-grained expert 提高组合空间：

```text
一个 token 可以：
Expert 3  + Expert 27 + Expert 81 + Expert 104
```

相当于用多个“小专业模块”拼出当前 token 所需要的能力。

模型容量因此可以扩得更大，而 active FLOPs 不必同比增加。

---

## 4.2 Shared Expert 为什么存在？

如果所有能力都 Routed：

```text
“语言基本语法”
“常见词义”
“通用 Transformer feature”
```

这些每个 token 都可能需要。

让 router 每次重新路由这些公共能力是浪费。

DeepSeekMoE 因此把 expert 分为：

```text
               MoE
        ┌───────┴────────┐
        │                │
 Shared Experts     Routed Experts
 始终激活             Top-K 激活
```

输出可以理解成：

```math
y =
E_{\text{shared}}(x)
+
\sum_{e\in TopK}p_e E_e(x)
```

这样：

- shared expert 学通用能力；
- routed expert 更容易专业化。

---

# 5. MoE 为什么把模型问题变成 Infra 问题？

如果所有 experts 都在一张 GPU：

```text
Router
 ↓
Expert
```

问题不大。

但一个 1T MoE 不可能把所有专家都塞在一张 GPU。

实际系统是：

```text
GPU0: Expert 0~7
GPU1: Expert 8~15
GPU2: Expert 16~23
...
```

现在 GPU0 上的 token 可能需要 Expert 47：

```text
GPU0 token
    │
    ├─ Router: Expert 47
    │
    ▼
Network
    │
    ▼
GPU5
    │
Expert 47
```

一个 MoE layer 实际执行：

```text
tokens
  │
  ▼
Router
  │
  ▼
Dispatch All-to-All
  │
  ▼
Grouped Expert GEMM
  │
  ▼
Combine All-to-All
  │
  ▼
output
```

所以从 Dense 到 MoE 后：

```math
\boxed{\text{FLOPs bottleneck}}
```

被部分转换成：

```math
\boxed{
\text{Communication}
+
\text{Routing}
+
\text{Load Balance}
+
\text{Small GEMM}
}
```

---

# 6. 为什么 Ultra-Sparse MoE 会让 Grouped GEMM 越来越重要？

假设 batch 中有：

```math
N=4096
```

个 tokens。

256 experts、Top-8 后，每个 expert 平均收到：

```math
4096\times 8/256=128
```

个 token。

如果 experts 再增加到 896，而每 token 激活 16：

```math
4096\times16/896\approx73
```

平均每 expert 只有几十 token。

Dense FFN 原本是：

```text
一个巨大的 GEMM

████████████████████████████████
```

MoE 变成：

```text
Expert 0   ███
Expert 1      ██
Expert 2        ████
Expert 3            █
...
```

GPU Tensor Core 喜欢大的、规整的矩阵。

很多小 GEMM 会导致：

- kernel launch overhead；
- occupancy 下降；
- tile 填不满；
- Tensor Core utilization 下降；
- expert token 数不均导致 ragged shape。

所以需要 **Grouped GEMM**：

不是：

```text
launch GEMM expert0
launch GEMM expert1
launch GEMM expert2
...
```

而是：

```text
一次 grouped kernel

Expert 0 matrix
Expert 1 matrix
Expert 2 matrix
...
```

统一调度多个不同 M/N shape 的 GEMM。

因此：

```text
MoE sparsity ↑
      ↓
experts ↑
      ↓
tokens/expert ↓
      ↓
small/ragged GEMM ↑
      ↓
Grouped GEMM / fused MoE kernel 重要性 ↑
```

这正是为什么 CUTLASS、DeepGEMM、TileLang、Triton grouped GEMM、Transformer Engine grouped_gemm 等 kernel 成为现代 MoE Infra 的核心。

---

# 7. DeepSeek-V3：为什么会出现 Auxiliary-Loss-Free Load Balancing？

MoE 有一个严重问题：

```text
Router 学着学着
        ↓
大量 token 喜欢少数 Expert
        ↓
Expert 17 爆满
Expert 23 几乎没 token
```

这叫 Expert Load Imbalance。

传统做法加入 auxiliary balancing loss：

```math
L = L_{\text{LM}}+\lambda L_{\text{balance}}
```

让 expert usage 尽量均匀。

问题：

> balance objective 和 language modeling objective 不完全一致。

$\lambda$ 太大：

```text
Router 被强迫均匀
→ 破坏真正最优 expert specialization
```

$\lambda$ 太小：

```text
load balance 失效
```

DeepSeek-V3 使用 **auxiliary-loss-free load balancing**：

核心思想不是给主 loss 再加一项，而是给 router score 加动态 bias。

概念上：

```math
s'_e(x)=s_e(x)+b_e
```

如果某 expert 最近负载过高：

```math
b_e \downarrow
```

负载太低：

```math
b_e \uparrow
```

因此：

```text
LM gradient
   │
   └─ 继续专注模型能力

Load controller
   │
   └─ 独立调整 routing bias
```

从 Infra 角度看，这是极其重要的：

> **Load Balance 不只是模型质量问题，而是决定 EP 中是不是会出现 straggler。**

假设 63 张 GPU expert compute 都是 4 ms：

```text
GPU0-62: 4ms
```

但某张 GPU 因 expert 热点需要 8 ms：

```text
GPU63: 8ms
```

下一阶段同步必须等 GPU63。

整个 layer latency：

```math
T\approx 8ms
```

所以负载均衡直接决定集群 MFU。

---

# 8. 为什么 DeepSeek-V3 的 DualPipe 与 MoE 是一套设计？

Dense Transformer layer 大致：

```text
Attention GEMM
      ↓
FFN GEMM
```

MoE layer：

```text
Attention Compute
      ↓
Dispatch All-to-All
      ↓
Expert GEMM
      ↓
Combine All-to-All
```

通信占比突然变大。

如果完全串行：

```math
T =
T_{attn}
+T_{dispatch}
+T_{expert}
+T_{combine}
```

理论上 MoE 省下的 FLOPs 会部分被通信吃掉。

DualPipe 不是只减少普通 Pipeline Bubble，而是把：

```text
Pipeline A Forward
        +
Pipeline B Backward
```

交错，然后进一步把 operator 拆细：

```text
Attention
Dispatch
MLP
Combine
B-input
W-grad
```

让不同 stream 上的：

```text
communication
      ↕
compute
```

重叠。

理想目标：

```math
T_{\text{compute+comm}}
\rightarrow
\max(T_{\text{compute}},T_{\text{comm}})
```

而不是：

```math
T_{\text{compute}}+T_{\text{comm}}
```

所以要把 DeepSeek-V3 看成一个整体：

```text
DeepSeekMoE
    │
    └─ 制造巨大 All-to-All

Aux-loss-free routing
    │
    └─ 避免 expert straggler

DeepEP / network-aware routing
    │
    └─ 提高 All-to-All efficiency

DualPipe
    │
    └─ 把通信藏进计算

Grouped GEMM / custom kernel
    │
    └─ 解决 fragmented expert compute
```

这就是典型的 **Model–System Co-design**。

---

# 9. 第二场革命：为什么 MLA 会出现？

MoE 主要解决 FFN compute。

Inference 还有另一个独立瓶颈：

# KV Cache

---

## 9.1 标准 MHA 的 KV Cache

标准 Multi-Head Attention：

```math
Q=XW_Q,\quad
K=XW_K,\quad
V=XW_V
```

注意力：

```math
A=\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_h}}
\right)V
```

在 autoregressive decode 中，以前 token 的 K/V 不变，因此保存起来：

```text
token 1 ──► K1,V1 ┐
token 2 ──► K2,V2 │
token 3 ──► K3,V3 ├─ KV Cache
...               │
token L ──► KL,VL ┘
```

生成下一个 token：

```text
Q_new
  │
  ├── K1
  ├── K2
  ├── ...
  └── KL
```

KV Cache 大致：

```math
M_{KV}
\propto
L\times N_{layer}\times N_{KV-head}\times d_{head}
```

因此上下文：

```text
4K → 32K → 128K → 1M
```

KV Cache 线性爆炸。

更麻烦的是：

> decode 每生成一个 token，都要重新读取大量历史 K/V。

因此 decoder 常常不是 FLOPs-bound，而是：

```math
\boxed{\text{HBM bandwidth bound}}
```

---

# 10. GQA/MQA 先解决了一部分 KV 问题

MHA：

```text
Q head 0 → K0,V0
Q head 1 → K1,V1
...
Q head H → KH,VH
```

GQA：

```text
多个 Q heads
     │
     └─ 共用一个 KV head
```

MQA：

```text
所有 Q heads
   │
   └─ 共用一组 K,V
```

所以：

```math
N_{KV-head}\downarrow
```

KV Cache 直接下降。

但 DeepSeek-V2 进一步做了 MLA。

---

# 11. MLA：Multi-head Latent Attention 到底具体怎么实现？

MLA 的核心不是简单“少几个 KV heads”，而是：

> **把每个 token 的 K/V 信息先压缩成一个低维 latent，再把这个 latent 放入 Cache。**

传统：

```text
hidden x_t
  │
  ├── W_K ──► K_t ─┐
  │                 ├─ cache
  └── W_V ──► V_t ─┘
```

MLA：

```text
hidden x_t
    │
    └── Down Projection
            │
            ▼
         c_t^KV
        low-rank latent
            │
            └── cache
```

需要参与 attention 时：

```text
c_t^KV
  │
  ├─ Up Projection → K_t
  └─ Up Projection → V_t
```

可以近似理解为：

```math
c_t^{KV}=W_{DKV}x_t
```

```math
K_t=W_{UK}c_t^{KV}
```

```math
V_t=W_{UV}c_t^{KV}
```

真实 MLA 还会把 RoPE 部分和 non-RoPE 内容拆开，以保证低秩吸收和位置编码兼容，但心智模型就是：

```text
“大 KV”
    ↓
low-rank compression
    ↓
“小 latent KV”
```

DeepSeek-V2 报告中，相比此前架构 KV cache 显著下降。

---

# 12. 为什么 MLA 是一个典型 Infra trade-off？

MLA 并不是“免费压缩”。

它本质上是：

```math
\boxed{\text{更多 projection compute}}
```

换：

```math
\boxed{\text{更少 KV memory + HBM traffic}}
```

现代 GPU：

- Tensor Core FLOPs 增长非常快；
- HBM 容量很贵；
- HBM bandwidth 增长速度远慢于 Tensor Core FLOPs。

因此：

```text
多做一点 GEMM
      ↓
少搬很多 KV bytes
```

可能非常划算。

这也是模型设计从：

> 少算一点

转向：

> 少搬一点数据

的重要例子。

---

# 13. MLA 解决了 KV bytes，但没有解决 sequence length

这是一个非常关键的区别。

MLA 让：

```math
\text{bytes/token}
\downarrow
```

但 attention 仍然要与历史位置交互。

decode：

```math
T_{\text{attention}}
\propto L
```

prefill：

```math
T_{\text{attention}}
\propto L^2
```

所以 1M context 下，即使每个 token 的 KV 很小：

```text
1,000,000 × 小 KV
```

还是很大。

这推动了下一阶段：

# Sparse Attention

---

# 14. DeepSeek Sparse Attention：DSA 到底怎么流？

核心思想：

> 对一个 query，真正重要的历史 token 通常远少于整个上下文。

所以不要：

```text
Query
  │
  └── attend to 1,000,000 historical positions
```

而是：

```text
Query
  │
  ▼
Lightweight Indexer
  │
  ▼
score all candidate positions
  │
  ▼
Top-K positions
  │
  ▼
Sparse Attention only on Top-K
```

可写成：

```math
I(q,K)=\operatorname{TopK}
\left(
s(q,k_1),...,s(q,k_L)
\right)
```

然后：

```math
\operatorname{Attn}(q)
=
\operatorname{softmax}
\left(
qK_{I}^T
\right)V_I
```

如果：

```math
L=1,000,000
```

而：

```math
K=2048
```

真正高成本 attention 的工作量就从百万级候选减少到 2048。

---

# 15. DSA 的 Infra Pipeline

真实系统不是简单一个 sparse_matmul：

```text
Query
  │
  ▼
Indexer Projection / Logits GEMM
  │
  ▼
Top-K Kernel
  │
  ▼
logical token → KV page mapping
  │
  ▼
Gather sparse KV
  │
  ▼
Sparse MLA Attention Kernel
```

因此一整套新的 kernel 问题出现：

1. Indexer logits GEMM；
2. Top-K selection；
3. paged KV index transform；
4. irregular gather；
5. sparse attention；
6. decode/prefill 分别优化。

这也是为什么 DSA 模型在某些 GPU 架构上并不能“直接 fallback”到普通 attention：

它依赖的是一整套 kernel stack，而不是一个 Python 层面的 mask。

---

# 16. Sparse Attention 优化完之后，为什么 Indexer 自己会成为瓶颈？

总时间：

```math
T
=
T_{\text{index}}(L)
+
T_{\text{topk}}(L)
+
T_{\text{sparse-attn}}(K)
```

随着 K 变小：

```math
T_{\text{sparse-attn}}\downarrow
```

但 indexer 仍然可能扫描：

```math
L=1M
```

于是：

```text
以前：
Attention ██████████████████
Indexer   ██

优化后：
Attention ███
Indexer   ████████
```

这正是 Amdahl's Law。

因此 2026 的最新结构开始进一步优化：

> **不是只减少 Attention，而是减少“寻找该 Attention 哪些位置”的成本。**

---

# 17. DeepSeek-V4：CSA 是怎么实现的？

DeepSeek-V4 的关键变化之一是：

# Compressed Sparse Attention

DSA：

```text
1M raw KV
   │
Indexer over 1M
   │
Top-K
```

CSA 先做 sequence compression：

```text
Raw KV sequence

t0 t1 t2 t3 t4 t5 t6 t7 ...
│  │  │  │
└──┴──┴──┴── learned compression
      │
      ▼
     c0

          t4 t5 t6 t7 ...
          │  │  │  │
          └──┴──┴──┴──
                │
                ▼
               c1
```

如果 compression ratio：

```math
m=4
```

则：

```math
L
\rightarrow
L/m
```

1M token：

```math
1,000,000
\rightarrow
250,000
```

然后才运行 sparse index：

```text
compressed KV
      │
      ▼
Indexer
      │
      ▼
Top-K compressed entries
      │
      ▼
Sparse Attention
```

因此：

```math
T_{\text{index}}
```

也大约随 candidate 数量一起下降。

---

## 17.1 为什么 CSA 还保留 Sliding Window？

Sequence compression 会损失局部细节。

例如：

```text
"foo = bar + 1"
```

压成块级表示后，某些 token-level dependency 可能变弱。

因此 CSA 会把：

```text
Sparse selected compressed KV
           +
Recent sliding-window raw KV
```

合并。

概念上：

```math
KV_{\text{used}}
=
KV_{\text{sparse-compressed}}
\cup
KV_{\text{recent-raw}}
```

所以它同时保留：

- long-range approximate retrieval；
- local fine-grained dependency。

---

# 18. DeepSeek-V4：HCA 又是什么？

CSA：

```text
先适度压缩
再 sparse select
```

HCA：

```text
更激进压缩
但对压缩后的序列做 dense attention
```

例如：

```math
m'=128
```

1M tokens：

```math
1,000,000 / 128 \approx 7812
```

也就是说把非常长的历史压成几千个“memory slots”。

然后：

```text
Query
  │
  └── dense attention over ~8K compressed memory
```

为什么要同时存在 CSA 和 HCA？

因为它们有不同 inductive bias：

### CSA

擅长：

```text
从远处找少数精确信息
```

类似：

> search / retrieval

### HCA

擅长：

```text
对整个长历史建立广泛但粗粒度理解
```

类似：

> compressed global memory

因此 DeepSeek-V4 使用 hybrid：

```text
Layer 0: CSA
Layer 1: HCA
Layer 2: CSA
Layer 3: HCA
...
```

从 Infra 角度，它实际上是在构建：

```text
Recent Raw KV             → L1-like
Moderately Compressed KV  → sparse searchable memory
Heavily Compressed KV     → global summary memory
```

这是非常典型的 **Memory Hierarchy**。

---

# 19. 从 DSA → CSA/HCA 的路线说明了什么？

DeepSeek 的长上下文路线：

```text
MLA
 │
 │ 压每个 token 的 KV bytes
 ▼
DSA
 │
 │ 减少真正执行 Attention 的 token 数
 ▼
CSA
 │
 │ 连 Indexer 要扫描的候选数也减少
 ▼
HCA
 │
 │ 对超长历史建立更粗粒度全局 memory
 ▼
Hybrid Memory Hierarchy
```

这是一条非常完整的瓶颈迁移链：

```math
\text{KV bytes}
\rightarrow
\text{Attention FLOPs}
\rightarrow
\text{Indexer FLOPs}
\rightarrow
\text{Memory hierarchy design}
```

---

# 20. Linear Attention：为什么另一批模型干脆不保存全部历史？

Sparse Attention 的前提仍然是：

> 历史 token / KV 大体还存在，只是不全部读取。

Linear / Recurrent Attention 更激进：

> 把历史不断压进固定大小 state。

---

# 21. Linear Attention 的基本数学心智模型

标准 softmax attention：

```math
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}(QK^T)V
```

存在：

```math
QK^T
```

所以需要 token-to-token interaction。

一些 linear attention 通过 feature map：

```math
\phi(Q), \phi(K)
```

改写为：

```math
\phi(Q)
\left(
\phi(K)^T V
\right)
```

可以维护 recurrent state：

```math
S_t
=
S_{t-1}
+
\phi(k_t)v_t^T
```

query：

```math
y_t=\phi(q_t)S_t
```

于是历史被压成：

```math
S_t
```

而不是：

```text
K1,V1
K2,V2
...
Kt,Vt
```

相对于 sequence length：

```math
M_{\text{state}}=O(1)
```

decode 也不需要每次扫全部历史。

---

# 22. Gated DeltaNet / Delta Attention 解决什么？

简单 linear state：

```math
S_t=S_{t-1}+\Delta_t
```

最大的问题：

> 历史只会不断累积，模型不容易“忘掉/改写”旧信息。

DeltaNet 一类方法引入类似 delta-rule 的更新，让新的 key/value 可以修改 state。

Gated DeltaNet 再加入可学习 gate：

```math
S_t =
\alpha_t \odot S_{t-1}
+
\Delta_t
```

其中：

```math
0\le\alpha_t\le1
```

可以理解为：

```text
旧 state
  │
  ├─ gate 决定保留多少
  │
  ▼
更新后的 state
```

于是模型可以：

- 保留长期信息；
- 忘记无用状态；
- 更新已有 memory。

---

# 23. Kimi Delta Attention：KDA 具体做了什么？

Kimi 的 KDA 可以理解为对 DeltaNet 进一步细粒度化。

一个关键变化：

> decay/gating 不只是 scalar，而可以做到更细粒度的 channel-wise 控制。

概念上：

```math
S_t
=
D_t \odot S_{t-1}
+
\Delta_t
```

其中 $D_t$ 可以针对不同 channel 有不同衰减。

直觉：

```text
state channel 0: 保留 99%
state channel 1: 保留 30%
state channel 2: 几乎覆盖
...
```

这样 fixed-size state 的表达能力比单一 decay 更强。

---

# 24. KDA 为什么训练时还能并行？

看到 recurrence：

```math
S_t=f(S_{t-1},x_t)
```

很容易以为：

> 那训练必须 token-by-token 串行？

现代 linear attention 通常会使用 chunk-wise 并行算法：

```text
sequence:
┌────────┬────────┬────────┬────────┐
│ chunk0 │ chunk1 │ chunk2 │ chunk3 │
└────────┴────────┴────────┴────────┘
```

chunk 内：

- 用矩阵形式并行处理；

chunk 间：

- 只传播 compact recurrent state。

因此：

```text
token-level recurrence
      ↓
algebraic rearrangement
      ↓
chunk-level parallel scan / matrix kernel
```

训练仍可以有效利用 GPU。

Kimi 配套的 FlashKDA 本质上就是在解决：

- chunk kernel fusion；
- recurrent state update；
- decode fused kernel；
- projection + convolution 等操作融合。

所以 Linear Attention 能否“理论 O(L) → 实际快”，仍然高度依赖 kernel。

---

# 25. 为什么 Kimi K3 不是纯 KDA，而是 3:1 KDA + Gated MLA？

纯 recurrent state 最大的问题：

> **历史被压缩后，精确细节可能不可逆地丢失。**

例如：

```text
第 731,421 token:
UUID = 98B31C...
```

要模型在 20 万 token 以后精确复制这个 UUID：

fixed-size state 很难保证。

因此 Kimi K3 使用：

```text
KDA
KDA
KDA
Gated MLA
KDA
KDA
KDA
Gated MLA
...
```

即大约：

```math
3:1
```

### KDA 层

负责：

```text
cheap long-range state propagation
```

### Gated MLA 层

负责：

```text
full / exact token-level retrieval
```

所以 hybrid attention 的本质是：

```math
\boxed{
\text{Compressed State}
+
\text{Exact Retrieval}
}
```

而不是：

> “Linear Attention 已经替代 Attention。”

---

# 26. Gated MLA 中的 Gate 是什么意义？

普通 MLA：

```math
y=\operatorname{MLA}(x)
```

Gated MLA：

```math
g=\sigma(W_gx)
```

```math
y=g\odot\operatorname{MLA}(x)
```

它让模型对 attention 输出进行 data-dependent control：

```text
MLA output
    │
    × gate(x)
    │
    ▼
actual contribution
```

从深模型稳定性角度，它减少某些 attention output 直接无条件注入 residual stream。

在 inference 实现里，这个 gate projection 甚至可以与主 attention 路径并行执行：

```text
             ┌─ MLA main path ─────────┐
x ───────────┤                          ├─ multiply
             └─ gate projection ───────┘
```

这说明一个新的趋势：

> **架构设计时已经开始考虑 kernel launch / CUDA stream overlap 的可实现性。**

---

# 27. Qwen3-Next：为什么 Qwen 也走 GDN + Attention Hybrid？

Qwen3-Next 采用的核心思路与 Kimi Linear 收敛：

```text
大部分层:
Gated DeltaNet

周期性层:
Global Attention
```

原因完全一样：

### GDN

优点：

- recurrent state；
- long-context decode 成本低；
- KV memory 小。

缺点：

- exact recall 弱。

### Global Attention

优点：

- 精确 token interaction；
- retrieval 能力强。

缺点：

- 长上下文贵。

所以：

```math
\text{Hybrid}
=
\text{cheap memory}
+
\text{exact memory}
```

---

# 28. Qwen3.8-Flash-Next：QSA 具体比普通 Sparse Attention多做了什么？

QSA：Qwen Sparse Attention。

核心问题已经不是：

> “Sparse Attention 能不能减少 attention？”

而是：

> “Indexer 自己能不能足够便宜？”

QSA 采用两个重要思路：

## 28.1 Compressed Lightweight Indexer

不要用和主 attention 同等维度的表示做 indexing。

先把 Q/K 映射到更便宜的 index space：

```text
Q
│
└─ small projection ──► q_index

K
│
└─ small projection ──► k_index
```

然后：

```math
s_i=q_{index}k_{index,i}
```

indexer GEMM 本身变小。

---

## 28.2 Micro-block Granularity

不是：

```text
1M token
→ 对 1M 个位置逐 token score
```

而是：

```text
1M token
→ 分成 micro blocks
→ block-level score
→ 选择重要 block
→ 在 block 内精确 attention
```

假设每 block 32 token：

```math
1M/32
\approx31250
```

Indexer candidate 数大幅下降。

这就是：

```text
Token-level Sparse:
Indexer O(L)

Micro-block Sparse:
Indexer O(L/B)
```

其中 $B$ 是 block size。

所以 QSA 优化的是：

```math
\boxed{\text{Sparse Attention 的 Index Cost}}
```

---

# 29. GLM-5.2 IndexShare：为什么“共享 Indexer”也很重要？

如果一个 Sparse Attention layer 有自己的 indexer：

```text
Layer 1 → index over 1M
Layer 2 → index over 1M
Layer 3 → index over 1M
Layer 4 → index over 1M
```

四层就做四遍。

GLM-5.2 的 IndexShare：

```text
        Shared Indexer
             │
       index result
      ┌──────┼──────┐
      ▼      ▼      ▼
    L1     L2      L3/L4...
```

官方设计是让一组 sparse attention layers 复用 indexer。

所以：

```math
T_{index}
```

按 layer 数进一步摊薄。

这说明 Sparse Attention 已进入一个新的工程阶段：

```text
第一代：
Attention sparse 化

第二代：
Indexer 本身压缩/共享/分块
```

---

# 30. GLM-5.3-Flash：为什么又走 Linear + Sparse？

GLM 最新路线与 Kimi/Qwen 收敛：

```text
Linear Attention
      +
Sparse Attention
```

原因不是“大家互相抄结构”，而是数学约束决定的：

### Linear state

解决：

```math
\text{long-context memory/compute}
```

### Sparse exact retrieval

解决：

```math
\text{information loss}
```

因此可以把这类架构理解成：

```text
                         Long Context Memory

                 ┌────────────┴────────────┐
                 │                         │
          Compressed State             Exact Store
                 │                         │
        KDA / GDN / Linear        MLA / Sparse Attention
                 │                         │
                 └────────────┬────────────┘
                              │
                           Query
```

---

# 31. 现代 Attention 其实正在变成“多级 Cache”

如果用 CPU Memory Hierarchy 类比：

| 模型结构 | 类比 |
|---|---|
| Sliding Window | L1 Cache |
| Linear recurrent state | compressed cache |
| HCA | global compressed memory |
| CSA/QSA/DSA | indexed sparse memory |
| Full MLA | exact main memory |
| External RAG | disk / remote store |

这不是严格硬件一一对应，但作为 Infra 心智模型非常有用。

未来长上下文模型越来越像：

```text
Query
  │
  ├─ 最近信息？──────────────► Sliding Window
  │
  ├─ 全局状态？──────────────► Linear / HCA state
  │
  ├─ 精确远程 token？────────► Sparse Retrieval
  │
  └─ 模型外知识？────────────► RAG / Tool
```

---

# 32. 第三场革命：Residual Stream 为什么突然开始被重新设计？

标准 PreNorm Transformer：

```math
x_{l+1}
=
x_l
+
F_l(\operatorname{Norm}(x_l))
```

每层 output 都固定系数 1 加入 residual：

```text
x0
 │
 + f1
 │
 + f2
 │
 + f3
 │
 + f4
 │
 ...
```

当网络更深、更异构时：

```text
Linear Attention
MoE
Sparse Attention
MoE
Linear Attention
...
```

Residual 开始暴露两个问题：

1. hidden magnitude 随深度增长；
2. 早期 layer 信息被大量后续 output “稀释”。

因此“深度方向的信息流”开始成为独立建模对象。

---

# 33. Kimi AttnRes：把层深方向也变成 Attention

标准 residual：

```math
h_l = h_{l-1}+f_l(h_{l-1})
```

虽然数学展开后包含所有过去层：

```math
h_l=x_0+\sum_{i<l}f_i
```

但所有信息以固定累加方式混合。

AttnRes 的核心是：

> 当前层可以直接“查询”以前不同层的 representation。

概念上：

```math
h_l
=
\sum_{i<l}
\alpha_{i\rightarrow l}v_i
```

其中：

```math
\alpha_{i\rightarrow l}
=
\operatorname{softmax}
(q_l^Tk_i)
```

因此：

```text
Layer 12
   │
   ├── 5%  read Layer 2
   ├── 10% read Layer 5
   ├── 60% read Layer 9
   └── 25% read Layer 11
```

这相当于把：

> depth dimension

也变成一个 retrieval problem。

---

# 34. Full AttnRes 为什么很贵？

如果第 $l$ 层都要存所有过去层：

```text
v0
v1
v2
...
v(l-1)
```

activation memory 和 pipeline communication 都会上升。

特别是 Pipeline Parallelism：

标准 pipeline stage 只需要给下一 stage 发：

```text
latest hidden state
```

AttnRes 可能需要访问更老的 depth states。

这对 distributed training 很不友好。

---

# 35. Block AttnRes 如何把它工程化？

Kimi 提出 Block AttnRes：

不是存每一层，而是把多层打成 block。

例如：

```text
Block 0: Layer 0~3
Block 1: Layer 4~7
Block 2: Layer 8~11
```

保留：

```text
Block summary 0
Block summary 1
Block summary 2
```

当前层只 attend：

- block-level summaries；
- 当前 block 内必要状态。

于是：

```text
Full AttnRes:
O(number of previous layers)

Block AttnRes:
O(number of previous blocks)
```

并且可以对 pipeline communication 做 cache-based 优化。

所以 Kimi 的贡献不只是提出一个 depth attention 数学公式，而是同时回答：

> **它在 Pipeline Parallel training 里怎么存、怎么传、怎么不让 activation 爆炸。**

---

# 36. DeepSeek mHC：为什么与 AttnRes 路线不同但目标相同？

mHC = Manifold-Constrained Hyper-Connections。

基础 Hyper-Connection 先把 residual stream 从一条拓宽成多条：

传统：

```text
residual x ∈ R^d
```

HC：

```text
X ∈ R^(n_hc × d)

branch0
branch1
branch2
branch3
```

每一层可以动态：

- 从多个 residual branch 读取；
- 写回多个 branch。

概念上：

```math
X_{l+1}
=
A_lX_l
+
B_lF(C_lX_l)
```

其中：

- $C_l$：layer input 如何从 residual branches 读；
- $B_l$：layer output 如何写回；
- $A_l$：原 residual state 如何混合传播。

问题：

动态 mixing matrix 很容易造成：

```math
\|X_l\|
```

随深度爆炸或衰减。

mHC 的核心是对 mixing map 做约束，例如让关键 mixing matrix 位于 **doubly stochastic matrices** 组成的 Birkhoff polytope 上。

直觉：

```text
不是允许任意:
[10  -3  5 ... ]

而限制成稳定的 probability-like mixing:
row sum = 1
column sum = 1
non-negative
```

这让 signal propagation 更可控。

---

# 37. mHC 的 Infra 代价是什么？

如果 residual width：

```math
n_{hc}=4
```

则边界 residual state：

```math
d
\rightarrow
4d
```

这意味着：

- activation memory ↑；
- residual read/write bandwidth ↑；
- PP communication ↑；
- checkpoint/recompute 复杂度 ↑。

所以 DeepSeek-V4 配套提到：

- recomputation；
- fused kernels；
- tensor-level checkpoint；
- 更细粒度 activation 管理。

这再次说明：

> **架构创新本身会制造新的 Infra 成本，而真正可用的结构必须同时给出系统实现。**

---

# 38. Qwen Gated Residual：4 路 residual stream 是怎么工作的？

Qwen3.8-Flash-Next 的 Gated Residual 也把 residual stream 扩成多 branch：

```text
R0
R1
R2
R3
```

每层通过 data-dependent gate 决定：

```text
读哪几个 branch？
每个 dimension 读多少？
输出写回哪些 branch？
```

与简单 HC 不同，gate 是 elementwise/data-dependent。

概念上：

```math
x_l=
\sum_i g^{read}_{l,i}\odot R_i
```

layer output：

```math
u_l=F_l(x_l)
```

写回：

```math
R_i'
=
R_i
+
g^{write}_{l,i}\odot u_l
```

因此 residual 不再是一条固定高速公路，而变成：

```text
             residual network

R0 ────────────────┐
R1 ────────┐       │
R2 ────┐   │       ├─ dynamic read → Layer
R3 ────┴───┴───────┘
                      │
                dynamic write
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
         R0          R1          R2/R3
```

从 Infra 角度，Qwen 还明确考虑 residual state 可以低精度存储，例如 FP8，以降低额外 memory traffic。

---

# 39. AttnRes / mHC / Gated Residual 应该如何统一理解？

虽然数学不同，它们都在解决：

```math
\boxed{\text{Depth-wise Information Routing}}
```

过去：

```text
模型扩展 = width + layers
```

现在：

```text
模型很深
+
每层 operator 异构
+
每层功能不同
```

所以：

> 层之间怎么传播信息，也值得被学习。

可以类比 GPU cluster：

```text
以前：
GPU 间只有简单链路就够

规模大后：
需要 NVSwitch / topology-aware routing
```

深层 Transformer 同理：

```text
以前：
x + F(x) 就够

规模大后：
需要 learned depth routing
```

---

# 40. Kimi K3：Stable LatentMoE 到底新在哪里？

Kimi K3 继续把 MoE 稀疏度推到非常高：

- 896 routed experts；
- 每 token 激活 16；
- 再加 shared experts；
- 总参数约 2.8T；
- active parameter 远小于 total parameter。

结构上：

```text
token
  │
  ├─ Shared Expert 0 ──────────┐
  ├─ Shared Expert 1 ──────────┤
  │                            │
  └─ Router                    │
      │                        │
      ├─ Expert e1 ──┐         │
      ├─ Expert e2 ──┤         │
      ├─ ...         ├─ sum ───┤
      └─ Expert e16 ─┘         │
                               ▼
                            output
```

“LatentMoE”本质是在更低/更紧凑的 expert representation 中做超大规模 expert specialization，从而把 total capacity 推高，同时控制 active compute。

---

# 41. Kimi K3 的 Quantile Balancing 在解决什么？

极端 MoE 下：

```text
896 experts
```

Router balance 变得非常敏感。

如果简单用 heuristic：

```text
expert load > target
bias -= eta

expert load < target
bias += eta
```

会面临：

- learning rate 敏感；
- load 分布长尾；
- 训练阶段动态变化大。

Quantile Balancing 的思路是：

> 根据 router score / load distribution 的分位结构做专家分配或 bias 调整。

本质上把：

```text
“专家是否过热”
```

从绝对阈值问题变成：

```text
“它在当前整个专家分布里处于哪个 quantile”
```

在 896 experts 这种规模下会更稳健。

从 Infra 角度，还是为了避免：

```math
\boxed{\text{Expert Straggler}}
```

---

# 42. SiTU / activation 为什么在极端 MoE 下也重要？

Expert MLP 内部 activation 会决定：

- 数值范围；
- quantization difficulty；
- outlier；
- low-precision kernel efficiency。

Kimi K3 使用 SiTU 类有界/更可控 activation 设计，配合 MXFP4/MXFP8。

原因是 FP4 很怕：

```text
极端 outlier
```

如果 activation range 很宽：

```text
大部分值挤在很小区间
少数 outlier 决定 scale
```

4-bit quantization error 会显著增大。

所以：

```text
activation function design
        │
        ▼
numerical distribution
        │
        ▼
FP4 quantization quality
        │
        ▼
inference/training kernel efficiency
```

这已经明显属于 Numeric–Architecture–Kernel Co-design。

---

# 43. FP8 / FP4：为什么越来越重要？

参数从 BF16 → FP8 → FP4：

```text
BF16: 16 bit
FP8 :  8 bit
FP4 :  4 bit
```

如果只看 storage：

```math
M_{FP8}\approx\frac12M_{BF16}
```

```math
M_{FP4}\approx\frac14M_{BF16}
```

但更重要的是：

> **每次从 HBM 搬权重的数据量下降。**

在 decode / Ultra-Sparse MoE 中：

expert GEMM 可能已经不再是纯 compute-bound。

尤其每 expert token 很少时：

```text
load expert weights
      ↓
只算几十个 token
```

Arithmetic Intensity 下降。

于是 weight bandwidth 成为瓶颈。

所以 FP4 可以同时：

- 减少模型显存；
- 减少 HBM traffic；
- 提高 Tensor Core throughput；
- 提高 expert weight residency。

---

# 44. 为什么 Quantization-Aware Training 比部署后量化更重要？

Post-Training Quantization：

```text
先 BF16 训练完
      ↓
部署时硬压成 FP4
```

模型从来没见过 quantization error。

QAT：

```text
训练阶段
  │
  ├─ simulate FP4 weights
  ├─ simulate scale / rounding
  └─ 模型主动适应 quantization noise
```

因此：

```math
W_{quant}=Q(W)
```

forward 里直接让模型适应 $Q(W)$。

Kimi K3 从较早 post-training/SFT 阶段就开始 QAT，说明：

> **未来 serving precision 会越来越成为训练目标的一部分。**

---

# 45. Qwen 的 N-gram Embedding：为什么是非常“系统型”的架构创新？

标准 token embedding：

```math
e_t=E[token_t]
```

Qwen3.8-Next 增加基于 local n-gram 的 lookup：

```text
... token(t-2), token(t-1), token(t)
                │
                ▼
          deterministic hash/index
                │
                ▼
         N-gram Embedding Table
                │
                ▼
         extra representation
```

关键是：

> Embedding lookup 几乎没有大 GEMM。

所以可以扩一个很大的 table，而每 token 只读少量 rows。

---

# 46. 为什么 N-gram Embedding 可以放 CPU？

假设 table 很大：

```text
51B embedding parameters
```

全部放 GPU HBM 非常昂贵。

但每 token 只查：

```text
少数几个 embedding vectors
```

于是可以：

```text
CPU RAM
 │
 │ async lookup/prefetch
 ▼
Pinned / transfer buffer
 │
 │ PCIe
 ▼
GPU
```

并且提前知道未来 batch 的 n-gram index，所以可以异步：

```text
GPU:
████ current layer compute ████

CPU/PCIe:
    ███ prefetch next ngram ███
```

理想情况下把 transfer 藏在 compute 下。

这意味着：

```math
\text{Model Capacity}
```

开始可以来自：

```math
P_{GPU}
+
P_{CPU-memory}
```

而不是所有参数都在 GPU。

---

# 47. 这为什么很像推荐系统？

推荐系统早就有：

```text
GPU:
Dense tower / MLP

CPU / huge memory:
Embedding tables
```

因为 embedding：

- 参数量巨大；
- 单次访问稀疏；
- lookup-bound 而不是 GEMM-bound。

LLM 的 N-gram memory 也开始出现类似性质。

所以未来可能越来越常见：

```text
GPU resident:
active dense / expert compute

CPU resident:
large sparse lookup memory

NVMe:
cold prefix / long-term cache
```

模型本身会变成多级内存系统。

---

# 48. Muon：为什么 optimizer 也是 Infra 话题？

AdamW：

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
```

```math
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
```

```math
W\leftarrow
W-\eta\frac{m_t}{\sqrt{v_t}+\epsilon}
```

它基本是 elementwise normalization。

Muon 更关注 matrix-shaped parameter 的整体 geometry。

简化理解：

```text
gradient matrix G
      │
momentum M
      │
orthogonalization
      │
near UV^T update direction
      │
update W
```

用 Newton-Schulz iteration 近似矩阵正交化。

---

# 49. 为什么更快收敛就是 Infra 优化？

假设两个 optimizer：

```text
AdamW:
目标 loss 需要 20T tokens

Muon:
目标 loss 需要 16T tokens
```

少 4T tokens。

如果训练是几千 GPU：

```text
节省的不是 5% kernel time
而可能是几十万/百万 GPU-hours
```

所以：

```math
\boxed{\text{Token Efficiency}=\text{Compute Efficiency}}
```

这就是为什么 Kimi K2/K3、DeepSeek-V4、Qwen3.8 都开始认真研究 Muon。

---

# 50. Kimi 的 MuonClip：为什么 optimizer 改了又会制造新的稳定性问题？

Kimi K2 把 Muon 放大到 1T MoE 时发现：

```text
attention logits
      ↑
      ↑
      ↑
可能出现极端增长
```

Attention：

```math
S=\frac{QK^T}{\sqrt d}
```

如果 Q/K norm 不受控：

```math
|S|\rightarrow very\ large
```

softmax：

```text
[1000, 2, -3, ...]
↓
几乎 one-hot
```

可能导致训练不稳定。

MuonClip / QK-Clip 的思想：

监测 attention max logits。

如果超过阈值：

```text
rescale Q/K projection weights
```

把 logit 拉回安全范围。

所以：

```text
Muon
 │
 └─ token efficiency ↑
       │
       └─ attention scale instability
              │
              └─ MuonClip
```

仍然是典型“优化一个瓶颈，暴露下一个”。

---

# 51. MTP：为什么 Multi-Token Prediction 越来越常见？

传统 next-token prediction：

```math
x_1,...,x_t
\rightarrow
x_{t+1}
```

MTP 增加辅助 head：

```text
shared hidden
    │
    ├─ predict t+1
    ├─ predict t+2
    ├─ predict t+3
    └─ ...
```

训练目标：

```math
L
=
L_{t+1}
+
\lambda_2L_{t+2}
+
\lambda_3L_{t+3}
+\cdots
```

它既可能增强 representation，也可以为 speculative decoding 提供 draft 信息。

---

# 52. 为什么 Autoregressive Decode 天生是 Infra 的痛点？

生成：

```text
token 1
  ↓
whole model
  ↓
token 2
  ↓
whole model
  ↓
token 3
```

token $t+1$ 必须等 token $t$。

所以即使单次模型 forward 已经非常快：

```math
\text{serial dependency}
```

仍然不可消失。

MTP/speculative decoding 的思路：

```text
一次 propose:
t+1, t+2, t+3, t+4

主模型 verify
      │
      └─ 一次接受多个 token
```

如果 average accepted length：

```math
a>1
```

完整 sequential model steps 近似减少：

```math
1/a
```

这就是 Train–Serve Co-design：

> **训练阶段增加结构，就是为了推理阶段减少串行 decode steps。**

---

# 53. 为什么 GLM-5.2 还继续优化 MTP？

MTP 不是：

```text
有一个 head → 就一定加速
```

真正服务收益取决于：

```math
\text{acceptance rate / acceptance length}
```

如果 draft 4 tokens：

```text
a b c d
```

主模型经常只接受第一个：

```text
accept a
reject b c d
```

加速很小。

因此后续模型会继续优化：

- MTP hidden representation；
- target dependency；
- draft quality；
- verify pipeline。

这说明 MTP 已从“训练 regularizer”演化成服务架构的一部分。

---

# 54. 现在把 Kimi K3 的整个 block 流程串起来

一个简化 Kimi K3 sequence：

```text
Input
  │
  ▼
Block AttnRes read
  │
  ▼
KDA
  │
  ▼
Residual / AttnRes write
  │
  ▼
Stable LatentMoE
  │
  ▼

Block AttnRes read
  │
  ▼
KDA
  │
  ▼
MoE
  │

Block AttnRes read
  │
  ▼
KDA
  │
  ▼
MoE
  │

Block AttnRes read
  │
  ▼
Gated MLA
  │
  ▼
MoE
  │
  └──── repeat
```

三个维度分别被不同结构处理：

```text
Sequence Dimension
    │
    ├─ KDA: compressed recurrent state
    └─ Gated MLA: exact global retrieval

Depth Dimension
    │
    └─ Block AttnRes

Channel / Parameter Dimension
    │
    └─ Stable LatentMoE
```

这就是 Kimi K3 最值得学习的地方：

> **它不是只沿一个 axis 做 scaling，而是分别设计 sequence、depth、width 三个维度。**

---

# 55. Qwen3.8-Next 也可以按三个维度拆

```text
Sequence
   │
   ├─ GDN
   └─ QSA

Depth
   │
   └─ 4-branch Gated Residual

Capacity
   │
   ├─ Ultra-Sparse MoE
   └─ N-gram Embedding
```

再加：

```text
Optimization
   │
   └─ Muon
```

所以 Qwen3.8-Next 的架构已经明显不像：

```text
Attention + FFN
```

而像：

```text
Token Memory System
+
Depth Routing System
+
Conditional Compute System
+
External Sparse Parameter Memory
```

---

# 56. DeepSeek-V4 同样可以这样拆

```text
Sequence
   │
   ├─ CSA
   └─ HCA

Depth
   │
   └─ mHC

Width / Capacity
   │
   └─ DeepSeekMoE

Decode
   │
   └─ MTP

Numerics
   │
   └─ FP4 / QAT

Optimizer
   │
   └─ Muon
```

DeepSeek 路线尤其体现：

> **每一轮模型架构变化都直接与 Infra kernel / storage / communication 配套。**

---

# 57. 四家路线的“风格”差异

## 57.1 DeepSeek：Hardware-aware Full-stack Co-design

DeepSeek 最鲜明的路线：

```text
模型结构
 ↕
训练框架
 ↕
GPU kernel
 ↕
通信网络
 ↕
Serving KV system
```

典型例子：

- DeepSeekMoE ↔ DeepEP；
- MoE communication ↔ DualPipe；
- MLA ↔ FlashMLA；
- DSA ↔ sparse index kernel；
- V4 CSA/HCA ↔ two-stage contextual parallelism；
- mHC ↔ recompute/fused kernel；
- MoE ↔ FP4 fused expert kernel。

---

## 57.2 Kimi：把 Scaling 分解为 Sequence / Depth / Width

Kimi K3 的结构尤其“正交”：

```text
Sequence:
KDA + MLA

Depth:
AttnRes

Width:
LatentMoE
```

再用：

```text
MoonEP / FlashKDA / MXFP4
```

把系统成本压下来。

---

## 57.3 Qwen：高度 Serving-Oriented 的异构架构

Qwen 很明显同时考虑：

- cloud serving；
- cost-sensitive API；
- host memory；
- long context；
- multimodal；
- Dense/MoE 不同部署 envelope。

N-gram embedding offload 是非常典型的“模型结构直接为 memory tier 设计”。

---

## 57.4 GLM：快速吸收主线结构并优化实际瓶颈

GLM 的路线更像：

```text
MoE
↓
DSA
↓
IndexShare
↓
Linear + Sparse Hybrid
↓
mHC
```

IndexShare 特别能体现系统思维：

> 主 sparse attention 已经优化了，那就继续优化 indexer。

---

# 58. 为什么 Dense 不会彻底被 MoE 淘汰？

Ultra-Sparse MoE 的优势需要：

- 大 batch；
- 高效 All-to-All；
- 好的 EP topology；
- grouped GEMM；
- expert balance；
- 较强 kernel support。

在单机或小规模部署：

Dense：

```text
没有 Router
没有 Dispatch
没有 Combine
没有 Expert imbalance
GEMM 大而规整
```

可能更容易达到高实际吞吐。

所以未来更可能：

```text
Cloud Frontier:
Ultra-Sparse MoE

Small Cluster:
Dense / moderate MoE

Local:
Dense + quantization

Long Context API:
Hybrid linear/sparse

Huge memory:
host/offload hierarchy
```

不是一个架构统治全部硬件环境。

---

# 59. 一条最重要的 Infra 规律：Arithmetic Intensity 在改变模型设计

Roofline 模型：

```math
\text{Performance}
=
\min(
\text{Peak FLOPs},
\text{Bandwidth}\times \text{Arithmetic Intensity}
)
```

Arithmetic Intensity：

```math
AI=
\frac{\text{FLOPs}}{\text{Bytes moved}}
```

Dense large GEMM：

```math
AI
```

通常高。

但现代结构不断产生低 AI 操作：

- MoE small expert GEMM；
- KV Cache read；
- sparse gather；
- indexer；
- residual branch；
- embedding lookup。

因此越来越多模型变成：

```math
\boxed{\text{Memory/Communication Bound}}
```

而不是 Compute Bound。

这解释了为什么现代架构创新越来越关注：

```text
少搬
少存
少同步
少扫描
多 overlap
```

而不只是：

```text
少 FLOPs
```

---

# 60. 把你最近学习的 Infra 知识全部串起来

你之前学的内容可以直接挂到这条模型路线：

```text
MoE
 │
 ├─ Expert Parallel
 │     │
 │     ├─ NCCL
 │     ├─ All-to-All
 │     ├─ NVLink / NVSwitch
 │     └─ InfiniBand
 │
 ├─ Expert compute
 │     │
 │     ├─ Grouped GEMM
 │     ├─ CUTLASS
 │     ├─ Transformer Engine
 │     ├─ Triton
 │     └─ TileLang
 │
 └─ overlap
       │
       ├─ CUDA Stream
       ├─ SM partition
       └─ DualPipe

Long Context
 │
 ├─ KV Cache
 │     │
 │     ├─ PagedAttention
 │     ├─ MLA
 │     └─ Prefix Cache
 │
 ├─ Sparse Attention
 │     │
 │     ├─ Top-K
 │     ├─ Gather
 │     └─ Sparse kernel
 │
 └─ Linear Attention
       │
       ├─ recurrent state
       ├─ chunk kernel
       └─ prefix-cache semantics

Serving
 │
 ├─ Continuous Batching
 ├─ Prefill/Decode Disaggregation
 ├─ Speculative Decoding
 └─ CPU/GPU/NVMe Memory Tiering
```

以后你看一个 kernel，可以先问：

> **到底是哪一种模型结构，让这个 kernel 成为了 bottleneck？**

例如：

```text
Grouped GEMM
    ↑
small expert GEMM
    ↑
Ultra-Sparse MoE
```

而不是孤立学习 Grouped GEMM。

---

# 61. 现代模型的“六层调度系统”

可以把最新模型抽象成六层：

## 第 1 层：Token / Sequence Routing

```text
当前 token 需要看哪些历史？
```

对应：

- QSA；
- DSA；
- CSA；
- Linear state。

## 第 2 层：Parameter Routing

```text
当前 token 需要哪些参数？
```

对应：

- MoE Router；
- Shared/Routed Expert。

## 第 3 层：Depth Routing

```text
当前层应该读取哪些过去层的表示？
```

对应：

- AttnRes；
- mHC；
- Gated Residual。

## 第 4 层：GPU Kernel Scheduling

```text
这些算子如何合并？
```

对应：

- fused kernel；
- Grouped GEMM；
- FlashKDA；
- FlashMLA。

## 第 5 层：GPU/Network Scheduling

```text
token 和 activation 去哪张 GPU？
```

对应：

- EP；
- DeepEP；
- MoonEP；
- NVLink/IB；
- DualPipe。

## 第 6 层：Memory Tier Scheduling

```text
哪些东西留在 HBM？
哪些可以去 CPU/NVMe？
```

对应：

- KV Cache；
- N-gram offload；
- prefix cache；
- heterogeneous KV。

---

# 62. 未来 LLM 越来越像一个“操作系统 + 数据库 + HPC 程序”

传统 Transformer：

```text
token
 ↓
Attention
 ↓
FFN
 ↓
Attention
 ↓
FFN
```

现代 Frontier Model：

```text
                           Query Token
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
      Sequence Router     Expert Router       Depth Router
           │                   │                   │
           ▼                   ▼                   ▼
   Linear/Compressed      Sparse Experts      Residual Memory
       State                   │                   │
           │                   │                   │
           ├─ Exact Sparse ────┤                   │
           │  Retrieval        │                   │
           ▼                   ▼                   ▼
      KV Memory           Expert Weights      Layer States
           │                   │                   │
     HBM / RAM / NVMe     HBM / Host          HBM / recompute
           │                   │                   │
           └───────────────────┼───────────────────┘
                               ▼
                         GPU Kernel Scheduler
                               │
                         Network Scheduler
                               │
                               ▼
                             Output
```

模型本身已经在决定：

- 什么值得计算；
- 什么值得保存；
- 什么值得精确检索；
- 什么可以压缩；
- 什么参数当前应该激活；
- 什么 memory tier 应该放在哪里。

---

# 63. 一张表总结“创新 → 解决瓶颈 → 新瓶颈”

| 结构 | 解决的问题 | 核心实现 | 新暴露的问题 |
|---|---|---|---|
| MoE | Dense FLOPs 太高 | Router + Top-K Expert | All-to-All / imbalance / small GEMM |
| DeepSeekMoE | Expert specialization 不够 | fine-grained + shared experts | Expert 数更多、通信更复杂 |
| Aux-loss-free balance | balance loss 伤模型 | dynamic routing bias | controller tuning |
| DualPipe | PP bubble + MoE comm | bidirectional PP + F/B overlap | 调度复杂、双份 stage 参数 |
| MLA | KV Cache / HBM traffic | low-rank KV latent | 长序列扫描仍存在 |
| DSA | 长 context attention FLOPs | indexer + Top-K sparse attention | Indexer cost |
| CSA | Indexer 仍扫太多 token | sequence compression + DSA | compression quality |
| HCA | 极长 context 全局建模贵 | aggressive compression + dense attention | 精细 retrieval 弱 |
| KDA/GDN | KV 随 L 增长 | fixed recurrent state | exact recall 弱 |
| Hybrid Linear+Attention | Linear recall 弱 | periodic exact attention | runtime 复杂 |
| QSA | Sparse indexer 太贵 | compressed index + micro-block | block selection quality |
| IndexShare | 每层重复 indexing | multi-layer shared index | index reuse compatibility |
| AttnRes | residual dilution | attention over depth | activation / PP comm |
| Block AttnRes | AttnRes 太贵 | block summary | 粒度 trade-off |
| mHC | deep signal instability | multi-stream residual + manifold constraint | activation / kernel complexity |
| Gated Residual | 固定 residual 太僵硬 | multi-branch + elementwise gate | residual bandwidth |
| N-gram Embedding | backbone capacity 太贵 | sparse lookup table | host transfer / cache locality |
| FP8/FP4 | HBM / weight bandwidth | low precision | quantization stability |
| Muon | token efficiency | matrix-level orthogonal update | stability / optimizer memory |
| MTP | decode 串行 | predict multiple future tokens | acceptance rate |

---

# 64. 我认为真正的“总纲”

如果把 2024–2026 所有变化压缩成一句因果链：

```text
Dense FLOPs 太贵
    ↓
MoE

MoE FLOPs 降低
    ↓
通信 + small GEMM 成瓶颈
    ↓
DeepEP/MoonEP + DualPipe + Grouped GEMM

KV Cache 太大
    ↓
MLA/GQA

KV bytes 降低
    ↓
Attention 仍扫描整个长序列
    ↓
Sparse Attention

Sparse Attention 降低主 Attention
    ↓
Indexer 变成瓶颈
    ↓
CSA / QSA / IndexShare

甚至 Indexing 也嫌贵
    ↓
Linear/Recurrent Attention

Linear State 便宜
    ↓
精确记忆变弱
    ↓
Linear + Sparse/MLA Hybrid

模型越来越深和异构
    ↓
Residual Stream 成瓶颈
    ↓
AttnRes / mHC / Gated Residual

Active FLOPs 越来越少
    ↓
HBM/Weight Movement 占比越来越大
    ↓
FP8/FP4 / Host Offload

单 token forward 越来越快
    ↓
Autoregressive serial dependency 越来越显眼
    ↓
MTP / Speculative Decoding
```

这正是 Amdahl's Law 在大模型架构上的连续体现。

---

# 65. 最终 Insight：未来不是“哪种 Attention 胜出”，而是 Hierarchical Memory

我不认为未来会是：

```text
Linear Attention 淘汰 Softmax Attention
```

也不认为：

```text
Sparse Attention 成为唯一答案
```

更有可能是：

```math
\boxed{
\text{Hierarchical Memory}
+
\text{Conditional Compute}
+
\text{Hardware-aware Scheduling}
}
```

模型同时存在：

```text
Recent exact memory
        │
Sliding Window

Compressed recurrent memory
        │
KDA / GDN

Compressed global memory
        │
HCA

Sparse exact memory
        │
CSA / QSA / DSA

Conditional parameter memory
        │
MoE

Host sparse parameter memory
        │
N-gram Embedding

External memory
        │
RAG / Tool / Search
```

这已经非常像现代计算机的：

```text
register
L1
L2/L3
RAM
SSD
remote store
```

区别只是：

> LLM 的 memory hierarchy 是可学习的。

---

# 66. 对 AI Infra 学习最重要的方法论

以后不要孤立地学习：

- CUDA Stream；
- Grouped GEMM；
- NCCL；
- FlashAttention；
- PagedAttention；
- TileLang；
- Triton；
- DeepEP；
- speculative decoding。

应该始终往上追问两层：

### 第一问

> **是什么模型结构，让这个 Infra 技术变得必要？**

例如：

```text
Grouped GEMM
  ← Ultra-Sparse MoE

Sparse Gather Kernel
  ← Sparse Attention

Prefix Cache
  ← Long-context Agent Serving

FlashKDA
  ← Linear Attention Hybrid

FP4 MoE Kernel
  ← Ultra-Sparse Expert + Weight-bandwidth bottleneck
```

### 第二问

> **这个 Infra 优化完成后，下一个瓶颈会去哪？**

这才是理解技术演进最重要的能力。

---

# 67. 参考资料

以下尽量优先列官方技术报告、官方模型卡、官方仓库或作者团队材料。

## DeepSeek

1. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**  
   https://arxiv.org/abs/2405.04434

2. **DeepSeek-V3 Technical Report**  
   https://arxiv.org/abs/2412.19437

3. **DeepSeek DualPipe**  
   https://github.com/deepseek-ai/DualPipe

4. **DeepSeek-V3.2 / DeepSeek Sparse Attention**  
   https://api-docs.deepseek.com/news/news251201/

5. **DeepSeek Transparency Center / V4 official model card & report links**  
   https://www.deepseek.com/en/transparency/

6. **DeepSeek-V4 Model Card**  
   Official model card linked by DeepSeek Transparency Center.

## Kimi / Moonshot

7. **Kimi K2 official repository and technical report**  
   https://github.com/MoonshotAI/Kimi-K2

8. **Kimi Linear / KDA Technical Report**  
   https://github.com/MoonshotAI/Kimi-Linear

9. **Attention Residuals**  
   https://arxiv.org/abs/2603.15031

10. **Kimi K3 official release / technical report**  
    https://www.kimi.com/news/kimi-k3-open-source

11. **Kimi K3 architecture / Infra release**  
    Includes MoonEP, FlashKDA and related serving work.

## Qwen

12. **Qwen3 Technical Report**  
    https://arxiv.org/abs/2505.09388

13. **Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate Cost-Efficiency**  
    https://qwen.ai/blog?id=qwen3.8-flash-next

14. **On the Design of Qwen3.8-Next Architecture**  
    https://arxiv.org/abs/2608.30320

15. **Qwen3.8-Flash-Next official repository**  
    https://github.com/QwenLM/Qwen3.8-Flash-Next

## GLM / Z.ai

16. **GLM-4.5 Technical Report / official model family**  
    https://arxiv.org/abs/2508.06471

17. **GLM-5 series official repository / model documentation**  
    https://github.com/zai-org/GLM-5

18. **GLM-5.3-Flash implementation documentation**  
    Hugging Face Transformers `glm5_next` model documentation.

---

# 68. 后续适合继续展开的专题

这份文档先建立“模型结构 ↔ Infra 瓶颈迁移”的总地图。后面每一项都值得单独形成一章：

1. **MLA 数学推导**：从 MHA → MQA/GQA → MLA，完整推导 KV cache bytes；
2. **DSA / CSA / QSA kernel 路径**：Indexer → Top-K → Page Mapping → Sparse MLA；
3. **Linear Attention**：DeltaNet / GDN / KDA 的 recurrence 与 chunkwise parallel 算法；
4. **MoE Infra**：Router → Dispatch → DeepEP/MoonEP → Grouped GEMM → Combine；
5. **Residual Evolution**：PreNorm → HC → mHC → AttnRes → Gated Residual；
6. **FP8/FP4**：数据格式、scale granularity、Tensor Core、QAT；
7. **Muon**：Newton-Schulz、optimizer state、分布式优化器系统；
8. **MTP / Spec Decode**：训练目标如何转换为 serving speedup；
9. **Hybrid Architecture Serving**：KDA state + MLA KV 如何做 prefix cache / PD disaggregation；
10. **未来 Memory Hierarchy LLM**：HBM / RAM / NVMe / RAG / learned memory 的统一视角。

---

> **一句话总结**
>
> 大模型结构的演化，本质上是瓶颈迁移：MoE 将瓶颈从计算迁移到通信和小 GEMM，MLA 将瓶颈从 KV 容量迁移到长序列扫描，Sparse Attention 再将瓶颈迁移到 Indexer，Linear Attention 则用固定状态消除扫描但牺牲精确记忆，因此最终形成 Hybrid Attention；与此同时，Ultra-Sparse MoE 把系统推向 Weight Movement、FP4、Grouped GEMM 和 All-to-All，深层异构网络又催生 AttnRes/mHC/Gated Residual，而 MTP 则开始直接优化 autoregressive decode 的串行性。未来的 Frontier LLM 越来越不是单纯的 Transformer，而是一个围绕 GPU、HBM、互联网络和多级 Memory Hierarchy 共同设计的计算系统。
