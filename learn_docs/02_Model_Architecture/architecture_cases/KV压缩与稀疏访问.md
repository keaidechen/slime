# KV压缩与稀疏访问

> 类型：专题详解。所属：[专题总览](README.md)。涉及模型版本、API 和性能数字的旧稿内容尚未逐条重新核验，见[版本与验证范围](../../版本与验证范围.md)。

<a id="read-18"></a>

<a id="9-第二场革命为什么-mla-会出现"></a>

## 第二场革命：为什么 MLA 会出现？

MoE 主要解决 FFN compute。

Inference 还有另一个独立瓶颈：


<a id="read-19"></a>

## KV Cache

---


<a id="read-20"></a>

<a id="91-标准-mha-的-kv-cache"></a>

### 1 标准 MHA 的 KV Cache

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


<a id="read-21"></a>

<a id="10-gqamqa-先解决了一部分-kv-问题"></a>

## GQA/MQA 先解决了一部分 KV 问题

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


<a id="read-22"></a>

<a id="11-mlamulti-head-latent-attention-到底具体怎么实现"></a>

## MLA：Multi-head Latent Attention 到底具体怎么实现？

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


<a id="read-23"></a>

<a id="12-为什么-mla-是一个典型-infra-trade-off"></a>

## 为什么 MLA 是一个典型 Infra trade-off？

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


<a id="read-24"></a>

<a id="13-mla-解决了-kv-bytes但没有解决-sequence-length"></a>

## MLA 解决了 KV bytes，但没有解决 sequence length

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


<a id="read-25"></a>

## Sparse Attention

---


<a id="read-26"></a>

<a id="14-deepseek-sparse-attentiondsa-到底怎么流"></a>

## DeepSeek Sparse Attention：DSA 到底怎么流？

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


<a id="read-27"></a>

<a id="15-dsa-的-infra-pipeline"></a>

## DSA 的 Infra Pipeline

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


<a id="read-28"></a>

<a id="16-sparse-attention-优化完之后为什么-indexer-自己会成为瓶颈"></a>

## Sparse Attention 优化完之后，为什么 Indexer 自己会成为瓶颈？

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


<a id="read-29"></a>

<a id="17-deepseek-v4csa-是怎么实现的"></a>

## DeepSeek-V4：CSA 是怎么实现的？

DeepSeek-V4 的关键变化之一是：


<a id="read-30"></a>

## Compressed Sparse Attention

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


<a id="read-31"></a>

<a id="171-为什么-csa-还保留-sliding-window"></a>

### 1 为什么 CSA 还保留 Sliding Window？

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


<a id="read-32"></a>

<a id="18-deepseek-v4hca-又是什么"></a>

## DeepSeek-V4：HCA 又是什么？

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

#### CSA

擅长：

```text
从远处找少数精确信息
```

类似：

> search / retrieval

#### HCA

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


<a id="read-33"></a>

<a id="19-从-dsa--csahca-的路线说明了什么"></a>

## 从 DSA → CSA/HCA 的路线说明了什么？

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

---

[所属专题](README.md) · [百科总览](../../README.md)
