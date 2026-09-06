# Transformer 执行与 KV 基础

<details>
<summary>本篇分段导航：按首读范围进入，其余二读</summary>

- [先画一个 Dense Transformer block](#read-01)
- [15. 为什么 LLM 推理必须理解 Prefill 与 Decode？](#read-02)
- [16. KV Cache：为什么一个“缓存”能成为系统核心？](#read-03)
- [17. 为什么 KV Cache 会产生“内存碎片”？](#read-04)
- [18. Virtual Memory 与 Paging：为什么 vLLM 会借鉴操作系统？](#read-05)
- [19. Internal Fragmentation 与 External Fragmentation](#read-06)
- [20. Copy-on-Write 为什么会出现在 LLM Serving？](#read-07)
- [21. Prefix Cache 与 Radix Tree 的直觉](#read-08)

</details>

<!-- learning-position -->
> **学习定位**：A1/A4 · 必修。
> **前置**：[基础课程](<README.md>)。
> **首读/二读**：先画 Dense block 的 shape，再读 prefill/decode、KV、分页、碎片与前缀复用。
> **进度与实验**：[学习清单](<../学习清单.md>) · [总入口](<../README.md>)。
<!-- /learning-position -->

> 前置：GPU 内存与 tensor shape。目标：先用 Dense block 建立训练/推理的数据布局，再理解动态 KV。预计 3–4 小时。


<a id="read-01"></a>

## 先画一个 Dense Transformer block

令 batch 为 B、序列为 S、hidden 为 H，输入为 [B,S,H]。Q/K/V 投影得到每个 token 的头向量；attention 混合可见 token，再由输出投影回到 H；MLP 先扩张再回投影，残差连接要求输出与输入形状相容。LayerNorm/RMSNorm、激活和位置编码也消耗算子与内存，不能只数大矩阵乘。

PyTorch Linear 的权重通常存为 [out_features,in_features]，数学写 XW 时 W 常定义为 [in,out]。读 TP 代码前先约定方向，否则“按行/列切”容易说反。

训练会为 backward 保存或重算 activation；推理通常不建训练图，但 decode 会跨步保留 KV。GQA 的 query head 数和 KV head 数不同，KV 核算要用后者。packing 后的总 token 数相同，也不表示 attention 可见 pair 数相同。


<a id="shared-15"></a>


<a id="read-02"></a>

## 15. 为什么 LLM 推理必须理解 Prefill 与 Decode？


一个请求：

```text
Prompt: 1000 tokens
```

推理通常分为两阶段。

#### Prefill

一次性处理整个 prompt。

此时：

```text
Q length ≈ 1000
K/V length ≈ 1000
```

有大量大矩阵乘法，GPU 并行度较高。

#### Decode

之后每一步只生成一个或少数 token：

```text
Q length = 1
K/V length = 1001
```

下一步：

```text
Q length = 1
K/V length = 1002
```

此时每一步计算量相对小，但必须读取历史 KV Cache。

所以 decode 常常表现为：

> **memory-bandwidth-bound。**

这也是为什么 KV Cache 管理会直接决定 serving throughput。

---


<a id="shared-16"></a>


<a id="read-03"></a>

## 16. KV Cache：为什么一个“缓存”能成为系统核心？


Attention：

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt d}\right)V
$$

自回归生成第 $t$ 个 token 时，前面 token 的 K/V 已经计算过。

如果每一步重新计算全部历史 K/V，非常浪费。

所以保存：

```text
token 1 → K1,V1
token 2 → K2,V2
...
token t → Kt,Vt
```

这就是 KV Cache。

一个粗略的每 token KV Cache 大小公式：

$$
\text{bytes/token}
=
2\times L\times H_{kv}\times d_h\times \text{bytes(dtype)}
$$

其中：

- `2`：K 与 V；
- $L$：Transformer 层数；
- $H_{kv}$：KV heads 数；
- $d_h$：head dimension。

例如一个仅作直觉演示的 MHA 配置：

```text
32 layers
32 KV heads
head_dim = 128
FP16 = 2 bytes
```

则：

$$
2\times32\times32\times128\times2
=524288\text{ bytes}
$$

约等于：

> **512 KiB / token**。

2000 token 就接近 1 GiB KV Cache。

现代 GQA/MQA 会明显降低这个数字，但它仍然可能成为 serving 中最大的动态显存消费者之一。

---


<a id="shared-17"></a>


<a id="read-04"></a>

## 17. 为什么 KV Cache 会产生“内存碎片”？


真实 serving 中请求长度不同：

```text
A: 700 tokens
B: 14000 tokens
C: 1200 tokens
D: finished
E: just arrived
```

输出长度在开始时通常又不知道。

如果提前给每个请求预留最大长度：

```text
Request A: [used][unused..............]
Request B: [used........][unused......]
```

会浪费很多显存。

如果不断 malloc/free 不同大小区域，又可能出现：

```text
used | hole | used | small hole | used
```

虽然空闲总量够，但找不到足够大的连续区域。

这就是碎片问题。

PagedAttention 的核心就是把这个问题类比成操作系统的 virtual memory / paging。

---


<a id="shared-18"></a>


<a id="read-05"></a>

## 18. Virtual Memory 与 Paging：为什么 vLLM 会借鉴操作系统？


操作系统希望进程看到：

```text
Virtual address
0 1 2 3 4 5 6 ...
```

但物理内存可以是：

```text
Physical page 7
Physical page 2
Physical page 91
...
```

中间通过 page table 映射：

```text
Virtual Page 0 → Physical Page 7
Virtual Page 1 → Physical Page 2
Virtual Page 2 → Physical Page 91
```

因此：

> **逻辑连续 ≠ 物理连续。**

PagedAttention 把 KV Cache 也拆成固定大小 block：

```text
Logical KV Block
       ↓ block table
Physical KV Block
```

这让请求可以按需增长，而不需要拥有一块连续的巨大显存。

---


<a id="shared-19"></a>


<a id="read-06"></a>

## 19. Internal Fragmentation 与 External Fragmentation


这是读 PagedAttention 必须区分的两个词。

#### Internal Fragmentation

已经分配了一块内存，但里面没有完全使用。

例如 block 容量 16 token：

```text
[10 tokens used][6 slots empty]
```

这 6 个 slot 是块内部浪费。

#### External Fragmentation

空闲内存存在，但被分散在很多小洞中：

```text
used | free | used | free | used
```

没有一块足够大的连续区域。

固定大小 page/block 可以显著缓解外部碎片。

---


<a id="shared-20"></a>


<a id="read-07"></a>

## 20. Copy-on-Write 为什么会出现在 LLM Serving？


假设同一个 prompt 要采样 4 个回答：

```text
Prompt
  ├── Answer A
  ├── Answer B
  ├── Answer C
  └── Answer D
```

Prompt 的 KV Cache 完全相同。

没必要复制 4 份：

```text
共享 prompt blocks
```

当某个 branch 要写入自己的新 token 时，再复制需要修改的最后一个共享 block。

这就是：

> **Copy-on-Write（写时复制）**。

它同样源自操作系统进程 `fork()` 的经典设计思想。

---


<a id="shared-21"></a>


<a id="read-08"></a>

## 21. Prefix Cache 与 Radix Tree 的直觉


假设很多请求都包含：

```text
You are a helpful assistant...
```

如果这段 prefix 的 KV 已经算过，就可以直接复用。

但真实请求共享关系可能是多层的：

```text
                System Prompt
                /           \
          Conversation A   Few-shot Prompt
             /    \              /   \
           A1     A2            Q1    Q2
```

简单哈希表只能表达“整个 prefix 是否命中”。

Radix Tree 可以表达：

> **任意长度的最长公共前缀和树状共享关系。**

这就是 RadixAttention 的核心数据结构基础。

---
