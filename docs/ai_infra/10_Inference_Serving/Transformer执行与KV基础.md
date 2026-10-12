# Transformer 执行与 KV 基础

> 相关基础：[基础概念](../02_Model_Architecture/Transformer与Attention基础.md)。关联：[机制与演进](推理内存系统.md)。

训练用梯度改参数；推理固定参数，根据输入预测输出。Serving（模型服务）还需要接收请求、排队、返回结果和回收资源。本篇先跟踪一条请求，下一篇再管理多条请求的共享显存。

Prompt 是用户输入及系统模板形成的 token 序列；Prefill（预填充）计算输入序列，Decode（逐步解码）消费新 token 并继续预测。Dense Transformer 指这里每个 token 使用相同的稠密层，区别于选择部分专家的 MoE。

<details>
<summary>文章导航</summary>

- [先画一个 Dense Transformer block](#先画一个-dense-transformer-block)
- [为什么 LLM 推理必须理解 Prefill 与 Decode？](#为什么-llm-推理必须理解-prefill-与-decode)
- [KV Cache：为什么一个“缓存”能成为系统核心？](#kv-cache为什么一个缓存能成为系统核心)
- [为什么 KV Cache 会产生“内存碎片”？](#为什么-kv-cache-会产生内存碎片)
- [Virtual Memory 与 Paging：为什么 vLLM 会借鉴操作系统？](#virtual-memory-与-paging为什么-vllm-会借鉴操作系统)
- [Internal Fragmentation 与 External Fragmentation](#internal-fragmentation-与-external-fragmentation)
- [Copy-on-Write 为什么会出现在 LLM Serving？](#copy-on-write-为什么会出现在-llm-serving)
- [Prefix Cache 与 Radix Tree 的直觉](#prefix-cache-与-radix-tree-的直觉)
- [完整推演](#完整推演)

</details>

> 前置：GPU 内存与 tensor shape。目标：先用 Dense block 建立训练/推理的数据布局，再理解动态 KV。


## 先画一个 Dense Transformer block

令 batch 为 B、序列为 S、hidden 为 H，输入为 [B,S,H]。Q/K/V 投影得到每个 token 的头向量；attention 混合可见 token，再由输出投影回到 H；MLP 先扩张再回投影，残差连接要求输出与输入形状相容。LayerNorm/RMSNorm、激活和位置编码也消耗算子与内存，不能只数大矩阵乘。

PyTorch Linear 的权重通常存为 [out_features,in_features]，数学写 XW 时 W 常定义为 [in,out]。读 TP 代码前先约定方向，否则“按行/列切”容易说反。

训练会为 backward 保存或重算 activation；推理通常不建训练图，但 decode 会跨步保留 KV。GQA 的 query head 数和 KV head 数不同，KV 核算要用后者。packing 后的总 token 数相同，也不表示 attention 可见 pair 数相同。


## 为什么 LLM 推理必须理解 Prefill 与 Decode？


一个请求：

```text
Prompt: 1000 tokens
```

推理通常分为两阶段。

#### Prefill

计算整个 prompt；工程上也可分块执行。最后一个输入位置的 logits 可以直接用于采样首个输出 token，所以首 token 通常来自 prefill 的末尾，而不是必须额外做一次 decode。

此时：

```text
Q length ≈ 1000
K/V length ≈ 1000
```

有大量大矩阵乘法，GPU 并行度较高。

#### Decode

随后把刚采样的 token 作为下一次输入，补上它的 KV，再预测下一个 token。以下是不使用推测解码的单请求例子：

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

Memory-bandwidth-bound 表示主要受显存带宽限制；大 batch、长上下文或不同模型也可能转成计算或通信受限，不能把这一判断当作所有 decode 的定律。Serving throughput 是单位时间服务完成的工作量。

---


## KV Cache：为什么一个“缓存”能成为系统核心？


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


## 为什么 KV Cache 会产生“内存碎片”？


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

vLLM 是一个模型推理引擎，PagedAttention 是其提出的分页 KV 管理与访问方法。它借鉴操作系统的 virtual memory（虚拟内存）与 paging（分页）思路。

---


## Virtual Memory 与 Paging：为什么 vLLM 会借鉴操作系统？


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


## Internal Fragmentation 与 External Fragmentation


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


## Copy-on-Write 为什么会出现在 LLM Serving？


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


## Prefix Cache 与 Radix Tree 的直觉


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

若只用完整 prompt 作为一个哈希键，难以直接查找最长可复用前缀；也可以设计按块哈希和前缀链来实现复用，并不是只有树才能做到。

Radix Tree（基数树）将公共前缀合并为共享路径，因而自然表达：

> **最长公共前缀和树状共享关系。实际可复用粒度仍受 KV 块大小与实现约束。**

这就是 RadixAttention 的核心数据结构基础。

---

## 完整推演

[批次与KV长度推演](<批处理排队与调度演进.md#数值展开运行时请求队列token-budgetcuda-graph-与结果返回>)：区分已采样token与已计算KV的token。


---

<!-- main-route -->

KV 的含义已经明确，实际服务还要管理变长请求与共享前缀。下一篇追踪 KV 的分配、复用和回收。

[上一篇：Pipeline Parallelism](<../08_Parallelism/PipelineParallelism.md>) · [主线 16/26](<../guides/系统阅读路线.md>) · [下一篇：推理内存系统](<推理内存系统.md>)

<!-- /main-route -->
