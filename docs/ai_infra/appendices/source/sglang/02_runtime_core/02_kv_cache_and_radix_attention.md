# 2.2 KV Cache、RadixAttention 与引用锁

相关基础：[Transformer 与 KV](<../../../../05_Inference/19_生成执行与采样.md#s1>)。


## 1. 为什么 KV cache 是 serving 的中心

自回归 decode 若每步重算全部历史，计算量不可接受。每层保存历史 token 的 K/V，新 token 只计算自己的 Q/K/V，再读取历史 KV attention。

KV 容量决定可同时运行的 token 数，也决定 scheduler 的 admission。模型权重相对静态，KV 则随请求高速分配、共享、增长、驱逐和释放。

### 从 Attention 公式推到缓存

先只看一层、一个头。输入隐藏向量 `X∈R^{T×d_model}` 经三个线性投影得到 Query（查询）`Q=XW_Q`、Key（键）`K=XW_K`、Value（值）`V=XW_V`；三个投影权重在本例中均为 `[d_model,d]`，Q/K/V 各为 `[T,d]`。T 是 token 数，`d_model` 是隐藏维度，d 是每头维度，均为无量纲计数。位置编码可能进一步变换 Q/K，缓存应保存该实现真正供 Attention 使用的状态。

某个新位置的 `q` 形状为 `[1,d]`，历史加当前位置的 `K/V` 为 `[T,d]`。先算 `s=qKᵀ/√d` 得 `[1,T]`，再做 softmax 得到非负且和为 1 的权重 `a`，最后 `o=aV` 得 `[1,d]`。除以 `√d` 是为了在各分量方差约为 1 且近似独立时，把 d 项内积约为 d 的方差缩回约 1，避免 softmax 过早饱和；它不是时间或字节单位的换算。

Softmax 将一组实数分数变成概率权重：`a_j=exp(s_j)/sum_k exp(s_k)`，j/k 遍历可见位置，分母保证总和为 1。训练得到的特征和分数在这里均按无物理单位数值处理；指数权重并不代表时间或存储量。因果 mask 将未来位置分数视为 `-∞`，其指数为 0，因而不能贡献输出。

可手算的退化例取 `d=1`，`q=[0]`，`K=[[1],[2],[3]]`，`V=[[2],[4],[8]]`。三个分数均为 0，softmax 为 `[1/3,1/3,1/3]`，输出为 `(2+4+8)/3=14/3`。本轮只需新位置的 q；旧位置的 Q 已完成职责，所以一般缓存 K/V，不缓存所有旧 Q。即便 K/V 已缓存，本轮仍要读它们来算这个加权和。

为什么只缓存不会让长上下文 decode 变成常数成本？长度为 T 的新 Query 仍与 T 个 key 做内积，约需 `2Td` 次乘加计数意义下的浮点运算，再对 value 加权约需 `2Td`，两项约 `4Td` FLOP，另有 softmax 等开销。完整因果 prefill 的可见位置对为 `1+2+…+T=T(T+1)/2`，所以同一头的这两项约为 `2dT(T+1)` FLOP。取 T=3，可见位置对是 `1+2+3=6`；下一次 decode 在总长 4 时只处理 4 对。KV 缓存省去旧位置的重算，但不能省去新位置与所需历史的交互。

### 每 token 到底占多少字节

普通 MHA（Multi-Head Attention，多头注意力）每个 Q 头有对应 K/V 头；GQA（Grouped-Query Attention，分组查询注意力）让多个 Q 头共享一个 K/V 头。内存公式计的是 KV 头数，不是 Q 头数。设本 rank 保存 L 层、每层 H 个 KV 头、每头 d 个元素，每元素占 s 字节，则：

```text
每层每 token 的 K = H × d × s 字节
每层 K 加 V        = 2 × H × d × s 字节
所有本地层         = c = 2 × L × H × d × s 字节/token
T 个不共享的位置   = T × c 字节
```

玩具模型取 `L=2,H=1,d=2,s=2`（BF16，16 位浮点，每元素 2 字节），得到 `c=16 字节/token`，6 个位置需 96 字节。若 A/B 各长 4、共享前 2 个位置，物理位置总数是 `2+2+2=6` 而非 8，KV 从 128 降到 96 字节；两条请求自己的映射表仍各有 4 项。

这里不含页取整、量化 scale、映射表、对齐和 workspace。TP 分片需代入实际本地 KV 头数；头被复制时不能简单除以 TP 度。PP 分层需代入本地层数。MLA（Multi-head Latent Attention，多头潜在注意力）保存压缩潜在状态等不同布局，应另按实际元素数算，不能套用普通 K/V 公式。

## 2. 两级映射

### Request-to-token pool

`ReqToTokenPool` 位于 `srt/mem_cache/memory_pool.py:251`。可把它看成二维页表：

```text
req_pool_idx, logical_token_position
              ↓
physical_kv_index
```

每个请求占一个 row/slot，row 中记录其各 token 的物理 KV index。

### Token-to-KV pool

例如 `MHATokenToKVPool` 位于同文件约 1658 行，实际按 layer/head/dtype 存 K/V。allocator 维护哪些 physical token slot 空闲。

解耦的价值：

- 请求逻辑 token 连续，物理 KV 可离散；
- prefix cache 可让多个请求的 row 指向同一物理 KV；
- eviction 只需回收物理 index；
- page size/量化/不同 attention layout 可换 backend。

## 3. Radix tree 存什么

`RadixCache` 位于 `srt/mem_cache/radix_cache.py:280`，节点 `TreeNode` 在约 217 行。压缩 radix tree 节点通常保存：

- `key`：一段 token ids（边标签压缩成数组）；
- `value`：对应的 physical KV indices；
- children/parent；
- last access / priority；
- `lock_ref`；
- host value/hash 等扩展 metadata。

根节点为空。路径拼接是完整 token prefix。

LoRA（Low-Rank Adaptation，低秩适配）冻结原权重 W，用较小的矩阵乘积 BA 表示可训练增量，使有效权重为 W+BA。减少的是可训练参数及其梯度、优化器状态，原权重仍需参与前向。

`extra_key` 构成命名空间：相同 token 在不同 LoRA、cache version 或隔离上下文下不能错误共享。仅用 token ids 做 key 可能产生 silent correctness bug。

## 4. prefix match 与节点 split

`match_prefix()`：

1. 按 page size 截到对齐长度；
2. 从 root 沿 token 段匹配；
3. 收集命中节点的 KV indices；
4. 更新访问 metadata；
5. 若匹配停在压缩节点内部，则 split 节点。

例：

```text
树中边 key = [10, 11, 12, 13]
查询       = [10, 11, 99]
```

公共前缀 `[10,11]` 停在边内部。split 后：

```text
[10,11]
   └─ [12,13]
```

这样 `[10,11]` 成为精确可锁定/插入的边界。split 只重组 metadata，不复制 KV。

## 5. insert 的三种情况

插入新序列时递归比较节点：

1. 无 child：创建叶子；
2. child key 完全匹配：进入 child；
3. 部分匹配：split，再挂旧 suffix 与新 suffix。

插入返回已存在 `prefix_len`。`cache_finished_req()` 会释放请求中与树已存 prefix 重复的 KV index，因为 cache 只需持有一份。

这段代码的所有权语义要细读：

```text
request 运行时：req row 持有/引用 KV
插入 cache：radix tree 接管一份长期引用
已重复部分：释放新算出的 duplicate physical slots
请求结束：释放 req slot，并解锁路径
```

## 6. `lock_ref` 为何不能省

LRU（Least Recently Used，最近最少使用）优先淘汰最长时间未被访问的缓存对象。被活跃请求引用的对象能否淘汰，还需先满足生命周期约束。

LRU 不能驱逐正在被 running request 使用的 prefix。请求命中节点后增加从节点到 root 的保护引用；结束/切换时减少。通常：

- `lock_ref > 0`：protected，不能 evict；
- `lock_ref == 0` 且是可驱逐叶子：进入 eviction 候选。

增加 child lock 可能使祖先从 evictable 变 protected；减少到零又会使路径重新可驱逐。`evictable_size_` 与 `protected_size_` 必须随状态转换精确更新。

例如同一长度为 2 的前缀，初始 `lock_ref=0`；A 命中后为 1，B 再命中后为 2，A 结束后降至 1，此时即便 A 的 request slot 已 free，两个 KV 位置仍不能驱逐；B 也结束后降到 0，才成为候选。锁计的是活跃保护关系，不是 KV 字节数；只有同一段从 0 变正或从正变 0 时，相关容量才在可驱逐与受保护类别间转换。

典型灾难：

- 少 `inc`：运行请求读到已复用的 KV，输出静默错误；
- 少 `dec`：cache 永久不可驱逐，最终 OOM；
- split 后 ref/parent 转移错：统计或保护范围错误；
- abort 路径漏解锁：压力下才泄漏。

## 7. page alignment

page size > 1 时，只能共享/回收完整 page；未对齐尾部需单独处理。`match_prefix` 和 `insert` 都会 page-align key。page 大：

- metadata 少、kernel/allocator 友好；
- 内部碎片和短 prefix 浪费增加。

page 小则相反。要按请求长度和共享模式 benchmark。

把页大小记为 P（token/page），请求长度为 T（token）。不共享且按整页分配时，需要 `ceil(T/P)` 页，槽位数为 `P×ceil(T/P)`；内部碎片是这个槽位数减 T。`ceil` 表示向上取整，因为最后一个 token 即使只占一小部分也需要一整页。可复用完整前缀长度则是 `P×floor(H/P)`，`floor` 表示向下取整，H 是实际相同的前缀长度。

例如 `P=4,T=7,H=5`：分配 2 页共 8 个 slot，空 1 个；前缀只有前 4 个 token 可以按整页共享，第 5 个相同 token 仍可能需要重算。分配用向上取整，匹配用向下取整，方向相反；混用会造成漏算容量或越界复用。保护与释放以页/节点实际所有权为准，不能单独释放仍有活跃元素的整页。

## 8. eviction 与 HiCache

驱逐策略可用访问时间/priority，从未锁定叶子开始，释放 KV 并向上级联清理空节点。不能先删父节点再判断 child 使用情况。

HiCache 把 cache 扩展到 GPU→host→远端存储层次。infra 需要额外理解：

- device/host 命中状态；
- async prefetch 与 timeout；
- pinned host 容量/NUMA；
- tombstone/并发回填；
- 慢层命中是否真的优于重算。

例如需要恢复 8 MiB KV，有效传输带宽为 2 GiB/s，纯搬运下界是 `8/2048 秒≈3.91 ms`；加 1 ms 的排队和元数据开销，约 4.91 ms。若重算相同前缀只要 3 ms，慢层“命中”并不划算。若重算要 20 ms，则值得尝试恢复。这里 GiB 是 `2^30` 字节，MiB 是 `2^20` 字节，带宽必须是实际有效值。

异步恢复时，host buffer 在 DMA（Direct Memory Access，直接内存访问）读完前必须保留，目标 GPU slot 在写完前不能交给 Attention。Pinned host memory 指主机页被固定以便传输，不表示数据已经到达 GPU；命中树节点、拥有目标地址和数据可读取是三个时刻。NUMA（Non-Uniform Memory Access，非统一内存访问）还会使不同 CPU/设备连接位置的搬运成本不同。

## 9. 必做不变量

建议在 debug/stress test 检查：

- 所有 req row 的有效 KV index 均在 allocator 已占用集合；
- free 与 allocated 不重叠；
- radix node value 长度与 key/page 对齐；
- protected node 不在 eviction candidate；
- `evictable + protected` 与树持有 token 总量一致；
- finish/abort/retract 后 slot、lock、KV 数量回到预期。

## 10. `ReqToTokenPool` 源码精读

`memory_pool.py:251`：

```python
self._alloc_size = size + 1
self.req_to_token = torch.zeros(
    (self._alloc_size, max_context_len),
    dtype=torch.int32,
    device=device,
)
self.free_slots = list(range(1, self._alloc_size))
self.req_generation = torch.zeros(self._alloc_size, dtype=torch.int64)
```

第 0 行故意留给 CUDA Graph padding dummy batch。这样 padded `req_pool_indices=0` 的读写落在安全页，不会破坏真实请求。

分配：

```python
if r.req_pool_idx is None:
    r.req_pool_idx = select_index[offset]
    self.req_generation[r.req_pool_idx] += 1
```

`req_generation` 使同一个 slot 的多次复用可区分。异步事件若携带旧 generation，可拒绝 late write，避免“请求 A 已释放，迟到操作写坏新请求 B”。

chunked request 可复用已有 slot，但源码断言它必须有 in-flight middle chunk 或 committed KV；普通新请求不应偷偷携带 slot。

## 11. Radix `TreeNode` 源码精读

`radix_cache.py:217`：

```python
self.children = defaultdict(TreeNode)
self.parent = None
self.key = None
self.value = None
self.lock_ref = 0
self.last_access_time = time.monotonic()
self.host_ref_counter = 0
self.host_value = None
self.hash_value = None
self.priority = priority
```

这里有两套保护：

- `lock_ref`：device KV 正被请求使用；
- `host_ref_counter`：HiCache host value 正被 storage/prefetch 操作使用。

device 已 evicted（`value is None`）不代表节点无用；若 `host_value` 仍在，可能从 host 恢复。树节点的逻辑存在、device resident、host backed 是三个不同状态。

## 12. `match_prefix` 的真实前处理

```python
key, _ = key.maybe_to_bigram_view(self.is_eagle)
key = key.page_aligned(self.page_size)
value, last_node = self._match_prefix_helper(self.root_node, key)
value = torch.cat(value) if value else empty
```

EAGLE 可把 key 切换为 bigram 视图，普通 token-level cache 心智模型不再完全适用。page alignment 在树查找前执行：

```text
prompt length=65, page_size=64
最多只有前 64 token 可作为完整 page 命中
```

在线官方 attention backend 文档也明确说明：page=64、32-token prompt 完全无法形成 prefix hit。

## 13. split 的所有权例子

原节点：

```text
key=[A,B,C,D]
value=[10,11,12,13]
lock_ref=2
children={...}
```

在 `[A,B]` split 后应成为：

```text
parent key=[A,B], value=[10,11]
child  key=[C,D], value=[12,13]
```

需正确转移：

- 原 children 给 suffix child；
- parent pointers；
- lock/protected 语义；
- last access/priority；
- host value/hash 按 page 同步切；
- evictable leaf set。

仅切 key/value 而忘记 host/hash，会在 HiCache restore 时对错页。

## 14. finish cache 的重复 KV

请求命中 prefix P 后继续生成 suffix S。运行时 req row 可能含：

```text
[P 的共享 physical indices | S 的新 indices]
```

插入完整 P+S 时树已含 P，`insert` 返回已存在的 prefix length。**原先命中的共享 P 不能释放**：它仍属于树，可能被别的请求读取。本快照只释放 `kv_indices[req.cache_protected_len:result.prefix_len]`，即原保护边界之后、新插入时才发现与树重复的那段新算 KV；树接管真正新增的 S。之后：

1. tag session leaf；
2. free page 未对齐尾部；
3. `dec_lock_ref(req.last_node)`；
4. 释放 req slot。

任何次序改变都要重新证明没有 double free 和 use-after-free。

例如 A 已引用树中 `[10,11]` 的物理位置 `[3,8]`，自己为后缀 `[12,13]` 分配 `[20,21]`。若 B 在 A 插入前已把 token `12` 的等价 KV 插入树，位置为 30，那么 A 的重复副本只有 slot 20；slot 3/8 仍共享，slot 21 可交给新树节点。把“前缀已存在”误解为释放 `[3,8,20]` 会破坏所有共享请求。

## 15. 页粒度为何同时影响复用与浪费

以长度 31/32/33/63/64/65 和页大小 1/16/64 观察取整边界。只考虑末页分配且不共享时，长度 33、页大小 16 需要 48 个 slot，末页有 15 个空位；页大小 64 则需要 64 个 slot，空位为 31。这个容量算例没有包含树锁、共享和 backend 的额外限制。

matched tokens、cached tokens 与实际分配量属于不同统计。更细粒度可能减少内部碎片并增加可复用前缀，更粗粒度可能改变 kernel 访问效率与元数据成本。因此 Attention 吞吐和 TTFT 的最优点依赖具体 backend 与请求分布，没有通用最佳 page size。

## 16. 本章延伸阅读

- [RadixAttention 原始博客](https://www.lmsys.org/blog/2024-01-17-sglang/)：逐步展示树的 insert、match、LRU eviction。
- [SGLang 论文](https://arxiv.org/abs/2312.07104)：RadixAttention 设计与 workload。
- [Attention Backend 官方指南](https://docs.sglang.io/docs/advanced_features/attention_backend)：page size、不同 backend 与 prefix reuse 约束。
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)：理解逻辑 block table 与物理 KV block。
- [HiCache 设计文档](https://docs.sglang.io/docs/advanced_features/hicache_design)：理解 device/host/storage 多层状态。
