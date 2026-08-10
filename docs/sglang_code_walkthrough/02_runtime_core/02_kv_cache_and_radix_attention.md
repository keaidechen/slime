# 2.2 KV Cache、RadixAttention 与引用锁

## 1. 为什么 KV cache 是 serving 的中心

自回归 decode 若每步重算全部历史，计算量不可接受。每层保存历史 token 的 K/V，新 token 只计算自己的 Q/K/V，再读取历史 KV attention。

KV 容量决定可同时运行的 token 数，也决定 scheduler 的 admission。模型权重相对静态，KV 则随请求高速分配、共享、增长、驱逐和释放。

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

LRU 不能驱逐正在被 running request 使用的 prefix。请求命中节点后增加从节点到 root 的保护引用；结束/切换时减少。通常：

- `lock_ref > 0`：protected，不能 evict；
- `lock_ref == 0` 且是可驱逐叶子：进入 eviction 候选。

增加 child lock 可能使祖先从 evictable 变 protected；减少到零又会使路径重新可驱逐。`evictable_size_` 与 `protected_size_` 必须随状态转换精确更新。

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

## 8. eviction 与 HiCache

驱逐策略可用访问时间/priority，从未锁定叶子开始，释放 KV 并向上级联清理空节点。不能先删父节点再判断 child 使用情况。

HiCache 把 cache 扩展到 GPU→host→远端存储层次。infra 需要额外理解：

- device/host 命中状态；
- async prefetch 与 timeout；
- pinned host 容量/NUMA；
- tombstone/并发回填；
- 慢层命中是否真的优于重算。

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

插入完整 P+S 时树已含 P，`insert` 返回 prefix length。代码释放新插入视图中与树重复的 physical indices，只让树接管新增 S。之后：

1. tag session leaf；
2. free page 未对齐尾部；
3. `dec_lock_ref(req.last_node)`；
4. 释放 req slot。

任何次序改变都要重新证明没有 double free 和 use-after-free。

## 15. 容量与碎片实验

构造长度分别为 31/32/33/63/64/65 的共享 prefix，page size 取 1/16/64，测：

- matched tokens；
- cached tokens；
- internal fragmentation；
- attention throughput；
- TTFT。

你会看到 cache granularity 与 kernel page efficiency 的直接冲突，没有通用最佳 page size。

## 16. 本章延伸阅读

- [RadixAttention 原始博客](https://www.lmsys.org/blog/2024-01-17-sglang/)：逐步展示树的 insert、match、LRU eviction。
- [SGLang 论文](https://arxiv.org/abs/2312.07104)：RadixAttention 设计与 workload。
- [Attention Backend 官方指南](https://docs.sglang.io/docs/advanced_features/attention_backend)：page size、不同 backend 与 prefix reuse 约束。
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)：理解逻辑 block table 与物理 KV block。
- [HiCache 设计文档](https://docs.sglang.io/docs/advanced_features/hicache_design)：理解 device/host/storage 多层状态。
