# DDP、分布式 Optimizer 与 Checkpoint

## 1. Megatron DDP 不只是 PyTorch DDP 包装

`DistributedDataParallel` 位于 `megatron/core/distributed/distributed_data_parallel.py:22`。它把参数 gradient 映射到连续 grad buffer，并用 bucket 控制通信。目标是：

- 避免大量小 tensor collective；
- backward 产生一桶 gradient 后尽早发起通信；
- 与计算重叠；
- 为 distributed optimizer 的 reduce-scatter 提供连续布局。

理解 bucket 要区分：

```text
parameter storage
main gradient buffer
communication bucket view
optimizer shard
```

`param.grad` 可能不是你想象的独立 tensor；Megatron 常用 `param.main_grad` 指向连续 buffer。写自定义 gradient 逻辑前必须确认目标字段和 dtype。

## 2. 从 all-reduce 到 reduce-scatter

普通 DP：

```text
每 rank 有完整 grad
  → all-reduce
每 rank 仍有相同完整 grad
```

distributed optimizer：

```text
完整逻辑 grad buffer
  → reduce-scatter
每 rank 的 optimizer 只消费其 shard 对应的 reduced grad view
  → 本地更新 master param / moments shard
  → all-gather 更新后的 param shard
每个 DP replica 获得下一次 forward 需要的参数
```

这里“optimizer 只消费 local shard”不等于完整 grad backing storage 已消失。当前 DDP buffer 路径通常先分配完整 `bucket.grad_data`，reduce-scatter 的输出写入其中属于本 rank 的 local view；因此官方显存公式仍计入未按 DP 除掉的 gradient bytes。要同时区分通信结果的逻辑 ownership 与 buffer 的物理分配。

主类位于 `megatron/core/optimizer/distrib_optimizer.py:113`。它与 ZeRO-1 思路相近，但深度集成 Megatron 的 buffer、TP/PP 参数布局、mixed precision 和 overlap。

### 一个 8 元素例子

DP=2，逻辑参数/grad 展平后 8 元素。rank0 管 optimizer index `[0:4)`，rank1 管 `[4:8)`：

1. 两 rank backward 得到各自样本的 8 元素 grad；
2. reduce-scatter 求和后，rank0 的 optimizer view 是前 4，rank1 的 optimizer view 是后 4；完整 grad buffer 通常仍作为通信/累积 backing storage 分配；
3. 各自更新自己的 FP32 master weight 与 Adam m/v；
4. all-gather 新 param，恢复 forward 所需完整 DP replica。

参数边界不一定与 shard 边界对齐，所以代码维护 model param、main param、group range、world range 等 mapping。阅读 distributed optimizer 最有价值的是画“逻辑一维 buffer 区间图”。

## 3. overlap 的时序

- `overlap_grad_reduce`：某个 backward bucket ready 后立即 RS/AR；
- `overlap_param_gather`：optimizer step 后或下一次 forward 前，按需要预取参数；
- `delay_grad_reduce/delay_param_gather`：把同步边界交给外层 schedule。

必须满足：任何 bucket 在读取前已经完成对应 collective。异步 handle、CUDA stream/event 与 `finish_grad_sync()`/`start_param_sync()` 的边界就是 correctness。

典型隐患：

- 自定义 hook 改写 gradient 发生在 reduce 已启动后；
- unused parameter 导致 bucket 永远不 ready；
- conditional computation 让 DP ranks 的 hook 顺序不一致；
- overlap 增加 buffer，显存估算漏项。

## 4. mixed precision 状态

BF16/FP16 训练常同时存在：

- 低精度 model param；
- FP32 master param；
- gradient（可能 FP32 accumulation）；
- Adam first/second moments；
- scaler/overflow state。

粗略 Adam、未分片情况下仅模型状态就可能达到每参数 16 bytes 量级，具体取决于参数/grad dtype 与实现。不要用“模型参数量×2 bytes”估算训练显存。

## 5. 分布式 checkpoint 的核心抽象

`megatron/core/dist_checkpointing/` 将“逻辑全局 tensor”与“当前 rank 的 shard”分离。sharded state dict 通常描述：

- global shape；
- local shard 与 global offset；
- replica id；
- flattened range；
- key 与 metadata。

保存策略负责把 shard 写到存储；加载策略根据新的并行布局重建目标 shard。这让改变 DP 规模、部分 TP/PP 布局时有机会重分片，而不是要求文件布局等于运行布局。

## 6. checkpoint 必须包含什么

要精确续训通常需要：

- model；
- optimizer master params/moments；
- LR/WD scheduler；
- iteration/consumed samples；
- RNG（Python/NumPy/PyTorch/CUDA/TP tracker）；
- data sampler/iterator state；
- loss scaler；
- 并行与模型配置 metadata。

只恢复 model 叫 warm start，不叫严格 resume。

### 两阶段验证

1. **结构验证**：key、global shape、dtype、shard coverage、无重叠/无空洞；
2. **语义验证**：连续跑 N+M step 与 N step 保存后恢复再跑 M step，比较 sample 顺序、loss、参数 hash。

## 7. 常见事故

- checkpoint 成功但恢复 loss 跳变：optimizer/RNG/data state 未恢复；
- 换 TP 后 shape error：某参数未声明正确 shard axis；
- 所有 rank 同时写同一文件：writer 协调配置错误；
- 保存卡住：异步保存队列、存储尾延迟或 rank 异常；
- 加载后首步 OOM：optimizer state materialization 峰值或 param AG；
- 只检查 rank0 参数：TP/PP shard 错误被遗漏。

## 8. DDP 构造函数源码精读

`distributed_data_parallel.py:22` 的类注释已经明确三件事：连续 grad buffer、按 bucket 通信、gradient accumulation dtype 可与参数 dtype 不同。

固定快照中的 bucket 选择：

```python
dp_group = process_group_dict["dp_group"]
if ddp_config.bucket_size is None:
    ddp_config.bucket_size = max(
        40_000_000, 1_000_000 * dp_group.size()
    )
if not ddp_config.overlap_grad_reduce:
    ddp_config.bucket_size = None

if disable_bucketing or pp_rank > 0:
    self.bucket_size = None
```

逐行解释：

- DP 很大时 bucket 也变大，避免 NCCL ring chunk 太小而落入 latency-bound；
- 不 overlap 时 `None` 表示不需要为了提前通信拆桶；
- 非首 PP stage 默认关闭 bucketing，因为其 DP 通信不在同样的 critical path 上，拆桶可能只增加开销；
- interleaved 后续 model chunk 也可能显式禁用 bucketing。

因此 bucket size 不是纯网络参数，它与 PP critical path 共同决定。

## 9. 参数到 buffer 的 view

构建 DDP 后，参数/gradient 通常映射到连续 storage：

```text
grad_data:
| param C grad | padding | param B grad | param A grad |
                ^ bucket boundary
```

参数遍历常按反向产生 gradient 的近似逆序布置，使先 ready 的 bucket 先通信。padding 用于 DP shard 对齐或通信效率。

自定义 gradient clipping/正则化必须在 finalize gradient 之后、optimizer step 之前读 `main_grad`。如果直接把 `param.grad = new_tensor`，可能脱离连续 buffer，后续 reducer 仍读取旧 view。

## 10. `DistributedOptimizer` range map 精读

`distrib_optimizer.py:113` 说明 shard 边界不尊重参数边界。核心计算近似：

```python
param_local_start = max(
    0, param_world_start - gbuf_world_range.start
)
param_local_end = min(
    gbuf_world_range.size,
    param_world_end - gbuf_world_range.start
)
if param_local_end > param_local_start:
    param_range_map[param] = {
        "gbuf_world": ...,
        "gbuf_local": ...,
        "param": sub_param_range,
    }
```

假设 bucket 有 12 元素，DP=3，每 rank 4 元素；参数 P 跨 `[3,8)`：

```text
rank0 owns [0,4): P 的 [3,4) → param slice [0,1)
rank1 owns [4,8): P 的 [4,8) → param slice [1,5)
rank2 owns [8,12): 不拥有 P
```

所以一个参数的 Adam states 可能分在两个 DP ranks。`param_range_map` 同时记录全局 buffer、bucket 内、本地 buffer 与参数内部四套坐标，任何 off-by-one 都会造成参数部分未更新或写错。

## 11. 官方显存公式

当前官方 distributed optimizer 指南给出每参数理论 bytes：

| 参数/主梯度 | 普通 optimizer | distributed optimizer（DP=d） |
|---|---:|---:|
| fp16 / fp16 | 20 | `4 + 16/d` |
| bf16 / fp32 | 18 | `6 + 12/d` |
| fp32 / fp32 | 16 | `8 + 8/d` |

这是 model-state 理论值，不包括 activation、padding、bucket、通信 workspace、allocator 碎片。以 BF16/FP32、d=8 为例是 7.5 bytes/param，而不是 18。

## 12. 分布式 checkpoint 对象模型

从 `megatron/core/dist_checkpointing/` 重点读：

```text
mapping.py / ShardedTensor
serialization.py
strategies/
validation.py
```

一个 `ShardedTensor` 可理解为：

```python
ShardedTensor(
    key="decoder.layers.0.mlp.weight",
    data=local_tensor,
    global_shape=(...),
    global_offset=(...),
    axis_fragmentations=(...),
    replica_id=(...),
)
```

保存前 validation 应证明所有非 replica shard 对 global tensor 覆盖恰好一次。加载时 target sharded state dict 描述“我现在需要哪些 shard”，backend 再从存储重组。

## 13. 恢复一致性实验

严格测试：

```text
Run A: seed固定，连续训练 20 step
Run B: 同初始状态训练 10 step → save → 新进程 load → 10 step
```

比较：

- step 10 后 checkpoint 参数 hash；
- step 11 的样本 ids；
- step 11–20 loss；
- 最终各 TP/PP shard 参数；
- optimizer m/v、LR、loss scaler；
- RNG tracker state。

只比较最终 validation 指标无法发现 resume 后短暂换样本或 optimizer 漂移。

## 14. 本章延伸阅读

- [Distributed Optimizer 官方指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)：包含字节公式、data flow 与 sharding 图。
- [Distributed Checkpoint API](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/dist_checkpointing.html)：理解 sharded state dict 与策略层。
- [ZeRO 论文](https://arxiv.org/abs/1910.02054)：用 stage 1/2/3 建立状态分片基线。
- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)：对比 PyTorch DCP 的 planner/storage 抽象。
