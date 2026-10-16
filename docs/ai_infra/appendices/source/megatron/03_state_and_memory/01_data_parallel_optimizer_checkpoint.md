# DDP、分布式 Optimizer 与 Checkpoint

相关基础：[通信与 tensor 基础](<../../../communication/两卡通信与torchrun.md>)。


优化器公式与矩阵耦合见[AdamW](<../../../../02_Model_Computation/05_损失自动微分与优化器.md#s2>)、[Muon](<../../../training/Muon与矩阵优化.md>)；本快照的 Adam/AdamW 后端选择、Muon 参数路由与 LayerWise owner 见[优化器执行走读](<04_adam_muon_execution.md>)。

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

### 为什么要分桶，而不是每个参数发一次通信

bucket 是将若干梯度放在一起通信的连续区间。设网络每次调用的固定启动成本 α=10 μs，有效带宽 β=10 GB/s，单个小梯度 1000 bytes。简化耗时 `t≈α+Q/β` 中，Q 为传输字节数；单个约 `10+0.1=10.1 μs`，100 次约 1010 μs。合成 100000 bytes 一次发送约 `10+10=20 μs`。这些是假设值，真实 collective 还受 rank 数与算法影响，但说明了小消息为什么昂贵。

代价是必须等整桶最后一个梯度就绪才能发出；桶越大，开始通信越晚。反向通常先产生靠近输出层的梯度，因此按其产生顺序组织 bucket，可以让先就绪的梯度在后续反向计算期间通信。bucket 不是模型的逻辑层，也不保证与参数边界一致。

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

master weight（主权重）通常是优化器更新使用的较高精度参数副本，计算时再转换为低精度副本。它能累积较小更新，但不能恢复低精度前反向中已丢失的信息。

1. 两 rank backward 得到各自样本的 8 元素 grad；
2. reduce-scatter 求和后，rank0 的 optimizer view 是前 4，rank1 的 optimizer view 是后 4；完整 grad buffer 通常仍作为通信/累积 backing storage 分配；
3. 各自更新自己的 FP32 master weight 与 Adam m/v；
4. all-gather 新 param，恢复 forward 所需完整 DP replica。

参数边界不一定与 shard 边界对齐，所以代码维护 model param、main param、group range、world range 等 mapping。阅读 distributed optimizer 最有价值的是画“逻辑一维 buffer 区间图”。

把八元素例子赋上数值：rank0 的本地梯度为 `[1,2,3,4,5,6,7,8]`，rank1 为 `[3,4,5,6,7,8,9,10]`。假设每个 rank 的本地目标已是等样本数的均值，全局目标是两个均值的平均，则目标梯度为 `[2,3,4,5,6,7,8,9]`。SUM RS 可先产生前半 `[4,6,8,10]`、后半 `[12,14,16,18]`，再各除以 2；实际实现也可能预先缩放，不能再重复除。

为了只演示 owner，用无动量 SGD（随机梯度下降）和学习率 0.1，初始参数全为 1。rank0 更新前四个为 `[0.8,0.7,0.6,0.5]`，rank1 更新后四个为 `[0.4,0.3,0.2,0.1]`。AG 后两副本都拿到完整新参数。Adam 换了局部更新公式和历史状态，RS → 本地更新 → AG 的归属关系相同。

若 rank1 跳过更新却仍参与 AG，拼回的模型会混合两版参数；若省略 AG，则每个副本只有自己负责的部分是新值。因此“梯度已同步”和“下一轮参数已同步”是两个独立完成条件。

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

考虑一桶的最小时序：计算 stream 完成最后一次梯度写入 → 通信 stream 等待该事件并启动 RS → optimizer 的 stream 等待 RS 完成 → 更新本地状态 → AG → 下一轮 forward 的消费者等待 AG。CPU 获得异步 work handle 只代表提交成功；相关 wait/event 才建立消费者顺序，不一定意味着整个 GPU 已经全局空闲。

缓冲区复用的条件也是这条依赖链：清零旧梯度必须晚于 optimizer 和仍在读取它的通信；覆盖参数 gather buffer 必须晚于所有用它的算子。单凭 `param.grad is None` 无法判断 `main_grad` 的 storage 已无人使用。

## 4. mixed precision 状态

BF16/FP16 训练常同时存在：

- 低精度 model param；
- FP32 master param；
- gradient（可能 FP32 accumulation）；
- Adam first/second moments；
- scaler/overflow state。

粗略 Adam、未分片情况下仅模型状态就可能达到每参数 16–20 bytes，具体取决于参数/grad dtype 与实现。不要用“模型参数量×2 bytes”估算训练显存。

低精度为什么还需要 master weight？若计算参数在某个数值范围内的相邻可表示值间距为 Δ，而一次更新仅为 Δ/10，直接在该低精度参数上相减可能舍入回原值。高精度主权重可以先累积这些小更新，再转换为计算参数。FP16/BF16 都占 2 bytes，但指数范围和尾数精度不同，不能据此认为数值行为相同。

loss scaling 则处理另一处损失：令缩放后目标 `L'=sL`，反向得到 `g'=sg`，在更新前用 `g=g'/s` 还原，避免某些小梯度在低精度计算中过早下溢。s 是无单位倍率；若 g' 出现 Inf/NaN，不能继续更新，应按 scaler 规则跳过并调整 s。主权重解决更新精度，loss scaling 解决反向动态范围，两者不是互相替代。

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

RNG（Random Number Generator，随机数生成器）具有随抽样推进的内部状态。仅保存初始 seed 不能恢复已推进的位置，重计算和恢复训练通常还需保留状态及对应的消费顺序。

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

### 把 bytes/parameter 公式拆成实物

BF16 参数、FP32 主梯度这一行最容易手算：计算参数 2 bytes + 完整梯度缓冲 4 bytes + FP32 主权重 4 bytes + Adam 的一阶/二阶矩各 4 bytes，共 `2+4+4+4+4=18`。Distributed Optimizer 只把最后三项按数据分片域大小 d 分开，于是 `2+4+(4+4+4)/d=6+12/d`。d=2 时为 12，d=8 时为 7.5 bytes/param；没有 activation 的含义。

FP16/FP16 行还计入更新所需 FP32 梯度转换空间，分片前为 `2+2+4+4+8=20`；按表中实现的归属，把 FP32 梯度、主权重和 m/v 分片后得到 `4+16/d`。FP32 参数路径不需要额外低精度计算副本/独立主权重，保留参数与梯度的 8 bytes，再分片 m/v 的 8 bytes，得到 `8+8/d`。这些是该公式的 buffer 假设，不能套到任意 optimizer、精度封装或 FSDP 实现。

设本 TP/PP rank 的逻辑参数量为十亿，d=8 的 BF16/FP32 账本是 `7.5×10⁹ bytes≈6.98 GiB`，因为 `1 GiB=2³⁰ bytes`。这还不含 padding、额外 workspace、通信缓冲和激活；不能直接用十亿全局参数量，忽略 TP/PP 已经切分的部分。

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


例如全局矩阵 `W:[4,2]` 原先按行分两份，rank0 的 offset=`(0,0)`、shape=`[2,2]`，rank1 的 offset=`(2,0)`、shape=`[2,2]`。目标改为四份时，每份 shape=`[1,2]`，offset 依次为 `(0,0)…(3,0)`；loader 按全局坐标取相应行，无须让旧文件名等于新 rank 号。

若两个 DP replicas 都存同一个 shard，它们的逻辑覆盖相同但 replica 身份不同，验证时不能当成两份互相冲突的主分片。若新模型把 W 的第 0 行从“词 A”改成“词 B”，字节和 shape 完全合法，语义却变了；checkpoint backend 无法只靠 offset 自动修复这种映射。

异步保存还多一条生命周期：先为当前 step 建立稳定快照，再由后台写 shard，最后完成全体结果和 metadata/tracker 的提交。若后台尚在读 GPU/CPU buffer，下一步就改写同一 storage，会保存混合时刻的值；需要复制到独立 staging buffer 或明确等待读完。写函数返回请求对象不代表恢复端已可读。更多提交边界见[源码问题第 14 节](<../06_reference/03_source_questions.md>)。

## 13. 恢复一致性是一组状态关系

考虑同一初始状态下的两条执行路径：A 连续更新 20 次，B 更新 10 次后保存，在新进程恢复并继续 10 次。这里的数字用于把恢复边界定位到第 10/11 步，并非实测结果。二者等价的前提是恢复了决定后续轨迹的全部状态。

第 10 步的 checkpoint 参数 hash 对应保存时刻；第 11 步的样本 ID 对应数据游标；第 11–20 步的 loss 对应后续目标；最终 TP/PP shard 参数对应模型更新。optimizer m/v、LR、loss scaler 和 RNG tracker 则分别控制更新方向与尺度、学习率、数值范围和随机流。

如果权重相同而 m/v 不同，下一步 Adam 更新就可能不同；如果优化器恢复但数据游标偏移，轨迹会从第 11 步分叉。低精度与非确定性 kernel 还要求明确容差。最终 validation 指标相近，无法单独排除这些短暂但真实的恢复偏差。

## 14. 一步执行链：梯度就绪不等于同步完成

出发点是数据吞吐或 optimizer 状态冗余；前者用复制模型分配样本，后者沿数据域分片状态。它们都必须保持同一全局目标的一步更新等价。

沿 [DDP](<../../../../../../Megatron-LM/megatron/core/distributed/distributed_data_parallel.py>)、[buffer](<../../../../../../Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py>)、[finalize](<../../../../../../Megatron-LM/megatron/core/distributed/finalize_model_grads.py>) 依次观察：

| 时刻 | 符号 | 要检查的责任 |
|---|---|---|
| 参数梯度产生 | `_make_backward_post_hook` | 非 fused 已累加路径将 `param.grad` 加到 `main_grad`；避免重复累加，随后清理 `param.grad` |
| 累积中的 microbatch | `no_sync` | bucket group 的 `is_last_microbatch=False`，保留累积而不提前最终规约 |
| 最后 microbatch 的 bucket 就绪 | `register_grad_ready` | 就绪计数满足条件后启动同步；shared/fused 路径不能假定一个参数永远只触发一次 |
| 提交同步 | `start_grad_sync` | 根据 Distributed Optimizer 等配置选择 AR/RS，检查 group 和 scale |
| 同步收尾 | `finish_grad_sync` | optimizer 的消费者边界必须等结果可读；不能只判断 work 已提交 |
| 其他梯度处理 | `finalize_model_grads` | 完成 SP、embedding 等所需归约及可选 token 分母缩放 |
| 更新与下一轮 | optimizer step、param sync | 更新本地拥有的 state，必要时 gather，使下轮读到新参数 |

结果验收：固定全局样本，将 1 个副本与 2 个副本的一步梯度/更新对齐；不等 token 数时按目标分母核对，而不是直接平均各 rank 的均值。

收益边界：普通 DDP 不减少每副本持久状态；Distributed Optimizer 的 local shard 不意味着原完整 grad buffer 一定释放。代价是 bucket/通信 workspace、网络尾部和同步点；bucket 太小增加启动次数，太大推迟可重叠窗口。具体观测与已有多卡入口见[并行证据边界](<../05_design_and_failures/03_parallelism_validation.md>)。

## 15. 本章延伸阅读

- [Distributed Optimizer 官方指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)：包含字节公式、data flow 与 sharding 图。
- [Distributed Checkpoint API](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/dist_checkpointing.html)：理解 sharded state dict 与策略层。
- [ZeRO 论文](https://arxiv.org/abs/1910.02054)：用 stage 1/2/3 建立状态分片基线。
- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)：对比 PyTorch DCP 的 planner/storage 抽象。
