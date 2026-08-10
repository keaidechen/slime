# Megatron-FSDP：状态分片与执行生命周期

## 1. 它与 Distributed Optimizer 的边界

Distributed Optimizer 主要分片 optimizer state 与主参数，并通过 gradient reduce-scatter / parameter all-gather 协作；FSDP 进一步把参数、梯度、optimizer state 的驻留与 materialize 生命周期组织成 unit。两者都减少 DP 维度冗余，但执行模型、通信粒度和 checkpoint 接口不同，不能只按“都是 ZeRO”理解。

Megatron-FSDP 提供 `no_shard`、`optim`、`optim_grads`、`optim_grads_params` 等基础策略。分片越彻底，静态显存越低，但 forward/backward 前后的参数 gather、梯度 reduce 和 buffer 管理越复杂。

## 2. FSDP unit 是性能边界

unit 太大：峰值 materialization 高、通信启动晚；unit 太小：collective 数量多、latency 和调度开销高。unit 选择还决定 prefetch、double buffering、通信计算 overlap 的窗口。Transformer layer 往往是自然起点，但不保证对 MoE、共享 embedding 或混合层最优。

核心源码入口：

- `Megatron-LM/megatron/core/distributed/fsdp/`
- `Megatron-LM/megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py`
- `Megatron-LM/megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py`

## 3. 一次迭代的状态变化

```text
sharded params
  -> pre-forward all-gather/materialize
  -> forward
  -> 可释放 full params
  -> pre-backward materialize
  -> backward + gradient reduce-scatter
  -> optimizer 更新 local shard
```

优化重点是让下一 unit 的 gather 与当前 unit 计算重叠，并及时释放不再需要的 full buffer。显存快照要区分 persistent shard、临时 full param、double buffer、gradient shard 与 allocator reserved。

## 4. 与 TP/PP/CP 的组合

FSDP 沿 DP 组分片，TP/PP/CP 已先改变本 rank 模型与 activation 的局部形态。开启 symmetric/user buffer、低精度 gradient communication 或 meta-device init 时，要验证与现有 TP/CP stream、allocator 配置和 checkpoint backend 的兼容性。

## 5. Checkpoint 与恢复

优先保存 sharded state dict，并用分布式 checkpoint backend 表达 global tensor 到 local shard 的映射。验收至少覆盖：同拓扑恢复、改变 DP 恢复、目标需要时改变 model-parallel 拓扑、optimizer state 恢复后下一步 loss/参数连续。

官方 FSDP 页面内容更新较快；配置字段以当前源码和[官方指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/megatron_fsdp.html)为准，不照搬旧版本命令。
