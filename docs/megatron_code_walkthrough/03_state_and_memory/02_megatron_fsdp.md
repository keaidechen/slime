# Megatron-FSDP：状态分片与执行生命周期

本章只描述仓库固定 commit 中的 Megatron-FSDP，不把 PyTorch FSDP2 或早期 `custom_fsdp` 的行为混进来。应用入口是 `--use-megatron-fsdp`；训练侧最终用 `megatron_FSDP` 包装每个 model chunk。

## 1. 它与 Distributed Optimizer 的边界

Distributed Optimizer 与 Megatron-FSDP 都沿数据并行域分片，但最重要的差异是训练参数是否长期驻留：

| 方案 | 低精度 training params | gradients | optimizer/main states | 计算前是否需要 materialize 参数 |
|---|---|---|---|---|
| DDP | replica | replica/AR | replica | 否 |
| DDP + Distributed Optimizer | 通常为 replica | optimizer 读 RS local view；完整 grad buffer 通常仍分配 | 分片 | step 后/forward 前做参数 AG，使 replica 可计算 |
| FSDP `optim` | replica | replica/AR | 分片 | 否 |
| FSDP `optim_grads` | replica | 分片/RS | 分片 | 否 |
| FSDP `optim_grads_params` | 分片 | 分片/RS | 分片 | 是 |

当前参数校验会在开启 Megatron-FSDP 时同时启用 Distributed Optimizer。这里不是把两个独立 wrapper 叠起来，而是 optimizer 复用 FSDP 的 param/grad buffer 与 shard metadata。阅读时从 `training.get_model` 看 wrapper，再到 `optimizer/__init__.py` 看 optimizer 如何识别 `ddp_config.use_megatron_fsdp`。

更完整的选择解释见[源码问题详解第 12 节](../06_reference/03_source_questions.md)。

## 2. 四种 sharding strategy 的准确语义

`--data-parallel-sharding-strategy` 接受：

```text
no_shard          ~= DDP
optim             ~= optimizer-state sharding / ZeRO-1
optim_grads       ~= optimizer + gradient sharding / ZeRO-2
optim_grads_params~= optimizer + gradient + training-param sharding / ZeRO-3
```

名称描述“哪些状态沿 DP shard”，不是四套不同模型数学。默认是 `optim_grads_params`。另有 `outer_dp_sharding_strategy` 支持 hybrid sharding：outer group 可复制，或在 inner stage-3 基础上再分 optimizer state；它依赖 `num_distributed_optimizer_instances` 形成 inner/outer DP group，不能只看一个 DP size。

源码锚点：

- `megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py:ShardingStrategy`
- `megatron/core/distributed/fsdp/src/megatron_fsdp/distributed_data_parallel_config.py`
- `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py:_init_dist_index`

## 3. FSDP unit 只在哪条路径上是生命周期边界

FSDP unit 的职责是界定一组 full parameters 必须存活到哪里：进入 unit 前 all-gather/materialize，forward/backward 使用后释放或 reshard。源码明确说明 `optim`、`optim_grads` 不分片 model params，因此不需要 FSDP units；unit 对 `optim_grads_params` 才是核心边界。

unit 太大：

- 单次 materialize 的 full param 峰值高；
- 下一 unit 的 prefetch 启动较晚；
- backward 同时存活的 full buffer 可能更多。

unit 太小：

- all-gather 数量和 launch latency 增加；
- hook/stream/event 调度更密；
- 参数桶变小，网络更难达到带宽上限。

Transformer layer 是自然起点，但 MoE grouped MLP、共享 embedding、MTP 与 fusion module 可能有不同参数使用边界。fine-grained param gather 还可把 hook 下沉到子模块；这会改变预取粒度，不能与“每层一个 unit”当成同一配置。

源码锚点：`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:MegatronFSDP` 中的 unit 构建与 `_param_list_for_submodule_unshard`；应用适配层是 `mcore_fsdp_adapter.py:FullyShardedDataParallel`。

## 4. 一次迭代的状态变化

对 `optim_grads_params`，简化生命周期为：

```text
persistent local param shards
  -> pre-forward AG / materialize current unit
  -> forward
  -> reshard/release不再需要的 full param
  -> pre-backward AG / materialize
  -> backward
  -> gradient RS，保留 local grad shard
  -> optimizer 更新 local main-param/moment shard
  -> 下一轮按 unit 再次 materialize
```

对 `optim`/`optim_grads`，training params 常驻，不发生上面的 unit param AG/释放；差异主要在 gradient 和 optimizer state 是否分片。因此不能用一张“sharded params → AG”时序图解释所有 strategy。

梯度 accumulation 还决定每个 microbatch 是否同步。`sync_model_each_microbatch`、`no_sync()`、`overlap_grad_reduce` 与 schedule 的 finalize hook 共同决定 RS/AR 边界；错误地重复同步同一 accumulation buffer 会让 gradient 被多次规约。

## 5. overlap、prefetch 与 double buffer

`overlap_param_gather` 让后续 unit 的参数 AG 尽量藏在当前 unit 计算后；`overlap_grad_reduce` 让已完成 backward 的 gradient bucket 尽早 RS。correctness 边界仍是：参数被读取前 AG 必须完成，optimizer 读取 grad shard 前 RS 必须完成。

常见 buffer 类别：

```text
persistent param shard
materialized full-param bucket
prefetch / double buffer
gradient accumulation buffer 或 grad shard
main param + optimizer moments
通信 user buffer / NCCL workspace
```

打开 overlap 或 double buffer 会用更多显存换遮蔽窗口。显存快照若只统计 persistent shard，会严重低估 forward/backward 峰值。开启 `--nccl-ub` 时，当前参数逻辑还会默认打开 FSDP double buffer 与 manual registration，应把这两个隐式变化记录进 A/B 配置。

## 6. TP、PP、CP、EP 组合时谁先切什么

顺序上先由模型构建确定本 rank 的 PP layers 与 TP shards，再由 FSDP 沿 `dp_cp` mesh 管理这些 local parameters。`mcore_fsdp_adapter._init_dist_index` 显式构造包含 DP/CP、TP 以及 expert mesh 的 `DeviceMesh`，并给 column/row/replicated 参数标注 TP 模式。

这意味着：

- FSDP 看到的“一个参数”往往已经是 TP-local shard；
- PP rank 只保存自己 model chunk 的 FSDP state；
- expert 参数使用 expert DP/TP/EP mesh，不能套普通 dense DP group；
- checkpoint metadata 必须同时表达 TP 与 FSDP shard 轴。

当前快照还有明确限制与性能冲突：

- Megatron-FSDP 只接受 Adam/SGD；
- 要求 `CUDA_DEVICE_MAX_CONNECTIONS` 不为 `1`，而 Hopper 及更早架构上的某些 TP/CP overlap 推荐值相反；组合可运行不等于性能最优；
- hybrid context parallel 明确不支持 Megatron-FSDP；
- single-grouped MoE weight/bias 路径尚不支持；
- `optim_grads_params` 不支持跨 DP replica 的 weight-hash 检查。

限制以 `megatron/training/arguments.py:validate_args` 为准，不能只参考旧启动命令。

## 7. Checkpoint 与恢复

当前参数校验要求 Megatron-FSDP 使用：

```text
--ckpt-format fsdp_dtensor
```

adapter 为参数标注 TP 类型并建立 DTensor device mesh，使 checkpoint 同时知道 DP/CP shard 与 TP column/row/replicated 语义。验收至少覆盖：

1. 同拓扑 save/load 后下一步连续；
2. 改 DP 后恢复；
3. 产品目标要求时，验证 TP/PP 变化的目标 state dict 与参数 mapping；
4. optimizer m/v、main params、scheduler、RNG 和 data state 同时恢复；
5. 保存中断时不发布不完整 checkpoint。

“backend 能重分片”不代表任意模型变更都能恢复。它能按 metadata 重组同一个 global tensor，不能自动推导 QKV/GQA layout、layer 重排或 tokenizer 改变。边界详见[源码问题详解第 13、14 节](../06_reference/03_source_questions.md)。

## 8. 性能与正确性验收

对每个候选 strategy，至少记录：

```text
构模后 persistent memory
首个 unit AG 峰值
forward / backward 峰值
optimizer step 峰值
param AG 与 grad RS 的 exposed tail
每个 bucket 的 numel、次数与所属 group
```

correctness 先用 `no_shard` 或 DDP baseline 对齐固定输入 loss/gradient，再逐级切换 `optim`、`optim_grads`、`optim_grads_params`。若直接从 DDP 跳到 stage-3 并同时打开 TP/PP/overlap/double-buffer，出现差异时无法判断是 shard mapping、schedule hook 还是异步通信 race。

官方页面更新较快；命令字段先以当前源码为准，再用[官方 Megatron-FSDP 指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/megatron_fsdp.html)补充动态能力。
