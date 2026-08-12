# 源码问题索引与详解

本章集中回答那些会同时跨过训练循环、并行组、通信和状态管理的问题。结论以仓库内 `Megatron-LM/` 固定 commit `21fe0fe1597932421f1bb0efd93f469376a7a255` 为准；源码锚点使用“路径 + 符号”，避免只有行号的答案随代码插入而失效。

## 快速索引

| 问题 | 先看结论 | 主要源码 |
|---|---|---|
| world size 为什么不能总写成六维乘积？ | dense 网格和 expert 网格是两套视图 | `parallel_state.initialize_model_parallel` |
| 一个 optimizer step 到底消费多少样本？ | `MBS × DP × microbatches` | `num_microbatches_calculator.py`、`training.train` |
| overflow 后哪些计数推进？ | iteration/sample 推进，LR 不推进 | `training.train_step`、`training.train` |
| loss 为什么返回 closure？ | schedule 只有在末 stage 才知道怎样规约 | `schedules.forward_step_calc_loss` |
| 变长样本怎样得到真正 token mean？ | 打开 per-token 路径，最后按总有效 token 归一化 | `finalize_model_grads.py` |
| batch 是不是每个 rank 都从 DataLoader 读取？ | TP source 取数并广播，随后 CP 再切序列 | `core.utils.get_batch_on_this_tp_rank` |
| Column/Row Parallel 的通信分别在哪？ | 通信可能在 forward，也可能藏在 autograd backward | `tensor_parallel/mappings.py` |
| SP 与 CP 到底差在哪？ | SP 配合 TP 切 replicated activation；CP 切 attention context | `tensor_parallel/layers.py`、`core.utils` |
| 为什么 TP 通常节点内、EP 更怕慢网络？ | 前者高频同步，后者是变长 token 重排 | `parallel_state.py`、`moe/token_dispatcher.py` |
| collective hang 为什么常不是 NCCL bug？ | rank 违反了同 group、同顺序、同 tensor 契约 | 所有 collective 调用点 |
| PP 为什么能“释放”已经发送的 activation？ | 只替换 payload，保留 autograd graph head | `schedules.deallocate_output_tensor` |
| Distributed Optimizer 与 Megatron-FSDP 怎样选？ | 前者保持训练参数常驻，后者可管理参数 materialize 生命周期 | `distrib_optimizer.py`、`distributed/fsdp/` |
| checkpoint 为什么能换 DP，却不保证任意换模型？ | 它重组已声明的同一 global tensor，不推导新架构映射 | `dist_checkpointing/mapping.py` |
| async checkpoint 什么时候才算完整？ | finalize 写出 metadata 后才完整 | `dist_checkpointing.serialization.save` |
| MoE auxiliary loss 要不要在任务 loss 手工相加？ | 当前实现自动接入 autograd，不应重复相加 | `moe/router.py`、`schedules.py` |
| router replay 究竟固定了什么？ | 固定 top-k expert ID，不冻结当前 score/probability | `moe/router_replay.py` |
| overlap 为什么可能不快？ | 只能遮蔽通信，还会增加 bucket/buffer/调度成本 | `distributed_data_parallel.py` |
| fixed seed 为什么不等于 strict resume？ | 数据、RNG、低精度和归约顺序都是状态 | `checkpointing.py`、`random.py` |

## 1. world size 为什么不能总写成 `TP×PP×CP×EP×DP`

对 dense/attention 部分，当前实现先计算：

```text
model_size = TP × PP × CP
DP = world_size / model_size
```

然后建立 `decoder_rank_generator(tp=TP, ep=1, dp=DP, pp=PP, cp=CP)`。MoE expert 部分则另建：

```text
expert_model_size = expert_TP × EP × PP
expert_DP = world_size / expert_model_size
expert grid = (expert_TP, EP, expert_DP, PP, CP=1)
```

因此 EP 不是再从已有 world size 外面乘一次；它会重新解释 expert 参数所在的 rank 轴，并改变 expert data-parallel group 的大小。`RankGenerator.__init__` 还断言一份 generator 中不能同时令 EP、CP 大于 1，正是为了让普通网格包含 CP、expert 网格包含 EP。

评审配置时应分别写：

```text
dense/attention grid: TP, PP, CP, DP
expert grid:          expert_TP, PP, EP, expert_DP
```

再验证二者生成相同的 PP groups。只写一条六维乘法式，会把“物理 rank 数”和“同一批 rank 的两种逻辑分组”混为一谈。

源码锚点：`megatron/core/parallel_state.py:RankGenerator`、`initialize_model_parallel`。

## 2. 一个 update 消费多少样本，microbatch 数怎样变化

常规 fixed batch 下：

```text
num_microbatches = GBS / (MBS × DP)
本 DP replica 每次 update 的样本数 = MBS × num_microbatches
全局每次 update 的样本数 = MBS × DP × num_microbatches
```

`train_step()` 只调用一次 `forward_backward_func`，但 schedule 在内部循环 `num_microbatches` 次。PP 不出现在 GBS 公式中：PP ranks 合作处理的是同一批样本，而不是新增数据副本。

`update_num_microbatches(consumed_samples, ...)` 会在每轮前更新 calculator。当前快照支持 step batch-size schedule；当 microbatch 数变化时，普通训练会先保存 checkpoint，再做一致性检查。若开启 `decrease_batch_size_if_needed`，实际 running GBS 可以向下取到可整除值，不能再把配置中的目标 GBS 当成本轮真实消费量。

源码锚点：`megatron/core/num_microbatches_calculator.py`、`megatron/training/training.py:train`。

## 3. mixed-precision overflow 后，iteration、样本和 LR 谁推进

`optimizer.step()` 返回 `update_successful`。`train_step()` 先在 model-parallel group 上做逻辑 AND，保证一个模型副本的所有 shard 对“这次更新是否成功”意见一致，然后：

```python
if update_successful:
    opt_param_scheduler.step(increment=...)
    skipped_iter = 0
else:
    skipped_iter = 1
```

但是外层 `train()` 无论 `skipped_iter` 是否为 1，都会在本轮结束后增加 `iteration` 和 `consumed_train_samples`。原因是数据已经从 iterator 取出并参与前后向；把 sample counter 留在原地会让恢复/批量计划误以为这些样本未被消费。

所以准确语义是：

| 状态 | 成功更新 | overflow/跳过更新 |
|---|---:|---:|
| model/optimizer 参数 | 推进 | 不推进 |
| LR/WD scheduler | 推进 | 不推进 |
| iteration 日志位置 | 推进 | 推进，并记录 skipped iteration |
| consumed samples | 推进 | 推进 |
| loss scale | 由 optimizer/scaler 更新 | 通常回退以尝试后续 step |

checkpoint 需要保存这些状态的同一提交点，不能用“参数有没有更新”替代“数据有没有消费”这个独立问题。

源码锚点：`megatron/training/training.py:train_step`、`train`。

## 4. `forward_step` 为什么返回延迟执行的 loss closure

任务入口 `pretrain_gpt.forward_step` 不直接把 loss 变成一个最终 scalar，而是返回 output 与绑定了 `loss_mask` 的 closure。pipeline schedule 才知道当前 model chunk 是否是逻辑末 stage：

- 中间 stage：output 是要发送的 activation，不能在这里算 LM loss；
- 末 stage：调用 closure，得到 loss sum、有效 token 数和日志项；
- forward-only：可以收集非 loss 输出；
- MIMO/VPP：是否为末 stage 还要结合当前 virtual stage，而不是只看 global rank。

这个边界让 schedule 负责“什么时候算和怎样缩放”，任务代码负责“loss 数学定义”。如果任务代码提前做 DP all-reduce 或除 microbatch 数，schedule 再规约一次就会重复缩放。

源码锚点：`pretrain_gpt.py:loss_func`、`megatron/core/pipeline_parallel/schedules.py:forward_step_calc_loss`。

## 5. microbatch mean 与全局 token mean 为什么可能不同

GPT 默认 closure 返回的是：

```text
loss_sum_for_this_microbatch, valid_token_count, reporting_dict
```

当 `calculate_per_token_loss=False` 时，schedule 对每个 microbatch 做：

```text
loss_mb = loss_sum_mb / max(valid_tokens_mb, 1) / num_microbatches
```

所以最终是“各 microbatch token mean 的平均”。只有每个 microbatch 的有效 token 数相同，它才等于全局 token mean。packed/变长数据中，`valid_tokens_mb` 往往不同。

当 `calculate_per_token_loss=True` 时，schedule 不在每个 microbatch 内除 token 数，而是累计 `total_num_tokens`，在 `finalize_model_grads` 阶段按 DP/CP 语义对总有效 token 做最终梯度归一化。这才是变长样本下严格的 token-weighted mean 路径。

新增 loss 时要先明确目标是：

```text
sample mean / sequence mean / microbatch mean / valid-token mean
```

然后只让一个层次承担归一化。不能看到 loss 数值接近就认为梯度 scale 正确。

源码锚点：`pretrain_gpt.py:loss_func`、`schedules.py:forward_step_calc_loss`、`distributed/finalize_model_grads.py`。

## 6. batch 是不是每个 rank 都从 DataLoader 取一份

不是。典型 GPT 路径分两步：

1. TP rank 0 持有预处理后的 batch；`get_batch_on_this_tp_rank` 在 TP group 内广播需要的 tensor。
2. 若 CP>1，`get_batch_on_this_cp_rank` 再按 sequence 维为 CP ranks 分片。

PP 会进一步减少字段：首 stage 主要需要 tokens/position，末 stage 主要需要 labels/loss mask，中间 stage 在 packed sequence 下仍可能需要 `cu_seqlens` 等 metadata。MTP 在当前 rank 上时可能要求完整字段，因此不能硬编码“所有中间 PP stage 都不取 batch”。

这给出三个排错不变量：

- 同一 TP group 广播后、CP 分片前，逻辑 batch 应一致；
- 不同 DP replicas 应消费不同样本；
- CP 分片后 tensor 内容不同，但合起来必须是同一全局序列且 mask 使用全局坐标。

源码锚点：`megatron/core/utils.py:get_batch_on_this_tp_rank`、`get_batch_on_this_cp_rank`。

## 7. Column/Row Parallel 的 forward 与 backward 通信在哪

以 `Y=XA` 为例：

| 层 | 权重切法 | forward | backward |
|---|---|---|---|
| Column Parallel | `A` 按输出列切 | 每 rank 产生 `Y_i`；通常不 gather | `dA_i` 本地；replicated input 的 `dX` 要跨 TP 求和 |
| Row Parallel | `A` 按输入行切 | 每 rank 产生 partial `Y`，再 AR；SP 时改成 sequence RS | `dX_i` 本地，输入布局需要时由相邻层的 autograd mapping 恢复 |

容易误读的是：源码 forward 里没有显式 `all_reduce(dX)`，不代表反向没有通信。`copy_to_tensor_model_parallel_region` 的契约就是“forward copy，backward all-reduce”；`gather_from_sequence_parallel_region` 则是“forward AG，backward RS”。通信被自定义 autograd Function 放到了对应方向。

Column→激活→Row 成对使用时，中间大 activation 始终保持 hidden shard，不必先组成完整 `[tokens, 4H]`。

源码锚点：`megatron/core/tensor_parallel/layers.py:ColumnParallelLinear`、`RowParallelLinear`；`tensor_parallel/mappings.py`。

## 8. SP 与 CP 的实际切分为什么不能画成同一张图

SP 使用 TP group，把 LayerNorm、dropout、residual 等原本在 TP ranks 复制的 activation 沿 sequence 维分开。进入 Column Parallel 前常做 sequence AG，离开 Row Parallel 后做 sequence RS；attention 的全局 context 语义本身没有因 SP 改写。

CP 则建立独立 CP group，让 attention query 只持有全局 sequence 的一部分，并通过 KV/P2P 等协议看到所需上下文。当前 GPT batch 默认还使用 zigzag 来平衡 causal attention 计算。以 `S=8, CP=2` 为例，先切成四个长度为 2 的 chunk：

```text
chunk0=[0,1], chunk1=[2,3], chunk2=[4,5], chunk3=[6,7]
CP rank0 <- chunk0 + chunk3 = [0,1,6,7]
CP rank1 <- chunk1 + chunk2 = [2,3,4,5]
```

这不是简单的 rank0=`[0..3]`、rank1=`[4..7]`。后半段 token 的 attention 更贵，把序列头尾配对能均衡 causal work。packed sequence 则可按每个 document 调用 TE 的 partition-index 逻辑，且要求相应长度满足切分约束。

源码锚点：`megatron/core/utils.py:_get_batch_on_this_cp_rank_per_sequence_balancing`、`_get_batch_on_this_cp_rank_per_document_balancing`。

## 9. 为什么 TP 通常节点内，而 EP all-to-all 更敏感

TP collective 出现在几乎每层的主路径，消息往往与单个 GEMM 相邻。跨节点后，较低带宽和较高 latency 会被重复支付，且 TP size 过大还会把 GEMM 切小，降低计算效率。因此把变化最快的 TP ranks 放在 NVLink/NVSwitch 域通常最稳妥。

EP dispatch 的特征不同：每个 token 根据动态路由发到 expert owner，send/recv split 受本批路由分布影响。all-to-all 的完成时间取决于最重的 peer、最慢的 rank 和最拥塞链路；极端 imbalance 还会同时造成大消息与 expert GEMM 长尾。它不一定比 TP 总通信字节更多，但更容易出现不均匀、变长和多 peer 同步。

所以拓扑策略应写成“优先级 + 实测”，而不是绝对规则：TP 看每层 exposed collective，EP 看 token histogram、split size、A2A tail 和 grouped-GEMM shape。

源码锚点：`parallel_state.initialize_model_parallel` 的 rank order；`megatron/core/transformer/moe/token_dispatcher.py`。

## 10. collective hang 为什么经常不是 NCCL 自身的 bug

collective 是组内 rank 共同执行的有序协议。第 `k` 次 collective 至少要满足：

```text
相同 process group
相容的 collective 类型
相容的 tensor count / shape / dtype
所有成员都最终到达
```

常见因果链是：rank 3 在更早的数据或 Python 代码抛异常；其余 ranks 继续进入下一次 all-reduce，于是最后看见的是 NCCL timeout。另一类是 PP/VPP 条件分支用 global rank 判断，导致某个 group 少一次调用。报错点是“等待发生处”，第一因却在更早的控制流分叉。

定位时给通信调用记录 `(iteration, microbatch, op_seq, group_name, numel, dtype)`，对比首个不一致序号；同时先查所有 rank 的第一条 exception。只把 timeout 调大，会延后症状而不会修复协议不一致。

## 11. PP 发送 activation 后为什么还能反向

`deallocate_output_tensor` 在 forward activation 发给下一 stage 后，把 `out.data` 换成单元素 tensor，释放大 payload，但 Python Tensor 对象与 `grad_fn` 仍存在。backward 收到下一 stage 发回的完整 gradient 后，`custom_backward` 直接调用 autograd engine，绕过普通 `torch.autograd.backward` 对 output/grad shape 的检查。

它能成立不是因为 autograd “完全不需要 forward 数据”，而是反向节点需要的输入已经由 saved tensors 保存，或者会由 activation recompute 恢复。被替换的是 graph head 的 output payload。

因此：

- send 后读取 `output.data` 的自定义监控会只看到一个元素；
- hook 应放在 deallocate 之前，或观察 autograd saved tensor；
- 自定义 Function 若偷偷依赖 output payload，而没有正确 `save_for_backward`，会在此优化下暴露错误。

源码锚点：`megatron/core/pipeline_parallel/schedules.py:deallocate_output_tensor`、`custom_backward`。

## 12. Distributed Optimizer 与 Megatron-FSDP 的关键边界

两者都沿数据并行域减少状态冗余，但参数驻留模型不同：

| 方案 | 训练参数常态 | gradient | optimizer/main state | 主要额外生命周期 |
|---|---|---|---|---|
| DDP | replicated | AR | replicated | bucket overlap |
| DDP + Distributed Optimizer | 低精度 model param 通常常驻为 replica | optimizer 读 RS local view；完整 grad buffer 通常仍分配 | 分片 | step 后/forward 前 param AG |
| Megatron-FSDP `optim` | model param replicated | replicated/AR | optimizer 分片 | 不要求 FSDP unit |
| Megatron-FSDP `optim_grads` | model param replicated | 分片/RS | 分片 | 不要求 FSDP unit |
| Megatron-FSDP `optim_grads_params` | training param 分片，计算前 materialize | 分片/RS | 分片 | FSDP unit 的 AG、释放、prefetch |

Distributed Optimizer 的“local grad shard”描述的是 optimizer ownership，不表示完整 DDP grad storage 自动缩成 `1/DP`；`param_and_grad_buffer.py:start_grad_sync` 从完整 `bucket.grad_data` 取 local view 作为 reduce-scatter 输出。只有参数也分片的 `optim_grads_params` 才需要用 FSDP unit 限定“参数在哪段 forward/backward 内必须保持 materialized”。unit 太大提高峰值，太小增加 collective/launch 次数；还可能启用 fine-grained param gather，让子模块 hook 管更小的参数桶。

选择时先问瓶颈：如果主要是 optimizer state，Distributed Optimizer 或较浅 FSDP 策略可能已足够；若训练参数 replica 本身放不下，才必须承担 stage-3 类参数 AG 生命周期。不要仅用“ZeRO-几”替代对 buffer、hook 和 schedule 的核对。

源码锚点：`megatron/core/optimizer/distrib_optimizer.py`、`megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py:ShardingStrategy`、`mcore_fsdp_adapter.py:FullyShardedDataParallel`、`megatron_fsdp.py:MegatronFSDP`。

## 13. checkpoint 能重分片什么，不能自动解决什么

`ShardedTensor` 描述同一逻辑 global tensor 的 `global_shape`、`global_offset`、local shape、replica id 等。保存端验证主 replica 是否无重叠、无空洞；加载端由当前目标 sharded state dict 声明“新拓扑上的本 rank 需要哪一片”，backend 再重组数据。

因此通常可处理：

- DP 改变；
- 已正确声明 shard axis 的 TP/PP 改变；
- optimizer state 能映射回对应 model parameter，且保存格式支持该重分片路径。

不能自动推导：

- QKV layout、GQA groups、SwiGLU 拼接方式改变后的数学映射；
- layer 插入/删除/重排；
- vocab/tokenizer 语义改变；
- shared/unshared embedding 改变；
- 缺失或错误的 `sharded_state_dict()` metadata。

后者需要显式 conversion mapping，不属于存储 backend 的职责。“能够 load”只说明结构契约通过，固定输入 logits 与续训下一步才验证语义。

源码锚点：`megatron/core/dist_checkpointing/mapping.py:ShardedTensor`、`validation.py:validate_sharding_integrity`、各 module 的 `sharded_state_dict`。

## 14. async checkpoint 什么时候才算完整

`dist_checkpointing.serialization.save(..., async_sharded_save=True)` 返回 `AsyncRequest`，调用者还必须调度并最终 finalize。当前实现把 `metadata_finalize_fn` 放进 finalize callbacks：所有 shard 完成后才由 rank 0 写 checkpoint backend/version metadata，并做 barrier。

这意味着目录存在、部分 tensor 文件可见，都不等于 checkpoint 已发布。恢复/清理工具应以完整性 metadata/manifest 和上层 tracker 的提交语义判断，不能扫描到新目录就立即加载。开启 integrity manifest 时还会在所有数据写完后额外读文件计算 hash，可靠性更强但有额外 I/O。

源码锚点：`megatron/core/dist_checkpointing/serialization.py:save`、`megatron/training/checkpointing.py`。

## 15. MoE auxiliary loss 是否要在任务 loss 中手工相加

当前实现不要求 `pretrain_gpt.loss_func` 再返回 `lm_loss + aux_loss`。router 计算 load-balancing/z-loss 后，通过 `MoEAuxLossAutoScaler.apply(activation, aux_loss)` 把 auxiliary loss 接入 autograd graph；其 forward 保持 activation 不变，backward 给 aux loss 注入相应梯度。

schedule 的 `forward_step_calc_loss` 再根据 microbatch、CP、loss scale 和 `calculate_per_token_loss` 设置 `MoEAuxLossAutoScaler` 的 scale。这样 auxiliary loss 与主 loss 共用一致的 accumulation/缩放边界。

真正的风险是：

- 配置 coeff 为 0，误以为框架仍会自动平衡；
- 在任务 closure 又手工加一次，造成双计；
- 自定义 schedule 没有设置 autoscaler scale；
- per-token 与 legacy microbatch mean 路径切换后，aux scale 没同步验证。

日志 tracker 记录 auxiliary loss，不代表用户还应把该日志标量加入 loss。

源码锚点：`megatron/core/transformer/moe/router.py:_apply_aux_loss`、`moe_utils.py:MoEAuxLossAutoScaler`、`schedules.py:forward_step_calc_loss`。

## 16. router replay 固定 expert 后，router 还有没有梯度

replay 保存/提供的是 top-k indices。`REPLAY_FORWARD` 或 `REPLAY_BACKWARD` 下，代码不重新执行 top-k，而是：

```python
top_indices = recorded_indices
probs = scores.gather(1, top_indices)
```

因此它固定“选哪几个 expert”，但这些 expert 的当前 routing score/probability 仍从本次 `scores` gather，梯度仍可沿被选 score 回传。它不是冻结 router 参数，也不是保存完整 router logits。

`RECORD`、`REPLAY_FORWARD`、`REPLAY_BACKWARD` 要与 layer/microbatch 顺序严格对应；backward replay 使用队列，顺序错位会把另一 microbatch 的 expert IDs 套到当前 token。全局实例列表还要在多次模型构建/测试之间清理，避免实例数与 replay tensor 数不匹配。

源码锚点：`megatron/core/transformer/moe/router_replay.py:RouterReplay`、`router.py:TopKRouter`。

## 17. overlap 打开后为什么可能不快，甚至更慢

overlap 只把通信搬到另一个可并行窗口，不会消除通信。收益近似取决于：

```text
exposed_comm_after = max(comm_time - coverable_compute_window, tail_and_sync)
```

为了制造 overlap，框架可能拆 bucket、增加 stream/event、使用额外 buffer，并把大 collective 变成更多小 collective。当前 DDP 还只在 PP rank 0 默认启用细粒度 bucketing；后续 PP stage 的 DP 通信不在相同 critical path，源码会把 bucket size 设为 `None`，避免无收益拆桶。

所以验收要比较端到端 step time 与 exposed tail，而不是只看 NCCL kernel 与 GEMM 在时间线上有重叠。常见“看起来 overlap、吞吐不变”的原因是通信本来不在 critical path，或新增 buffer/launch 开销抵消了遮蔽收益。

源码锚点：`megatron/core/distributed/distributed_data_parallel.py:DistributedDataParallel.__init__`、各 `start_*_sync`/`finish_*_sync`。

## 18. strict resume 为什么比固定 seed 多得多

严格续训要求恢复同一条状态轨迹：model、optimizer/master state、scheduler、loss scaler、iteration/consumed samples、data iterator/sampler、Python/NumPy/PyTorch/CUDA RNG、TP RNG tracker，以及低精度 recipe 的动态状态。

即便这些都恢复，改变 TP/PP/DP 后 collective 次序、浮点归约树或 kernel 也可能改变，通常只能要求容差内数值接近，而不是 bitwise 相等。FP8 delayed scaling 若 amax/scale history 没恢复，恢复后前若干 step 尤其容易偏离。

最有区分力的测试不是最终 validation，而是：

```text
连续 N+M step
vs.
N step -> save -> 新进程 load -> M step
```

比较恢复后第一批 sample IDs、第一步 loss/grad norm、optimizer m/v、少量参数 hash，再逐层缩小差异。第一步就不同优先查数据/RNG/scale；多步后才漂移再查非确定 kernel、归约次序和异步 race。

源码锚点：`megatron/training/checkpointing.py`、`megatron/core/tensor_parallel/random.py`、Transformer Engine extra state。
