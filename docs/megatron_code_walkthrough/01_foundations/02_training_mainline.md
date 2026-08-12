# 从入口到一次参数更新

## 1. 三个用户注入点

`Megatron-LM/pretrain_gpt.py` 最值得先读的不是参数列表，而是传给训练框架的三个函数：

- `model_provider(...)`：构建当前 PP/VPP stage 应拥有的模型部分；
- `train_valid_test_datasets_provider(...)`（约 444 行）：构建数据集；
- `forward_step(...)`：取 batch、调用 model，返回 output 与延迟执行的 `loss_func`。

最终调用 `megatron.training.pretrain(...)`。这种设计把“训练框架控制流”和“任务特定逻辑”分开：pipeline schedule 并不知道 GPT loss 的细节，只遵守 `forward_step_func` 契约。

## 2. `pretrain()` 的阶段

入口位于 `megatron/training/training.py:1013`。建议按阶段而不是逐行阅读：

```text
解析配置/初始化分布式
  → 设置随机数、JIT/融合、全局 timer
  → 构建 model + optimizer + lr scheduler
  → 加载 checkpoint
  → 构建 train/valid/test iterator
  → train()
  → validation/test/checkpoint
```

初始化顺序很重要。进程组必须在依赖 rank topology 的模型 shard 之前创建；checkpoint 必须在 optimizer、scheduler 和模型对象存在后恢复；数据 sampler 又必须知道 consumed samples。

`setup_model_and_optimizer()` 位于 `training.py:1999`，负责：

1. `get_model()` 按 PP/VPP 构建一个或多个 model chunk；
2. 包装 mixed precision / DDP；
3. 创建 optimizer 和 optimizer parameter scheduler；
4. 根据配置加载或初始化状态。

看到 `model` 是 list 不要惊讶：interleaved pipeline 下，一个物理 rank 持有多个 virtual stage（model chunk）。

## 3. `train()` 与 `train_step()`

长期循环位于 `training.py:3278`，单步边界位于 `training.py:2290`。抽象后的单步是：

```python
optimizer.zero_grad()
losses = forward_backward_func(
    forward_step_func=forward_step_func,
    data_iterator=data_iterator,
    model=model,
    num_microbatches=get_num_microbatches(),
    ...
)
update_successful, grad_norm, num_zeros = optimizer.step()
if update_successful:
    opt_param_scheduler.step(increment=...)
```

真实代码还插入了 timer、rerun state machine、vision grads、embedding sync、参数 hash 检查等，但主因果链不变。

### 延迟 loss closure 为什么关键

`forward_step` 返回 `(output_tensor, loss_func)`，最后一个 pipeline stage 才执行 loss closure。这样 schedule 可以统一处理：

- 中间 stage：output 是要发给下一 stage 的 activation；
- 最后 stage：output 被 loss closure 规约为 scalar；
- forward-only：closure 可返回任意收集结果；
- token-level loss：可携带 token count 做正确归一化。

`megatron/core/pipeline_parallel/schedules.py` 的 `forward_step()` 文档约定支持两元或三元 loss 返回。修改任务时最容易出错的是在 microbatch、token、DP 三个层次重复或遗漏归一化。

## 4. microbatch calculator

不要把 micro batch、global batch 和 gradient accumulation 混为一谈：

```text
M = global_batch / (micro_batch × DP)
```

每次 optimizer update 调用一次 `forward_backward_func`，内部跑 M 个 microbatch。当前快照的 step batch-size schedule 会让 M 随 consumed samples 改变；旧 `rampup_batch_size` 参数已被 calculator 标为 deprecated 并忽略。PP schedule、loss scale、LR scheduler 的 sample increment 都依赖当前 M。

例：GBS=512、MBS=2、DP=32，则 M=8。PP=8 时只有 8 个 microbatch，恰好能填满但 bubble 仍明显；若 M=1，绝大多数 stage 会等待。

## 5. 数据如何到各个 rank

典型 GPT batch 只在 TP source rank 取数据，再在 TP group 中 broadcast，使同一 TP 组看到一致 token/label/mask。PP 的非首/末 stage 可能不需要完整 batch，但具体字段由 `get_batch_on_this_tp_rank` 一类工具决定。

infra 排障时要确认：

- DP rank 的 sample 是否不同；
- 同一 TP group 的 token 是否一致；
- consumed samples 是否随 checkpoint 正确恢复；
- packed sequence 的 `cu_seqlens`、position IDs、loss mask 是否一致。

## 6. 一次 step 的状态提交边界

mixed precision 下 `optimizer.step()` 可能因 overflow 失败。只有成功时才推进 LR/WD schedule 和参数更新计数；但外层 `train()` 无论本轮是否因 overflow 跳过参数更新，都会推进 iteration 与 `consumed_train_samples`，因为 batch 已从 iterator 消费。日志会用 `skipped_iter` 区分两种状态。checkpoint 最安全的语义是保存完整 iteration 边界上的整组状态，而不是半个 accumulation window。逐项状态表见[源码问题详解第 3 节](../06_reference/03_source_questions.md)。

## 7. 动手走读

在小模型上给以下符号加 rank-aware 日志（只打印必要 rank）：

1. `pretrain_gpt.forward_step`
2. `schedules.forward_step`
3. `schedules.backward_step`
4. `training.train_step`
5. `optimizer.step`

每条日志打印 `(global_rank, tp_rank, pp_rank, dp_rank, microbatch_id, tensor.shape)`。你应能从日志还原出单卡、TP=2、PP=2 三种时间线。

## 8. 固定快照源码精读

### `train_step` 的真正边界

在 `megatron/training/training.py:2290`，简化后的关键代码为：

```python
while rerun_state_machine.should_run_forward_backward(data_iterator):
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        forward_only=False,
    )
```

逐行理解：

1. 外层不是普通的“一次循环”，而受 rerun state machine 控制；检测到可疑结果时，框架可能重放 forward/backward。
2. `model_chunk.zero_grad_buffer()` 清 Megatron 连续 grad buffer；只调 `optimizer.zero_grad()` 不一定够。
3. `model` 始终按 list 处理，为 VPP 的多个 model chunk 保留统一接口。
4. `forward_backward_func` 已经包含所有 microbatch 的前后向，不是单个 microbatch。
5. 返回的 `losses_reduced` 主要供日志；真正 gradient 已写入 buffer。

当前版本还在调用前处理 `reuse_grad_buf_for_mxfp8_param_ag`：参数 all-gather buffer 与 grad buffer 复用时，清零次序影响参数是否被覆盖。这说明“显存复用”改变了控制流正确性，不能只当 allocator 优化。

### `forward_step`/loss closure 的调用时机

schedule 接收的契约近似：

```python
def forward_step(data_iterator, model):
    tokens, labels, loss_mask, ... = get_batch(data_iterator)
    output = model(tokens, ...)
    return output, partial(loss_func, loss_mask)
```

中间 PP stage 的 `output` 直接进入 P2P；末 stage 才调用 closure。若 closure 中执行 DP all-reduce，需要保证只有拥有 loss 的 stage 进入对应 group，且日志 reduction 不应参与 autograd。

### 一次更新的 shape 账本

假设 `GBS=64, MBS=2, DP=4, PP=2, TP=2`：

```text
num_microbatches = 64 / (2×4) = 8
每个 DP replica 本次处理 16 samples
每个 microbatch 2 samples
每个 PP stage 执行 8 次 forward + 8 次 backward
每个 TP rank 只持有其 layer shard，但看到相同 microbatch token
```

默认路径若先对每个 microbatch 的有效 token 求平均、schedule 再除 microbatch 数，得到的是“microbatch mean 的平均”。只有各 microbatch 有效 token 数相同，它才等于全局 token mean；packed/变长数据应核对 `calculate_per_token_loss` 路径。若自定义 loss 已除以 8 又被 schedule 再除一次，gradient 会缩小 8 倍。完整推导见[源码问题详解第 5 节](../06_reference/03_source_questions.md)。

## 9. 推荐打断点顺序

1. `pretrain_gpt.py` 最底部 `pretrain(...)`；
2. `training.pretrain` 创建模型前后；
3. `setup_model_and_optimizer`；
4. `train` 调 `train_step`；
5. `get_forward_backward_func`；
6. `schedules.forward_step`；
7. `pretrain_gpt.forward_step` 与 `loss_func`；
8. optimizer `step()` 返回处。

多进程不宜直接同时用交互 debugger。先在单卡断点，再用 rank-filter 日志：

```python
if torch.distributed.get_rank() in {0, 1}:
    print(f"rank={...} pp={...} shape={tuple(output.shape)}", flush=True)
```

## 10. 本章延伸阅读

- [Megatron Core Quickstart](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)：把最小训练命令与入口代码对应起来。
- [训练示例](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/training-examples.html)：核对 batch、并行和模型参数。
- [Pipeline schedule API](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/pipeline_parallel.html)：重点读 `forward_step_func` 的返回契约。
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)：查看当前主线与本仓库固定 commit 的差异。
