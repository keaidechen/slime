# Pipeline Parallel：1F1B、交错调度与 P2P

## 1. schedule 选择

`get_forward_backward_func()` 位于 `megatron/core/pipeline_parallel/schedules.py:48`，按 PP/VPP 状态返回：

- `forward_backward_no_pipelining`（约 672 行）；
- `forward_backward_pipelining_without_interleaving`（约 2127 行）；
- `forward_backward_pipelining_with_interleaving`（约 984 行）。

训练主循环只调用统一接口，schedule 负责 microbatch 顺序和 stage 通信。

## 2. GPipe 与 1F1B

以 PP=4、microbatch=8 为例：

- all-forward-all-backward：先把 8 个前向都推进，再反向；activation 峰值高；
- 1F1B：warmup 后每做一个 forward 尽快做一个 backward；限制在途 activation。

非交错 1F1B 分三段：

```text
warmup：只 forward，让流水线填满
steady：一个 forward + 一个 backward
cooldown：只 backward，让流水线排空
```

rank 越靠前，warmup microbatch 越多。最后 stage 最早获得 loss 并开始 backward。

理论 bubble 的粗略比例可看作 `(PP-1)/(M+PP-1)`，M 是 microbatch 数。它忽略通信、stage 不均和重叠，但足以解释为什么 M 太小会很差。

## 3. P2P 的 tensor 语义

相邻 stage 交换：

- forward：activation 从 stage i 发到 i+1；
- backward：activation gradient 从 i+1 发回 i。

`megatron/core/pipeline_parallel/p2p_communication.py` 封装 send/recv 组合。优化实现会把 `send_forward_recv_backward` 等操作批处理或重叠，减少同步点。

排查 P2P shape mismatch 时核对：

- sequence-first/batch-first；
- SP 后 sequence 长度是否除以 TP；
- decoder-only 与 encoder-decoder 的 tensor 数；
- variable sequence length；
- packed sequence；
- pipeline layout 中相邻模块的接口。

## 4. activation 的“伪释放”

schedule 中的 `deallocate_output_tensor()` 会在 activation 发出后把 tensor 的 `.data` 缩成 scalar，但保留 `.grad_fn`。普通 `torch.autograd.backward` 会检查 output 和 grad shape 相同，因此代码用 `custom_backward()` 直接调用 C++ autograd engine。

这是一处不常见但关键的优化：

```text
发送前：output.data 占完整 activation
发送后：保留 autograd 图，丢弃不再需要的 payload
反向时：用下一 stage 发回的完整 grad 驱动 grad_fn
```

风险：如果自定义代码在 send 后仍读取 output data，就会看到被缩小的 tensor。不能把“保留 Python Tensor 对象”误解为“数据仍在”。

## 5. virtual pipeline / interleaving

VPP 让一个物理 rank 持有多个不连续 model chunks。例如 16 层、PP=4、每 rank 两个 chunk，可近似布置：

```text
rank0: layers 0-1, 8-9
rank1: layers 2-3, 10-11
rank2: layers 4-5, 12-13
rank3: layers 6-7, 14-15
```

调度在虚拟 stage 间交错，缩短 bubble 单元；代价是：

- model 变成 list；
- 数据 iterator 可能也是 list；
- P2P 邻接关系和 first/last stage 判断带 virtual rank；
- 参数同步、embedding sync、checkpoint key 更复杂；
- 通信次数增加。

`forward_backward_pipelining_with_interleaving()` 中最难读的是 microbatch→model chunk 映射、warmup 数、forward/backward chunk 顺序。建议用 PP=2、VPP=2、M=8 手工列一张表，再对照辅助函数。

## 6. 自定义 pipeline layout

`megatron/core/transformer/pipeline_parallel_layer_layout.py` 和官方 `pipeline_parallel_layout` 文档支持非均匀 layer 类型/数量。MoE、MTP、embedding、loss 的成本不同，按“层数均分”未必按时间均分。

合理切分流程：

1. 单层/模块 profile 得到 forward+backward 时间和 activation size；
2. 以 stage 最大时间最小化为目标切分；
3. 同时控制 P2P tensor 大小和显存；
4. 多 microbatch 验证真实 steady state；
5. 检查首末 stage 的 embedding/output 额外成本。

## 7. 常见故障

- 吞吐低但 GPU 利用率周期性空洞：M 太小、stage 不均、P2P 暴露；
- OOM 只出现在早期 stage：warmup 在途 activation 多；
- interleaved 后 loss 错：chunk iterator 或 first/last virtual stage 判断错；
- hang 在 backward recv：某 stage 提前异常或发送 tensor 数/shape 不一致；
- recompute 没降预期显存：检查 checkpoint 粒度和 schedule 保留的 activation。

## 8. 非交错 schedule 源码精读

入口 `schedules.py:2127` 首先强制一个本地 model chunk：

```python
if isinstance(model, list):
    assert len(model) == 1
    model = model[0]
if isinstance(data_iterator, list):
    assert len(data_iterator) == 1
    data_iterator = data_iterator[0]

config = get_model_config(model)
if config.overlap_p2p_comm:
    raise ValueError(
        "Non-interleaved pipeline parallelism does not support "
        "overlapping p2p communication"
    )
```

这三点很重要：

- API 统一接受 list，但非交错只能有一个 chunk；
- iterator 与 chunk 一一对应；
- 当前快照非交错 schedule 明确拒绝 `overlap_p2p_comm`，不能看到 flag 就假设任意 schedule 可用。

接着构建 `P2PCommunicator`，从 PP group 获取邻接 ranks，并计算 tensor shapes。SP 开启时，文档明确指出 sequence 维会除以 TP size；若自定义 tensor 的 sequence 维没有同样切分，recv buffer 大小会错。

## 9. warmup 数量手算

非交错 1F1B 中，PP rank `r` 的典型 warmup：

```text
warmup = min(PP - r - 1, num_microbatches)
```

PP=4、M=8：

| PP rank | warmup F | steady 1F1B 次数 | cooldown B |
|---:|---:|---:|---:|
| 0 | 3 | 5 | 3 |
| 1 | 2 | 6 | 2 |
| 2 | 1 | 7 | 1 |
| 3 | 0 | 8 | 0 |

表中的 forward/backward 总数都为 8。rank0 同时保留最多在途 activation，因此常是 activation OOM 首发位置。

## 10. `deallocate_output_tensor` 精读

等价逻辑：

```python
def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    if not deallocate_pipeline_outputs:
        return
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)
```

它只换 `.data`，保留 autograd graph。反向时：

```python
Variable._execution_engine.run_backward(
    tensors=(output,),
    grad_tensors=(output_grad,),
    ...
)
```

绕过 Python `torch.autograd.backward` 的 shape check。这里的安全前提是 backward 节点真正需要的输入已被 autograd saved tensors 保存或通过 recompute 恢复，而不再需要 `output.data`。

自定义 hook 若在 send 后做 `output.detach().clone()`，得到的可能只有一个元素。应把观测放在 deallocate 前或用 autograd saved tensor hook。

## 11. interleaved 的 chunk 映射

设 PP=2、VPP=2、M=8，逻辑 stage 数是 4，但物理放置类似：

```text
physical rank0: virtual stage 0, 2
physical rank1: virtual stage 1, 3
```

microbatch→chunk 不是简单 `mb % 2`，还要区分 forward/backward 方向和一个 group 内的 microbatch 数。阅读 `get_model_chunk_id` 一类辅助函数时，应列：

```text
(clock, physical_rank, microbatch_id, chunk_id, F/B, send_to, recv_from)
```

然后检查每个 `(microbatch,chunk)` 恰好一次 forward/一次 backward。

## 12. 自定义 layout 例子

官方 layout 字符串例如：

```bash
--pipeline-model-parallel-layout "Et*3|(tt|)*29,m|L"
```

符号 `E/t/m/L` 分别代表 embedding、decoder layer、MTP、loss，`|` 切 stage，括号与乘号重复。它解决 DeepSeek 类模型中 embedding/MTP/loss 与普通 layer 代价不等的问题。

不要直接照抄 layout：先对目标 GPU 和实际 seq/batch profile，因为 attention、MoE、MTP 的相对成本会随 shape 改变。

## 13. 本章延伸阅读

- [Pipeline Parallel API](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/pipeline_parallel.html)：查 schedule 与 P2P communicator。
- [高效大规模流水线训练论文](https://arxiv.org/abs/2104.04473)：interleaved 1F1B 与 sequence parallel 的理论来源。
- [自定义 Pipeline Layout 官方指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/pipeline_parallel_layout.html)：包含 DeepSeek-V3 PP16/VPP2 的完整布局表。
- [GPipe 论文](https://arxiv.org/abs/1811.06965)：作为 all-forward/all-backward baseline 理解 bubble 与 activation。
