# 03. Rank 网格、进程组、Tensor/Sequence Parallel

## 1. 先画 rank 网格

`initialize_model_parallel()` 位于 `megatron/core/parallel_state.py:547`，`RankGenerator` 位于约 446 行。它把线性的 global rank 映射到多维并行坐标，再为不同维度组合创建 process group。

假设：

```text
TP=2, PP=2, DP=2, world=8, order="tp-pp-dp"
```

若 TP 是变化最快维度，可把 rank 想成：

| DP | PP | TP | global ranks |
|---:|---:|---:|---|
| 0 | 0 | 0,1 | 0,1 |
| 0 | 1 | 0,1 | 2,3 |
| 1 | 0 | 0,1 | 4,5 |
| 1 | 1 | 0,1 | 6,7 |

于是 rank 0：

- TP group `{0,1}`
- PP group `{0,2}`
- DP group `{0,4}`

真实配置还可能有 CP、EP、embedding group、position embedding group、data-modulo-expert group。必须用实际 `order` 和代码验证。

### `RankGenerator` 的核心思想

多维坐标展平：

```text
rank = Σ coordinate[i] × stride[i]
```

生成某个 group，就是固定非目标维度坐标，枚举目标维度坐标。代码复杂主要因为：

- EP 与 DP 的关系；
- independent EP 维度；
- order 中允许/禁止的相邻关系；
- 多种 data-parallel group（是否包含 CP、是否 modulo EP）。

调试 group 最有效的方法是启动时让每个 rank 输出所有语义 rank 和 group members，然后按 global rank 排序。

## 2. Column Parallel Linear

`ColumnParallelLinear` 位于 `megatron/core/tensor_parallel/layers.py:778`。对 `Y = X A`，按 A 的输出列切：

```text
A = [A0, A1]
Y0 = X A0
Y1 = X A1
```

每个 rank 持有 `A_i`，输入 X 通常相同，本地得到输出 shard。若下游能消费 shard，无需立刻 all-gather；若要求完整 Y 才 gather。

反向：

- `dA_i = Xᵀ dY_i` 本地计算；
- `dX_i = dY_i A_iᵀ` 后需要跨 TP 求和得到完整 `dX`。

代码中的 autograd mapping 会把这些通信隐藏在 copy/gather/reduce primitive 后面。

## 3. Row Parallel Linear

`RowParallelLinear` 位于 `layers.py:1142`。按 A 的输入行切，同时 X 按最后一维切：

```text
X = [X0, X1]
A = [A0; A1]
Y = X0 A0 + X1 A1
```

各 rank 算 partial output，随后 all-reduce（或与 sequence parallel 配合的 reduce-scatter）得到结果。

Column→激活→Row 的组合避免在 MLP 中间巨大维度上收集完整 tensor。

## 4. Vocab parallel

`VocabParallelEmbedding` 位于 `layers.py:198`。每个 TP rank 持有词表范围 `[vocab_start, vocab_end)`：

1. 对不在本 rank 范围的 token 做 mask；
2. 本地 token id 减去 `vocab_start`；
3. embedding lookup；
4. mask 位置置零；
5. TP all-reduce 得到完整 embedding。

parallel cross entropy 同理避免收集完整 `[tokens, vocab]` logits：先做跨 shard global max，再 global sum-exp，并只从拥有 target id 的 rank 取 target logit。这是数值稳定分布式 log-sum-exp 的典型模板。

## 5. Sequence Parallel 不等于 Context Parallel

SP 主要切 LayerNorm/Dropout 等原本在 TP ranks 上复制的 activation sequence 维；attention/MLP 的 TP 区域仍按 tensor shard 工作。典型变换：

```text
sequence-sharded
  --all-gather--> TP layer 输入
  --reduce-scatter--> sequence-sharded 输出
```

CP 则让 attention 本身处理 context shard，目标是长序列下的 attention activation/KV。二者可以同时存在。

### 为什么 TP 常配 SP

没有 SP 时，TP 虽切了权重和部分 activation，但 LayerNorm/residual 等 activation 仍在每个 TP rank 复制。长序列下这部分显存可观；SP 用已有通信的 AG/RS 形式保持计算语义同时去掉复制。

## 6. 通信重叠的正确理解

`--tp-comm-overlap` 不是“免费加速”。要生效通常需要：

- 足够大的 GEMM，可遮住 collective；
- TE/通信 backend 支持；
- shape 满足分块；
- 额外通信 buffer 可分配；
- stream 依赖正确。

若 kernel 很小，拆分会增加 launch latency；若网络慢于可覆盖窗口，尾部仍暴露；若 overlap buffer 导致 OOM，整体反而失败。

## 7. hang 排查清单

1. 所有 rank 是否以相同顺序创建 group？
2. 同一 group 中是否所有成员都进入相同 collective？
3. tensor shape/count/dtype 是否一致？
4. rank 是否在更早的 Python exception 后退出？
5. PP stage 条件是否误用了 global rank？
6. 设置 `TORCH_DISTRIBUTED_DEBUG=DETAIL`、`NCCL_DEBUG=INFO`，配合每个 collective 前后的序号日志。

不要在每个 rank 盲目打印海量日志；用 `(iteration, microbatch, op_seq, group_name, shape)` 可直接比较。

## 8. `RankGenerator` 源码精读

固定快照 `parallel_state.py:446`：

```python
class RankGenerator:
    def __init__(self, tp, ep, dp, pp, cp, order, rank_offset=0):
        self.world_size = tp * dp * pp * cp * ep
        self.name_to_size = {
            "tp": tp, "pp": pp, "dp": dp, "ep": ep, "cp": cp
        }
        for token in order.split("-"):
            self.ordered_size.append(self.name_to_size[token])

    def get_ranks(self, token):
        mask = self.get_mask(self.order, token)
        return generate_masked_orthogonal_rank_groups(
            self.world_size, self.ordered_size, mask
        )
```

`mask=True` 的维度在组内变化，其他维度固定。例如 `order=tp-dp-pp`，请求 `"tp-dp"` 会固定 PP 坐标，枚举 TP×DP。

源码还有关键断言：同一个 `RankGenerator` 不允许 EP 与 CP 同时大于 1；默认网格和 expert 网格分开生成。这就是为什么简单公式不能完整描述 expert/data groups。

### 手算 stride

`order=tp-cp-dp-pp`，size `[2,2,2,2]`，变化最快的是 TP：

```text
stride(tp)=1
stride(cp)=2
stride(dp)=4
stride(pp)=8
rank = tp + 2×cp + 4×dp + 8×pp
```

global rank 13：

```text
pp=1, remainder=5
dp=1, remainder=1
cp=0, tp=1
```

它的 TP group 固定 `(cp=0,dp=1,pp=1)`，成员 `{12,13}`。

## 9. Column/Row Parallel 的反向表

| Layer | forward 输入 | forward 输出 | forward 通信 | backward 关键通信 |
|---|---|---|---|---|
| Column | X replicated/SP | Y 按 output 切 | 可选 AG X | dX all-reduce/RS |
| Row | X 按 input 切 | Y replicated/SP | AR/RS partial Y | dX 本地，必要时 AG |

以 `X:[m,k]`, `A:[k,n]`, TP=2：

```text
Column:
  rank0 A0:[k,n/2] → Y0:[m,n/2]
  rank1 A1:[k,n/2] → Y1:[m,n/2]

Row:
  rank0 X0:[m,k/2], A0:[k/2,n] → P0
  rank1 X1:[m,k/2], A1:[k/2,n] → P1
  Y=P0+P1
```

Megatron 原始 MLP 设计把 Column 和 Row 连起来，使中间 `[m,4h]` 永不完整落在单卡。

## 10. 词表并行交叉熵的数值推导

每 rank 持有 logits shard `z_r`。全局：

```text
m = max_r(max(z_r))
denom = Σ_r Σ_j exp(z_rj - m)
loss = log(denom) + m - target_logit
```

实现需：

1. TP all-reduce MAX 得 `m`；
2. 本地减 m 后 exp/sum；
3. TP all-reduce SUM 得 denominator；
4. 只有拥有 target id 的 rank 提取 target logit，其他置零；
5. TP all-reduce SUM 得全局 target logit。

这避免收集全词表 logits。若 target id 没减本 rank vocab start，读到的就是错误位置。

## 11. 进程组观测脚本

可临时在初始化后打印：

```python
info = {
    "g": torch.distributed.get_rank(),
    "tp": parallel_state.get_tensor_model_parallel_rank(),
    "pp": parallel_state.get_pipeline_model_parallel_rank(),
    "dp": parallel_state.get_data_parallel_rank(),
    "cp": parallel_state.get_context_parallel_rank(),
}
print(info, flush=True)
torch.distributed.barrier()
```

输出会乱序，最好每个 rank 写独立文件或用 rank0 gather object。调试结束必须删除 barrier，避免污染性能。

## 12. 本章延伸阅读

- [Megatron-LM 论文](https://arxiv.org/abs/1909.08053)：精读 Figure 3 的 MLP 和 attention TP。
- [Megatron Core 并行策略指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)：核对 TP/SP 的推荐组合。
- [Tensor Parallel API](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/tensor_parallel.html)：查 `ColumnParallelLinear`、`RowParallelLinear` 与 mapping primitives。
- [PyTorch Distributed collectives](https://pytorch.org/docs/stable/distributed.html)：确认 collective 的输入输出与异步 handle 语义。
