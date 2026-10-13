# SP 执行走读：激活分片之后，谁负责梯度

基线：本仓库 Megatron commit `21fe0fe1597932421f1bb0efd93f469376a7a255`。先读 [TP 的 Column/Row 组合](<01_parallel_state_and_tensor_parallel.md>) 和 [SP/CP 数值推演](<../../../../04_Training/15_模型并行与执行日程.md#s4-数值展开-sp-与-cp布局转换在线-softmax-与远端梯度>)。本章采用 dense MLP、TP=2、无 CP 的示例。

## 1. 出发点：权重切了，norm 与 residual 激活还在复制

TP 的 FC1/FC2 可以分摊大矩阵，但 LayerNorm/RMSNorm、dropout 与 residual 的一些激活仍可能在 TP ranks 上重复保存。对于逐 token 的操作，每个 token 的 hidden 完整即可，不要求同时持有所有 token。

SP 将这段激活沿 token/sequence 切开，复用 TP group。它没有新增独立的 SP rank 轴，也没有让 attention 只看本地半段 context。

## 2. 怎么做：先画完整前向形状

设 `S=8,B=1,H=4,F=8,TP=2`，忽略 bias，采用 sequence-first：

| 位置 | rank0 | rank1 | 数值含义 |
|---|---|---|---|
| norm 输入/输出 | `[4,1,4]`，token 0..3 | `[4,1,4]`，token 4..7 | token shard，hidden 完整 |
| FC1 需要的输入 | `[8,1,4]` | `[8,1,4]` | AG 后每卡看到全部 tokens |
| FC1 + activation | `[8,1,4]` | `[8,1,4]` | 两卡分别持有 F 的不同部分 |
| FC2 本地乘积 | `[8,1,4]` | `[8,1,4]` | 对相同输出位置的不同贡献 |
| SUM RS 后 | `[4,1,4]`，token 0..3 | `[4,1,4]`，token 4..7 | 数值完整，再按 token 分片 |

FC1 输出和 FC2 输出在这个例子里 shape 相同，但前者是不同 feature，后者是相同 feature 的 partial sum。shape 必须和 placement 一起记录。

## 3. 到源码里找三处责任边界

1. [tensor_parallel/layers.py](<../../../../../../Megatron-LM/megatron/core/tensor_parallel/layers.py>)：从 `ColumnParallelLinear.forward` 的 `sequence_parallel` 分支追到线性层 autograd 实现，观察输入 AG 与反向 dgrad RS。Row 路径追 `RowParallelLinear.forward` 的 sequence RS。
2. [tensor_parallel/mappings.py](<../../../../../../Megatron-LM/megatron/core/tensor_parallel/mappings.py>)：检查 `_GatherFromSequenceParallelRegion` 与 `_ReduceScatterToSequenceParallelRegion` 的 forward/backward。
3. [distributed/finalize_model_grads.py](<../../../../../../Megatron-LM/megatron/core/distributed/finalize_model_grads.py>)：追 `finalize_model_grads` → `_allreduce_non_tensor_model_parallel_grads`，核对 norm 等复制参数梯度。

生产使用 TE spec 时，线性层可能由 [extensions/transformer_engine.py](<../../../../../../Megatron-LM/megatron/core/extensions/transformer_engine.py>) 中的 TE 模块完成；不能假定 Python local layer 的每个函数都实际执行。先确认 module 类型，再按相同形状/归约契约检查 TE 路径。

## 4. 反向不是照搬前向通信名

| 前向变换 | 反向责任 | 原因 |
|---|---|---|
| sequence AG，后续为 TP 切分计算 | SUM RS | 同一输入被不同 TP shard 使用，梯度贡献必须相加 |
| partial output SUM RS | AG | 各局部矩阵都需要对应输出梯度，再算本地 dW/dX |
| norm 本地逐 token 计算 | 局部 dX；参数梯度另行 TP SUM | gamma/beta 复制，但各卡处理不同 tokens |

注意 `_GatherFromSequenceParallelRegion.backward` 的 `tensor_parallel_output_grad`：为真时做 RS；为假时只是 split，因为后续计算可能是重复计算。不能把所有 AG 的 backward 都机械写成 RS，需看使用者的数学语义。

## 5. 被漏掉的 norm 参数梯度

设每 rank 两个 token 的 norm 输出都是 `[-1,1]`，上游梯度为全 1，则每 rank：

```text
dGamma_local = [-2,2]
dBeta_local  = [2,2]
```

跨 SP 切分合并后应该是 `dGamma=[-4,4]`、`dBeta=[4,4]`。只同步 DP 组不能补上 TP ranks 之间缺失的 token 贡献。

当前 `_allreduce_non_tensor_model_parallel_grads` 区分两个列表：带 `sequence_parallel` 标记的参数走 SUM；带 `average_gradients_across_tp_domain` 标记的参数走 AVG，并有独立优先级。不要把所有非 TP 参数都统一 SUM，也不要自己在 optimizer 前重复规约一次。

调试入口记录 `(param_name, sequence_parallel, average_gradients_across_tp_domain, grad_norm, tp_group_members)`。先在 finalize 前后比较，再看 optimizer 读取的是 `main_grad` 还是包装后的 local gradient。

## 6. 最终结果与代价

理想情况下，norm/residual 等对应激活从 `[S,B,H]` 变成 `[S/TP,B,H]`；但 FC1 前的 full-token gather buffer 仍存在，整体峰值不会简单除以 TP。

TP 中某些 AR 可以拆成匹配的 RS/AG，SP 因而不必等价于把一整套新通信叠到原 AR 上。但真实字节数、buffer 复用与 overlap 取决于 local/TE backend，不能据此承诺“SP 没有通信成本”。小 tensor、额外同步或通信 buffer 峰值都可能抵消收益。

与 CP 组合时，norm 入口常可理解为 `[S/(CP×TP),B,H]`；进入 TP 子图 gather 后恢复的是本 CP rank 的 `[S/CP,B,H]`，不是全局 `[S,B,H]`。attention 仍需 CP 协议处理远端上下文。


## 7. 等价性与失效边界

先运行 `python3 docs/ai_infra/examples/parallel_runtime_walkthroughs.py`，查看 `PASS sequence_context`。这是数学参考，不是实际 SP kernel 测试。

在 TP=2、相同输入和全局参数下，SP off/on 的差异应来自布局。dropout=0 或已对齐的分布式 RNG 排除了随机性差异，此时以下边界决定等价性：

- 对齐输出、dX、FC1/FC2 梯度和 norm 参数梯度；按 token index 重组，不只比较 loss。
- 若只有 norm 梯度缩小，检查参数标记及 finalize；若 dX 错而 dW 正确，检查 AG backward 的 reduction 语义。
- 记录 norm 与 FC1/FC2 边界 shape、峰值 HBM 和 AG/RS exposed time。
- PP 发送 shape 也会受 SP 影响，不能在 sender 分片后仍按全序列分配 receiver。

观测表和已有 GPU 测试入口见[并行证据边界](<../05_design_and_failures/03_parallelism_validation.md>)。
