# ETP：把一个专家的矩阵切到多张卡

ETP 是 Expert Tensor Parallelism。EP 决定不同 expert 放在哪里；ETP 决定一个 expert 内部怎样切；expert DP（EDP）决定同一 expert shard 有几份副本。三者必须分开。

## 1. 出发点：EP 之后，单个 expert 仍可能放不下

设一个 MoE 层有 8 个 experts，EP=4，每个 EP owner 管理 2 个 experts。如果其中一个 expert 的矩阵、梯度或中间激活已经过大，只增加 EP 不一定能解决：即使一张卡只管理一个 expert，这个 expert 仍可能超出容量。

ETP 进一步切 expert 的 FC1/FC2。它适合单 expert 足够大的情形；对细粒度小 expert，切分可能只得到更小的 GEMM 和更多通信。首先区分“全部专家太多”和“单专家太大”。

## 2. 怎么做：列切 FC1，行切 FC2

采用数学记法 `Y = activation(X A) B`，省略 bias，`X:[N,H]`、`A:[H,F]`、`B:[F,H]`。PyTorch 权重实际存储常为 `[out,in]`，不能直接按这里的书写轴切 storage。

ETP=X_degree 时：

```text
A = [A0, A1, ...]             Ai: [H, F/X_degree]
B = [B0; B1; ...]             Bi: [F/X_degree, H]
Zi = activation(X Ai)         Zi: [N, F/X_degree]
Pi = Zi Bi                   Pi: [N, H]，是 partial sum
Y = Σ_i Pi
```

每个 ETP rank 拥有同一 expert 的不同 intermediate 片段。激活函数逐元素作用，所以可在各片段独立计算。SwiGLU 需要 gate/up 两个分支按对应 intermediate 索引配对切分，不能把拼接后的两个分支错当两份独立 expert。

在本快照普通 AlltoAll dispatcher 路径中，先在 expert TP 组 gather token，再计算局部 expert，最后对 partial output 做 reduce-scatter，送回原来 token 分布。不是每个 expert 的 FC2 都再额外做一次相同 reduction；通信可能由 dispatcher 接管，详见[代码走读](../13_Source_Analysis/megatron/02_parallelism/05_ep_etp_execution.md)。

## 3. 两卡数值例子：SUM 不能写成平均

令 `H=2,F=4,N=1`，采用 ReLU，输入和矩阵为：

```text
X = [1, 2]
A = [[1, 0, 1, 2],
     [0, 1, 1, 0]]
B = [[1, 0], [0, 1], [1, 1], [2, -1]]
```

| 项目 | ETP rank 0 | ETP rank 1 |
|---|---|---|
| A 的列 | 0,1 | 2,3 |
| B 的行 | 0,1 | 2,3 |
| ReLU(X Ai) | `[1,2]` | `[3,2]` |
| 局部 Pi | `[1,2]` | `[7,1]` |

相加得到 `[8,3]`，与未切分计算相同。平均会得到 `[4,1.5]`，改变模型。

取 `loss=sum(Y)`，上游梯度 `[1,1]`：

```text
dZ0 = [1,1]，dZ1 = [2,1]
dX0 = [1,1]，dX1 = [4,2]
dX  = [5,3]
dA0/dA1、dB0/dB1 分别由对应 shard 本地计算
```

`dX` 要累加同一 token 经过不同 intermediate shard 的贡献；expert 参数梯度则只在持有**相同 shard**的 EDP 副本间归约。ETP 组内不同 shard 的参数梯度不能互相平均。

对于 top-k MoE，数学上还要按 router probability 加权并按 token index 合并。实际实现可能把 probability 乘法融合到 expert activation/FC2 前；检查代码时要确认恰好加权一次。

## 4. 与 TP、EP、EDP 的八卡例子

限制在标准 decoder、PP=1，显式选择 `order=tp-cp-ep-dp-pp`，W=8。

```text
dense:  TP=2, CP=2, DP=2       → 2×2×2 = 8
expert: ETP=2, EP=2, EDP=2     → 2×2×2 = 8
```

假设 4 个 experts，EP owner 0 管 e0/e1，owner 1 管 e2/e3：

| global rank | dense (tp,cp,dp) | expert (etp,ep,edp) | expert 参数所有权 |
|---:|---|---|---|
| 0 | (0,0,0) | (0,0,0) | e0/e1 的 shard 0，副本 0 |
| 1 | (1,0,0) | (1,0,0) | e0/e1 的 shard 1，副本 0 |
| 2 | (0,1,0) | (0,1,0) | e2/e3 的 shard 0，副本 0 |
| 3 | (1,1,0) | (1,1,0) | e2/e3 的 shard 1，副本 0 |
| 4 | (0,0,1) | (0,0,1) | e0/e1 的 shard 0，副本 1 |
| 5 | (1,0,1) | (1,0,1) | e0/e1 的 shard 1，副本 1 |
| 6 | (0,1,1) | (0,1,1) | e2/e3 的 shard 0，副本 1 |
| 7 | (1,1,1) | (1,1,1) | e2/e3 的 shard 1，副本 1 |

rank 0 的 ETP 组 `{0,1}` 合作计算同一 expert；EP 组 `{0,2}` 交换路由 token；EDP 组 `{0,4}` 同步同一 expert shard 的训练梯度。这里 dense TP 组恰好等于 ETP 组，不是一般规律。

仍用这 8 个 ranks，可另选 `ETP=1, EP=4, EDP=2`：rank 0..3 分别持有 e0..3 的完整矩阵，rank 4..7 是第二份副本。dense 的 `TP=2,CP=2,DP=2` 无需跟着变。这就是两套布局在同一物理 world 上解释不同层；具体组合还必须满足配置/backend 约束。

## 5. 最终结果：容量收益可以算，速度需要测

不计共享专家、router、bias 和 padding，一个有 `N_experts` 个普通两层 expert 的 MoE 层，expert 权重总元素为 `2×N_experts×H×F`。每 rank 的 expert 权重约为：

```text
2 × N_experts × H × F / (EP × ETP)
```

SwiGLU 的两路 FC1 加 FC2 则约为 `3×N_experts×H×F/(EP×ETP)`。这是 expert 参数项，不能再无条件除 EDP；EDP 是复制，只有启用相应状态分片才减少对应状态。

固定 W 和 PP 时，`EP×ETP×EDP=W/PP`。若把 ETP 加倍同时把 EP 减半，`EP×ETP` 不变，expert 参数总量/卡可能不变，但每 rank 的专家数变多、单 expert shard 变小，通信和计算布局会变。不能声称“增大 ETP 必然使总显存下降”。

## 6. 代价与组合边界

- Gather 后，每个 ETP rank 需要看到本组该 expert 的完整 token 集合；输入 token buffer 不按 ETP 简单缩小。
- Intermediate 从 F 变成 F/ETP；太窄的 grouped GEMM 可能利用率下降，通信占比上升。
- EP 的 A2A 与 ETP 的 AG/RS 是不同通信域，可能叠加网络与 stream 竞争。
- ETP 不会修复 router 负载倾斜；热点 expert 仍是热点，只是由多个 shard 合作完成。
- 显式设置 `--expert-tensor-parallel-size`：本快照未设置时回落到普通 TP，不能默认当作 1。
- 切换 degree 后，checkpoint 需要正确重建 expert ID、FC1/FC2 shard 和 optimizer state；能 load 不等于一步更新等价。

## 7. 如何验证

在仓库根目录运行：

```bash
python3 docs/ai_infra/examples/etp_reference.py
```

预期四项 PASS：输出和手算梯度、有限差分、EP+ETP token 重组、两网格成员。脚本刻意包含不均衡路由、空 expert 和多条 token 贡献；它是纯 Python 数学实验，没有启动进程组。

真实多卡比较须固定全局 batch、模型参数和路由输入，报告改变 ETP 后 EP/EDP 是否随之变化。至少对照 outputs、输入梯度、每 expert 权重梯度、一步更新，再比较 AG/RS 时间、每 expert token 数、GEMM shape 和峰值内存。完整记录方式见[验收章节](../13_Source_Analysis/megatron/05_design_and_failures/03_parallelism_validation.md)。

依据：本仓库 `parallel_state.py`、`moe/token_dispatcher.py`、`moe/experts.py`；外部术语和映射可参照 [NVIDIA MoE 指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)。

[返回并行专题](README.md) · [八类阅读主线](八类并行阅读主线.md)

<!-- series-nav -->

[上一篇：Expert Parallelism：模型稀疏，系统不一定轻松](<ExpertParallelism.md>) · [专题目录](<README.md>) · [下一篇：Embedding 与词表并行](<Embedding与词表并行.md>)

<!-- /series-nav -->
