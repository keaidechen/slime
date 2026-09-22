# 手工推演 DDP：不等长数据、梯度累积与 Bucket 时间线

> 模型：两张卡、普通梯度平均语义、同步更新。数字是教学构造；自定义通信 hook、Join 和其他 reduction 规则需要重新推导。概念见[DDP](DP与DDP.md)。

## 输入、目标与参考答案

模型 $y=w_1x_1+w_2x_2$，初值 $w=[1,2]$，每个样本 $\ell=(y-a)^2/2$。全局目标是本次更新所有有效样本的 mean，N=4；不包含被 mask 的占位样本。

| rank | microbatch | x | a | y | loss | 未缩放梯度 |
|---|---|---|---|---|---|---|
| 0 | 0 | [1,0] | 0 | 1 | 0.5 | [1,0] |
| 0 | 1 | [0,1] | 0 | 2 | 2 | [0,2] |
| 0 | 1 | [1,1] | 0 | 3 | 4.5 | [3,3] |
| 1 | 0 | [2,0] | 0 | 2 | 2 | [4,0] |
| 1 | 1 | 占位输入，mask=0 | — | — | 0 | [0,0] |

单卡参考：loss sum=9，mean=2.25，gradient sum=[8,5]，mean gradient=[2,1.25]。SGD 学习率 0.1 后，参数应为 [0.8,1.875]。

## 把归一化放到正确位置

若默认 DDP 将 D 个 rank 的梯度平均，每个 rank 应反传 $\widetilde L_r=(D/N)\sum_{i\in r}\ell_i$。这里 D=2，N=4，缩放因子为 0.5。

| 时刻 | rank 0 累积梯度 | rank 1 累积梯度 | 是否规约 |
|---|---|---|---|
| 清空梯度 | [0,0] | [0,0] | 否 |
| microbatch 0 后 | [0.5,0] | [2,0] | 否，使用 no_sync |
| microbatch 1 的本地贡献加入后 | [2,2.5] | [2,0] | 最后一次反向触发同步 |
| SUM AllReduce 后 | [4,2.5] | [4,2.5] | 数学中间结果 |
| 除以 D 后 | [2,1.25] | [2,1.25] | 可以执行相同更新 |

具体实现可能把除法放在规约前后，本表拆开只是为了核算。由于已经按整次更新的 N 缩放，不应再除以 microbatch 数 2。

错误做法是先算各 rank 的 local mean，再平均：rank 0 为 [4/3,5/3]，rank 1 为 [4,0]，得到 [8/3,5/6]，已经偏离全局目标。日志可以把 detached loss sum/count 汇总，但这不自动修正 backward 的梯度。

## 空 microbatch 为什么还需要参与执行协议

表中的占位步表示执行相容的 forward/backward、产生零贡献，并保持各 rank 的 collective 顺序。实际模型若含数据相关分支或其他跨 rank 操作，仅构造一个无关的零 loss 未必能触发相同通信。不能让 rank 1 直接跳过最后一次反向，而 rank 0 继续等待同步。

`no_sync` 应覆盖对应 forward 和 backward。全局有效数量 N 需要在这次更新的缩放前已知，或采用等价的、经过推导的后缩放；把 count 的额外 collective 混进不同顺序也会导致等待。

## Bucket 的独立时间算例

为解释重叠，下面换成包含两组参数的抽象网络；时间不由上面两个标量参数的计算量推导。假设通信通道串行，bucket 通信均为 2 ms：

| bucket | rank 0 ready | rank 1 ready | 全部就绪 | 通信区间 |
|---|---|---|---|---|
| B0：后层梯度 | 2 ms | 3 ms | 3 ms | [3,5] ms |
| B1：前层梯度 | 6 ms | 7 ms | 7 ms | [7,9] ms |

两卡计算最迟在 7 ms 结束，更新要等到 9 ms。通信总服务时间为 4 ms，暴露在计算之后的尾部为 2 ms。rank 0 若从 2 ms 提交 B0 到 5 ms 完成，中间包含等 rank 1 的时间，不能把 3 ms 全当作传输时间。

若合成一个 bucket，仍假设总服务时间为 4 ms，则要从 7 ms 执行到 11 ms。真实大 bucket 的服务时间还可能因带宽利用率更好而变化，本例只隔离 ready time 的影响。

## 从推演定位真实实现

检查参数注册/就绪顺序、bucket 大小、梯度是否为 bucket view、no_sync 范围、loss/count 缩放和优化器执行时刻。先与单卡全局 batch 比较一步梯度，再看 trace 中的 ready、提交和完成。

数值复核：[纯 Python 推演脚本](../labs/parallel_runtime_walkthroughs.py)。依据：[PyTorch DDP 设计](https://docs.pytorch.org/docs/stable/notes/ddp.html)；具体行为以实际框架版本和 hook 为准。

## 贯穿到当前实现

[对照Megatron bucket和采集窗口](../12_Performance_Reliability/贯穿案例_分布式训练等待与通信重叠.md)。

---

[所属领域](README.md) · [手工推演总入口](../手工推演索引.md)
