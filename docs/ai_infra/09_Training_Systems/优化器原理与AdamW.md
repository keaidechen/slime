# 优化器原理：从 SGD、Momentum 到 AdamW

> 前置：[自动微分](../01_Math_Numerics/计算图与自动微分.md)、[训练 step](训练步与优化器.md)。本篇解释梯度怎样变成更新；全局归约、unscale、裁剪和跳步的完整时序仍由训练 step 正文维护。

## 梯度给出局部方向，优化器决定怎样使用历史

记更新前参数为 $\theta_{t-1}$，本次梯度为 $g_t=\nabla L_t(\theta_{t-1})$，学习率为 $\eta_t$。梯度已经按预期目标归约；改变有效 token 分母会先改变这里的输入。不同优化器也会产生不同参数，因此真实训练中后续梯度通常不再相同。

最基本的 SGD 为 $\theta_t=\theta_{t-1}-\eta_t g_t$。如果某个方向的梯度持续同号，另一个方向受噪声影响反复变号，可以通过历史平滑减少第二种振荡。本篇采用未归一化动量：

$$b_t=\beta b_{t-1}+g_t,\qquad \theta_t=\theta_{t-1}-\eta_t b_t,\qquad b_0=0.$$

恒定梯度 $g$ 下，$b_t=(1-\beta^t)g/(1-\beta)$，所以动量不仅平滑方向，也改变幅度。另一种写法 $m_t=\beta m_{t-1}+(1-\beta)g_t$ 在同一初值下满足 $m_t=(1-\beta)b_t$；比较实现时需要连同学习率一起看。Nesterov 还改变实际使用的方向，不能只比较名为 momentum 的数值。

## Adam：每个坐标使用自己的历史尺度

Adam 保存一阶矩和平方梯度的二阶原点矩，两者与参数同形状：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t\odot g_t.$$

这里 $\odot$ 为逐元素乘法；$v_t$ 不是减去均值后的方差。零初始化时，恒定梯度下有 $m_t=(1-\beta_1^t)g$、$v_t=(1-\beta_2^t)g^2$，由此得到偏差校正：

$$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
d_t=\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},\qquad
\theta_t=\theta_{t-1}-\eta_t d_t.$$

除法和平方根均逐元素。较大的历史平方梯度会增大对应分母，但这不是计算完整 Hessian，也不是按矩阵奇异方向归一化。偏差校正补偿零初始化，不意味着能消除数据非平稳性或梯度估计偏差。算法来源：[Adam 原论文](https://arxiv.org/abs/1412.6980)。

当 $t=1$、$\epsilon=0$ 且每个梯度非零时，$d_1=\operatorname{sign}(g_1)$。这是首步的特殊结果，后续依赖历史，并非每步都做 sign 更新。正 epsilon 还会改变小梯度区域的尺度。

| 配置 | 控制什么 | 需要观察的后果 |
|---|---|---|
| $\eta_t$ | 最终更新尺度及其调度 | 更新/参数范数、loss 和验证曲线 |
| $\beta_1$ | 一阶历史的保留 | 方向平滑与方向改变后的响应 |
| $\beta_2$ | 平方梯度历史的保留 | 尺度变化后的适应、尖峰的持续影响 |
| $\epsilon$ | 小分母区域的保护和尺度 | 小梯度更新；放在根内或根外不是同一公式 |
| step $t$ | 偏差校正和调度的位置 | 跳步、缺梯度参数和恢复后的计数 |

指数历史的权重按 $\beta^k$ 衰减；$1/(1-\beta)$ 可作时间尺度的粗略直觉，不是固定长度窗口。实际 API 的参数和执行选择见 [PyTorch Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)。

## AdamW：让衰减不进入矩状态

若目标添加 $\lambda\|\theta\|_2^2/2$，送入 Adam 的梯度变为 $g_t+\lambda\theta_{t-1}$。衰减会进入 m/v 并被自适应分母缩放。AdamW 则对任务梯度更新 m/v，单独收缩参数：

$$\theta_t=(1-\eta_t\lambda)\theta_{t-1}-\eta_t d_t.$$

这里明确采用学习率乘 weight decay 的约定；其他文献可能把收缩系数单独命名。普通无动量 SGD 中，两种形式按这个约定一致；对 Adam 通常不一致。[AdamW 原论文](https://arxiv.org/abs/1711.05101)。

反例：$\theta_0=[1,2]$、任务梯度为零、m/v 为零、$\eta=0.1$、$\lambda=0.1$、$\epsilon=10^{-8}$。Adam 加 L2 的首步输入为 $[0.1,0.2]$，更新后约为 $[0.9,1.9]$；AdamW 的自适应项为零，得到 $[0.99,1.98]$。这里任务梯度是显式零张量；`grad=None` 是否跳过衰减要查具体实现。

哪些参数进入 decay 组是配方的一部分。Bias、norm、embedding 和共享参数应按所用规则记录；“命令行叫 adam”也不证明使用耦合 L2，见[源码中的后端选择](../13_Source_Analysis/megatron/03_state_and_memory/04_adam_muon_execution.md)。

## 两步数值推演：历史怎样改变方向

固定外部梯度序列 $g_1=[2,-1]$、$g_2=[0,3]$，初始参数 $[1,-2]$，学习率 0.1。Momentum 取 $\beta=0.5$；Adam/AdamW 取 $\beta_1=\beta_2=0.5$，初始状态为零，$\epsilon=0$；本例每步 v 的每个元素均为正。只有 AdamW 加 $\lambda=0.1$。这些数值为了手算，不是训练推荐值，也不是收敛实验。

| Adam 状态 | 第一步 | 第二步 |
|---|---|---|
| $m_t$ | $[1,-0.5]$ | $[0.5,1.25]$ |
| $v_t$ | $[2,0.5]$ | $[1,4.75]$ |
| $\hat m_t$ | $[2,-1]$ | $[2/3,5/3]$ |
| $\hat v_t$ | $[4,1]$ | $[4/3,19/3]$ |
| $d_t$ | $[1,-1]$ | $[1/\sqrt3,5/\sqrt{57}]\approx[0.577350,0.662266]$ |

第二步第一个坐标当前梯度为零，Adam 仍因历史动量继续更新；第二个坐标的当前梯度从负转正，历史也使更新不同于直接减去 0.3。

| 优化器 | 第一步参数 | 第二步参数 |
|---|---|---|
| SGD | $[0.8,-1.9]$ | $[0.8,-2.2]$ |
| Momentum | $[0.8,-1.9]$ | $[0.7,-2.15]$ |
| Adam，无衰减 | $[0.9,-1.9]$ | $[0.842265,-1.966227]$ |
| AdamW | $[0.89,-1.88]$ | $[0.823365,-1.927427]$ |

AdamW 第二步例如第一坐标为 $0.99\times0.89-0.1/\sqrt3$。同一个真实 loss 上，各方法走到不同参数后，第二步梯度会改变；固定梯度例只隔离更新规则。

CPU 对照见[优化器数值参考](../examples/optimizer_walkthroughs.py)。跨 rank 归约、裁剪、AdamW 更新与 Inf 跳步接成完整一次 step 的推演见[训练更新](训练步与优化器.md#一次完整更新从缩放梯度到adamw与跳步)。

## 状态、融合与分片

| 方法 | 本节配方的持久优化器张量 | 更新的耦合范围 |
|---|---|---|
| SGD，无动量 | 无 | 单个参数元素 |
| Momentum | 一个动量张量 | 单个参数元素的时间历史 |
| Adam/AdamW | m 和 v | 单个参数元素的时间历史 |
| 标准 Muon | 一个动量张量 | 一块逻辑矩阵内的奇异方向 |

这里不计参数、梯度、master weight、步计数与工作区；AMSGrad、自适应 Muon 等变体另有状态。相同精度下，两个 Adam 矩改成一个动量可减少这部分持久字节，但矩阵算法还会增加临时计算与通信。

Adam 的逐元素结构允许 owner 在拿到正确全局梯度与对应 m/v 后更新自己的元素分片。全局 norm、非有限值决策、共享参数和参数同步仍需要跨 rank 协作。Fused/foreach 改变提交和读写方式，CPU offload 改变执行位置，它们不是新的更新公式；数值等价还需检查 dtype、舍入和操作顺序。[PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)。

恢复训练应保留参数组到张量的身份、m/v、计数、学习率调度与必要精度状态。只恢复权重或重新初始化矩，会改变上述第二步。一般恢复协议见[Checkpoint](分布式Checkpoint与恢复.md)，低精度状态见[训练精度](低精度训练与状态管理.md)。

下一篇[Muon 与矩阵更新](Muon与矩阵优化.md)把“逐元素历史尺度”扩展到“矩阵整体更新”，并说明这一改变如何影响 TP 和 DP。

[训练目录](README.md) · [Megatron 优化器走读](../13_Source_Analysis/megatron/03_state_and_memory/04_adam_muon_execution.md)
