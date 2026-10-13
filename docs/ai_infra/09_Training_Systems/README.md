# 训练系统

连接数据、目标、梯度、优化器、内存和恢复，解释一次正确训练更新如何扩展。

## 主题地图

| 条目 | 定位 |
|---|---|
| [训练步与优化器](训练步与优化器.md) | 一次训练 step：目标、梯度、状态与更新 |
| [优化器原理与 AdamW](优化器原理与AdamW.md) | SGD、Momentum、Adam/AdamW 的动机、状态、偏差校正与两步推演 |
| [Muon 与矩阵优化](Muon与矩阵优化.md) | 极因子、Newton–Schulz、缩放、参数分组与分布式反例 |
| [数据管线与训练演进](数据管线与训练演进.md) | 训练数据管线：从样本到持续供给的 microbatch |
| [训练显存优化](训练显存优化.md) | 分片，以及受依赖、缓冲和传输窗口约束的激活保留／重算／卸载联合选择 |
| [训练并行方案设计](训练并行方案设计.md) | 固定模型、设备与期限，比较分片、重算、TP 和恢复开销 |
| [分布式Checkpoint与恢复](分布式Checkpoint与恢复.md) | 分布式 Checkpoint 与恢复 |
| [低精度训练与状态管理](低精度训练与状态管理.md) | 计算、累积、通信、优化器与 scale 的独立状态 |
| [预训练微调与后训练的系统边界](预训练微调与后训练的系统边界.md) | Pretraining、SFT、LoRA、QLoRA、DPO 与在线 RL |

## 完整推演与历史主线

| 条目 | 可以逐项核对的内容 |
|---|---|
| [两步优化器更新](优化器原理与AdamW.md#两步数值推演历史怎样改变方向) | 同一外部梯度序列的 SGD、Momentum、Adam 与 AdamW |
| [Muon 矩阵推演](Muon与矩阵优化.md#先用三次迭代看清计算过程) | 2×2 矩阵、奇异值变换、有限迭代误差 |
| [分片反例](Muon与矩阵优化.md#分片反例完整矩阵与各行独立更新不同) | 局部各行正交化与全矩阵更新不等价 |
| [一次完整更新：从缩放梯度到AdamW与跳步](<训练步与优化器.md#一次完整更新从缩放梯度到adamw与跳步>) | unscale、全局norm、裁剪、AdamW与Inf跳步 |
| [Packing 的目标与分母](数据管线与训练演进.md#packing的目标边界与有效分母) | 标签、mask、跨样本边界与有效 token 归约 |
| [数据游标与Checkpoint提交](<分布式Checkpoint与恢复.md#数据游标与checkpoint提交>) | 预取游标、训练消费游标、恢复与提交 |

全局入口：[手工推演索引](../手工推演索引.md) · [技术演进索引](../技术演进索引.md)。

## 读完本专题之后

完整副本遇到状态容量限制，分片和重算引入通信与额外计算；异步数据和保存提高重叠，也增加一致性责任。



## 查阅与关联

[系列总览](../README.md) · [术语索引](../术语索引.md) · [问题索引](../问题索引.md)

## 贯穿到当前实现

[从样本到参数更新](../12_Performance_Reliability/贯穿案例_从Rollout到参数更新.md)。

[训练状态与数据恢复](../12_Performance_Reliability/贯穿案例_换权失败与训练恢复.md)。

## 原理、源码与性能的对应入口

| 系统问题 | 通用正文 | 固定源码走读 | 性能或案例 |
|---|---|---|---|
| 优化器方向与状态 owner | [AdamW](优化器原理与AdamW.md)、[Muon](Muon与矩阵优化.md) | [Megatron 优化器执行](../13_Source_Analysis/megatron/03_state_and_memory/04_adam_muon_execution.md) | [训练性能](../12_Performance_Reliability/04_megatron.md) |
| 数据目标与更新 | [Packing](数据管线与训练演进.md)、[训练步](训练步与优化器.md) | [Megatron 数据](../13_Source_Analysis/megatron/01_foundations/00_ecosystem_install_and_data.md)、[优化器](../13_Source_Analysis/megatron/03_state_and_memory/01_data_parallel_optimizer_checkpoint.md) | [训练性能](../12_Performance_Reliability/04_megatron.md) |
| 状态、分片与重算 | [显存优化](训练显存优化.md)、[低精度](低精度训练与状态管理.md) | [Megatron 总览](../13_Source_Analysis/megatron/README.md) | [训练恢复案例](../12_Performance_Reliability/贯穿案例_换权失败与训练恢复.md) |
| 后训练目标与可训练状态 | [微调边界](预训练微调与后训练的系统边界.md) | [slime 算法](../13_Source_Analysis/slime/05_rl_algorithms.md) | [Rollout 到更新](../12_Performance_Reliability/贯穿案例_从Rollout到参数更新.md) |

源码入口说明相应实现，不能证明它支持表中每种配方。


## 内容深化与验证

[训练效率与收敛](训练效率与收敛.md)：batch、梯度噪声与临界尺度，强／弱扩展的达标收益和 GPU-hours，以及学习率时钟、loss spike 与质量对照。

## Omni 多任务训练

[Omni 训练与后训练](Omni训练与后训练.md)连接媒体对齐、冻结/解冻、因果与目标 mask、多码本归一化、连续生成目标及交互奖励。先读表示与计算图，再进入本篇，前后关系见 [Omni 阅读路线](../guides/Omni扩展阅读路线.md)。
