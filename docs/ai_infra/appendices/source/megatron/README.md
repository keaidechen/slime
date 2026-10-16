# Megatron：从一次更新理解分布式训练

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../README.md>)。

可以先把训练想成一个普通程序：读入一批数据，计算预测与误差，通过反向传播求参数的梯度，再由优化器更新参数。模型变大以后，单张 GPU 的内存装不下全部状态，计算也太慢；Megatron 要解决的是怎样把这个程序拆到多张 GPU，同时保持同一个训练目标。后面的每种并行和优化，都是对“拆什么、谁保存、怎样合并、何时可用”的不同回答。

建议第一次按下面六组顺序阅读。基础篇建立单次更新和 Transformer 的形状；并行篇把形状映射到设备；状态篇追踪梯度、优化器和恢复；模型篇解释新结构如何改变这些边界；排障篇将解释变成可检查的证据；参考篇用于回查。每篇的手算例子是缩小后的数学模型，不是 GPU 性能测量。

本系列默认以本地 `Megatron-LM` commit `21fe0fe1597932421f1bb0efd93f469376a7a255` 为源码基线。文中的行号用于辅助定位，升级后应按函数名查找；Transformer Engine 等外部库的具体实现还取决于它们自己的版本。

先约定几个反复使用的记号，避免把 shape 中的数字当成字节数：

| 记号 | 含义 | 示例与计量 |
|---|---|---|
| `B`、`S` | 本次 microbatch 的样本数、每条序列的位置数 | `B=2,S=4` 表示 8 个 token 位置；有效训练 token 可能更少 |
| `H`、`F` | hidden 特征数、前馈网络的中间特征数 | `[S,B,H]` 的元素数为 `S×B×H` |
| `M` | 一次更新累积的 microbatch 数 | 是批次数，不是样本数 |
| `W`、rank | 进程总数、进程编号 | 常规场景一个进程控制一张 GPU，rank 不是层号 |
| `dX` | 损失对 X 的梯度 `∂L/∂X` | 与 X 同形状；不是 X 在时间上的变化量 |

模型数值通常无物理单位；交叉熵使用自然对数时以 nat 表示。显存以元素数乘每元素字节数计算：例如 BF16 的 `[4,2,8]` 占 `4×2×8×2=128` bytes，不含其他张量与分配器开销。通信时间由 bytes 除以 bytes/s 得到，后文会明确区分 GB 与 GiB。

- [01_foundations资料](<01_foundations/README.md>)
- [02_parallelism资料](<02_parallelism/README.md>)
- [03_state_and_memory资料](<03_state_and_memory/README.md>)
- [04_models_and_features资料](<04_models_and_features/README.md>)
- [05_design_and_failures资料](<05_design_and_failures/README.md>)
- [06_reference资料](<06_reference/README.md>)
