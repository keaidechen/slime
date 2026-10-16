# 03_state_and_memory资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../../README.md>)。

模型能完成前向，还不等于能完成训练。反向需要中间激活，优化器需要历史状态，下一次前向需要所有参与者读到同一版新参数；断电重启还要恢复数据与随机数位置。这一组按“梯度汇总 → 状态分片 → 临时内存管理 → 更新算法”逐步展开。

阅读时为每份状态标注三件事：它属于哪个逻辑参数/样本，物理上保存在哪个设备，最后一个使用它的操作是谁。模型状态的静态字节数只是起点，FSDP 参数重建、通信重叠和 optimizer 临时空间共同决定峰值。

- [DDP、分布式 Optimizer 与 Checkpoint](<01_data_parallel_optimizer_checkpoint.md>)
- [Megatron-FSDP：状态分片与执行生命周期](<02_megatron_fsdp.md>)
- [显存技术、确定性与 CUDA Graph](<03_memory_determinism_cuda_graph.md>)
- [AdamW 与 Muon：参数分组、矩阵更新与状态 owner](<04_adam_muon_execution.md>)
