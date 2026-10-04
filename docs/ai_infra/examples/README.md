# 数学参考与状态示例

这些程序是文章推导的辅助材料。正文已经给出输入、推导和结果，阅读不依赖运行程序。数学参考检查教学模型的等价关系，不代表实际 GPU、多机通信或模型质量。

| 程序 | 解释对象 | 依赖与证据边界 |
|---|---|---|
| [parallel_runtime_walkthroughs.py](parallel_runtime_walkthroughs.py) | DDP/FSDP、TP、PP、SP/CP、专家网格、KV 与调度 | Python 标准库；矩阵、有限差分和离散状态模型 |
| [etp_reference.py](etp_reference.py) | 专家内部 TP 与 EP×ETP 网格 | Python 标准库；固定路由概率，不覆盖 router 参数梯度 |
| [training_rl_walkthroughs.py](training_rl_walkthroughs.py) | 更新、packing、checkpoint、PD、PPO/GAE 与异步队列 | Python 标准库；教学协议不等于仓库实现 |
| [loss_normalization.py](loss_normalization.py) | 不等长样本与归约分母 | Python 标准库；只检查给定目标口径 |
| [case_source_evidence.py](case_source_evidence.py) | 源码身份、内容哈希与符号定位 | Python 3.10+；源码漂移检查，不执行训练 |
| [vector_add.py](vector_add.py) | Triton 向量加与尾块 mask | PyTorch、Triton 和 CUDA；不属于本轮 CPU 数学验证 |
| [collectives.py](collectives.py) | torchrun 进程组与 collective 语义 | PyTorch 和分布式运行环境 |
| [ray_backpressure.py](ray_backpressure.py) | 队列并发与有界在途任务 | Ray；示意 actor 的背压机制 |

FlashAttention 的 [online_attention.py](../13_Source_Analysis/flash_attention/examples/online_attention.py)随其源码专题保存，用标准库解释在线归一化、split 合并与反向公式。

[算例索引](../手工推演索引.md) · [证据范围](../reference/README.md) · [系列总览](../README.md)
