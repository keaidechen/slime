# 数学参考与状态示例

这些程序是文章推导的辅助材料。正文已经给出输入、推导和结果，阅读不依赖运行程序。数学参考检查教学模型的等价关系，不代表实际 GPU、多机通信或模型质量。

| 程序 | 解释对象 | 依赖与证据边界 |
|---|---|---|
| [design_budget.py](<design_budget.py>) | 工作负载、相关驻留、突发、规模预算与生命周期成本 | Python 标准库；确定性事件与数学核对，无真实模型质量或性能 |
| [design_cases.py](<design_cases.py>) | 同一模型的训练和推理候选比较与敏感性 | Python 标准库；容量和阶段时间模型，无 GPU 压测 |
| [hardware_pooling.py](<hardware_pooling.py>) | 异构计算、远端 KV、Attention–FFN 往返 | Python 标准库；教学带宽、延迟与成本阈值 |
| [environment_placement.py](<environment_placement.py>) | 环境生命周期、快照供给与端边云首播放 | Python 标准库；不启动沙箱、不执行网络或媒体模型 |
| [parallel_runtime_walkthroughs.py](<parallel_runtime_walkthroughs.py>) | DDP/FSDP、TP、PP、SP/CP、专家网格、KV 与调度 | Python 标准库；矩阵、有限差分和离散状态模型 |
| [omni_walkthroughs.py](<omni_walkthroughs.py>) | 音频帧/字节、播放缺口、块身份、取消与背压、损失梯度、迭代恢复及阶段容量 | Python 标准库；教学模拟；可选固定源码文件 hash 核对，不运行真实媒体模型 |
| [etp_reference.py](<etp_reference.py>) | 专家内部 TP 与 EP×ETP 网格 | Python 标准库；固定路由概率，不覆盖 router 参数梯度 |
| [optimizer_walkthroughs.py](<optimizer_walkthroughs.py>) | AdamW 两步历史、Muon 矩阵迭代与分片反例 | Python 标准库；解析答案与谱分解对照，不运行真实优化器后端 |
| [rl_infra_contracts.py](<rl_infra_contracts.py>) | 词表归约与梯度、DPO、策略梯度、KL、过滤偏移、ESS、终止与截断、组尾部和重试 | Python 标准库；有限差分与反例，不运行真实训练、奖励服务或环境 |
| [training_rl_walkthroughs.py](<training_rl_walkthroughs.py>) | 更新、packing、checkpoint、PD、PPO/GAE 与异步队列 | Python 标准库；教学协议不等于仓库实现 |
| [loss_normalization.py](<loss_normalization.py>) | 不等长样本与归约分母 | Python 标准库；只检查给定目标口径 |
| [case_source_evidence.py](<case_source_evidence.py>) | 源码身份、内容哈希与符号定位 | Python 3.10+；源码漂移检查，不执行训练 |
| [vector_add.py](<vector_add.py>) | Triton 向量加与尾块 mask | PyTorch、Triton 和 CUDA；不属于本轮 CPU 数学验证 |
| [collectives.py](<collectives.py>) | torchrun 进程组与 collective 语义 | PyTorch 和分布式运行环境 |
| [ray_backpressure.py](<ray_backpressure.py>) | 队列并发与有界在途任务 | Ray；示意 actor 的背压机制 |

FlashAttention 的 [online_attention.py](<../appendices/source/flash_attention/examples/online_attention.py>)随其源码专题保存，用标准库解释在线归一化、split 合并与反向公式。

[算例索引](<../appendices/indexes/手工推演索引.md>) · [证据范围](<../reference/README.md>) · [系列总览](<../README.md>)


## 内容深化与验证

[numerics_quantization.py](<numerics_quantization.py>) 核对 BF16 舍入、INT4、裁剪与输出误差；[mini_transformer.py](<mini_transformer.py>) 对全部 36 个参数做有限差分；[system_tradeoffs.py](<system_tradeoffs.py>) 核对 batch、存储和排队算例；[cpu_evidence.py](<cpu_evidence.py>) 产生真实 CPU 时间线、原始计时和进程中断恢复证据。均只依赖 Python 标准库，GPU 与框架 backend 不在验证范围内。


[resource_statistics.py](<resource_statistics.py>)核对拓扑与弹性 batch、视觉／视频 KV、扩散噪声转换，并精确枚举小样本配对 bootstrap；不执行集群调度或真实多模态模型。
