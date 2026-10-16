# 02_parallelism资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../../README.md>)。

并行不是把同一段程序简单启动多次：DP 分不同样本，TP 分一个矩阵，PP 分不同层，SP/CP 分不同范围的序列工作，EP/ETP 分不同专家及其矩阵。先读策略篇选择切分对象，再按 TP → PP → SP → CP → EP/ETP 阅读；后两篇会复用前面的矩阵和通信知识。

每幅形状图都要同时回答两个问题：“这些卡保存的是不同元素，还是同一元素的不同贡献？”前者通常要拼接，后者通常要求和。下文用 `[1,2]` 乘一个小矩阵、两个流水阶段和四个路由 token 展开这一区别；进程组决定到底与谁拼接或求和。

- [并行策略选择与物理拓扑](<00_strategy_and_topology.md>)
- [Rank 网格、进程组、Tensor/Sequence Parallel](<01_parallel_state_and_tensor_parallel.md>)
- [Pipeline Parallel：1F1B、交错调度与 P2P](<02_pipeline_parallel.md>)
- [SP 执行走读：激活分片之后，谁负责梯度](<03_sequence_parallel_execution.md>)
- [CP 执行走读：从 batch 坐标到 attention backend](<04_context_parallel_execution.md>)
- [EP / ETP：从 token 路由走到专家梯度](<05_ep_etp_execution.md>)
