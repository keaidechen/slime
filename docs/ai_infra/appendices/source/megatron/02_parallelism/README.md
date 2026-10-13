# 02_parallelism资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../../README.md>)。

- [并行策略选择与物理拓扑](<00_strategy_and_topology.md>)
- [Rank 网格、进程组、Tensor/Sequence Parallel](<01_parallel_state_and_tensor_parallel.md>)
- [Pipeline Parallel：1F1B、交错调度与 P2P](<02_pipeline_parallel.md>)
- [SP 执行走读：激活分片之后，谁负责梯度](<03_sequence_parallel_execution.md>)
- [CP 执行走读：从 batch 坐标到 attention backend](<04_context_parallel_execution.md>)
- [EP / ETP：从 token 路由走到专家梯度](<05_ep_etp_execution.md>)
