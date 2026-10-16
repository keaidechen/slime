# 05_production_engineering资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../../README.md>)。

先把指标算对，再验证扩展，最后处理过载与失败。5.1 用时间戳和容量曲线区分 TTFT、ITL、TPOT、吞吐与 goodput；5.2 用小张量建立参考结果；5.3 用取消后的 slot 账本检查恢复边界。

优化结论应同时回答工作负载是否相同、端到端改善多少、输出与资源生命周期是否仍正确。这里的故障矩阵是应验证的目标，并非声称每个 backend 已自动提供所有恢复保证。

- [5.1 Benchmark、Profiling 与可观测性](<01_benchmark_profiling_observability.md>)
- [5.2 模型与 Backend 扩展、正确性验证](<02_extension_and_correctness.md>)
- [5.3 可靠性、容量保护与发布运维](<03_reliability_capacity_operations.md>)
