# SGLang资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../README.md>)。

这里也可以作为一组从基本编程知识出发的源码教材。首次阅读先把一条短请求走完，再看缓存、设备执行和扩展；不需要先熟悉所有论文或缩写。章节中的小整数、玩具张量和假设时间用于手算，不代表真实模型配置或性能测试结果。

本系列的源码基线是仓库内 SGLang 快照 `f5155d960286db25952217f343ee0d3c358f7f77`。公式分清模型数学、容量估算与实现事实；命令默认值、功能组合和行号以该快照对应源码为准。在线文档可能已经变化。

| 阅读阶段 | 先解决的问题 | 可以亲手检查的结果 |
|---|---|---|
| 基础 | 一条请求为什么经历多轮执行 | 三 token prompt 的输入、输出和 KV 边界 |
| Runtime | 如何批处理、共享内存并正确异步执行 | batch 形状、物理索引、准入预算、采样概率 |
| 扩展部署 | 单实例何时需要多卡或拆开阶段 | 通信内容、每 rank 字节数与容量账 |
| 接口模型 | API 配置如何成为模型实际输入 | 默认值、媒体展开、版本与缓存失效 |
| 生产工程 | 如何证明优化有效且服务可恢复 | 延迟口径、参考数值、取消后的资源守恒 |
| Diffusion | 图像/视频生成为什么有不同循环 | latent 形状、逐步更新、并行与近似误差 |
| 参考 | 如何查一个陌生词或核实支持声明 | 术语对应的执行边界与源码入口 |

读每个例子时可在纸上记三列：逻辑请求已到哪一步、设备这轮读写哪些数据、谁仍持有资源。它们不同步时，通常就是需要继续追源码的位置。

- [01_foundations资料](<01_foundations/README.md>)
- [02_runtime_core资料](<02_runtime_core/README.md>)
- [03_scaling_and_deployment资料](<03_scaling_and_deployment/README.md>)
- [04_interfaces_and_models资料](<04_interfaces_and_models/README.md>)
- [05_production_engineering资料](<05_production_engineering/README.md>)
- [06_diffusion资料](<06_diffusion/README.md>)
- [07_reference资料](<07_reference/README.md>)
