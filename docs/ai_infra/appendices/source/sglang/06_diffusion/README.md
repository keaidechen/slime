# 06_diffusion资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../../README.md>)。

这组从独立的图像生成循环开始，不把 LLM decode 的 KV 增长模型套过来。6.1 用小 latent 和两步 Euler 更新解释流水线；6.2 在此基础上引入 CFG、序列并行、特征缓存与量化；6.3 从第一步差异定位到最终质量。

先确认模型预测的是噪声、干净样本还是速度，再使用对应 scheduler 公式。读完应能区分采样 timestep 与墙钟时间、兼容请求合批与 LLM 连续批处理、等价执行优化与会改变质量的近似计算。

- [6.1 Pipeline 架构、服务接口与动态批处理](<01_pipeline_architecture_and_serving.md>)
- [6.2 并行、Backend、缓存与量化](<02_parallelism_cache_and_quantization.md>)
- [6.3 Profiling、新模型接入与正确性](<03_profiling_extension_and_correctness.md>)
