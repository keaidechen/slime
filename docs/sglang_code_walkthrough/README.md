# SGLang 系统学习与源码走读

这套中文文档基于本仓库的 SGLang 快照 `f5155d960286db25952217f343ee0d3c358f7f77`，把官方文档按工程问题重新组织。代码、命令、参数、类名和模型名保留英文；中文正文解释调用链、状态、正确性边界和性能权衡。

## 文档结构

```text
01_foundations/              基础概念、进程拓扑和端到端请求
02_runtime_core/             Scheduler、KV、执行层和生成控制
03_scaling_and_deployment/   并行、PD、多节点、路由和硬件平台
04_interfaces_and_models/    API、采样、模型类型、多模态和后训练
05_production_engineering/   Benchmark、观测、扩展、可靠性和运维
06_diffusion/                Diffusion pipeline、优化、扩展和验证
07_reference/                官方文档覆盖索引与术语表
```

各目录内重新编号。编号表示推荐阅读顺序，不表示模块调用顺序。

## 完整学习路线

1. [基础与请求链路](01_foundations/README.md)
2. [Runtime 核心](02_runtime_core/README.md)
3. [扩展、分布式与部署](03_scaling_and_deployment/README.md)
4. [接口、模型与后训练](04_interfaces_and_models/README.md)
5. [生产工程](05_production_engineering/README.md)
6. [SGLang Diffusion](06_diffusion/README.md)
7. [参考资料](07_reference/README.md)

## 按角色选择路线

| 目标 | 建议路线 |
|---|---|
| API/应用开发 | 1.2 → 4.1 → 4.2 → 5.3 |
| Runtime 开发 | 1.1 → 1.2 → 2.1 → 2.2 → 2.3 → 2.4 |
| 分布式推理 | Runtime 路线 → 3.1 → 3.2 → 3.3 |
| 模型接入 | 2.3 → 4.2 → 5.2 |
| RL/后训练 | 1.2 → 2.1 → 4.3 → 3.1 → 5.3 |
| 性能/生产运维 | 1.1 → 2.1 → 3.2 → 5.1 → 5.3 |
| Diffusion | 1.1 → 6.1 → 6.2 → 6.3 |

## 阅读方法

第一次阅读只跟一个普通文本生成请求，关闭 speculative、LoRA、PD 和 VLM。始终维护三本账：

| 账本 | 关键字段 | 创建 | 释放 |
|---|---|---|---|
| Request | rid、input/output、finish、grammar | API/scheduler 接收 | finish/abort |
| Batch | forward mode、seq lens、sampling info | scheduler iteration | result process 后 |
| KV | req slot、physical indices、radix lock | admission/prefix match | cache/evict/free |

再逐项打开高级功能，观察它增加了哪一种状态，并用 trace 验证 CPU scheduler、通信和 GPU forward 的时间关系。

## 一手资料

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang 官方文档](https://docs.sglang.io/)
- [SGLang 论文](https://arxiv.org/abs/2312.07104)
- [官方学习材料](https://github.com/sgl-project/sgl-learning-materials)

官方文档与本系列的完整映射见 [官方文档覆盖索引](07_reference/01_official_docs_coverage.md)。
