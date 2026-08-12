# SGLang 系统学习与源码走读

这套中文文档基于本仓库的 SGLang 快照 `f5155d960286db25952217f343ee0d3c358f7f77`。它不是官方参数手册的翻译，而是一套面向代码阅读和故障定位的工程地图：从一个问题出发，找到入口、状态所有者、跨进程消息、核心分支、失败清理和观测点。

> 源码快照很重要。SGLang 迭代很快；类名、参数或流程与其他版本不一致时，以本仓库 `sglang/` 子模块代码为准。

## 1. 文档如何组织

```text
01_foundations/              基础概念、进程拓扑和端到端请求
02_runtime_core/             Scheduler、KV、执行层和生成控制
03_scaling_and_deployment/   并行、PD、多节点、路由和硬件平台
04_interfaces_and_models/    API、采样、模型、多模态、控制面和换权
05_production_engineering/   Benchmark、观测、扩展、可靠性和运维
06_diffusion/                Diffusion pipeline、优化、扩展和验证
07_reference/                官方文档覆盖索引与术语表
```

目录编号只表示推荐学习顺序，不表示进程调用顺序。为了避免重命名造成外部链接失效，现有文件路径保持稳定；每个章节内部改用“问题 → 结论 → 源码链路 → 边界/失败 → 排查”的结构。

## 2. 先按问题找文档

| 你正在问的问题 | 先看 | 再看 |
|---|---|---|
| 一条请求从 HTTP 进入后经过哪些进程？ | [1.2 进程拓扑与请求链路](01_foundations/02_process_topology_and_request_path.md) | [2.1 Scheduler](02_runtime_core/01_scheduler_and_batch.md) |
| 请求为什么还没进 GPU 就失败了？ | [4.1 API、规范化与采样](04_interfaces_and_models/01_api_request_and_sampling.md) | [1.2 请求链路](01_foundations/02_process_topology_and_request_path.md) |
| `temperature=0`、`n`、stop、JSON schema 最终怎样生效？ | [4.1 API、规范化与采样](04_interfaces_and_models/01_api_request_and_sampling.md) | [2.4 约束生成](02_runtime_core/04_speculative_structured_sampling.md) |
| prefill/decode 为什么能动态拼 batch？ | [2.1 Scheduler](02_runtime_core/01_scheduler_and_batch.md) | [2.3 ModelRunner](02_runtime_core/03_model_runner_attention_cuda_graph.md) |
| KV cache 属于谁、什么时候释放？ | [2.2 KV 与内存](02_runtime_core/02_kv_cache_and_radix_attention.md) | [1.2 请求链路](01_foundations/02_process_topology_and_request_path.md) |
| 一个 HF 模型最终选择原生实现还是 Transformers fallback？ | [4.2 模型、多模态与 Fallback](04_interfaces_and_models/02_model_support_and_multimodal.md) | [2.3 ModelRunner](02_runtime_core/03_model_runner_attention_cuda_graph.md) |
| 图片如何变成 LLM 能消费的 token/features？ | [4.2 模型与多模态](04_interfaces_and_models/02_model_support_and_multimodal.md) | [1.2 请求链路](01_foundations/02_process_topology_and_request_path.md) |
| embedding、rerank、classification 与生成走的是同一条路吗？ | [4.2 模型任务](04_interfaces_and_models/02_model_support_and_multimodal.md) | [4.1 API 边界](04_interfaces_and_models/01_api_request_and_sampling.md) |
| 在线换权时在途请求会不会混用两个版本？ | [4.3 控制面与在线换权](04_interfaces_and_models/03_control_plane_and_post_training.md) | [5.3 可靠性与运维](05_production_engineering/03_reliability_capacity_operations.md) |
| 更新权重失败后能否自动回滚？ | [4.3 更新一致性边界](04_interfaces_and_models/03_control_plane_and_post_training.md) | [5.3 可靠性与运维](05_production_engineering/03_reliability_capacity_operations.md) |
| LoRA 动态卸载为何不会影响正在执行的请求？ | [4.3 LoRA 生命周期](04_interfaces_and_models/03_control_plane_and_post_training.md) | [2.2 Cache](02_runtime_core/02_kv_cache_and_radix_attention.md) |
| TP/DP/EP/PP 分别切什么？ | [3.1 并行与 PD](03_scaling_and_deployment/01_distributed_and_pd_disaggregation.md) | [3.2 部署与路由](03_scaling_and_deployment/02_deployment_topology_and_routing.md) |
| PD 分离后请求和 KV 怎样交接？ | [3.1 并行与 PD](03_scaling_and_deployment/01_distributed_and_pd_disaggregation.md) | [3.2 部署与路由](03_scaling_and_deployment/02_deployment_topology_and_routing.md) |
| 性能差应该先测哪一段？ | [5.1 Benchmark 与观测](05_production_engineering/01_benchmark_profiling_observability.md) | [1.2 请求时间线](01_foundations/02_process_topology_and_request_path.md) |
| 如何修改 runtime 而不破坏资源释放？ | [5.2 扩展与测试](05_production_engineering/02_extension_and_correctness.md) | [5.3 可靠性](05_production_engineering/03_reliability_capacity_operations.md) |
| Diffusion 是否复用 LLM 的 autoregressive scheduler？ | [6.1 Diffusion 架构](06_diffusion/01_pipeline_architecture_and_serving.md) | [6.2 优化与扩展](06_diffusion/02_parallelism_cache_and_quantization.md) |

## 3. 完整学习路线

1. [基础与请求链路](01_foundations/README.md)
2. [Runtime 核心](02_runtime_core/README.md)
3. [扩展、分布式与部署](03_scaling_and_deployment/README.md)
4. [接口、模型与控制面](04_interfaces_and_models/README.md)
5. [生产工程](05_production_engineering/README.md)
6. [SGLang Diffusion](06_diffusion/README.md)
7. [参考资料](07_reference/README.md)

## 4. 按角色选择路线

| 目标 | 建议路线 |
|---|---|
| API/应用开发 | 1.2 → 4.1 → 4.2 → 5.3 |
| Runtime 开发 | 1.1 → 1.2 → 2.1 → 2.2 → 2.3 → 2.4 |
| 分布式推理 | Runtime 路线 → 3.1 → 3.2 → 3.3 |
| 模型接入 | 2.3 → 4.2 → 5.2 |
| RL/后训练 | 1.2 → 2.1 → 4.3 → 3.1 → 5.3 |
| 性能/生产运维 | 1.1 → 2.1 → 3.2 → 5.1 → 5.3 |
| Diffusion | 1.1 → 6.1 → 6.2 → 6.3 |

## 5. 每次读代码维护四本账

| 账本 | 关键字段 | 创建/取得所有权 | 释放/交还所有权 |
|---|---|---|---|
| Request | `rid`、input/output、finish、grammar、LoRA ID | API/TokenizerManager 接收 | finish、abort、异常清理 |
| Batch | forward mode、seq lens、sampling info | scheduler iteration 组批 | result process 后过滤/重组 |
| KV | req slot、physical indices、radix lock | admission/prefix match | cache、evict、free |
| Version | model weight、LoRA ID、tokenizer/template、sampling config | 请求进入或控制面发布 | 请求完成或版本退役 |

第一次阅读只跟普通文本生成请求，关闭 speculative、LoRA、PD 和 VLM。之后每打开一个高级功能，都问四个问题：它增加了什么状态？状态由谁拥有？跨进程如何传递？失败时谁负责回收？

## 6. 文档中的结论等级

为避免把工程建议写成代码保证，本文使用三种措辞：

- **当前实现**：能在本快照源码中直接找到对应控制流；
- **源码观察**：实现和注释/常见预期之间存在容易踩坑的细节；
- **生产建议**：代码未提供完整保证，需要部署层补足。

特别是在线换权、跨 rank 失败和断连清理，不应因为“API 返回 success”就推断出未在源码中实现的事务语义。

## 7. 一手资料

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang 官方文档](https://docs.sglang.io/)
- [SGLang 论文](https://arxiv.org/abs/2312.07104)
- [官方学习材料](https://github.com/sgl-project/sgl-learning-materials)

官方文档与本系列的完整映射见 [官方文档覆盖索引](07_reference/01_official_docs_coverage.md)。
