# SGLang 系统学习与源码走读

> 全库学习入口：[总目录](<../../learn_docs/README.md>)；[分阶段清单](<../../learn_docs/学习清单.md>)。本系列负责框架实现，公共基础在[基础课](<../../learn_docs/00_Foundations/README.md>)中补齐。引擎机制统一在对应的完整源码章节中维护。

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

目录编号只表示推荐学习顺序，不表示进程调用顺序。完整章节保留原文件名，分组介绍与导航集中在本页；每个章节内部改用“问题 → 结论 → 源码链路 → 边界/失败 → 排查”的结构。

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

<a id="part-1"></a>

### 第一部分：基础与请求链路

这一部分先建立性能、内存和状态机心智模型，再跟踪一个请求穿过 HTTP、Tokenizer、Scheduler、模型执行和 Detokenizer 的完整路径。

0. [Runtime 概念与算例](01_foundations/00_runtime_concepts.md)
1. [Infra 工程师的学习地图](<01_foundations/01_learning_map.md>)
2. [进程拓扑与端到端请求路径](<01_foundations/02_process_topology_and_request_path.md>)

读完后应能画出进程与 IPC 拓扑，区分 TTFT、ITL、TPOT、E2E、throughput 和 goodput，并说清 request、batch、KV 三种生命周期。

<a id="part-2"></a>

### 第二部分：Runtime 核心

这一部分从调度、内存、执行、生成控制四条主线解释 `sglang.srt`。建议按顺序阅读，因为后一篇会使用前一篇建立的 request/batch/KV 状态。

1. [Scheduler、请求状态机与连续批处理](<02_runtime_core/01_scheduler_and_batch.md>)
2. [KV Cache、RadixAttention 与引用锁](<02_runtime_core/02_kv_cache_and_radix_attention.md>)
3. [ModelRunner、Attention Backend 与 CUDA Graph](<02_runtime_core/03_model_runner_attention_cuda_graph.md>)
4. [推测解码、约束输出与采样正确性](<02_runtime_core/04_speculative_structured_sampling.md>)

<a id="part-3"></a>

### 第三部分：扩展、分布式与部署

这一部分回答三个不同问题：单个实例如何跨设备执行、多个实例如何组成服务、不同硬件平台如何验证能力边界。

1. [分布式执行与 Prefill-Decode 解耦](<03_scaling_and_deployment/01_distributed_and_pd_disaggregation.md>)
2. [部署拓扑、多节点与路由](<03_scaling_and_deployment/02_deployment_topology_and_routing.md>)
3. [硬件平台与安装选择](<03_scaling_and_deployment/03_hardware_platforms.md>)

<a id="part-4"></a>

### 第四部分：接口、模型与控制面

这一部分专门解释 runtime 两侧的边界：左侧是外部协议怎样变成内部请求，右侧是模型/权重怎样成为可执行状态。三章各自回答一组不同的问题，避免把 HTTP、模型加载和在线换权混成一条模糊的“serving 流程”。

#### 章节职责

| 章节 | 核心问题 | 主要状态所有者 |
|---|---|---|
| [4.1 API、请求规范化与采样](<04_interfaces_and_models/01_api_request_and_sampling.md>) | 用户参数最后变成什么？采样与 stop 在哪里生效？ | OpenAI serving、`GenerateReqInput`、`SamplingParams`、`SamplingBatchInfo` |
| [4.2 模型支持、多模态与 Fallback](<04_interfaces_and_models/02_model_support_and_multimodal.md>) | 模型类如何选择？媒体如何变成 LLM 输入？不同 task 在哪里分流？ | `ModelRegistry`、model loader、multimodal processor、task serving |
| [4.3 控制面、在线权重更新与后训练](<04_interfaces_and_models/03_control_plane_and_post_training.md>) | 换权如何隔离请求？失败是否可回滚？LoRA 如何安全上下线？ | `TokenizerManager`、weight updater、scheduler、`LoRARegistry` |

#### 推荐阅读顺序

先完成 [1.2 进程拓扑与请求链路](<01_foundations/02_process_topology_and_request_path.md>)，再读 4.1。模型接入或 VLM 问题读 4.2；RL、在线换权、动态 LoRA 和运维控制面读 4.3。

下面三个判断贯穿本部分：

1. 外部协议对象不是 scheduler 直接消费的对象，中间至少经历协议转换、输入规范化和 tokenization；
2. “模型可由 Transformers 加载”不等于“模型已走 SGLang 原生高性能路径”；
3. “控制面操作被串行化”不等于“跨 rank 更新具备事务回滚”。

每章末尾都有源码定位清单，路径相对仓库的 `sglang/python/sglang/`。

<a id="part-5"></a>

### 第五部分：生产工程

这一部分不再按功能开关组织，而是按生产工作的三个闭环组织：测量、变更和运行。

1. [Benchmark、Profiling 与可观测性](<05_production_engineering/01_benchmark_profiling_observability.md>)
2. [模型与 Backend 扩展、正确性验证](<05_production_engineering/02_extension_and_correctness.md>)
3. [可靠性、容量保护与发布运维](<05_production_engineering/03_reliability_capacity_operations.md>)

<a id="part-6"></a>

### 第六部分：SGLang Diffusion

Diffusion 与自回归 LLM 的主循环不同，单独成篇：

1. [Pipeline 架构、服务接口与动态批处理](<06_diffusion/01_pipeline_architecture_and_serving.md>)
2. [并行、Backend、缓存与量化](<06_diffusion/02_parallelism_cache_and_quantization.md>)
3. [Profiling、新模型接入与正确性](<06_diffusion/03_profiling_extension_and_correctness.md>)

<a id="part-7"></a>

### 第七部分：参考资料

- [官方文档完整覆盖索引](<07_reference/01_official_docs_coverage.md>)：官方正文、Advanced Features 和 Cookbook 到本系列章节的映射。
- [中英文术语表](<07_reference/02_glossary.md>)：统一 serving、并行、生成和 Diffusion 术语。

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
