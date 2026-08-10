# Megatron-LM / Megatron Core 中文系统导读与代码走读

这不是 NVIDIA 文档的逐句镜像，而是一套基于本仓库固定版本重写的中文教材。它把官方 User Guide、关键 API Guide 与 `Megatron-LM` 源码按工程问题重新组织：先跑通训练，再理解并行和状态，最后进入高级模型与性能排障。

## 阅读基线与范围

- 源码基线：本仓库 `Megatron-LM/` 子模块 commit `21fe0fe1597932421f1bb0efd93f469376a7a255`；符号名优先于易漂移的行号。
- 在线核对：[Megatron Core 官方文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/)与 [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)。
- `Megatron Core` 是 `megatron/core/` 下的可组合训练库；`Megatron-LM` 是使用它的端到端参考训练程序；`Megatron Bridge` 负责与 Hugging Face 等格式互转。
- API 自动生成页不逐个翻译函数签名；本系列解释稳定抽象、数据流和关键源码入口。完整覆盖边界见[官方文档覆盖矩阵](06_reference/01_official_docs_coverage.md)。

## 六个主题层

| 层次 | 章节 | 读完能回答什么 |
|---|---|---|
| 1. 基础与主线 | [生态、安装、数据](01_foundations/00_ecosystem_install_and_data.md) · [学习地图](01_foundations/01_learning_map.md) · [一次参数更新](01_foundations/02_training_mainline.md) · [模型与 ModuleSpec](01_foundations/03_model_and_transformer.md) | 程序如何启动，样本如何进入模型，一次 step 在哪里提交 |
| 2. 并行 | [选择与拓扑](02_parallelism/00_strategy_and_topology.md) · [TP/SP 与进程组](02_parallelism/01_parallel_state_and_tensor_parallel.md) · [PP](02_parallelism/02_pipeline_parallel.md) | 每个维度切什么、在哪个 group 通信、如何组合 |
| 3. 状态与显存 | [DDP/Optimizer/Checkpoint](03_state_and_memory/01_data_parallel_optimizer_checkpoint.md) · [Megatron-FSDP](03_state_and_memory/02_megatron_fsdp.md) · [显存、确定性与 CUDA Graph](03_state_and_memory/03_memory_determinism_cuda_graph.md) | 参数、梯度、优化器和 activation 由谁持有，怎样保存与复现 |
| 4. 模型与高级特性 | [模型与 Tokenizer](04_models_and_features/01_models_and_tokenizers.md) · [MoE/MLA/MTP](04_models_and_features/02_moe_mla_mtp.md) · [Hybrid/多模态/RL](04_models_and_features/03_hybrid_multimodal_rl.md) · [CP/MoE/低精度深读](04_models_and_features/04_context_moe_precision_deep_dive.md) | 新架构怎样落到模块、路由、布局和精度状态 |
| 5. 实战 | [性能与排障](05_practice/01_performance_debugging.md) · [配置评审清单](05_practice/02_configuration_review.md) | 如何从 OOM、hang 或吞吐下降定位到正确层次 |
| 6. 参考 | [覆盖矩阵](06_reference/01_official_docs_coverage.md) · [术语表](06_reference/02_glossary.md) | 去哪里继续查，术语如何统一 |

## 推荐路线

第一次接触时按表格从上到下读。已有分布式训练经验时，可先读“选择与拓扑”，再按问题跳转：

- OOM：状态与显存 → 性能排障。
- collective hang：TP/SP 进程组 → PP P2P → 性能排障。
- 新模型：ModuleSpec → 模型与 Tokenizer → 对应高级特性。
- 改并行度后恢复：DDP/Optimizer/Checkpoint → FSDP → 配置评审。

每章建议三遍：第一遍只画控制流；第二遍记录 tensor 的 global/local shape、所属 group 与 dtype；第三遍做最小实验并用日志或 profiler 验证。

## 文档维护约定

1. 引用源码使用仓库相对路径和符号名，不复制大段实现。
2. 随上游升级先更新覆盖矩阵，再更新受影响章节。
3. 配置建议必须说明硬件、模型、序列长度与并行拓扑，不能把示例参数当通用最优值。
4. 对 nightly 才存在的功能明确标为“上游动态能力”，不要假定当前快照一定支持。
