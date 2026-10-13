# slime资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../README.md>)。

- [RL Infra 的设计空间：编排、数据流与策略版本](<00_rl_infra_survey.md>)
- [01 顶层架构与 Ray 编排：一次 RL 训练是如何被组织起来的](<01_architecture_and_ray_orchestration.md>)
- [02 Rollout 子系统：Server 模式的 SGLang 推理是如何生成训练数据的](<02_rollout_sglang_server_mode.md>)
- [03 数据流：Sample、Data Buffer、Partial Rollout 与异步](<03_data_buffer_partial_rollout_async.md>)
- [04 权重同步与显存管理：RL 系统的"头号瓶颈"是如何被攻克的](<04_weight_sync_and_memory.md>)
- [05 RL 算法实现：从 logprob 到 loss 的完整链路](<05_rl_algorithms.md>)
- [06 Megatron 训练后端与 HF↔Megatron 格式转换](<06_megatron_backend_and_mbridge.md>)
- [07 自定义接口与 Agentic RL：把任意数据生成流程接入训练闭环](<07_customization_and_agentic.md>)
- [08 奖励模型与评估体系：RM Hub、Dynamic Filter 与 Eval Pipeline](<08_rm_hub_and_eval.md>)
- [09 工程化与可观测性：调试、容错、Profiling、CI 与可复现](<09_engineering_observability.md>)
- [10 TransferQueue：RL 系统的独立数据平面](<10_transferqueue.md>)
- [13 训练侧内部实现：slime 内建 HF↔Megatron 权重转换](<13_megatron_bridge_internals.md>)
- [tau-bench + Qwen3-4B 训练全流程详解](<tau-bench_qwen3_4B.md>)
