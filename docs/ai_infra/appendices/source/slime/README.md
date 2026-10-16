# slime 源码教材

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../README.md>)。


这里从“会写程序”出发：先理解一条回答怎样成为训练数据，再看分布式执行为什么需要服务、队列、通信组和版本协议。建议按 00→01→02→03→05 阅读闭环，随后用 04/06/13 理解资源与权重，07/08 理解任务接入与反馈，09 学习验证方法；tau-bench 篇把这些机制接成一个具体案例，10 是独立数据平面的对照阅读。

贯穿例子的最小规模是 **2 个 prompt（输入任务）×每题 2 条回答**。`B` 表示任务组数，`G` 表示每组采样数，`P` 表示提示 token 数，`R` 表示 response 区间长度，`L=P+R`；有效训练 token 数是 mask 的和，未必等于 `R`。token 是分词器产生的整数编号，不保证等于一个汉字或一个单词。不同篇章另有局部符号时会重新定义。

本轮实现核对基于 **2026-10-16 工作区、主仓库 HEAD `ff067fd`**。源码片段以符号和调用关系为定位依据，旧稿的行号可能漂移。当前部署编排已拆到 `slime/backends/sglang_utils/deployment.py`，观测代码位于 `slime/observability/`；工作区没有 `train_async.py`，对应内容作为历史时序设计解释，不能直接照文件名启动。手算数字是教学输入，不是性能实测；量化、并发和异步的收益均须结合适用条件理解。

- [RL Infra 的设计空间：编排、数据流与策略版本](<00_rl_infra_survey.md>)
- [01 顶层架构与 Ray 编排：一次 RL 训练是如何被组织起来的](<01_architecture_and_ray_orchestration.md>)
- [02 Rollout 子系统：Server 模式的 SGLang 推理是如何生成训练数据的](<02_rollout_sglang_server_mode.md>)
- [03 数据流：Sample、Data Buffer、Partial Rollout 与异步](<03_data_buffer_partial_rollout_async.md>)
- [04 权重同步与显存管理：RL 系统的"头号瓶颈"是如何被攻克的](<04_weight_sync_and_memory.md>)
- [05 RL 算法实现：从 logprob 到 loss 的完整链路](<05_rl_algorithms.md>)
- [06 Megatron 训练后端与 HF↔Megatron 格式转换](<06_megatron_backend_and_mbridge.md>)
- [07 自定义接口与 Agentic RL：把任意数据生成流程接入训练闭环](<07_customization_and_agentic.md>)
- [08 奖励模型与评估体系：RM Hub、Dynamic Filter 与 Eval Pipeline](<08_rm_hub_and_eval.md>)
- [09 工程化与可观测性：从现象定位到可验证的原因](<09_engineering_observability.md>)
- [10 TransferQueue：RL 系统的独立数据平面](<10_transferqueue.md>)
- [13 训练侧内部实现：slime 内建 HF↔Megatron 权重转换](<13_megatron_bridge_internals.md>)
- [tau-bench + Qwen3-4B：从工具交互到一次参数更新](<tau-bench_qwen3_4B.md>)
