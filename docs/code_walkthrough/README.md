# 从 0 系统学习 RL Infra：slime 代码走读系列

本系列面向零基础读者，以本仓库（THUDM/slime）为教材系统学习 RL 基础设施。核心分两个部分：**00-09 讲 slime 自身的架构与实现**（阅读顺序上的主线，覆盖 slime 用到的每一层机制），**10-13 是延伸阅读**，深入 slime 依赖的第三方库（TransferQueue 是对比参考，非 slime 依赖；sglang/Megatron-LM/Megatron-Bridge 是 slime 真实依赖的库，讲它们内部如何响应 slime 的调用）。每篇遵循"背景 → 调用链 → 代码精读 → 实例 → 深入拆解"的结构。

## 阅读顺序

### 主线：slime 自身架构（00-09，建议按序读完）

| # | 文档 | 子领域 | 一句话内容 |
|---|---|---|---|
| 00 | [RL Infra 全景调研](00_rl_infra_survey.md) | 领域地图 | slime / verl / OpenRLHF / NeMo-RL / AReaL / ROLL / TRL / TorchForge / RLinf 的近期工作、Roadmap 与十大工程问题 |
| 01 | [顶层架构与 Ray 编排](01_architecture_and_ray_orchestration.md) | 编排 | `train.py` 主循环、placement group 探测重排、colocate/分离、rank0 rendezvous、断点续训 |
| 02 | [Rollout 子系统](02_rollout_sglang_server_mode.md) | 推理架构 | server 模式、model/server_group/engine 三层结构、router、异步生成循环、over-sampling、abort |
| 03 | [数据流与异步](03_data_buffer_partial_rollout_async.md) | 数据/长尾/异步 | `Sample` 不变量、DataSource/buffer、partial rollout、fully async |
| 04 | [权重同步与显存管理](04_weight_sync_and_memory.md) | 系统瓶颈 | NCCL/IPC/磁盘三种传输、分桶与格式转换、offload 错峰、FP8、异构 PD rank 分配 |
| 05 | [RL 算法实现](05_rl_algorithms.md) | 算法 | PPO/GRPO/GSPO/CISPO、KL 估计、GAE、TIS、CP 布局、分布式 logsumexp |
| 06 | [Megatron 后端与格式转换](06_megatron_backend_and_mbridge.md) | 训练后端 | 原生透传、bridge 模式、mbridge/megatron_to_hf、routing replay、TensorBackuper、StatelessAdam |
| 07 | [自定义接口与 Agentic RL](07_customization_and_agentic.md) | 应用层 | 三层自定义接口、`slime/agent/` TITO 层、TrajectoryManager drift 处理、examples 全景 |
| 08 | [奖励模型与评估体系](08_rm_hub_and_eval.md) | reward/评估 | RM Hub 分发架构、数学答案等价性判断、Dynamic Filter、EvalDatasetConfig、eval early stop |
| 09 | [工程化与可观测性](09_engineering_observability.md) | 工程化 | 分离调试、权重对账、容错（健康监控状态机）、trace、**手把手 profiling 五步法**、CI、可复现 |

### 延伸阅读：第三方依赖库源码（10-13，按需查阅，非必读路径）

| # | 文档 | 定位 | 一句话内容 |
|---|---|---|---|
| 10 | [TransferQueue：独立数据平面](10_transferqueue.md) | **对比参考**（slime 未采用） | Ascend TransferQueue 源码走读：BatchMeta、Controller 账本、Sampler、存储后端、与 slime 数据流对照 |
| 11 | [引擎内部实现（SGLang 篇）](11_engine_internals_sglang.md) | slime 真实依赖 | `sglang/` 源码走读：RL 端点三层调用链、NCCL 组管理、IPC 还原、torch_memory_saver、abort、router 一致性哈希 |
| 12 | [训练侧内部实现：Megatron-LM 篇](12_megatron_lm_internals.md) | slime 真实依赖 | `Megatron-LM/` 源码走读：get_model/mpu 通信组/DistributedOptimizer/1F1B 流水线调度 |
| 13 | [训练侧内部实现：Megatron-Bridge 篇](13_megatron_bridge_internals.md) | slime 真实依赖 | `Megatron-Bridge/` 源码走读：AutoBridge/CONFIG_MAPPING/ParamMapping，HF↔Megatron 双向转换 |

> Megatron-LM（训练执行引擎）与 Megatron-Bridge（格式转换/建模层）拆成两篇分别讲透，避免体量和主题差异悬殊的两个库被迫塞进一篇文档。

### 综合实战篇（独立示例，贯穿主线多篇知识点）

- **[tau-bench_qwen3_4B.md](tau-bench_qwen3_4B.md)**：以 tau-bench 客服 agent benchmark 为例，走一遍从启动脚本、资源配置、GRPO 算法参数到训练循环的完整端到端实践，是检验你是否吃透 01/02/05/07 篇的最佳练习——建议在读完主线后回来对照这篇，而不是当作"补充笔记"随便翻翻。

每篇 00-13 都包含「深入拆解」小节——针对该篇里最复杂/最容易一带而过的机制（如 Ray 探测重排、TP 下的分布式 logsumexp 自定义 autograd、TensorBackuper 的 pinned-memory 影子权重、TrajectoryManager 的 re-tokenization drift 分类、GRPOGroupNSampler 的连续段扫描、Megatron RankGenerator 的 rank 网格公式、数学答案等价性判断的括号计数状态机等），给出可直接对照源码验证的代码片段、具体数字例子与设计意图分析，而不只是描述"做了什么"。

## 前置知识

- PyTorch 基础与张量并行（TP）的直觉即可起步；流水线（PP）、专家并行（EP）、上下文并行（CP）在各篇用到处均有解释；
- 了解 PPO/GRPO 的算法概念有助于读第 05 篇，但篇内从公式到代码都有展开；
- Ray 不需要先学——01 篇用"可远程调用的有状态进程"一个模型就够用了。

## 约定

- 代码引用格式：`文件路径:行号区间`，行号以写作时仓库快照为准，上游演进后请用符号名搜索定位；
- 各篇末尾有动手建议，RL infra 是"读十遍不如跑一遍"的领域。
