# 从 0 系统学习 RL Infra：slime 代码走读系列

本系列面向零基础读者，以本仓库（THUDM/slime）为教材系统学习 RL 基础设施。两篇主线文档 + 八篇子领域走读，每篇遵循"背景 → 调用链 → 代码精读 → 实例"的结构。

## 阅读顺序

| # | 文档 | 子领域 | 一句话内容 |
|---|---|---|---|
| 00 | [RL Infra 全景调研](00_rl_infra_survey.md) | 领域地图 | slime / verl / OpenRLHF / NeMo-RL / AReaL / ROLL / TRL / TorchForge / RLinf 的近期工作、Roadmap 与十大工程问题 |
| 01 | [顶层架构与 Ray 编排](01_architecture_and_ray_orchestration.md) | 编排 | `train.py` 主循环、placement group、colocate/分离、同步 vs one-step-async |
| 02 | [Rollout 子系统](02_rollout_sglang_server_mode.md) | 推理架构 | server 模式、router、异步生成循环、over-sampling、abort |
| 03 | [数据流与异步](03_data_buffer_partial_rollout_async.md) | 数据/长尾/异步 | `Sample` 不变量、DataSource/buffer、partial rollout、fully async |
| 04 | [权重同步与显存管理](04_weight_sync_and_memory.md) | 系统瓶颈 | NCCL/IPC/磁盘三种传输、分桶与格式转换、offload 错峰、FP8 |
| 05 | [RL 算法实现](05_rl_algorithms.md) | 算法 | PPO/GRPO/GSPO/CISPO、KL 估计、GAE、TIS、CP 布局 |
| 06 | [Megatron 后端与格式转换](06_megatron_backend_and_mbridge.md) | 训练后端 | 原生透传、bridge 模式、mbridge/megatron_to_hf、routing replay |
| 07 | [自定义接口与 Agentic RL](07_customization_and_agentic.md) | 应用层 | 三层自定义接口、`slime/agent/` TITO 层、examples 全景 |
| 08 | [工程化与可观测性](08_engineering_observability.md) | 工程化 | 分离调试、权重对账、容错、trace/profiling、CI、可复现 |
| 09 | [TransferQueue：独立数据平面](09_transferqueue.md) | 数据流中间件 | Ascend TransferQueue 源码走读（本仓库根目录第三方源码）：BatchMeta、Controller 账本、Sampler、存储后端、与 slime 数据流对照 |
| 10 | [引擎内部实现（SGLang 篇）](10_engine_internals_sglang.md) | 推理引擎内部 | `sglang/` 源码走读：RL 端点三层调用链、NCCL 组管理、IPC 还原、torch_memory_saver、abort、logprob/回放数据、router 一致性哈希 |
| 11 | [训练侧内部实现](11_megatron_and_bridge_internals.md) | 训练引擎内部 | `Megatron-LM/` + `Megatron-Bridge/` 源码走读：get_model/mpu 通信组/DistributedOptimizer/流水线调度；AutoBridge/CONFIG_MAPPING/ParamMapping |

另有一篇独立示例走读：[tau-bench Qwen3-4B](tau-bench_qwen3_4B.md)。

## 前置知识

- PyTorch 基础与张量并行（TP）的直觉即可起步；流水线（PP）、专家并行（EP）、上下文并行（CP）在各篇用到处均有解释；
- 了解 PPO/GRPO 的算法概念有助于读第 05 篇，但篇内从公式到代码都有展开；
- Ray 不需要先学——01 篇用"可远程调用的有状态进程"一个模型就够用了。

## 约定

- 代码引用格式：`文件路径:行号区间`，行号以写作时仓库快照为准，上游演进后请用符号名搜索定位；
- 各篇末尾有动手建议，RL infra 是"读十遍不如跑一遍"的领域。
