# 从 0 系统学习 RL Infra：slime 代码走读系列

本系列面向零基础读者，以本仓库（THUDM/slime）为教材系统学习 RL 基础设施。**00-09 是 slime 主线，10-12 是数据平面/第三方引擎专题，13 回到 slime 自己的 HF↔Megatron 转换实现**。每篇尽量遵循“问题 → 调用链 → 源码符号 → 例子 → 边界条件”的结构。

> **版本说明（重要）**：本文档已按 slime v0.3.1 代码重新核对。v0.3.1 已移除 Megatron-Bridge 与 `bridge` mode，HF checkpoint 的加载、导出和 rollout 热更新均由 `slime/backends/megatron_utils/{hf_to_megatron,megatron_to_hf}/` 内建实现。旧文件名 `06_megatron_backend_and_mbridge.md`、`13_megatron_bridge_internals.md` 为避免外部链接失效而保留，内容讲的是当前实现。

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
| 06 | [Megatron 后端与格式转换](06_megatron_backend_and_mbridge.md) | 训练后端 | 原生模型 provider、HF checkpoint 直载、双向转换、routing replay、TensorBackuper、StatelessAdam |
| 07 | [自定义接口与 Agentic RL](07_customization_and_agentic.md) | 应用层 | 三层自定义接口、rollout sample hooks、`slime/agent/` TITO 层、TrajectoryManager drift 处理、examples 全景 |
| 08 | [奖励模型与评估体系](08_rm_hub_and_eval.md) | reward/评估 | RM Hub 分发架构、数学答案等价性判断、Dynamic Filter、EvalDatasetConfig、eval early stop |
| 09 | [工程化与可观测性](09_engineering_observability.md) | 工程化 | 分离调试、权重对账、容错（健康监控状态机）、trace、**手把手 profiling 五步法**、CI、可复现 |

### 专题深入（10-13，按需查阅，非必读路径）

| # | 文档 | 定位 | 一句话内容 |
|---|---|---|---|
| 10 | [TransferQueue：独立数据平面](10_transferqueue.md) | **对比参考**（slime 未采用） | Ascend TransferQueue 源码走读：BatchMeta、Controller 账本、Sampler、存储后端、与 slime 数据流对照 |
| 11 | [引擎内部实现（SGLang 篇）](11_engine_internals_sglang.md) | slime 真实依赖 | `sglang/` 源码走读：RL 端点三层调用链、NCCL 组管理、IPC 还原、torch_memory_saver、abort、router 一致性哈希 |
| 12 | [训练侧内部实现：Megatron-LM 篇](12_megatron_lm_internals.md) | slime 真实依赖 | `Megatron-LM/` 源码走读：get_model/mpu 通信组/DistributedOptimizer/1F1B 流水线调度 |
| 13 | [训练侧内部实现：内建权重转换篇](13_megatron_bridge_internals.md) | **slime 自身实现** | `hf_to_megatron/`、`megatron_to_hf/`、`HfWeightIteratorDirect`：加载、热更新、HF 导出的三条路径 |

> 12 讲 Megatron-LM 训练执行引擎；13 讲 slime 自己维护的格式转换层。两者一边负责“怎么训练”，一边负责“同一份参数如何在 HF/Megatron 命名和并行布局之间移动”。

### 综合实战篇（独立示例，贯穿主线多篇知识点）

- **[tau-bench_qwen3_4B.md](tau-bench_qwen3_4B.md)**：以 tau-bench 客服 agent benchmark 为例，走一遍从启动脚本、资源配置、GRPO 算法参数到训练循环的完整端到端实践，是检验你是否吃透 01/02/05/07 篇的最佳练习——建议在读完主线后回来对照这篇，而不是当作"补充笔记"随便翻翻。

每篇 00-13 都包含「深入拆解」小节——针对该篇最复杂的机制（如 Ray 探测重排、TP 下的分布式 logsumexp、TensorBackuper 的 pinned-memory 影子权重、TrajectoryManager 的 re-tokenization drift、GRPOGroupNSampler 的连续段扫描、Megatron RankGenerator 的 rank 网格公式、HF↔Megatron 的 QKV/TP 转换）给出可以用符号名回到源码验证的例子。

## 常见问题索引

如果不是按顺序学习，而是带着问题查代码，可以从这里进入：

| 问题 | 先读 |
|---|---|
| 一轮 rollout、训练、权重同步的准确先后顺序是什么？下一轮 rollout 会不会落后一拍？ | [01](01_architecture_and_ray_orchestration.md)、[tau-bench §5.5](tau-bench_qwen3_4B.md) |
| colocate 为什么没有 `--update-weight-transport=ipc`，却走了 CUDA IPC？ | [04 §1](04_weight_sync_and_memory.md) |
| `old_log_probs` 到底来自 SGLang、旧 actor，还是训练侧重算？ | [05](05_rl_algorithms.md)、[tau-bench 常见疑问](tau-bench_qwen3_4B.md) |
| partial rollout 续写旧前缀算不算 off-policy，怎么 mask？ | [03 §3](03_data_buffer_partial_rollout_async.md) |
| 自定义生成、sample hook、RM、dynamic filter 的执行顺序是什么？ | [07 §1](07_customization_and_agentic.md)、[08](08_rm_hub_and_eval.md) |
| HF checkpoint 现在如何加载进 Megatron，为什么不再需要 Bridge？ | [06 §3-4](06_megatron_backend_and_mbridge.md)、[13](13_megatron_bridge_internals.md) |
| QKV/gate-up/TP/EP 分片究竟在哪一层转换？ | [13](13_megatron_bridge_internals.md) |
| `--use-kl-loss --kl-loss-coef 0` 是否仍会加载并前向 reference model？ | [tau-bench 常见疑问](tau-bench_qwen3_4B.md) |

## 前置知识

- PyTorch 基础与张量并行（TP）的直觉即可起步；流水线（PP）、专家并行（EP）、上下文并行（CP）在各篇用到处均有解释；
- 了解 PPO/GRPO 的算法概念有助于读第 05 篇，但篇内从公式到代码都有展开；
- Ray 不需要先学——01 篇用"可远程调用的有状态进程"一个模型就够用了。

## 约定

- 代码定位以“仓库相对路径 + 符号名”为准；文中的固定行号只用于帮助阅读当前快照，代码演进后请用 `rg '符号名' 路径` 重新定位；
- 文档里的代码块若标有“精简”或使用伪代码，只表达控制流，不保证可直接复制执行；
- 各篇末尾有动手建议，RL infra 是"读十遍不如跑一遍"的领域。
