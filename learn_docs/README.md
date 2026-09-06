# RL Infra 学习知识库

面向已有 RL 算法经验、希望学习 **RL 框架与调度、训练／推理性能优化** 的工程师。基础从进程、PyTorch 存储和 GPU 执行开始，通信与 Kernel 保留为必要基础；模型与底层实现按阶段深入。

**第一次从[基础课](00_Foundations/README.md)开始，按[学习清单](学习清单.md)记录进度。** 清单以每周 20+ 小时、约 16 周组织阅读与实验，提供首读范围和验收，不要求一次通读所有长文。

## 一条贯穿主线

```text
RL 系统概览 → 单机/PyTorch/GPU → 两卡通信与并行
 → Megatron 训练 → SGLang 推理与调度
 → slime 资源/样本/换权/异步 → 性能优化与工程交付
```

性能测量从基础阶段开始：先正确计时，再看时间线，最后对确定的热点下钻 Kernel。先看系统全景，基础学完后沿主线二读。

## 按阶段学习

| 模块 | 主入口 | 学会什么 |
|---|---|---|
| A0–A1 系统全景与基础 | [基础课](00_Foundations/README.md) | 进程、tensor/storage、GPU、prefill/decode 与首份 trace |
| A2 通信与内存 | [通信专题](02_Distributed_Communication_Memory/00_专题总览.md) | rank/group、显存、Collective、NCCL、拓扑与重叠 |
| A3 并行与训练 | [并行原理](03_Parallelism/00_专题总览.md) · [Megatron](../docs/megatron_code_walkthrough/README.md) | shape/owner/通信、step、optimizer/checkpoint |
| A4 推理与调度 | [SGLang](../docs/sglang_code_walkthrough/README.md) | 请求、Scheduler、KV、ModelRunner、采样与路由 |
| A5 RL 框架 | [slime](../docs/code_walkthrough/README.md) | Ray、buffer、长尾、换权、共置/分离与正确性 |
| A6 性能分析 | [统一性能教材](../docs/performance_analysis_guide/README.md) | 原理、工具、训练、推理、RL、动态负载与案例 |
| A7 可靠性与交付 | [清单：交付阶段](学习清单.md#delivery) | 恢复、资源回收、回归与报告 |
| A8 Kernel 与模型 | [Kernel](01_Kernel_GPU_Programming_Compiler/00_专题总览.md) · [模型](model_arc/README.md) · [论文](../docs/paper_read/README.md) | 小 Kernel 实践；按瓶颈深入 MoE/长上下文/PD |

## 带着问题查阅

| 问题 | 先看 |
|---|---|
| 删除 tensor 后显存为什么没降？ | [PyTorch 存储](00_Foundations/02_PyTorch张量与自动微分.md) → [显存分析](../docs/performance_analysis_guide/02_pytorch.md#concept-08) |
| rank 与 GPU 编号是什么关系？ | [两卡通信](00_Foundations/06_两卡通信与torchrun.md) → [进程与地址空间](02_Distributed_Communication_Memory/01_进程设备与地址空间.md) |
| TP、DP、EP 怎样组合？ | [多维并行](03_Parallelism/10_多维并行与DeviceMesh.md) → [Megatron 问题](../docs/megatron_code_walkthrough/06_reference/03_source_questions.md) |
| 请求为什么排队、占 KV？ | [KV 基础](00_Foundations/05_Transformer执行与KV基础.md) → [Scheduler](../docs/sglang_code_walkthrough/02_runtime_core/01_scheduler_and_batch.md) |
| NCCL 时间长就是网络慢吗？ | [通信与到达偏斜](../docs/performance_analysis_guide/04_megatron.md#concept-09) |
| RL 优化怎样避免改错算法？ | [数值正确性](00_Foundations/08_RL数据与数值正确性.md) → [slime loss](../docs/code_walkthrough/05_rl_algorithms.md) |
| 换权与缓存如何配合？ | [slime 换权](../docs/code_walkthrough/04_weight_sync_and_memory.md) → [SGLang 控制面](../docs/sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md) |
| 怎样记录结论？ | [实验工作表](../docs/performance_analysis_guide/experiment_worksheet.md) → [报告模板](../docs/performance_analysis_guide/performance_report_template.md) |

## 使用约定

每篇“学习定位”说明模块、阅读等级、前置与首读范围。分层必修表示入门部分必须学，其余按问题回读；专项/参考内容完整保留。示意数字与教学代码不能当作 H20 实测。

基础概念集中维护，源码保留实现差异和必要短回顾；论文讲设计取舍，性能教材讲测量验证。原 Profiling 专题、长笔记和引擎摘要已经整合，旧路径为兼容入口，不需重复阅读。

- [当前重构结果与完整迁移索引](重构说明与迁移索引.md)
- [初始映射讨论稿](RL_Infra_文档映射清单.md)
- [版本与验证范围](版本与验证范围.md)

先问题和直觉，再讲形状/状态/实现；代码定位优先用路径与符号名；公式用 `$...$` / `$$...$$`；区分教学假设、论文报告、本仓库实现与实测。
