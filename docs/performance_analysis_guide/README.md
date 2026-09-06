# 训练、推理与 RL 性能分析：统一教材

本目录同时负责原理、逐步操作、框架实验与报告。原 learn_docs 的 Profiling 17 篇已整合到这里，旧目录和跳转页已删除，原理与操作统一在本目录维护。初学者从 00/01/02 开始；GPU/进程前置见[基础课](../../learn_docs/00_Foundations/README.md)。

## 主线与首读范围

| 顺序 | 章节 | 首次掌握 |
|---|---|---|
| 0 | [性能语言与方法](00_concepts.md) | 关键路径、口径、可证伪假设、测量扰动 |
| 1 | [环境、基线与遥测](01_baseline.md) | 固定 workload，理解 GPU 指标 |
| 2 | [PyTorch、CPU 与显存](02_pytorch.md) | 正确计时、首份 trace、数据供应和内存 |
| 3 | [CUDA 工具与 Roofline](03_cuda_kernels.md) | 先 nsys，再按热点 ncu；编译/SASS 二读 |
| 4 | [训练与分布式](04_megatron.md) | step、MFU、通信、rank 偏斜和 bubble |
| 5 | [推理](05_inference.md) | workload、KV、容量曲线与尾延迟 |
| 6 | [RL 全链路](06_slime.md) | 角色等待、换权、offload、有效样本 |
| 实验 | [六个递进实验](07_labs.md) | 随学习阶段完成，不一次全部执行 |
| 专项 | [动态负载](09_dynamic_workloads.md) | MoE、长上下文、RL 版本与 trace |
| 工程 | [线上观测](10_observability.md) | metric/log/trace、基数、告警 |
| 案例 | [GPU 空闲案例](11_case_gpu_idle.md) | 从现象到干预，数字为教学构造 |

## 工具与记录

- [软件教程](software_tutorials.md)：只读当前需要的工具章节。
- [术语词典](glossary.md)：先看最常用 30 词，其余随问题查。
- [工具与资料索引](08_toolbox_and_sources.md)：按症状选工具。
- [实验现场工作表](experiment_worksheet.md)：保留原始环境、假设、证据和测量。
- [正式报告模板](performance_report_template.md)：整理复现、结论、正确性与回归。

## 完成标准

每个实验至少有固定输入、正确性基线、重复测量、原始证据和清楚的适用范围。Profiler 数据用于诊断；性能收益使用无 profiler 的公平基准验证。脚本与 CLI 必须匹配实际环境，未运行的 H20/NCCL/框架实验不能填写虚构结果。

课程中每个“深入边界”小节已经与操作合在同一页，可按页内导航二读。回到[16 周学习清单](../../learn_docs/学习清单.md)选择本周范围，合并来源见[迁移索引](../../learn_docs/重构说明与迁移索引.md)。
