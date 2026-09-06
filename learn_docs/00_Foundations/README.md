# Infra 基础课

面向已有 RL 算法背景、系统知识从零学习的读者。用小进程、小张量和短请求建立直觉，再走训练／推理框架。CPU 练习不要求 GPU；GPU 实验使用实际 H20 环境。

| 顺序 | 课程 | 首次完成标准 |
|---|---|---|
| 0 | [课程与系统地图](00_课程与系统地图.md) | 角色、数据、权重和控制图 |
| 1 | [Linux 与运行环境](01_Linux与运行环境.md) | 环境与进程记录 |
| 2 | [PyTorch 张量与自动微分](02_PyTorch张量与自动微分.md) | storage/stride 与梯度算例 |
| 3 | [进程、线程与四种异步](03_进程线程与四种异步.md) | 提交/执行/等待对照图 |
| 4 | [GPU 执行与内存](04_GPU执行与内存基础.md) | Kernel、内存层级与正确计时 |
| 5 | [Transformer 与 KV](05_Transformer执行与KV基础.md) | Dense shape 图与 KV 核算 |
| 6 | [两卡通信与 torchrun](06_两卡通信与torchrun.md) | 两进程结果核对 |
| 7 | [Ray 与队列调度](07_Ray与队列调度.md) | 有界提交实验与样本账 |
| 8 | [RL 数据与数值正确性](08_RL数据与数值正确性.md) | loss/mask/version 算例 |
| 9 | [第一个 Triton Kernel](09_第一个TritonKernel.md) | 正确性与 shape 扫描报告 |

第 6 课连接通信/并行；第 7–8 课在进入 slime 前回读；第 9 课可在首次 profiler 后做，也可随热点分析二读。完整周期见[学习清单](../学习清单.md)。

原论文共享基础已拆入第 4–5 课和 [Hopper 专项](../01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md)；原长笔记按主题并入 Kernel、通信、并行与性能。旧路径保留导航，内容去向见[重构说明](../重构说明与迁移索引.md)。
