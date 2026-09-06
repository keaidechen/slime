# RL Infra 文档映射清单：从算法工程师到框架、调度与性能工程

> 历史设计稿：下文记录重构前的建议，包含“拟议”“尚未修改”等当时状态。当前已执行的合并、扩充和学习路线请看[重构说明](重构说明与迁移索引.md)与[学习清单](学习清单.md)。

> 讨论稿 v1 · 2026-09-06\
> 读者：已有 RL 算法背景，Infra 按初学者处理。目标：RL 框架与调度、训练／推理性能优化，同时建立通信与 Kernel 基础。时间：每周 20+ 小时；资源：可使用多张 H20。\
> 范围：原有六部分共 **158 篇 Markdown，51,709 行**，不含本清单。第 13—16 篇 Profiling 分章已确认补齐。\
> 审阅方式：遍历全部文件及章节结构，结合前一轮与本轮代表性正文抽读。映射属于课程与内容组织建议，不代表逐篇技术事实、命令和源码版本已校验。

## 1. 本轮结论与保留原则

现有基础内容有必要保留。对这位读者，“懂 RL 算法”不能推导出“懂 CUDA 异步、进程、张量布局、通信组、缓存生命周期和系统性能”。

本轮建议保留全部 158 篇的内容资产，不直接删除或搬迁原文。后续可以整编重复正文，但必须先记录段落去向，保留独特的直觉、图、算例、推导、边界条件与学习问答。归为参考表示不占当前主线必读时间，不表示没有价值。

三个具体调整：

1. 把基础课列为明确必修，有可完成的小实验；不能只在源码篇前放一串术语链接。
2. 同一长文允许两次进入：第一次学直觉、形状和状态；第二次学实现、优化与推导。
3. 源码、论文和实操各有职责。基础定义可统一，教学中的短回顾与具体例子仍可就近保留。

## 2. 阅读等级与动作说明

| 等级 | 含义 | 本轮文档数 |
|---|---|---:|
| 必修 | 目标方向直接需要；按所在阶段学，不要求现在全部读完 | 77 |
| 分层必修 | 指定首读范围必须学；其余保留为二读或深入 | 38 |
| 专项 | 主线之后选题深入，例如 MoE、PD、Attention Kernel | 12 |
| 参考 | 保留完整内容，遇到相应模型、平台或问题时查阅 | 15 |
| 导航 | 路线、目录与局部入口，不单独计算学习工作量 | 16 |

115 篇带有必修内容，不等于 115 篇都要逐字通读或做 115 个实验。课程将同一问题的原理、源码、实验编成一个学习单元，并标明读哪些段落。

表中“映射模块”是未来课程的逻辑归属，**不是已经执行的文件迁移路径**。跨模块表示不同阶段回读同一篇，不复制多份正文。“分段整编”“分两遍”等均为拟议动作；本次只新增清单。

## 3. 新体系的逻辑模块与前置关系

| 模块 | 职责 | 前置 | 核心验收 |
|---|---|---|---|
| A0 目标与 RL 系统全景 | 看见一轮 rollout→reward→train→换权，以及角色、数据、控制的区别 | RL 算法经验 | 画出角色与一轮同步流程；此时不要求读懂底层源码 |
| A1 单机系统、PyTorch 与 GPU 基础 | Linux 运行环境、进程/线程、tensor/storage/autograd、GPU 执行与内存、正确计时 | A0 只需概览 | 追踪一次 PyTorch 运算、解释首份 trace、估计并实测显存 |
| A2 分布式通信与状态 | rank/group、Collective、拓扑、NCCL、跨节点路径、重叠 | A1 | 两卡通信输入输出核对、消息大小扫描、实际机器拓扑图 |
| A3 训练执行与并行 | DP/FSDP/TP/PP/SP/CP/EP、Megatron step、optimizer/checkpoint | A1+A2 | 用 local/global shape 解释一个训练 step，核对梯度与配置账本 |
| A4 推理执行与调度 | prefill/decode、batch、KV、Scheduler、ModelRunner、路由 | A1；多卡部分需 A2/A3 的 TP 基础 | 跟一条请求到完成/取消，解释容量曲线与缓存变化 |
| A5 RL 框架与协同调度 | Ray、数据队列、换权、colocate、partial/async、算法系统一致性 | A2+A3+A4；先补 Ray/async 前导 | 画清样本与权重版本流，解释等待、长尾和正确性边界 |
| A6 性能分析方法与实操 | 可信基线、时间线、通信/显存/Kernel 定位与单变量验证 | 入门随 A1，框架分析随 A3–A5 | 一份原始 trace、一个可证伪假设、一份含正确性验证的 A/B 报告 |
| A7 可靠性与工程交付 | 失败恢复、资源回收、可观测性、测试、配置与代码修改 | 对应运行系统已理解 | 一个可复现问题、最小改动、回归验证与恢复演练 |
| A8 底层与模型专项 | Triton/Attention Kernel、MoE、长上下文、PD、编译器及架构比较 | 按专题依赖 A1–A6 | 必做一个小 Kernel 实验；另选一个与实际瓶颈相关的深入题 |

阅读主序：A0 概览 → A1 → A2 → A3 → A4 → A5。A6 从 A1 开始伴随，A7 随对应系统加入；A8 中 Kernel 入门为共同基础，其他专项按选题进入。

可以提前看 A4 的 prefill/decode 概念帮助理解基础，但不要在尚不理解进程、队列和显存时直接从 Scheduler 长调用链起步。A3 先通过 DP/TP 小例子理解切分，再引入 DeviceMesh 的抽象总结。

## 4. 从初学者角度，需要补齐的“台阶”

下表表示现有内容中尚未形成足够连贯的初学者课程；不意味着这些词在当前文档中完全没出现。优先抽取已有解释，只对缺口新增短课与小例子。

| 拟补单元 | 为何需要 | 可复用现有材料 | 最小完成标准 |
|---|---|---|---|
| B01 Linux 与运行环境 | 能启动 Python 不等于能定位容器、驱动、子进程与端口问题 | 性能 baseline/software_tutorials；Megatron 安装；SGLang hardware | 保存环境、启动命令与日志，找到 PID/GPU/端口，解释驱动/runtime/框架边界 |
| B02 PyTorch 的系统视角 | shape/stride/storage/view/copy 与 autograd 生命周期直接影响性能、显存和分片 | PyTorch 性能篇、显存篇、Kernel 基础 | 用小 tensor 比较 view/copy/contiguous，跟一次 forward/backward 的状态释放 |
| B03 CPU 进程、线程与异步 | “异步”在 asyncio、Ray、CUDA、NCCL 中含义不同 | 进程地址篇、slime Ray/rollout、CPU 数据加载篇 | 分别画四种异步的提交、等待、完成，解释 CPU 阻塞与 GPU 空闲 |
| B04 torchrun 与两卡最小分布式 | rank、device、group 的定义需要运行实例支撑 | 进程地址篇、Collective、DDP | 两进程输出 PID/global/local rank/device，演示一个 collective 并核对结果 |
| B05 Ray 与队列调度 | actor、ObjectRef、资源声明、placement、背压是目标方向的基础 | slime 01/02/03、TransferQueue 对比 | 做小生产者/消费者示例，观察队列变长、超时、取消与资源声明 |
| B06 Transformer 的执行与数据布局 | 算法经验不能保证熟悉模型 runtime 中的 tensor shape、KV 与 padding | 共享 GPU 基础、MLA、模型构建、词表并行 | 画一个 Dense block 的 shape/dtype，计算 prefill/decode 的状态差别 |
| B07 RL batch 与数值正确性 | packing、mask、token mean、logprob 与版本边界是系统改动底线 | slime 03/05/13、Megatron source_questions、SGLang sampling | 小数据验证 mask/归一化/变长样本；解释训练与推理 logprob 差异来源 |
| B08 请求和状态的生命周期 | 不只是找到成功路径，还要知道失败后谁清理 | SGLang request/KV/control/reliability；slime 容错 | 跟普通请求、取消、失败三条路径，记录 request/KV/weight version 的所有者 |
| B09 调度与容量的基本模型 | 并发、到达率、服务率、长尾、背压与负载均衡需要统一语言 | SGLang Scheduler、Orca、RL 角色并行、推理性能 | 固定生成长度分布，画并发/吞吐/尾延迟/队列关系；解释生产消费不匹配 |
| B10 最小 CUDA/Triton 实践 | 只看 DSL 比较还不能把 Kernel 概念与执行联系起来 | CUDA 执行篇、Triton 篇、Kernel 实验手册 | 跑通并解释一个小算子，比较合理基线、数值误差与不同 shape 的时间 |

## 5. 基础长文如何保留与分段使用

### 5.1 AI Infra 学习笔记

保留其问答价值，按现有章节建立迁移账本：

| 原章节 | 主归属 | 首读与后读 |
|---|---|---|
| §01 GPU 硬件层级 | A1 | GPU/SM/执行单元必读；精确型号资源表参考 |
| §02 架构代际演进 | A8 | 首次只理解资源变化为何影响 workload；型号时间线与能力逐项核验 |
| §03–06 CUDA 执行、编译、驻留与调度 | A1→A8 | 执行/异步/Stream 必读；高级资源细节二读 |
| §07 GEMM 与工具链 | A1→A8 | 先理解算子与库的层次，再学实现与后端对比 |
| §08–10 进程、IPC 与互联 | A2 | 保留长解释和历史直觉，接两卡与拓扑实验 |
| §11–12 TP 通信与 activation | A3 | 保留完整推算；补齐假设、shape、单位、适用并行配置 |
| §13 Hockney 模型 | A2/A6 | 通信篇作为主入口，原问答作为算例与辨析 |
| §14 利用率与总复盘 | A6 | 将特定百分比与普遍判断区分，连接实际测量口径 |

### 5.2 论文共享基础

这篇虽然位于 paper_read，实际可作为重要入门教材来源：

- §1–14：CPU/GPU、HBM、内存层级、执行层次、fusion、tiling、Roofline、occupancy → A1。
- §15–21：prefill/decode、KV、分页、碎片、COW、prefix tree → A4。
- §22–25：论文地图、词汇和回顾 → 导航/自测。
- §26：TMA、WGMMA、warp specialization 等 → A8，实验适用性以实际机器和软件为准。

### 5.3 模型与论文长文

MoE/MLA/FlashInfer 等不压缩成只有结论的短摘要。先增加“本次读到哪里”的入口，再保留完整论证。需要物理拆分时，以一个独立问题为单元，原路径保留导航与段落迁移索引。

## 6. 重复主题的职责映射

| 主题 | 原理主入口 | 其他内容保留什么 | 避免的重复 |
|---|---|---|---|
| GPU 执行与内存 | A1：共享基础+Kernel 01/02，整编后建立统一入口 | 长笔记保留硬件/软件层级辨析和算例 | 每篇重写全套 GPU 百科 |
| 通信与分片 | A2 通信原理、A3 并行原理 | Megatron 讲具体 group/执行；slime 讲角色间布局转换 | 把不同框架的具体实现当作通用保证 |
| KV Cache | A4 内存原理 | 论文讲设计；SGLang 讲 pool/锁/释放；性能教程讲实验 | 删除独特的分页、COW、radix 教学例子 |
| 性能分析 | A6 原理专题 | performance_analysis_guide 负责逐步操作；框架篇保留专属观测点 | 两套独立“从零安装 profiler”正文 |
| 权重同步 | A5 slime 04/13 | slime 11 讲接线；SGLang 控制面讲端点内部和失败边界 | 混淆抽象版本切换要求与实际事务保证 |
| MoE | A8 模型概念与数据流 | 通信讲 dispatch/combine；并行讲布局；框架讲实现；Kernel 讲 GEMM | 全部合成难以定位的一篇超长文 |
| 实验记录 | 工作表保存原始证据 | performance_report_template 保存正式结论与复现步骤 | 工作表和报告各自维护互相冲突的测量规范 |
| 术语与版本 | 公共 glossary + 版本索引 | 框架局部词典与首次出现的一句话解释保留 | 只剩跳转链接，初学者无法就地理解 |

## 7. 全部原文逐篇映射

每个文件恰好出现一行。点击原文可回到当前内容。表内动作均为建议；没有执行移动、删除或合并。

### 7.1 learn_docs：入口与历史笔记（2 篇）

| 原文 | 映射模块 | 阅读等级 | 拟议动作 | 首读范围与后续用途 |
|---|---|---|---|---|
| [README.md](<./README.md>) | A0 | 导航 | 升级总入口 | 统一全库课程、前置依赖、进度与查阅入口；原专题入口保留 |
| [AI_Infra_学习笔记_2026-08-27.md](<./AI_Infra_学习笔记_2026-08-27.md>) | A1/A2/A3/A8 | 分层必修 | 分段整编，保留问答原文 | 首读 GPU 层级、执行、进程；TP 通信推算在 A3 回读；架构代际放参考层，修复旧路径 |

### 7.2 learn_docs/01_Kernel_GPU_Programming_Compiler（10 篇）

| 原文 | 映射模块 | 阅读等级 | 拟议动作 | 首读范围与后续用途 |
|---|---|---|---|---|
| [00_专题总览.md](<./01_Kernel_GPU_Programming_Compiler/00_专题总览.md>) | A1/A8 | 导航 | 保留并改路线 | 区分单卡基础与 Kernel 专项；不要求先读完所有 DSL |
| [01_CUDA执行模型与调度.md](<./01_Kernel_GPU_Programming_Compiler/01_CUDA执行模型与调度.md>) | A1 | 必修 | 保留，补最小运行例 | Kernel/Block/Warp/SM、异步 launch、Stream/Event；高级调度二读 |
| [02_GPU内存编程与数据搬运.md](<./01_Kernel_GPU_Programming_Compiler/02_GPU内存编程与数据搬运.md>) | A1→A8 | 分层必修 | 保留，分两遍 | 首读内存层级、带宽、合并访存、tiling；bank conflict 与高级搬运二读 |
| [03_GEMM与高性能Kernel模式.md](<./01_Kernel_GPU_Programming_Compiler/03_GEMM与高性能Kernel模式.md>) | A1→A8 | 分层必修 | 保留，补 shape 示例 | 先懂 GEMM、Tensor Core、batch/shape；流水线和 warp specialization 后读 |
| [04_CUDA算子库_CUTLASS与CuTe.md](<./01_Kernel_GPU_Programming_Compiler/04_CUDA算子库_CUTLASS与CuTe.md>) | A8 | 分层必修 | 保留，分两遍 | 先区分 CUDA/cuBLAS/CUTLASS 的职责；CuTe layout 和模板实现按热点深入 |
| [05_Triton_TileLang_CUDA-Tile与Helion.md](<./01_Kernel_GPU_Programming_Compiler/05_Triton_TileLang_CUDA-Tile与Helion.md>) | A8 | 分层必修 | 保留，拆主线与工具比较 | Triton 基本编程与一个小算子实践；其余 DSL 保留为比较参考 |
| [06_PyTorch编译栈.md](<./01_Kernel_GPU_Programming_Compiler/06_PyTorch编译栈.md>) | A1→A8 | 分层必修 | 保留，配 graph break 实验 | 首读 eager/compile、fusion、graph break、冷启动；编译器内部二读 |
| [07_IR_PTX_SASS与编译模式.md](<./01_Kernel_GPU_Programming_Compiler/07_IR_PTX_SASS与编译模式.md>) | A1→A8 | 分层必修 | 保留，分两遍 | 首读源码→PTX→SASS、JIT/AOT、架构兼容；IR/LLVM 深入参考 |
| [08_案例_FlashAttention与GroupedGEMM.md](<./01_Kernel_GPU_Programming_Compiler/08_案例_FlashAttention与GroupedGEMM.md>) | A8 | 专项 | 保留案例 | 先完成 A1/A4，再追 FlashAttention 与 Grouped GEMM 的 IO/shape/调度 |
| [09_实验与排查手册.md](<./01_Kernel_GPU_Programming_Compiler/09_实验与排查手册.md>) | A1/A8 | 必修 | 保留，接统一实验索引 | 正确计时与正确性基线先做；自定义 Kernel 比较在 A8 做 |

### 7.3 learn_docs/02_Distributed_Communication_Memory（16 篇）

| 原文 | 映射模块 | 阅读等级 | 拟议动作 | 首读范围与后续用途 |
|---|---|---|---|---|
| [00_专题总览.md](<./02_Distributed_Communication_Memory/00_专题总览.md>) | A2 | 导航 | 保留并补前置 | 先走两进程两卡示例，再进入通信与状态专题 |
| [01_进程设备与地址空间.md](<./02_Distributed_Communication_Memory/01_进程设备与地址空间.md>) | A2 | 必修 | 保留，补初学者前导 | process/rank/local_rank/device/context；UVA 与 IPC 生命周期第二遍 |
| [02_GPU内存组成与显存核算.md](<./02_Distributed_Communication_Memory/02_GPU内存组成与显存核算.md>) | A1/A2 | 必修 | 保留，补手算到实测 | 参数/梯度/优化器/activation/KV，区分 allocated/reserved/device used |
| [03_通信原语与Collective.md](<./02_Distributed_Communication_Memory/03_通信原语与Collective.md>) | A2 | 必修 | 保留，补两卡张量演示 | 逐个列输入输出与 rank，演示 AllReduce/AllGather/ReduceScatter/AllToAll |
| [04_Collective通信算法.md](<./02_Distributed_Communication_Memory/04_Collective通信算法.md>) | A2→A8 | 分层必修 | 保留，增加消息时序 | 先懂 ring/tree 的步骤与开销；PAT/协议内部后读 |
| [05_节点内GPU互联.md](<./02_Distributed_Communication_Memory/05_节点内GPU互联.md>) | A2 | 必修 | 保留，配本机拓扑记录 | PCIe/NVLink/NVSwitch、亲和性；用实际 H20 机器结果辨认路径 |
| [06_跨节点网络.md](<./02_Distributed_Communication_Memory/06_跨节点网络.md>) | A2→A7 | 分层必修 | 保留，区分入门与网络深入 | 节点间路径、NIC/RDMA/GPUDirect 必读；RoCE 调优和 SHARP 专项 |
| [07_NCCL软件栈.md](<./02_Distributed_Communication_Memory/07_NCCL软件栈.md>) | A2→A7 | 分层必修 | 保留，拆稳定原理与版本动态 | 通信组初始化、传输与 hang 排查；device-side 新能力参考 |
| [08_通信性能模型.md](<./02_Distributed_Communication_Memory/08_通信性能模型.md>) | A2 | 必修 | 保留，配消息尺寸扫描 | α–β、有效带宽、algbw/busbw；先手算，再解释实测曲线 |
| [09_通信计算重叠.md](<./02_Distributed_Communication_Memory/09_通信计算重叠.md>) | A2/A3 | 必修 | 保留，配依赖时间线 | 异步不等于重叠；用 Stream/Event 与 NCCL trace 验证 |
| [10_训练显存优化.md](<./02_Distributed_Communication_Memory/10_训练显存优化.md>) | A3 | 必修 | 保留，明确与并行篇边界 | 显存账本和优化取舍；FSDP 执行生命周期链接 A3 原理篇 |
| [11_推理内存系统.md](<./02_Distributed_Communication_Memory/11_推理内存系统.md>) | A4 | 必修 | 保留为推理内存原理入口 | KV 核算、paging、prefix reuse；调度细节接 SGLang 与论文 |
| [12_分布式Checkpoint与恢复.md](<./02_Distributed_Communication_Memory/12_分布式Checkpoint与恢复.md>) | A3/A7 | 必修 | 保留，配恢复实验 | 状态完整性、一致性切面、重分片；中断恢复不能只检查加载成功 |
| [13_案例_MoE-AllToAll.md](<./02_Distributed_Communication_Memory/13_案例_MoE-AllToAll.md>) | A8 | 专项 | 保留，接 MoE 实验 | 逐步走 dispatch→expert→combine；用真实 token 分布解释通信 |
| [14_案例_Prefill-Decode分离.md](<./02_Distributed_Communication_Memory/14_案例_Prefill-Decode分离.md>) | A4→A8 | 分层必修 | 保留，分原理与部署 | 先懂 KV 为什么跨节点搬、何时得不偿失；PD 实测后做 |
| [15_实验与排查手册.md](<./02_Distributed_Communication_Memory/15_实验与排查手册.md>) | A2/A7 | 必修 | 保留，接统一实验索引 | 拓扑→P2P→Collective→多机，分别记录显存与带宽证据 |

### 7.4 learn_docs/03_Parallelism（17 篇）

| 原文 | 映射模块 | 阅读等级 | 拟议动作 | 首读范围与后续用途 |
|---|---|---|---|---|
| [00_专题总览.md](<./03_Parallelism/00_专题总览.md>) | A3 | 导航 | 保留并重排入口 | 先看 DP/TP 的小例子，再用统一布局语言总结 |
| [01_统一切分语言.md](<./03_Parallelism/01_统一切分语言.md>) | A3 | 必修 | 保留，后移抽象介绍 | 先用两个矩阵解释 shard/replicate/partial，再进入 DeviceMesh API |
| [02_DP与DDP.md](<./03_Parallelism/02_DP与DDP.md>) | A3 | 必修 | 保留，配梯度对齐实验 | 梯度平均、bucket、累积/no_sync、不同 token 数的归一化 |
| [03_ZeRO与FSDP.md](<./03_Parallelism/03_ZeRO与FSDP.md>) | A3 | 必修 | 保留，配状态生命周期图 | ZeRO 三种状态与 FSDP gather/reshard；区分原理和 Megatron 实现 |
| [04_TensorParallelism.md](<./03_Parallelism/04_TensorParallelism.md>) | A3 | 必修 | 保留，配两卡 Linear 实验 | Column/Row Linear 的 local shape、forward/backward 通信 |
| [05_SequenceParallelism.md](<./03_Parallelism/05_SequenceParallelism.md>) | A3 | 必修 | 保留，分术语与实现 | Megatron SP 与其他 sequence 切分的区别；先跟 TP 相邻层 |
| [06_PipelineParallelism.md](<./03_Parallelism/06_PipelineParallelism.md>) | A3 | 必修 | 保留，补手画 microbatch 时序 | GPipe/1F1B、activation 生命周期与 bubble；Zero Bubble 二读 |
| [07_ContextParallelism.md](<./03_Parallelism/07_ContextParallelism.md>) | A3→A8 | 分层必修 | 保留，分入门与长序列专项 | 理解切序列、交换 KV 和负载不均；具体算法与大规模实测后读 |
| [08_ExpertParallelism.md](<./03_Parallelism/08_ExpertParallelism.md>) | A3→A8 | 分层必修 | 保留，分入门与 MoE 专项 | expert/token/rank 映射及 A2A 先学；ETP/folding 后读 |
| [09_Embedding与词表并行.md](<./03_Parallelism/09_Embedding与词表并行.md>) | A3/A5 | 必修 | 保留，接 RL 数值正确性 | 分布式 logits/logsumexp、词表切分与全局 token loss |
| [10_多维并行与DeviceMesh.md](<./03_Parallelism/10_多维并行与DeviceMesh.md>) | A3 | 必修 | 保留，补小网格再放大 | 先画 2×2 rank group，再看大配置；勿以缩写乘积替代布局 |
| [11_训练并行方案设计.md](<./03_Parallelism/11_训练并行方案设计.md>) | A3 | 必修 | 保留，改为配置实战入口 | 根据容量、batch 语义和拓扑推配置，形成可核对账本 |
| [12_推理并行方案设计.md](<./03_Parallelism/12_推理并行方案设计.md>) | A4 | 必修 | 保留，配 TP/副本对比 | 并行对容量、吞吐、延迟与 KV 的不同影响 |
| [13_RL角色级并行.md](<./03_Parallelism/13_RL角色级并行.md>) | A0→A5 | 分层必修 | 保留，前后两遍 | A0 看角色与两层并行；A5 深读共置、分离、异步与 freshness |
| [14_自动并行与代价模型.md](<./03_Parallelism/14_自动并行与代价模型.md>) | A8 | 专项 | 保留 | 理解代价模型和搜索约束，作为调度设计拓展 |
| [15_案例_DeepSeek与DualPipe.md](<./03_Parallelism/15_案例_DeepSeek与DualPipe.md>) | A8 | 专项 | 保留 | PP/EP 基础完成后对照 DualPipe；不直接照抄配置 |
| [16_实验与配置工作表.md](<./03_Parallelism/16_实验与配置工作表.md>) | A3/A4/A5 | 必修 | 保留，接统一实验索引 | 所有并行实验统一填布局、状态、通信、batch 与实际拓扑 |

### 7.5 learn_docs/04_Profiling_Performence_Analysis（17 篇）

| 原文 | 映射模块 | 阅读等级 | 拟议动作 | 首读范围与后续用途 |
|---|---|---|---|---|
| [00_专题总览.md](<./04_Profiling_Performence_Analysis/00_专题总览.md>) | A6 | 导航 | 保留，修路径命名计划 | 17 篇现已齐全；区分原理入口与性能教程操作入口 |
| [01_方法论与测量陷阱.md](<./04_Profiling_Performence_Analysis/01_方法论与测量陷阱.md>) | A1/A6 | 必修 | 保留为方法原理 | 测量边界、warmup、重复、干预与噪声；操作链接教程 |
| [02_GPU指标与遥测.md](<./04_Profiling_Performence_Analysis/02_GPU指标与遥测.md>) | A1/A6 | 必修 | 保留 | GPU/SM/DRAM 指标、采样与降频；不能只读 GPU-Util |
| [03_NsightSystems时间线.md](<./04_Profiling_Performence_Analysis/03_NsightSystems时间线.md>) | A1/A6 | 必修 | 保留 | CPU/GPU/通信关键路径；首次采集命令接软件教程 |
| [04_NsightCompute_Kernel分析.md](<./04_Profiling_Performence_Analysis/04_NsightCompute_Kernel分析.md>) | A6→A8 | 分层必修 | 保留 | 先会选热点与读报告；计数器重放和 kernel 深入后读 |
| [05_PyTorchProfiler.md](<./04_Profiling_Performence_Analysis/05_PyTorchProfiler.md>) | A1/A6 | 必修 | 保留 | operator、shape、stack、memory 对应关系；配首次 trace |
| [06_Roofline与瓶颈分类.md](<./04_Profiling_Performence_Analysis/06_Roofline与瓶颈分类.md>) | A1→A6 | 分层必修 | 保留 | 先懂算力/带宽限制，再补口径与其他瓶颈类别 |
| [07_CPU与数据加载.md](<./04_Profiling_Performence_Analysis/07_CPU与数据加载.md>) | A1/A6 | 必修 | 保留，补 Python 调度前置 | DataLoader、pin memory、H2D、CPU 空洞与隐式同步 |
| [08_显存与OOM.md](<./04_Profiling_Performence_Analysis/08_显存与OOM.md>) | A1/A6 | 必修 | 保留 | 显存峰值、引用保留、碎片、snapshot 与回归 |
| [09_分布式通信Profiling.md](<./04_Profiling_Performence_Analysis/09_分布式通信Profiling.md>) | A2/A6 | 必修 | 保留 | 先分到达晚、排队与真实传输慢，再选通信工具 |
| [10_并行策略与Bubble.md](<./04_Profiling_Performence_Analysis/10_并行策略与Bubble.md>) | A3/A6 | 必修 | 保留 | 把不同并行策略映射到 trace；需要先理解 A3 |
| [11_训练性能指标.md](<./04_Profiling_Performence_Analysis/11_训练性能指标.md>) | A3/A6 | 必修 | 保留 | step 边界、有效 token、MFU/HFU、扩展效率 |
| [12_推理性能分析.md](<./04_Profiling_Performence_Analysis/12_推理性能分析.md>) | A4/A6 | 必修 | 保留 | TTFT/TPOT、负载长度、开闭环与 goodput；配容量曲线 |
| [13_专项负载_MoE_长上下文_RL.md](<./04_Profiling_Performence_Analysis/13_专项负载_MoE_长上下文_RL.md>) | A5/A6/A8 | 分层必修 | 保留，新补齐 | RL 角色时间线与版本 lag 必读；MoE/长上下文跟专项 |
| [14_线上可观测性.md](<./04_Profiling_Performence_Analysis/14_线上可观测性.md>) | A7 | 必修 | 保留，新补齐 | 指标/日志/trace、基数与常驻采样，建立小型观测面板 |
| [15_案例_GPU利用率低.md](<./04_Profiling_Performence_Analysis/15_案例_GPU利用率低.md>) | A6 | 必修 | 保留，新补齐 | 从模糊现象到可证伪假设；在首份 trace 后做案例 |
| [16_实验与排查工作表.md](<./04_Profiling_Performence_Analysis/16_实验与排查工作表.md>) | A6/A7 | 必修 | 保留，新补齐 | 现场工作表保留；正式报告引用 performance_report_template |

### 7.6 learn_docs/model_arc（6 篇）

| 原文 | 映射模块 | 阅读等级 | 拟议动作 | 首读范围与后续用途 |
|---|---|---|---|---|
| [AI_Infra_Model_Architecture_Evolution_DeepSeek_Kimi_Qwen_GLM_2024_2026.md](<./model_arc/AI_Infra_Model_Architecture_Evolution_DeepSeek_Kimi_Qwen_GLM_2024_2026.md>) | A8 | 参考 | 保留，按架构索引 | 用于查模型差异和资料来源；具体型号/能力单独核对版本 |
| [MLA.md](<./model_arc/MLA.md>) | A4→A8 | 分层必修 | 保留长文，增加短入口 | 先读 MHA/GQA/MLA 的 KV 形状与容量影响；推导与实现二读 |
| [MoE-Infra.md](<./model_arc/MoE-Infra.md>) | A3→A8 | 分层必修 | 保留长文，按问题分段导航 | 首读 router→dispatch→GEMM→combine；通信/调优/模型案例专项回读 |
| [Sparse-Attention.md](<./model_arc/Sparse-Attention.md>) | A8 | 专项 | 保留 | 先学标准 attention 与 KV；再看稀疏结构对访问和负载的影响 |
| [Linear-Attention.md](<./model_arc/Linear-Attention.md>) | A8 | 专项 | 保留 | 递归状态与 KV 的差别、混合模型资源特征；数学长推导后读 |
| [Residual-Evolution.md](<./model_arc/Residual-Evolution.md>) | A8 | 参考 | 保留 | 训练稳定性、结构与执行依赖的专题参考，不作为并行入门前置 |

### 7.7 docs/code_walkthrough（16 篇）

| 原文 | 映射模块 | 阅读等级 | 拟议动作 | 首读范围与后续用途 |
|---|---|---|---|---|
| [README.md](<../docs/code_walkthrough/README.md>) | A0/A5 | 导航 | 保留为 slime 子入口 | 链接总学习路线；区分初次角色概览与完整源码阅读 |
| [00_rl_infra_survey.md](<../docs/code_walkthrough/00_rl_infra_survey.md>) | A0 | 分层必修 | 保留，拆入门与动态综述 | 首读工程问题地图；各框架近况与 Roadmap 为版本化参考 |
| [01_architecture_and_ray_orchestration.md](<../docs/code_walkthrough/01_architecture_and_ray_orchestration.md>) | A0→A5 | 分层必修 | 保留，补 Ray 前导 | 先读角色与同步主循环；placement/rank/rendezvous 在 A2、Ray 基础后读 |
| [02_rollout_sglang_server_mode.md](<../docs/code_walkthrough/02_rollout_sglang_server_mode.md>) | A5 | 必修 | 保留 | 读服务拓扑、生成循环、并发限制与 abort；前置 SGLang 请求基础 |
| [03_data_buffer_partial_rollout_async.md](<../docs/code_walkthrough/03_data_buffer_partial_rollout_async.md>) | A5 | 必修 | 保留，按状态机分段 | 先 Sample/buffer 不变量，再 partial/async/backpressure，串到算法语义 |
| [04_weight_sync_and_memory.md](<../docs/code_walkthrough/04_weight_sync_and_memory.md>) | A5 | 必修 | 保留，明确测量前提 | 布局转换、NCCL/IPC/disk、offload；“头号瓶颈”应按 workload 限定 |
| [05_rl_algorithms.md](<../docs/code_walkthrough/05_rl_algorithms.md>) | A5 | 必修 | 保留算法系统边界 | 已有算法可快读；logprob、mask、全局归一化、重要性修正重点读 |
| [06_megatron_backend_and_mbridge.md](<../docs/code_walkthrough/06_megatron_backend_and_mbridge.md>) | A5 | 必修 | 保留 slime 接入职责 | 训练 actor/provider/格式转换入口；Megatron 内部链接专门系列 |
| [07_customization_and_agentic.md](<../docs/code_walkthrough/07_customization_and_agentic.md>) | A5 | 必修 | 保留，分基础扩展与 Agentic | 先 hook/自定义 rollout，再多轮工具等待、轨迹对齐与失败 |
| [08_rm_hub_and_eval.md](<../docs/code_walkthrough/08_rm_hub_and_eval.md>) | A5 | 必修 | 保留 | reward/filter/eval 的顺序与样本组语义；算法背景不替代工程校验 |
| [09_engineering_observability.md](<../docs/code_walkthrough/09_engineering_observability.md>) | A5/A7 | 必修 | 保留框架专属部分 | 分离调试、对账、健康状态、CI；通用 profiler 操作引用性能教程 |
| [10_transferqueue.md](<../docs/code_walkthrough/10_transferqueue.md>) | A5→A8 | 专项 | 保留对比专题 | 主线 buffer 学完后比较独立数据平面；标明不是 slime 当前依赖 |
| [11_engine_internals_sglang.md](<../docs/code_walkthrough/11_engine_internals_sglang.md>) | A5 | 必修 | 保留跨框架接线图 | slime 调用→SGLang 端点→执行与释放；内部机制引用 SGLang 系列 |
| [12_megatron_lm_internals.md](<../docs/code_walkthrough/12_megatron_lm_internals.md>) | A3/A5 | 必修 | 保留接入摘要与调用链 | 回答 slime 如何进入 Megatron；重复引擎解释由 Megatron 系列承载 |
| [13_megatron_bridge_internals.md](<../docs/code_walkthrough/13_megatron_bridge_internals.md>) | A5 | 必修 | 保留独立转换专题 | HF↔Megatron、QKV/TP/EP 的往返校验；旧 Bridge 文件名暂保留 |
| [tau-bench_qwen3_4B.md](<../docs/code_walkthrough/tau-bench_qwen3_4B.md>) | A5/A7 | 专项 | 保留综合项目 | 基础闭环后再做多轮 Agent；不作为第一次跑 RL 的默认实验 |

### 7.8 docs/megatron_code_walkthrough（20 篇）

| 原文 | 映射模块 | 阅读等级 | 拟议动作 | 首读范围与后续用途 |
|---|---|---|---|---|
| [README.md](<../docs/megatron_code_walkthrough/README.md>) | A3 | 导航 | 保留子入口 | 挂接基础前置与 RL 训练后端路线，保留源码版本 |
| [01_foundations/00_ecosystem_install_and_data.md](<../docs/megatron_code_walkthrough/01_foundations/00_ecosystem_install_and_data.md>) | A3 | 必修 | 保留，补最小启动说明 | 环境、数据、第一次 step；明确预训练示例与 RL backend 的关系 |
| [01_foundations/01_learning_map.md](<../docs/megatron_code_walkthrough/01_foundations/01_learning_map.md>) | A3 | 导航 | 保留 | 只承担 Megatron 源码地图，基础通识链接统一入口 |
| [01_foundations/02_training_mainline.md](<../docs/megatron_code_walkthrough/01_foundations/02_training_mainline.md>) | A3 | 必修 | 保留，补一次 step 例子 | microbatch→forward/backward→optimizer；含 loss closure 与更新边界 |
| [01_foundations/03_model_and_transformer.md](<../docs/megatron_code_walkthrough/01_foundations/03_model_and_transformer.md>) | A3 | 必修 | 保留 | GPTModel/ModuleSpec/Transformer Engine、local shape 与模型构建 |
| [02_parallelism/00_strategy_and_topology.md](<../docs/megatron_code_walkthrough/02_parallelism/00_strategy_and_topology.md>) | A3 | 必修 | 保留实现约束与配置 | 通用选型链接并行原理，本篇讲 Megatron 的约束与拓扑落实 |
| [02_parallelism/01_parallel_state_and_tensor_parallel.md](<../docs/megatron_code_walkthrough/02_parallelism/01_parallel_state_and_tensor_parallel.md>) | A3 | 必修 | 保留 | rank group、TP/SP、词表并行，结合两卡张量核对 |
| [02_parallelism/02_pipeline_parallel.md](<../docs/megatron_code_walkthrough/02_parallelism/02_pipeline_parallel.md>) | A3 | 必修 | 保留 | 1F1B/交错/P2P、activation 生命周期，配 stage 时间线 |
| [03_state_and_memory/01_data_parallel_optimizer_checkpoint.md](<../docs/megatron_code_walkthrough/03_state_and_memory/01_data_parallel_optimizer_checkpoint.md>) | A3/A7 | 必修 | 保留 | DDP bucket、DistributedOptimizer、状态保存恢复；配中断验证 |
| [03_state_and_memory/02_megatron_fsdp.md](<../docs/megatron_code_walkthrough/03_state_and_memory/02_megatron_fsdp.md>) | A3→A8 | 分层必修 | 保留 | 先区分 DistributedOptimizer 与 FSDP；具体策略源码第二遍读 |
| [03_state_and_memory/03_memory_determinism_cuda_graph.md](<../docs/megatron_code_walkthrough/03_state_and_memory/03_memory_determinism_cuda_graph.md>) | A3/A6 | 必修 | 保留 | offload、确定性、CUDA Graph 的具体生命周期与限制 |
| [04_models_and_features/01_models_and_tokenizers.md](<../docs/megatron_code_walkthrough/04_models_and_features/01_models_and_tokenizers.md>) | A3/A5 | 必修 | 保留 | 模型/tokenizer/checkpoint 契约；模型目录只按任务查阅 |
| [04_models_and_features/02_moe_mla_mtp.md](<../docs/megatron_code_walkthrough/04_models_and_features/02_moe_mla_mtp.md>) | A8 | 专项 | 保留 | MoE/MLA/MTP 的具体模块与 router replay，承接模型基础 |
| [04_models_and_features/03_hybrid_multimodal_rl.md](<../docs/megatron_code_walkthrough/04_models_and_features/03_hybrid_multimodal_rl.md>) | A8 | 参考 | 保留并按主题导航 | RL 相关入口可提前；Hybrid/Mamba/多模态/Energon 后续按需 |
| [04_models_and_features/04_context_moe_precision_deep_dive.md](<../docs/megatron_code_walkthrough/04_models_and_features/04_context_moe_precision_deep_dive.md>) | A8 | 专项 | 保留 | CP/EP 组合与低精度深读；先有标准 Dense 训练基线 |
| [05_practice/01_performance_debugging.md](<../docs/megatron_code_walkthrough/05_practice/01_performance_debugging.md>) | A3/A6/A7 | 必修 | 保留框架故障树 | OOM/hang/straggler→源码与配置；通用采集操作接性能指南 |
| [05_practice/02_configuration_review.md](<../docs/megatron_code_walkthrough/05_practice/02_configuration_review.md>) | A3/A7 | 必修 | 保留为配置验收 | 每个 Megatron 实验填写，避免只保存启动命令 |
| [06_reference/01_official_docs_coverage.md](<../docs/megatron_code_walkthrough/06_reference/01_official_docs_coverage.md>) | A3 | 参考 | 保留 | 官方能力与本快照的覆盖边界，不要求逐项背诵 |
| [06_reference/02_glossary.md](<../docs/megatron_code_walkthrough/06_reference/02_glossary.md>) | A3 | 参考 | 保留局部词典 | 基础公共术语指向总词典，Megatron 专属术语保留 |
| [06_reference/03_source_questions.md](<../docs/megatron_code_walkthrough/06_reference/03_source_questions.md>) | A3/A5 | 分层必修 | 保留，分配到相关章节 | 先读 world size/batch/token mean；overflow/reshard 等在对应实验回读 |

### 7.9 docs/paper_read（13 篇）

| 原文 | 映射模块 | 阅读等级 | 拟议动作 | 首读范围与后续用途 |
|---|---|---|---|---|
| [00_共享基础_GPU与LLM推理硬件基础.md](<../docs/paper_read/00_共享基础_GPU与LLM推理硬件基础.md>) | A1/A4/A8 | 分层必修 | 保留全文，抽取为入门教材来源 | 首读 §1–14 GPU 与性能直觉；A4 读 §15–21 KV/分页；§26 高级硬件后读 |
| [01_FlashAttention论文详解.md](<../docs/paper_read/01_FlashAttention论文详解.md>) | A4→A8 | 分层必修 | 保留论文完整推导 | 先读 IO 与 tiling 的动机、精确 attention；online softmax 推导专项二读 |
| [02_PagedAttention-vLLM论文详解.md](<../docs/paper_read/02_PagedAttention-vLLM论文详解.md>) | A4 | 分层必修 | 保留，增加首读导航 | 逻辑块/物理块/浪费/COW 的例子必读；论文实验与实现差异第二遍 |
| [03_SGLang-RadixAttention论文详解.md](<../docs/paper_read/03_SGLang-RadixAttention论文详解.md>) | A4 | 分层必修 | 保留，增加首读导航 | prefix 复用、radix tree、锁/驱逐与调度；论文其他设计第二遍 |
| [04_FlashAttention-2论文详解.md](<../docs/paper_read/04_FlashAttention-2论文详解.md>) | A8 | 专项 | 保留 | FA1 基础后读非 matmul 开销、warp 分工与并行粒度 |
| [05_FlashAttention-3论文详解.md](<../docs/paper_read/05_FlashAttention-3论文详解.md>) | A8 | 专项 | 保留 | 异步搬运/计算、流水线与资源；实验能力以实际 H20 环境核验 |
| [06_Orca与Continuous-Batching论文详解.md](<../docs/paper_read/06_Orca与Continuous-Batching论文详解.md>) | A4 | 分层必修 | 保留，调度课程主读 | 静态 batch→iteration-level；区分论文设计与当前引擎实现 |
| [07_DistServe与Prefill-Decode分离论文详解.md](<../docs/paper_read/07_DistServe与Prefill-Decode分离论文详解.md>) | A4→A8 | 分层必修 | 保留 | 首读 prefill/decode 干扰与传 KV 成本；PD 部署与复现实验专项 |
| [08_TensorRT-LLM_Runtime架构详解.md](<../docs/paper_read/08_TensorRT-LLM_Runtime架构详解.md>) | A8 | 参考 | 保留为 Runtime 比较 | 移到逻辑上的系统架构比较栏，暂不改原路径 |
| [09_vLLM-V1_Runtime架构详解.md](<../docs/paper_read/09_vLLM-V1_Runtime架构详解.md>) | A4→A8 | 分层必修 | 保留为 Runtime 比较 | SGLang 主线后对照 token budget、KV 管理与 worker 边界 |
| [10_SGLang_Runtime架构详解.md](<../docs/paper_read/10_SGLang_Runtime架构详解.md>) | A4 | 必修 | 保留概念桥梁 | 在源码前建立进程/调度/执行全景；具体符号以源码系列为准 |
| [11_LLM推理Runtime技术演化总览.md](<../docs/paper_read/11_LLM推理Runtime技术演化总览.md>) | A0/A4 | 导航 | 保留技术地图 | 先看调度、KV、Kernel 三条线；不作为全部论文读完的要求 |
| [FlashInfer 论文详解.md](<../docs/paper_read/FlashInfer 论文详解.md>) | A4→A8 | 分层必修 | 保留长文，增加分段入口 | A4 看 backend 的职责与 KV layout；A8 深入 ragged/page/kernel 调度 |

### 7.10 docs/performance_analysis_guide（13 篇）

| 原文 | 映射模块 | 阅读等级 | 拟议动作 | 首读范围与后续用途 |
|---|---|---|---|---|
| [README.md](<../docs/performance_analysis_guide/README.md>) | A6 | 导航 | 保留统一实验入口 | 以学习阶段索引实验；与原理专题互链 |
| [00_concepts.md](<../docs/performance_analysis_guide/00_concepts.md>) | A1 | 必修 | 保留初学者教程 | 延迟/吞吐/关键路径/测量边界，先于专业 profiler |
| [01_baseline.md](<../docs/performance_analysis_guide/01_baseline.md>) | A1 | 必修 | 保留 | 固定环境、workload、重复统计；建立 H20 环境记录 |
| [02_pytorch.md](<../docs/performance_analysis_guide/02_pytorch.md>) | A1/A6 | 必修 | 保留手把手操作 | 从 CUDA 正确计时到首份 operator/kernel/memory trace |
| [03_cuda_kernels.md](<../docs/performance_analysis_guide/03_cuda_kernels.md>) | A1→A8 | 分层必修 | 保留，按工具层次拆阅读范围 | 先 nsys/NVTX；热点确定后 ncu；Triton/SASS 在 A8 |
| [04_megatron.md](<../docs/performance_analysis_guide/04_megatron.md>) | A3/A6 | 必修 | 保留训练实验操作 | step 分解、并行扫描；原理链接 A3，源码定位链接 Megatron |
| [05_inference.md](<../docs/performance_analysis_guide/05_inference.md>) | A4/A6 | 必修 | 保留推理实验操作 | 长度/并发/速率控制、容量曲线、KV 与延迟分布 |
| [06_slime.md](<../docs/performance_analysis_guide/06_slime.md>) | A5/A6 | 必修 | 保留 RL 综合实验 | 分离两侧、测换权/offload、观察角色关键路径与质量约束 |
| [07_labs.md](<../docs/performance_analysis_guide/07_labs.md>) | A1/A3/A4/A5/A8 | 必修 | 保留六实验并拆阶段入口 | 实验 1–2 在 A1，3 在 A8 或热点出现时，4/5/6 随训练/推理/RL |
| [08_toolbox_and_sources.md](<../docs/performance_analysis_guide/08_toolbox_and_sources.md>) | A6 | 参考 | 保留查阅索引 | 按问题找工具；资料清单不变成额外必读课程 |
| [glossary.md](<../docs/performance_analysis_guide/glossary.md>) | A1–A8 | 参考 | 保留为公共术语入口 | 首次读最先掌握的 30 词，后续随问题查，不先背完整词典 |
| [performance_report_template.md](<../docs/performance_analysis_guide/performance_report_template.md>) | A6/A7 | 必修 | 保留正式报告模板 | 实验工作表收原始记录，本模板组织可复现结论与回归 |
| [software_tutorials.md](<../docs/performance_analysis_guide/software_tutorials.md>) | A1/A6/A7 | 分层必修 | 保留工具操作手册 | 按首次使用取章节；先 nvidia-smi/Profiler/Perfetto/nsys，后 DCGM/ncu |

### 7.11 docs/sglang_code_walkthrough（28 篇）

| 原文 | 映射模块 | 阅读等级 | 拟议动作 | 首读范围与后续用途 |
|---|---|---|---|---|
| [README.md](<../docs/sglang_code_walkthrough/README.md>) | A4/A5 | 导航 | 保留子入口 | 把 Runtime 与 RL 路线接入总课程，保留源码快照边界 |
| [01_foundations/README.md](<../docs/sglang_code_walkthrough/01_foundations/README.md>) | A4 | 导航 | 保留 | 进程/请求基础入口，补 A1/A2 前置链接 |
| [01_foundations/01_learning_map.md](<../docs/sglang_code_walkthrough/01_foundations/01_learning_map.md>) | A4 | 必修 | 保留，配请求观察 | 训练与推理差异、指标、KV 核算；复杂模型术语按需回查 |
| [01_foundations/02_process_topology_and_request_path.md](<../docs/sglang_code_walkthrough/01_foundations/02_process_topology_and_request_path.md>) | A4→A5 | 分层必修 | 保留长文，增加普通请求首读路线 | 先普通文本请求的对象和跨进程流，再读异常、控制请求与特殊分支 |
| [02_runtime_core/README.md](<../docs/sglang_code_walkthrough/02_runtime_core/README.md>) | A4 | 导航 | 保留 | 按 scheduler→KV→runner→生成控制顺序 |
| [02_runtime_core/01_scheduler_and_batch.md](<../docs/sglang_code_walkthrough/02_runtime_core/01_scheduler_and_batch.md>) | A4 | 必修 | 保留调度主章 | waiting/running、token budget、prefill/decode、overlap 与请求状态 |
| [02_runtime_core/02_kv_cache_and_radix_attention.md](<../docs/sglang_code_walkthrough/02_runtime_core/02_kv_cache_and_radix_attention.md>) | A4 | 必修 | 保留内存实现主章 | pool、页映射、radix、lock_ref、evict/free；配 finish/abort 检查 |
| [02_runtime_core/03_model_runner_attention_cuda_graph.md](<../docs/sglang_code_walkthrough/02_runtime_core/03_model_runner_attention_cuda_graph.md>) | A4/A6 | 必修 | 保留 | ForwardBatch、backend、CUDA Graph 与运行 shape，连到热点分析 |
| [02_runtime_core/04_speculative_structured_sampling.md](<../docs/sglang_code_walkthrough/02_runtime_core/04_speculative_structured_sampling.md>) | A4→A8 | 分层必修 | 保留，按功能分层 | 采样正确性和约束输出先读；speculative 树与性能专项 |
| [03_scaling_and_deployment/README.md](<../docs/sglang_code_walkthrough/03_scaling_and_deployment/README.md>) | A4 | 导航 | 保留 | 单实例正确后再进入多副本、TP 与 PD |
| [03_scaling_and_deployment/01_distributed_and_pd_disaggregation.md](<../docs/sglang_code_walkthrough/03_scaling_and_deployment/01_distributed_and_pd_disaggregation.md>) | A4→A8 | 分层必修 | 保留，拆普通并行与 PD | TP/DP 基础必修；PD KV 交接、EP、多机复杂组合后读 |
| [03_scaling_and_deployment/02_deployment_topology_and_routing.md](<../docs/sglang_code_walkthrough/03_scaling_and_deployment/02_deployment_topology_and_routing.md>) | A4/A5 | 必修 | 保留 | 副本、路由、readiness、多节点启动，关联 RL placement |
| [03_scaling_and_deployment/03_hardware_platforms.md](<../docs/sglang_code_walkthrough/03_scaling_and_deployment/03_hardware_platforms.md>) | A1/A4 | 分层必修 | 保留 | 先核实实际环境/backend；其他硬件平台比较参考 |
| [04_interfaces_and_models/README.md](<../docs/sglang_code_walkthrough/04_interfaces_and_models/README.md>) | A4/A5 | 导航 | 保留 | 采样、模型支持与控制面按依赖串联 |
| [04_interfaces_and_models/01_api_request_and_sampling.md](<../docs/sglang_code_walkthrough/04_interfaces_and_models/01_api_request_and_sampling.md>) | A4/A5 | 必修 | 保留 | 请求规范化、sampling、tokenizer/template、logprob 与 RL 语义 |
| [04_interfaces_and_models/02_model_support_and_multimodal.md](<../docs/sglang_code_walkthrough/04_interfaces_and_models/02_model_support_and_multimodal.md>) | A4→A8 | 分层必修 | 保留，分文本基础与多模态 | 模型注册、加载、fallback 契约先读；多模态分支专项 |
| [04_interfaces_and_models/03_control_plane_and_post_training.md](<../docs/sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>) | A5 | 必修 | 保留控制面实现主章 | 在线换权、暂停/恢复、cache 失效、失败边界；不假设事务保证 |
| [05_production_engineering/README.md](<../docs/sglang_code_walkthrough/05_production_engineering/README.md>) | A6/A7 | 导航 | 保留 | 链接统一实验和 SGLang 专属运维/扩展 |
| [05_production_engineering/01_benchmark_profiling_observability.md](<../docs/sglang_code_walkthrough/05_production_engineering/01_benchmark_profiling_observability.md>) | A4/A6 | 必修 | 保留框架观测点 | 通用计时/工具安装接性能教程，保留 runtime 指标与因果链 |
| [05_production_engineering/02_extension_and_correctness.md](<../docs/sglang_code_walkthrough/05_production_engineering/02_extension_and_correctness.md>) | A7 | 必修 | 保留 | 模型/backend 契约、数值容差与回归，供首次代码改动使用 |
| [05_production_engineering/03_reliability_capacity_operations.md](<../docs/sglang_code_walkthrough/05_production_engineering/03_reliability_capacity_operations.md>) | A5/A7 | 必修 | 保留 | admission、超时/取消、资源回收与故障注入 |
| [06_diffusion/README.md](<../docs/sglang_code_walkthrough/06_diffusion/README.md>) | A8 | 参考 | 保留专题入口 | 作为扩展方向，不设为当前 LLM RL 主线前置 |
| [06_diffusion/01_pipeline_architecture_and_serving.md](<../docs/sglang_code_walkthrough/06_diffusion/01_pipeline_architecture_and_serving.md>) | A8 | 参考 | 保留 | Diffusion 请求和 pipeline 与 LLM 调度对比 |
| [06_diffusion/02_parallelism_cache_and_quantization.md](<../docs/sglang_code_walkthrough/06_diffusion/02_parallelism_cache_and_quantization.md>) | A8 | 参考 | 保留 | Diffusion 并行/cache/量化按相关任务查阅 |
| [06_diffusion/03_profiling_extension_and_correctness.md](<../docs/sglang_code_walkthrough/06_diffusion/03_profiling_extension_and_correctness.md>) | A8 | 参考 | 保留 | Diffusion profiling 与正确性验证，后续方向使用 |
| [07_reference/README.md](<../docs/sglang_code_walkthrough/07_reference/README.md>) | A4 | 参考 | 保留 | 官方覆盖与局部术语索引 |
| [07_reference/01_official_docs_coverage.md](<../docs/sglang_code_walkthrough/07_reference/01_official_docs_coverage.md>) | A4/A7 | 参考 | 保留 | 框架能力和当前源码版本核对入口 |
| [07_reference/02_glossary.md](<../docs/sglang_code_walkthrough/07_reference/02_glossary.md>) | A4 | 参考 | 保留局部词典 | 框架专属名词就近解释，公共术语链接总词典 |

## 8. 建议的入口结构与迁移顺序

拟议逻辑结构如下；这一轮只新增本清单，其他目录保持原状。

```text
learn_docs/README.md                   全库唯一课程入口
  路线与能力验收                       A0–A8、阅读范围、实验索引
  单机与GPU基础                        原文整编 + B01/B02/B03/B06
  分布式通信与并行                    现有 02/03 + B04
  训练系统                            导向 megatron_code_walkthrough
  推理系统                            导向 sglang_code_walkthrough
  RL框架与调度                        导向 code_walkthrough + B05/B07/B08/B09
  性能与实验                          原理专题 + performance_analysis_guide
  Kernel与模型专项                    现有 01 深入部分、model_arc、paper_read
  参考与历史笔记                      原文索引、版本依据、段落迁移账本
```

建议按以下次序实际改文档：

1. 建立总入口、单元级前置关系与实验索引，并给原文添加阅读等级和首读范围。
2. 先整编 A1/A2 的基础链：保留直觉与例子，补进程/张量/两卡通信的小台阶。
3. 打通 A3/A4 的“原理→源码→实验”；统一一组小模型、小张量和显存账本示例。
4. 打通 A5 的 Ray/队列/换权/异步主线；算法正确性贯穿每个优化实验。
5. 最后处理正文合并、长文拆分和目录命名，维护旧入口和链接。

后续每次拆分或合并必须记录：原路径与章节、新路径与章节、独特内容去向、链接验证结果。不能仅凭题目相同就删正文。

已发现的导航事项：

- Profiling 第 13—16 章现已齐全，此项不再列为内容缺失。
- 实际目录为 `04_Profiling_Performence_Analysis`，部分链接使用 `Performance`。后续统一拼写时应成批修改内部引用并检查兼容入口。
- 长笔记仍有 `AI_Infra_知识库/` 旧路径，需要修复。
- slime 的 06/13 文件名含旧 Bridge 名称，当前内容为内建转换。先保留路径并说明职责，避免为改名制造额外断链。
- 本轮不沿用正文中的“最新版本”或日期作为正确性证明。动态型号、版本号、API、源码 commit 和命令另做事实核对；稳定基础与动态资料分开维护。

## 9. 约 16 周、每周 20+ 小时的课程预算草案

这是控制课程规模的预算，不是就业结果承诺，也不是要求按日历跳过没懂的基础。采用约 320 小时的保守基数；额外时间用于复盘和环境问题。每周建议约 7 小时原理/源码、10 小时实践、3 小时整理；没有 GPU 实验的基础周可将实践用于小张量、CPU 进程或手画状态图。

| 周次 | 学习重心 | 阅读范围控制 | 交付物 |
|---|---|---|---|
| 1–3，约 60h | A0+A1，A6 入门 | 必要 GPU/PyTorch/进程基础；暂不全面学习所有 Kernel DSL | 环境记录、tensor/显存实验、正确计时、第一份 trace；可加一条简单生成请求观察 |
| 4–5，约 40h | A2 与 A3 的 DP/TP 入门 | 两卡实例先于 DeviceMesh 抽象；网络深度按实际环境扩展 | 拓扑图、Collective 正确性与带宽扫描、两卡梯度/Linear 对齐 |
| 6–8，约 60h | A3 训练执行与性能 | Dense step、状态、TP/PP/SP；CP/EP 先懂基本机制 | Megatron step 图、并行/显存账本、训练瓶颈分析、Checkpoint 恢复 |
| 9–11，约 60h | A4 推理执行与性能 | 普通请求、Scheduler、KV、ModelRunner；论文按首读范围 | 请求状态图、容量曲线、prefix/KV 实验、一次取消回收验证 |
| 12–14，约 60h | A5 框架与协同调度 | Ray、buffer、换权、共置/分离、partial/async 与正确性 | RL 一轮闭环、阶段耗时、布局转换核对、样本/版本账本 |
| 15–16，约 40h | A7 交付+A8 有界深入 | 基础 Kernel 小实验必做；大型专项只选一题 | 一个小 Kernel 报告、一项框架/调度或性能代码改动及验证报告 |

Kernel 小实验也可在第 3 周首次接触、最后两周深化，避免全部压到结尾。MoE、复杂 PD 和多种新模型的源码深读不同时塞入这 16 周；选一条与你的实测瓶颈有关的线继续深入。若主线基础未通过验收，缩减专项数量，保留基础练习。

## 10. H20 实验梯度与结果要求

当前只知道能使用多张 H20，不假定其节点数、NVLink/NIC/网络拓扑、软件版本、权限或运行时间额度。每个实验先记录实际环境；不把文中 H100 或其他机器的吞吐、带宽、Kernel 能力直接当作 H20 实测结论。本轮没有连接 GPU 或执行实验。

| 实验 | 资源梯度 | 连接模块 | 最小结果与正确性检查 |
|---|---|---|---|
| E01 正确计时与 tensor 生命周期 | 1 卡 | A1/A6 | warmup、同步、重复统计；解释 view/copy 与显存保留 |
| E02 小算子与 profiler | 1 卡 | A1/A8 | 合理 PyTorch/库基线、数值误差、shape 扫描、热点 trace |
| E03 Collective 与通信模型 | 2 卡起，随后扩到实际单节点可用卡数 | A2 | 输入输出核对；消息尺寸曲线；algbw/busbw 口径；记录拓扑 |
| E04 DP/TP 与梯度语义 | 2 卡起 | A3 | 小模型或 Linear 的单卡/多卡结果与梯度核对，确认 batch/token mean |
| E05 Megatron 训练配置 | 2/4 卡起，容量需要时扩展 | A3/A6 | 固定 workload，单变量比较 step、峰值显存与通信；区分强/弱扩展 |
| E06 SGLang 请求与容量 | 单实例起，再比较 TP 与多个副本 | A4/A6 | 固定输入输出长度/采样，统计吞吐、延迟分布、KV，验证完成/取消释放 |
| E07 RL 同步闭环与换权 | 能容纳选定小模型的最小配置 | A5 | 角色耗时、权重往返/版本核对、mask/logprob 与有效样本指标 |
| E08 共置、分离与异步 | 先具备 E07 基线，再增加角色资源池 | A5/A6 | 同资源预算比较或明确成本口径；记录队列、lag、长尾及质量指标 |
| E09 多节点与故障恢复 | 确认实际跨节点条件后 | A2/A7 | 单节点对照、到达偏斜与传输区分；进程中断恢复、样本进度与状态核对 |

先用足够小、可解释的负载建立基线，再扩大卡数与模型；卡多并不要求第一次实验就覆盖所有复杂并行。E08 的短程指标用于正确性和系统行为验证，不能仅凭短程 loss 或 tokens/s 推断长期训练质量提升。

## 11. 下一轮重构的验收标准

- 158 篇原文都有去向，独特例子与推导可追溯；不因“参考”标签删除内容。
- 从 A1 第一课到首个实验，不要求读者预先懂 rank、Stream、IPC 或 profiler。
- 每个课程单元包含：前置、首读范围、二读范围、源码入口、实验、完成标准。
- 公共概念有明确主入口，框架篇保留足够就地理解的短解释。
- 每个实践记录环境、workload、正确性、测量口径与原始证据。
- 稳定原理、论文设计、本仓库实现、工程建议的边界明确。
- 先验证目录与链接，再宣告迁移完成；章节标题相似不视为内容等价。

当前待讨论的主要设计点是基础讲解的密度与分段方式。建议先拿 A1 的“进程→PyTorch 运算→GPU 执行→正确计时”做一组样章，确认深度后再整编其余章节。
