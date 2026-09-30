# 从零读懂 FlashAttention：CUDA、论文与源码

这套文档面向**会读基本 Python、没有写过 CUDA**的读者。目标是在文档里看懂一个 attention 调用如何变成 GPU 上的协作计算：关键源码直接摘录，并逐段解释变量、地址、线程分工和执行前后的状态。矩阵乘法、Softmax、CUDA 线程组织和 C++ 模板都会在首次出现时解释。

源码来自本仓库 [`flash-attention/`](../../flash-attention/)，固定提交为 `e9cf2c1651d2303191eb40a739a3c135fda00999`。本系列以 NVIDIA 的 **FA2、固定长度、无 dropout 前向**为主线，再逐步引入训练和新硬件。代码中的 AMD、稀疏 attention、MLA 等分支属于后续专题。

## 怎么读

如果目标是先自己写出基础 kernel，再读本系列源码，先完成 [CUDA 渐进实践第 4–9 章](../cuda_tutorial/README.md)：Softmax → 普通 Attention → 在线 Attention → Tensor Core/CuTe → 异步流水。那里提供可修改的教学程序和验证练习，本系列继续负责真实 FA2 实现的逐段精读。

| 顺序 | 文档 | 读完能回答的问题 |
|---|---|---|
| 00 | [CUDA 入门：谁计算，数据放哪里](00_cuda_basics.md) | kernel、thread、warp、block、SM 是什么？`<<<...>>>` 如何读？ |
| 01 | [从普通 Attention 到在线 Softmax](01_algorithm_and_online_softmax.md) | 为什么可以分块？最大值改变后，旧结果怎样修正？ |
| 02 | [FA2 前向：逐层走到一个线程块](02_fa2_forward.md) | Python → C++ → CUDA 如何连接？一个 block 负责哪些输出？ |
| 03 | [读懂 CuTe 布局、线程分工与流水](03_layout_and_pipeline.md) | `gQ`、`sQ`、fragment 是什么？搬运和计算如何衔接？ |
| 04 | [反向、变长与 KV cache](04_backward_varlen_kvcache.md) | 为什么保存 LSE？不同长度的样本怎样索引？decode 为什么需要 split？ |
| 05 | [Hopper 与 CuTe DSL：FA3/FA4](05_hopper_and_cute.md) | producer/consumer、TMA、异步 MMA 在解决什么问题？ |
| 06 | [动手实验、验证与排错](06_experiments.md) | 没有 GPU 能验证什么？有 GPU 后如何检查结果和计时？ |

### 零经验读者分三遍走

只需先会 Python 的列表、循环、函数和索引；不要求预先会 CUDA、CuTe 或反向传播。文中的 `§` 表示章内小节，例如“01 §8”是第 01 章第 8 节。代码与公式中的下标从 0 开始。

| 阶段 | 建议顺序 | 到这里能做什么，就可以继续 |
|---|---|---|
| 第一遍：建立图景 | 00 → 01 §1–7 → 06 §1–2 → 02 §1 和 §6–12 | 手算 `49/15`；说明一个 block 如何扫描 K/V 并写出自己的 O |
| 第二遍：精读实现 | 02 全章 → 03 → 回读 01 §8 | 算出地址、线程坐标；解释为什么要等待、归约、修正旧输出 |
| 第三遍：按用途深入 | 训练读 04 §1–4、§10；推理读 04 §5–9、§11；最后读 05 | 理解梯度累加，或变长/缓存/split；再比较硬件流水 |

第一遍的 02 §6–12 中，可以先把 `copy` 理解为搬运、`gemm` 理解为矩阵乘、`softmax_rescale_o` 理解为更新 m/l/a，跟着正文状态表走。第二遍再展开线程布局和模板。01 §8 的四线程归约依赖 03 的布局知识，第一次不必卡在这里。

没有 NVIDIA GPU 也可以完成前两遍和 CPU 实验。各章自查题用于判断是否掌握主线；05 的架构细节属于进阶阅读。本系列的学习目标是理解这份 FlashAttention 实现；独立编写和调优 CUDA kernel 还需要编译、调试与性能分析的实践。

## 贯穿主线的两个例子

**数学例子**只有一行 Q、四个 K/V，用手算看清在线归一化。**工程例子**采用 `Q/K/V.shape = (2, 256, 4, 64)`、FP16、`causal=False`、`dropout_p=0`，在常规 FA2 前向分支中追踪 16 个线程块、每块 128 个线程。这两种尺寸分别服务于手算和真实 kernel 配置，不能把手算中的块大小直接当成 CUDA 配置。

01 §1.1 和 04 §3.1 另用两个 key 的小例子衔接前向与梯度，帮助第一次接触 attention 的读者入门。

文中代码分三种，并明确标注：

- **源码摘录**：来自固定提交，新增精读段落附原文件行号和永久链接。紧接着说明每段做什么、为什么需要、怎样代入具体数值。摘录可能只覆盖函数的一个分支，不作为独立程序运行。
- **教学伪代码**：去掉并行布局和预取，用串行步骤表达算法，不能直接拿来替换 kernel。
- **可运行示例**：在 [`examples/online_attention.py`](examples/online_attention.py)，只依赖 Python 标准库。

## 源码精读具体覆盖什么

各章先讲概念，再展开真实实现；源码链接供核对，主要执行逻辑已放在正文。

| 阅读位置 | 在文中完成的推导 |
|---|---|
| 01 §8 | `Softmax` 首轮初始化、旧最大值备份、`exp2`、四线程 shuffle、延迟分母归约、最终 LSE |
| 02 全章 | Python padding/scale → C++ 指针/stride/分发 → 一个 block 的两轮 K/V 扫描 → 输出写回 |
| 03 §10 | 从布局算出线程 37 搬哪些 Q、持有哪些 score；展开 mask、边界拷贝、GEMM helper 和 shared 复用 |
| 04 §10–11 | dO/O 点积、dP 原地变成 dS、dQ 原子累加和确定性缓冲区、分页地址、split 的带权合并 |
| 05 §8–9 | TMA acquire/wait/release、异步 MMA 的完成条件、FA4 旧 O 修正、pipeline 轮次与 JIT 编译/执行 |

每段源码尽量回答四件事：**进入时存什么 → 哪条语句改变什么 → 退出时存什么 → 后面谁会读取它**。先沿一个固定配置把这四件事连起来，再看可选分支。

## 论文与源码各提供什么

[FA1 论文](https://arxiv.org/abs/2205.14135)提供 IO 优化和重计算的出发点；[FA2 论文](https://arxiv.org/abs/2307.08691)进一步讨论减少非矩阵乘操作以及 CTA/warp 分工；[FA3 论文](https://arxiv.org/abs/2407.08608)讨论 Hopper 上的异步重叠与低精度。我们用当前源码逐项核对这些思路的落点。FA1 原始 kernel 与当前默认 FA2 实现有差别，FA4 章节则明确按当前实现讲解。

仓库还有更长的 [FA1](../paper_read/01_FlashAttention论文详解.md)、[FA2](../paper_read/04_FlashAttention-2论文详解.md)、[FA3](../paper_read/05_FlashAttention-3论文详解.md) 论文笔记；可以在读完本系列基础部分后回看。

## 三个容易混淆的入口

| 调用入口 | 主要源码 | 本系列位置 |
|---|---|---|
| `from flash_attn import flash_attn_func` | `flash_attn/flash_attn_interface.py`、`csrc/flash_attn/` | 02～04 的 FA2 主线 |
| `from flash_attn_3 import flash_attn_interface` | `hopper/` | 05 的 FA3 部分 |
| `from flash_attn.cute import flash_attn_func` | `flash_attn/cute/` | 05 的 FA4 部分 |

FA2 的 C++ 代码也使用 **CuTe** 布局库；FA4 的 **CuTe DSL** 是另一套用 Python 语法编写、编译 GPU kernel 的路径。看到 `cute` 这个词时先确认所在目录。

## 源码准备与版本

在 slime 仓库根目录执行：

```bash
git submodule update --init flash-attention
git -C flash-attention rev-parse HEAD
```

只读源码、运行 CPU 教学示例不需要安装 FlashAttention。编译 NVIDIA C++ 路径时，还需按[上游 README](../../flash-attention/README.md)准备受支持的 GPU、CUDA、PyTorch，并初始化 CUTLASS，例如 `git -C flash-attention submodule update --init csrc/cutlass`。AMD 依赖和 CuTe DSL 安装按各自说明处理。

这个子模块的存在不会自动切换 slime 的训练或推理 backend。升级它时，应重新核对文中的函数、tile 配置和接口约束；本系列里的具体数字均以以上提交为准。
