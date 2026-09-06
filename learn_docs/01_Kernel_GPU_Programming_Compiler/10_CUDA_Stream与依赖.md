# CUDA Stream、依赖与调度层次

<!-- learning-position -->
> **学习定位**：A1/A2 · 必修。
> **前置**：[基础课程](<../00_Foundations/README.md>)。
> **首读/二读**：区分 Stream、Block 与 Warp 的调度；画 producer/consumer 的 Event 依赖，再读高级例子。
> **进度与实验**：[学习清单](<../学习清单.md>) · [总入口](<../README.md>)。
<!-- /learning-position -->

> 前置：GPU 执行模型。目标：区分 host 提交、Stream 操作顺序、Block 驻留和 Warp 发射。首读各层职责，再跟一对 producer/consumer 操作画依赖。


<a id="notebook-06"></a>

## 学习问答：原笔记 §06

### 06｜CUDA Stream、Block 调度与 Warp Scheduler：三个“调度”为什么看起来相似，却完全不是一层？

这一章专门解决一个很容易混淆的问题：**CUDA Stream 和 Warp Scheduler 都像是在决定“接下来执行谁”，但它们实际上相差了多个层级。** 要把它们分清，最有效的方法不是背定义，而是沿着一次 GPU 工作从 Host 到 SM 的路径往下看。

#### 先给结论：调度至少分三层

```text
CPU / Framework
│
│ enqueue Kernel / Memcpy / Event
▼
CUDA Stream
│  表达 operation 的顺序、依赖与并发机会
▼
GPU Front End / CTA(Block) Scheduling
│  把可以运行的 Block 分配到资源足够的 SM
▼
SM
│
├─ Resident Warp 0
├─ Resident Warp 1
├─ Resident Warp 2
└─ ...
      │
      ▼
Warp Scheduler
│  从 Ready / Eligible Warp 中选择
│  并 Issue 下一条指令
▼
CUDA Core / Tensor Core / LSU / TMA / ...
```

所以可以用一句话区分：

> **CUDA Stream 决定“哪些 GPU operation 在依赖上允许先后或并发”；CTA/Block 调度决定“哪个 Block 进入哪个 SM”；Warp Scheduler 决定“已经驻留在这个 SM 上的哪个 Warp 此刻发射下一条指令”。**

#### 第一层｜CUDA Stream：软件可见的 GPU 工作队列

CUDA Stream 可以理解为一条**有序的 GPU operation 队列**。进入同一个 Stream 的操作具有顺序语义；不同 Stream 中、彼此没有显式依赖的工作，则有机会在 GPU 上重叠。

```text
Stream 0:
Kernel A
   ↓
Kernel B
   ↓
Memcpy C

Stream 1:
Kernel D
   ↓
Kernel E
```

这里明确保证的是：

```text
A → B → C
D → E
```

但 `A/B/C` 与 `D/E` 之间如果没有 event / dependency，就可能重叠。

**注意：Multi-stream ≠ 一定并行。** Stream 只是告诉运行时与 GPU：“这些工作在依赖关系上可以并发。”如果 Kernel A 已经吃满 SM、Register、Shared Memory、Tensor Core 等关键资源，Kernel D 即使在另一条 Stream 中，也可能无法真正 concurrent execution。

```text
资源有余量：
Stream 0: [====== Kernel A ======]
Stream 1:    [=== Kernel D ===]
                 ↑ 可能重叠

资源已占满：
Stream 0: [========== Kernel A ==========]
Stream 1:                              [Kernel D]
```

这也解释了为什么 CUDA Stream 常用来做：

- Compute ↔ Memcpy overlap；
- 多个较小 Kernel 的 concurrent execution；
- 多个独立 GEMM 的 multi-stream 并行；
- Pipeline / communication-compute overlap。

#### 第二层｜Block / CTA 调度：哪些工作真正进入 SM？

一个 Kernel 被 launch 之后，会产生一个 Grid；Grid 中的 Block/CTA 并不会一次性全部进入 GPU。GPU 需要根据每个 SM 当前剩余的：

- Register File；
- Shared Memory；
- resident warp/thread slots；
- Block slots；
- 以及其他架构资源；

把尚未执行的 Block 动态分配给可容纳它们的 SM。

因此：

> **Stream 的“Kernel 已经可以运行”不等于 Kernel 的所有 Block 已经驻留；Kernel 已经被 dispatch 也不等于所有 Warp 都正在执行。**

#### 第三层｜Warp Scheduler：SM 内的硬件指令调度

当 Block 已经驻留某个 SM 后，硬件把线程按 32 个组织为 Warp。此时 SM 内可能同时驻留很多 Warp：

```text
SM
├─ Warp 0：等待 Global Memory
├─ Warp 1：Ready
├─ Warp 2：等待前序 Tensor Core 结果
├─ Warp 3：Ready
├─ Warp 4：等待 Barrier
└─ Warp 5：Ready
```

Warp Scheduler 会在 instruction issue 时刻从 Eligible Warp 中选择一个或多个 Warp，将其下一条指令发往对应执行管线。

```text
cycle N:
Warp 0  blocked
Warp 1  eligible  ┐
Warp 2  blocked   │
Warp 3  eligible  ├─→ scheduler 选择并 issue
Warp 5  eligible  ┘
```

这就是 GPU **latency hiding（延迟隐藏）**的核心：某个 Warp 在等 HBM、Tensor Core 结果或同步时，不让整个 SM 一起干等，而是 issue 其他已经 ready 的 Warp。

#### CUDA Stream vs Warp Scheduler：一张表彻底区分

| 维度 | CUDA Stream | Warp Scheduler |
|---|---|---|
| 所属层级 | CUDA 软件执行/运行时模型 | SM 内硬件微架构 |
| 调度对象 | Kernel、Memcpy、Event 等 GPU operation | Warp 的下一条 instruction |
| 主要控制者 | 程序 + CUDA Runtime/Driver | GPU 硬件 |
| 粒度 | Kernel/operation 级，较粗 | Warp/instruction issue 级，极细 |
| 时间尺度 | 常见是 μs～ms 级工作 | GPU cycle / pipeline issue 级 |
| 用户能否直接创建/选择 | 可以显式创建 Stream、Event | 不能直接指定“下一 cycle 运行 Warp X” |
| 主要目标 | 表达依赖、异步与并发机会 | 隐藏 latency、持续喂饱执行单元 |

#### 为什么二者“感觉很像”？

因为它们都在利用一个共同思想：

> **当任务 A 暂时无法继续时，看看是否有与它无依赖的任务 B 可以先做。**

但二者处理的对象完全不同：

```text
Stream 层：
Kernel A 还在跑，Kernel B 是否允许并发？

Warp Scheduler 层：
Warp 0 在等内存，Warp 3 是否 ready，可以 issue 下一条指令？
```

这是一个很重要的系统性认识：**AI Infra 里很多“调度”概念看起来相似，区分它们首先要问“调度对象的粒度是什么”。**

#### 把完整执行链再画一次

```text
PyTorch / CUDA C++ / Triton / TileLang
│
│ 产生或选择 GPU Kernel
▼
CUDA Runtime / Driver
│
▼
CUDA Stream
│ operation dependency / concurrency
▼
Kernel Launch
│
▼
Grid
│
▼
Block / CTA Scheduler
│
▼
SM
│
▼
Resident Warps
│
▼
Warp Scheduler
│
▼
Issue Instruction
│
├─ CUDA Core
├─ Tensor Core
├─ LSU
└─ TMA / other pipelines
```

> **记忆：Stream 管 Kernel；CTA Scheduler 管 Block→SM；Warp Scheduler 管 SM 内 Warp→Instruction。**

---
