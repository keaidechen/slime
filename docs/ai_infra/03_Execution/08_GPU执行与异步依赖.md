# 第 8 章　GPU执行与异步依赖

区分进程、运行时和设备的职责，用依赖与生命周期解释异步提交、并发和正确释放。

本章内容：

- [1. 主机、设备与执行边界](#s1)
- [2. 进程、线程与四种异步：谁提交，谁执行，谁等待](#s2)
- [3. CUDA Stream、依赖与调度层次](#s3)

<a id="s1"></a>

<a id="s1-主机设备与执行边界"></a>

## 1. 主机、设备与执行边界


<a id="s1-从程序到设备工作"></a>

### 从程序到设备工作

应用进程运行在操作系统管理的地址空间中，通过框架、运行时和驱动提交设备工作。GPU 在自己的执行资源上处理 Kernel 和传输；CPU 与 GPU 的进度没有天然的一一对应关系。

例如 Python 执行 `y = x @ w` 后继续记录 CPU 时间，记录到的可能只是提交开销，此时 GPU 仍在做矩阵乘。随后若要把 y 中的数值复制到 CPU 打印，程序才需要等待相关计算完成，于是等待看起来发生在“打印”这一行。相反，一次只修改形状与步长的视图操作可能没有设备计算。理解运行时间时，应把 Python 调用、设备工作和结果读取分别放到时间线上。

<a id="s1-三种调度层次"></a>

### 三种调度层次

操作系统调度 CPU 线程；运行时安排请求、任务和设备提交；GPU 将线程块等工作映射到硬件资源。某层队列里有工作，不意味着下一层已经有可执行任务。

GPU 空闲可能源于 CPU 数据准备、解释器或调度开销，也可能源于等待通信、依赖或资源释放。仅增加 CUDA stream 数量不能解决尚未产生输入的问题。

<a id="s1-异步与并发"></a>

### 异步与并发

异步描述的是提交方是否必须等到工作完成。例如 CPU 把任务 A 交给 GPU 后继续准备任务 B，A 的提交就是相对于 CPU 异步的。并发描述多个任务在同一时间段内都有进展：一个线程交替处理两个请求，也可以是并发。并行则要求它们在某些时刻实际同时执行，例如 CPU 准备 B 的同时 GPU 正在算 A。因此一次异步提交并不能证明两个 GPU Kernel 同时运行；它们仍可能在设备队列中依次执行。

两个 stream 中的工作若依赖同一结果，仍需建立顺序；若都需要大量相同资源，也可能不能有效重叠。设备是否支持某类并发也应按具体平台确认。

<a id="s1-资源生命周期"></a>

### 资源生命周期

资源生命周期描述一份资源从创建、使用到允许回收的整个过程。例如 CPU 已经提交 GPU 对输入缓冲 X 的读取，此时主机函数返回，但 X 仍属于在途计算的输入；只有 GPU 完成最后一次读取后，存储才可以安全改作其他用途。

Context 保存设备执行上下文，stream 表达操作顺序，event 标记完成点，allocator 管理存储复用，通信组保存参与者关系。这些资源的结束时刻未必一致。请求取消时，用户可能已经不再等待结果，设备却仍在访问缓冲，因此取消处理也需要追踪在途访问并安排清理，不能仅删除请求对象。

<a id="s1-演进的问题链"></a>

### 演进的问题链

最直接的执行方式是“准备一批、计算一批、等待结果，再准备下一批”。它容易检查，但 GPU 计算时 CPU 可能空闲，CPU 准备时 GPU 又可能空闲。加入异步队列后，CPU 可以提前准备后续批次，从而利用这些空档；代价是必须记住每批输入是否仍被设备使用、结果是否已经可读。

当批次很多且每次工作很小时，逐项提交本身又可能成为成本，因此还会将多个操作预先组织成可重复提交的图。请求的到达、长度和取消却不断变化，运行时必须决定何时能复用固定执行方案，何时重新安排工作。后文的 stream、事件与请求状态机，分别用来解决这些顺序和完成追踪问题。

相关：[进程线程与异步](<#s2>)、[Stream 依赖](<#s3>)、[编译与图提交](<10_编译器与运行时.md#s1>)。

参考：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)、[OSTEP Processes](https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-intro.pdf)。

---

<a id="s2"></a>

<a id="s2-进程线程与四种异步谁提交谁执行谁等待"></a>

## 2. 进程、线程与四种异步：谁提交，谁执行，谁等待

> 前置：Python 函数、运行环境。目标：遇到“异步”能说出具体执行者与完成条件。


<a id="s2-用餐厅理解边界再回到工程对象"></a>

### 用餐厅理解边界，再回到工程对象

进程像拥有独立账本的门店；线程像门店内共享账本的工作人员。协程是单个执行线程上的可暂停任务：遇到可等待操作就让其他任务前进。这个类比只用于区分状态与调度，不能据此推断 CPU 或 GPU 的实际并行度。

| 对象 | 状态与执行 | RL 中的例子 |
|---|---|---|
| OS process | 独立地址空间，跨进程数据需显式传递或共享映射 | trainer rank、SGLang worker |
| thread | 同进程共享地址空间，需要处理并发访问 | tokenizer 或服务工作线程 |
| coroutine | 在事件循环中合作调度 | 等待多个 HTTP rollout 响应 |
| GPU stream | GPU 操作的有序提交队列 | GEMM、Memcpy、NCCL 工作 |

在常见启用 GIL 的 CPython 中，多个线程不会自动使纯 Python CPU 计算并行；原生库可能释放 GIL。协程适合隐藏可等待 I/O，不能靠 `async def` 自动把重计算移到后台。


<a id="s2-四种异步不能互相替代"></a>

### 四种异步不能互相替代

| 层次 | 提交得到什么 | 谁执行 | 什么时候能消费结果 |
|---|---|---|---|
| asyncio | coroutine/task/future | 事件循环推进，底层可能有网络/线程 | await 完成且处理异常 |
| Ray | ObjectRef | 远程 worker/actor | 依赖就绪或 ray.get 返回 |
| CUDA | host 调用返回，工作可能仍在队列 | GPU | 相关 stream/event 的依赖满足 |
| 分布式 collective | Work 或已提交的设备操作 | 通信参与者与设备 | 遵守 API 的 wait/stream 完成语义 |

`await HTTP` 完成不代表服务端为下一次权重更新完成了所有跨 rank 工作；CPU 不阻塞也不等于 GPU 已有两个 kernel 同时执行。


<a id="s2-协程的提交与完成顺序"></a>

### 协程的提交与完成顺序

```python
import asyncio
import time

async def request(i):
    await asyncio.sleep(0.1)  # 模拟可等待 I/O，不模拟 GPU 计算
    return i

async def main():
    begin = time.perf_counter()
    values = await asyncio.gather(*(request(i) for i in range(4)))
    print(values, "elapsed:", time.perf_counter() - begin)

asyncio.run(main())
```

四个等待可以重叠，总时间通常接近一次等待加调度开销。将函数内的 `await asyncio.sleep` 换成阻塞的 `time.sleep`，事件循环就无法在这段时间推进其他协程。这解释了为什么服务端某个同步预处理步骤可能拖慢所有请求。


<a id="s2-并发不自动限制队列"></a>

### 并发不自动限制队列

一次提交一百万个任务，可能只是把一百万份元数据和未来结果放进队列。要限制的是 **in-flight 工作数量或字节数**，不只是 worker 数。

policy lag（策略滞后）描述生成样本使用的策略与消费样本时的训练策略之间的差距，可按版本、时间或分布差异计量。版本号差相同，不保证概率分布变化相同。

例如生成侧每秒完成 12 个样本，训练侧每秒消耗 10 个；持续运行时积压每秒增加 2 个。即便内存还足够，样本等待更久也可能增加 policy lag。背压让生产方在容量接近上限时等待、减少投递或进行有规则的拒绝。


<a id="s2-谁负责结束与清理"></a>

### 谁负责结束与清理

把一条请求写成状态机：

```text
created → queued → running → finished
             └→ canceled      └→ failed / canceled
```

每条边都要问：响应 future 是否完成，队列是否移除，GPU/KV 是否仍被引用，统计是否重复计数。客户端停止等待，不一定意味着服务端停止计算；需要显式取消协议与资源释放确认。


<a id="s2-从基础连接到源码"></a>

### 从基础连接到源码

- 进程/rank/device 的对应：[进程与地址空间](<../appendices/memory/进程设备与地址空间.md>)。
- GPU 操作依赖：[CUDA Stream 与依赖](<#s3>)。
- 远程任务与背压：[Ray 与队列调度](<../06_Post_Training/27_权重发布异步与资源调度.md#s4>)。
- 普通请求到取消的路径：[SGLang 请求链](<../appendices/source/sglang/01_foundations/02_process_topology_and_request_path.md>)。

验收：为一次 rollout 画出 CPU 进程、HTTP 等待、GPU 执行和结果返回四条泳道；标明每个“异步”的提交点与完成点。不要把四条泳道画成一条同步调用栈。

---

<a id="s3"></a>

<a id="s3-cuda-stream依赖与调度层次"></a>

## 3. CUDA Stream、依赖与调度层次

> 前置：GPU 执行模型。目标：区分 host 提交、Stream 操作顺序、Block 驻留和 Warp 发射。


<a id="s3-从提交一次计算到拿到结果"></a>

### 从“提交一次计算”到“拿到结果”

上一章的 Kernel 是 GPU 要做的工作。CPU 端称为 **Host（主机端）**，GPU 端称为 **Device（设备端）**。CPU 发起 Kernel Launch（启动/提交）后，通常可以继续执行下一行代码，此时 GPU 可能还没开始，也可能正在计算。这叫相对于 Host 的**异步执行**：函数返回不代表结果已就绪。

**Stream（流）**表达一系列 GPU 操作的先后顺序。操作既包括 Kernel，也包括 Memcpy（内存复制）与 Event（事件标记）。最简单的同流流程是：

```text
复制输入到 GPU → Kernel 计算 → 复制输出到 CPU → CPU 等待完成 → 读取输出
```

同一个 Stream 里的常规操作按序执行；CPU 要读取异步复制的结果，必须先确认对应操作已经完成。Host 缓冲区是否使用 pinned memory（页锁定内存，复制期间保持物理页稳定）等条件会影响复制能否真正异步，不能仅凭接口名判断性能。

<a id="s3-两条-stream-怎样表达依赖"></a>

### 两条 Stream 怎样表达依赖？

假设 A 写入张量 X，B 要读取 X，C 与两者独立。把 A、B 放入不同 Stream 并不会自动建立依赖；应该先在 A 后记录事件 E，再让 B 所在的 Stream 等待 E：

```text
Stream 0：A 写 X（3 ms）→ 记录事件 E
Stream 1：等待事件 E → B 读 X（2 ms）
Stream 2：C（4 ms）
```

这里“记录事件”是在队列中放一个完成标记，不是让 CPU 等到 A 完成；“Stream 等事件”约束该 Stream 后续操作，也不要求 CPU 阻塞。CPU 的 synchronize（同步等待）则用于确实需要读取结果的地方。[CUDA 异步执行文档](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)给出了这几种机制的接口。

依赖链 A→B 至少要 5 ms；若资源足够且忽略提交开销，C 可在这 5 ms 内完成。三者总耗时的理想下界为 $\max(3+2,4)=5$ ms，而不是把 B 提前执行得到错误结果。若 A 与 C 争用同一资源，实际时间会更长。

**正确性还包括数据的生命周期**：在 GPU 完成读取前，不得覆盖或回收其输入缓冲区。跨 Stream 传递张量时，需要同时保证生产者先完成和内存仍然有效，框架可能提供额外的内存管理接口。

### 异步错误为什么常在下一行暴露

假设 Kernel A 写越界，随后主机提交 B，最后把一个统计量复制回 CPU。由于提交是异步的，异常可能直到复制或同步时才被观察到；栈顶出现的统计代码未必是最初出错的位置。若只删掉这一行，故障通常只是移到下一个同步点。

排查时保留最早的设备错误和输入形状，在小复现中逐段加入同步，缩小“最后一次成功同步”到“第一次失败同步”的区间。PyTorch 的 `CUDA_LAUNCH_BLOCKING=1` 可帮助定位提交点，但会改变执行时序与性能，不能用这个模式测正常吞吐。[CUDA 异步语义](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)解释了错误延迟报告的原因。

另一个更隐蔽的失败来自内存复用。Stream A 创建张量，Stream B 读取它；即使 B 已等待 A 的生产完成，Python 提前丢掉最后一个引用后，分配器仍需知道 B 尚未读完。PyTorch 的 [record_stream](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)用于登记这种使用关系；它不替代生产者到消费者的执行依赖。这里需要同时画两条边：数据就绪后才能读，最后一次读完后才能复用存储。

全局同步可能暂时掩盖此类竞争，却付出串行执行的成本。最终修复应表达必要的依赖与生命周期，再移除诊断用同步进行验证。

<a id="s3-概念辨析"></a>

### 概念辨析

LSU 是内存读写单元，TMA 是部分 GPU 的张量搬运加速器；Runtime/Driver 指负责提交工作的运行时软件和驱动。

下面把同一次工作继续向硬件内部展开。首次阅读只需掌握“操作依赖 → Block 驻留 → Warp 发射”三层；Ready/Eligible 表示具备发射条件，Issue 表示向执行单元发出指令，Resident 表示已经驻留。


<a id="s3-cuda-streamblock-调度与-warp-scheduler三个调度为什么看起来相似却完全不是一层"></a>

#### CUDA Stream、Block 调度与 Warp Scheduler：三个“调度”为什么看起来相似，却完全不是一层？

这一章专门解决一个很容易混淆的问题：**CUDA Stream 和 Warp Scheduler 都像是在决定“接下来执行谁”，但它们实际上相差了多个层级。** 要把它们分清，最有效的方法不是背定义，而是沿着一次 GPU 工作从 Host 到 SM 的路径往下看。

<a id="s3-先给结论调度至少分三层"></a>

##### 先给结论：调度至少分三层

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

<a id="s3-第一层cuda-stream软件可见的-gpu-工作队列"></a>

##### 第一层｜CUDA Stream：软件可见的 GPU 工作队列

CUDA Stream 可以理解为一条**有序的 GPU operation 队列**。进入同一个 Stream 的操作具有顺序语义；不同 Stream 中、彼此没有显式依赖、也没有默认 Stream 等隐式依赖的工作，则有机会在 GPU 上重叠。

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

但 `A/B/C` 与 `D/E` 之间在没有显式和隐式依赖、且资源允许时，就可能重叠。

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

<a id="s3-第二层block--cta-调度哪些工作真正进入-sm"></a>

##### 第二层｜Block / CTA 调度：哪些工作真正进入 SM？

一个 Kernel 被 launch 之后，会产生一个 Grid；Grid 中的 Block/CTA 并不会一次性全部进入 GPU。GPU 需要根据每个 SM 当前剩余的：

- Register File；
- Shared Memory；
- resident warp/thread slots；
- Block slots；
- 以及其他架构资源；

把尚未执行的 Block 动态分配给可容纳它们的 SM。

因此：

> **Stream 的“Kernel 已经可以运行”不等于 Kernel 的所有 Block 已经驻留；Kernel 已经被 dispatch 也不等于所有 Warp 都正在执行。**

<a id="s3-第三层warp-schedulersm-内的硬件指令调度"></a>

##### 第三层｜Warp Scheduler：SM 内的硬件指令调度

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

<a id="s3-cuda-stream-vs-warp-scheduler一张表彻底区分"></a>

##### CUDA Stream vs Warp Scheduler：一张表彻底区分

| 维度 | CUDA Stream | Warp Scheduler |
|---|---|---|
| 所属层级 | CUDA 软件执行/运行时模型 | SM 内硬件微架构 |
| 调度对象 | Kernel、Memcpy、Event 等 GPU operation | Warp 的下一条 instruction |
| 主要控制者 | 程序 + CUDA Runtime/Driver | GPU 硬件 |
| 粒度 | Kernel/operation 级，较粗 | Warp/instruction issue 级，极细 |
| 时间尺度 | 常见是 μs～ms 级工作 | GPU cycle / pipeline issue 级 |
| 用户能否直接创建/选择 | 可以显式创建 Stream、Event | 不能直接指定“下一 cycle 运行 Warp X” |
| 主要目标 | 表达依赖、异步与并发机会 | 隐藏 latency、持续喂饱执行单元 |

<a id="s3-为什么二者感觉很像"></a>

##### 为什么二者“感觉很像”？

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

<a id="s3-把完整执行链再画一次"></a>

##### 把完整执行链再画一次

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

> **记忆：Stream 管操作依赖；CTA Scheduler 管 Block→SM；Warp Scheduler 管 SM 内 Warp→Instruction。**

---

## 本章回顾

异步提交只改变主机等待方式，不自动建立设备间依赖。正确执行同时要求消费者等待数据就绪，以及缓冲在最后一次访问完成前保持有效。

**检查理解：** 两个 Stream 分别写入和读取同一缓冲时，执行依赖和内存存活要分别保证什么？

下一章讨论矩阵乘与Attention算子。将复用和依赖落实为矩阵乘分块、Attention 在线归约及专家算子调度，解释优化收益与边界。

<!-- book-nav -->

[上一章：硬件与存储层次](<07_硬件与存储层次.md>) · [全书目录](<../README.md>) · [下一章：矩阵乘与Attention算子](<09_矩阵乘与Attention算子.md>)

<!-- /book-nav -->
