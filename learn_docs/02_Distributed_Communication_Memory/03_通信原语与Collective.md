# 通信原语与 Collective

<!-- learning-position -->
> **学习定位**：A2 · 必修。
> **前置**：[两卡通信](<../00_Foundations/06_两卡通信与torchrun.md>)。
> **首读/二读**：逐个列输入输出与 rank，演示 AllReduce/AllGather/ReduceScatter/AllToAll。
> **进度与实验**：[学习清单](<../学习清单.md>) · [总入口](<../README.md>)。
<!-- /learning-position -->

Collective Communication（集合通信）是 communicator 中一组 rank 共同参与的数据变换。理解它们最有效的方法不是背名字，而是画出**输入分布 → 输出分布**。

## 核心原语

假设 4 个 rank，每个 rank 上 `Xi` 大小均为 $N$ bytes：

| 原语 | 输出语义 | 大模型中的典型用途 |
|---|---|---|
| Broadcast | root 的 X 复制到所有 rank | 参数/配置发布 |
| Reduce | 所有 X 规约到 root | 聚合统计量 |
| AllReduce | 所有 X 规约，结果给所有 rank | DP 梯度同步、TP partial sum |
| AllGather | 每 rank 的 shard 拼接到所有 rank | FSDP 参数展开、sequence shard 恢复 |
| ReduceScatter | 规约后按 shard 分发 | ZeRO/FSDP 梯度分片、TP |
| All-to-All | 每 rank 给每个 rank 一个不同 shard | MoE token dispatch、某些 CP/SP 转置 |
| Gather/Scatter | 多对一拼接 / 一对多拆分 | 控制面、小规模汇集 |
| Send/Recv | point-to-point（P2P，点对点） | pipeline activation、邻居交换 |

NCCL 的 communicator 需要所有相关 rank 以匹配的顺序调用 collective；某个 rank 漏调或 tensor count 不一致，常表现为 hang，而非立即抛出易懂异常。

## AllReduce = ReduceScatter + AllGather

逻辑上可把 AllReduce 分解为：

```mermaid
flowchart LR
    I["每 rank：完整输入"] --> RS["ReduceScatter：规约并分片"]
    RS --> S["每 rank：1/N 规约 shard"]
    S --> AG["AllGather：收集全部 shard"]
    AG --> O["每 rank：完整规约结果"]
```

这个分解非常重要：如果后续计算只需要自己那一片，就不应马上 AllGather。ZeRO/FSDP 正是通过延迟或按 layer 执行 gather，把“永久复制”变成“按需暂存”。

## All-to-All 不是 AllGather

设 rank 0 的输入被切为 `[x00,x01,x02,x03]`，其中第二个下标表示目标 rank。All-to-All 后 rank 1 获得 `[x01,x11,x21,x31]`。每个数据只送到其目标，而不是复制给所有人。

MoE 中目标由 router 动态决定，所以发送量可能不均匀。即使总字节相同，热点 rank、变长 split、pack/unpack、排序、跨节点比例都会使 All-to-All 比规则 AllGather 更难优化。

## In-place、异步与完成语义

- in-place 只描述输入输出 buffer 复用，不等于零拷贝或零 workspace。
- `async_op=True` 或把 NCCL 操作 enqueue 到 CUDA stream，只说明 host 不阻塞；消费者仍必须通过 stream dependency/event 等待数据完成。
- collective 的“完成”可分为 host 已提交、device 已完成、远端可见、所有 rank 都安全复用 buffer。不同 API 的契约必须查文档。

## one-sided 与 two-sided

传统 send/recv 和 collective 通常是 two-sided：发送与接收双方都参与匹配。Remote Memory Access（RMA，远程内存访问）允许一方对已注册远端窗口执行 put/get，并用 signal/wait 同步。one-sided 能简化不规则通信，但内存注册、可见性、ordering 和生命周期责任更显式。

PyTorch 2.14 的实验性 `nccl2` backend 引入 one-sided window 能力；这属于快速演进接口，不能把 release blog 当稳定 API 文档。

## 选择原语的三个问题

1. 结果究竟要复制到所有 rank，还是只保留 shard？
2. 发送模式是规则等长，还是 token-driven 变长？
3. 下一算子能否直接消费分片布局，从而消除一次通信？

第三问通常收益最大。优化通信的最高境界往往不是把 collective 加速 10%，而是通过布局和算子融合删掉它。

## 资料

- [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [PyTorch Distributed collectives](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch 2.14 Release Blog](https://pytorch.org/blog/pytorch-2-14-release-blog/)



<a id="notebook-09"></a>

## 学习问答：原笔记 §09

### 09｜从操作系统 IPC 到 GPU 通信原语：今天 NCCL / RDMA 的思想是怎么演化出来的？

问题：GPU 通信看起来有 Send/Recv、AllReduce、共享显存、P2P、RDMA、GPUDirect 等很多概念。它们与传统操作系统进程间通信是什么关系？为什么会一步步演化成今天的大模型训练通信体系？

#### 核心主线

> OS 进程隔离\
> ↓\
> IPC：Pipe / Socket / Shared Memory\
> ↓\
> Message Passing + Shared Memory 两类范式\
> ↓\
> 跨机器网络通信\
> ↓\
> MPI：Send/Recv + Collective\
> ↓\
> DMA / RDMA：减少 CPU 和内存拷贝\
> ↓\
> GPU 独立显存出现\
> ↓\
> CUDA Copy / P2P / CUDA IPC\
> ↓\
> GPUDirect RDMA\
> ↓\
> NCCL：GPU Collective / P2P\
> ↓\
> DDP / TP / PP / EP / FSDP

#### 第一阶段｜OS 为什么需要 IPC？

进程的核心特征之一是虚拟地址空间隔离：Process A 中的地址 0x1000 和 Process B 中的 0x1000 不代表同一块物理数据。默认情况下，一个进程不能直接解引用另一个进程的普通指针。因此操作系统需要提供 IPC。

#### 第二阶段｜CPU 世界形成两类基本通信范式

| **范式**        | **典型机制**                            | **核心思想**                         | **后来映射到 GPU 世界**                   |
|-----------------|-----------------------------------------|--------------------------------------|-------------------------------------------|
| Message Passing | pipe / socket / send / recv             | 把数据作为“消息”从发送方交给接收方   | MPI_Send/Recv、ncclSend/Recv              |
| Shared Memory   | shm / mmap + mutex / semaphore / atomic | 让多个进程看到同一块内存，再解决同步 | CUDA IPC、共享 GPU buffer、NVSHMEM 等思想 |

这里出现了以后所有通信系统都绕不开的两个基本问题：

**Communication = Data Movement + Synchronization**

#### 第三阶段｜从单机 IPC 到跨机器 Message Passing

进程跨服务器后不再共享物理内存，通信必须经过网络。socket 将单机 send/recv 的抽象自然推广到网络：用户缓冲区 → kernel/network stack → NIC → 网络 → 对端 NIC → 对端进程。

#### 第四阶段｜HPC 把通信抽象成 MPI

当进程规模达到几百、几千甚至更多时，手写成百上千个 Send/Recv 会非常复杂。MPI 因此把常见通信模式标准化为两类：

- Point-to-Point：Send / Recv。

- Collective：Broadcast、Reduce、AllReduce、Gather、AllGather、Scatter、ReduceScatter、AllToAll 等。

非常重要：AllReduce 并不是 GPU 或深度学习发明的。它早已是 HPC/MPI 的经典 collective；分布式训练只是发现“梯度同步”天然就是一个 AllReduce 问题。

#### 第五阶段｜DMA / RDMA：从“CPU 搬数据”到“设备自己搬”

普通数据搬运如果处处依赖 CPU copy，会浪费 CPU 周期并增加内存带宽压力。DMA 让设备在 CPU 配置好 descriptor 后自行读写内存；RDMA 再把这种思想扩展到远端机器，使网络通信越来越像对远端内存执行 read/write。

> 传统思路：CPU 参与多次 copy / protocol processing\
>
> DMA： Device ↔ Host Memory，CPU 主要负责下发与管理\
>
> RDMA： NIC A ═════ fabric ═════ NIC B → Remote Memory

#### 第六阶段｜GPU 出现：系统里多了一块独立 device memory

GPU 最初更像 CPU 的加速外设，数据需要通过 cudaMemcpy 在 Host RAM 与 GPU VRAM/HBM 之间搬运。多 GPU 出现后，又产生 GPU0→GPU1 的 device-to-device 通信需求。

> 最早： GPU0 → Host RAM → GPU1\
> 改进： GPU0 ── P2P / Peer Access ──→ GPU1

CUDA P2P 解决“同机 GPU 之间怎样直接搬数据”，底层实际路径可以是 PCIe，也可以是 NVLink/NVSwitch。注意：NVLink 是物理互联，cudaMemcpyPeer / Send / AllReduce 才是更上层的通信操作。

#### 第七阶段｜不同进程各自管理 GPU：CUDA IPC 再次出现

如果 Process A 管 GPU0、Process B 管 GPU1，A 中的 GPU pointer 对 B 并不天然有效。这和 CPU 进程地址空间隔离是同一个问题。CUDA IPC 的做法是：A 导出 GPU memory/event 的可传递 handle，经标准 OS IPC 把 handle 交给 B，B 再把它映射成自己可用的 device-side 资源。

> Process A / GPU0\
> cudaIpcGetMemHandle()\
> │\
> ├── OS IPC 只传 handle / metadata ──→ Process B\
> │ cudaIpcOpenMemHandle()\
> └────────────────────────────────────→ 映射/访问 GPU memory

因此 GPU IPC 并没有取代 OS IPC，而是在 OS IPC 上再加了一层 GPU memory/resource 语义。

#### 第八阶段｜跨机器 GPU 通信：GPUDirect RDMA

跨节点时，最笨的路径是 GPU HBM → CPU DRAM → NIC → 网络 → NIC → CPU DRAM → GPU HBM。GPUDirect RDMA 的关键目标是让支持的 HCA/NIC 直接 DMA GPU memory，避免不必要的 host-memory staging。

> GPU HBM → PCIe → HCA ═════ InfiniBand/RoCE ═════ HCA → PCIe → GPU HBM\
> ↑ 直接 DMA GPU memory，避免 CPU DRAM 中转 ↑

#### 第九阶段｜NCCL：把“GPU 通信怎么做”封装成高性能原语

当程序需要自己处理 P2P、PCIe/NVLink 拓扑、NIC/HCA、RDMA、ring/tree、chunk/channel 等细节时，工程复杂度会极高。NCCL 把这些细节封装起来，对上提供熟悉的 communication primitives。

| **类型**           | **典型 NCCL/MPI 原语**    | **大模型训练中的典型用途**                      |
|--------------------|---------------------------|-------------------------------------------------|
| Point-to-Point     | Send / Recv               | Pipeline Parallel 的 activation / gradient 传递 |
| Reduction          | Reduce / AllReduce        | DDP 梯度同步、部分 TP 聚合                      |
| Partition + Gather | ReduceScatter / AllGather | FSDP/ZeRO、TP/SP                                |
| Permutation        | AllToAll                  | MoE Expert Parallel token dispatch              |

#### 把通信分成五层来理解

| **层次**     | **例子**                                  | **回答的问题**           |
|--------------|-------------------------------------------|--------------------------|
| L5 并行策略  | DDP / TP / PP / EP / FSDP                 | 模型为什么需要通信？     |
| L4 通信原语  | AllReduce / AllGather / RS / Send/Recv    | 要交换什么数据？         |
| L3 通信库    | NCCL / MPI / NVSHMEM                      | 由谁实现这些原语？       |
| L2 Transport | CUDA P2P / SHM / RDMA / IB Verbs          | 数据具体怎样被搬运？     |
| L1 Hardware  | PCIe / NVLink / NVSwitch / InfiniBand HCA | 数据最终走哪条物理路径？ |

#### 最重要的 Insight

今天 GPU 通信并不是一套凭空出现的新理论，而是 OS IPC 的隔离/共享思想、MPI 的 message-passing 与 collective 抽象、DMA/RDMA 的设备搬运机制，再叠加 CUDA 的 device memory/stream 模型后形成的。理解这条演化线后，NCCL、NVLink、InfiniBand、GPUDirect RDMA 就会自然落在不同层次上。
