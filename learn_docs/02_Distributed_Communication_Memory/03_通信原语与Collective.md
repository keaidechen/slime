# 通信原语与 Collective

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

