# 从两个进程、两张卡学通信

> 前置：[进程与异步](03_进程线程与四种异步.md)、tensor 的 shape/dtype。预计 3–4 小时。CPU 可先运行 Gloo 演示；H20 上用 NCCL。

## 1. 一个 launcher，两个 Python 进程

`torchrun --nproc-per-node=2` 创建两个执行同一脚本的 worker。每个 worker 有自己的 PID、变量与模型；脚本通过环境变量得知自己是哪个参与者。

| 名字 | 两卡单节点例子 | 用途 |
|---|---|---|
| RANK | 0、1 | 全局参与者编号 |
| LOCAL_RANK | 0、1 | 当前节点内 worker 编号 |
| WORLD_SIZE | 两者都是 2 | 全局参与者数 |
| device | 显式绑定到当前 worker 使用的 GPU | 模型和 tensor 放在哪里 |

逻辑 rank 不是永久的物理卡号。必须同时看可见设备映射。初始化进程组是让各参与者对通信域达成一致，不是自动帮你创建或切分模型。

## 2. 运行完整示例

脚本：[collectives.py](../labs/collectives.py)。从仓库根目录运行：

```bash
# CPU 版：先核对数据语义，不代表 GPU 性能
torchrun --standalone --nproc-per-node=2 learn_docs/labs/collectives.py --backend gloo

# Linux + 两张可见 GPU + PyTorch NCCL
torchrun --standalone --nproc-per-node=2 learn_docs/labs/collectives.py --backend nccl
```

脚本只创建小 tensor，打印身份并检查 AllReduce/AllGather 的结果，正常退出时销毁进程组。日志先后顺序可能变化，结果不应变化。初始化失败先查两个 worker 是否都启动，再查设备与 backend；不要先调整几十个 NCCL 环境变量。

## 3. 手算两种结果

开始时 rank 0 为 `[1,1]`，rank 1 为 `[2,2]`。

| 操作 | rank 0 输出 | rank 1 输出 |
|---|---|---|
| AllReduce SUM | `[3,3]` | `[3,3]` |
| AllGather 原始输入 | `[[1,1],[2,2]]` | `[[1,1],[2,2]]` |

AllReduce 合并数值，AllGather 保留每个分片后拼起来。两者输出形状与后续用途不同。“把数据同步一下”不足以确定该用哪一种。

ReduceScatter 可以先理解为：把各 rank 的完整输入按元素规约，再让每个 rank 只拿一部分。例如输入 `[1,2,3,4]` 与 `[10,20,30,40]`，规约结果是 `[11,22,33,44]`，两卡各拿前后两项。其具体 backend/API 支持要核对实际环境。

## 4. 为什么通信会一直等

所有参与者需要按匹配的顺序、shape、dtype 和通信域参与。以下是概念性的错误，不要在共享任务中故意制造无限等待：

```text
rank 0：进入 collective A
rank 1：因为 if 分支跳过 A，直接进入 collective B
```

如果 rank 1 在数据加载时先报错，rank 0 可能最后表现为通信超时。超时是你看到问题的位置，原始错误可能在另一个进程更早的日志中。

## 5. 从正确性进入性能

通过小张量校验后，再做消息尺寸扫描。每个点记录参与卡数、拓扑、dtype、消息 bytes、warmup、重复、时间分布。用于显式计时的同步只放在测量边界；不要在生产路径每个通信后无条件全局 barrier。

一个简化模型是 `T ≈ α + bytes / bandwidth`：小消息更容易被固定延迟支配，大消息更容易暴露带宽。真实 collective 还要算算法步骤数、链路复用与到达偏斜，见[通信性能模型](../02_Distributed_Communication_Memory/08_通信性能模型.md)。

## 6. 验收

- 手算并验证两种 collective 的结果。
- 解释 PID、rank、local_rank 与 device 的区别。
- 比较小消息与大消息，不把 CPU/Gloo 吞吐当作 NCCL 基线。
- 能指出“某 rank 先失败”和“所有 rank 到齐但网络慢”需要不同证据。

下一步：[通信原语](../02_Distributed_Communication_Memory/03_通信原语与Collective.md) → [DP/DDP](../03_Parallelism/02_DP与DDP.md) → [TP](../03_Parallelism/04_TensorParallelism.md)。API 依据：[PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html)、[torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)。
