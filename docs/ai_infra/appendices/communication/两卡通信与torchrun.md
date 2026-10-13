# 从两个进程、两张卡学通信

> 前置：[进程与异步](<../../03_Execution/08_GPU执行与异步依赖.md#s2>)、tensor 的 shape/dtype。CPU/Gloo 与 GPU/NCCL 表达相同 collective 数学，但具有不同运行条件。


## 一个 launcher，两个 Python 进程

`torchrun --nproc-per-node=2` 创建两个执行同一脚本的 worker。每个 worker 有自己的 PID、变量与模型；脚本通过环境变量得知自己是哪个参与者。

| 名字 | 两卡单节点例子 | 用途 |
|---|---|---|
| RANK | 0、1 | 全局参与者编号 |
| LOCAL_RANK | 0、1 | 当前节点内 worker 编号 |
| WORLD_SIZE | 两者都是 2 | 全局参与者数 |
| device | 显式绑定到当前 worker 使用的 GPU | 模型和 tensor 放在哪里 |

逻辑 rank 不是永久的物理卡号。必须同时看可见设备映射。初始化进程组是让各参与者对通信域达成一致，不是自动帮你创建或切分模型。


## 启动参数怎样变成进程身份

[collectives.py](<../../examples/collectives.py>)中的同一逻辑可以映射到不同 backend。以下命令表达启动器与 worker 脚本的职责分界：

```bash
# CPU 版：先核对数据语义，不代表 GPU 性能
torchrun --standalone --nproc-per-node=2 docs/ai_infra/examples/collectives.py --backend gloo

# Linux + 两张可见 GPU + PyTorch NCCL
torchrun --standalone --nproc-per-node=2 docs/ai_infra/examples/collectives.py --backend nccl
```

这条命令要分成“启动器参数”和“教学脚本参数”来读：

| 输入部分 | 作用 |
|---|---|
| `torchrun` | PyTorch 分布式启动器，负责创建 worker 进程并设置 rank 相关环境变量 |
| `--standalone` | 使用本机独立 rendezvous（会合）服务，适合这个单机示例 |
| `--nproc-per-node=2` | 在当前节点启动 2 个 worker；不是让一个进程开 2 个线程 |
| `docs/ai_infra/examples/collectives.py` | 每个 worker 都执行的 Python 脚本 |
| `--backend gloo` | 传给教学脚本，选择 Gloo；本例中 tensor 放在 CPU |
| `--backend nccl` | 传给教学脚本，选择 NCCL；需要 CUDA-enabled PyTorch 和至少两张可见 GPU |

脚本正常时会打印两行字典，一行对应一个 worker。`pid` 应不同，`rank`/`local_rank` 应分别为 0 和 1，`world_size` 应为 2，`sum` 应为 `[3.0, 3.0]`，`gather` 应包含两个 rank 的原始输入。两行谁先出现不固定，不能用日志顺序推断 rank 顺序。

脚本只创建小 tensor，打印身份并检查 AllReduce/AllGather 的结果，正常退出时销毁进程组。日志先后顺序可能变化，结果不应变化。初始化失败先查两个 worker 是否都启动，再查设备与 backend；不要先调整几十个 NCCL 环境变量。


## 手算两种结果

开始时 rank 0 为 `[1,1]`，rank 1 为 `[2,2]`。

| 操作 | rank 0 输出 | rank 1 输出 |
|---|---|---|
| AllReduce SUM | `[3,3]` | `[3,3]` |
| AllGather 原始输入 | `[[1,1],[2,2]]` | `[[1,1],[2,2]]` |

AllReduce 合并数值，AllGather 保留每个分片后拼起来。两者输出形状与后续用途不同。“把数据同步一下”不足以确定该用哪一种。

ReduceScatter 可以先理解为：把各 rank 的完整输入按元素规约，再让每个 rank 只拿一部分。例如输入 `[1,2,3,4]` 与 `[10,20,30,40]`，规约结果是 `[11,22,33,44]`，两卡各拿前后两项。其具体 backend/API 支持要核对实际环境。


## 为什么通信会一直等

所有参与者需要按匹配的顺序、shape、dtype 和通信域参与。以下是概念性的错误，不要在共享任务中故意制造无限等待：

```text
rank 0：进入 collective A
rank 1：因为 if 分支跳过 A，直接进入 collective B
```

如果 rank 1 在数据加载时先报错，rank 0 可能最后表现为通信超时。超时是你看到问题的位置，原始错误可能在另一个进程更早的日志中。


## 从正确性进入性能

通信数值正确还不能说明性能。参与卡数、拓扑、dtype、消息字节、warmup 与重复时间分布共同定义比较对象。测量边界的同步用于明确起止，而每次通信后的全局 barrier 会改变生产路径的重叠关系。

一个简化模型是 `T ≈ α + bytes / bandwidth`：小消息更容易被固定延迟支配，大消息更容易暴露带宽。真实 collective 还要算算法步骤数、链路复用与到达偏斜，见[通信性能模型](<../../03_Execution/11_互联通信与成本.md#s2>)。


## 同一程序的两种边界

PID 标识操作系统进程，rank 标识通信组参与者，local_rank 描述节点内编号，device 则是进程实际使用的设备。它们的映射必须一致，但不是同一种身份。AllReduce 合并同位置数值，AllGather 拼接参与者的数据；结果相似的 shape 不代表相同语义。

小消息常由固定开销主导，大消息才更接近带宽限制。CPU/Gloo 可以解释通信数学，却不能替代 GPU/NCCL 性能。如果某 rank 先失败，其他 rank 可能一直等待；如果所有 rank 按相同顺序到达而传输很慢，证据才指向另一类链路或资源问题。

[通信原语](<../../03_Execution/11_互联通信与成本.md#s1>) · [DP/DDP](<../../04_Training/14_数据并行与状态分片.md#s2>) · [TP](<../../04_Training/15_模型并行与执行日程.md#s1>)。依据：[PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html)、[torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)。
