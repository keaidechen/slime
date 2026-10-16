# Megatron 框架与执行边界

相关基础：[通信与 tensor 基础](<../../../communication/两卡通信与torchrun.md>)。


## 1. 先建立正确边界

DDP（DistributedDataParallel，分布式数据并行）让各数据并行参与者持有模型副本，处理不同数据并同步梯度。全局目标的平均分母与框架的梯度缩放规则必须配合。

Megatron 不是一个“把模型包进 DDP”的脚本，而是四层系统：

```text
pretrain_gpt.py / examples
  └─ megatron.training：参数、初始化、训练循环、日志、checkpoint
      └─ megatron.core：模型组件、并行状态、调度器、optimizer、checkpoint
          └─ PyTorch distributed + Transformer Engine + CUDA/NCCL
```

阅读时始终问：当前代码属于控制面还是数据面？控制面创建进程组、选 schedule、组织 batch；数据面执行 GEMM、attention、collective 和内存搬运。很多问题来自把两者混在一起。

### 先把一张 GPU 上的数学闭环走通

用一个只有标量参数 w 的模型代替 GPT：输入 `x=2`，预测 `y=wx`，目标 `t=1`，损失 `L=(y-t)²/2`。初始 `w=1` 时，`y=2,L=0.5`。链式法则逐项给出 `∂L/∂y=y-t=1`、`∂y/∂w=x=2`，因此 `dL/dw=2`；学习率 `η=0.1` 的梯度下降更新为 `w'=w-η·dL/dw=0.8`。梯度告诉我们改变参数会怎样改变损失，优化器才真正写入新参数。

Transformer 把这些标量换成矩阵，把一层乘法换成很多层，但仍是这个闭环。activation 是前向算出的中间值，例如 y；参数是跨 step 保留的 w；梯度是本次目标对参数的导数。把 activation、梯度和参数都叫“模型数据”，会掩盖它们不同的存活时间。

## 2. 必须掌握的知识点

### A. 分布式与硬件基础

pinned memory（页锁定主机内存）在注册期间保持物理页稳定，可供某些异步 DMA 路径使用。它消耗主机资源，是否真正与计算重叠还取决于设备、stream 依赖和缓冲区生命周期。

Roofline 用算术强度 I=F/Q 连接计算与数据搬运：可达到的计算速率上界为 min(P_peak, BW×I)。F 是 FLOPs，Q 是选定存储层级的传输字节数，BW 是该层带宽，峰值要匹配计算精度。

FLOP 是一次浮点运算；FLOPs 表示运算总数，FLOP/s 表示速率。GEMM 是通用矩阵乘法，例如 `[m,k]×[k,n]` 约需 `2mkn` 次运算，一次乘加按两次计。若一个算子做 1000 FLOPs、从显存传 100 bytes，则 `I=10 FLOP/byte`；假设峰值 100 FLOP/s、带宽 6 bytes/s，带宽只供得起 `6×10=60 FLOP/s`，所以用时至少 `max(1000/100,100/6)≈16.7 s`。这是故意缩小的硬件数字；这里的 Q 不能一会儿指显存流量、一会儿指网络流量。

rank 是分布式进程编号，global rank 在整个作业内唯一，local rank 通常用于选择本机 GPU。process group 是参与某种通信的一组进程；collective 是这组进程共同完成的操作。CUDA stream 是 GPU 工作的有序队列，同一 stream 的操作按序执行；不同 stream 的数据依赖需要 event 或等价等待机制。CPU 发出 kernel 请求就可以继续运行，只有消费者依赖已经满足，GPU 结果才可读取。HBM（高带宽显存）保存 GPU 张量，pinned memory 仍是主机内存，二者不应混算。

- CUDA stream/event、异步 kernel launch、pinned memory；
- NCCL 的 all-reduce、reduce-scatter、all-gather、all-to-all、send/recv；
- NVLink/NVSwitch、PCIe、InfiniBand/RoCE 的拓扑与带宽层级；
- `torchrun`、global/local rank、world size、process group、collective 顺序一致性；
- roofline 直觉：GEMM 往往 compute-bound，decode/小算子/通信往往 bandwidth 或 latency-bound。

TP 通常每层都有通信，因此低延迟、高带宽的节点内互联更适合它。EP 的 All-to-All 还受到路由倾斜与跨节点带宽影响；collective hang 则可能来自某个 rank 提前异常、组成员或调用顺序不一致。三者分别涉及拓扑、流量分布与协议，详见[源码问题第 9、10 节](<../06_reference/03_source_questions.md>)。

### B. Transformer 训练基础

- pre-norm Transformer、QKV、GQA、RoPE、SwiGLU、cross entropy；
- activation、参数、gradient、master weight、optimizer moment 的生命周期；
- mixed precision、loss scaling、gradient clipping；
- gradient accumulation 与 micro/global batch：

```text
global_batch_size
  = micro_batch_size × data_parallel_size × num_microbatches
```

流水线下 `num_microbatches` 还决定 bubble 和 1F1B 是否有足够工作。

### C. 并行维度

| 维度 | 切分对象 | 典型通信 | 主要收益 | 主要代价 |
|---|---|---|---|---|
| DP | batch | gradient all-reduce / RS+AG | 扩吞吐 | 模型状态复制或重建 |
| TP | 单层张量 | all-reduce / AG / RS | 单层可放下 | 高频、低粒度通信 |
| PP | layer 深度 | stage 间 P2P | 参数和 activation 分摊 | bubble、调度复杂 |
| SP | 非 TP 区域的 sequence | 与 TP collective 配对 | 降 activation | 额外布局约束 |
| CP | attention 的 sequence/context | ring/P2P/AG | 长上下文 | attention 通信 |
| EP | experts | token all-to-all | 稀疏模型扩展 | 负载不均、A2A |

注意：不能机械套用 `world = TP×PP×CP×EP×DP`。EP 常与其他维度共享/重排 rank 轴，实际分组必须以 `parallel_state.py` 创建的 group 为准。

### D. 执行与调度

- `pretrain()` 的初始化顺序；
- model provider / dataset provider / forward step 三个注入点；
- no-pipeline、1F1B、interleaved 1F1B；
- virtual pipeline stage 和 model chunk；
- activation deallocation、recompute、通信重叠；
- optimizer step 成功/overflow、scheduler advance、checkpoint 的原子性边界。

### E. 模型构建与扩展

- `GPTModel` 与 embedding/decoder/output layer；
- `TransformerBlock`、`TransformerLayer`；
- `ModuleSpec`：结构描述与具体实现解耦；
- local PyTorch spec 与 Transformer Engine spec；
- 参数初始化、权重共享、pipeline stage 上的条件构建；
- packed sequence、sequence/context parallel 对 tensor layout 的影响。

### F. 训练状态与可靠性

- DDP bucket 与 grad buffer；
- distributed optimizer 的 shard/reduce-scatter/all-gather；
- sharded state dict 与 distributed checkpoint；
- RNG state、data sampler state、iteration、optimizer/scheduler 状态；
- deterministic training、straggler、hang、silent data corruption；
- checkpoint 重分片与模型架构变更的兼容边界。

### G. 性能工程

- tokens/s/GPU、model FLOPs utilization、step time 分解；
- overlap flags 的依赖关系和额外 buffer；
- recompute/offload 的 compute、PCIe 和显存交换；
- PP bubble、microbatch 数、VPP、layer 不均衡；
- MoE capacity、token imbalance、grouped GEMM、DeepEP；
- FP8 recipe、amax/scaling、精度回归。

## 3. 从一个设备到多个设备，问题怎样变化

单卡 GPT 给出完整 step 的调用与 shape；TP=2 将 column/row 的局部结果与 collective 接起来；PP=2 则增加跨 stage 的激活和梯度依赖。microbatch 从 1 增至 8，可以在理想调度模型中减少气泡比例，但会改变在途状态与实际开销。

四卡下 DP=4 与 TP=2×DP=2 分摊的是不同对象。Distributed Optimizer 进一步改变优化器状态归属与通信。Checkpoint 改用另一 DP/TP 布局时，则必须区分可重分片张量和其他恢复状态。若一个 rank 的控制分支少发一次 collective，余下参与者的等待来自协议分歧，最后停住的位置未必是第一因。

## 4. 阅读源码的方法

- 先搜符号，不依赖永久行号；
- 每条调用链记录 tensor 的逻辑 shape、物理 shard、所属 process group；
- 一次只改变一个并行维度；
- 将 profiler timeline 与 Python 控制流对齐；
- 所有性能结论都同时报告模型、序列、batch、dtype、GPU、拓扑和版本。

## 5. 需要先会到什么程度

### PyTorch autograd

不要求会写复杂自定义算子，但必须理解：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * x
y.backward()       # 沿保存的计算图计算导数
assert x.grad.item() == 4.0
```

`x` 是需要梯度的 leaf（叶子）张量；`y.grad_fn` 指向生成 y 的反向节点。反向依赖某些前向输入，框架会保存它们，或由重计算重新生成。view 与原张量共享底层 storage；删掉一个变量不保证 storage 释放，其他 view、saved tensor 或异步操作仍可能使用它。in-place 操作直接改写已有 storage，因此会影响所有别名。hook 是在指定执行边界调用的回调，例如梯度就绪后将它写入通信桶。

流水线会在满足特殊前提后替换 output 的 `.data`，再用自定义反向入口驱动原图。这不是普通 autograd 的通用写法；普通 `backward` 对输出与上游梯度的形状有检查，不能照搬“缩成标量后正常 backward”。具体安全边界放在 [PP 章节](<../02_parallelism/02_pipeline_parallel.md>)解释。

这直接关系到流水线的 activation 伪释放。还要能区分 leaf tensor、view、in-place、hook，以及 `param.grad` 与 Megatron `param.main_grad`。

### 分布式通信

用 4 个 rank 手算以下操作：

```text
rank0 [a0,a1] ┐
rank1 [b0,b1] ├─ all-reduce → 每 rank [Σ0,Σ1]

4 rank 各有 8 元素
  reduce-scatter → 每 rank 得到 2 个已归约元素
  all-gather     → 拼回 8 元素
```

必须知道 collective 是“组内所有 rank 共同参与的有序协议”。一个 rank 少调用一次、换了 group 或 tensor count 不同，其他 rank 表现为 hang。

用两个 rank 的整数检验三种常用操作：输入分别 `[1,2,3,4]`、`[10,20,30,40]`，SUM all-reduce（AR，全归约）让两者都得到 `[11,22,33,44]`；SUM reduce-scatter（RS，归约后分发）让 rank0 得 `[11,22]`、rank1 得 `[33,44]`；对这两个结果 all-gather（AG，全收集）又在两边拼回 `[11,22,33,44]`。AG 只拼接，绝不会自动求平均。all-to-all（A2A，全互换）则按目的地发送不同片段，用于专家路由；send/recv 是两个参与者间的点对点（P2P）发送与接收。

### GPU 性能

至少能看懂 profiler 中：

- GEMM 的 M/N/K 是否太小；
- NCCL kernel 是否与 GEMM 重叠；
- CPU 是否在 kernel 之间留下 launch gap；
- H2D/D2H 是否在独立 stream；
- allocator 是否频繁申请或发生大峰值。

## 6. 从执行主线进入专题

主链和 GPT 解释函数与 shape，TP/SP 解释算子两侧的布局转换，PP/VPP 解释在途 microbatch。DDP/optimizer 把梯度写入 buffer 和更新状态连接起来，checkpoint 解释这些状态怎样跨重启延续。

CP/MoE 分别增加长上下文与条件计算的通信；FP8、recompute 和 offload 改变精度、计算和内存之间的交换。读完这些关系，再看完整并行方案，才能把每个配置接到具体状态和成本，而不是只组合缩写。

## 7. 本章延伸阅读

- [Megatron-LM 原始论文：模型内并行](https://arxiv.org/abs/1909.08053)：先读 TP 的列并行/行并行图。
- [Megatron Core 官方并行策略指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)：用于核对当前版本支持的组合与参数。
- [Megatron Core 官方产品页](https://developer.nvidia.com/megatron-core)：了解 Core、LM、Bridge 与 Transformer Engine 的边界。
- [ZeRO 论文](https://arxiv.org/abs/1910.02054)：理解 optimizer/gradient/parameter 三类状态为何可以分片。
- [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)：理解 interleaved pipeline 与 TP/PP/DP 的组合。
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)：理解 sequence parallel 与选择性激活重计算。
