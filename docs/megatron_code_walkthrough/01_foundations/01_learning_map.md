# Infra 工程师的 Megatron 学习地图

## 1. 先建立正确边界

Megatron 不是一个“把模型包进 DDP”的脚本，而是四层系统：

```text
pretrain_gpt.py / examples
  └─ megatron.training：参数、初始化、训练循环、日志、checkpoint
      └─ megatron.core：模型组件、并行状态、调度器、optimizer、checkpoint
          └─ PyTorch distributed + Transformer Engine + CUDA/NCCL
```

阅读时始终问：当前代码属于控制面还是数据面？控制面创建进程组、选 schedule、组织 batch；数据面执行 GEMM、attention、collective 和内存搬运。很多问题来自把两者混在一起。

## 2. 必须掌握的知识点

### A. 分布式与硬件基础

- CUDA stream/event、异步 kernel launch、pinned memory；
- NCCL 的 all-reduce、reduce-scatter、all-gather、all-to-all、send/recv；
- NVLink/NVSwitch、PCIe、InfiniBand/RoCE 的拓扑与带宽层级；
- `torchrun`、global/local rank、world size、process group、collective 顺序一致性；
- roofline 直觉：GEMM 往往 compute-bound，decode/小算子/通信往往 bandwidth 或 latency-bound。

验收题：为什么 TP 通常优先放在节点内？为什么 EP 的 all-to-all 对网络更敏感？collective hang 为什么经常不是 NCCL 本身的 bug？

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

## 3. 建议的学习实验

1. 单 GPU 跑极小 GPT，打印一个 step 的调用链和 tensor shape。
2. 2 GPU TP=2，用 profiler 找到 column/row parallel 的 collective。
3. 2 GPU PP=2，microbatch 从 1 增至 8，观察 bubble。
4. 4 GPU 比较 DP=4、TP=2×DP=2 的显存和吞吐。
5. 开启 distributed optimizer，记录 optimizer state 和通信变化。
6. 制造一个 rank 条件分支，让 collective 序列不一致，学习定位 hang。
7. 保存 checkpoint 后改变 DP/TP 布局加载，检查哪些状态可重分片。

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
y = f(x)           # y.grad_fn 保存反向图入口
y.data = scalar    # payload 可被替换，grad_fn 仍可能存在
torch.autograd.backward(y, grad_y)
```

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

### GPU 性能

至少能看懂 profiler 中：

- GEMM 的 M/N/K 是否太小；
- NCCL kernel 是否与 GEMM 重叠；
- CPU 是否在 kernel 之间留下 launch gap；
- H2D/D2H 是否在独立 stream；
- allocator 是否频繁申请或发生大峰值。

## 6. 一条 8 周学习路线

| 周 | 主题 | 交付物 |
|---|---|---|
| 1 | 单卡主链和 GPT | 一次 step 的函数/shape 图 |
| 2 | TP/SP | 2 卡 collective timeline |
| 3 | PP/VPP | 手绘 1F1B 与实测 bubble |
| 4 | DDP/optimizer | grad buffer 区间图 |
| 5 | checkpoint/restart | 连续训练等价性报告 |
| 6 | CP/MoE | ring softmax 与 token dispatch 图 |
| 7 | FP8/recompute/offload | 显存—吞吐 trade-off |
| 8 | 综合调优 | 一份目标模型并行方案评审 |

## 7. 本章延伸阅读

- [Megatron-LM 原始论文：模型内并行](https://arxiv.org/abs/1909.08053)：先读 TP 的列并行/行并行图。
- [Megatron Core 官方并行策略指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)：用于核对当前版本支持的组合与参数。
- [Megatron Core 官方产品页](https://developer.nvidia.com/megatron-core)：了解 Core、LM、Bridge 与 Transformer Engine 的边界。
- [ZeRO 论文](https://arxiv.org/abs/1910.02054)：理解 optimizer/gradient/parameter 三类状态为何可以分片。
- [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)：理解 sequence parallel 与 interleaved pipeline 的来源。
