# Megatron 配置背后的状态、拓扑与失败边界

一套训练配置描述的不只是 GPU 数量，而是模型状态放在哪里、每次更新消费什么数据，以及哪些依赖必须在更新前完成。global batch 固定时提高 DP，会减少每卡样本或 microbatch 数；性能变化因此包含调度和算子颗粒度的变化。

## 模型与批量共同决定执行形状

参数量、层数、hidden、heads/GQA、FFN、词表、seq length 与 dense/MoE 定义模型；MBS、GBS、DP、microbatch 数和 token budget 定义工作量。参数、梯度、归约精度、loss scaling 和 FP8 recipe 决定数值路径。源码 commit、镜像、GPU、driver、CUDA/NCCL/PyTorch/TE 限定实际实现。

## 两套网格怎样映射到硬件

在本文的标准 decoder 布局中，dense 网格为 `W=TP×CP×DP×PP`，expert 网格为 `W=ETP×EP×EDP×PP`。整除条件与两套 PP groups 的相容性决定组是否有效。TP、PP、CP、EP、ETP、DP/EDP 是否跨节点，以及 rank 到 host/GPU/NIC 的映射，决定通信路径。

未指定 ETP 时需要按当前参数解析的最终值理解，不能默认 1。ETP 变大还可能改变 EP/EDP。每 stage 层数、VPP chunks 和 microbatch 依赖决定 warmup 与 bubble；消息大小、频率和可重叠区间决定通信成本。

## 稳态分片与瞬时峰值

参数、gradient、master/moment、activation、通信 workspace、graph/offload buffer 与 allocator 余量在不同阶段重叠。构模、forward、backward、optimizer/param gather 的峰值可能来自不同对象。DDP+Distributed Optimizer 与 FSDP 也具有不同参数生命周期和 checkpoint state dict，不能由“都分片”推断恢复方式相同。

## 数据身份是正确性的组成部分

manifest、blend、split、cache 与 tokenizer metadata 定义样本。mock data 提供计算路径的参考，真实数据增加预处理和 data-wait；二者不具有相同供给成本。loss、grad norm 和短程 hash 可以揭示变化，但恢复还需保持样本位置、LR、参数和目标拓扑之间的关系。

## 性能与失败怎样共同限制配置

max-rank step time 的 p50/p95、tokens/s/GPU、明确口径的 MFU、allocated/reserved 峰值、网络与 PP bubble 描述不同成本。累计 CUDA/NCCL 时长与关键路径的暴露区间分开核算；少量稳态 trace 用于归因，不能替代无 profiler 的分布。

rank 异常、数据慢节点、checkpoint 写失败、恢复时 DP 改变、极端路由和 NaN/Inf 分别破坏不同前提。合理的失败边界包括及时传播第一因、清理资源、阻止坏 checkpoint 发布，以及明确的 fallback。作业退出本身不能证明恢复正确。

[数学、实现与性能的边界](<03_parallelism_validation.md>) · [并行方案原理](<../../../parallelism/并行方案的约束与取舍.md>)
