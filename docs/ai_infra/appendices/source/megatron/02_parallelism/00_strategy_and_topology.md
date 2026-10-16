# 并行策略选择与物理拓扑

相关基础：[通信与 tensor 基础](<../../../communication/两卡通信与torchrun.md>)。


## 1. 每个维度解决不同约束

先判断“放不下”的是哪一部分。若权重和优化器状态很大，增加保存完整副本的 GPU 不会降低单卡占用；若只是序列太长导致 activation 峰值高，分片优化器也未必有用。并行方案应从容量账本和一次更新的工作量出发。

DP（Data Parallelism，数据并行）让多个模型副本处理不同样本；TP（Tensor Parallelism，张量并行）把层内矩阵拆开；PP（Pipeline Parallelism，流水线并行）把不同层放在不同设备。SP（Sequence Parallelism，序列并行）在 TP 的部分区域沿 token 维保存激活；EP（Expert Parallelism，专家并行）把不同专家放到不同设备。ZeRO 是 Zero Redundancy Optimizer 的简称，FSDP 是 Fully Sharded Data Parallel，二者围绕数据副本间的状态冗余进行分片。表中 AR/RS/AG 分别是全归约、归约分发、全收集，具体整数例子见[框架基础](<../01_foundations/01_framework_overview.md>)。

CP（Context Parallelism，上下文并行）把同一序列的位置拆给多个参与者。每个查询仍需按 Attention 规则获得其他分片的信息，因此必须交换 K/V 或等价状态，并正确合并归一化。

ETP（Expert Tensor Parallelism，专家张量并行）把同一个专家的矩阵拆给多个参与者；它与把不同专家分给不同设备的 EP 是两个切分维度。

| 维度 | 切分对象 | 典型通信 | 首要目标 |
|---|---|---|---|
| DP | batch | gradient AR 或 RS/AG | 扩吞吐 |
| ZeRO/FSDP | 数据复制域内的模型状态 | 按策略进行参数 AG、梯度 AR/RS | 降低状态冗余；不是额外独立 rank 轴 |
| TP | 单层权重/activation | 高频 AR、AG、RS | 让单层放得下 |
| PP | layer 深度 | stage 间 send/recv | 分摊参数和 activation |
| SP | TP 区域内的 sequence | 与 TP collective 配对 | 降低非 TP 区域 activation |
| CP | attention context | ring/P2P/AG | 支持长序列 |
| EP | experts | token all-to-all | 扩展稀疏参数 |
| ETP | 单 expert 的矩阵 | expert TP 组内 AG/RS 等 | 分摊单 expert 的权重/计算 |

八类的出发点、收益、代价与原理/源码对应见[统一阅读主线](<../../../../04_Training/14_数据并行与状态分片.md>)。SP 复用 TP 组；ETP 不一定等于普通 TP。

DP 是 dense/attention 网格中的剩余副本维度，但不能机械套用一个乘法式。当前实现分别建立包含 CP 的 decoder grid 和包含 EP 的 expert grid；EP 会改变 expert DP 的解释，而不是在同一 world size 外再乘一次。真实分组以 `megatron/core/parallel_state.py` 和 `RankGenerator` 为准，公式推导见[源码问题详解第 1 节](<../06_reference/03_source_questions.md>)。

## 2. 选择顺序

EDP（Expert Data Parallelism，专家数据并行）表示同一专家参数分片的多个数据副本，需合并各副本处理不同样本得到的梯度。它不能替代专家矩阵内部的 ETP 合并。

1. 先用 DP，直到单副本放不下或 batch 已受约束。
2. TP 优先放在 NVLink/NVSwitch 域内，因为每层都可能通信。
3. 参数或 activation 仍超限时增加 PP，并保证有足够 microbatch 隐藏 bubble。
4. 长上下文引入 CP；MoE 引入 EP，并重新评估 all-to-all 网络。
   单 expert 容量仍超限时比较 ETP；若 expert GEMM 已小，ETP 可能增加开销。显式记录 ETP 和 EDP，避免继承 TP 的默认值却误以为 ETP=1。
5. 再叠加 overlap、recompute、offload、FP8 和 CUDA Graph。每次只改变一个主要变量。

这个顺序是排查起点，不是“必须先开满 DP”。例如单副本参数、梯度和优化器占 18 GiB，activation 占 8 GiB，设备可用预算 24 GiB，总计 26 GiB：DP=2 仍是每卡 26 GiB；若某种状态分片把其中 12 GiB 的状态减半，则每卡变为 `6+12/2+8=20 GiB`，可能已足够。若工作负载换成状态 6 GiB、activation 24 GiB，则同样的状态优化难以解决问题，应看 SP/CP、recompute 或 microbatch。数字是预算例子，通信临时空间仍须另留余量。

## 3. 逻辑拓扑必须落到物理拓扑

同一个 `TP=8` 在单机 NVSwitch 和跨节点 InfiniBand 上不是同一方案。设计时显式写出：rank → host → local GPU → NIC；每种 process group 是否跨节点；collective 的消息大小与频率；是否与计算处于同一 critical path。

经验规则不是硬约束：TP 通常节点内；PP 可跨节点；DP 更能容忍跨节点；EP 对网络抖动和负载不均非常敏感。最终以实测 topology、NCCL 选择和 profiler 时间线为准。

## 4. 一个配置的基本算术

设每个数据副本消耗 `TP × PP × CP` 个 rank，dense 模型常见：

```text
DP = world_size / (TP × PP × CP)
GBS = MBS × DP × num_microbatches
```

同一 world 上，expert 侧另外满足：

```text
expert_DP = world_size / (ETP × EP × PP)
```

两套网格的 PP groups 必须一致。固定 world size 调 ETP 时，EP 或 EDP 需要相应改变；只有 EDP ranks 才复制同一 expert shard。具体 group members 见[ETP 八卡例子](<../../../../04_Training/15_模型并行与执行日程.md#s6>)。

再检查层数能否按 PP/VPP layout 分配、attention head 是否可按 TP 切分、序列长度是否满足 CP 切分约束、MoE expert 数是否可按 EP 切分。算术整除只是必要条件，不代表性能合理。

### 八卡配置从算术走到数据

取 `W=8,TP=2,PP=2,CP=1`，则 dense DP=`8/(2×2×1)=2`。若 `MBS=1,GBS=8`，每副本需累积 `M=8/(1×2)=4` 批。两个 TP ranks 合算一个 stage，两个 PP stages 合算一个模型；四个 rank 合作处理每个副本的数据，而不是各取四份新数据。

若模型含 MoE，另设 `ETP=1,EP=2,PP=2`，则 `EDP=8/(1×2×2)=2`。若 ETP 改为 2 并保持 EP=2，EDP 就降为 1。world size 仍是 8，不是 16；只是专家参数的副本数少了。这是两套网格共享同一批物理 rank 的例子，实际 rank order 还必须让 PP groups 一致。

成本也要按单位展开。一个 FP16/BF16 激活 `[S=4,B=1,H=8]` 为 64 bytes；这是一次逻辑 P2P payload，不是整步通信量。乘上微批次数、前反向次数、stage 边界数，还要考虑 TP/SP 后局部 shape，才得到相应边的流量。小例子用于核算，不用于估算实际网络效率：真实小消息通常受启动延迟主导。

## 5. 方案评审的输出

一份可执行方案必须包含拓扑图、并行组、每 rank 状态/activation 估算、microbatch 和 bubble、主要 collective、checkpoint 恢复策略、correctness baseline 与失败回滚。可直接使用[配置评审清单](<../05_design_and_failures/02_configuration_review.md>)。
