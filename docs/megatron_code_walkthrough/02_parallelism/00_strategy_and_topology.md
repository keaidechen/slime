# 并行策略选择与物理拓扑

## 1. 每个维度解决不同约束

| 维度 | 切分对象 | 典型通信 | 首要目标 |
|---|---|---|---|
| DP | batch | gradient AR 或 RS/AG | 扩吞吐 |
| TP | 单层权重/activation | 高频 AR、AG、RS | 让单层放得下 |
| PP | layer 深度 | stage 间 send/recv | 分摊参数和 activation |
| SP | TP 区域内的 sequence | 与 TP collective 配对 | 降低非 TP 区域 activation |
| CP | attention context | ring/P2P/AG | 支持长序列 |
| EP | experts | token all-to-all | 扩展稀疏参数 |

DP 是 dense/attention 网格中的剩余副本维度，但不能机械套用一个乘法式。当前实现分别建立包含 CP 的 decoder grid 和包含 EP 的 expert grid；EP 会改变 expert DP 的解释，而不是在同一 world size 外再乘一次。真实分组以 `megatron/core/parallel_state.py` 和 `RankGenerator` 为准，公式推导见[源码问题详解第 1 节](../06_reference/03_source_questions.md)。

## 2. 选择顺序

1. 先用 DP，直到单副本放不下或 batch 已受约束。
2. TP 优先放在 NVLink/NVSwitch 域内，因为每层都可能通信。
3. 参数或 activation 仍超限时增加 PP，并保证有足够 microbatch 隐藏 bubble。
4. 长上下文引入 CP；MoE 引入 EP，并重新评估 all-to-all 网络。
5. 再叠加 overlap、recompute、offload、FP8 和 CUDA Graph。每次只改变一个主要变量。

## 3. 逻辑拓扑必须落到物理拓扑

同一个 `TP=8` 在单机 NVSwitch 和跨节点 InfiniBand 上不是同一方案。设计时显式写出：rank → host → local GPU → NIC；每种 process group 是否跨节点；collective 的消息大小与频率；是否与计算处于同一 critical path。

经验规则不是硬约束：TP 通常节点内；PP 可跨节点；DP 更能容忍跨节点；EP 对网络抖动和负载不均非常敏感。最终以实测 topology、NCCL 选择和 profiler 时间线为准。

## 4. 一个配置的基本算术

设每个数据副本消耗 `TP × PP × CP` 个 rank，dense 模型常见：

```text
DP = world_size / (TP × PP × CP)
GBS = MBS × DP × num_microbatches
```

再检查层数能否按 PP/VPP layout 分配、attention head 是否可按 TP 切分、序列长度是否满足 CP 切分约束、MoE expert 数是否可按 EP 切分。算术整除只是必要条件，不代表性能合理。

## 5. 方案评审的输出

一份可执行方案必须包含拓扑图、并行组、每 rank 状态/activation 估算、microbatch 和 bubble、主要 collective、checkpoint 恢复策略、correctness baseline 与失败回滚。可直接使用[配置评审清单](../05_practice/02_configuration_review.md)。
