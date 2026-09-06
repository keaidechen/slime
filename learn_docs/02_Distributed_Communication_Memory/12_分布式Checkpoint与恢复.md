# 分布式 Checkpoint 与恢复

Checkpoint（检查点）不是简单的 `torch.save(model)`。大规模任务要同时解决：分片状态一致性、并行写入、存储峰值、拓扑变化后的 reshard、版本兼容与故障恢复时间。

## 需要保存什么

| 状态 | 是否通常需要 | 说明 |
|---|---|---|
| model parameters/buffers | 是 | 可能按 TP/PP/DP/EP 分片 |
| optimizer states | 训练恢复需要 | 通常比参数更大 |
| learning-rate scheduler | 是 | 否则训练轨迹变化 |
| random number generator state | 精确恢复需要 | CPU/CUDA/data pipeline 均可能有 RNG |
| data loader progress | 通常需要 | 避免重复/跳过样本 |
| gradient scaler | mixed precision 需要 | 影响 loss scaling |
| topology/metadata | 是 | 用于解释 shard 与版本 |

## 一致性切面

所有 rank 必须对“保存的是哪一个 step”达成一致。若某些 rank 已 optimizer step、另一些尚未，文件即使完整也不可恢复。常用办法是在安全点 quiesce，或构建 copy-on-write snapshot 后让训练继续、异步落盘。

```mermaid
sequenceDiagram
    participant T as Training ranks
    participant S as Snapshot buffers
    participant W as Async writers
    participant O as Object storage
    T->>S: 一致性切面复制/引用
    T-->>T: 继续训练
    S->>W: 分片排队
    W->>O: 并行上传 + checksum
    O-->>W: commit manifest
```

最后写 manifest/commit marker 很重要：恢复端只加载已经原子提交的版本，避免把半套 shard 当成有效 checkpoint。

## Sharded checkpoint 与 reshard

Sharded checkpoint 让各 rank 并行保存本地 shard，避免 rank 0 聚合导致 OOM 与单点 I/O。恢复时若 world size 或 mesh 改变，需要 load-time reshard：读取逻辑全局 tensor metadata，把旧 shard 映射成新 shard。

PyTorch Distributed Checkpoint（DCP，分布式检查点）支持多 rank 并行 save/load 和 load-time reshard。兼容性仍取决于 state dict 结构、tensor key、dtype、planner 和应用版本，不能只依赖文件扩展名。

## Checkpoint 频率的期望成本

若故障平均间隔为 $M$，一次 checkpoint 花费 $C$，经典近似的最优周期与 $\sqrt{2CM}$ 同阶。直觉是：太频繁浪费 I/O，太稀疏则故障后重算多。大集群的系统级 Mean Time Between Failures（MTBF，平均故障间隔）会随组件数量增长而下降，因此 checkpoint 必须进入吞吐模型。

## 可靠性验证

一个“成功写完”的 checkpoint 仍可能不可用。必须定期做：

1. checksum 和 manifest 完整性验证；
2. 在独立 job 中实际 load；
3. 改变 DP size 的 reshard 演练；
4. 至少跑若干 step 比较 loss/optimizer continuity；
5. 注入 rank failure、storage timeout、部分文件缺失；
6. 测量 Recovery Time Objective（RTO，恢复时间目标）与 Recovery Point Objective（RPO，恢复点目标）。

## Silent Data Corruption

Silent Data Corruption（SDC，静默数据损坏）不会总以进程 crash 呈现。ECC、checksum、梯度/权重统计、loss anomaly、冗余验证与 checkpoint lineage 共同构成防线。NCCL RAS 能诊断部分 hardware/communicator 状态，但不验证模型 tensor 数值正确性。

## 资料

- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [DeepSpeed Universal Checkpointing](https://arxiv.org/abs/2406.18820)
- [NCCL RAS](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html)

