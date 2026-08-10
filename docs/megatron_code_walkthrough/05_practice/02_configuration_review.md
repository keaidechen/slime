# 训练配置评审与上线清单

## 1. 配置摘要

- 模型：参数量、层数、hidden、heads/GQA、FFN、词表、seq length、dense/MoE。
- 批量：MBS、GBS、DP、num microbatches、token budget。
- 精度：param/grad/reduction dtype、loss scaling、FP8 recipe。
- 环境：commit、镜像、GPU、驱动、CUDA/NCCL/PyTorch/TE。

## 2. 并行与拓扑

- 写出 `world_size` 分解及所有整除约束。
- 画 rank 到 host/GPU/NIC 的映射。
- 标出 TP/PP/CP/EP/DP group 是否跨节点。
- 估算主要 collective 的大小、频率、是否可 overlap。
- PP 写出每 stage 层数、VPP chunks、warmup 与 bubble。

## 3. 显存与状态

分别估算参数、gradient、master/moment、activation、通信 workspace、graph/offload buffer 与 allocator 余量。记录四个峰值：构模后、forward、backward、optimizer/param gather。说明使用 DDP+Distributed Optimizer 还是 FSDP，以及 checkpoint state dict 形式。

## 4. 数据与可复现

- 数据 manifest、blend、split、cache 和 tokenizer metadata 有版本。
- mock-data 计算上限与真实数据 data-wait 都已测。
- baseline loss/grad norm 与短程 hash 已保存。
- save/resume 连续性与目标拓扑恢复已验证。

## 5. 性能验收

报告 max-rank step time 的 p50/p95，而非只报 rank0 平均；同时给 tokens/s/GPU、MFU 口径、峰值 allocated/reserved、网络吞吐和 pipeline bubble。Profiler 只截取 warmup 后少量稳定 step，并区分累计 CUDA/NCCL 时间与真正暴露在 critical path 的时间。

## 6. 回滚与故障演练

每个激进特性都有单独开关和已验证 fallback。至少演练 rank 异常、数据慢节点、checkpoint 写失败、恢复时 DP 改变、极端 MoE 路由、NaN/Inf。成功标准是及时发现第一因、资源清理、坏 checkpoint 不被发布，而不仅是作业最终退出。

