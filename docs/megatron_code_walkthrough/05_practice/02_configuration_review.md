# 训练配置评审与上线清单

<!-- learning-position -->
> **学习定位**：A3/A7 · 必修。
> **前置**：[通信与 tensor 基础](<../../../learn_docs/00_Foundations/06_两卡通信与torchrun.md>)。
> **首读/二读**：每个 Megatron 实验填写，避免只保存启动命令。
> **进度与实验**：[学习清单](<../../../learn_docs/学习清单.md>) · [总入口](<../../../learn_docs/README.md>)。
<!-- /learning-position -->

<a id="beginner-example"></a>

## 入门例子：给清单填一个可被别人复核的配置

不要只勾“并行正确”。写下模型参数量、序列长度分布、global/microbatch、梯度累积次数、各 group 的 rank 列表和实际 GPU 映射。普通参数与 expert 参数分别记录 owner。

例如 global batch 固定时提高 DP，会减少每卡样本或 microbatch 数；若改动后实际有效 token 总数不同，性能对比需要重做口径。恢复实验则要同时检查样本位置、LR 和参数，而不只是文件能读。

提交评审时至少附一张状态显存表、一张 step 时间线、完整命令及正确性结果。不确定项注明尚缺什么测量，避免把估算写成实测。

## 机制与实现

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
