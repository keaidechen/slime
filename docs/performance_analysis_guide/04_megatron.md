# 04 Megatron 训练框架性能分析

Megatron 的难点不是 kernel 更多，而是同一个 step 同时包含多种并行通信和流水线调度。分析目标是把 step time 拆成可行动的部分，并找到最慢 rank 的关键路径。

## 1. 先用最少概念理解五种并行

| 并行 | 切分什么 | 典型通信 | 常见性能问题 |
|---|---|---|---|
| DP | 数据副本 | gradient reduce-scatter/all-reduce、parameter all-gather | 跨节点带宽、bucket、overlap |
| TP | 单层 tensor | all-gather/reduce-scatter/all-reduce | GEMM 变小、频繁通信、跨节点 TP |
| PP | 层 | send/recv activation/gradient | pipeline bubble、stage 不均衡 |
| CP | sequence/context | P2P ring、all-gather、all-to-all | 长上下文通信与 attention overlap |
| EP | MoE experts | token dispatch/combine all-to-all | token 不均、跨节点 A2A、小 expert GEMM |

首先画出本次运行的 `TP × PP × CP × EP × DP` 和 rank 到 node/GPU/NIC 的映射。没有拓扑图时，任何通信结论都不可靠。

Megatron Bridge 的 [Parallelisms Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html) 给出了组合关系；本仓库的源码导读见 [Megatron 并行策略与拓扑](../megatron_code_walkthrough/02_parallelism/00_strategy_and_topology.md)。

## 2. 定义要测的 step

建议先写成：

```text
T_step =
  T_data_wait
  + T_exposed_forward
  + T_exposed_backward
  + T_exposed_communication
  + T_pipeline_bubble
  + T_optimizer
  + T_checkpoint/logging
```

`exposed` 表示未被其他工作隐藏的部分。某个 NCCL kernel 跑 8 ms，但完全与 12 ms GEMM 重叠，它对 step time 的直接贡献接近 0。

最低指标：

- 稳态 step time 的 p50/p95；
- tokens/s/GPU；
- forward/backward/optimizer/data/checkpoint timers；
- 各 rank 的 min/max，明确最慢 rank；
- allocated/reserved/peak memory；
- loss/grad norm；
- PP microbatch 数与各 stage layer 数；
- MoE 每 expert/rank token 分布。

## 3. 第一次基线：先不抓 trace

### 3.1 固定配置

保存：

- MBS、GBS、sequence length、token packing；
- TP/PP/CP/EP/DP/VPP；
- dtype、FP8、recompute、distributed optimizer；
- overlap、fusion、CUDA Graph；
- 模型层数、hidden、heads、experts；
- 节点、GPU、网络和 commit。

GBS 的常见关系：

```text
GBS = MBS × DP × gradient_accumulation_steps
```

改并行度时要保持 token budget/GBS 不变，否则 step time 和吞吐都无法公平归因。

### 3.2 使用 Megatron timers

Megatron Core 的 `Timers` 支持 `max`、`minmax`、`all` 等跨 rank 汇总，并可在记录前选择 barrier。优先用 `minmax`，因为平均值会隐藏 straggler。不同 Megatron/Bridge 版本的 CLI 或 config 名称不同，先查当前入口的 `--help` 或配置类；timer 语义见 [Megatron Core timers API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.timers.html)。

注意：人为加入 barrier 会改变时间线，甚至因参与 rank 不一致造成 hang。只有明确理解进程组时才启用 timer barrier。

### 3.3 建立最小正确配置

新模型或新并行组合先从：

```text
单卡/最小 GPU 数
-> BF16
-> 无高级 overlap
-> 无 FP8
-> 无 CUDA Graph
-> 固定小数据
```

先保证 loss、checkpoint 和恢复正确，再逐个加入 DP、TP、PP、CP/EP、overlap、FP8、graph。性能功能不是 bring-up 的调试工具。

## 4. 先定位是哪一种并行问题

### 4.1 DP

现象：

- backward 尾部出现长 reduce-scatter/all-reduce；
- forward 前 parameter all-gather 暴露；
- 单节点好，多节点明显掉速；
- DP rank 到 collective 的时间差很大。

检查顺序：

1. `nccl-tests` 是否达到合理带宽；
2. GPU-NIC affinity、跨 NUMA、接口选择；
3. bucket 大小是否太小/太大；
4. gradient reduce 与 backward 是否实际 overlap；
5. parameter gather 与 forward 是否实际 overlap；
6. 最慢 rank 是否在 collective 前已落后。

不要看到长 NCCL kernel 就直接调 NCCL 环境变量。先证明网络/通信在关键路径。

### 4.2 TP

现象：

- TP 增大后 GEMM shape 变小，单 kernel TFLOP/s 下降；
- 每层出现频繁 collective；
- TP 跨低带宽节点；
- 低精度加速收益很小，因为 host/communication 主导。

实验：保持 GPU 总数和 GBS 不变，对比 `TP=8, PP=1` 与 `TP=4, PP=2`。同时看 GEMM shape、PP bubble 和通信暴露，不能只看 TP 通信。

### 4.3 PP

现象：

- step 开头/结尾有明显 warmup/flush 空闲；
- 某 stage 一直比其他 stage 晚；
- first/last stage 因 embedding/loss 更重；
- microbatch 太少，bubble 比例高。

检查：

1. microbatch 数；
2. 每 stage 层数和非 Transformer 工作；
3. 每层实际时间，而不是只按层数均分；
4. VPP 是否减少 bubble，却增加过多通信/调度；
5. send/recv 是否与计算 overlap。

PP stage 的负载应按实际 profile 重平衡。NVIDIA 的性能指南指出 embedding/projection 可能使首尾 stage 更重，也说明 VPP 减 bubble 的同时会增加 stage 间通信：[Megatron Bridge Performance Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html)。

### 4.4 CP

现象：

- 长 sequence OOM 得到缓解，但 attention 通信暴露；
- 不同 `cp_comm_type` 时间线不同；
- TP+CP 的组合影响 GEMM 与 attention shape。

不要只增大 TP 或 CP。保持总 GPU 数，做 TP×CP 小矩阵 sweep，并记录吞吐、显存和通信暴露。

### 4.5 EP/MoE

现象：

- dispatch/combine all-to-all 很长；
- 某些 rank expert token 多，成为 straggler；
- expert GEMM 很小；
- token drop/rebalance 改变正确性或模型行为。

至少记录每个 expert/rank 的 token 数分布。先验证路由和 correctness，再测试 grouped GEMM、dispatcher/backend、EP placement 和通信 overlap。官方文档强调 EP overlap 的收益与 workload 有关，小 EP 甚至可能持平或变慢：[Megatron Bridge Communication Overlap](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/communication-overlap.html)。

## 5. 用 PyTorch Profiler 分析 Megatron

如果使用 Megatron Bridge，可以配置目标 step 和 rank：

```python
from megatron.bridge.training.config import ProfilingConfig

cfg.profiling = ProfilingConfig(
    use_pytorch_profiler=True,
    profile_step_start=10,
    profile_step_end=13,
    profile_ranks=[0],
    record_shapes=True,
)
```

rank 0 不一定代表瓶颈。选择 rank 时要覆盖：

- PP first/middle/last stage；
- 已知最慢 rank；
- 跨节点通信边界；
- MoE token 最多/最少的 rank。

先每类选 1 个，不要所有 rank 同时开 `with_stack`。官方配置与 memory snapshot 示例见 [Megatron Bridge Profiling](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/profiling.html)。

非 Bridge 或 fork 版本应找到自己的 `ProfilingConfig`/CLI；不要假设同名参数行为一致。

## 6. 用 Nsight Systems 看全局时间线

Megatron Bridge 官方方式：框架配置 step/rank 范围，再由 nsys 监听 CUDA Profiler API：

```python
from megatron.bridge.training.config import ProfilingConfig

cfg.profiling = ProfilingConfig(
    use_nsys_profiler=True,
    profile_step_start=10,
    profile_step_end=13,
    profile_ranks=[0, 1],
)
```

```bash
nsys profile \
  -s none \
  -t cuda,nvtx \
  -o /tmp/megatron_profile \
  --force-overwrite true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  python your_train.py
```

确认当前 fork 的训练循环确实调用 start/stop；Slime 对 Megatron 原生 nsys 开关的差异见第 6 章。

时间线阅读：

1. 标出每个 microbatch 的 F/B；
2. 标出 PP warmup、1F1B steady、flush；
3. 标出 TP/CP/EP/DP collective；
4. 计算通信的**暴露区间并集**，不要累加所有 stream duration；
5. 找最晚 rank 的第一个分叉点；
6. 再锁定关键路径中的 GEMM/attention/communication。

## 7. 通信 overlap 不能只看“同时出现”

开启 overlap 后要验证三件事：

1. wall time 是否缩短；
2. 计算 kernel 是否因争抢 SM/带宽而变慢；
3. 新 buffer 是否增加峰值显存。

可能出现：通信与计算重叠了，但两者都变慢，最终 step 没改善。比较时间线中的 exposed interval 与端到端 step，而不是比较 kernel 累计和。

NVIDIA 当前指南中的典型 DP 配置是 `overlap_grad_reduce` 与 `overlap_param_gather`，TP overlap 还需要结合 sequence parallel 和具体 backend；这些不是所有版本、所有硬件的通用默认值。先按当前框架配置文档启用，再 A/B 验证。

## 8. rank straggler：先找“谁晚到”

Megatron Core 提供 `StragglerDetector`，可收集每 rank elapsed、GPU util、clock、温度、功耗等并报告 min/max。API 见 [StragglerDetector](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.utils.html)。

若自行加低频采样，可 all-gather 每 rank 的阶段时间；不要每 step 大规模同步。找到最慢 rank 后检查：

- host/NUMA/GPU/NIC placement；
- GPU clock、温度、ECC/Xid；
- 数据长度和 dataloader shard；
- PP stage 层数；
- MoE token 数；
- checkpoint writer/存储；
- Python GC 或日志；
- collective 前的第一处分叉。

最后卡住的 NCCL 调用常常只是受害者。

## 9. NCCL 的分析顺序

### 9.1 硬件基线

先跑匹配消息大小的 `nccl-tests`，记录 algbw/busbw。

### 9.2 日志

临时开启：

```bash
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET \
NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log \
python your_train.py
```

不要长期把 debug tuning 环境变量留在生产脚本。当前 NCCL 文档明确区分系统配置项和只用于调试的变量：[NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)。

### 9.3 hang/desync

PyTorch ProcessGroupNCCL 提供 flight recorder 和 desync debug：

```bash
TORCH_NCCL_TRACE_BUFFER_SIZE=20000 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
TORCH_NCCL_DESYNC_DEBUG=1 \
python your_train.py
```

具体变量依 PyTorch 版本，查 [ProcessGroupNCCL Environment Variables](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html)。这是排 hang/correctness 的工具，不是常驻性能开关。

## 10. Megatron OOM 的四个检查点

分别记录：

```text
模型构建后
第一个 forward 峰值
第一个 backward 峰值
optimizer step / parameter all-gather 峰值
```

归因：

- forward：activation、attention workspace、graph capture；
- backward：saved tensor、recompute 临时、grad bucket；
- optimizer：master weights/moments、all-gather、cast buffer；
- 第二步后上涨：异步 handle、cache、Python 引用、碎片。

Megatron Bridge 可用 `record_memory_history=True` 和 `memory_snapshot_path`；或按第 2 章手动 Memory Snapshot。理论估算只用来提出假设，最终以快照验证。

## 11. 配置 sweep 的安全顺序

### 11.1 容量与并行

```text
MBS/sequence packing
-> TP×PP
-> CP（长上下文）
-> EP/dispatcher（MoE）
```

### 11.2 性能功能

```text
fusion
-> distributed optimizer
-> DP overlap
-> TP/PP/EP overlap
-> recompute（显存与算力权衡）
-> FP8
-> CUDA Graph
```

顺序不是绝对，但每次只引入一个可归因变量。对每个候选保存 correctness、吞吐、step 分解、显存和 trace 证据。

## 12. 症状速查

| 症状 | 首查 | 不要先做 |
|---|---|---|
| GPU 周期性空洞 | PP bubble、data、checkpoint、GC | 直接优化 GEMM |
| NCCL 尾部长 | 谁最晚进入、拓扑、消息大小 | 随机设置 NCCL env |
| 小 kernel 极多 | MBS、TP/CP 过分切分、fusion | 直接 ncu 全任务 |
| 首尾 PP rank OOM | embedding/loss、warmup activation | 只看全局平均显存 |
| MoE rank 抖动 | expert token 分布、A2A | 只看 rank0 |
| FP8 提升小 | host/communication bound、shape | 认定 FP8 未生效 |
| overlap 开启但不快 | exposed interval、资源争用 | 看累计通信时间 |

## 本章完成标准

- 能画出并行组和 rank placement。
- 能用 min/max timer 指出最慢 rank 和阶段。
- 能在时间线上区分 PP bubble 与 TP/DP/EP 通信。
- 能说明通信累计时间与 exposed communication 的区别。
- 能设计保持 GBS/token budget 的 TP×PP A/B。
- 能用正确性、吞吐、显存和 trace 共同验收。

## 参考资料

- [Megatron Bridge Performance Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html)
- [Megatron Bridge Profiling](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/profiling.html)
- [Megatron Bridge Communication Overlap](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/communication-overlap.html)
- [Megatron Core timers](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.timers.html)
- [NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
- [Scaling Language Model Training with Megatron](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)
