# 实验与排查工作表

<!-- learning-position -->
> **学习定位**：A6/A7 · 必修。
> **前置**：[性能语言与测量](<00_concepts.md>)。
> **首读/二读**：实验现场填写原始证据；正式结论使用独立报告模板。
> **进度与实验**：[学习清单](<../../learn_docs/学习清单.md>) · [总入口](<../../learn_docs/README.md>)。
<!-- /learning-position -->

复制本页，为一次性能问题建立可审计记录。

## A. 问题定义

```text
workload / owner:
symptom and first occurrence:
user impact / SLO:
baseline window:
regressed window:
primary metric and unit:
success threshold:
correctness constraint:
```

## B. 环境指纹

```text
code commit / image digest:
framework / runtime:
CUDA / driver / NCCL:
GPU / count / MIG:
CPU / NUMA / memory:
NVLink / PCIe / NIC / fabric:
power / clock policy:
container CPU / memory limits:
dataset / model / tokenizer version:
```

MIG = Multi-Instance GPU（多实例 GPU）；NUMA = Non-Uniform Memory Access（非统一内存访问）。

## C. Workload shape

```text
train: global/micro batch, grad accumulation, seq p50/p95/max
serve: ISL/OSL distribution, arrival pattern, concurrency/rate
parallel: DP/TP/PP/SP/CP/EP/FSDP
dtype / quantization:
MoE: experts, top-k, tokens/expert distribution
RL: roles, placement, policy lag, rollout length
```

DP/TP/PP/SP/CP/EP 分别是数据、张量、流水线、序列、上下文与专家并行；FSDP 是 Fully Sharded Data Parallel（全分片数据并行）。

## D. 快速体检

| 检查 | Baseline | Current | 结论 |
|---|---:|---:|---|
| throughput / latency | | | |
| step/TTFT p50/p99 | | | |
| GPU/SM/DRAM activity | | | |
| HBM allocated/reserved/device | | | |
| power/clock/temp/throttle | | | |
| Xid/ECC/link error | | | |
| CPU/host memory/I/O | | | |
| queue/KV/preemption | | | |
| rank/stage/expert skew | | | |

HBM = High Bandwidth Memory（高带宽内存）；TTFT = Time to First Token（首 token 时间）。

## E. 假设表

| # | 可证伪假设 | 需要的最小证据 | 单变量实验 | 结果 |
|---|---|---|---|---|
| H1 | | | | |
| H2 | | | | |
| H3 | | | | |

把“GPU 利用率低”改写成可验证假设，例如“rank 17 的 DataLoader p99 使其他 rank 提前到达 AllReduce”。

## F. 分层采集

```text
Level 0 — logs/metrics window:
Level 1 — NVTX/PyTorch Profiler rank and steps:
Level 2 — Nsight Systems process/rank and capture range:
Level 3 — Nsight Compute kernel regex and launch count:
Memory — snapshot ranks/phases:
Communication — Flight Recorder/RAS/nccl-tests:
```

每升一级前写清“上一级证据为何不足”。

## G. Timeline 核算

| Phase | p50 | p99 | rank max/min | overlap | exposed critical-path |
|---|---:|---:|---:|---:|---:|
| data/H2D | | | | | |
| forward | | | | | |
| backward | | | | | |
| communication | | | | | |
| optimizer | | | | | |
| queue/scheduler | | | | | |
| checkpoint/eval | | | | | |

## H. Kernel 记录

| Kernel + shape | Duration | Calls | AI | Compute/Memory SOL | Occupancy | Stall | Action |
|---|---:|---:|---:|---:|---:|---|---|
| | | | | | | | |

AI = Arithmetic Intensity（算术强度）；SOL = Speed of Light（理论极限占比）。

## I. 通信账本

| Collective | Group | Bytes | Frequency | Arrival skew | Total | Exposed | Topology |
|---|---|---:|---:|---:|---:|---:|---|
| | | | | | | | |

## J. 显存账本

| 类别 | Steady | Peak | Visible to PyTorch snapshot? | Lifetime/owner |
|---|---:|---:|---|---|
| parameters/grad/optimizer | | | | |
| activations | | | | |
| temporary/workspace | | | | |
| communication | | | | |
| KV / CUDA Graph | | | | |
| context/external | | | | |
| fragmentation/headroom | | | | |

## K. 结果与复现

```text
root cause:
causal evidence:
change:
before → after (p50/p99, N runs):
accuracy/correctness:
peak memory / power / error impact:
tested shapes/scales:
rejected alternatives:
known limits:
rollback:
```

## L. 回归与线上化

- [ ] benchmark 配置和随机种子入库；
- [ ] 指标口径、单位、有效 token 定义固定；
- [ ] warm/cold 分开；
- [ ] 代表 shape 和至少一个边界 shape；
- [ ] performance threshold 与 noise budget；
- [ ] correctness gate；
- [ ] dashboard/alert 可发现同类问题；
- [ ] trace/dump 有过期、脱敏和权限策略；
- [ ] profiler/DCGM counter 冲突已恢复；
- [ ] runbook 包含回退步骤。

## 资料入口

- [专题总览](../../learn_docs/04_Profiling_Performence_Analysis/00_专题总览.md)
- [方法论与测量陷阱](../../learn_docs/04_Profiling_Performence_Analysis/01_方法论与测量陷阱.md)
- [案例：GPU 利用率低](../../learn_docs/04_Profiling_Performence_Analysis/15_案例_GPU利用率低.md)
