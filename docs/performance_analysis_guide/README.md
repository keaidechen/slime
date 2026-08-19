# 大模型训练、推理与强化学习性能分析：从零到一

这套文档写给第一次接触性能分析的人。你不需要预先了解 CUDA、算子、分布式训练或推理调度；按顺序完成每一章的操作，就能建立一条从“程序很慢”到“找到可验证原因”的完整分析路径。

本文档以 Linux + NVIDIA GPU + PyTorch 为主，框架重点是 Megatron、SGLang/vLLM 和本仓库的 Slime。AMD、NPU、CPU-only 环境虽然也有对应工具，但命令和指标不同，不是本教程的主线。

## 你最终要学会什么

完成教程后，你应该能够独立回答这些问题：

1. 任务慢在数据、Python、CPU 发射、GPU 计算、显存、通信、调度，还是框架阶段之间的等待？
2. `torch.matmul` 这样的 Python 调用最终触发了哪些 ATen 算子和 CUDA kernel？
3. 应该使用 `nvidia-smi`、PyTorch Profiler、Nsight Systems 还是 Nsight Compute？
4. Megatron 的 TP/PP/DP/CP/EP 分别会在时间线上留下什么特征？
5. SGLang/vLLM 的 TTFT、TPOT、ITL、吞吐、并发和 KV Cache 如何一起分析？
6. Slime 中训练、rollout、reward、权重同步和 offload/onload 谁是端到端瓶颈？
7. 如何设计只有一个变量的 A/B 实验，并写出别人可以复现的性能报告？

## 文档目录与建议顺序

| 顺序 | 文档 | 学完后的能力 |
|---|---|---|
| 前置 | [缩写与术语词典](./glossary.md) | 看懂 GPU、CUDA、分布式、推理和 RL 的常用缩写 |
| 0 | [性能分析的基本语言](./00_concepts.md) | 区分延迟、吞吐、利用率、瓶颈和关键路径 |
| 1 | [建立环境清单与可信基线](./01_baseline.md) | 记录环境、监控资源、做可复现 A/B 实验 |
| 配套 | [常用性能分析软件教程](./software_tutorials.md) | 安装、采集并操作 Perfetto、Nsight、TensorBoard 等软件 |
| 2 | [PyTorch：从 Python 代码看到算子和 kernel](./02_pytorch.md) | 正确计时、抓 trace、分析显存和编译图 |
| 3 | [CUDA 与底层算子](./03_cuda_kernels.md) | 使用 Nsight Systems/Compute、Roofline、Triton 和 SASS 工具 |
| 4 | [Megatron 训练框架](./04_megatron.md) | 分解 step time，定位并行通信、bubble、straggler 和 OOM |
| 5 | [SGLang 与 vLLM 推理框架](./05_inference.md) | 构造公平负载，分析 prefill/decode、调度、KV Cache 和容量拐点 |
| 6 | [Slime 强化学习全链路](./06_slime.md) | 分离训练与 rollout，分析权重同步和系统流水线平衡 |
| 7 | [循序渐进实验课](./07_labs.md) | 用六个实验完成从单算子到 Slime 的完整练习 |
| 8 | [工具箱与中英文资料索引](./08_toolbox_and_sources.md) | 按问题选工具，并继续阅读一手资料 |
| 模板 | [性能分析报告模板](./performance_report_template.md) | 固化证据、结论、A/B 结果和回滚方案 |

如果缩写表里大部分词都陌生，先读词典的“最先掌握的 30 个词”，不需要一次背完。不要跳过第 0、1 章。性能分析最常见的错误不是“不会用高级工具”，而是没有 warmup、没有同步 CUDA、改变了多个变量，或只看单次平均值。每当后续章节首次要求使用一种软件，就打开配套软件教程完成对应小节。

## 一条必须记住的主线

无论分析哪个框架，都按同一个顺序：

```text
复现问题
  -> 固定 workload 与环境
  -> 看端到端指标和资源曲线
  -> 把时间分到阶段
  -> 用系统时间线找关键路径
  -> 只对关键路径里的热点 kernel 下钻
  -> 提出一个可证伪假设
  -> 只改一个变量做 A/B
  -> 同时验证正确性与性能
```

工具不是按“高级程度”选择，而是按问题层级选择：

| 你现在的问题 | 首选工具 |
|---|---|
| 哪张卡空闲、功耗/频率/显存是否异常？ | `nvidia-smi dmon`、DCGM |
| 哪个 Python/ATen 算子最耗时？ | PyTorch Profiler |
| CPU、GPU、NCCL 和多个进程如何互相等待？ | Nsight Systems |
| 某个 CUDA kernel 为什么慢？ | Nsight Compute |
| Python 线程卡在哪里？ | `py-spy`、`cProfile` |
| Python/native 内存是谁分配的？ | Memray |
| PyTorch CUDA allocator 为什么 OOM？ | PyTorch Memory Snapshot / Memory Viz |
| 分布式 collective 是否异常？ | Megatron timer、NCCL trace、`nccl-tests`、Nsight Systems |
| 推理服务是否达到容量上限？ | SGLang/vLLM benchmark + Prometheus/DCGM |
| Slime 是训练慢还是 rollout 慢？ | `perf/*` 指标，再分别 profile 两侧 |

## 开始前的最低条件

建议准备：

- 一台 Linux GPU 机器；单卡也能完成前 4 个实验。
- 可用的 PyTorch CUDA 环境。
- 能执行 `nvidia-smi`。
- 若要完成底层分析，安装与驱动兼容的 Nsight Systems 和 Nsight Compute。
- 若要完成框架章节，准备一个能够跑通的小模型、小 batch 配置。不要用生产规模开始学习 profiler。

先执行：

```bash
python - <<'PY'
import platform
import torch

print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

nvidia-smi
which nsys || true
which ncu || true
```

如果没有 `nsys`/`ncu`，仍可先完成第 0～2 章。安装方式与容器权限见第 3 章。

## 本教程的证据约定

文档中的结论分三类：

- **测量事实**：来自日志、计数器或 trace，例如“rank 7 的 step 比其他 rank 慢 20%”。
- **解释假设**：根据事实推测原因，例如“可能是 MoE token 路由不均”。
- **验证结果**：只改变一个变量后，假设被支持或否定。

在没有 A/B 验证前，不要把解释假设写成结论。

## 与仓库已有文档的关系

这套教程负责“从零学习方法”。完成基础章节后，可以继续阅读仓库中更贴近源码的材料：

- [Slime 工程化与可观测性](../code_walkthrough/09_engineering_observability.md)
- [Megatron 性能与排障](../megatron_code_walkthrough/05_practice/01_performance_debugging.md)
- [SGLang Benchmark、Profiling 与可观测性](../sglang_code_walkthrough/05_production_engineering/01_benchmark_profiling_observability.md)
- [Slime 官方性能分析简表](../zh/developer_guide/profiling.md)
- [Slime rollout trace 可视化](../zh/developer_guide/trace.md)

## 版本说明

性能工具和框架参数变化很快。本文档按 2026-08-19 的官方资料与当前仓库代码整理。执行命令前先运行 `--help`，以本机安装版本为准。教程会把版本敏感或内部 API 明确标出来。
