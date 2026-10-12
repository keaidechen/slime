# 性能语言、关键路径与成本模型

<details>
<summary>文章导航</summary>

- [1. 性能分析不是找最大的数字](#1-性能分析不是找最大的数字)
- [2. 六个基础概念](#2-六个基础概念)
- [3. 从 Python 到 GPU 的层次](#3-从-python-到-gpu-的层次)
- [4. GPU 为什么会慢：四个大类](#4-gpu-为什么会慢四个大类)
- [5. 训练、推理、RL 分别测什么](#5-训练推理rl-分别测什么)
- [6. 成本模型的适用范围](#6-成本模型的适用范围)
- [7. 症状到工具的第一棵决策树](#7-症状到工具的第一棵决策树)
- [8. 性能目标决定证据的粒度](#8-性能目标决定证据的粒度)
- [9. 跨层定位：八个系统问题](#9-跨层定位八个系统问题)
- [参考资料](#参考资料)

</details>

相关基础：[必要基础](<../00_Physical_Foundations/README.md>)。


Profiler 是记录程序执行时间或资源行为的性能分析工具；trace 是包含事件、时间与依赖的执行记录。本章先不操作这些工具。目标是先建立一套不会混淆的语言。你需要知道“测什么”“为什么慢”和“下一步该看哪里”。


## 1. 性能分析不是找最大的数字

假设 trace 中：

- GEMM kernel 累计 60 ms；
- NCCL 通信累计 30 ms；
- CPU 等待累计 20 ms；
- 整个 step 实际是 70 ms。

不能把三项相加得到 110 ms，因为计算、通信和 CPU 工作可能重叠。真正决定总时长的是**关键路径**：任何优化如果没有缩短关键路径，就不会缩短 step。

以后看到一个“很大的累计耗时”，先问：

1. 它在关键路径上吗？
2. 它是否与别的工作重叠？
3. 缩短它会让后续工作更早开始吗？


### 手算一条关键路径

假设 A 耗时 4 ms，随后 B（6 ms）和 C（3 ms）可并行，D（2 ms）必须等 B、C 都结束。忽略资源争用，总时间为 $4+\max(6,3)+2=12$ ms，关键路径是 A→B→D。

将 C 减半为 1.5 ms，总时间仍是 12 ms；将 B 减半为 3 ms，总时间变成 9 ms；把 B 继续减至 1 ms，则 C 成为限制，总时间仍需 9 ms。瓶颈会转移，这就是局部加速无法无限叠加的原因。

## 2. 六个基础概念

### 2.1 延迟 latency

完成一次操作花多长时间。

- 训练：一次 iteration/step 的秒数。
- 推理：一次请求的端到端时间。
- kernel：设备实际执行该 kernel 的微秒数；主机 launch 调用耗时是另一种计时口径。

延迟要报告分布，而不只是平均值：`p50` 是中位数，`p95` 表示 95% 样本不超过该值，`p99` 用来观察长尾。

### 2.2 吞吐 throughput

单位时间完成多少工作。

- 训练：samples/s、tokens/s、tokens/s/GPU。
- 推理：requests/s、input tokens/s、output tokens/s。
- 算子：TFLOP/s、GB/s。

提高并发常常能提高吞吐，但会增加排队和延迟。吞吐更高不代表用户体验更好。

### 2.3 利用率 utilization

某资源在采样窗口内有多忙。`nvidia-smi` 的 GPU 利用率只说明一段时间内 GPU 上是否有 kernel 活跃，不能证明 kernel 已接近硬件峰值。

下面几种情况都可能显示“GPU 利用率 100%”：

- 高效 Tensor Core GEMM；
- 很慢的低效率 kernel；
- 大量小 kernel 连续运行；
- 内存带宽已满但计算单元很闲。

所以利用率是报警器，不是诊断结论。

### 2.4 饱和 saturation

继续增加负载后，吞吐不再明显上升，但延迟和排队快速上升，此时系统达到容量拐点。推理服务一定要通过并发或 RPS（Requests Per Second，每秒到达请求数）逐档扫描测试找到该点。

### 2.5 瓶颈 bottleneck

当前限制端到端性能的资源或阶段。瓶颈会随 workload 改变：小 batch 可能受 CPU launch 限制，大 batch 可能变成算力或显存带宽限制。


### 2.6 开销 overhead

为管理真正计算而付出的成本，例如 Python 调度、kernel launch、图编译、日志、checkpoint、序列化、HTTP 和 Ray object store。


## 3. 从 Python 到 GPU 的层次

先把图中的名字按职责分组：ATen 是 PyTorch 的基础张量算子层；cuBLAS/cuDNN 是计算库；Triton 是编写与编译 GPU 计算的工具，Inductor 是 PyTorch 的编译后端；PTX 是虚拟指令表示，SASS 是设备机器指令表示。Nsight 是 NVIDIA 的分析工具系列。初读只需知道它们分别属于哪一层，不必掌握各工具用法。

以后看到“算子”一词，需要先问它指哪一层：

```text
业务/框架阶段
  例如 rollout、forward、backward、optimizer
        ↓
Python Module / Function
  例如 TransformerLayer、torch.matmul
        ↓
PyTorch ATen operator
  例如 aten::mm、aten::scaled_dot_product_attention
        ↓
后端库或生成代码
  例如 cuBLAS、cuDNN、FlashAttention、Triton、Inductor
        ↓
CUDA kernel
  Nsight/PyTorch trace 中看到的具体 kernel 名
        ↓
PTX / SASS 指令与 GPU 硬件
```

一个 Python 操作可能触发多个 ATen 算子；一个 ATen 算子也可能触发多个 CUDA kernel。`torch.compile` 还可能把多个 ATen 算子融合成一个 Triton kernel。因此不能只靠函数名猜测底层执行。


## 4. GPU 为什么会慢：四个大类

### 4.1 没有足够工作：latency/launch bound

特征：

- kernel 很短且数量很多；
- GPU 时间线有大量小间隙；
- CPU 线程持续发射 kernel；
- 增大 batch 后效率明显变好。

常见原因：micro-batch 太小、shape 太碎、Python/调度慢、没有融合、CUDA Graph 未命中。

### 4.2 计算受限：compute bound

特征：

- 大 GEMM/attention 占主导；
- Tensor Core/SM 吞吐接近硬件上限；
- 显存带宽没有满；
- 减少 FLOPs 或降低精度可能改善性能。

### 4.3 内存受限：memory bound

特征：

- DRAM/L2 吞吐高，计算吞吐相对低；
- elementwise、normalization、embedding、KV Cache 访问占比高；
- kernel fusion、减少读写、提高数据复用更可能有效。

### 4.4 等待受限：communication / synchronization / I/O bound

特征：

- GPU 时间线出现空洞或长 NCCL kernel；
- 某些 rank 先到 collective，等待最慢 rank；
- 数据加载、网络、存储或 checkpoint 与计算没有重叠；
- 单卡快，多卡扩展差。

NVIDIA 的 GPU 性能背景指南用算术强度和并行度来区分延迟、计算和内存限制；Nsight Compute 的 Roofline 是把这种判断数据化的工具。参见 [GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) 与 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。


## 5. 训练、推理、RL 分别测什么


### 5.1 训练

最低指标集：

- step time 的 p50/p95；
- tokens/s 和 tokens/s/GPU；
- data、forward、backward、optimizer、通信、checkpoint 分段；
- 每个 rank 的时间，至少保留 min/max 和最慢 rank；
- allocated/reserved/峰值显存；
- 正确性护栏：loss、grad norm、样本/token 数。

MFU（Model FLOPs Utilization，模型浮点运算利用率）用模型有效计算速率与选定精度的硬件峰值比较，只有在 FLOPs 公式、精度峰值和统计范围一致时才可比较。不同文章中的 MFU 不能直接横比。

### 5.2 推理

最低指标集：

- TTFT：Time To First Token，首 token 等待时间；
- TPOT：首 token 之后平均每个输出 token 的处理时间；
- ITL：相邻 token 的间隔，用于观察抖动；
- E2E latency；
- request/input/output/total token throughput；
- p50/p95/p99；
- 并发、到达率、输入/输出长度分布、cache 命中率；
- 错误、超时、abort/retract。

同一框架在 `input=128/output=128` 和 `input=8192/output=128` 上不是同一个 workload，吞吐不能直接比较。

### 5.3 强化学习

RL 是训练和在线推理组成的流水线，还要测：

- rollout 生成时间；
- reward/tool/environment 的非生成时间；
- 训练等待 rollout 的时间；
- logprob/ref/teacher forward；
- actor train；
- 权重同步；
- 模型 offload/onload；
- 每轮有效 token 数与样本长度分布。

局部最快不等于端到端最快。给训练侧更多 GPU 可能让它更早结束，却更久地等待 rollout。


## 6. 成本模型的适用范围

| 模型 | 回答的问题 | 容易遗漏的因素 |
|---|---|---|
| Roofline | 计算与某层数据流量决定怎样的吞吐上界？ | 小任务、依赖与并行度 |
| $\alpha+n\beta$，其中 $\beta=1/BW$ | 启动延迟与消息大小怎样影响传输时间？ | 拥塞、拓扑与到达偏斜 |
| Amdahl | 局部加速能缩短多少端到端时间？ | 工作量变化与新瓶颈 |

若可优化部分占 30%，局部加速 2 倍，工作量不变时理想整体加速为 $1/(0.7+0.3/2)\approx1.176$。同样，计算 8 ms、通信 5 ms 只有在依赖允许、资源能够充分重叠时，才可能接近 8 ms；通信若消费计算结果，或二者争用带宽，就不能直接取最大值。

模型给出假设下的约束，观测判断执行是否满足这些假设。环境、工作量、计时与预热、控制变量和正确性如何限定结论，在[证据与因果](证据与因果.md)中集中解释。

## 7. 症状到工具的第一棵决策树

此图供遇到问题时查阅，不是安装清单。py-spy 看 Python 执行，pidstat/iostat 看进程与存储活动；PyTorch Profiler 连接框架算子和设备，Nsight Systems 看系统时间线，Nsight Compute 看单个 Kernel；Memory Snapshot 与 Memray 分别帮助分析设备分配器和主机内存。

```text
端到端变慢
├─ GPU 利用率低或有周期性空洞
│  ├─ CPU 忙：py-spy / PyTorch Profiler / Nsight Systems
│  ├─ data/IO 忙：pidstat / iostat / dataloader timer
│  └─ 多 rank 互等：rank timer / NCCL / Nsight Systems
├─ GPU 持续忙
│  ├─ 已锁定热点 op：PyTorch Profiler
│  └─ 已锁定热点 kernel：Nsight Compute / Roofline
├─ OOM 或显存逐步上涨
│  ├─ PyTorch allocator：Memory Snapshot
│  └─ Python/native host memory：Memray
├─ 推理延迟随并发爆炸
│  └─ benchmark sweep + scheduler/KV/queue metrics
└─ Slime 一轮很慢
   ├─ wait_time_ratio 高：分析 rollout
   └─ wait_time_ratio 低且 actor_train 慢：分析训练
```


## 8. 性能目标决定证据的粒度

如果目标是用户首 token 延迟，端到端请求、排队和 Prefill 才是直接对象；如果目标是训练 tokens/s/GPU，工作量、整步关键路径和资源预算需要固定。局部算子的 TFLOP/s 是中间解释，不能替代前两种目标。

明确 workload、当前基线和候选层次之后，才知道最小证据是长期指标、系统 trace 还是单 kernel 计数器。没有目标指标，“变快”就缺少可判断的对象。

## 9. 跨层定位：八个系统问题

本节供深入 Kernel 或编译器时回看。DSL 是领域专用语言，codegen 是生成代码，ISA 是指令集架构，cubin 是 GPU 二进制文件；这里新增的工具名表示实现选项，不是完成这条主线前必须另学的课程。

### 从 workload 到通信路径

1. 这是 **workload / 算子抽象** 问题，还是具体 Kernel 实现问题？例如 GEMM / Grouped GEMM 是“要算什么”，cuBLASLt / CUTLASS / Triton 是“怎么实现”。

2. 这是 **代码/编译** 问题，还是 **运行时/调度** 问题？例如 nvcc / Triton compiler / TileLang codegen 与 CUDA Runtime / Driver 不属于同一层。

3. 当前看到的是哪种编译产物？CUDA C++ / Triton DSL 最终可能经历 PTX → cubin/SASS；不要把 CUDA Runtime 当成 GPU ISA。

4. 当前工作粒度是什么？Operation / Kernel / Grid / Block(CTA) / Warp / Thread / Tile 中的哪一层？

5. “谁在调度谁”？Stream 管 operation 的顺序与并发机会；GPU CTA scheduler 管 Block→SM；Warp Scheduler 管 SM 内 Warp→instruction issue。

6. 资源属于谁？Register 属于线程执行状态但物理驻留于 SM；Shared Memory 属于 Block 协作；L2 属于整 GPU；HBM 是设备全局显存；Hopper/Blackwell 还引入 TMA/TMEM 等新数据通路与存储层级。

7. 当前瓶颈是 Tensor Core compute、HBM/片上数据搬运、launch/scheduling、expert load imbalance，还是 GPU 间通信？Grouped GEMM 要特别看每个 Expert 的 `M_i` 分布，而不是只看总 FLOPs。

8. 如果已经跨 GPU / 节点：数据沿 PCIe / NVLink / NVSwitch / HCA / InfiniBand 中的哪条路径走？Collective 的 latency 与 bandwidth 成本分别是什么，能否与 GEMM/Attention overlap？

先看“模型产生了什么 workload”，再看“由哪个 library / DSL / compiler 实现”，然后看“Kernel 如何进入 Stream、Block 如何进入 SM、Warp 如何被 issue”，最后沿着 Register/Shared Memory/HBM/NVLink/IB 追踪数据流。绝大多数 CUDA、FlashAttention、MoE、Megatron、NCCL 与推理 Runtime 问题，都可以挂到 **Workload → Compilation → Scheduling → Resource → Data Movement → Communication** 这六个维度上。

## 参考资料

- [Roofline 原始报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

- [NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- [NVIDIA Deep Learning Performance](https://docs.nvidia.com/deeplearning/performance/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [PyTorch Benchmark Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [SGLang Bench Serving Guide](https://docs.sglang.ai/developer_guide/bench_serving)

<!-- main-route -->

成本模型给出了候选瓶颈；下一篇讨论观测怎样区分原因，以及什么条件下两组数字能比较。

[上一篇：权重、样本与异步一致性](<../11_RL_Systems/权重样本与异步一致性.md>) · [主线 23/26](<../guides/系统阅读路线.md>) · [下一篇：从观测到因果结论](<证据与因果.md>)

<!-- /main-route -->
