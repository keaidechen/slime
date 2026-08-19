# 08 工具箱与中英文资料索引

本章不是让你把所有工具都学一遍，而是让你按问题选最小工具。前半部分是工具地图，后半部分是经过筛选的一手资料和博客。建议先完成实验，再按遇到的问题查阅对应资料。

## 1. 一张工具选择表

| 层级 | 工具 | 最适合回答 | 典型输出 | 开销/风险 |
|---|---|---|---|---|
| 业务/框架 | 原生日志、TensorBoard、W&B | step、reward、tokens/s、阶段耗时是否异常 | scalar/曲线 | 低 |
| 服务 | SGLang/vLLM benchmark | TTFT、TPOT、ITL、吞吐和容量拐点 | JSON/JSONL | 低至中 |
| GPU 设备 | `nvidia-smi dmon`、DCGM | 利用率、显存、功耗、频率是否异常 | 时间序列 | 低 |
| CPU/OS | `time`、`pidstat`、`vmstat`、`iostat`、`perf` | CPU、调度、缺页、I/O 是否阻塞 | 文本/record | 低至中 |
| Python CPU | `cProfile`、`py-spy` | Python 函数或线程卡在哪里 | 表/火焰图 | 中；采样较轻 |
| PyTorch | `torch.utils.benchmark` | 单个 Python/C++ operator 的可靠 microbenchmark | 统计表 | 低 |
| PyTorch | PyTorch Profiler/Kineto | 哪个 ATen op、kernel、shape 或分配最耗时 | trace/table | 中至高 |
| CUDA 系统 | Nsight Systems | CPU、CUDA、NCCL、多进程怎样等待 | `.nsys-rep` | 中 |
| CUDA kernel | Nsight Compute | 单个 kernel 为什么慢 | `.ncu-rep` | 高；会 replay |
| CUDA 正确性 | Compute Sanitizer | 越界、race、未初始化访问 | 错误报告 | 很高 |
| CUDA 二进制 | `cuobjdump`、`nvdisasm` | 最终 cubin/PTX/SASS 是什么 | 汇编/元数据 | 低 |
| Triton | interpreter、`device_print`、dump 环境变量 | 自定义 Triton kernel 的值、IR 和汇编 | console/IR | 低至高 |
| 显存 | PyTorch Memory Snapshot/Memory Viz | allocator 可见 tensor、峰值和碎片 | pickle/交互图 | 中；历史很大 |
| 主机内存 | Memray | Python/native 分配和泄漏 | `.bin`/火焰图 | 中至高 |
| 通信 | NCCL logs、PyTorch flight recorder、`nccl-tests` | collective hang、带宽、rank 长尾 | 日志/dump | 低至中 |
| 在线观测 | Prometheus/Grafana | 长时间容量、队列、错误和资源趋势 | metrics/dashboard | 低 |

经验法则：先用低开销工具缩小范围，再用高开销工具验证。Profiler 产生的数据不是无扰动观察；开 profiler 后的吞吐不能替代关闭 profiler 时的基线。

## 2. 按症状选择工具

### 2.1 “GPU 利用率低”

1. `nvidia-smi dmon` 看是否持续低，还是采样错过短 burst；
2. 框架指标看请求/数据是否足够；
3. `pidstat`/`py-spy` 看 CPU 供给；
4. Nsight Systems 找 GPU gap 前的依赖；
5. 只有热点 kernel 自身慢时才用 Nsight Compute。

### 2.2 “GPU 利用率高但吞吐低”

1. 确认 workload 的 token、shape、精度和有效输出相同；
2. PyTorch Profiler 看 op 组成是否变化；
3. Nsight Systems 看通信、重复计算或错误 kernel；
4. Nsight Compute 判断 memory/compute/latency bound；
5. 检查 GPU clock、功耗限制和温度。

### 2.3 “偶发卡住或 collective 超时”

1. 保留每 rank 带时间戳日志；
2. 开适量 `NCCL_DEBUG`，不要永久使用最详细级别；
3. 使用 PyTorch ProcessGroupNCCL flight recorder；
4. 在同样节点拓扑跑 `nccl-tests`；
5. Nsight Systems 对齐最先偏离的 rank；
6. 检查数据、MoE token、CPU/I/O 和硬件 straggler，不要先认定是 NCCL bug。

### 2.4 “OOM，但 allocated 看起来还有空间”

1. 同时记录 allocated、reserved 和设备总显存；
2. PyTorch Memory Snapshot 查 allocator 内部；
3. 对比 `nvidia-smi`，估计 allocator 外部分配；
4. 检查 NCCL、CUDA Graph、KV Cache、第三方 kernel workspace；
5. 主机内存增长另用 Memray，不要混为 CUDA OOM。

### 2.5 “推理 p99 很高，平均值正常”

1. 保存每请求输入/输出长度和到达时间；
2. 对齐 queue、batch、KV Cache 和错误/retry；
3. SGLang sample trace 找长尾 tool/reward；
4. 短 profile 比较正常请求和长尾请求；
5. 分开网络/client、排队、prefill 和 decode。

## 3. 资料阅读方法

每篇资料只带着一个问题读：

1. 它测量的边界是什么？
2. 输出里的时间、计数器或指标怎样定义？
3. 工具引入了多大扰动？
4. 示例 workload 与你的 workload 有什么不同？
5. 哪条结论可以通过单变量实验验证？

旧博客最容易过时的是 CLI 参数、环境变量和默认行为；最不容易过时的是测量方法、时间线因果关系和硬件基本原理。命令始终以当前 `--help` 和官方文档为准。

## 4. 中文资料：建议阅读顺序

以下优先选择项目或厂商官方中文资料。

### 4.1 Slime：先理解完整 RL 系统

1. [Slime 简介（官方中文博客）](https://thudm.github.io/slime/blogs/introducing_slime.html)

   先读系统定位、训练/推理解耦、colocate 和 debug 模式。它帮助你理解为何 RL 性能不能只看 actor TFLOPS。

2. [本仓库 Slime 性能分析指南](../zh/developer_guide/profiling.md)

   对应 `sleep_rollout`、router `/workers` 和 `tools/profile_rollout.py`，可直接跟着第 6 章操作。

3. [本仓库 Debug 指南](../zh/developer_guide/debug.md)

   重点读 rollout/train 分离、固定数据回放、第一步精度检查和非法显存访问排查。

4. [本仓库 rollout trace 文档](../zh/developer_guide/trace.md)

   用于 sample、tool、reward、prefill/decode 的端到端长尾分析。

### 4.2 NVIDIA：从系统时间线进入 CUDA

1. [使用 Nsight Systems 优化 CUDA 内存传输](https://developer.nvidia.cn/blog/optimizing-cuda-memory-transfers-with-nsight-systems/)

   适合第一次理解 H2D/D2H、pageable/pinned memory、copy 与 compute overlap。

2. [使用 NVIDIA Nsight Systems 加速数据中心和 HPC 性能分析](https://developer.nvidia.cn/zh-cn/blog/accelerating-data-center-and-hpc-performance-analysis-with-nvidia-nsight-systems/)

   了解系统级时间线、NVTX 和多节点/多进程分析思路。

3. [在 NVIDIA Grace Hopper 上分析 LLM 训练工作流](https://developer.nvidia.cn/zh-cn/blog/profiling-llm-training-workflows-on-nvidia-grace-hopper/)

   重点学习如何把 PyTorch Profiler、Nsight Systems 和 Nsight Compute 串起来，不要照搬特定硬件数字。

4. [CUDA 性能优化指南（中文）](https://developer.nvidia.cn/blog/cuda-performance-guide-cn/)

   用作内存层级、并发和优化原则的中文补充。具体工具界面以最新官方手册为准。

5. [CUDA 开发者工具教程视频系列](https://developer.nvidia.cn/blog/new-video-series-cuda-developer-tools-tutorials/)

   适合更偏视频学习的人，用来熟悉 Nsight 的基本操作。

## 5. 英文资料：PyTorch 基础

建议按以下顺序阅读，并在每篇后完成对应实验。

1. [PyTorch Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

   学 `profile`、activities、operator table 和 trace。对应实验 2。

2. [PyTorch Benchmark Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)

   学 warmup、线程和同步正确的 microbenchmark，避免用普通 Python timer。

3. [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

   把 profiler 发现的问题映射到常见训练优化。不要一次打开所有建议；逐项 A/B。

4. [PyTorch Profiler API](https://docs.pytorch.org/docs/stable/profiler.html)

   查 schedule、trace handler、shape、stack、memory、FLOPs 等精确定义和开销。

5. [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)

   学 snapshot 与 Memory Viz，并理解 PyTorch 不可见的 CUDA 分配。

6. [PyTorch Dispatcher Tutorial](https://docs.pytorch.org/tutorials/advanced/dispatcher.html) 与 [Custom Operators Manual](https://docs.pytorch.org/docs/main/library.html)

   用于理解 operator schema、dispatch key、CPU/CUDA/autograd 注册，解决“Python API 到底调用哪个实现”。

7. [Profiling `torch.compile`](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)、[`torch._logging`](https://docs.pytorch.org/docs/stable/logging) 与 [Troubleshooting](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html)

   用于 graph break、recompile、编译时间和生成 kernel。日志很大，只开启与当前问题相关的类别。

## 6. 英文资料：CUDA 与底层算子

1. [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

   查 capture range、CUDA Graph、multiprocess、sampling、NVTX 和 CLI 参数。对应实验 3 的系统层。

2. [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 与 [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)

   学 replay、section、Speed of Light、Roofline、occupancy 和 warp stall。不要从完整训练直接 `--set full`。

3. [GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)

   补齐 SM、Tensor Core、内存层级、算术强度和 latency hiding 的硬件背景。

4. [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)

   查 `cuobjdump`、`nvdisasm`、PTX/SASS、line info。用于 kernel 名已知后的二进制下钻。

5. [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

   用 memcheck、racecheck、initcheck 查正确性。它不是测速工具，运行会极慢。

6. [Triton Debugging](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html) 与 [Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

   学 interpreter、`device_assert`/`device_print`、`do_bench` 和自定义 matmul 的性能验证。

7. [Triton 官方仓库的编译与 dump 提示](https://github.com/triton-lang/triton)

   查 `TRITON_KERNEL_DUMP`、`TRITON_DUMP_DIR`、MLIR/LLVM dump 等当前支持方式；环境变量会随版本演进。

## 7. 英文资料：Megatron 与分布式通信

1. [Megatron Bridge Parallelisms Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)

   第一次系统理解 TP、PP、DP、CP、EP 和组合约束。

2. [Megatron Bridge Profiling](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/profiling.html)

   给出 PyTorch Profiler 和 Nsight Systems 的框架集成方式；参数以当前 Bridge 版本为准。

3. [Megatron Bridge Performance Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html)

   将并行、batch、precision、activation recomputation 和性能联系起来。

4. [Megatron Communication Overlap](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/communication-overlap.html)

   重点是从时间线验证 overlap，而不是仅确认开关为真。

5. [Megatron Core Timers](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.timers.html) 与 [Straggler Detector](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.utils.html)

   用于阶段计时、rank 间偏差和慢节点排查。

6. [NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)、[NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) 与 [nccl-tests](https://github.com/NVIDIA/nccl-tests)

   用于先验证硬件/网络 collective 基线，再解释框架通信。不要把 debug 环境变量永久写入所有任务。

7. [PyTorch ProcessGroupNCCL Environment Variables](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html) 与 [Flight Recorder Tutorial](https://docs.pytorch.org/tutorials/prototype/flight_recorder_tutorial.html)

   用于 collective hang 和 desynchronization 的环形缓冲证据。

## 8. 英文资料：SGLang 与 vLLM

### 8.1 SGLang

1. [SGLang Bench Serving Guide](https://docs.sglang.ai/developer_guide/bench_serving)

   参数化 workload、request rate、并发、数据集和输出指标，是容量实验的主参考。

2. [SGLang Benchmark and Profiling](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md)

   查当前 profiler API、活动类型和 trace 输出。

3. [SGLang Observability](https://docs.sglang.ai/advanced_features/observability.html)

   查 Prometheus metrics、request dump/replay 和生产观测。

4. [SGLang v0.4: Zero-Overhead Batch Scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)

   这是一个很好的“博客如何使用 trace 证明调度优化”的例子。重点看实验设计和时间线，不照搬版本数字。

### 8.2 vLLM

1. [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)

   查 `vllm bench latency/serve/throughput` 的当前参数和指标。

2. [Profiling vLLM](https://docs.vllm.ai/en/stable/contributing/profiling/)

   查当前 PyTorch profiler、Nsight Systems、multiprocessing 与 CUDA Graph 采集方式。旧博客中的环境变量可能已过时。

3. [vLLM Performance Dashboard](https://docs.vllm.ai/en/latest/benchmarking/dashboard/)

   学如何持续跟踪 commit、硬件和 workload 下的回归；不要把公开 dashboard 数字直接当作自己环境的容量。

## 9. 其他工具的一手入口

- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html)：查询与 `dmon` 字段定义。
- [NVIDIA DCGM Profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)：集群级 GPU 遥测和计数器。
- [Perfetto UI](https://ui.perfetto.dev/) 与 [Perfetto Documentation](https://perfetto.dev/docs/)：打开 Chrome/PyTorch trace、SQL 分析时间线。
- [Memray Documentation](https://bloomberg.github.io/memray/)：Python/native 内存追踪与火焰图。
- [py-spy](https://github.com/benfred/py-spy)：低侵入 Python 采样、线程栈和火焰图。
- [Linux perf](https://perf.wiki.kernel.org/index.php/Main_Page)：CPU PMU、采样和调度分析。
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)：指标采集、label 与查询；避免高基数请求级 label。
- [TensorBoard Profiler](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)：查看 profiler trace 的一种 UI；PyTorch 也可生成 TensorBoard trace handler 输出。

## 10. 资料可信度与版本记录

本目录按 2026-08-19 的官方资料和当前 Slime 仓库整理，遵循以下优先级：

```text
当前本地源码与 --help
  > 对应版本官方文档
  > 项目维护者官方博客
  > 厂商技术博客
  > 社区博客和论坛经验
```

社区博客适合发现关键词，不适合直接复制生产参数。采用任何资料中的建议时，在性能报告中记录：链接、阅读日期、适用版本、你实际验证的 A/B 结果。
