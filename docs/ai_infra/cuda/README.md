# CUDA 算子实现：从执行模型到 FlashAttention

一次 kernel 如何分工、数据放在哪里、谁必须等待谁，是本专题的共同线索。文章通过完整代码和正文算例，依次解释基础算子、Softmax、在线 Attention、Tensor Core、CuTe 与异步流水，最终对应到固定版本的 FA2 前向路径。

## 阅读路线

| 顺序 | 章节 | 核心问题 |
|---|---|---|
| 0 | [环境与必要的 C++](00_setup.md) | 区分驱动、Toolkit、PyTorch；读懂指针和 kernel 启动 |
| 1 | [把 CUDA 程序放到真实 GPU 上理解](01_execution_and_memory.md) | 连起 grid/block/warp 与 SM/调度器/执行管线，追踪数据路径并计算驻留预算 |
| 2 | [三个核心 kernel](02_core_kernels.md) | 理解加法、归约、分块矩阵乘；知道每次 barrier 的原因 |
| 3 | [正确性、性能与 slime](03_debug_profile_and_slime.md) | 用 CPU 对照、Sanitizer、Event 和 profiler 验证结论；继续读项目 kernel |
| 4 | [行归约与 Softmax](04_row_softmax.md) | 理解 shuffle、跨 warp 归约、稳定化与全 mask 行 |
| 5 | [普通 Attention](05_materialized_attention.md) | 追踪 QK → Softmax → PV，核对 shape、stride 和 mask |
| 6 | [分块在线 Attention](06_online_attention.md) | 推导 m/l/a 更新，解释旧输出缩放和块顺序 |
| 7 | [Tensor Core 与 CuTe](07_mma_and_cute.md) | 理解 C++ 模板、warp 协作、矩阵片段与坐标布局 |
| 8 | [异步双缓冲](08_async_pipeline.md) | 画出缓冲区的加载、就绪、消费、复用时序 |
| 9 | [读一条固定 FA2 前向路径](09_read_fa2.md) | 将教学实现对应到真实 kernel，推演线程 37 与两轮 K/V |

按 **0–3 执行基础 → 4–6 算法 → 7–8 协作与流水 → 9 FA2 源码** 连续阅读。推导与反例已在正文展开，示例程序作为辅助材料。通用术语见 [硬件与系统](../03_Hardware_Systems/README.md)，完整源码分析见 [FlashAttention 系列](../13_Source_Analysis/flash_attention/README.md)，测量解释见 [性能分析](../12_Performance_Reliability/README.md)。

## 配套实现及边界

| 示例 | 对应章节 | 验证内容与依赖 |
|---|---|---|
| [cpu_walkthrough.py](examples/cpu_walkthrough.py) | 1–2 | Python 标准库；教学资源预算、二维线程编号、归约和 tiled GEMM |
| [cuda_basics.cu](examples/cuda_basics.cu) | 1–3 | CUDA；实际硬件/编译资源查询、基础算子和 Event 计时 |
| [attention_cpu.py](examples/attention_cpu.py) | 4–7、9 | Python 标准库；在线状态、错误反例、120 组分块/遍历顺序对照与布局覆盖 |
| [attention_lab.cu](examples/attention_lab.cu) | 4–6 | CUDA；warp/block Softmax、普通/在线 Attention，逐元素对照 FP64 CPU 参考 |
| [wmma_tile.cu](examples/wmma_tile.cu) | 7 | 兼容 WMMA 的 GPU/Toolkit；FP16 输入、FP32 累加的小矩阵乘 |
| [cute_layout.cu](examples/cute_layout.cu) | 7 | CUDA Toolkit + CUTLASS；host 上的布局/转置视图检查，不启动 GPU kernel |
| [async_tiles.cu](examples/async_tiles.cu) | 8 | SM80 或更新兼容 GPU/Toolkit；双缓冲、尾块补零和跨线程读 |

GPU 示例共享 [错误报告和数值检查](examples/lab_common.cuh)。编译、架构与依赖说明放在对应章节；基础算子与 Attention 不依赖其他仓库，CuTe 需要 CUTLASS 头文件。

CPU 参考覆盖算法状态和坐标关系，不能验证 GPU 并发语义。`.cu` 示例尚未在本次整理中编译或实测，教学实现也不代表 FA2 的性能实现。

## 值得配合使用的公开教程与开源代码

以下入口于 2026-09-30 查阅。阅读顺序是本教程的建议，不代表上游的课程分级。

| 资料 | 用法与首读范围 | 许可与定位 |
|---|---|---|
| [NVIDIA Modern CUDA C++ 教程](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/README.md) | 有 notebook、讲义和练习；配合阅读 execution spaces、kernels、streams、shared memory | 公开教学资源；根目录 [LICENSE](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/LICENSE) 为 CC BY-NC-SA 4.0，含非商业限制，不能视为任意用途的宽松开源代码 |
| [NVIDIA cuda-samples](https://github.com/NVIDIA/cuda-samples) | 先查 `vectorAdd`，再读 `matrixMul`、`reduction`，用于核对完整 API 使用 | 开源示例集；根目录 [LICENSE](https://github.com/NVIDIA/cuda-samples/blob/master/LICENSE) 为 BSD 三条款；具体文件保留自身声明 |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | 完成本教程后看 `dev/cuda` 中逐步优化的算子，再读训练主线 | [MIT](https://github.com/karpathy/llm.c/blob/master/LICENSE)；面向 LLM 的进阶实现，不适合作为第一份 CUDA 程序 |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | 首读 Part 1 与 Part 2，遇到具体语义再查后续章节 | 官方参考文档，免费在线阅读；不是本项目附带的开源教材 |


[返回总入口](../README.md)
