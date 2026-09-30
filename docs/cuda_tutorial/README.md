# 从零开始写 CUDA

面向会写基本 Python、尚未写过 GPU kernel 的读者。先沿一次 kernel 执行，把 CUDA 分工、GPU 硬件与数据存储连起来，再亲手实现基础算子、Softmax、普通与在线 Attention，最后用 Tensor Core、CuTe 和异步流水小实验接上真实 FA2 前向。本文及配套示例为本项目新编的教学材料；基础与 Attention 实验不依赖第三方仓库，CuTe 小实验另需 CUTLASS 头文件。

## 阅读路线

| 顺序 | 章节 | 完成标志 |
|---|---|---|
| 0 | [环境与必要的 C++](00_setup.md) | 区分驱动、Toolkit、PyTorch；读懂指针和 kernel 启动 |
| 1 | [把 CUDA 程序放到真实 GPU 上理解](01_execution_and_memory.md) | 连起 grid/block/warp 与 SM/调度器/执行管线，追踪数据路径并计算驻留预算 |
| 2 | [三个核心 kernel](02_core_kernels.md) | 读懂并运行加法、归约、分块矩阵乘；知道每次 barrier 的原因 |
| 3 | [正确性、性能与 slime](03_debug_profile_and_slime.md) | 用 CPU 对照、Sanitizer、Event 和 profiler 验证结论；继续读项目 kernel |
| 4 | [行归约与 Softmax](04_row_softmax.md) | 理解 shuffle、跨 warp 归约、稳定化与全 mask 行 |
| 5 | [普通 Attention](05_materialized_attention.md) | 亲手接通 QK → Softmax → PV，核对 shape、stride 和 mask |
| 6 | [分块在线 Attention](06_online_attention.md) | 推导并验证 m/l/a 更新，解释旧输出缩放和块顺序 |
| 7 | [Tensor Core 与 CuTe](07_mma_and_cute.md) | 理解 C++ 模板、warp 协作、矩阵片段与坐标布局 |
| 8 | [异步双缓冲](08_async_pipeline.md) | 画出缓冲区的加载、就绪、消费、复用时序 |
| 9 | [读一条固定 FA2 前向路径](09_read_fa2.md) | 将教学实现对应到真实 kernel，推演线程 37 与两轮 K/V |

按 **0–3 基础 → 4–6 算法 → 7–8 协作与流水 → 9 FA2 源码** 阅读。每章先手算，再运行例子，最后完成修改练习；程序打印 PASS 不能替代自己解释结果。目标是读通固定长度、无 dropout 的 FA2 前向，暂不要求理解所有版本与所有分支。

完成后继续读 [FlashAttention 源码导读](../flash_attention_code_walkthrough/README.md) 的完整摘录；概念查询见 [AI Infra 百科](../../learn_docs/README.md)，工具操作见 [性能分析指南](../performance_analysis_guide/README.md)。本套侧重渐进实践，已有导读侧重固定提交的真实实现。

全程使用同一条解释顺序：**这段代码由谁执行 → 落到哪个硬件作用域 → 数据放哪里 → 谁需要等待谁 → 哪种资源限制并行度**。第 1 章建立总图，后续每个算子都回到这组问题；硬件介绍与 CUDA 语法不再分开阅读。

## 本地练习

在项目根目录运行无需 GPU 的算法推演：

```bash
python3 docs/cuda_tutorial/examples/cpu_walkthrough.py
python3 docs/cuda_tutorial/examples/attention_cpu.py
```

在装有 NVIDIA GPU、驱动和 CUDA Toolkit 的 Linux 机器运行完整 CUDA 实验：

```bash
nvcc -O2 -lineinfo -std=c++17 docs/cuda_tutorial/examples/cuda_basics.cu -o /tmp/slime_cuda_basics
/tmp/slime_cuda_basics
```

程序先打印实际设备资源、`matmul_tiled` 的编译资源用量与理论驻留容量，再检查多组边界尺寸、打印三个 `PASS`，随后打印向量加法的平均 kernel 时间。容量估算不是实测调度或吞吐；时间取决于机器，本文不提供虚构的基准数字。若编译器默认目标与 GPU 不兼容，按 [环境章节](00_setup.md)设置目标架构。

Attention 的完整 CUDA 实验：

```bash
nvcc -O2 -lineinfo -std=c++17 docs/cuda_tutorial/examples/attention_lab.cu -o /tmp/slime_attention_lab
/tmp/slime_attention_lab
```

| 示例 | 对应章节 | 验证内容与依赖 |
|---|---|---|
| [cpu_walkthrough.py](examples/cpu_walkthrough.py) | 1–2 | Python 标准库；教学资源预算、二维线程编号、归约和 tiled GEMM |
| [cuda_basics.cu](examples/cuda_basics.cu) | 1–3 | CUDA；实际硬件/编译资源查询、基础算子和 Event 计时 |
| [attention_cpu.py](examples/attention_cpu.py) | 4–7、9 | Python 标准库；在线状态、错误反例、120 组分块/遍历顺序对照与布局覆盖 |
| [attention_lab.cu](examples/attention_lab.cu) | 4–6 | CUDA；warp/block Softmax、普通/在线 Attention，逐元素对照 FP64 CPU 参考 |
| [wmma_tile.cu](examples/wmma_tile.cu) | 7 | 兼容 WMMA 的 GPU/Toolkit；FP16 输入、FP32 累加的小矩阵乘 |
| [cute_layout.cu](examples/cute_layout.cu) | 7 | CUDA Toolkit + CUTLASS；host 上的布局/转置视图检查，不启动 GPU kernel |
| [async_tiles.cu](examples/async_tiles.cu) | 8 | SM80 或更新兼容 GPU/Toolkit；双缓冲、尾块补零和跨线程读 |

第 7/8 章给出架构与依赖要求、具体命令；第 4/8 章给出 Sanitizer 检查命令。新 GPU 示例共用 [lab_common.cuh](examples/lab_common.cuh) 的错误报告和数值检查。所有输入生成、参考结果和 PASS 条件都在源码中，没有省略 host 驱动。

**验证范围**：本机已运行两份 Python 推演，包括在线状态反例和坐标覆盖；文档内 C++ 模板小程序已在本机编译运行通过。编写时本机没有 `nvcc`、NVIDIA GPU 运行环境，CUTLASS 嵌套子模块也尚未初始化，因此所有 `.cu` 示例的编译、设备正确性、Sanitizer 和性能数据仍需在目标机器验证。CPU 推演不验证 GPU 并发语义，教学在线 Attention 也不代表 FA2 的性能实现。

## 值得配合使用的公开教程与开源代码

以下入口于 2026-09-30 查阅。阅读顺序是本教程的建议，不代表上游的课程分级。

| 资料 | 用法与首读范围 | 许可与定位 |
|---|---|---|
| [NVIDIA Modern CUDA C++ 教程](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/README.md) | 有 notebook、讲义和练习；配合阅读 execution spaces、kernels、streams、shared memory | 公开教学资源；根目录 [LICENSE](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/LICENSE) 为 CC BY-NC-SA 4.0，含非商业限制，不能视为任意用途的宽松开源代码 |
| [NVIDIA cuda-samples](https://github.com/NVIDIA/cuda-samples) | 先查 `vectorAdd`，再读 `matrixMul`、`reduction`，用于核对完整 API 使用 | 开源示例集；根目录 [LICENSE](https://github.com/NVIDIA/cuda-samples/blob/master/LICENSE) 为 BSD 三条款；具体文件保留自身声明 |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | 完成本教程后看 `dev/cuda` 中逐步优化的算子，再读训练主线 | [MIT](https://github.com/karpathy/llm.c/blob/master/LICENSE)；面向 LLM 的进阶实现，不适合作为第一份 CUDA 程序 |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | 首读 Part 1 与 Part 2，遇到具体语义再查后续章节 | 官方参考文档，免费在线阅读；不是本项目附带的开源教材 |

希望把上游源码放到本目录时，可从项目根目录执行以下**可选**命令；本次整理没有执行下载，也没有添加 submodule：

```bash
git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git docs/cuda_tutorial/upstream/cuda-samples
git clone --depth 1 https://github.com/karpathy/llm.c.git docs/cuda_tutorial/upstream/llm.c
git -C docs/cuda_tutorial/upstream/cuda-samples rev-parse HEAD
git -C docs/cuda_tutorial/upstream/llm.c rev-parse HEAD
```

保留上游许可证并记录实际 commit。上游构建方式、硬件要求随版本变化，按所下载版本的 README 操作；这里只提供入口，不承诺最新分支兼容你的 Toolkit。
