# CUDA 算子实现：从执行模型到 FlashAttention

一次 kernel 如何分工、数据放在哪里、谁必须等待谁，是本专题的共同线索。文章通过完整代码和正文算例，依次解释基础算子、Softmax、在线 Attention、Tensor Core、CuTe 与异步流水，最终对应到固定版本的 FA2 前向路径。

读者只需会变量、循环、数组和函数。kernel 是提交给 GPU 执行的函数；本专题要解释的是：怎样把原本一个循环完成的工作分给许多线程，同时保留正确的结果。CUDA 是 NVIDIA 的并行计算平台与编程模型；FlashAttention 是一类减少 Attention 中间数据读写的实现，FA2 指 FlashAttention-2。

建议按 0→9 阅读：前四篇建立执行、存储和测量方法；4–6 篇先算清 Softmax 与 Attention，再改变它们保存中间量的方式；7–9 篇才引入硬件矩阵指令、布局和流水。遇到模板名时，先辨认它描述的是形状、线程分工还是内存布局，再沿代码找实际执行的操作。

阅读时先建立三个区别：**算法**规定算什么，**线程分工与布局**规定谁处理哪些数据、如何寻址，**硬件资源与同步**决定这些工作怎样被承接和推进。各小节先说明当前概念是什么、解决什么问题，再用算例、代码或时序展开；已学概念在后续章节按需回顾。编译、验证和源码导读小节则先交代要验证或追踪什么，再列命令与文件。

全文用 `A[M,K]` 表示形状为 M 行、K 列的张量，下标从 0 开始；`@` 表示矩阵乘，`*` 在标量公式中表示普通乘法。shape 是各维长度，stride 是该维下标增加 1 时跨过的**元素数**；字节地址还需乘元素大小。`ln`/`log` 默认是自然对数，`exp(x)=e^x`；`log2` 与 `exp2` 才使用底数 2。FP16/FP32/FP64 分别指 16/32/64 位浮点格式，在这里每个元素占 2/4/8 字节。维度、索引、概率和 score 都没有物理单位；带宽与容量公式会单独标明字节、秒及其倍数。

每篇的手算小例子用来检查数值和所有权，不代表性能实验规模。没有 NVIDIA GPU 时，可先执行下面两个标准库脚本，再把正文的中间值对上输出：

```bash
python3 docs/ai_infra/cuda/examples/cpu_walkthrough.py
python3 docs/ai_infra/cuda/examples/attention_cpu.py
```

CUDA 编译命令从第 0 章开始给出，后续新示例在相应章节单独说明。读完一段代码，应能说清输入形状、每个输出由谁写、谁需要等待，以及最后一个读者结束后哪块存储可以复用。

## 实验材料

下面是按依赖顺序排列的章节入口。“核心问题”说明每章要建立的理解；可执行示例和环境要求另见后面的材料表。

| 材料编号 | 实验说明 | 核心问题 |
|---|---|---|
| 0 | [环境与必要的 C++](<00_setup.md>) | 区分驱动、Toolkit、PyTorch；读懂指针和 kernel 启动 |
| 1 | [把 CUDA 程序放到真实 GPU 上理解](<01_execution_and_memory.md>) | 连起 grid/block/warp 与 SM/调度器/执行管线，追踪数据路径并计算驻留预算 |
| 2 | [三个核心 kernel](<02_core_kernels.md>) | 理解加法、归约、分块矩阵乘；知道每次 barrier 的原因 |
| 3 | [正确性、性能与 slime](<03_debug_profile_and_slime.md>) | 用 CPU 对照、Sanitizer、Event 和 profiler 验证结论；继续读项目 kernel |
| 4 | [行归约与 Softmax](<04_row_softmax.md>) | 理解 shuffle、跨 warp 归约、稳定化与全 mask 行 |
| 5 | [普通 Attention](<05_materialized_attention.md>) | 追踪 QK → Softmax → PV，核对 shape、stride 和 mask |
| 6 | [分块在线 Attention](<06_online_attention.md>) | 推导 m/l/a 更新，解释旧输出缩放和块顺序 |
| 7 | [Tensor Core 与 CuTe](<07_mma_and_cute.md>) | 理解 C++ 模板、warp 协作、矩阵片段与坐标布局 |
| 8 | [异步双缓冲](<08_async_pipeline.md>) | 画出缓冲区的加载、就绪、消费、复用时序 |
| 9 | [读一条固定 FA2 前向路径](<09_read_fa2.md>) | 将教学实现对应到真实 kernel，推演线程 37 与两轮 K/V |

本目录是第 8—9 章的配套实验资料，编号用于对应示例文件，不增加正文的阅读步骤。通用术语见 [硬件与系统](<../appendices/hardware/README.md>)，完整源码分析见 [FlashAttention 系列](<../appendices/source/flash_attention/README.md>)，测量解释见 [性能分析](<../appendices/performance/README.md>)。

## 配套实现及边界

**配套实现**用于把正文的公式、坐标或依赖关系变成可检查的结果。CPU 脚本检查算法与索引，CUDA 程序检查目标 GPU 上的实际执行，二者验证范围不同。先根据手头环境选择材料，再按对应章节运行。

| 示例 | 对应章节 | 验证内容与依赖 |
|---|---|---|
| [cpu_walkthrough.py](<examples/cpu_walkthrough.py>) | 1–2 | Python 标准库；教学资源预算、二维线程编号、归约和 tiled GEMM |
| [cuda_basics.cu](<examples/cuda_basics.cu>) | 1–3 | CUDA；实际硬件/编译资源查询、基础算子和 Event 计时 |
| [matmul_tiled.html](<examples/matmul_tiled.html>) | 2 | 浏览器；完整 A/B/C 的输入输出映射、逐项累加、合作加载与 kernel 逐段解释，无需 GPU |
| [attention_cpu.py](<examples/attention_cpu.py>) | 4–7、9 | Python 标准库；在线状态、错误反例、120 组分块/遍历顺序对照与布局覆盖 |
| [attention_lab.cu](<examples/attention_lab.cu>) | 4–6 | CUDA；warp/block Softmax、普通/在线 Attention，逐元素对照 FP64 CPU 参考 |
| [wmma_tile.cu](<examples/wmma_tile.cu>) | 7 | 兼容 WMMA 的 GPU/Toolkit；FP16 输入、FP32 累加的小矩阵乘 |
| [cute_layout.cu](<examples/cute_layout.cu>) | 7 | CUDA Toolkit + CUTLASS；host 上的布局/转置视图检查，不启动 GPU kernel |
| [async_tiles.cu](<examples/async_tiles.cu>) | 8 | SM80 或更新兼容 GPU/Toolkit；双缓冲、尾块补零和跨线程读 |

GPU 示例共享 [错误报告和数值检查](<examples/lab_common.cuh>)。编译、架构与依赖说明放在对应章节；基础算子与 Attention 不依赖其他仓库，CuTe 需要 CUTLASS 头文件。

CPU 参考覆盖算法状态和坐标关系，不能验证 GPU 并发语义。`.cu` 示例尚未在本次整理中编译或实测，教学实现也不代表 FA2 的性能实现。

## 值得配合使用的公开教程与开源代码

这些是补充教程、实现样本与官方参考：教程适合循序练习，参考文档用于核对具体语义，性能实现用于后续比较设计。阅读本专题不要求先读完这些资料，遇到相应问题再查指定入口即可。

以下入口于 2026-09-30 查阅，按需要核对接口、实现和依赖。

| 资料 | 用法与首读范围 | 许可与定位 |
|---|---|---|
| [NVIDIA Modern CUDA C++ 教程](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/README.md) | 有 notebook、讲义和练习；配合阅读 execution spaces、kernels、streams、shared memory | 公开教学资源；根目录 [LICENSE](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/LICENSE) 为 CC BY-NC-SA 4.0，含非商业限制，不能视为任意用途的宽松开源代码 |
| [NVIDIA cuda-samples](https://github.com/NVIDIA/cuda-samples) | 先查 `vectorAdd`，再读 `matrixMul`、`reduction`，用于核对完整 API 使用 | 开源示例集；根目录 [LICENSE](https://github.com/NVIDIA/cuda-samples/blob/master/LICENSE) 为 BSD 三条款；具体文件保留自身声明 |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | 完成本教程后看 `dev/cuda` 中逐步优化的算子，再读训练主线 | [MIT](https://github.com/karpathy/llm.c/blob/master/LICENSE)；面向 LLM 的进阶实现，不适合作为第一份 CUDA 程序 |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | 首读 Part 1 与 Part 2，遇到具体语义再查后续章节 | 官方参考文档，免费在线阅读；不是本项目附带的开源教材 |


[返回总入口](<../README.md>)
