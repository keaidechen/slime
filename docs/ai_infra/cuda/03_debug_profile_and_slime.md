# 3. 正确性、性能与 slime

[上一章](<02_core_kernels.md>) · [返回目录](<README.md>) · [下一章：行归约与 Softmax](<04_row_softmax.md>)

## 3.1 先证明答案可信

按以下顺序检查：编译与 Runtime API 是否报错；所有输出是否与 CPU 对照相符；边界尺寸是否正常；是否存在越界和数据竞争；最后才看运行时间。浮点加法不满足结合律，归约树和串行循环可能产生不同舍入，不能盲目要求逐 bit 一致。

在 GPU 机器编译后执行：

```bash
compute-sanitizer --tool memcheck /tmp/slime_cuda_basics
compute-sanitizer --tool racecheck /tmp/slime_cuda_basics
compute-sanitizer --tool synccheck /tmp/slime_cuda_basics
```

`memcheck` 查越界、非法地址等；`racecheck` 重点查 shared memory 访问冲突；`synccheck` 查同步原语的错误用法。三者没有报告不等于证明所有并发行为正确，仍要分析索引、数据依赖和所有权。不要使用 Sanitizer 下的程序耗时评价性能。工具行为和要求见 [Compute Sanitizer 文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

| 现象 | 优先检查 |
|---|---|
| `nvcc: command not found` | Toolkit 是否安装、PATH 是否指向它 |
| `no kernel image is available` | 编译的目标架构、设备与 Toolkit 支持范围 |
| `illegal memory access` | 长度、二维 stride、尾块边界、指针所在设备及生命周期 |
| `invalid configuration argument` | block/grid 配置、线程总数、shared memory 请求 |
| 偶尔错、换优化级别就错 | 未初始化数据、竞争、错误的同步假设 |
| 第一次特别慢 | 上下文初始化、JIT、缓存和频率状态 |

需要定位异步错误时可临时运行 `CUDA_LAUNCH_BLOCKING=1 /tmp/slime_cuda_basics`，让报错位置更接近问题启动点；这会改变执行方式，正常计时时去掉。

## 3.2 怎样计时才有意义

H2D（Host-to-Device）是从主机内存向设备内存传输，D2H 是反向传输。提交复制的 CPU 调用返回，与设备已完成复制，是两个不同事件。

示例在输入拷贝完成、正确性检查通过后，预热 10 次，再用同一默认 stream 的 CUDA Events 包住 100 次加法启动；等待 stop event 后读取总毫秒数并除以 100。分配、H2D/D2H 拷贝和 CPU 对照都不在计时范围内。

这是设备时间线上的批量执行间隔：若 CPU 提交不够快，间隔中也可能包含 GPU 等待提交的空隙。因此不要将它解释成纯指令执行周期。短 kernel 应多次独立测量，记录波动；也不要在被 profiler 或 Sanitizer 干扰的运行中取最终基准。

衡量端到端收益则需另设计时区间，把必要的数据准备、拷贝、同步和结果处理都计入。只在 CPU 上测量 `kernel<<<...>>>()` 那一行，通常主要测到提交成本。

有效带宽可用向量加法的 `12*n / seconds / 1e9` 估算，单位 GB/s；这是按算法逻辑字节数定义的指标。本例反复访问相同数组，可能主要命中 cache，不能直接把结果当 HBM 带宽。矩阵乘常用 `2*M*N*K / seconds / 1e12` 估算 TFLOP/s，要同时记录 dtype、累加精度和所用算法。

## 3.3 先看时间线，再看热点 kernel

```bash
nsys profile --trace=cuda,nvtx --sample=none -o /tmp/slime_cuda_timeline /tmp/slime_cuda_basics
nsys stats /tmp/slime_cuda_timeline.nsys-rep
ncu --set basic --kernel-name regex:matmul_tiled --launch-count 1 /tmp/slime_cuda_basics
```

这是工具入门命令。最后一条抓到本程序第一次矩阵乘，即 `1×1×1` 的正确性用例，只用于认识工具界面；做性能实验时应单独固定较大工作负载并预热，或准确筛选目标 launch，不能从这个极小算例推导 GPU 吞吐。

[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 用来看 CPU 提交、拷贝、kernel 与空闲间隙的先后关系。[Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) 用来看特定 kernel 的资源使用与瓶颈；采样可能需要硬件计数器权限，其回放与插桩也会扰动执行。工具不可用时先检查安装版本、驱动支持及服务器权限，不要把 profiler 问题误诊为 kernel 错误。

## 3.4 最值得先掌握的性能判断

pinned memory（页锁定主机内存）在注册期间保持物理页稳定，可供某些异步 DMA 路径使用。它消耗主机资源，是否真正与计算重叠还取决于设备、stream 依赖和缓冲区生命周期。

| 观察或假设 | 可尝试的改变 | 必须核对的代价 |
|---|---|---|
| 访存地址离散 | 调整线程与连续维度的映射 | 布局变换本身是否更贵 |
| 大量重复加载 | shared tile、寄存器复用 | barrier、寄存器/shared 占用 |
| 大量很短的 kernel | 合适的算子融合；固定流程考虑 CUDA Graph | 数值行为、额外驻留状态与图的约束 |
| H2D 拷贝可与计算重叠 | pinned host memory、异步拷贝与多个 stream | 数据依赖、缓冲区生命周期、硬件并发能力 |
| block 太少 | 增加独立工作量或重新分块 | 是否增加冗余计算与合并成本 |

spill（寄存器溢出）指寄存器资源不足等情况下，编译器把部分线程私有值放入 local memory。local 表示线程私有地址空间，底层访问可能经过显存与缓存，因此增加访存成本。

occupancy 指驻留活跃 warp 数相对上限的比例。更多 warp 有时有助于隐藏延迟，但过度减少寄存器会引发 spill；最大 occupancy 不是性能目标。stream 多也不保证加速：同一 GPU 上计算、带宽和容量仍然共享。

把 profiler 结果接回 [第 1 章的执行过程](<01_execution_and_memory.md#launch-journey>)：resident warp 少，先查资源预算与 grid 是否足够大；resident 多但 eligible 少，查依赖、访存和 barrier；有可发射工作仍不快，再看执行管线与带宽瓶颈。`cuda_basics.cu` 打印的 occupancy API 结果是理论容量，不能代替 profiler 的 achieved occupancy，更不是 `nvidia-smi` 的 GPU 利用率。

可用 `时间 ≳ max(计算量/算力, 数据量/带宽)` 做粗略下界判断，再补上启动、同步和依赖开销。这里的算力和带宽必须对应实际 dtype 与存储层级。详细推导及实验方法已有 [CUDA 工具与 Roofline](<../appendices/performance/03_cuda_kernels.md>)，无需在第一遍掌握所有指标。

## 3.5 回到 slime：从能读懂的 kernel 开始

以下链接指向当前工作区源码，后续代码更新时函数名和实现可能变化。

| 入口 | 先追什么 | 对应本教程 |
|---|---|---|
| [INT4 CUDA 实现](<../../../slime/backends/megatron_utils/kernels/int4_qat/fake_int4_quant_cuda.cu>) 的 `int4_quant_1x32_kernel` | `row`、`warp_id`、`lane_id` 如何定位元素；`warpReduceMax` 怎样生成量化 scale | 线程索引、stride、归约、尾部 mask |
| [FP8 Triton 实现](<../../../slime/backends/megatron_utils/kernels/fp8_kernel.py>) 的 `_blockwise_cast_to_fp8_triton` | tile 的行列偏移、masked load、absmax、scale 和写回 | 二维索引、边界填充、局部归约 |
| [FlashAttention 导读](<../appendices/source/flash_attention/README.md>) | Q/K/V 分块、在线 Softmax、shared 与寄存器之间的复用 | tile、归约、同步与数据移动 |
| [slime Profiling](<../../zh/developer_guide/profiling.md>) | 找到 rollout 的真实热点，再决定是否深入 kernel | 系统时间线与热点定位 |

比如 INT4 实现中的 `tid >> 5` 相当于整数除以 32，`tid & 0x1F` 相当于取模 32。`tid=37` 时，`warp_id=1`、`lane_id=5`；首次循环 `item=1`，所以 `col=1*32+5=37`。再沿 `x[row*stride_xm + col*stride_xn]` 找到它读取的地址，就已经能把线程分工与量化值连起来。

Triton 的一个 program 处理一个 tile，但 `tl.arange` 的一个元素不能简单理解成一个固定 CUDA 线程；具体映射由编译器和布局决定。CUDA 基础帮你理解约束，而不是把两套语法逐字替换。

写 PyTorch 自定义算子时，还需处理 device、dtype、shape/stride、当前 CUDA stream、输入输出生命周期和 autograd。理解三个基础 kernel 后再进入扩展工程；本教程独立程序的默认 stream 用法不能原样代替框架中的 stream 管理。
