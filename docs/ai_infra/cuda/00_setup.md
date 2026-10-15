# 0. 环境与必要的 C++

[返回目录](<README.md>) · [下一章：执行与内存](<01_execution_and_memory.md>)

## 0.1 先认识要安装的三层

普通 C++ 循环由 CPU 执行；将函数写成 CUDA kernel 后，还需要编译出 GPU 能运行的代码，并由 CPU 通过驱动提交。把“编译程序”和“运行程序”拆开，才能理解为什么 GPU 可见、框架可用、源码可编译是三个不同的检查。

host 指负责分配、提交和取回结果的主机端，device 指执行本教程 kernel 的 NVIDIA GPU 端。CUDA Toolkit 是开发工具集合，API 是应用程序编程接口，Runtime API 是程序调用的 `cudaMalloc` 等运行时接口，驱动则负责更底层的设备交互。compute capability（计算能力版本，如 8.0）描述 GPU 的架构功能，不是 Toolkit 的版本号；`sm_80` 是对应的编译目标名称。

| 层 | 作用 | 检查方法 |
|---|---|---|
| NVIDIA 驱动 | 让操作系统与 GPU 通信 | `nvidia-smi` |
| CUDA Toolkit | 提供 CUDA 头文件、库和 `nvcc` 编译器 | `nvcc --version` |
| PyTorch 等框架 | 调用 GPU 库、生成或启动 kernel | 可选：检查 `torch.cuda.is_available()` |

`nvidia-smi` 中的 CUDA Version 表示驱动支持的 CUDA 版本上限信息，不证明安装了对应版本的 Toolkit。PyTorch 能用 GPU，也不代表系统存在 `nvcc`：运行预编译算子与编译 `.cu` 文件是两件事。

本教程选择 **Linux + NVIDIA GPU + C++17**。macOS/Apple Silicon 可阅读全部章节、运行 CPU 推演；Metal/MPS 不能执行这里的 CUDA kernel。有远程 NVIDIA 服务器时，在服务器上编译并运行。不要为了入门直接升级 slime 训练环境，优先使用与已有驱动、项目环境兼容的 Toolkit；安装步骤以 [NVIDIA Linux 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)为准。

在目标 Linux 机器检查：

```bash
nvidia-smi
nvcc --version
```

然后在项目根目录编译第一份完整程序：

```bash
nvcc -O2 -lineinfo -std=c++17 docs/ai_infra/cuda/examples/cuda_basics.cu -o /tmp/slime_cuda_basics
/tmp/slime_cuda_basics
```

`nvcc` 协调 host 与 device 代码的编译；生成的可执行文件仍由 CPU 启动，运行到 kernel launch 时才把工作交给 GPU。程序会查询硬件资源、检查三个基础算子的结果并计时，具体输出在第 1–3 章解释。

`-O2` 开启优化，`-lineinfo` 保留供 profiler 使用的源码行信息，`-std=c++17` 选择 C++ 语言标准，`-o` 指定可执行文件。使用支持 `-arch=native` 的 Toolkit 且编译机器可以看到目标 GPU 时，可额外加 `-arch=native`；跨机器编译应按目标 GPU 的 compute capability 设置 `-arch=sm_XX` 或 `-gencode`，不能把示意字符串 `sm_XX` 原样使用。架构选项见 [nvcc 文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)。

## 0.2 只补当前需要的 C++

Python 的 `a[i] + b[i]` 在 C++ 中也这样写，但 C++ 指针本身不携带长度与形状：

```cpp
void add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
```

逐项解释：`float` 是单精度浮点数，`float*` 是指向浮点数据的地址，`const float*` 表示不能通过这个指针修改数据。`int n` 把数组长度另行传入。`++i` 让下标加一，花括号包住函数或循环体，分号结束语句。这里默认 `a`、`b`、`c` 都至少有 `n` 个元素；越界不会像 Python 一样自动抛出 `IndexError`。

示例在 CPU 上用 `std::vector<float>` 保存数组。`v.size()` 是元素数，`v.data()` 是首元素指针。CUDA 分配和拷贝使用**字节数**，所以 `n` 个 `float` 需要 `n * sizeof(float)` 字节。`size_t` 用来表达非负的对象大小。

例如 `a=[1,2,3]`、`b=[10,20,30]` 都是形状 `[3]` 的数组，`n=3`，每份占 `3×4=12` 字节；结果为 `[11,22,33]`。`a+2` 指向第三个 float，地址相对 `a` 增加 8 字节，而不是 2 字节。`a[i]` 等价于解引用 `*(a+i)`。若把六个数解释成 `[2,3]` 矩阵，第二行第三列在 `a[1*3+2]`；同一个指针也可以被误解释成 `[3,2]`，编译器不会替你发现形状错了。

`const float*` 约束的是通过该指针写入的权限，不承诺其他线程不会修改同一存储。类似地，复制指针只复制地址，不会复制数组，也不会延长数组的生命周期。后续每次传指针，都要同时确认长度、布局和存储是否仍然有效。

| 写法 | 本教程中的含义 |
|---|---|
| `__global__ void f(...)` | 定义由 host 启动、在 GPU 上执行的 kernel，返回值为 `void` |
| `__device__` | 定义在 GPU 上调用的函数 |
| `__host__` | 定义 CPU 函数；普通函数默认是 host 函数 |
| `__shared__ float tile[256]` | 当前 block 内线程共享的暂存数组 |
| `constexpr int B = 256` | 编译期常量；本教程用它固定线程数或 tile 大小 |
| `cond ? x : y` | 条件成立取 `x`，否则取 `y` |
| `step >>= 1` | 正整数 `step` 右移一位，本例等价于除以二 |

这些语法的官方定义见 [CUDA C++ 入门](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)。

## 0.3 CPU 与 GPU 如何连接

```text
CPU 创建输入 → 分配 GPU 显存 → 拷贝输入到 GPU
            → 启动 kernel → 等待所需结果 → 拷贝结果到 CPU → 比较答案
```

`cudaMalloc(&d_a, bytes)` 给设备指针 `d_a` 分配空间；`&d_a` 表示把“保存指针的变量的地址”交给 API，让 API 写回分配结果。`cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice)` 指定目标、来源、字节数和方向。`cudaFree(d_a)` 释放显存。普通独立显存上的 `d_a` 不能直接在 CPU 上用 `d_a[i]` 读取。

统一内存 `cudaMallocManaged` 是另一种编程方式，但它不会消除实际数据移动的成本。本教程先用显式分配与拷贝，让数据位置清楚可见。

沿前面的三个元素算例，可以把资源的生命周期写成下面这张表。H2D（Host-to-Device）是从 CPU 内存拷到 GPU，D2H（Device-to-Host）是反向拷贝。这里使用示例的同步 `cudaMemcpy` 和同一默认 stream（有序任务队列，下一章详述），不涉及跨 stream 并发：

| 时刻 | CPU 内存 | GPU 内存 | 可以进行的下一步 |
|---|---|---|---|
| 创建输入并分配 | a/b 已有值，c 等待结果 | d_a/d_b/d_c 已分配，内容尚未定义 | 拷贝 a/b，各 12 字节 |
| 输入拷贝后 | a/b 保留用于参考 | d_a/d_b 可供后续 kernel 读取 | 提交加法 kernel |
| launch 返回 | CPU 可继续执行其他代码 | kernel 可能尚未完成，d_c 不能视为已就绪 | 提交有序的结果拷贝 |
| D2H 拷贝返回 | c 已包含 `[11,22,33]` | 本次计算和结果读取已完成 | 比较结果，再释放设备分配 |

“变量 `d_a` 还存在”和“它指向的显存还有效”是两件事；释放后保留的地址不能继续使用。更一般的异步程序必须等最后一个 GPU 使用者完成，才能复用或释放它使用的缓冲区。第 1 章用 stream/event 表达这个条件，第 8 章将同一原则应用到 kernel 内的 shared 缓冲区。

示例的 `CUDA_CHECK(...)` 检查每次 Runtime API 调用，失败时打印文件、行号和错误文字后退出。kernel 启动使用特殊语法，随后要单独调用 `cudaGetLastError()`；异步执行错误还需要在同步点检查，下一章会解释。

## 自查

MiB 是二进制单位，1 MiB=2²⁰=1,048,576 字节；MB 是十进制单位，1 MB=1,000,000 字节。下面用字节数除以 1,048,576 得到 MiB。

1. 一百万个 `float` 占多少字节？答：4,000,000 字节，约 3.81 MiB。
2. 为什么只传一个 `float*` 无法知道矩阵宽度？答：指针没有 shape/stride 元数据，需要额外传递。
3. PyTorch 能调用 GPU、`nvcc` 找不到时，缺少什么？答：编译工具可能未安装或不在 PATH，不能据此断言 GPU/驱动失效。
