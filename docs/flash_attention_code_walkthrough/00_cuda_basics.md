# 00 · CUDA 入门：谁计算，数据放哪里

CUDA 是 NVIDIA GPU 的编程平台；本系列用到它的 C++ 扩展语法。CPU 侧代码常称 **host**，GPU 侧代码常称 **device**。读完本节，你应能解释下一段代码中的每个部分：

```cpp
kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
```

这是 [`flash_fwd_launch_template.h`](../../flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h) 中真实的启动语句。先不用理解 `Kernel_traits`，把它看作一组已经选好的配置。

## 1. CPU 发任务，GPU 执行 kernel

Python 函数通常在 CPU 上执行。它可以创建 GPU 张量，再通过 CUDA 接口提交 GPU 工作。**kernel** 是一次在 GPU 上由许多线程执行的函数。一次 `flash_attn_func(...)` 会经过 Python 和 C++，最终启动 kernel；这不意味着 Python 解释器在 GPU 上运行。

一个 CUDA **thread（线程）**拥有自己的编号和局部状态。许多线程执行同一份函数，但利用编号处理不同数据。对 `y[i] = x[i] * 2`，最简单的分工是一条线程处理一个 `i`。

下面是**教学 CUDA 片段**，省略内存分配、输入拷贝、错误处理和结果拷回，不能作为完整程序独立运行：

```cpp
__global__ void times_two(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = 2.0f * x[i];
    }
}
// 假设 x、y 已是有效的 GPU 指针，n = 1000。
times_two<<<4, 256>>>(x, y, n);
```

`__global__` 声明可从 CPU 启动的 GPU 函数。`float*` 是一段浮点数内存的起始地址，`x[i]` 通过地址找到第 i 个值。`const` 表示这个函数不通过该指针修改输入。

4 个 block × 每块 256 个线程 = 1024 个线程。block 0 的线程 0 处理元素 0；block 1 的线程 0 处理元素 256；block 3 的线程 231 处理元素 999。最后 24 个线程由 `i < n` 挡住，不访问越界内存。以后看到 attention 的边界检查，可以先想起这个例子。

## 2. thread、warp、block、grid 与 SM

| 名称 | 含义 | 在这个例子中 |
|---|---|---|
| thread | 一个执行 kernel 的线程 | 负责一个数组元素 |
| warp | NVIDIA GPU 上一组 32 个线程 | 256 线程的 block 有 8 个 warp |
| block / CTA | 可以使用共享内存协作的一组线程 | 256 个线程构成一块 |
| grid | 一次 kernel 启动中的全部 block | 本次有 4 块 |
| SM | GPU 中承载线程块执行的硬件单元 | 调度这些 block 的地方 |

**block 是任务组织，SM 是执行硬件**。一个 SM 可同时驻留多个 block，也可能因资源限制只驻留较少的 block。grid 可以远大于 SM 数目，任务分批执行。普通 kernel 不能假定 block 0 一定先于 block 1 完成。

要把这些术语放进同一次执行过程，先读 [CUDA 实践第 1 章：程序如何落到真实 GPU](../cuda_tutorial/01_execution_and_memory.md)。其中将 block→SM、warp→调度器、指令→执行管线和数据→存储层次画在同一张图里，并用线程数、寄存器和 shared 的资源预算解释驻留限制；后面的矩阵乘、Softmax 和 FA2 都沿用这套对应关系。

`threadIdx.x` 是块内线程编号；`blockIdx.x` 是网格中的块编号。`blockDim.x` 是每块线程数，`gridDim.x` 是块数。CUDA 允许用 x/y/z 三个轴编号，目的是方便表达多维任务；三维 grid 并不表示 GPU 上存在对应的三维物理摆放。

以上执行模型可对照 [NVIDIA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)。FlashAttention 中每块线程共同处理一个矩阵块，不再是一线程独占一个最终输出。

## 3. 三种主要存储位置

| 存储 | 谁使用 | FlashAttention 中的典型内容 |
|---|---|---|
| global memory（由设备内存支持，访问可以命中 cache） | 持有有效地址的设备线程可访问 | 输入 Q/K/V，最终输出 O |
| shared memory（共享内存） | 普通场景下同一 block 的线程共享 | 即将反复使用的 Q/K/V tile |
| register（寄存器） | 每个线程持有自己的值 | 局部 score、输出累加片段、softmax 统计量 |

L1/L2 cache 也会影响实际访问速度；这里的三分法用于理解程序显式安排的数据，不代表硬件只有这三层。寄存器资源不足时可能发生 spill，将部分值放到有显存开销的存储中。

shared 不是 global load 必须经过的一层 cache，寄存器也不是每条线程永久独占的一颗硬件核心内部的小盒子。物理寄存器文件属于 SM，编译与执行为线程分配其中的资源；数据落点和访问作用域应分开判断。

把数据放入共享内存需要一次搬运，是否划算取决于后续复用。矩阵乘法里，一个 Q 元素会参与许多不同 K 的点积；同一 K/V tile 又会被多个 Q 行使用。这就是搬进片上存储后反复计算的机会。

这里的 **tile（数据块）**是一块矩形数据，例如 Q 的 128 行、64 列；**thread block（线程块）**是一组执行任务的线程。一个线程块可以依次处理多个数据块；遇到“块”时先分清它指数据还是线程。

### 3.1 张量、精度与字节数

**tensor（张量）**在本文可以理解为带形状的多维数组：一个数是标量，一列数是向量，二维表是矩阵。`shape=(2,3)` 表示两行三列；`dtype` 表示每个元素按什么数值格式存储。

| dtype | 每元素字节数 | 本系列中的用途 |
|---|---:|---|
| FP16 / BF16 | 2 | 常见 Q/K/V 输入；两者位数相同、数值表示规则不同 |
| FP32 | 4 | 常见 score、输出累加值和 Softmax 统计量 |
| FP64 | 8 | 实验中的高精度参考计算 |
| int32 | 4 | 长度、索引等整数数据 |

“FP32 累加”表示将乘积加到 FP32 结果里，并不表示输入也变成 FP32。浮点数只有有限精度，因此数学上等价的计算顺序可能产生不同的末尾数位。`inf` 表示无穷大，`NaN` 表示无效数值结果，例如 `0/0`。内存计量中 `1 KiB=1024` 字节，`1 MiB=1024²` 字节。

**stride（步长）**描述某个下标增加 1 时跨过多少元素。连续二维数组 `[[a,b,c],[d,e,f]]` 按 `a,b,c,d,e,f` 存储，行步长为 3、列步长为 1。`contiguous` 表示满足相应的连续排布规则；转置视图可能保留同一份数据而改变步长。第 03 章会把这个例子推广到 Q/K/V。

## 4. 为什么要同步

设线程 0 往共享内存写一个值，线程 1 要读它。如果没有约定，线程 1 可能先读到旧值。CUDA 的 `__syncthreads()` 是同一 block 内的协作屏障：参与的线程到齐，并建立相关内存可见性后再继续。它不会等待整个 grid。见 [CUDA C++ 入门](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)。

概念上的协作顺序如下：

```text
线程们各搬一部分输入到 shared memory
                    ↓
             block 内同步
                    ↓
线程们读取彼此搬来的数据，完成各自计算
                    ↓
       确认读取结束，再复用这块 shared memory
```

不能把需要全块参与的屏障放进只有部分线程进入的条件分支。异步拷贝还需要等待相应搬运完成；单独出现一个 `__syncthreads()` 不等于所有类型的异步操作都已完成。第 03 节会追源码中的等待与复用顺序。

## 5. 看懂 `<<<grid, threads, shared_bytes, stream>>>`

这四个位置依次表示：启动多少 block、每个 block 多少线程、每块申请多少字节动态共享内存、将任务放在哪条 CUDA stream。

**stream** 可以先理解为 GPU 任务队列。同一 stream 中的操作按其排序规则衔接；CPU 提交 kernel 后通常不需要立即等它算完。因此测 GPU 时间不能只测一次 Python 调用耗时，第 06 节会解释计时方法。

回到 FlashAttention：

```cpp
const int num_m_block =
    (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
dim3 grid(num_m_block, params.b, params.h);
```

这是 `run_flash_fwd` 的源码摘录。`dim3` 保存三个非负整数。Q 有 `seqlen_q` 行，每块处理 `kBlockM` 行，向上取整得到所需块数，再乘 batch 数和 head 数。这里分的是**Q 行块**，每个 block 内部还要扫描所需 K/V 块。

## 6. 读 C++/CUDA 时先认识这些符号

| 写法 | 零基础读法 |
|---|---|
| `__device__` | 这个函数在 GPU 上执行，由 GPU 代码调用 |
| `__global__` | kernel 入口，可以从 CPU 发起一次并行启动 |
| `__shared__` | 同一个 block 的线程共享的存储 |
| `extern __shared__ char smem_[]` | 启动时指定字节数的动态共享内存区域 |
| `template<int D>` | 用编译时参数 D 生成专门版本，例如 D=64 |
| `constexpr` | 这个值可在编译时确定 |
| `if constexpr (...)` | 编译时选择分支，未选中的分支不生成执行代码 |
| `typename` / `using` | 声明、引用一个类型；第一次可把别名展开成自然语言 |
| `Element` / `ElementAccum` | 本实现中的类型别名；主线例子分别为 FP16 元素和 FP32 累加值 |
| `auto` | 让编译器推断变量类型 |
| `params.q_ptr` | 参数对象中的 Q 数据指针 |
| `Tensor& acc_o` | 引用调用者的对象，函数内修改会影响调用者看到的结果 |
| `cute::copy` / `Kernel_traits::kNThreads` | `::` 选择某个命名空间或类型中的函数/成员 |
| `condition ? a : b` | 条件成立取 a，否则取 b，类似 Python 的 `a if condition else b` |
| `&&` / `\|\|` / `!` | 逻辑与、或、非；前两者会短路求值 |
| `nullptr` | 空指针，常表示没有提供某个可选数组 |
| `static_assert(...)` | 编译时检查条件，不满足就无法编译该实例 |
| `#pragma unroll` | 提示编译器展开循环；是否有利需结合代码规模和资源评估 |

这些写法不要求一次掌握全部 C++。第一遍读代码，只要能区分**地址、形状、数据值、编译时配置**，就可以沿主线继续。

例如 `softmax.template softmax_rescale_o<true, false>(acc_s, acc_o, scale)` 应分两层读：尖括号里的布尔值选择编译版本，圆括号里传入当前数据。成员访问后的 `template` 是 C++ 告诉编译器如何解析模板名的语法。源码中的 `/*Is_first=*/true` 只是给 `true` 加了注释，并非 Python 式关键字实参。`[&] { ... }` 是引用外部变量的匿名函数；分发宏用它包住一段待执行代码。

## 7. 自查

- 1000 个元素、每块 256 线程，需要几块？**4 块**，末块有越界保护。
- 128 个线程有几个 warp？**4 个**，不是 128 个硬件核心。
- 100 个 block 是否必须等于 100 个 SM？**不必**，block 会被硬件调度。
- `__syncthreads()` 能否让另一个 block 的输出就绪？**不能**，其作用域是当前 block。

下一节先把 GPU 放在一边，用四个数字理解 attention 分块为什么成立。
