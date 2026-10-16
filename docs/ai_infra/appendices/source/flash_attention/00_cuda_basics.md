# 00 · 读 FA2 启动代码：线程、存储与模板约定

本篇把[GPU 执行模型](<../../../03_Execution/07_硬件与存储层次.md#s2>)映射到 FA2 源码中的名字。已经理解 thread、warp、block 和内存层次时，可以重点看启动配置与 C++ 模板；需要完整小程序时再查 [CUDA 算子实现](<../../../cuda/README.md>)。

```cpp
kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
```

这是 [`flash_fwd_launch_template.h`](<../../../../../flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h>) 中的启动语句。`Kernel_traits` 表示已选配置，`params` 携带指针、长度和 stride；下面解释这些量怎样对应一次执行。

## 1. CPU 提交与设备线程

普通程序可能用两层循环逐行、逐列计算矩阵；GPU 要同时处理许多这样的任务，所以程序必须说明“启动多少执行者、每个执行者负责哪些下标”。**CUDA** 是 NVIDIA GPU 的编程平台；**kernel（核函数）**是一次由许多 GPU 线程共同执行的函数。CPU 是发出工作请求的主机处理器，GPU 是承接并行计算的设备，两边各有执行进度。

Python 经 C++/CUDA 接口提交 kernel，解释器并不在 GPU 上运行。CUDA 线程利用编号处理不同数据，例如一维索引 `i=blockIdx.x*blockDim.x+threadIdx.x`。`__global__` 标记 kernel 入口，`float*` 表示浮点数据地址，`const` 限定经该指针的写入。

教学例中 1000 个元素、每 block 256 线程，需要 4 个 block：block 1 的线程 0 对应元素 256，block 3 的线程 231 对应元素 999，最后 24 个线程由 `i<n` 保护。FA2 中边界保护承担同样的职责，但线程协作处理矩阵 tile，不再是一线程独占一个最终输出。完整索引与尾块示例见[基础 kernel](<../../../cuda/02_core_kernels.md>)。

## 2. 从 grid 到 SM 的对应

Grid 是一次启动的所有 block；每 block 的 256 个线程形成 8 个 warp，128 个线程则形成 4 个 warp。Block 是逻辑任务，SM 是承载它的硬件，一个 SM 的驻留量受线程、寄存器与 shared memory 预算限制。Grid 可以远大于 SM 数，普通 kernel 也不能假定 block 按编号完成。

**warp（线程束）**在本系列 NVIDIA 架构中包含 32 条线程；**SM（Streaming Multiprocessor，流式多处理器）**拥有调度、计算和片上存储资源。一个普通 block 的线程会在同一 SM 上协作，不同 block 即使碰巧在同一 SM 上，也不能直接把彼此的普通 shared memory 当作公共数组。线程是程序执行状态，不等于一条线程永久占用一个运算单元；warp 等数据时，SM 可以调度其他已经就绪的 warp。

例如假设设备只有 2 个 SM，每个 SM 因资源限制最多驻留 1 个 block，而 grid 有 5 个 block。最初只能容纳其中 2 个；有 block 完成并归还资源后，才能继续安排剩余任务。不能在 block 0 中忙等 block 4 写一个标志来代替 grid 同步：block 4 可能还没有机会运行。跨 block 的阶段依赖通常交给不同 kernel 和 stream 排序处理，第 04 篇反向就是这样的例子。

`threadIdx`、`blockIdx` 分别定位块内线程与 grid 内的块，`blockDim`、`gridDim` 给出各维大小。x/y/z 是逻辑编号轴，不代表 GPU 的三维物理摆放。[执行与内存算例](<../../../cuda/01_execution_and_memory.md>)展开对应的资源预算。

## 3. 三种主要存储位置

| 存储 | 谁使用 | FlashAttention 中的典型内容 |
|---|---|---|
| global memory（由设备内存支持，访问可以命中 cache） | 持有有效地址的设备线程可访问 | 输入 Q/K/V，最终输出 O |
| shared memory（共享内存） | 普通场景下同一 block 的线程共享 | 即将反复使用的 Q/K/V tile |
| register（寄存器） | 每个线程持有自己的值 | 局部 score、输出累加片段、softmax 统计量 |

L1/L2 cache 也会影响实际访问速度；这里的三分法用于理解程序显式安排的数据，不代表硬件只有这三层。寄存器资源不足时可能发生 spill，将部分值放到有显存开销的存储中。

shared 不是 global load 必须经过的一层 cache，寄存器也不是每条线程永久独占的一颗硬件核心内部的小盒子。物理寄存器文件属于 SM，编译与执行为线程分配其中的资源；数据落点和访问作用域应分开判断。

把数据放入共享内存需要一次搬运，是否划算取决于后续复用。矩阵乘法里，一个 Q 元素会参与许多不同 K 的点积；同一 K/V tile 又会被多个 Q 行使用。这就是搬进片上存储后反复计算的机会。

这里的显存常由 **HBM（High Bandwidth Memory，高带宽内存）**提供；它容量大，但读取需要穿过比片上寄存器、共享内存更长的数据路径。假设两个 query 都需要同一条含 4 个数的 K 行：各自独立加载共提出 8 个元素的读取需求，协作搬一次到 shared 后可在片上分别使用。实际 HBM 流量还受缓存影响，这个算例解释的是软件可见的复用机会，并不声称一定减少一半物理显存事务。

这里的 **tile（数据块）**是一块矩形数据，例如 Q 的 128 行、64 列；**thread block（线程块）**是一组执行任务的线程。一个线程块可以依次处理多个数据块；遇到“块”时先分清它指数据还是线程。

### 3.1 形状、步长与精度

本系列常见 Q/K/V 使用每元素 2 字节的 FP16/BF16，累加值和 Softmax 状态使用 4 字节 FP32；CPU 数学参考使用高精度浮点，长度和索引常使用 4 字节 int32。FP32 累加不改变输入 dtype，等价计算顺序也可能因舍入产生不同末位。FP64 为 8 字节，`inf`/`NaN` 分别表示无穷与无效结果；KiB/MiB 使用 1024 的幂计量。

`shape` 描述大小，`stride` 描述下标增加 1 时跨过的元素数。连续 `[2,3]` 数组的 stride 为 `[3,1]`；转置视图能只改元数据而保留底层存储。FA2 的地址计算必须结合实际 stride，不能只看数学形状。通用定义见[张量与布局](<../../../02_Model_Computation/03_张量形状与数值表示.md#s1>)，Q/K/V 的具体地址在第 03 章展开。

以 FP16 数组 `[[10,11,12],[20,21,22]]` 为例，元素 `(1,2)` 的元素偏移为 `1×3+2×1=5`，字节偏移为 `5×2=10`。这两种偏移的单位不同：形状与 stride 用来定位第几个元素，最后再由数据类型决定占多少字节。FP 是 floating point（浮点）；BF16 是 bfloat16。16 位格式能表示的数有限，FP32 累加可减少求和舍入，但无法恢复输入存储时已经舍掉的细节。

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

用两条线程搬 `shared=[a,b]` 来检查这个规则：线程 0 写 a，线程 1 写 b，随后两条线程都要读 a+b。第一次屏障防止读到尚未写好的 b；如果接着要把同一数组改放 `[c,d]`，还要确认两条线程都读完 a/b，才能覆盖。前者回答“新数据到了吗”，后者回答“旧数据用完了吗”。同步点本身不复制数组，也不释放数组；它建立允许后续读写发生的先后关系。

## 5. 看懂 `<<<grid, threads, shared_bytes, stream>>>`

这四个位置依次表示：启动多少 block、每个 block 多少线程、每块申请多少字节动态共享内存、将任务放在哪条 CUDA stream。

**stream** 可以先理解为 GPU 任务队列。同一 stream 中的操作按其排序规则衔接；CPU 提交 kernel 后通常不需要立即等它算完。因此测 GPU 时间不能只测一次 Python 调用耗时，第 06 节会解释计时方法。

假设 CPU 依次在同一 stream 提交“准备 Q → attention → 使用 O”，GPU 会按依赖顺序处理，CPU 无需在每一步后停下来。若改用不同 stream，程序就需要显式事件等依赖，不能把 Python 的调用先后当成跨队列的数据就绪证明。CPU 函数返回、kernel 执行完、另一条 stream 可以消费 O，是三个要分别判断的时刻。

存储的寿命也有两层。block 完成后，其 shared memory 和寄存器资源可由硬件供后续 block 复用；Q/K/V/O 的显存张量则仍由调用者或框架管理。`ctx` 为反向保留的张量还不能提前覆盖；即便 Python 名字被删除，分配器也必须遵守尚未完成的设备访问。优化时所谓“复用缓冲区”，指旧读者全部结束后才能换用途，而非看到函数返回就立刻改写。

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

线程分工解释了由谁计算，仍未解释为什么能只保存一个局部 tile。相关资料用四个 key 的数值推导在线 Softmax，再将状态更新对应到源码。
