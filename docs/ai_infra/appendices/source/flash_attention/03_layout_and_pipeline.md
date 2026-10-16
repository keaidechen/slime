# 03 · 读懂 CuTe 布局、线程分工与流水


本章先用 §1–9 解释视图、片段和流水，再在 §10 代入线程编号。第一次精读只需算明白线程 37 的例子；swizzle 的位运算和完整 CuTe 模板实现可以留待实际编写 kernel 时学习。

## 1. 先区分“数据在哪里”和“如何给数据编号”

一段内存只是按地址排列的数。shape 描述逻辑形状，stride 描述每个坐标轴移动一步时，地址增加多少个元素。一个普通 `2×3` 行优先矩阵的 stride 是 `(3,1)`，元素 `(i,j)` 的偏移是 `3*i+j`。

把同一段内存看成 `3×2` 的转置视图，可以使用 shape `(3,2)`、stride `(1,3)`。此时只改变索引方式，没有发生数据搬运。某些复杂布局转换需要真实的拷贝，必须查看是否出现 `copy` 等操作，不能只从名字中的 `transpose` 判断。

给底层六个数编号为 `[10,11,12,20,21,22]`：原视图 `(1,0)` 的偏移为 `1×3+0=3`，读到 20；转置视图 `(0,1)` 的偏移为 `0×1+1×3=3`，也读到 20。若仅把 shape 改成 `(3,2)`，却仍使用新的连续 stride `(2,1)`，则 `(0,1)` 会读到 11——那是按不同方式重新分组，不是矩阵转置。所谓 **layout（布局）**，就是从逻辑坐标到存储偏移的映射；CuTe 还用布局表达“哪个线程持有哪些坐标”。

本系列的连续输入 `Q.shape=(2,256,4,64)`，以元素为单位的 stride 是 `(65536,256,64,1)`：

```text
q[b,t,h,c] 的偏移 = b*65536 + t*256 + h*64 + c
```

例如 `q[0,128,2,0]` 偏移为 32896。`mQ(_,bidh,_)` 选定一个 head 后，相邻 token 仍隔着 256 个元素，不能假设间距自动变成 64。

## 2. `make_tensor` 和 `local_tile` 在这里建立视图

以下是 [`flash_fwd_kernel.h`](<../../../../../flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h>) 的**源码摘录**，`mQ` 已由指针、shape 和 stride 构成：

```cpp
Tensor gQ = local_tile(mQ(_, bidh, _),
                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                      make_coord(m_block, 0));
```

把它读成：“固定 head，在整份 Q 上选第 `m_block` 个 `kBlockM×kHeadDim` 的矩形。”CuTe 的 `_` 表示保留该维度，类似 Python 切片的 `:`。此时只是建立对 global memory 数据的 tile 视图；创建 `gQ` 不等于已经把 Q 搬到共享内存。

`Int<128>`、`Shape<...>` 是 C++ 类型中携带的编译时整数和形状。CuTe 利用这些信息生成地址计算和指令布局。先按“已知大小的矩形”理解即可，稍后再研究模板如何实现。

CuTe 的 `_0{}`、`_1{}` 等是编译时整数对象，分别表示 0、1；`Shape<_8,_64>{}` 表示编译时确定的 `8×64` 形状。`size<1>(tensor)` 取第 1 个维度的大小，`get<0>(coord)` 取坐标的第 0 项。形状还可以嵌套，例如 `((2,2),2,16)`：`size<0,1>` 表示第一组的第二项。后面的 mask 用这种层次把一个寄存器片段拆成行坐标和列坐标。

## 3. 一张数据流图解释变量名前缀

```mermaid
flowchart LR
    A[显存 gQ / gK / gV] -->|global 到 shared 拷贝| B[共享内存 sQ / sK / sV]
    B -->|按 MMA 操作数布局读取| C[线程寄存器 fragment]
    C --> D[acc_s: QK 的局部结果]
    D --> E[在线 softmax 与 rP]
    E --> F[acc_o: PV 的累加结果]
    F -->|归一化和写回| G[显存 O 与 LSE]
```

| 变量 | 怎么读 | 是整块还是线程视角 |
|---|---|---|
| `gQ` | global Q tile 的视图 | 整个逻辑 tile |
| `sQ` | shared Q tile 的视图 | 当前 block 的共享数据 |
| `tQgQ` | Q 拷贝任务在 global 端的线程分片 | 当前线程负责搬的那部分 |
| `tQsQ` | 同一拷贝任务在 shared 端的线程分片 | 当前线程负责写的那部分 |
| `tSrQ` | score GEMM 使用的 Q 寄存器片段 | 当前线程的操作数 |
| `acc_s` | score 累加片段 | 当前线程持有的一部分 score |
| `acc_o` | output 累加片段 | 当前线程持有的一部分输出 |

前缀是本实现的命名习惯，不是 CUDA 语言规则。`Tensor acc_o` 看起来像一个完整矩阵，但它实际代表每条线程所拥有的 accumulator fragment（累加器片段）。所有线程的片段合起来才覆盖逻辑输出 tile。

## 4. 搬运分工与计算分工可以不同

源码先创建 `gmem_tiled_copy_QKV`，调用 `get_thread_slice(tidx)`，然后用 `partition_S(gQ)` 与 `partition_D(sQ)` 确定当前线程的源和目标。这里 S/D 是 source/destination，与 attention 的 score 矩阵 S 无关。

随后 `tiled_mma.get_thread_slice(tidx)` 建立**矩阵乘法的线程分工**。同一条线程可以负责搬运一组相邻元素，却在计算时负责另一种排列的片段。原因是显存访问希望相邻线程访问相邻地址，而 Tensor Core 的指令对输入/输出片段有规定的布局。

[`kernel_traits.h`](<../../../../../flash-attention/csrc/flash_attn/src/kernel_traits.h>) 中本例的 `kGmemElemsPerLoad = sizeof(uint128_t)/sizeof(Element) = 16/2 = 8`：一次拷贝单元可携带 8 个 FP16 元素。`kGmemThreadsPerRow=64/8=8`，所以搬运一行 64 个元素可以让 8 条线程各搬连续的 8 个。每条线程会随布局迭代搬多份数据，不能据此认为 128 个线程只搬 128 个元素。

这种安排追求 **coalescing（合并访存）**：一次指令中各线程的地址尽量集中，让硬件用较少的内存事务服务它们。用一个假设的“每事务覆盖 32 字节、起点对齐”模型，8 条线程各读一个连续 FP32 数，地址覆盖 32 字节，可落在一个区间；若每条线程相隔 128 字节，则落在 8 个区间。两者都只要 8 个数，搬运效率却可能很不同。真实事务还取决于指令宽度、架构和缓存；本例仅解释地址聚集为何有价值，不能用它推算 FA2 的精确事务数。

## 5. Tensor Core 的小块如何拼成大 tile

MMA 指令完成小矩阵的乘加 `C += A @ B`。本例 traits 选择 `SM80_16x8x16_F32F16F16F32_TN`，可以先读作：使用特定硬件矩阵乘原语，输入为 FP16，结果用 FP32 累加。名字中的 `16x8x16` 是该操作的 M/N/K 尺寸，**不是整个 attention tile 的尺寸**。

矩阵乘通用记号是 `A[M,K] @ B[K,N] → C[M,N]`，其中 K 表示被求和的维度。它恰好与 attention 的 Key 都写成 K，但两者含义不同：

| 当前矩阵乘 | M：结果行数 | N：结果列数 | K：求和长度 |
|---|---:|---:|---:|
| QK，计算 score | 128 个 query | 128 个 key | 64 个特征 |
| PV，累加输出 | 128 个 query | 64 个输出特征 | 128 个 key |

`kBlockM/kBlockN` 在这个前向 traits 中分别固定 Q/KV 的块高；它们不会随着第二次乘法自动改名。

traits 的 `TiledMma` 用这些小操作覆盖 `128×128` 的 score 块以及 `128×64` 的输出块。`Layout<Shape<Int<kNWarps>,_1,_1>>` 将 warp 布局沿 M 方向组织；一个 warp 可以在多个子块上重复执行 MMA。

从纯矩阵覆盖数来算，QK 要覆盖 `128/16=8` 组 M 子块、`128/8=16` 组 N 子块，并沿特征累加 `64/16=4` 份，合计 `8×16×4=512` 个这种基本 MMA 操作。PV 则为 `(128/16)×(64/8)×(128/16)=512` 个。这里数的是整个 block 逻辑工作中的 warp 级基本操作，不是单条线程独立算 512 个小矩阵；实际指令分发由 `TiledMma` 与编译结果决定。

沿 Q 行把工作分给 warp，能让不同 warp 持有不同输出行；它们复用 K/V，却不必把不同 warp 对同一输出行的部分和搬到 shared 再合并。同一行内部仍可能跨 lane 归约，所以“减少 warp 之间的通信”不等于“没有线程间通信”。后面的四线程 Softmax 正是剩余的一类协作。

`utils.h` 的 [`gemm` 与 `gemm_rs`](<../../../../../flash-attention/csrc/flash_attn/src/utils.h>) 帮助理解这一层：内部按归约维度循环，加载下一组寄存器片段，再调用 `cute::gemm`。`gemm_rs` 的一个操作数已经在寄存器中，适合 softmax 之后直接使用 `rP`。这解释了为什么局部 score 无需写回 HBM 再被下一次矩阵乘读取。

## 6. 为什么 shared memory 使用 swizzle

共享内存内部有多个 **bank（存储分区）**。多个不同地址的请求集中到同一 bank 时，可能需要分多次服务，称为 bank conflict；同地址广播等情况需另行区分。**swizzle** 是对地址映射作有规律的重排，让后续线程访问更适合硬件布局；逻辑矩阵的元素值没有改变。

`kernel_traits.h` 的 `SmemLayoutAtomQ` 把 `Swizzle<...>` 与基础行列布局组合。global 端按连续地址读取，shared 端可以按适合 MMA 读取的布局存放。`copy` 根据源/目标布局，将同一个逻辑坐标的元素放到正确位置；物理地址被重排不表示 Q/K/V 的数学行列被随意交换。

可以用一个只有 4 个 bank 的**教学模型**手算。假设每个标量占一个存储字，bank 编号为 `字偏移 % 4`，普通 `4×4` 行优先矩阵的 `(r,c)` 偏移为 `4r+c`。四条线程同时读第 0 列，偏移是 `[0,4,8,12]`，都去 bank 0。改用 `4r+(c XOR r)` 存储，其中 XOR 是逐位异或，则这一列偏移变成 `[0,5,10,15]`，分散到 bank 0、1、2、3。每一行中 `c XOR r` 只是对 0–3 的置换，因此所有元素仍能唯一定位。

这是解释重排目的的小模型，不是源码 `Swizzle` 参数的展开。真实共享内存的 bank 数、字宽和 Tensor Core 加载方式要按目标架构解释。最关键的约束是：写入与读取必须使用匹配的映射；只改写入布局，读取仍按 `4r+c`，会把不同元素当成原来的元素，造成数值错误。

`sVt` 通过 `sV.data()` 和转置布局建立另一个视图。它没有因此新增一份完整 V。把“布局视图”“片段分配”“真实搬运”区分开，就能避免看到很多 Tensor 名字时误判显存用量。

## 7. 片上资源也会用完

本例 `Share_Q_K_smem=false`，Q/K/V tile 均为 `128×64` 个 FP16 数：

```text
sQ = 128*64*2 = 16384 字节
sK = 128*64*2 = 16384 字节
sV = 128*64*2 = 16384 字节
合计 = 49152 字节 = 48 KiB / block
```

这是由该配置的 `kSmemQSize`、`kSmemKVSize`、`kSmemSize` 得出的动态共享内存量。score、输出累加器还占用寄存器等资源；不要把上面三项当成整个 kernel 的全部资源消耗。

spill（寄存器溢出）指寄存器资源不足等情况下，编译器把部分线程私有值放入 local memory。local 表示线程私有地址空间，底层访问可能经过显存与缓存，因此增加访存成本。

tile 越大，往往复用越多，但每块需要的资源也越多，可能减少同时驻留的 block 数，甚至引发寄存器 spill。**occupancy（占用率）**通常指活跃 warp 数相对硬件可容纳上限的比例。它不是越高越快的单一指标；数据复用、指令效率和等待时间同样重要。

例如假设某 SM 可供此 kernel 使用 96 KiB shared、65536 个 32 位寄存器，warp 上限为 64。每 block 用 48 KiB shared、128 线程，并假设编译后每线程用 200 个寄存器，那么 shared 限制为 `floor(96/48)=2` 块，寄存器限制为 `floor(65536/(128×200))=2` 块。仅按这些约束可驻留 2 块，即 8 个 warp，占 warp 上限的 `8/64=12.5%`。这是忽略分配粒度和其他限制的预算练习，不是某款 GPU 的实测占用率。

如果把 shared 用量增至每块 64 KiB，shared 一项就把驻留上限降到 1 块。单块复用增加可能节省搬运，却也可能使 SM 在等待内存时没有足够的其他 warp 可执行。相反，只为了提高占用率而缩小 tile，又可能增加 K/V 重读和循环开销。优化必须同时看资源预算和实际等待原因。

## 8. 沿源码看一次异步搬运

`compute_attn_1rowblock` 中的 `cp_async_fence` 与 `cp_async_wait<N>` 管理异步 global→shared 拷贝组。这里的 fence 是该拷贝流水的提交边界，不应直接当作“所有内存操作完成”。`wait<0>` 等待相关已提交拷贝完成；`wait<1>` 允许保留一个较新的待完成组，具体安全性依赖此前提交顺序。

**pipeline（流水）**把多块数据的阶段交错安排；**prefetch（预取）**就是当前计算还没结束时，提前发起以后要用的数据加载。`cp.async` 是这里异步 global→shared 搬运使用的指令机制。发起异步操作后线程可继续执行，真正消费结果前再等待。

下面是本例主循环的**简化时序**，用于理解源码，省略 layout、边界和寄存器预取细节：

```text
先把 Q 和首个 K tile 放入 shared
等待本轮 K 可用
发起本轮 V 拷贝
计算本轮 QK                  ← 可与 V 的搬运重叠
等待 V 可用，并完成必要的 block 同步
发起下一轮 K 拷贝            ← 复用已经消费完的 K 缓冲区
本轮在线 softmax 与 PV       ← 可与下一轮 K 的搬运重叠
进入下一轮，等待所需 K 就绪
```

为什么覆写 sK 前安全性很重要？如果某些线程还在读旧 K，另一些线程已写入新 K，结果会依赖执行时机，产生 data race（数据竞争）。所以同步既要保证“生产完成后才能读”，也要保证“消费完成后才能覆盖”。源码里一对等待与同步经常同时承担这两种衔接需求，应结合前后读写检查。

`wait<N>` 中的 N 是允许仍在途的已提交组数上限，不是 tile 编号。假设依次提交 G0、G1，`wait<1>` 允许较新的 G1 仍未结束，但保证较旧的 G0 已结束；它不授权读取 G1 的目标。即使组已经完成，也仍要核对其他线程搬来的共享数据是否经过必要的同步。FA2 主线采用的 `wait<0>` 更容易理解：先完成当前线程相关的全部已提交组，再做块内会合。[PTX 的 cp.async 组与等待语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group)给出了这类约束。

流水能隐藏多少等待也可以估算。若一块数据加载用 3 个时间单位、计算用 5 个单位，处理两块且完全串行要 `2×(3+5)=16`；允许加载第二块与第一块计算重叠，则为 `3+5+5=13`。重叠没有减少 6 个单位的搬运工作，只减少了暴露在关键路径上的等待。必须有可并行的硬件工作、可用的缓冲区和足够长的计算段；若加载更慢、资源竞争严重或只处理一块，收益会减小。下一篇硬件流水会沿这个模型继续推演。

## 9. 在线 softmax 如何跨线程合作

一行 score 可能分布在多个线程寄存器中。每条线程先对自己持有的列取最大值，再通过 `Allreduce<4>` 合并；该 helper 在 `utils.h` 中使用 shuffle 指令在协作线程之间交换寄存器值。这里 4 是这套 MMA 行片段布局的协作组大小，不是“整块只有 4 个线程”。

m 必须及时统一：各片段要使用同一个指数基准。l 的局部和则可以暂存在各线程中，因为在基准相同的前提下，“先各自累加，最后再合并”与“每轮都合并”在数学上等价。`softmax.h` 的 `reduce_sum` 只做线程内归约，而 `normalize_softmax_lse` 的 `quad_allreduce_` 在末尾完成跨线程求和。这是代码里可以直接看见的通信削减。

## 10. 源码精读：算清楚一条线程究竟拿到哪些数

本节继续使用 `M=N=128、head_dim=64、4 个 warp` 的配置。下面的线程编号、坐标都是由该配置推导，不能直接套用到其他 kernel。

### 10.1 两套分工对象，分别回答两个问题

源码摘录：[flash_fwd_kernel.h，176–190 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L176-L190)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:176-190 -->
```cpp
typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

typename Kernel_traits::TiledMma tiled_mma;
auto thr_mma = tiled_mma.get_thread_slice(tidx);
Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)
```

逐句理解：

1. `gmem_tiled_copy_QKV` 包含整块线程的搬运规则。`get_thread_slice(tidx)` 取当前线程的规则，没有启动新线程。
2. `partition_S(gQ)` 和 `partition_D(sQ)` 用这条规则选择源、目标坐标。它们是一对视图；此处没有读写 Q。
3. `tiled_mma` 包含计算的分工。`partition_fragment_A/B` 建立适合矩阵乘操作数的线程寄存器片段。片段建立后还需 `copy` 填数据。
4. 同一个 `tidx` 同时拥有搬运任务和计算任务。这两个任务的坐标集合可以不同，共享内存承担中间交接。

搬运布局的定义如下：

源码摘录：[kernel_traits.h，118–133 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/kernel_traits.h#L118-L133)。

<!-- source: csrc/flash_attn/src/kernel_traits.h:118-133 -->
```cpp
static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                              Stride<Int<kGmemThreadsPerRow>, _1>>;

// We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
// from the same address by the same threadblock. This is slightly faster.
using Gmem_copy_struct = std::conditional_t<
    Has_cp_async,
    SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
    AutoVectorizingCopyWithAssumedAlignment<128>
>;
using GmemTiledCopyQKV = decltype(
    make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                    GmemLayoutAtom{},
                    Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
```

代入本例，线程布局是 `Shape<16,8>`，stride 是 `(8,1)`，因此线程坐标 `(r,c)` 对应 `tidx=8*r+c`。每个线程坐标携带 `Shape<1,8>` 的 8 个相邻 FP16 值，于是：

```text
线程 t 首次搬运的行 = t // 8
线程 t 首次搬运的列 = 8*(t % 8) ... 8*(t % 8)+7
一个线程布局覆盖 16 行 × 64 列
128 行需要重复 8 次，行号每次增加 16
```

例如线程 37 搬 `gQ` 的局部行 `4,20,36,52,68,84,100,116`，每行的列 `40...47`。共搬 `8×8=64` 个数；整个 block 共搬 `128×64=8192` 个数，恰好是一个 Q tile。这里列出的是**逻辑坐标**；shared 端还有 swizzle，实际物理地址不能照普通行优先公式套用。

### 10.2 计算时，线程 37 的 score 寄存器不是这些坐标

mask 必须知道寄存器中的每个 score 对应哪对 query/key。因此 mask 的坐标计算，恰好把隐含的 MMA 分工暴露出来。

调用端提供每个线程的行起点：

源码摘录：[flash_fwd_kernel.h，335–337 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L335-L337)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:335-337 -->
```cpp
mask.template apply_mask<Is_causal, Is_even_MN>(
    acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
);
```

mask 内部先整理寄存器的坐标轴：

源码摘录：[mask.h，140–145 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/mask.h#L140-L145)。

<!-- source: csrc/flash_attn/src/mask.h:140-145 -->
```cpp
// Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
// Do we need both row and column indices, or just column incides?
static constexpr bool Col_idx_only = !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask;
const int lane_id = threadIdx.x % 32;
const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
```

再按行、列分别展开：

源码摘录：[mask.h，167–179 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/mask.h#L167-L179)。

<!-- source: csrc/flash_attn/src/mask.h:167-179 -->
```cpp
for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
    #pragma unroll
    for (int i = 0; i < size<0, 0>(tensor); ++i) {
        const int row_idx = row_idx_base + i * 8;
        const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
        const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
        #pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
            const int col_idx_base = col_idx_offset + nj * 8;
            #pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                const int col_idx = col_idx_base + j;
```

`(MMA=4,MMA_M,MMA_N)` 中的 4 被拆成 `2×2`：每个小片段含两个行位置、两个列位置。`make_tensor(tensor_.data(), 新布局)` 保留原寄存器数据，只换索引方式。上述代码对本例给出：

```text
warp = tidx // 32
lane = tidx % 32
行 = m_block*128 + warp*16 + lane//4 + i*8 + mi*64
列 = n_block*128 + (lane%4)*2 + j + nj*8
其中 i,j ∈ {0,1}，mi ∈ {0,1}，nj ∈ {0,...,15}
```

选择上一章的 `m_block=1、n_block=1`，代入线程 37：`warp=1、lane=5`。

`lane` 是 warp 内的线程编号，范围为 0～31；`tidx` 是整个 block 内的编号。这里 `37//32=1`、`37%32=5`。行式中的 `warp*16+lane//4=17`，再加 `i*8` 和 `mi*64`，得到局部行 `17、25、81、89`，最后加 Q tile 起点 128，才变成下面的全局行号。

| 内容 | 线程 37 持有的坐标 |
|---|---|
| score 的 query 行 | `145、153、209、217` |
| 每行的 key 列 | `130、131、138、139、…、250、251`，共 32 列 |
| score 数量 | `4 行 × 32 列 = 128` 个 FP32 累加值 |

线程 36、37、38、39 拥有相同的 4 个 query 行，各自持有不同的 32 列。四条线程合起来覆盖这些行的全部 128 个 key。这就是 Softmax 中 `Allreduce<4>` 的布局依据：它交换的是**同一行的不同列片段**，不是随便挑四条线程求平均。

把公式对 128 条线程全部展开，恰好覆盖 `128×128` 个不同 score 坐标，没有遗漏或重复。它也解释了为什么“一个线程算一个 attention 输出元素”无法描述这个 kernel。

### 10.3 mask 真正在这些寄存器上改什么

源码摘录：[mask.h，188–191 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/mask.h#L188-L191)。

<!-- source: csrc/flash_attn/src/mask.h:188-191 -->
```cpp
if constexpr (Causal_mask) {
    if (col_idx >= col_idx_limit_right) {
        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
    }
```

`col_idx_limit_right` 是**第一个不允许访问的 key 下标**，所以比较符是 `>=`。在等长 causal attention 中，query 145 的上界是 146：key 145 允许，key 146 起被改成负无穷。后续 `exp(-inf)=0`，这些位置不会贡献分母和输出。

`N_q=2、N_k=5` 时，query 0 的上界是 `0+1+5-2=4`，因此 key 0、1、2、3 允许。这是源码中的右下对齐，而不是把所有非方阵都套上 `j<=i`。

注意代码有 `if constexpr (Need_masking)`：无 causal、无 local、无 ALiBi、尺寸整齐时，整个 mask 体可以在编译时消失。上面的坐标推导仍描述同一 MMA 布局，但这种分支无需真的逐元素写 mask。

### 10.4 边界检查为什么同时需要坐标和布尔数组

真正执行搬运的 helper 核心如下：

源码摘录：[utils.h，309–323 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/utils.h#L309-L323)。

<!-- source: csrc/flash_attn/src/utils.h:309-323 -->
```cpp
#pragma unroll
for (int m = 0; m < size<1>(S); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            if (Is_even_K || predicate_K(k)) {
                cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
            } else if (Clear_OOB_K) {
                cute::clear(D(_, m, k));
            }
        }
    } else if (Clear_OOB_MN) {
        cute::clear(D(_, m, _));
    }
}
```

这里模板参数名中的 `K` 指当前拷贝矩阵的末维，不一定是 attention 的 Key 张量。对于 Q/K 输入，通常就是 head dimension。

逐层展开成初学者可以执行的规则：

- 外层 `m` 枚举该线程负责的 token 行。`identity_MN` 保存逻辑坐标，因此可以比较行号与实际长度 `max_MN`。
- 内层 `k` 枚举特征方向的拷贝片段。`predicate_K(k)` 表示该片段是否落在有效特征范围内。
- 有效位置执行 `cute::copy`。无效特征若 `Clear_OOB_K=true`，把目标清零，避免后续矩阵乘读到旧值。
- 无效行是否清零由 `Clear_OOB_MN` 决定。最终写 O 到显存时，无效行/列应跳过写入，不能试图“清零”数组外的地址。
- `Is_even_MN/Is_even_K=true` 表示编译时已知这一维无需边界检查。`||` 的短路使对应坐标/谓词不必被读取。

例如有效长度为 130、tile 长度为 128，第二个 tile 只有局部行 0、1 有效。逻辑局部行 4 的线程任务会被行条件挡住。不能把“grid 向上取整覆盖了 256 行”误读成“用户必须提供 256 行输入”。

### 10.5 追进 `gemm`：计算前的寄存器加载也有流水

源码摘录：[utils.h，146–159 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/utils.h#L146-L159)。

<!-- source: csrc/flash_attn/src/utils.h:146-159 -->
```cpp
Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
#pragma unroll
for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
        if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
        if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
}
```

`retile_D` 让目标寄存器以拷贝指令要求的方式编号，仍指向原寄存器片段。`A_in_regs/B_in_regs` 为假时，先把第 0 片操作数从 shared 载入寄存器。循环在计算第 i 片之前发起下一片的寄存器加载，从而给加载与计算交叠留下机会。

本例 QK 的归约轴是 head dimension 64，MMA 的 K 维为 16，可以理解为四轮累加：

```text
acc_s += Q[:,  0:16] @ K[:,  0:16].T
acc_s += Q[:, 16:32] @ K[:, 16:32].T
acc_s += Q[:, 32:48] @ K[:, 32:48].T
acc_s += Q[:, 48:64] @ K[:, 48:64].T
```

这段是**数学分解示意**，每一行在实际 kernel 中仍由全体线程的 MMA 片段共同完成。它位于一个 K tile 内部；不要与外层沿 token 遍历多个 K tile 的循环混淆。

PV 调用的 helper 少了 A 的 shared 加载：

源码摘录：[utils.h，174–182 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/utils.h#L174-L182)。

<!-- source: csrc/flash_attn/src/utils.h:174-182 -->
```cpp
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}
```

此处 A 是已经在寄存器里的 `tOrP`，B 是 V。循环只加载 V 的下一个片段，再执行乘加。因而前面 score→指数权重→FP16 `rP` 的转换，可以直接供 PV 消费。

### 10.6 从指针证明 V 转置没有复制，Q 空间可以复用

源码摘录：[flash_fwd_kernel.h，167–174 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L167-L174)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:167-174 -->
```cpp
Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                        typename Kernel_traits::SmemLayoutQ{});
// Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                        typename Kernel_traits::SmemLayoutKV{});
Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
```

本例不共享 Q/K 空间，三个区间按元素偏移分别是 `[0,8192)`、`[8192,16384)`、`[16384,24576)`。`sVt` 直接用 `sV.data()` 构造，所以 V 与 V 转置视图指向同一段存储。

前向尾声的 `sO` 则从 `smem_` 起始位置构造，复用原来的 Q 空间。此时 Q 的全部矩阵乘已结束，才能覆盖它。由此可以用一个具体原则检查同步：**新写入是否可能早于任意线程对旧数据的最后一次读取？**

同样，每轮 `cp_async_wait<0>()` 负责等待当前线程所提交的相关异步拷贝完成，紧随其后的 `__syncthreads()` 让整块线程在共享数据的交接处会合。某条线程自己的拷贝完成，并不代表其准备读取的、由其他线程搬来的元素都已完成；两者不能随意互相替代。


## 11. 常见误解与边界

- `local_tile(...)` 是否意味着发生了一次显存拷贝？**不意味着；这里是在选取视图。**
- `sV` 与 `sVt` 是否一定有两份存储？**不是，它们在这里基于同一数据地址。**
- `row_sum` 中途是否已经是一整行的总和？**不是，当前分支延迟到最后做跨线程归约。**
- 为何不能随便删除 `wait` 或屏障？**会破坏数据就绪或缓冲区复用的先后约束。**

下一节把同样的索引与分块思想用于反向和推理。
