# 2. 三个核心 kernel

[上一章](<01_execution_and_memory.md>) · [目录](<README.md>) · [下一章：验证与性能](<03_debug_profile_and_slime.md>)

完整实现集中在 [cuda_basics.cu](<examples/cuda_basics.cu>)，没有 PyTorch 或第三方库依赖。下面的代码片段解释其中的关键部分；编译时使用完整文件。三个算子分别对应**独立输出、多个输入合成一个输出、多个输出复用输入**，这些模式也出现在量化、归一化和 attention 中。

沿用 [第 1 章的软硬件映射](<01_execution_and_memory.md#hardware-map>)，每个例子同时检查三件事：block 在一个 SM 上拥有哪些输出，数据存在哪一层，以及哪条同步保护了哪次读写。程序开头会打印设备资源与 `matmul_tiled` 的编译属性，供下面的源码预算对照。

## 2.1 向量加法：先把完整生命周期走通

目标为 `c[i] = a[i] + b[i]`。上一章已解释 kernel，这里按 `run_vector_add` 看 host 端：

```cpp
size_t bytes = static_cast<size_t>(n) * sizeof(float);
CUDA_CHECK(cudaMalloc(&d_a, bytes));
CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
// b、c 的分配以及 b 的输入拷贝见完整文件。
vector_add<<<blocks, BLOCK>>>(d_a, d_b, d_c, n);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
```

分配出来的 `d_c` 不需要先清零，因为每个有效输出都会被完整写一次；这与对未初始化值做 `+=` 完全不同。拷回后逐元素与 CPU 结果比较，最后 `cudaFree` 释放每个分配。

这个问题每个元素做一次加法、逻辑上读两个 `float` 并写一个 `float`，即 1 FLOP / 12 字节。计算密度很低，数据足够大时通常应优先考虑访存；数组很小时，启动成本也会很显著。这是算法的工作量估算，实际 HBM 字节数会受缓存影响。

硬件上，每个 warp 的 lane 读取相邻 a/b 地址，操作数由访存路径进入线程的寄存器值，再交给浮点管线计算；这里没有声明 shared，也不需要先搬进 shared 才能相加。等待数据时能否有其他 warp 可发射，取决于 SM 的驻留量和指令依赖，而不只取决于 grid 总共有多少线程。

若计算改为 `c[i] = alpha*a[i] + b[i]`，改变的是逐元素表达式，索引与边界条件不变。`n=257` 在 `BLOCK=256` 或 128 时都包含尾块，因此每条线程仍需检查 `i<n`。

## 2.2 归约：怎样让多个线程合成一个值

目标为 `sum(x)`。直接让所有线程写 `out[0] += x[i]` 会产生数据竞争。我们的实现先在每个 block 内归约，输出一个 partial sum，再由 CPU 汇总 partials，刻意把跨 block 的依赖留在清楚的阶段边界上。

```cpp
__shared__ float values[BLOCK];
int t = threadIdx.x;
int i = blockIdx.x * BLOCK + t;
values[t] = i < n ? x[i] : 0.0f;
__syncthreads();

for (int step = BLOCK / 2; step > 0; step >>= 1) {
    if (t < step) values[t] += values[t + step];
    __syncthreads();
}
if (t == 0) partial[blockIdx.x] = values[0];
```

用 8 个线程处理 `[1,2,3,4,5,6,7,8]`：

| 阶段 | 本轮活动线程 | 下一轮需要的值 |
|---|---|---|
| 初始化 | 0..7 | `[1,2,3,4,5,6,7,8]` |
| step=4 | 0..3 | 前四项变成 `[6,8,10,12]` |
| step=2 | 0..1 | 前两项变成 `[16,20]` |
| step=1 | 0 | 第一项变成 `36` |

第一道 barrier 确保所有输入都已写入 shared。每轮的 barrier 确保下一轮读取的是更新后的值。barrier 位于 `if` 外面，即使某个线程这一轮不做加法，也参与同步。每个 block 写不同的 `partial[blockIdx.x]`，所以不存在不同 block 争抢同一输出的问题。

这个实现要求 **block 大小为 2 的幂，且实际启动线程数等于 `BLOCK`**。把启动改成 192 而不修改算法会破坏它；模板、常量和 launch 必须一起核对。输入长度不必是 2 的幂，最后一个 block 用零填充。

`values[256]` 是该 block 在其所在 SM 上使用的 1 KiB shared。八个 warp 协作读写它，barrier 覆盖整个 block；其他驻留在同一 SM 的 block 仍有各自独立的 values。写到 global 的 partial 则是后续阶段能够读取的结果，这把“SM 内协作”和“跨 block 汇总”分开了。

生产代码可以继续在 GPU 上归约 partials，或使用成熟的归约库。另一种方案是每块向结果做 `atomicAdd`：它能保护该地址的原子更新，但输出必须先初始化，竞争会影响性能，浮点累加顺序可能变化；原子更新也不是整个 grid 的 barrier。

warp shuffle 能直接在线程间交换寄存器值，是后续优化方向。使用 `_sync` 系列时要明确参与 mask 和有效来源 lane，不能认为“同一 warp 永远自动同步”就随意删 barrier。第一遍先掌握 shared 版本。

归约运算决定中性值：求最大值时，越界元素必须填负无穷。`[-7,-4,-9]` 的最大值为 -4；若 padding lane 填 0，结果会错误地变成 0。

## 2.3 矩阵乘：为什么值得用 shared memory

矩阵乘为 `C[M,N] = A[M,K] @ B[K,N]`，其中：

```text
C[row,col] = A[row,0]*B[0,col] + ... + A[row,K-1]*B[K-1,col]
```

最简单的方案是一个线程负责一个 `C[row,col]`，在本线程里循环 K 次。下面是用于比较的 **naive kernel**，用于展示逐元素累加的数据流；默认程序运行后面的 tiled 版本：

```cpp
__global__ void matmul_naive(const float* a, const float* b, float* c,
                             int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) acc += a[row*k+j] * b[j*n+col];
    c[row*n+col] = acc;
}
```

同一行的多个输出会用到相同的 A 元素，同一列的多个输出会用到相同的 B 元素。tiled 版本让一个 block 负责 `16×16` 个输出，每轮合作加载一块 A 和一块 B，再复用这些值。

```text
block (bx, by)
  负责 C[by*16 : by*16+16, bx*16 : bx*16+16]
  对 base = 0,16,32,... 逐轮执行：
    256 个线程合作加载 A tile 和 B tile
    barrier：等加载完成
    每个线程将 16 对乘积加入自己的 acc
    barrier：等所有线程读完后，再复用 shared
  写回各自的 C 元素
```

核心代码与完整文件一致：

```cpp
as[ty][tx] = row < m && base + tx < k ? a[row*k + base+tx] : 0.0f;
bs[ty][tx] = base + ty < k && col < n ? b[(base+ty)*n + col] : 0.0f;
__syncthreads();
for (int j = 0; j < TILE; ++j) acc += as[ty][j] * bs[j][tx];
__syncthreads();
```

第一道 barrier 保护“写完再读”；第二道保护“读完再覆盖”。即使某个线程对应的 C 在边界之外，它仍可能负责加载其他有效线程需要的 A/B 数据，所以 tiled kernel 不能照搬 naive 的提前 `return`。

把这一轮放回硬件：`dim3(16,16)` 是一个拥有 8 个 warp 的 block，整体落到一个 SM；`as/bs` 共申请 `2*16*16*4=2048` 字节 shared，`acc` 是每线程独立的累加值，通常使用该 SM 的寄存器资源。warp 0 横跨两行各 16 个线程，`as[ty][tx]` 的合作加载地址必须按这组实际 lane 来分析。此版使用标量 FP32 乘加，没有因为矩阵呈二维就自动成为 Tensor Core 运算。

若把 TILE 从 16 改为 32，block 变为 1024 线程、32 个 warp，shared 变为 8 KiB。更大的数据复用同时带来更多线程和存储资源需求；必须先检查设备与 kernel 的线程限制，再核对 [驻留预算](<01_execution_and_memory.md#residency>)和实际时间。

具体代入 `M=17, N=19, K=23`，grid 为 `(2,2)`。取 `bx=1, by=0, tx=2, ty=3`：该线程负责 `C[3,18]`，输出线性偏移为 `3*19+18=75`。第一轮 `base=0`，它搬运 `A[3,2]` 与 `B[3,18]`；计算时它使用的是整条 `as[3,j]` 与 `bs[j,2]`，其中多数元素由别的线程加载。第二轮 `base=16`，只剩 K 方向的 7 项有效，剩余 9 项填零。

一个完整 tile 阶段加载 `2*16*16=512` 个浮点数，完成 `16*16*16` 次乘加，按乘和加各计一 FLOP 得到 8192 FLOPs。只考虑这一阶段输入，计算密度为 `8192/(512*4)=4 FLOP/byte`；输出写回、边界和缓存均未计入，因此这不是实测带宽结论。收益来自明确的数据复用，是否更快仍需测量。

这个实现是 FP32 教学 GEMM，没有 Tensor Core、寄存器分块或异步流水，不能作为 cuBLAS/FlashAttention 的性能替代品。Tensor Core 使用特定矩阵运算指令与布局，不会因为你写了乘加循环就自动得到同样的实现。

例如 `17×23 @ 23×19` 同时包含 M、N、K 三个维度的尾块，能说明补零与写回边界各自的作用。增大 TILE 会改变线程数、shared 占用和驻留容量；减少重复加载并不保证端到端更快。

## 2.4 边界尺寸与误差范围

配套示例覆盖向量长度 `1,31,32,255,256,257,1003`，以及矩阵 `(M,N,K)` 为 `(1,1,1)`、`(16,16,16)`、`(17,19,23)`、`(31,7,33)`。这些尺寸分别暴露单元素、warp/block 边界、尾部填充和非方阵索引问题。

输入包含正数、负数和零。为了让第一步排错清楚，示例使用可精确表示的二进制分数；随机输入、强消去和更大 K 会带来不同的误差分布。当前 `check_close` 使用 `1e-4 + 1e-4*abs(expected)` 容差并拒绝非有限结果，该容差只服务于这里的小型 FP32 示例，不是所有 dtype 和形状的通用阈值。

API 与同步语义可查 [CUDA C++ 入门](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)；更多优化实现可读 [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)。
