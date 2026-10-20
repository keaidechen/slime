# 7. 从一个线程一个结果，到 Tensor Core 与 CuTe 布局

[上一章](<06_online_attention.md>) · [目录](<README.md>) · [下一章](<08_async_pipeline.md>)

这一章有两个独立示例：[wmma_tile.cu](<examples/wmma_tile.cu>) 实际计算一个矩阵块，[cute_layout.cu](<examples/cute_layout.cu>) 验证形状、步长与同一份内存的多个视图。先学“共同持有一个矩阵”和“坐标怎样变成地址”，再看真实 FA2 的 `TiledMMA`，不需要先读完整 CUTLASS。

第 6 章已经减少了中间数据，但点积与 PV 仍由普通浮点循环完成。Tensor Core 是执行特定矩阵乘加的硬件资源；要利用它，线程必须按指令规定共同提供矩阵片段。MMA 是 Matrix Multiply-Accumulate（矩阵乘累加），WMMA 是 Warp Matrix Multiply-Accumulate 的 CUDA C++ 接口。CuTe 是 CUTLASS 项目中描述张量、布局及线程分工的 C++ 模板抽象；CUTLASS 是 NVIDIA 的高性能线性代数模板库。布局工具的价值在于把“连续地搬运”和“按硬件要求计算”这两种分工连接起来。

## 7.1 先跨过 C++ 模板这道门

**C++ 模板**是用类型或编译期常量定制代码的机制，例如生成不同元素类型、不同 tile 宽度的版本。模板实例是编译时确定的一份具体代码；运行时函数参数则提供本次处理的数据。CuTe 大量用模板表达形状和布局，因此先区分“选择哪份代码”和“向代码传入什么数据”。

CUDA 源码里的尖括号经常描述编译期配置，而不是运行时输入。用普通 C++ 看一个完整的小程序：

```cpp
#include <cstdio>

template <int Width, typename T>
struct Tile {
    using Element = T;
    static constexpr int columns = Width;
    template <bool Twice>
    T scale(T x) const {
        if constexpr (Twice) return x * 2;
        else return x;
    }
};

template <typename Config>
auto evaluate(typename Config::Element x) {
    Config config;
    return config.template scale<true>(x);
}

int main() {
    using Config = Tile<16, float>;
    auto value = evaluate<Config>(3.0f);
    std::printf("columns=%d, value=%.1f\n", Config::columns, value);
}
```

将其保存为 `/tmp/slime_template_demo.cpp`，用 `c++ -std=c++17 /tmp/slime_template_demo.cpp -o /tmp/slime_template_demo` 编译后运行，预期为 `columns=16, value=6.0`，不需要 GPU。

| 符号 | 对应例子 | 读法 |
|---|---|---|
| `template<int Width, typename T>` | `Tile<16,float>` | 用整数与类型定制一份代码 |
| `using Element=T` | `Config::Element` | 给类型取别名 |
| `typename Config::Element` | evaluate 的参数类型 | 告诉编译器：这个依赖模板参数的名称是类型 |
| `.template scale<true>` | 对模板对象调用成员模板 | 帮编译器解析 `<...>`；没有启动 kernel |
| `if constexpr` | Twice 为编译期布尔值 | 按实例选择分支，未选分支不参与该实例的执行 |
| `auto`、`decltype(expr)` | 推导返回值或表达式类型 | 可以避免手写很长的类型名 |

CuTe 的 `Int<16>{}` 同时涉及类型和对象：`Int<16>` 是携带常量 16 的类型，后面的 `{}` 创建这个类型的对象。`Shape<Int<16>,Int<8>>{}` 因而表示编译期已知的形状，而不是分配一个 `16×8` 浮点数组。

为何要在编译期选择这些值？矩阵指令只接受规定的形状与类型，固定 tile 后，编译器也能展开循环并确定片段大小。代价是不同配置可能产生多份代码，编译和二进制体积增长；输入实际长度仍可以是运行时值，边界仍需检查。`Tile<16,float>`、函数调用 `scale<true>(x)`、kernel 启动 `f<<<grid,block>>>(...)` 中的尖括号各有用途：分别是模板类型、模板函数实例和 CUDA 启动配置，不应只凭标点判断代码是否正在执行 GPU 工作。

## 7.2 WMMA：32 条线程共同做一次矩阵乘

**WMMA** 是以一个 warp 为协作单位的矩阵乘累加 C++ 接口，用来表达 `D=A@B+C`。它将输入与累加结果组织成 **fragment（矩阵片段）**：整个逻辑矩阵由 warp 共同持有，各线程的 fragment 只保存其中一部分。它与第 2 章“每线程独立算一个输出”的分工不同，也不为每个线程额外分配一张完整矩阵。

第 2 章中，每个线程用一个 `float acc` 算一个 C 元素。现在一个完整 warp 共同完成 `16×16` 的输出，归约维度 K=16，输入 FP16，累加 FP32：

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
wmma::fill_fragment(cf, 0.0f);
wmma::load_matrix_sync(af, a, 16);
wmma::load_matrix_sync(bf, b, 16);
wmma::mma_sync(cf, af, bf, cf);
wmma::store_matrix_sync(c, cf, 16, wmma::mem_row_major);
```

完整程序以 `<<<1,32>>>` 启动，32 个 lane 都执行这些操作。每个线程中的 fragment 只是协作矩阵的一部分，不能把它当成该线程独占的完整矩阵。WMMA API 的 fragment 到 lane 的元素映射不应自行假设；为了核对答案，示例先 store 成普通行优先 C，再逐元素与 CPU 比较。

数学操作是 `D=A@B+C`，A/B/C/D 在本例均为 `[16,16]`，代码将 D 写回 cf，所以表现为 `cf=af@bf+cf`。`fill_fragment` 将累加器初始化为零，两个 load 让各 lane 获得其操作数片段，mma 更新寄存器片段，store 才将结果写入普通 global 数组 c。这里只执行一次 K=16 的乘加；更大的归约维度需要重复加载下一对 A/B 片段，并保留 cf 累加。

手算时可将左上角嵌入 `A₂=[[1,2],[3,4]]`、`B₂=[[5,6],[7,8]]`，16×16 的其余输入全部置零。于是 C 左上角是 `[[19,22],[43,50]]`，其他位置为零。例如 19=`1*5+2*7+14个零乘积`。硬件依然执行规定的 16×16×16 形状，不会因非零数据只有 2×2 就自动变成任意小形状指令；这种补零适合理解，性能上会浪费工作。

`sync` 表示这些 API 的 warp 协作要求，不能把它当作 CPU 已能读取结果的保证，也不能替代生产 shared 输入所需的 block barrier。若数据由别的 warp 搬入 shared，仍须先保证生产完成且可见，再由计算 warp 执行 load；本例直接读已拷贝好的 global 输入。输出在 GPU 写完并完成 D2H 后才由 CPU 检查。

在硬件图里，block 仍然分配到一个 SM，warp 的指令仍由该 SM 的调度与执行资源处理；变化的是矩阵乘加使用 Tensor Core 相关的运算路径。Tensor Core 并没有替代 block/warp 层次，fragment 也不是另一份 HBM 张量。此示例总共只有一个 block，只演示协作与数值，不能衡量整张 GPU 的矩阵吞吐。

`load_matrix_sync(...,16)` 中最后的 16 是内存中的 leading dimension：行优先矩阵相邻行起点相差 16 个元素。不要把它当成 warp 数或字节数。本例直接使用 `cudaMalloc` 返回的对齐地址；使用任意偏移或切片时还要满足 WMMA 的地址及 leading dimension 对齐要求。接口约束见 [CUDA WMMA 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)。

下面以支持 SM80 的目标机器为例；其他 GPU 应使用其受 Toolkit 支持的目标架构，见第 0 章。此处 WMMA 运算要求兼容的 compute capability ≥7.0 设备，并且所用 Toolkit 仍支持为该设备生成代码：

```bash
nvcc -O2 -lineinfo -std=c++17 -arch=sm_80 docs/ai_infra/cuda/examples/wmma_tile.cu -o /tmp/slime_wmma_tile
/tmp/slime_wmma_tile
```

默认输入含正负数并且不是常量矩阵，因此错误的转置和索引更容易暴露。程序与 **FP16 转换后的输入**比较：若拿原始 FP32 输入直接当参考，可能把量化误差错当成计算错误。

FP32 累加能减轻长串求和的舍入，但不能恢复输入转 FP16 时丢掉的信息。例如按通常的最近舍入，1 附近 FP16 相邻数相差 `2^-10≈0.0009765625`，1.0003 转换后成为 1；后续即使用 FP32 乘加，也不会重新知道被舍弃的 0.0003。报告性能与误差时要同时写输入、累加和输出精度。Tensor Core 提高矩阵运算吞吐的条件还包括足够工作量、合适形状与及时的数据供给，不能只凭使用了 MMA 就断言整个算子更快。

## 7.3 从 WMMA 走向 FA2 使用的 MMA atom

**MMA atom** 是 CuTe 对一种基础矩阵乘累加操作及其线程/数值布局规则的封装，**TiledMMA** 再把 atom 按规则组织成更大的计算分工。它们属于代码与布局抽象；实际矩阵指令由硬件执行。学习 atom 是为了知道某个输出坐标属于哪个 lane，以及哪些 lane 应共同做行归约。

WMMA 是建立协作计算概念的示例，FA2 这条路径通过 CuTe 封装更具体的 MMA 指令，不直接调用上述 WMMA API。当前工作区 `kernel_traits.h` 在相应架构与类型分支里选择 `SM80_16x8x16_F32F16F16F32_TN`。

对该 SM80 `m16n8k16` FP32 accumulator 布局，一个 warp 产生 `16×8=128` 个结果，32 条线程各有 4 个。令 `lane=0..31`，四个结果的逻辑坐标可写成：

```text
row = lane//4 + i*8
col = (lane%4)*2 + j
i,j ∈ {0,1}
```

例如 lane 5 拥有 `(1,2)、(1,3)、(9,2)、(9,3)`。lane 4/5/6/7 拥有相同的两行、不同列，合起来覆盖这两行全部 8 列。需要做逐行归约时，这四个 lane 就是自然的协作组。

`m16n8k16` 中 m=16、n=8 是结果 tile 的行列数，k=16 是本次乘加的归约长度：A `[16,16]` 与 B `[16,8]` 生成 C `[16,8]`。每个输出累加 16 个乘积，按惯例共 `2*16*8*16=4096` FLOPs。4 个 accumulator/线程来自 `16*8/32=4`，是输出值数，不是每线程发出四次独立矩阵指令。TiledMMA 的组合不会改变每个底层指令的参与要求。

PTX（Parallel Thread Execution）是 NVIDIA 的虚拟指令表示，还需经后续编译生成目标 GPU 的机器代码；查看 PTX 可以理解部分操作，但不能据此确定最终寄存器占用与指令调度。

这张坐标表针对上述 MMA 指令的 accumulator；**不能拿来索引 WMMA 的 fragment.x，也不能直接套给 A/B 操作数或其他指令形状**。硬件布局可核对 [PTX ISA 的矩阵片段说明](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)。`attention_cpu.py` 会枚举这 128 个坐标，检查既不重复也不遗漏；第 9 章再扩展到 FA2 的 `128×128` tile。

## 7.4 CuTe Layout 先理解成“坐标到偏移的函数”

**Layout（布局）**是把逻辑坐标映射到线性偏移的规则；shape 指定合法坐标范围，stride 等规则决定每个坐标落到哪里。**Tensor 视图**将一份数据存储与 Layout 组合，让程序按逻辑坐标访问数据。创建视图本身不分配或搬运矩阵，这一点决定了“转置视图”和“复制转置后的数据”是不同操作。

一份内存为 `[10,11,12,20,21,22]`。行优先 `2×3` 视图对应：

```text
shape=(2,3), stride=(3,1)
offset(r,c)=3*r+c
(1,2) → 5 → 数据 22
```

其转置视图为 `shape=(3,2), stride=(1,3)`，`offset(r,c)=r+3*c`。于是转置视图的 `(2,1)` 仍然访问偏移 5，并未搬运数据。

把两个视图真正展开，原矩阵是 `[[10,11,12],[20,21,22]]`，转置视图是 `[[10,20],[11,21],[12,22]]`。在转置视图沿列走一步跨过 3 个 float，沿行走一步只跨过 1 个；如果后续 kernel 错按“每行连续两项”的公式 `2*r+c` 读取，就把视图错当成已经搬好的连续矩阵。只有另外分配并复制成 `[10,20,11,21,12,22]`，才得到 stride=(2,1) 的连续转置结果。视图创建节省搬运，但消费者必须支持它的真实 stride。

`cute_layout.cu` 中对应的写法是：

```cpp
auto layout = make_layout(make_shape(Int<2>{}, Int<3>{}),
                          make_stride(Int<3>{}, Int<1>{}));
auto view = make_tensor(data, layout);
```

`make_layout` 构造映射，`make_tensor` 把地址与映射组合成视图。示例会通过转置视图写入 99，再确认原数组和原视图同时看见 99，用这个现象区分“别名视图”和“另一份拷贝”。

嵌套 shape 则是把一个维度分成多个子坐标，例如 `((2,4),3)`；给出 stride `((1,2),8)` 后，坐标 `((i,j),k)` 的偏移是 `i+2*j+8*k`。`((1,2),1)` 对应 13。先展开这个加法，再理解更复杂的 `size<0,1>` 和布局组合。入门定义见 [CuTe Layouts](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/cute/01_layout.html)。

这里 i=0/1、j=0…3、k=0…2，共 `2*4*3=24` 个逻辑元素；偏移最大为 `1+2*3+8*2=23`，该映射恰好覆盖 0…23。一般布局未必如此紧凑：shape=(2,3)、stride=(4,1) 有 6 个逻辑元素，但偏移集合为 `{0,1,2,4,5,6}`，底层至少需要 7 个元素的位置。逻辑元素数量与存储跨度不同；某些广播布局还会让多个坐标指向同一地址。创建视图不检查你的底层分配是否足够，也不改变谁拥有这块存储。

## 7.5 CuTe 示例的依赖边界

**依赖边界**说明运行示例还需要哪些外部工具、头文件和版本。本节的布局映射在 host 上计算，不启动 GPU kernel；但示例使用 CuTe/CUTLASS 头文件并通过 nvcc 构建，因此仍依赖 CUDA Toolkit。先确认这一点，再执行下面的可选准备步骤。

本项目的 FlashAttention 子模块记录了 CUTLASS 子模块 commit `7127592069c2fe01b041e174ba4345ef9b279671`。如果 `flash-attention/csrc/cutlass/include/cute/tensor.hpp` 尚不存在，在目标开发机器从项目根目录执行：

```bash
git -C flash-attention submodule update --init csrc/cutlass
git -C flash-attention/csrc/cutlass rev-parse HEAD
nvcc -O2 -std=c++17 --expt-relaxed-constexpr \
  -I flash-attention/csrc/cutlass/include \
  docs/ai_infra/cuda/examples/cute_layout.cu -o /tmp/slime_cute_layout
/tmp/slime_cute_layout
```

这是可选的依赖下载步骤；本次写作未执行。布局示例本身不启动 GPU kernel，但上述构建方式仍需要 CUDA Toolkit 和 CUTLASS 头文件；不是无依赖的 macOS 示例。没有该环境时，可以手算地址并运行标准库 Python 的布局覆盖检查。

## 7.6 一条线程为什么有两套坐标

**搬运分工**指定线程负责传递哪些数据，**计算分工**指定线程参与矩阵指令时持有哪些操作数和累加值。这两种分工可以不同：搬运希望连续、对齐，MMA 则要求特定的操作数分布，shared memory 充当交接处。同一逻辑元素的 global 与 shared 物理地址布局也可以不同。

真实代码用 `TiledCopy` 表达分块搬运分工，用前面的 `TiledMMA` 表达计算分工。以下接口用于提取线程自己的视图或执行操作：

| 名称 | 要回答的问题 |
|---|---|
| `get_thread_slice(tidx)` | 取整套分工规则中属于当前线程的部分 |
| `partition_S` / `partition_D` | 当前线程从哪些源坐标搬到哪些目标坐标 |
| `partition_fragment_A/B` | 当前线程需要哪些矩阵乘操作数片段 |
| `partition_fragment_C` | 当前线程持有哪些 accumulator 片段 |
| `copy` / `gemm` | 真正执行搬运或计算；建立视图本身不等于执行这些操作 |

### bank conflict 与 shared 布局

**bank** 是 shared memory 的硬件存储分区；**bank conflict** 指一次访问中，不同地址落到同一个 bank，超出其同时服务能力，需要拆分处理的情况。重排 shared 布局，是为了让消费者需要的地址更好地分散到不同 bank；这与第 1 章减少 global sector 数的合并访存是不同问题。

以常见的 32 个 bank、连续 32-bit 字分配到连续 bank 的标量 FP32 访问模型为例，`bank=(字节偏移/4)%32`。对行优先 `tile[32][32]`，一个 warp 的 lane t 读 `tile[t][0]`，字偏移为 32t，全部命中 bank 0 的不同地址，产生 bank conflict，访问需要拆分处理。若每行填充到 33 项，偏移为 33t，bank=t，冲突就消失；代价是多 `32*4=128` 字节 shared。若所有 lane 读取同一个字，则可以广播，不能当成上述“不同地址落同一 bank”的冲突。模型及适用范围见 [Shared Memory and Memory Banks](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)。

因此某种排列可让 global 连续读，却让 shared 的列读发生冲突。**padding** 是在布局中增加不承载有效元素的填充位置，以改变行步长等地址关系；**swizzle** 则通过可逆的地址重排改变 bank 分布。前者会增加容量需求，两者都需要按真实消费者的访问方式验证，尤其 FP16 向量访问和矩阵加载不能机械套用上面的标量公式。

XOR 是按位异或，相同位得 0、不同位得 1。例如一个**纯教学**的 4×4 布局用 `physical_col=logical_col XOR row` 重排每行；每行仍是原列的一次排列。读者必须用同样的映射才能取回原元素。真实 `Swizzle<...>` 作用于指定地址位，不能把这个 4×4 公式直接套入 FA2；先理解“逻辑坐标没变、物理摆放改变”，再读已有 [布局与流水精读](<../appendices/source/flash_attention/03_layout_and_pipeline.md>) 的 swizzle 和线程 37 算例。

row=1 时，逻辑列 0/1/2/3 对应物理列 1/0/3/2；逻辑 `(1,2)` 因而写到该行物理列 3。读取也应用同一映射才能拿回该值。因为 `(c XOR r) XOR r=c`，每行内可以逆变换、没有丢失元素。它改变的是存储位置，数学上的 query、key 和特征含义不变。

## 7.7 四种布局概念的边界

本章的**四种布局相关概念**分别描述数据块的大小、线程怎样排列、各线程持有哪些值，以及值放在什么地址。它们回答不同问题，不能仅凭一个 shape 就推断其余三者：

| 概念 | 回答的问题 | 例子 |
|---|---|---|
| tile 形状 | 本次处理多大的逻辑矩阵块？ | score `[128,128]` |
| 线程排列 | 哪些线程组成协作组，如何编号？ | 一个 block 的 4 个 warp |
| 每线程持有的值 | 某个线程负责哪些逻辑坐标？ | lane 5 的 4 个 MMA accumulator 坐标 |
| 内存布局 | 逻辑坐标对应哪个存储偏移？ | 行优先 stride、shared swizzle |

模板参数可以描述上述规则，但模板语法本身不是一种硬件布局。若将前文普通 C++ 示例的 Width 从 16 改为 32，只会改变 `columns`；`scale<false>(3.0f)` 返回 3，这些操作与线程如何协作无关。转置视图改变坐标到地址的映射，不复制底层数据。

在 m16n8 accumulator 的映射中，lane 31 持有 `(7,6)、(7,7)、(15,6)、(15,7)`；这些坐标描述每条线程的寄存器值。`wmma::mma_sync` 则要求整个 warp 协作，不能放在只允许 lane 0 进入的分支中。区分 tile 形状、线程排列、每线程持有的值和实际内存布局，才能解释后续流水中的搬运与计算。
