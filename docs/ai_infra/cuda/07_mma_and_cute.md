# 7. 从一个线程一个结果，到 Tensor Core 与 CuTe 布局

[上一章](06_online_attention.md) · [目录](README.md) · [下一章](08_async_pipeline.md)

这一章有两个独立示例：[wmma_tile.cu](examples/wmma_tile.cu) 实际计算一个矩阵块，[cute_layout.cu](examples/cute_layout.cu) 验证形状、步长与同一份内存的多个视图。先学“共同持有一个矩阵”和“坐标怎样变成地址”，再看真实 FA2 的 `TiledMMA`，不需要先读完整 CUTLASS。

## 7.1 先跨过 C++ 模板这道门

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

## 7.2 WMMA：32 条线程共同做一次矩阵乘

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

在硬件图里，block 仍然分配到一个 SM，warp 的指令仍由该 SM 的调度与执行资源处理；变化的是矩阵乘加使用 Tensor Core 相关的运算路径。Tensor Core 并没有替代 block/warp 层次，fragment 也不是另一份 HBM 张量。此示例总共只有一个 block，只演示协作与数值，不能衡量整张 GPU 的矩阵吞吐。

`load_matrix_sync(...,16)` 中最后的 16 是内存中的 leading dimension：行优先矩阵相邻行起点相差 16 个元素。不要把它当成 warp 数或字节数。本例直接使用 `cudaMalloc` 返回的对齐地址；使用任意偏移或切片时还要满足 WMMA 的地址及 leading dimension 对齐要求。接口约束见 [CUDA WMMA 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)。

下面以支持 SM80 的目标机器为例；其他 GPU 应使用其受 Toolkit 支持的目标架构，见第 0 章。此处 WMMA 运算要求兼容的 compute capability ≥7.0 设备，并且所用 Toolkit 仍支持为该设备生成代码：

```bash
nvcc -O2 -lineinfo -std=c++17 -arch=sm_80 docs/ai_infra/cuda/examples/wmma_tile.cu -o /tmp/slime_wmma_tile
/tmp/slime_wmma_tile
```

默认输入含正负数并且不是常量矩阵，因此错误的转置和索引更容易暴露。程序与 **FP16 转换后的输入**比较：若拿原始 FP32 输入直接当参考，可能把量化误差错当成计算错误。

## 7.3 从 WMMA 走向 FA2 使用的 MMA atom

WMMA 是建立协作计算概念的示例，FA2 这条路径通过 CuTe 封装更具体的 MMA 指令，不直接调用上述 WMMA API。当前工作区 `kernel_traits.h` 在相应架构与类型分支里选择 `SM80_16x8x16_F32F16F16F32_TN`。

对该 SM80 `m16n8k16` FP32 accumulator 布局，一个 warp 产生 `16×8=128` 个结果，32 条线程各有 4 个。令 `lane=0..31`，四个结果的逻辑坐标可写成：

```text
row = lane//4 + i*8
col = (lane%4)*2 + j
i,j ∈ {0,1}
```

例如 lane 5 拥有 `(1,2)、(1,3)、(9,2)、(9,3)`。lane 4/5/6/7 拥有相同的两行、不同列，合起来覆盖这两行全部 8 列。需要做逐行归约时，这四个 lane 就是自然的协作组。

这张坐标表针对上述 MMA 指令的 accumulator；**不能拿来索引 WMMA 的 fragment.x，也不能直接套给 A/B 操作数或其他指令形状**。硬件布局可核对 [PTX ISA 的矩阵片段说明](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)。`attention_cpu.py` 会枚举这 128 个坐标，检查既不重复也不遗漏；第 9 章再扩展到 FA2 的 `128×128` tile。

## 7.4 CuTe Layout 先理解成“坐标到偏移的函数”

一份内存为 `[10,11,12,20,21,22]`。行优先 `2×3` 视图对应：

```text
shape=(2,3), stride=(3,1)
offset(r,c)=3*r+c
(1,2) → 5 → 数据 22
```

其转置视图为 `shape=(3,2), stride=(1,3)`，`offset(r,c)=r+3*c`。于是转置视图的 `(2,1)` 仍然访问偏移 5，并未搬运数据。

`cute_layout.cu` 中对应的写法是：

```cpp
auto layout = make_layout(make_shape(Int<2>{}, Int<3>{}),
                          make_stride(Int<3>{}, Int<1>{}));
auto view = make_tensor(data, layout);
```

`make_layout` 构造映射，`make_tensor` 把地址与映射组合成视图。示例会通过转置视图写入 99，再确认原数组和原视图同时看见 99，用这个现象区分“别名视图”和“另一份拷贝”。

嵌套 shape 则是把一个维度分成多个子坐标，例如 `((2,4),3)`；给出 stride `((1,2),8)` 后，坐标 `((i,j),k)` 的偏移是 `i+2*j+8*k`。`((1,2),1)` 对应 13。先展开这个加法，再理解更复杂的 `size<0,1>` 和布局组合。入门定义见 [CuTe Layouts](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/cute/01_layout.html)。

## 7.5 CuTe 示例的依赖边界

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

真实代码分别定义搬运的 `TiledCopy` 与计算的 `TiledMMA`：

| 名称 | 要回答的问题 |
|---|---|
| `get_thread_slice(tidx)` | 取整套分工规则中属于当前线程的部分 |
| `partition_S` / `partition_D` | 当前线程从哪些源坐标搬到哪些目标坐标 |
| `partition_fragment_A/B` | 当前线程需要哪些矩阵乘操作数片段 |
| `partition_fragment_C` | 当前线程持有哪些 accumulator 片段 |
| `copy` / `gemm` | 真正执行搬运或计算；建立视图本身不等于执行这些操作 |

搬运希望连续、对齐；MMA 则要求特定的操作数分布，因此不必采用同一分工。shared memory 充当交接处。对相同逻辑元素，global 与 shared 的物理地址布局也可以不同。

例如一个**纯教学**的 4×4 布局用 `physical_col=logical_col XOR row` 重排每行；每行仍是原列的一次排列。读者必须用同样的映射才能取回原元素。真实 `Swizzle<...>` 作用于指定地址位，不能把这个 4×4 公式直接套入 FA2；先理解“逻辑坐标没变、物理摆放改变”，再读已有 [布局与流水精读](../13_Source_Analysis/flash_attention/03_layout_and_pipeline.md) 的 swizzle 和线程 37 算例。

## 7.7 四种布局概念的边界

模板参数、逻辑坐标和物理存储各有不同含义。前文模板中的 `Width=32` 只改变 `columns`，`scale<false>(3.0f)` 返回 3；这与线程如何协作无关。转置视图改变坐标到地址的映射，不复制底层数据。

在 m16n8 accumulator 的映射中，lane 31 持有 `(7,6)、(7,7)、(15,6)、(15,7)`；这些坐标描述每条线程的寄存器值。`wmma::mma_sync` 则要求整个 warp 协作，不能放在只允许 lane 0 进入的分支中。区分 tile 形状、线程排列、每线程持有的值和实际内存布局，才能解释后续流水中的搬运与计算。
