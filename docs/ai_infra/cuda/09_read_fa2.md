# 9. 从在线状态读一条固定的 FA2 前向路径

[上一章](<08_async_pipeline.md>) · [目录](<README.md>) · [继续源码精读](<../appendices/source/flash_attention/README.md>)

本章负责把前文的算子与布局分析接到真实代码，先限定一个可完整追踪的目标：**FP16、固定长度、无 dropout、无 causal 的 FA2 前向**。不要求同时读懂 backward、变长、分页、split-KV 或 FA3/FA4。

源码基准为本项目 [flash-attention](<../../../flash-attention>) 的 commit `e9cf2c1651d2303191eb40a739a3c135fda00999`；以下硬件原语讨论针对 SM80 分支。输入为 `Q/K/V.shape=(2,256,4,64)`，连续布局，`dropout_p=0`、`causal=False`，无额外 bias/softcap。

## 9.1 先认领你要追踪的 block

根据 [flash_fwd_launch_template.h](<../../../flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h>) 的 `run_mha_fwd_hdim64`，无 dropout 分支选择：

```cpp
Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, T>
```

依次表示 head_dim=64、Q 块高=128、KV 块高=128、4 个 warp，以及这份 traits 定义的两个布尔选项。当前配置每块 128 线程，一个 block 拥有一个 batch/head 的 128 行 Q 输出。grid 为 `(ceil(256/128),2,4)=(2,2,4)`，共 16 blocks。

这与第 6 章“一块一个 query”不同：现在一个 block 同时处理 128 个 query，以便在矩阵乘中共享数据和使用 Tensor Core。

将配置映射到硬件，一个 block 的四个 warp 在一个 SM 上协作；当前 `Share_Q_K_smem=false`，三个 FP16 Q/K/V tile 各为 `128*64*2=16 KiB`，合计 48 KiB shared/block。score 和输出 accumulator 由线程的寄存器片段持有，实际寄存器分配与可能的 spill 要查编译报告，不能只由 shared 数量推算 occupancy。

一次循环中，Q/K/V 的 global 访问经过相应缓存与搬运路径到 shared，MMA 所需片段送入矩阵运算路径，Softmax 则进行非矩阵的归约、指数与缩放，最后得到可写回 global 的 O。这让“为什么分块、为什么换布局、为什么要等待”同时对应到 SM 的存储和执行资源。

本教学输入总共仅 16 blocks；对于 SM 数大于 16 的设备，这个 kernel 无法同时给每个 SM 一个 block。它适合源码追踪，不是饱和整卡性能的基准。增大 batch、head 或序列长度会改变 grid 和资源使用情况；读取性能数据时必须同时记录 workload。

## 9.2 给熟悉的概念找到源码名字

| 前文的示例 | 真实代码里的对应位置 | 主要差别 |
|---|---|---|
| 计算一块 QK 分数 | `gemm`、`acc_s` | 线程共同持有 score tile，用 MMA 计算 |
| 稳定 Softmax | `Softmax::softmax_rescale_o` | 行分布在多个线程中；局部和可延迟合并 |
| 更新旧 numerator | `acc_o` 乘 `scores_scale` | 每线程持有多个输出片段 |
| 当前权重乘 V | `rP`、`gemm_rs` | P 片段留在寄存器，并转换成运算需要的类型/布局 |
| 共享 tile 与读写阶段 | `sQ/sK/sV`、`copy`、wait/barrier | CuTe 布局与异步搬运 |
| 最终 numerator/denominator | `normalize_softmax_lse` | 合并需要的统计量、归一化、保存 LSE |

表中“对应”是算法角色，不意味着可以逐行替换。尤其教学版已经把 score 乘了 scale；真实这条路径的原始 QK 分数和行最大值尚未乘 scale，`exp2` 时结合 `scale_softmax_log2` 完成缩放与换底。

## 9.3 用一条线程同时追搬运和计算

固定当前 block 的 Q/KV tile 坐标都从局部 0 开始，选择线程 37：

```text
tidx=37 → warp=1, lane=5
```

搬 Q 时，这份 traits 让 8 条线程各搬一行的 8 个 FP16 数。线程 37 的首个线程坐标是 `row=37//8=4, col_group=37%8=5`，因此搬局部列 40..47；后续重复沿行推进 16：

```text
搬运 Q 的行：4,20,36,52,68,84,100,116
每行的列：40..47
总共 64 个 FP16 数
```

而该线程持有 score 的局部行是 `17,25,81,89`，每行持有列 `2,3,10,11,...,122,123`。这里的 score 列表示 key，下标含义已经与 Q 的特征列不同。

其扩展坐标规则为：

```text
row = warp*16 + lane//4 + i*8 + mi*64
col = (lane%4)*2 + j + nj*8
i,j ∈ {0,1}; mi ∈ {0,1}; nj ∈ {0,...,15}
```

`attention_cpu.py` 枚举全部 128 线程，检查恰好覆盖 `128×128` 个不同坐标。它验证的是这套坐标公式的覆盖；真实代码对应的 mask 推导、MMA 定义仍应通过 [布局精读 §10](<../appendices/source/flash_attention/03_layout_and_pipeline.md>)核对。

线程 36/37/38/39 持有相同的行、互补的列，因此 Softmax 的 `Allreduce<4>` 是把同一行的部分统计量合并。它既不是任意四条线程，也不是让整块所有线程混合求一个最大值。

## 9.4 追两轮 K/V：哪些状态必须保留

该配置序列长度 256、KV 块高 128，所以每个 block 遍历两块 K/V。在当前实现中顺序是 **后半块 1 → 前半块 0**。CPU 参考已验证反向遍历的算法等价性，但实际浮点累加顺序不同，不能要求逐 bit 一样。

| 阶段 | score 临时状态 | 跨轮状态 | 搬运或同步 |
|---|---|---|---|
| prologue | 尚未计算 | acc_o 清零 | 加载 Q 和 K 块 1 |
| 首轮 QK | 生成块 1 的 acc_s | m/l 尚待首轮 Softmax 初始化 | 等 K，加载 V 块 1，与 QK 存在重叠机会 |
| 首轮 Softmax/PV | acc_s 变指数权重，再成为 rP | 初始化 m/l，acc_o 累加块 1 的贡献 | 等 V；旧 K 消费完后预取 K 块 0 |
| 次轮 QK | 重新清零并计算 acc_s | 原 m/l/acc_o 必须保留 | 等 K 块 0，加载 V 块 0 |
| 次轮 Softmax/PV | 当前块指数权重 | 按新最大值缩放旧 l/acc_o，再加新贡献 | 等 V，完成 PV |
| 收尾 | score 不再需要 | 合并统计量，acc_o 除以 l | 输出转换与写回 |

这张表故意不把 Softmax 构造函数当成“状态已经初始化”：当前实现的首轮分支负责初始化 row_max/row_sum。每轮应清空的是当前 score accumulator，不能连跨轮输出也一起清空。

## 9.5 不同时打开所有文件

按以下顺序阅读，每一步先回答问题再进入下一步：

| 文件或导读 | 当前要回答的问题 |
|---|---|
| [FA2 前向导读 §1–5](<../appendices/source/flash_attention/02_fa2_forward.md>) | Python 参数如何变成指针、stride 与编译实例？ |
| [kernel_traits.h](<../../../flash-attention/csrc/flash_attn/src/kernel_traits.h>) | 这个实例有多少线程、哪些布局、多少 shared memory？ |
| [flash_fwd_kernel.h](<../../../flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h>) 的 `compute_attn_1rowblock` | block 拥有哪些行，首轮加载了什么，循环保留了什么？ |
| [softmax.h](<../../../flash-attention/csrc/flash_attn/src/softmax.h>) 与 [在线 Softmax 精读](<../appendices/source/flash_attention/01_algorithm_and_online_softmax.md>) | m/l/a 的更新各对应哪一段，什么时候跨线程归约？ |
| [utils.h](<../../../flash-attention/csrc/flash_attn/src/utils.h>) 与 [布局精读](<../appendices/source/flash_attention/03_layout_and_pipeline.md>) | `gemm`/`gemm_rs` 的操作数从哪里来，什么时候可复用？ |

第一次可以把 CuTe 已定义的 `copy` 和 `gemm` 当成带明确输入输出的操作；第二次用线程 37 展开分工，第三次才深入模板内部和 PTX。每一遍都应能给数据标上“位置、所有者、生命周期”。

## 9.6 哪些实现变化会破坏算法与依赖

正确实现同时保持数学状态、数据布局与同步依赖：

| 假设改动 | 预期影响 |
|---|---|
| 每轮开始都 clear(acc_o) | 丢掉前面 key 块的输出贡献 |
| 最大值增大时只缩放 l，不缩放 acc_o | 分子分母使用不同指数基准 |
| 把 score 的行归约改为全 block 一个最大值/分母 | 混合不同 query，改变算法 |
| 认为创建 `gQ` 视图就完成了 global→shared 拷贝 | 省掉真实加载，后续读不到应有数据 |
| 在当前 K 仍被读取时预取下一块到同一存储 | 写覆盖未结束的读，产生竞争 |
| 将 rP 每块单独归一化 | 丢失各块在整行中的相对权重 |

这些反例把第 4–8 章的数学状态和执行依赖对应到同一条 FA2 路径。要独立修改性能实现，还需在目标 GPU 上编译、比较数值并用 Sanitizer/profiler 验证。反向与 FA3/FA4 放到这一条路径读通之后，沿现有导读继续即可。
