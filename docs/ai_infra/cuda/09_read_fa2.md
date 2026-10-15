# 9. 从在线状态读一条固定的 FA2 前向路径

[上一章](<08_async_pipeline.md>) · [目录](<README.md>) · [继续源码精读](<../appendices/source/flash_attention/README.md>)

本章负责把前文的算子与布局分析接到真实代码，先限定一个可完整追踪的目标：**FP16、固定长度、无 dropout、无 causal 的 FA2 前向**。不要求同时读懂 backward、变长、分页、split-KV 或 FA3/FA4。

源码基准为本项目 [flash-attention](<../../../flash-attention>) 的 commit `e9cf2c1651d2303191eb40a739a3c135fda00999`；以下硬件原语讨论针对 SM80 分支。输入为 `Q/K/V.shape=(2,256,4,64)`，连续布局，`dropout_p=0`、`causal=False`，无额外 bias/softcap。

四个维度依次是 batch 数、序列长度、head 数、每个 head 的特征维度：每个样本有 256 个 token，每个 token 在四个 head 上各有 64 个特征。输出 O 具有相同形状。dropout 是训练中随机丢弃部分权重的操作，这里关闭；bias 是附加到分数上的偏置，softcap 是限制分数幅度的一种变换，这里也不启用。这样第 6 章的数学语义可以直接作为对照。

前文已解决“怎样不存整张 S/P”的问题，FA2 还要解决“怎样让同样的算法在硬件上高效执行”。本章会看见三类实现选择：多个 query 组成一个计算块以复用 K/V，矩阵乘使用 MMA，跨轮分母只在必要时合并并在最后归一化。FA2 的工作划分与减少非矩阵运算的动机见 [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)；下面的具体布局、常量与执行顺序则以本地固定提交为准。

## 9.1 先认领你要追踪的 block

根据 [flash_fwd_launch_template.h](<../../../flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h>) 的 `run_mha_fwd_hdim64`，无 dropout 分支选择：

```cpp
Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, T>
```

依次表示 head_dim=64、Q 块高=128、KV 块高=128、4 个 warp、`Is_Q_in_regs_=false`、`Share_Q_K_smem_=false`，T 是输入元素类型。第一个布尔选项没有要求把整个 Q tile 持续保存在寄存器中，第二个不让 Q/K 共用同一片 shared；MMA 仍会按需把 Q 操作数片段读入寄存器，false 不是“Q 从不进寄存器”。当前配置每块 128 线程，一个 block 拥有一个 batch/head 的 128 行 Q 输出。grid 为 `(ceil(256/128),2,4)=(2,2,4)`，共 16 blocks。

这与第 6 章“一块一个 query”不同：现在一个 block 同时处理 128 个 query，以便在矩阵乘中共享数据和使用 Tensor Core。

具体选 blockIdx=(1,1,2)：x=1 表示第二块 query，y=1 表示第二个样本，z=2 表示第三个 head。它拥有 `O[1,128:256,2,:]`，共 `[128,64]` 个输出；遍历全部 256 个 key 后才完成这些输出。其他 block 拥有不重叠的输出切片，因此这条路径不需要跨 block 归约 O。

连续 `(batch,seq,head,dim)` 布局的元素偏移为：

```text
offset(b,q,h,c) = ((b*256+q)*4+h)*64+c
stride = (65536,256,64,1)  # 元素数；FP16 字节偏移再乘 2
```

固定 head 后，相邻 query 行的起点相隔 4×64=256 个元素，并不是 64。完整张量连续，不代表其中的每个 `[seq,dim]` head 切片都紧凑连续。当前 block 的首元素偏移是 `65536+128*256+2*64=98432`；源码通过传入的 row/head/batch stride 保留这层地址关系。

将配置映射到硬件，一个 block 的四个 warp 在一个 SM 上协作；当前 `Share_Q_K_smem=false`，三个 FP16 Q/K/V tile 各为 `128*64*2=16 KiB`，合计 48 KiB shared/block。score 和输出 accumulator 由线程的寄存器片段持有，实际寄存器分配与可能的 spill 要查编译报告，不能只由 shared 数量推算 occupancy。

每轮的形状链为 `Q[128,64] @ Kᵀ[64,128] -> score[128,128]`，随后指数权重 `[128,128] @ V[128,64] -> acc_o[128,64]`。score 是当前 KV 块的临时结果，acc_o 则跨两块保留。按逻辑值数，score 有 16384 个 FP32 值，分给 128 线程平均每线程 128 个；acc_o 有 8192 个，平均每线程 64 个。这说明寄存器需求很可观，但不能据此断言编译结果恰好是 192 个寄存器/线程：地址、统计量、操作数片段、生命周期复用和 spill 都会影响实际分配。第 1 章的资源预算仍然适用。

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

令 r_j 为原始 QK 点积，sigma=`1/sqrt(64)=1/8`，数学分数 s_j=sigma*r_j。因为 sigma>0，原始最大值 m_r=max(r_j) 对应分数最大值 m=sigma*m_r。由 `e^x=2^(x*log2(e))`，代码要计算的是：

```text
exp(s_j-m) = exp(sigma*(r_j-m_r))
           = exp2((r_j-m_r)*sigma*log2(e))
scale_softmax_log2 = sigma*log2(e)
```

`log2(e)≈1.442695` 是换底常数，无量纲。例如原始 r=`[0,8*ln(2)]`，缩放后 s=`[0,ln(2)]`；以 m_r=8*ln(2) 为基准，第一项 exp2 的输入是 `−ln(2)*log2(e)=−1`，得到 1/2，第二项得到 1。若先把 score 乘 1/8，之后又使用带 1/8 的 scale，就错误地缩放了两次。

同理，旧状态的缩放为 `exp2((m_r_old-m_r_new)*scale_softmax_log2)`。最终自然对数域的 LSE 则是 `sigma*m_r+ln(l)`；不能把未经缩放的 m_r 直接代入第 6 章的 `m+ln(l)`。数学上这些表达等价，有限精度下乘减是否融合、rP 转成 FP16 等选择仍会引入舍入差异。

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

推导这份搬运量：一行 64 个 FP16，每线程一次搬 8 个，即 16 字节，所以需要 8 线程/行；128 线程一次覆盖 16 行，而 tile 高 128，需沿行重复 8 次。线程 37 每次搬 8 个、共 8 次，总计 64 个元素，128 线程总计 8192 个，恰好覆盖 `[128,64]`。对刚才 blockIdx=(1,1,2)，其首个元素是 Q[1,132,2,40]，global 元素偏移 `98432+4*256+40=99496`，FP16 字节偏移为 198992。shared 的物理偏移还要经过 swizzle，不能把这个 global 公式原样当成 shared 地址。

而该线程持有 score 的局部行是 `17,25,81,89`，每行持有列 `2,3,10,11,...,122,123`。这里的 score 列表示 key，下标含义已经与 Q 的特征列不同。

其扩展坐标规则为：

```text
row = warp*16 + lane//4 + i*8 + mi*64
col = (lane%4)*2 + j + nj*8
i,j ∈ {0,1}; mi ∈ {0,1}; nj ∈ {0,...,15}
```

`attention_cpu.py` 枚举全部 128 线程，检查恰好覆盖 `128×128` 个不同坐标。它验证的是这套坐标公式的覆盖；真实代码对应的 mask 推导、MMA 定义仍应通过 [布局精读 §10](<../appendices/source/flash_attention/03_layout_and_pipeline.md>)核对。

代入 warp=1、lane=5：`warp*16+lane//4=17`，i 取 0/1 增加 0/8，mi 取 0/1 增加 0/64，得到 17、25、81、89；列起点为 `(5%4)*2=2`，j 取 0/1，再由 nj 沿列每次推进 8。于是每线程是 4 行×32 列=128 个 score。对第二个 Q tile、首轮 KV 块 1，这些局部坐标的第一项实际是 query 145 对 key 130 的分数，与搬运的 Q 行 132 并不相同：装载者把数据交到 shared 后，计算者按另一份分工取用它。

线程 36/37/38/39 持有相同的行、互补的列，因此 Softmax 的 `Allreduce<4>` 是把同一行的部分统计量合并。它既不是任意四条线程，也不是让整块所有线程混合求一个最大值。

all-reduce 表示合并后每个参与者都得到结果，与第 4 章 down-shuffle 只在 lane 0 得到完整结果不同。这里最大值必须及时合并，让四条线程使用共同指数基准；row_sum 则可以暂存各线程负责列的局部和。假设某行四条线程的旧局部和是 `[1,2,3,4]`，共同 alpha=1/2，新块各自贡献 `[4,3,2,1]`，它们分别更新成 `[4.5,4,3.5,3]`。最后合并为 15，等于 `1/2*(1+2+3+4)+(4+3+2+1)`。这是分配律允许的延迟归约，省掉了中间轮次不必要的跨线程求和；若四线程的 alpha 不同就不能这样解释。

与此同时，acc_o 通过 PV 的 MMA 累加对应输出通道的贡献，并没有每轮除以分母。收尾的 `normalize_softmax_lse` 才合并 row_sum 并归一化。这正是第 6 章“保留未归一化分子”在真实代码中的意义；把中间 row_sum 当成整行分母，会读错这段实现。

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

用一行上的两组常量核对反向遍历：设 Q[0]=1、其他 63 个分量为零；前半 128 个 key 的第一分量为 8*ln(2)，后半为 0，其他分量也为零。则除以 sqrt(64)=8 后，前半 score 全为 ln(2)，后半全为 0；再令某个 V 通道前半全为 3、后半全为 1。首轮先处理后半，概念上的整行状态为 `m=0,l=128,a=128`。第二轮 m 升到 ln(2)，alpha=1/2，新块权重都为 1，于是 `l=64+128=192`、`a=64+384=448`，输出 `448/192=7/3`。直接计算也是 `(128*1+128*2*3)/(128+128*2)=7/3`。这是实数手算值，实际 FP16 输入转换后需容差比较；源码里的 m_r 与局部 row_sum 再按前两节对应。

再检查两类 shared 的复用条件。首轮 QK 结束并在块内协调后，K 块 1 不再被读，可以向 sK 搬 K 块 0；这时 V 块 1 仍需供首轮 PV 读取，不能一起覆盖。进入下一轮，在上轮 PV 的消费者已越过相应 block barrier 后，sV 才接收 V 块 0。Q 则两轮都要使用，当前 traits 下保持在自己的 shared 区域。`cp_async_wait<0>` 确保搬运完成，barrier 协调各线程阶段；没有任一条单独的指令可以代替整套生命周期证明。

收尾还包含一次布局交接：FP32 acc_o 除以分母后转换成 FP16 rO，借用原 sQ 的 shared 存储建立 sO 视图，按输出布局写入；块内同步后，再由负责 global 写回的线程读取 sO。当前 Q/K 不共享存储，最后一轮 QK 后已有 wait 与 block barrier，随后 Softmax/PV 不再读取 Q，所以写 sO 前无需另加一道保护 Q 读者的 barrier。其他配置要重新核对读写依赖；源码对 `Share_Q_K_smem` 分支另有 barrier，不能推广成“任意复用 shared 都能删 barrier”。输出写回完成后 block 的 shared/寄存器资源可回收，global O/LSE 仍保留给后续使用者；host 使用 O 仍需相应的完成依赖。

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
