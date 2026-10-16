# 01 · 从普通 Attention 到在线 Softmax


**第一次阅读范围：§1–7。** §8 的源码精读可以在读过 02、03 后再回来；理解分块公式时还不需要理解寄存器布局。

## 1. Q、K、V 分别参与哪一步

先把 attention 理解为“为每个查询计算一组权重，再按权重汇总信息”。**Q（query）**参与提出查询，**K（key）**参与计算匹配分数，**V（value）**是最终被汇总的内容。K 的第 j 行与 V 的第 j 行一一对应。它们都是数值向量；本函数接收已经准备好的 Q/K/V，不负责把原始文字变成这些向量。

**token** 是序列中的一个位置单位，例如文字切分后的一个片段；**batch** 是一起处理的多条序列；**head** 是一组 attention 计算通道。本节先固定 batch 和 head，只看其中一组计算。

先只考虑一个样本、一个 head（注意力头）。Q 有 $N_q$ 行，每行 d 个数；K 有 $N_k$ 行，每行 d 个数；V 有 $N_k$ 行，每行 $d_v$ 个数。本系列 FA2 常规例子取 $d_v=d$。

公式中 `Σ` 表示沿指定下标求和，`exp(x)=e^x`，`ln(x)` 和未注明底数的 `log(x)` 均表示自然对数；`exp(ln(x))=x`。`i` 是 query 下标，`j` 是 key 下标，`t` 是特征下标。Python 的 `@` 表示矩阵乘法：左边每行与右边每列做点积；它与逐元素的 `*` 不同。

对 Q 的第 i 行，先和每个 K 行做点积。点积就是对应元素相乘再相加：

$$S_{ij}=\alpha\sum_{t=0}^{d-1}Q_{it}K_{jt},\qquad \alpha=1/\sqrt d.$$

这里 $Q_{it}$ 是第 i 个 query 的第 t 个特征，$K_{jt}$ 是第 j 个 key 的同一特征；求和产生一个数 $S_{ij}$，而不是一个向量。Q/K 是模型学到的数值表示，本篇把它们看作无物理单位的实数；指数函数的输入 S 也按无量纲量处理。$d$ 是参与点积的特征数量，$\alpha$ 是模型定义中的缩放系数，不是字节数或硬件 tile 大小。

为什么常用 $1/\sqrt d$？先作一个帮助理解的统计假设：Q/K 的各分量相互独立、均值为 0、方差为 1。方差衡量相对均值的平方偏离，标准差是方差的平方根。在此假设下，每项乘积 $Q_tK_t$ 均值为 0、方差为 1，不同项的方差可以相加。具体说，乘积均值为 0，平方的平均值可按独立性拆成 Q 与 K 各自平方平均值的乘积 `1×1=1`，所以乘积方差为 1。将 d 项相加，再乘 alpha，得到：

$$\operatorname{Var}\left(\sum_{t=0}^{d-1}Q_tK_t\right)=d,
\qquad
\operatorname{Var}\left(\alpha\sum_{t=0}^{d-1}Q_tK_t\right)=\alpha^2d=1.$$

例如 d 从 1 增到 64，未缩放点积的典型幅度从标准差 1 变成 8；乘 1/8 后回到 1。过大的分数差会让 Softmax 权重挤到少数位置。真实模型不一定满足独立、单位方差假设，所以这是尺度选择的动机，不是对任意输入的保证；接口也允许显式指定其他 scale。

Q 与所有 K 的相关程度组成一行 score。Softmax 把这行变成和为 1 的权重，然后对 V 的各行求加权和：

$$P_{ij}=\frac{e^{S_{ij}}}{\sum_{k=0}^{N_k-1} e^{S_{ik}}},\qquad O_{it}=\sum_j P_{ij}V_{jt}.$$

| 中间结果 | 形状 | 一行表示什么 |
|---|---|---|
| Q | $N_q\times d$ | 一个 query 的特征 |
| K | $N_k\times d$ | 一个 key 的特征 |
| S、P | $N_q\times N_k$ | 某个 query 对全部 key 的 score、权重 |
| V | $N_k\times d_v$ | 被加权的向量 |
| O | $N_q\times d_v$ | 该 query 得到的加权结果 |

`QK^T` 中的转置让每一行 Q 能与每一行 K 做点积；它是数学表达，不要求代码真的把整份 K 复制成一份转置矩阵。

### 1.1 从输入到输出的最小例子

取 `Q=[[1]]`、`K=[[0],[ln(3)]]`、`V=[[10],[20]]`。这里只有一个 query、两个 key，每行一个特征，因而 `alpha=1`：

```text
点积与缩放：S = [1*0, 1*ln(3)] = [0, ln(3)]
取指数：    exp(S) = [1, 3]
归一化：    P = [1/4, 3/4]
加权汇总：  O = [(1/4)*10 + (3/4)*20] = [17.5]
```

“归一化”在这里就是除以总和，使这一行权重相加为 1。如果每行 V 改成两个特征，就用同一组权重分别汇总两个特征，得到两个输出数。增加 query 行则重复上述工作，各 query 的 Softmax 分母独立。

具体把 V 改为 `[[10,1],[20,-1]]`，其形状是 `[2,2]`，P 仍是形状 `[1,2]` 的行向量 `[1/4,3/4]`，得到 `O=[[17.5,-0.5]]`，形状 `[1,2]`。Softmax 沿 key 轴归一化；它不在两个输出特征之间再做一次归一化。O 可以为负，也不要求特征相加为 1，因为它是对 V 的加权结果，而非概率数组。

再让 query 和特征都增加到 2，取下面三张 `[2,2]` 矩阵，逐步检查两个矩阵乘的轴：

```text
Q = [[sqrt(2), 0],      K = [[0, 0],         V = [[10,  1],
     [0, sqrt(2)]]          [ln(3), ln(2)]]       [20, -1]]
alpha = 1/sqrt(2)
Q @ K.T = [[0, sqrt(2)*ln(3)], [0, sqrt(2)*ln(2)]]
S       = [[0, ln(3)], [0, ln(2)]]
P       = [[1/4, 3/4], [1/3, 2/3]]
O       = [[17.5, -0.5], [50/3, -1/3]]
```

例如 `O[1,0]=(1/3)*10+(2/3)*20=50/3`，`O[1,1]=(1/3)*1+(2/3)*(-1)=-1/3`。第一轮乘法沿两个特征求点积，第二轮沿两个 key 汇总 V；碰巧两个长度都为 2，不表示这两个轴具有同一含义。

**mask（遮蔽）**规定哪些 query/key 配对可以参与。causal mask 表示因果限制：等长序列中不读取未来位置；local mask 只保留一个位置窗口。被遮蔽位置的权重应为 0。下面先不使用 mask，第 04 章再处理长度不同时的规则。

## 2. 为什么大矩阵会造成问题

一个 head、`N_q=N_k=4096` 时，一张 score 矩阵有 16,777,216 个数。如果每个数 2 字节，就是 32 MiB；Q 在 `d=64` 时只占 0.5 MiB。batch 和 head 还会继续放大这些数字。

朴素实现先计算 S 并写入显存，再读取 S 算 P、写回 P，最后读取 P 乘 V。FlashAttention 的分块实现让当前 score 块在片上使用完即可丢弃。这个 IO 优化出发点来自 [FA1 论文](https://arxiv.org/abs/2205.14135)。完整 dense attention 仍要计算有效 Q/K 配对；节省中间显存不代表计算量变为线性。

这里 **IO** 指数据读写，dense 指不主动省略允许参与的配对。可以把耗时粗分为“搬数”和“算数”：矩阵乘硬件很快时，反复写回、读取大中间矩阵可能成为主要开销。FlashAttention 尽量让中间结果在片上用完；实际加速幅度仍取决于形状、硬件和实现，不能只由省了多少显存推算。

把成本拆开计算更清楚。仍取单 head、等长 $N=4096$、$d=d_v=64$，并约定一次乘加记为 2 次浮点运算（FLOP）：QK 约有 $2N^2d$ 次，PV 也约有 $2N^2d$ 次，共 $4N^2d=4,294,967,296$ 次，不计 Softmax。N 翻倍，配对数量和这部分计算量约变成 4 倍。

另一方面，如果 S/P 均以 2 字节存储，仅“写 S、读 S、写 P、读 P”就涉及 $4\times32=128$ MiB 的逻辑读写；这不是所有实现的实测 HBM 流量，缓存、融合与 FP32 中间值都会改变它。分块去掉这两张全尺寸中间矩阵，却仍要读取 Q/K/V、写 O/LSE，并可能为不同 Q 块重复读取 K/V。因而“中间显存随 N 线性增长”和“执行中总读取字节数随 N 线性增长”不能互换。

## 3. Softmax 先减最大值

直接算 `exp(1000)` 会溢出。对一行 score 取最大值 m，改写为：

$$\frac{e^{s_j}}{\sum_k e^{s_k}}=\frac{e^{s_j-m}}{\sum_k e^{s_k-m}}.$$

分子分母同时乘了同一常数，结果不变。所有指数的输入不大于 0，数值更稳定。问题是扫描第一块 K 时，还不知道未来块的最大值。在线 Softmax 就是在未来出现更大值时修正已有结果。

展开中间一步可见公共因子为何消失：$e^{s_j}=e^m e^{s_j-m}$，因此分母也为 $e^m\sum_k e^{s_k-m}$，约去 $e^m$ 即得上式。对 `[1000,1000+ln(3)]`，直接取指数会出问题，减最大值后是 `[-ln(3),0]`，指数为 `[1/3,1]`，除以 `4/3` 仍得到 `[1/4,3/4]`。

减最大值只消除了大正指数的溢出风险；很负的差值仍可能下溢为 0，输入中的 NaN 也不会被自动修复。后面谈“数学等价”时默认输入有限且本行有有效 key，空行要另定行为。

## 4. 只保存 m、l、a

对一个 query，已扫描的 key 集合上保存：

$$m=\max_j s_j,\qquad l=\sum_j e^{s_j-m},\qquad a=\sum_j e^{s_j-m}v_j.$$

m、l 是标量，a 是长度为 $d_v$ 的向量。a 还没有除以 l，所以这里叫**未归一化输出**。即便扫描了几千个 key，也只需保存这些量。

新块 score 为向量 s，更新步骤是：

$$m'=\max(m,\max s),\qquad r=e^{m-m'},$$
$$p_j^{\mathrm{block}}=e^{s_j-m'},\qquad l'=rl+\sum_jp_j^{\mathrm{block}},$$
$$a'=ra+\sum_jp_j^{\mathrm{block}}v_j.$$

结束时才计算 $O=a/l$。这里 $p^{\mathrm{block}}$ 只是当前共同基准下的指数权重，还不是和为 1 的最终概率。源码里虽然把它命名为 `rP`，也不能据此假设每个 tile 都单独归一化了。

上标 `'` 表示更新后的值，l 是小写字母 L。修正因子可直接由一条旧权重推导：`exp(s_old-m') = exp(s_old-m) * exp(m-m')`。因此所有旧权重都乘同一个 r，旧分母 l 和旧加权分子 a 也都乘 r；无需重新读取旧 score。这里“在线”表示逐块处理、持续维护状态，与联网无关。

### 4.1 为什么三个状态足够，而局部输出不够

设 A 是已经处理的 key 集合，B 是新块，它们不重叠。把合并后的分母按集合拆开，再代入旧状态定义：

$$\begin{aligned}
l' &= \sum_{j\in A\cup B}e^{s_j-m'}\\
   &= \sum_{j\in A}e^{s_j-m}e^{m-m'}+\sum_{j\in B}e^{s_j-m'}\\
   &= rl+\sum_{j\in B}p_j^{\mathrm{block}}.
\end{aligned}$$

对第 t 个输出特征同样展开：

$$\begin{aligned}
a'_t &= \sum_{j\in A\cup B}e^{s_j-m'}V_{jt}\\
     &= r\sum_{j\in A}e^{s_j-m}V_{jt}+\sum_{j\in B}e^{s_j-m'}V_{jt}\\
     &= ra_t+\sum_{j\in B}p_j^{\mathrm{block}}V_{jt}.
\end{aligned}$$

因此每次更新后，m/l/a 都仍满足“对已处理全部 key 的定义”，这是循环的不变量。第一块建立它，后续块保持它，扫描结束就覆盖了全部有效 key。m、l、r 和指数权重均无量纲；a 的各分量与 V 的对应分量同单位，除以无量纲的 l 得到同单位的 O。

如果只保留归一化输出 $O_A=a/l$，就丢掉了 A 的总权重信息。两个局部输出相同，可能一个来自很小的总权重，另一个来自很大的总权重；合并新块时它们应有不同影响。保留 m 和 l 正是在保留这种信息。也可以每步维护归一化 O，但必须做 $O'=(rlO+\sum_Bp_jV_j)/l'$；保存 a、最后只除一次，可少做中间除法与相关逐元素操作。

初始化可在数学上写为 `m=-inf,l=0,a=0`。教学脚本跳过空块，在第一块有效数据到来时令旧贡献为 0；否则直接机械计算 `exp(-inf-(-inf))` 并不合法。GPU 源码通过首轮分支和空值保护落实同一要求。

## 5. 四个 key 的完整手算

设缩放后的 score 为 `[0, ln(2), ln(4), ln(8)]`，V 的四个标量为 `[1, 2, 3, 4]`，每次处理两个 key。用标量 V 只是为了手算；真实 V 每行是向量，各分量遵循相同规则。

第一块：

- 最大值 `m = ln(2)`。
- 相对指数为 `[1/2, 1]`，所以 `l = 1.5`。
- `a = (1/2)×1 + 1×2 = 2.5`。

第二块的最大值升至 `ln(8)`，旧状态要乘 `r = exp(ln(2)-ln(8)) = 1/4`：

| 量 | 旧块缩放后的贡献 | 新块贡献 | 合并结果 |
|---|---:|---:|---:|
| l | `1.5/4 = 0.375` | `1/2 + 1 = 1.5` | `1.875` |
| a | `2.5/4 = 0.625` | `(1/2)×3 + 1×4 = 5.5` | `6.125` |

最终 `O = 6.125 / 1.875 = 49/15 ≈ 3.266667`。一次性计算的权重是 `[1,2,4,8]/15`，结果同样为 `(1+4+12+32)/15`。不缩放旧 a/l 就得不到这个答案。

这里可以用两个错误做对照。每块分别 Softmax 得到 `O_A=5/3`、`O_B=11/3`，直接平均为 `8/3`，并非 `49/15`；相同的 key 个数不代表相同的概率总质量。若第二块只缩放 l、不缩放 a，则输出变为 `(2.5+5.5)/1.875=64/15`，甚至大于所有 V 的最大值 4。对于非空行、无 dropout 的 Softmax 加权和，权重非负且和为 1，每个输出特征都应位于该特征输入值的最小值与最大值之间；这个性质可用于手算自查。

这个手算顺序是 key 块 0→1；第 02 章真实 kernel 使用 1→0。反过来算时，先得到 `(m,l,a)=(ln(8),1.5,5.5)`，再加上相对于 `ln(8)` 的旧半块贡献：分母 `1/8+1/4=0.375`，分子 `1/8*1+1/4*2=0.625`。最终仍为 `(ln(8),1.875,6.125)`。本例两块均有有效 key；空块处理在 §7 说明。

[`examples/online_attention.py`](<examples/online_attention.py>) 会输出这两步的状态，并比较分块结果与普通 attention。

## 6. 对照 `softmax.h` 的真实状态

[`softmax.h`](<../../../../../flash-attention/csrc/flash_attn/src/softmax.h>) 的 `Softmax::softmax_rescale_o` 将这一公式直接落实到状态更新。下面是其非首块分支中的**源码摘录**：

```cpp
float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
row_sum(mi) *= scores_scale;
for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
    acc_o_rowcol(mi, ni) *= scores_scale;
}
```

它正是在把旧 l 和旧 a 乘上 r。`mi` 是当前线程所持有片段中的行索引，`ni` 是片段的列索引；不是全矩阵坐标。

| 数学量 | 代码中的对应物 | 需要注意 |
|---|---|---|
| m | `row_max` | 基础分支保存的是未乘 scale 的 score 最大值 |
| l | `row_sum` | 中间阶段是线程局部部分和，最终还要跨线程归约 |
| a | `acc_o` / `acc_o_rowcol` | 由多个线程的寄存器片段共同组成 |
| 当前 score 块 | `acc_s` / `scores` | 同一片存储随后被改写为指数权重 |
| r | `scores_scale` | 最大值基准变化时的旧状态缩放 |
| O | 归一化后的 `acc_o` | 最后转换成输出 dtype 并写回 |

为什么看到 `exp2f` 而非 `exp`？因为 $e^x=2^{x\log_2 e}$。基础分支令 `softmax_scale_log2 = alpha × log2(e)`，将缩放和换底一起完成。公式里的 s 已经乘过 alpha，而源码的原始 `acc_s` 尚未乘；对照时要保持同一单位。

`reduce_max` 会跨一组协作线程合并最大值，让它们采用相同基准；`reduce_sum` 在这里先保存线程局部和，`normalize_softmax_lse` 再做最终跨线程求和。第 03 节解释为什么这样仍成立。

**reduction（归约）**指把多个值合成少量统计量，例如 `[1,4,2]` 求和变成 7，求最大值变成 4。它既可以在一个线程内部完成，也可以由多个线程合作完成；`allreduce` 表示参与者最后都拿到合并结果。

## 7. LSE 与边界条件

LSE 是 **log-sum-exp**：

$$LSE=\log\sum_j e^{s_j}=m+\log l.$$

它每行只有一个数。反向时可通过 $P_{ij}=e^{S_{ij}-LSE_i}$ 重算概率，无需保存整张 P。

其来源是 $\sum_j e^{s_j}=e^m l$，取自然对数后得到 $\log(e^m l)=m+\log l$。四 key 例子的 LSE 为 `ln(8)+ln(15/8)=ln(15)`；重算第三个概率为 `exp(ln(4)-ln(15))=4/15`。LSE 自身不是概率，也不是 l；它可以大于 1。一个含 $B_M$ 个 query 行的 tile 只需 $B_M$ 个 LSE 数，而完整 P 要 $B_MN_k$ 个数。

被 mask 的 score 通常视作负无穷，指数贡献为 0。但一整行都无效时，`-inf - (-inf)` 会产生 NaN，且 `a/l` 是 `0/0`。真实代码有专门处理：输出为零；FA2 普通路径与 split 路径在无效行的 LSE 哨兵值上还可能不同。我们的教学脚本统一用 `-inf` 表示空行 LSE，不将其当成所有后端的逐位契约。

“哨兵值”是表示特殊情况的约定值，不是有效行套公式得到的普通数。有效行的分块公式在实数运算中与普通 attention 等价；GPU 使用有限精度和不同的乘加顺序，因此应按误差容限比较，而不是要求所有后端逐位相同。

## 8. 源码精读：`Softmax` 对象的整个状态更新

前面的公式现在可以直接对照实现。以下逐段覆盖首块、后续块、指数运算、线程间归约以及最终输出，不需要自行去源文件寻找未展开的数学步骤。

### 8.1 对象没有把统计量预先清零

源码摘录：[softmax.h，128–134 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/softmax.h#L128-L134)。

<!-- source: csrc/flash_attn/src/softmax.h:128-134 -->
```cpp
template <int kNRows>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};
```

`kNRows` 是**当前线程负责的行片段数**；`TensorT` 在每条线程中保存这一小组 FP32 值。`row_max`、`row_sum` 是对象的两个状态，构造函数体为空。因此正确性依赖第一轮明确初始化，而不能在一个新对象上直接执行 `Is_first=false`。

完整 `128×128` score tile 是由整个 block 持有的逻辑结果，不能把此处 `kNRows` 理解成 128，也不能认为每条线程各分配一整份矩阵。

### 8.2 首块：先让四条协作线程得到同一最大值

源码摘录：[softmax.h，137–145 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/softmax.h#L137-L145)。

<!-- source: csrc/flash_attn/src/softmax.h:137-145 -->
```cpp
__forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0>(scores))::value == kNRows);
    if (Is_first) {
        FLASH_NAMESPACE::template reduce_max</*zero_init=*/true>(scores, row_max);
        FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
        FLASH_NAMESPACE::reduce_sum</*zero_init=*/true>(scores, row_sum);
    } else {
```

`make_tensor(acc_s.data(), ...)` 不分配另一份 score；它把 MMA 的三维片段视图重排成“本线程持有的行×列”，让后续代码可以按行操作。随后三句依次做：

1. `reduce_max<true>`：用当前 score 初始化最大值，包含线程内和四线程组内的最大值归约。
2. `scale_apply_exp2`：把 score 原地替换成相对于最大值的指数权重。
3. `reduce_sum<true>`：从当前指数权重初始化线程局部和。

第一句的 `<true>` 对应 `zero_init=true`，意思是**不合并旧 summary，而从当前数据开始**；对于 max 操作，并不是“先与数字 0 比较”。否则全为负数的 score 会得到错误最大值。

首轮无需缩放旧输出，因为调用者已经将 `acc_o` 清零。但本函数仍不计算 PV；它只把 `acc_s` 准备成下一次 GEMM 的权重。

### 8.3 后续块：旧最大值必须先复制出来

源码摘录：[softmax.h，146–166 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/softmax.h#L146-L166)。

<!-- source: csrc/flash_attn/src/softmax.h:146-166 -->
```cpp
    Tensor scores_max_prev = make_fragment_like(row_max);
    cute::copy(row_max, scores_max_prev);
    FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, row_max);
    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
    static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
    #pragma unroll
    for (int mi = 0; mi < size(row_max); ++mi) {
        float scores_max_cur = !Check_inf
            ? row_max(mi)
            : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
        float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
        row_sum(mi) *= scores_scale;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
    }
    FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
    // We don't do the reduce across threads here since we don't need to use the row_sum.
    // We do that reduce at the end when we need to normalize the softmax.
    FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
}
```

`make_fragment_like(row_max)` 给旧最大值分配一个同布局片段，`cute::copy` 保存旧值；下一句 `reduce_max<false>` 才把新块最大值并入 `row_max`。如果没有这份旧值，更新之后只能看到新 m，就算不出旧状态的修正因子。

`acc_o_rowcol` 同样只是旧输出寄存器的另一种视图。`scores_scale` 对应 $r=e^{\alpha(m_{old}-m_{new})}$；源码最大值还未乘 alpha，所以在指数里带着 `softmax_scale_log2`。

以本节 alpha=1 的四 key 手算为例，进入第二块时 `scores_max_prev=ln(2)`，更新后的 `row_max=ln(8)`，得到 `scores_scale=1/4`。代码让每条线程的 `row_sum` 部分和乘 1/4，并让该线程所有相应输出列乘 1/4。所有片段一起完成后，就是整行 l/a 的正确缩放。

最后再次把新 score 改成指数权重。`reduce_sum<false>` 含义变成“在已经修正过的旧部分和上继续加”，不需要再另存一份 l。

`Check_inf` 是避免全遮蔽 tile 产生 `-inf-(-inf)` 的保护：当新最大值仍是负无穷时，用 0 作为计算用的安全基准。它并不把无效 score 变成有效 score；后面的指数仍为 0。

### 8.4 `exp2` 如何同时实现缩放与稳定化

源码摘录：[softmax.h，73–89 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/softmax.h#L73-L89)。

<!-- source: csrc/flash_attn/src/softmax.h:73-89 -->
```cpp
// If max is -inf, then all elements must have been -inf (possibly due to masking).
// We don't want (-inf - (-inf)) since that would give NaN.
// If we don't have float around M_LOG2E the multiplication is done in fp64.
const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
#pragma unroll
for (int ni = 0; ni < size<1>(tensor); ++ni)  {
    // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
    // max * log_2(e)) This allows the compiler to use the ffma
    // instruction instead of fadd and fmul separately.
    // The following macro will disable the use of fma.
    // See: https://github.com/pytorch/pytorch/issues/121558 for more details
    // This macro is set in PyTorch and not FlashAttention
    #ifdef UNFUSE_FMA
        tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
    #else
        tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
    #endif
```

本节前向传入的 `scale=alpha×log2(e)`，`Scale_max` 默认为 true，故每个 score 变成：

$$2^{x\alpha\log_2e-m\alpha\log_2e}=e^{\alpha(x-m)}.$$

`tensor(mi,ni)=...` 是原地赋值：该片段旧 score 从这里开始就不存在了，下一次读到的是指数权重。因为后续只需权重和统计量，保留旧 score 没有必要。

当 max 为负无穷时 `max_scaled=0`，于是无效 score 算 `exp2(-inf-0)=0`。`UNFUSE_FMA` 两个编译分支控制乘加的舍入方式：一个显式分开乘法，另一个允许融合。它们表达同一数学式，但低位舍入可能不同。

### 8.5 为什么 `reduce_max` 和 `reduce_sum` 看起来不对称

源码摘录：[softmax.h，47–63 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/softmax.h#L47-L63)。

<!-- source: csrc/flash_attn/src/softmax.h:47-63 -->
```cpp
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}
```

`reduce_` 先做线程内归约，再做 `quad_allreduce_`。`reduce_max` 调用它，因此四条线程会及时拿到统一 m。`reduce_sum` 只调用 `thread_reduce_`，所以此时每条线程的 l 是其列片段的部分和。

这可以正确工作，是因为 m 已统一。假设同一行在四条线程中分成列集合 A/B/C/D，每条线程先保有 `l_A/l_B/l_C/l_D`。下一 tile 更新时，它们用同一个 r 缩放各自 l，再加各自新列指数和。循环结束合起来就得到完整 l；无需每轮先交换 l。

`quad_allreduce_` 底层使用下列递归：

源码摘录：[utils.h，111–131 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/utils.h#L111-L131)。

<!-- source: csrc/flash_attn/src/utils.h:111-131 -->
```cpp
template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};
```

对四线程组而言，先用 XOR 2 配对，再用 XOR 1 配对。若四条线程的局部最大值为 `[1,4,2,3]`，第一步得到 `[2,4,2,4]`，第二步均为 4。shuffle 交换的是寄存器值，没有先写一张 shared memory 矩阵。

这个“4”来自当前 MMA 行数据在 lane 间的分布；第 03 节会从实际 mask 坐标公式推导为什么恰好是这四条线程协作。

### 8.6 最终归一化同时产出 LSE

源码摘录：[softmax.h，170–187 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/softmax.h#L170-L187)。

<!-- source: csrc/flash_attn/src/softmax.h:170-187 -->
```cpp
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };
};
```

第一条 `quad_allreduce_` 把部分和合成完整 l。`inv_sum=1/l`，每个输出列乘这个数，未归一化 a 才成为 O。`lse=row_max×softmax_scale+log(sum)` 正是已统一单位的 $m+\log l$。

无 dropout 时 `scale=inv_sum`；有 dropout 时再乘 `rp_dropout`，补偿概率丢弃的缩放。我们的基础主线中它为 1。

`sum==0` 表示可能没有有效 key。代码选择安全倒数 1，零输出累加器仍为零，并写 LSE 哨兵：普通路径为正无穷，split 局部路径为负无穷。`sum != sum` 则是 NaN 检测，但不应据此认为本函数能修复任意上游 NaN。

返回值是 LSE 片段；O 是通过引用参数 `acc_o` **原地修改**的。调用方随后分别写回它们。这一点很重要：如果只看 `return lse`，容易误以为函数只在计算 LSE，没有完成输出归一化。

## 9. 常见误解与边界

- 为什么只缩放 l、不缩放 a 不行？**两者必须使用同一指数基准，否则比例改变。**
- 每块单独 softmax，再把输出平均能否得到最终答案？**一般不能；各块归一化分母不同。**
- 从后往前扫描 K 会改变数学结果吗？**实数运算下不变；浮点舍入可能变化。**
- `rP` 的行和必须是 1 吗？**不必，它可以是尚未最终归一化的指数权重。**

下一节把这些变量放回一条真实的 Python → CUDA 调用链。
