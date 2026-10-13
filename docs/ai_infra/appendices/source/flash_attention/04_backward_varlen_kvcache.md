# 04 · 反向、变长与 KV cache


本章有两条独立主线：**训练**按 §1–4 → §10 阅读；**推理与存储**按 §5–9 → §11 阅读。尚未接触梯度的读者，可先完成下面的小例子再读矩阵公式。

## 1. backward 是在求什么

训练时，模型输出最终会影响一个标量 loss（损失）。`dO[i,t]` 表示 loss 对输出元素 `O[i,t]` 的变化有多敏感。attention 的 backward 接收 dO，算出 loss 对输入 Q/K/V 的敏感度 dQ/dK/dV，交给前一层继续传播。

第一遍可以把梯度看作“从后面传来的同形状数组”。`dQ` 与 Q 同形状，`dK` 与 K 同形状，`dV` 与 V 同形状。`torch.autograd.Function` 的 `ctx` 保存前向结束后、反向开始前仍需保留的状态。

### 1.1 梯度的一个数到底是什么意思

若 `y=3*x`、`loss=y²`，在 `x=2` 时，`y=6`，loss 对 y 的梯度为 `dY=2*y=12`；y 对 x 的变化率是 3，因此 `dX=12*3=36`。这就是链式法则：沿计算依赖把后面传来的梯度乘以前面这一步的变化率；如果有多条影响路径，就把贡献相加。

这里 `dX` 是 loss 对 x 的导数，不是把变量 x 乘上字母 d，也不是下一次要给 x 加上的更新量。改变 x 一个很小的量 `epsilon`，loss 的变化近似为 `dX*epsilon`。优化器怎样用这个梯度更新模型参数，属于 attention backward 之后的工作。

attention 接收的 `dO` 同样来自后面的计算。测试中可以直接指定 `dO=G`，等价于选择一个简单损失 `loss=sum(O*G)`，方便独立检查这个算子。

LSE（log-sum-exp）是 log(∑exp(xᵢ))，可写成 m+log(∑exp(xᵢ−m))，其中 m=max(xᵢ)。它把指数归一化的分母放到对数域计算，减最大值可避免大正数的指数溢出。

第 02 节已展开 [`FlashAttnFunc.forward`](<../../../../../flash-attention/flash_attn/flash_attn_interface.py>) 的 `ctx.save_for_backward(...)`：当需要梯度时保存 Q、K、V、O、LSE、随机数状态。这里没有保存一张完整的 P。问题于是变成：反向需要概率时从哪里获得它？

## 2. LSE 让局部概率可以重建

因为 $LSE_i=\log\sum_j e^{S_{ij}}$，所以

$$P_{ij}=e^{S_{ij}-LSE_i}.$$

重新算当前 tile 的 $Q_iK_j^T$，做与前向一致的 scale/mask，再减去保存的逐行 LSE，就能还原这一 tile 的概率。只要当前 tile 用完就释放，不需要在显存里保存完整 P。

在 [`flash_bwd_kernel.h`](<../../../../../flash-attention/csrc/flash_attn/src/flash_bwd_kernel.h>) 的 `compute_dq_dk_dv_1colblock` 中，先用 `gemm` 重算 `acc_s`，再调用：

```cpp
FLASH_NAMESPACE::scale_apply_exp2</*scale_max=*/false>(
    scores, lse, params.scale_softmax_log2);
```

这是**源码摘录**。`scale_max=false` 提醒 helper：这里传入的统计量是已经包含 score 缩放的 LSE，不是前向里的原始最大值。指数仍采用换底后的 `exp2`。理解了这句，就找到了重计算在代码中的直接落点。

## 3. 用三步理解梯度公式

以下公式暂不含 dropout、softcap、位置偏置；S 定义为**缩放后**的 score，$S=\alpha QK^T$。

先从 $O=PV$ 传播：

$$dV=P^TdO,\qquad dP=dOV^T.$$

它们都是矩阵乘。接着通过逐行 softmax：

$$D_i=\sum_jP_{ij}dP_{ij},\qquad dS_{ij}=P_{ij}(dP_{ij}-D_i).$$

D 是每行一个数。展开 dP 后得到一个实用恒等式：

$$D_i=\sum_t dO_{it}\left(\sum_jP_{ij}V_{jt}\right)=\sum_t dO_{it}O_{it}.$$

所以 D 可以直接由 O 与 dO 求点积，无需先保存整行 P 或 dP。最后回到 Q/K：

$$dQ=\alpha\,dSK,\qquad dK=\alpha\,dS^TQ.$$

### 3.1 把这些矩阵式接回一个能手算的例子

沿用第 01 章的 `Q=[[1]]、K=[[0],[ln(3)]]、V=[[10],[20]]`，前向得到 `P=[1/4,3/4]、O=17.5`。选 `loss=O`，因此 `dO=1`：

| 步骤 | 代入结果 | 为什么 |
|---|---|---|
| `dV = P.T @ dO` | `[0.25, 0.75]` | 第一个 V 增加一点，O 只增加其 1/4；第二个增加其 3/4 |
| `dP = dO @ V.T` | `[10, 20]` | 若单独改变某个权重，其对应的 V 决定 O 的变化率 |
| `D = sum(P*dP)` | `17.5` | 这是 dP 按当前概率求出的加权平均，也等于 `O*dO` |
| `dS = P*(dP-D)` | `[-1.875, 1.875]` | Softmax 的权重相互关联：提高一个位置的 score 会改变其他位置的权重 |
| `dQ = alpha*dS @ K` | `[1.875*ln(3)]` | 两条 key 路径对同一个 query 的贡献相加 |
| `dK = alpha*dS.T @ Q` | `[-1.875, 1.875]` | 各 key 接收各自 score 路径的贡献 |

在独立计算 `dP` 时暂把 P 看成可变的中间数组；随后 `dS` 才把“P 必须由 Softmax 产生”的关系纳入。这里公式 `P*(dP-D)` 的 `*` 是逐元素乘法，D 按行重复用于该行的每一列。文中的 `dP/dS` 是整块逻辑矩阵，真实 kernel 仍只维护当前 tile 的线程片段。

| 数学工作 | 代码观察点 |
|---|---|
| 计算逐行 D | [`flash_bwd_preprocess_kernel.h`](<../../../../../flash-attention/csrc/flash_attn/src/flash_bwd_preprocess_kernel.h>)：`compute_dot_do_o` |
| 重算 score 与 P | `flash_bwd_kernel.h`：`acc_s`、`scores`、`lse` |
| 计算 dP | 同文件：`acc_dp` 的 GEMM |
| 做 `P * (dP - D)` | 同文件：`pointwise_mult`、`dP_sum` |
| 计算 dQ/dK/dV | 同文件：`acc_dq`、`acc_dk`、`acc_dv` |
| 汇总并转输出类型 | [`flash_bwd_launch_template.h`](<../../../../../flash-attention/csrc/flash_attn/src/flash_bwd_launch_template.h>)：`flash_bwd_convert_dq_kernel` |

`dP_sum` 就是上面的逐行 D，不是一张与 P 同大小的矩阵。CUDA 中 scale 可能合并到最终转换等步骤，逐句对照时不要因为某一条 GEMM 前未见 alpha 就认定遗漏。

## 4. 为什么反向有多个 kernel 和原子加

前向按 Q 行块分工，通常每个输出位置由其所属 block 完成。反向的当前常规实现按 K/V 列块并行，见 `run_flash_bwd_seqk_parallel`：先计算 D 等预处理状态，再启动主计算，最后转换/合并 dQ。

一块 K/V 对自己的 dK/dV 累加有较好的局部性，但它也会对很多 Q 行的 dQ 贡献一部分。多个 block 可能更新同一 dQ 地址。普通“读旧值、加贡献、写回”会发生丢失更新，所以非确定性路径使用 `atomicAdd` 等原子累加手段。

原子加保证并发更新不会简单覆盖彼此，却不保证不同运行的浮点加法顺序完全相同。浮点加法不满足严格结合律，所以低位可能变化。`deterministic=True` 使用额外分开的 dQ 累加缓冲区和受控的后处理，代价是更多内存和时间；它也不承诺与另一种 attention 实现逐位一致。

重计算的意义是用局部矩阵乘换掉巨大的中间显存读写。FA1 的训练设计和当前反向代码都能帮助理解这种取舍，但不能由“少保存 P”推断 backward 完全没有临时缓冲区。

## 5. 变长样本：用前缀和找到起点

两条序列分别有 2 和 3 个 token。padding 方式把它们都补到 3；varlen（variable length，变长）方式把有效 token 串接为 5 行：

```text
扁平 token 下标   0  1 | 2  3  4
属于哪条样本     样本0 |  样本1
cu_seqlens       [0, 2, 5]
```

`cu_seqlens` 是累计长度数组，通常为 GPU 上的 int32，长度为 `batch+1`。第 b 条的起点是 `cu_seqlens[b]`，长度是 `cu_seqlens[b+1]-cu_seqlens[b]`。Q/K 长度不同时各有一份数组，不能默认它们相等。`max_seqlen_q/k` 给出长度上界，辅助启动和边界处理。

Python `FlashAttnVarlenFunc` 走 `varlen_fwd/varlen_bwd`。[`block_info.h`](<../../../../../flash-attention/csrc/flash_attn/src/block_info.h>) 的 `BlockInfo` 读取累计长度，算出 `sum_s_q`、`actual_seqlen_q` 等。`q_offset` 在 varlen 场景下使用 `sum_s_q * row_stride`；固定长度场景则使用 `bidb * batch_stride`。

“把样本接起来”只改变存储方式。每条序列的 attention 范围仍独立，样本 1 不能看到样本 0 的 K。这个约束由样本偏移、实际长度和 block 范围共同保证。

## 6. causal mask 的右下对齐

等长的 **causal 自注意力**中，query i 只能看 key `j <= i`；关闭 causal 的主线例子则没有这个限制。**decode（逐步生成）**阶段常常只有几个新 query，却有很长的历史 K。FA2 用右下对齐规则：

$$j\le i+N_k-N_q.$$

`N_q=2,N_k=5` 时，允许矩阵是：

```text
1 1 1 1 0
1 1 1 1 1
```

`N_q=5,N_k=2` 时则为：

```text
0 0
0 0
0 0
1 0
1 1
```

前 3 行完全被遮蔽，输出为零。若用普通左上角 `tril` 作非方阵参考，会得到不同答案。这是接口 docstring 明确规定、初学者最容易在测试中弄错的地方。

## 7. GQA：多个 Q head 共享同一组 K/V

MHA 是多头注意力；GQA 是分组查询注意力；MQA 是只有一个 KV head 的情况。假设 Q 有 6 个 head，KV 有 2 个，那么 Q head 0、1、2 使用 KV head 0；Q head 3、4、5 使用 KV head 1。

前向源码选择 K/V head 时使用 `bidh / params.h_h_k_ratio`，这里比例为 3，是整数除法。由此能看出该路径根据 head 编号寻址，而不必把全部 K/V 真正复制三份。反向时共享 KV 的多个 Q head 对同一 KV 梯度的贡献也需要合并，不能照搬一对一的 head 假设。

## 8. KV cache 与分页是存储管理

生成第 t 个 token 时，旧 token 的 K/V 可以复用，只计算新 token 的 K/V 并追加到 cache。FA2 的 [`flash_attn_with_kvcache`](<../../../../../flash-attention/flash_attn/flash_attn_interface.py>) 可以就地追加 K/V，并对更新后的 cache 做 attention；它不支持 backward。

调用者提供容量足够的 cache 与已有长度 `cache_seqlens`，并在后续调用中维护正确的长度。`cache_seqlens` 不能简单理解为“函数会替调用者自动推进的计数器”。

连续 cache 的形状为 `(batch_cache, capacity, H_kv, d)`。分页 cache 改为 `(num_blocks, page_block_size, H_kv, d)`，另有 `block_table` 把每条序列的逻辑页映射到物理页。

取 FA2 接口允许的 `page_block_size=256`，某序列 `block_table=[7,3,...]`。它的逻辑 token 300 位于逻辑页 `300//256=1`，页内偏移 `300%256=44`，因此访问物理页 3 的第 44 行。页可以不连续，但数学上的 token 顺序没有改变。

## 9. split KV 的结果怎样合并

当 Q 很短时，按 Q 分块产生的 block 太少，GPU 可能缺少可并行工作。split KV 将 K 分给多个任务，分别得到局部归一化输出 $O_r$ 与 $L_r=\log Z_r$。最终为：

这里 r 是 split 编号，`Z_r=sum(exp(S_ij))` 是这一分区对当前 query 的**未减最大值时的分母**。实际计算仍使用稳定形式，只通过 `L_r` 表示它，并不直接求可能溢出的 `Z_r`。

$$L=\log\sum_r e^{L_r},\qquad O=\sum_r e^{L_r-L}O_r.$$

用[第 01 章](<01_algorithm_and_online_softmax.md>)四个 key 的手算例子：第一块输出为 `5/3`，分母 Z 为 3；第二块输出为 `11/3`，分母 Z 为 12。合并权重应为 `3/15` 和 `12/15`，最终仍得 `49/15`；直接平均会出错。

这不与第 01 章的 `l=1.5` 矛盾：l 使用减去局部最大值后的基准，`Z_r=exp(m_r)*l_r`。两个 split 各自的 l 都是 1.5，但 m 分别是 `ln(2)`、`ln(8)`，所以 Z 分别为 3、12，LSE 分别为 `ln(3)`、`ln(12)`。只比较 l 会丢掉基准信息；保存 LSE 才能正确合并。

源码在 `flash_api.cpp` 决定 split 数，在 `flash_fwd_launch_template.h` 启动局部计算和 combine，具体合并在 `flash_fwd_kernel.h` 的 `combine_attn_seqk_parallel`。split 带来临时结果和第二次合并的成本，因此不是分得越多越好。

## 10. 源码精读：一次反向传播的中间值如何变化

以下固定无 dropout、无 softcap、Q/K head 数相同，先把主干看清。反向使用自己的 `Kernel_traits`；不能把前向 `128×128` 的 tile 大小直接套给它。

### 10.1 预处理：从 O、dO 得到逐行 D

源码摘录：[flash_bwd_preprocess_kernel.h，35–50 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_bwd_preprocess_kernel.h#L35-L50)。

<!-- source: csrc/flash_attn/src/flash_bwd_preprocess_kernel.h:35-50 -->
```cpp
Tensor o_reshaped = make_tensor(o.data(), do_reshaped.layout());
Tensor do_fp32 = FLASH_NAMESPACE::convert_type<float>(do_reshaped);
Tensor o_fp32 = FLASH_NAMESPACE::convert_type<float>(o_reshaped);
#pragma unroll
for (int mi = 0; mi < size<0>(do_reshaped); ++mi) {
    float dP_sum_cur = do_fp32(mi, 0) * o_fp32(mi, 0);
    #pragma unroll
    for (int ni = 1; ni < size<1>(do_reshaped); ni++) {
        dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni);
    }
    FLASH_NAMESPACE::SumOp<float> sum_op;
    dP_sum_cur = FLASH_NAMESPACE::Allreduce<THREADS_PER_ROW>::run(dP_sum_cur, sum_op) * scale;
    if (threadIdx.x % THREADS_PER_ROW == 0) {
        dP_sum(mi * gdP_col_stride + threadIdx.x / THREADS_PER_ROW) = dP_sum_cur;
    }
}
```

这里首先把 O/dO 的片段转成 FP32，减少点积累加时的精度损失。每条线程先乘加自己持有的特征列，再用 `Allreduce<THREADS_PER_ROW>` 合并同一行的各份结果。最后只让 `threadIdx.x % THREADS_PER_ROW == 0` 的线程写入：归约后多条线程都有相同结果，写一次即可。

`gdP_col_stride` 描述结果片段写到全局缓冲区时的间隔；它不参与梯度数学。`scale` 来自 `params.p_dropout`，在无 dropout 时等于 1。有 dropout 时这个字段表示**保留概率**，对应源码把部分缩放延迟到最终梯度的约定。

用一行最小例子代入：`O=[2,3]、dO=[5,7]`，先得到乘积 `[10,21]`，合并为 `D=31`。`softmax_d` / `dP_sum` 保存的就是这样的逐行统计量。

### 10.2 主计算：重算概率，再把 dP 的寄存器原地变成 dS

首先重新做 QK、mask，并执行前面介绍的 `scale_apply_exp2<false>`。helper 中 `Scale_max=false` 的分支把 LSE 乘 `log2(e)`，因此实际计算是：

```text
scores = exp2(raw_QK * alpha * log2(e) - LSE * log2(e))
       = exp(alpha * raw_QK - LSE)
       = P
```

此时 `scores` 已经不再存 score，而是概率；同一变量名跨阶段承载不同意义，是阅读这段代码的重要难点。

随后 dO 和 V 相乘：

源码摘录：[flash_bwd_kernel.h，577–595 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_bwd_kernel.h#L577-L595)。

<!-- source: csrc/flash_attn/src/flash_bwd_kernel.h:577-595 -->
```cpp
FLASH_NAMESPACE::gemm</*A_in_regs=*/false, /*B_in_regs=*/Kernel_traits::Is_V_in_regs>(
    acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
    smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV
);

// Reshape acc_dp from (MMA=4, MMA_N, MMA_N) to (row=(2, MMA_N), col=(2, MMA_N))
Tensor dS = make_tensor(acc_dp.data(), scores.layout());
auto pointwise_mult = [](<../../../13_Source_Analysis/flash_attention/float p, float dp, float d>) {
    return p * (!Is_dropout || p >= 0 ? dp - d : d);
};
#pragma unroll
for (int mi = 0; mi < size<0>(dS); ++mi) {
    #pragma unroll
    for (int ni = 0; ni < size<1>(dS); ++ni) {
        float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
        if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
        dS(mi, ni) = scaled_ds;
    }
}
```

按执行顺序读：

1. `gemm` 令 `acc_dp` 累加 `dO @ V.T`，得到当前 tile 的 dP。
2. `make_tensor(acc_dp.data(), scores.layout())` 为**同一批寄存器**建立按行列索引的视图 `dS`，没有新分配一张完整梯度矩阵。
3. 无 dropout 时，`pointwise_mult(p,dp,d)` 化简为 `p*(dp-d)`。
4. 内层读取 `scores(mi,ni)=P_ij`、`dS(mi,ni)=dP_ij`、`dP_sum(mi)=D_i`。其中 D 只依赖行，所以同一行所有列复用一个 D。
5. `dS(mi,ni)=scaled_ds` 覆盖原 dP。执行后，`acc_dp` 的存储也已变成 dS。若启用 softcap，还要乘 `dtanh`，把梯度继续穿过前向的 tanh。

可以手算一行来检查减法究竟减的是什么：

```text
P  = [0.25, 0.75]
dP = [2, 6]
D  = 0.25*2 + 0.75*6 = 5
dS = [0.25*(2-5), 0.75*(6-5)] = [-0.75, 0.75]
```

梯度分量之和为 0：把一行所有 score 同时加同一个常数不会改变 softmax，所以这种共同平移的梯度为 0。这是比“找到了 pointwise_mult 函数”更有用的理解检查。

### 10.3 为什么 dQ 需要原子加，而一个 K 块可以累积自己的 dK/dV

对固定的 K/V tile，主 kernel 遍历相关 Q tile：

```text
该 K 块的 dV += P.T @ dO
该 K 块的 dK += dS.T @ Q       （alpha 在后续乘）
当前 Q 块的 dQ 贡献 = dS @ K   （alpha 在后续乘）
```

前两项一直累加到当前 K 块负责的结果。第三项还缺其他 K 块的贡献，所以要写到共享的全局 `dQaccum` 缓冲区。实际乘法与写入分别是：

源码摘录：[flash_bwd_kernel.h，655–656 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_bwd_kernel.h#L655-L656)。

<!-- source: csrc/flash_attn/src/flash_bwd_kernel.h:655-656 -->
```cpp
FLASH_NAMESPACE::gemm(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq,
            smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt);
```

源码摘录：[flash_bwd_kernel.h，672–679 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_bwd_kernel.h#L672-L679)。

<!-- source: csrc/flash_attn/src/flash_bwd_kernel.h:672-679 -->
```cpp
if (!Seq_parallel) {
    cute::copy(gmem_tiled_copy_dQaccum, acc_dq_reshaped, tdQgdQaccum);
} else {
    // if (cute::thread0()) { print(acc_dq.layout()); printf("\n"); print(acc_dq_reshaped.layout()); printf("\n"); print(tdQgdQaccum.layout()); printf("\n"); }
    CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }
}
```

本节的序列并行路径传入 `Seq_parallel=true、Is_last=false`，会执行这里的原子加。设两个 block 分别贡献 3 和 5，目标初值是 0：普通读改写可能都读到 0，最终只留下 3 或 5；`atomicAdd` 保证两次更新合起来得到 8。它解决并发累加问题，无法让有限精度加法变成严格结合运算。

### 10.4 deterministic 路径仍然能出现 atomicAdd

这段源码值得直接阅读，因为“确定性等于把所有 atomicAdd 删除”在这里并不成立：

源码摘录：[flash_bwd_kernel.h，834–837 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_bwd_kernel.h#L834-L837)。

<!-- source: csrc/flash_attn/src/flash_bwd_kernel.h:834-837 -->
```cpp
// If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
for (int n_block = blockIdx.x; n_block < (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN; n_block += gridDim.x) {
    compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, false, false, /*Seq_parallel=*/true>(params, bidb, bidh, n_block);
}
```

地址还加了以下偏移：

源码摘录：[flash_bwd_kernel.h，122–125 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_bwd_kernel.h#L122-L125)。

<!-- source: csrc/flash_attn/src/flash_bwd_kernel.h:122-125 -->
```cpp
const index_t row_offset_dq_accum = binfo.q_offset(dq_accum_batch_stride, dq_accum_row_stride, bidb)
    + ((m_block_max - 1) * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128ll * bidb)) * dq_accum_row_stride + bidh * params.d_rounded
    // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
    + (!params.deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);
```

普通路径的额外偏移是 0，各 K block 竞争相同的 dQ 缓冲区。确定性路径为不同 `blockIdx.x` 分配不同的累加区；每个 block 按 `n_block += gridDim.x` 的固定步长处理自己的 K 块序列。原子指令可以保留，但不同 block 不再争抢同一份 dQ 累加结果。

例如 6 个 K tile、`gridDim.x=2`：block 0 依次处理 0、2、4，block 1 处理 1、3、5，各自写到独立缓冲区；后处理按固定顺序合并这两份结果。这里是**解释调度公式的例子**，实际 `gridDim.x` 由 SM 数、batch 和 head 数决定。

固定长度接口的分配也与这个设计对应：普通路径使用一份 FP32 `dq_accum`，由预处理 kernel 清零所需区域；确定性路径分配带 `nsplits` 维度的 `torch::zeros`。不清零就直接 atomicAdd，会把未初始化值混入梯度。

### 10.5 最后才乘 alpha：追踪一个缩放因子的完整去向

源码摘录：[flash_bwd_preprocess_kernel.h，241–252 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_bwd_preprocess_kernel.h#L241-L252)。

<!-- source: csrc/flash_attn/src/flash_bwd_preprocess_kernel.h:241-252 -->
```cpp
Tensor tdQrdQaccum = make_fragment_like(tdQgdQaccum);
clear(acc_dq);
for (int s = 0; s < nsplits; ++s) {
    cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, tdQrdQaccum);
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) += tdQrdQaccum(i); }
    tdQgdQaccum.data() = tdQgdQaccum.data() + params.dq_accum_split_stride;
}
#pragma unroll
for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
// Convert acc_dq from fp32 to fp16
Tensor rdQ = FLASH_NAMESPACE::convert_type<Element>(acc_dq);
```

这里先把 `acc_dq` 清零，再依次加载每份部分结果。普通路径 `nsplits=1`；确定性路径可能多份，每轮通过 `dq_accum_split_stride` 移动到下一份缓冲区。

之后统一乘 `params.scale_softmax_rp_dropout`。由 C++ 参数设置可知它等于 `alpha / 保留概率`，无 dropout 时就是 `alpha`。因此前面 GEMM 算 `dS @ K` 而没有当场乘 alpha 是有意安排，并未漏掉链式法则。

最后 `convert_type<Element>` 把 FP32 累加结果转换成输出 dtype，再进行共享内存重排和显存写回。dK 同样在主 kernel 的尾声乘这个尺度；dV 不需要由 `S=alpha QK.T` 引入的 alpha。

把整个反向连接起来，便能解释为什么是多个 kernel：**先准备逐行 D 和累加区 → 各 K 块计算并累加 → 合并 dQ、补缩放、转类型**。这些 kernel 在同一 CUDA stream 中按提交顺序执行，所以后一阶段可以消费前一阶段写出的结果。

## 11. 源码精读：变长、分页与 split 如何改变地址和归约

### 11.1 `BlockInfo` 中的 -1 是哨兵，不是合法 token 起点

源码摘录：[block_info.h，15–24 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/block_info.h#L15-L24)。

<!-- source: csrc/flash_attn/src/block_info.h:15-24 -->
```cpp
template<typename Params>
__device__ BlockInfo(const Params &params, const int bidb)
    : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
    , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
    , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    , leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])
    , seqlen_k_cache((!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - leftpad_k)
    , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] - leftpad_k : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
```

C++ 构造函数冒号后的列表依次初始化字段：

- `sum_s_q` 在变长输入中读取前缀和；没有前缀和则写 -1，表示使用固定 batch stride。
- `actual_seqlen_q` 取相邻前缀和之差，或者固定长度。
- K 多了 `is_seqlens_k_cumulative`：同一个参数字段可能装累计长度，也可能装每条 cache 的现有长度，不能一律做相邻差分。
- `seqlen_k_cache` 扣除左侧 padding；`actual_seqlen_k` 还考虑本次追加的新 K，或由 `seqused_k` 指定的有效范围。

Q 的起始地址选择如下：

源码摘录：[block_info.h，28–31 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/block_info.h#L28-L31)。

<!-- source: csrc/flash_attn/src/block_info.h:28-31 -->
```cpp
template <typename index_t>
__forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
    return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
}
```

`cu_seqlens_q=[0,2,5]、row_stride=256、bidb=1` 时，返回元素偏移 `2*256=512`。如果是补齐到长度 3 的定长存储，`batch_stride=3*256=768`，相同 batch 编号应返回 768。两者不是不同 attention 公式，而是不同的存储排布。

### 11.2 分页地址：先查物理页，再加页内偏移

split/cache 前向中读取 K tile 的代码如下：

源码摘录：[flash_fwd_kernel.h，591–597 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L591-L597)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:591-597 -->
```cpp
const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
const int block_table_idx = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
const int block_table_offset = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
const index_t row_offset_k = block_table == nullptr
    ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
      + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
    : block_table[block_table_idx] * params.k_batch_stride + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
```

`block_table` 先跳到当前样本对应的页表。`(n_block_max-1)*kBlockN` 是当前首个处理 tile 的逻辑 token 起点；整数除法取逻辑页号，减法取页内偏移。然后：

```text
K 元素偏移 = 物理页号 * k_batch_stride
            + 页内 token * k_row_stride
            + KV head * k_head_stride
```

分页情形的 `k_batch_stride` 实际跨越一个物理页，不是跨越一整条逻辑序列。变量名沿用了参数结构的命名，理解时必须结合张量形状。

取 `H_kv=2、d=64、page_size=256`，连续页存储的 stride 为：页 `32768`、token `128`、head `64`。页表 `[7,3]` 下的逻辑 token 300、KV head 1，对应偏移：

```text
page = 300//256 = 1，inside = 44，physical_page = 3
offset = 3*32768 + 44*128 + 1*64 = 104000
```

kernel 先定位整个 tile 的起点，再由线程布局加上 tile 内坐标；上例是把同一地址规则应用到单个 token，便于核算。

循环转到前一个 K tile 时也不能直接减去固定字节数：

源码摘录：[flash_fwd_kernel.h，907–914 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L907-L914)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:907-914 -->
```cpp
if (block_table == nullptr) {
    tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
} else {
    const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
    const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
    const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
    const int block_table_offset_next =(n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
    tKgK.data() = tKgK.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
```

跨逻辑页时，物理页号可能增大，也可能减小。因此分页分支计算**页号差 × 页 stride + 页内偏移差 × 行 stride**；连续分支才直接减一个 tile 的行数。漏掉这一分支，页表碰巧连续时可能正常，一旦物理页打散就会读取错误 token。

### 11.3 combine kernel 对“每个 split 的 LSE”再做一次稳定归约

源码摘录：[flash_fwd_kernel.h，1190–1204 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L1190-L1204)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:1190-1204 -->
```cpp
// Compute the logsumexp of the LSE along the split dimension.
ElementAccum lse_max = lse_accum(0);
#pragma unroll
for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }
MaxOp<float> max_op;
lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
float lse_sum = expf(lse_accum(0) - lse_max);
#pragma unroll
for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += expf(lse_accum(l) - lse_max); }
SumOp<float> sum_op;
lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
// For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
// lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
```

按一个 query 行来理解，输入是 `[L_0,L_1,...]`，每个值来自一个 split。它先找最大 L，计算 `sum(exp(L_r-max))`，再加回 max 得到全局 LSE。这与第 01 节稳定 softmax 的步骤相同，只是被归约的维度从 key 变成 split。

`lse_max=-inf` 表示所有 split 都没有有效 key。代码把指数基准临时改为 0，避免 `-inf-(-inf)`；总和为 0 时，把最终 LSE 设为 `+inf` 哨兵。后面局部 `-inf` 减全局 `+inf` 的指数是 0，因此空 split 不贡献输出。

接下来把局部 LSE 改成合并权重，并暂存 shared：

源码摘录：[flash_fwd_kernel.h，1216–1223 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L1216-L1223)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:1216-1223 -->
```cpp
// Store the scales exp(lse - lse_logsum) in shared memory.
#pragma unroll
for (int l = 0; l < kNLsePerThread; ++l) {
    const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
    const int col = tidx / kRowsPerLoadTranspose;
    if (row < params.num_splits && col < kBlockM) { sLSE[row][col] = expf(lse_accum(l) - lse_logsum); }
}
__syncthreads();
```

`sLSE` 此时已经存 `exp(L_r-L)`，不再是 LSE。同步后，不同线程才能安全地读取其他线程写出的权重。真正合并 O 的内层是：

源码摘录：[flash_fwd_kernel.h，1257–1265 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L1257-L1265)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:1257-1265 -->
```cpp
for (int m = 0; m < size<1>(tOrOaccum); ++m) {
    int row = get<0>(tOcOaccum(0, m, 0));
    ElementAccum lse_scale = sLSE[split][row];
    #pragma unroll
    for (int k = 0; k < size<2>(tOrOaccum); ++k) {
        #pragma unroll
        for (int i = 0; i < size<0>(tOrOaccum); ++i) {
            tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
        }
```

`row` 对应 query，`split` 对应 K 的分区。每个输出特征列使用相同的逐行权重，执行 `全局输出 += 权重 * 局部归一化输出`。这解释了三个看似额外的步骤为何必要：先归约 LSE 得到正确分母，再共享逐行权重，最后逐列合并 O。


## 12. 常见误解与边界

- O/LSE 为什么能帮助反向节省显存？**它们允许逐块重算 P，并用 O、dO 求逐行 D。**
- `cu_seqlens=[0,2,5]` 的第二条序列长度是多少？**3，起点是扁平行 2。**
- 分页 KV 是否改变 softmax 公式？**不改变，只改变逻辑 token 到物理地址的映射。**
- 两个 split 的输出为什么不能平均？**其原始归一化分母可能不同。**

下一节把基本算法保持不变，观察 Hopper 和 Blackwell 如何重新安排搬运、计算与同步。
