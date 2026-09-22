# 手工推演 SP 与 CP：布局转换、在线 Softmax 与远端梯度

> 两rank。先演示Megatron式SP，再演示切分上下文的Attention；两者切分的算子范围不同。概念：[SP](SequenceParallelism.md)、[CP](ContextParallelism.md)。

## SP：每个token的hidden仍然完整

X形状[S=4,H=2]，四行分别[1,3]、[2,4]、[3,5]、[4,6]。rank 0持前两行，rank 1持后两行。沿hidden做LayerNorm，教学设epsilon=0、gamma=1、beta=0，各行均值为2/3/4/5，方差均为1，输出均为[-1,1]。

每rank可以局部归一化，因为每个token的两个hidden分量都在本地。若切hidden而不是切sequence，这个结论不成立。

## 穿过TP子图

设中间宽度4，TP=2：

| 边界 | rank 0 / rank 1 的局部shape | 数据语义 |
|---|---|---|
| LN输入和输出 | [2,2] | sequence shard |
| AllGather后 | [4,2] | 完整token集合 |
| ColumnLinear输出 | [4,2] | 中间hidden shard |
| RowLinear局部输出 | [4,2] | 全序列上的partial |
| SUM ReduceScatter后 | [2,2] | token分片的完整结果 |

最后一步同时做两件事：把两rank对同一输出的贡献相加，再把结果按token分发。只scatter而不reduce会丢失模型贡献。

反向沿这些逻辑变换的伴随传播：输出的sequence shard梯度先被聚合到TP子图需要的布局，最终对LN输出的各TP局部贡献做求和并回到原token owner。此处的求和与DP平均不同。

LN的gamma/beta在这两个TP rank上复制，但各rank只处理部分token，因此参数梯度还需合并。假设LN输出上游梯度全1，则每rank的dGamma=[-2,2]、dBeta=[2,2]，跨sequence分片求和后为[-4,4]和[4,4]。只说“LN可局部做”而漏掉参数梯度同步，会得到错误训练。

## CP：查询留在本地，历史跨rank流动

另设单head、d=1、四个token，Q均为1，K=[0,ln2,ln3,ln4]，V=[1,2,4,8]，使用causal mask。rank 0持token0/1，rank 1持token2/3。

因为exp(K)=[1,2,3,4]，单卡参考输出为：

| query | 可见key | 权重分母 | 加权分子 | 输出 |
|---|---|---:|---:|---:|
| 0 | 0 | 1 | 1 | 1 |
| 1 | 0,1 | 3 | 5 | 5/3 |
| 2 | 0,1,2 | 6 | 17 | 17/6 |
| 3 | 0,1,2,3 | 10 | 49 | 4.9 |

rank 1若只处理本地KV，则query3输出=(3×4+4×8)/7=44/7，错误。它需要远端token0/1的贡献。

## 怎样合并块而不平均两个Softmax

对一个查询，维护最大值m、归一化和l、未归一化输出向量n。合并块a/b时：

$$m=\max(m_a,m_b),\quad l=e^{m_a-m}l_a+e^{m_b-m}l_b,$$
$$n=e^{m_a-m}n_a+e^{m_b-m}n_b,\quad O=n/l.$$

对query3，本地块2/3有 $m_b=\ln4,l_b=7/4,n_b=11$；远端块0/1有 $m_a=\ln2,l_a=3/2,n_a=5/2$。换到公共最大值ln4后，l=0.5×1.5+1.75=2.5，n=0.5×2.5+11=12.25，O=4.9。

两个块的各自输出均值不能直接平均。完全被mask的块应作为零贡献跳过；初始空状态也需避免对负无穷作未定义减法。

## 反向：远端K/V也有梯度

取loss=O3，其余查询上游梯度为0。p=[0.1,0.2,0.3,0.4]，因此dV=p，$dscore_j=p_j(V_j-O_3)$，得到[-0.39,-0.58,-0.27,1.24]。因Q3=1、d=1，dK相同；$dQ_3=\sum_j dscore_jK_j\approx1.020354$，其他query的dQ为0。

这些贡献由query3所在的rank 1产生，但K0/V0与K1/V1属于rank 0。必须将相应梯度贡献送回owner，并累加来自其他query分片的贡献。只把forward的KV传来而不处理backward归属，训练就不完整。

## 长度相同不等于工作相同

S=8时，连续切分[0..3]与[4..7]的causal pair数分别为10和26；配成[0,1,6,7]与[2,3,4,5]则各18。改变的是分配与索引，不能改变原始位置和mask。对于packed样本，样本边界也必须随重排保留。

验证：[SP归一化、CP输出与有限差分梯度](../labs/parallel_runtime_walkthroughs.py)。依据：[Ring Attention](https://arxiv.org/abs/2310.01889)、[Ulysses](https://arxiv.org/abs/2309.14509)。

---

[所属领域](README.md) · [手工推演总入口](../手工推演索引.md)
