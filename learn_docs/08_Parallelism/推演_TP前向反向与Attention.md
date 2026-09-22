# 手工推演 TP：两层 MLP 的前向、反向与 Attention 布局

> 两张卡，数学权重约定为 [in,out]，省略 bias。PyTorch Linear 的存储方向通常相反，读代码时要转换。概念见[Tensor Parallelism](TensorParallelism.md)。

## 单卡参考模型

$$X=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
W_1=\begin{bmatrix}1&0&1&2\\0&1&1&0\end{bmatrix},\quad
W_2=\begin{bmatrix}1&0\\0&1\\1&1\\2&-1\end{bmatrix}.$$

计算 $A=\operatorname{ReLU}(XW_1)$、$Y=AW_2$，loss 为 Y 的全部元素之和。这里所有激活都大于0，所以 ReLU 的局部导数为1，避免零点约定干扰推演。

## 第一层沿输出列切分

rank 0 保存 W1 的前两列，rank 1 保存后两列。X 在两卡逻辑复制：

| rank | W1 shard [2,2] | A shard [2,2] |
|---|---|---|
| 0 | [[1,0],[0,1]] | [[1,2],[3,4]] |
| 1 | [[1,2],[1,0]] | [[3,2],[7,6]] |

完整 A 为 [[1,2,3,2],[3,4,7,6]]，但不必实际 AllGather。因为第二层正好可以消费同样的中间维切片。

## 第二层沿输入行切分

rank 0 保存 W2 的前两行，rank 1 保存后两行，各计算完整输出形状的部分和：

| rank | 局部乘积 [2,2] | 语义 |
|---|---|---|
| 0 | [[1,2],[3,4]] | 部分和，不能当完整 Y |
| 1 | [[7,1],[19,1]] | 部分和，不能当完整 Y |
| SUM AllReduce | [[8,3],[22,5]] | 两卡均获得完整 Y |

loss=38。这里是 SUM，不是数据并行的平均；除以2会把模型输出改小一半。

## 反向逐步算

上游 G=dL/dY 为全1矩阵。第二层每卡计算 $dW_{2,r}=A_r^TG$、$dA_r=GW_{2,r}^T$：

| 对象 | rank 0 | rank 1 |
|---|---|---|
| dW2 shard | [[4,4],[6,6]] | [[10,10],[8,8]] |
| dA shard | 每行[1,1] | 每行[2,1] |
| dW1 shard=$X^TdA_r$ | [[4,4],[6,6]] | [[8,4],[12,6]] |
| dX 局部贡献 | 每行[1,1] | 每行[4,2] |
| SUM 后 dX | 每行[5,3] | 每行[5,3] |

因此前向只需在第二层后合并部分输出，反向只需在第一层输入梯度处合并部分贡献。权重梯度保持对应 TP shard；若还有 DP，则在相同 TP 坐标的 DP group 同步。

这里的梯度语义按逻辑模型推导。实现会用特定 autograd 通信算子表达“前向求和、反向把同一上游梯度提供给分支”等关系，不能把每个物理副本都当作独立 loss 再无条件规约，否则可能重复计数。

## 通信负载账本

本例每个合并位置有4个元素，假设FP32，逻辑 payload=16 B。它不是所有链路的累计字节。两rank ring AllReduce 模型中每rank发送总量为 $2(D-1)M/D=M=16$ B，接收同量；真实小消息主要受固定开销影响。

## Attention 如何对应同一种布局

设 B=1、S=2、H=4、head数2、head_dim=2、TP=2。各卡拥有一个完整head的Q/K/V，对两个token做该head的Attention；head内部仍需要正确的causal mask。

| 位置 | 每卡形状 | 语义 |
|---|---|---|
| 输入 X | [1,2,4] | replicated |
| Q/K/V | 各[1,2,1,2] | head shard |
| Attention输出 | [1,2,2] | head shard |
| 输出投影权重 | [2,4] | 输入行 shard |
| 输出投影结果 | [1,2,4] | partial，需要SUM |

反向中，输出投影把同一逻辑 dY 传给各 head 分支；每个分支分别经过 attention backward 和 QKV 投影 backward，产生对 X 的局部贡献，最后求和。GQA 的KV head少、MLA缓存表示特殊时，不能机械套用此head整除分法。

验证：[纯 Python 单卡/分片前反向对照](../labs/parallel_runtime_walkthroughs.py)。依据：[Megatron-LM](https://arxiv.org/abs/1909.08053)。

---

[所属领域](README.md) · [手工推演总入口](../手工推演索引.md)
