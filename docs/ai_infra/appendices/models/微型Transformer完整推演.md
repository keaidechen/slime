# 微型 Transformer：从两个 token 到全部参数梯度


## 固定模型，避免隐含分支

词表大小 3，hidden size 2，一个 head，一个 Pre-Norm block，FFN 宽度 2，batch 为 1。输入 token 为 [0,1]，next-token labels 为 [1,2]，两个位置都计入 loss。没有 dropout、bias、可学习 Norm gain、门控 FFN 或 tied embedding。位置 p 加固定向量 [0.1p,−0.1p]，不用 RoPE；causal mask 只允许位置 i 访问 j≤i。

本例使用 RMSNorm(x)=x/√(mean(x²)+10⁻⁵)，FFN 激活使用 SiLU(z)=z/(1+exp(−z))，block 末尾再做一次 RMSNorm 后接词表 head。参数共 36 个：embedding 6、Q/K/V/O 共 16、FFN 两个矩阵共 8、head 6。

| 参数 | 数值，按输入维×输出维排列 |
|---|---|
| Embedding | [[0.5,−0.25],[0.25,0.75],[−0.5,0.5]] |
| Wq | [[0.5,0.25],[−0.25,0.5]] |
| Wk | [[0.25,−0.5],[0.5,0.25]] |
| Wv | [[0.5,0],[0.25,0.5]] |
| Wo | [[0.5,−0.25],[0.25,0.5]] |
| Wup | [[0.25,0.5],[−0.5,0.25]] |
| Wdown | [[0.5,0.25],[−0.25,0.5]] |
| Whead | [[0.5,−0.25,0.25],[−0.25,0.5,0]] |

## 前向时每个张量去了哪里

查表并加位置后，X=[[0.5,−0.25],[0.35,0.65]]。记 N=RMSNorm(X)，则 Q=NWq、K=NWk、V=NWv，形状均为 [2,2]。每个可见位置的分数为 qᵢ·kⱼ/√2，按行归一化得到 P，再计算 A=PV。

$$U=X+AW_o,\quad Z=\operatorname{RMSNorm}(U)W_{up},$$
$$H=\operatorname{SiLU}(Z),\quad Y=U+HW_{down},\quad C=\operatorname{RMSNorm}(Y)W_{head}.$$

| 状态 | 形状／本次结果 |
|---|---|
| 第一个位置的 P | [1]，未来位置没有进入 softmax |
| 第二个位置的 P | [0.391331984,0.608668016] |
| 第一个位置 logits | [0.785365759,−0.574008727,0.332240931] |
| 第二个位置 logits | [0.063932597,0.421637368,0.183167520] |
| 每位置交叉熵 | [1.997252138,1.149592019] |
| 两位置 mean loss | 1.573422079 |

交叉熵为 Lᵢ=log∑ⱼexp(Cᵢⱼ)−Cᵢ,label。实现先减去该行最大值，再在 log-sum-exp 后加回，避免直接指数化大值。loss 分母为 2；没有额外除以词表大小。

## 反向怎样沿依赖返回

从 head 开始，设 softmax 概率为 R，则：

$$\frac{\partial L}{\partial C_{ij}}=\frac{R_{ij}-\mathbf{1}[j=label_i]}{2}.$$

若 C=BWhead，则 dWhead=BᵀdC、dB=dC·Wheadᵀ。本次 head 梯度为：

$$\begin{bmatrix}
0.454119878&-0.427013224&-0.027106654\\
0.042294520&0.452133630&-0.494428150
\end{bmatrix}.$$

对每行 Norm，设 r=√(∑xⱼ²/D+ε)，上游梯度为 g，输入梯度为：

$$dx_j=g_j/r-x_j\frac{\sum_k g_kx_k}{D r^3}.$$

残差 Y=U+F(U) 的梯度有直通和 FFN 两路，二者相加。SiLU 的导数为 σ(z)+zσ(z)(1−σ(z))。矩阵乘反向继续生成 dWdown、dWup，并把 FFN 输入梯度加回 U。不能只保留其中一条 U 的使用路径。

Attention 中，先由 A=PV 得到 dP=dA·Vᵀ、dV=PᵀdA。每个可见位置的 softmax 反向是：

$$dS_{ij}=P_{ij}\left(dP_{ij}-\sum_k P_{ik}dP_{ik}\right).$$

再经过除以 √2 和 QKᵀ，得到 dQ、dK；被 mask 的分数不参与，梯度为零。Q/K/V 三条投影都消费同一个 N，所以三路 dN 需要相加。最终 embedding 按 token ID 累加贡献；若同一 token 在多个位置出现，梯度也会相加。固定位置向量不作为参数更新。

## 保存、释放与重算

| 反向消费者 | 需要的信息 | 可以采用的替代方案 |
|---|---|---|
| Linear | 输入、权重及上游梯度 | 按层重算输入；权重须与前向版本相容 |
| RMSNorm | 输入、归一化统计 | 保存统计或重算，取决于实现 |
| SiLU | 输入或足以恢复导数的状态 | 重算 FFN 中间量 |
| Attention | Q/K/V 与归一化信息 | 分块在线算法可避免保存完整 P |
| Embedding | token ID 和输入梯度 | 按索引归约，不必保存 one-hot 矩阵 |

数学上有 [2,2] 的 score/P，并不意味着所有高效实现都把它们物化到 HBM。教学脚本保存很多标量节点和局部导数，因此 Python 对象内存远大于理想张量字节数。不能拿该程序的内存来估计生产 Transformer 激活开销。

训练会保存上述反向依赖；prefill／decode 没有同样的反向图，但需要缓存适用的 K/V。把“训练激活”和“推理 KV”统一称为缓存会掩盖生命周期差异。

## 怎样验证自己没有漏掉一条梯度

[完整程序](<../../examples/mini_transformer.py>)实现标量反向模式，并对全部 36 个参数做中心有限差分：

$$g_{num}=\frac{L(\theta+h)-L(\theta-h)}{2h},\qquad h=10^{-5}.$$

本次 CPU 运行最大绝对差约 2.03×10⁻¹¹，检查阈值为 2×10⁻⁶。阈值是小模型的实验选择，不能直接当作 BF16 backend 的容差。还检查改变未来 token 不影响首位置 logits，以及学习率 0.01 的一次 SGD 将 loss 降到约 1.561201606。

有限差分验证脚本内的梯度与脚本内的目标一致，不证明所有模型语义正确；因此还要独立核对 causal 可见性、标签偏移和 loss 分母。浮点误差与低精度需要另做对照，见[数值基础](<../../02_Model_Computation/03_张量形状与数值表示.md#s2>)。

[模型目录](<README.md>) · [训练效率](<../../04_Training/17_训练效率收敛与预算.md#s1>) · [可复现实验](<../../07_Practice/30_训练综合案例.md#s3>)
