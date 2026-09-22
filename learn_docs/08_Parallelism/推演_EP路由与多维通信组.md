# 手工推演 EP 与多维通信组：token 去哪里，梯度由谁更新

> 分为独立的EP数据流与DP×TP网格两个例子。二者不暗示所有框架都能把EP作为额外维度直接相乘。概念：[EP](ExpertParallelism.md)、[DeviceMesh](多维并行与DeviceMesh.md)。

## 四个token、两个专家

token输入为标量[1,2,3,4]，rank 0拥有t0/t1和专家E0，rank 1拥有t2/t3和专家E1。专家函数E0(x)=2x，E1(x)=3x。每个token选择两个专家，权重如下；路由权重固定，不计算router梯度。

| token | 来源rank | E0权重 | E1权重 | 完整参考输出 |
|---|---|---:|---:|---:|
| t0=1 | 0 | 0.75 | 0.25 | 2.25 |
| t1=2 | 0 | 0.25 | 0.75 | 5.5 |
| t2=3 | 1 | 0.5 | 0.5 | 7.5 |
| t3=4 | 1 | 0.25 | 0.75 | 11 |

## Dispatch、计算与Combine

每个来源rank按目标专家把token打包，保留(token_id,source_rank,expert_id,weight)映射。rank 0向rank 1发送t0/t1供E1计算；rank 1向rank 0发送t2/t3供E0计算。发往本地专家的记录也参与排列，但不必经过网络。

E0在rank 0处理全部四个token，产生[2,4,6,8]；E1在rank 1产生[3,6,9,12]。返回输出到token来源，按原token位置加权求和，得到表中结果。不能按专家输出的当前数组顺序直接相加，必须恢复映射。

本例top-2使四个token产生八份expert assignment。若真实hidden=H、元素b字节，跨rank每方向每次发送2Hb字节的激活；combine若形状精度相同，再发送同量。不含索引、权重、对齐与协议，不能把八份assignment全算为远端流量。

## 反向与专家参数梯度

取loss为四个输出之和，则每token上游梯度为1。传给每个专家输出的梯度就是其路由权重；专家反向再乘专家系数，来源rank累加两份dX，得到[2.25,2.75,2.5,2.75]。

设专家系数是可训练标量a0=2、a1=3，则dA0=0.75×1+0.25×2+0.5×3+0.25×4=3.75；dA1=6.25。SGD lr=0.1后a0=1.625、a1=2.375。若专家还跨数据副本复制，需在相同专家身份的复制组同步梯度，不能把不同专家梯度当成同一参数规约。

真实router训练还需处理权重的梯度、选择机制及辅助目标；固定路由算例只验证dispatch/combine的线性关系。

## 另一个例子：八卡 DP×TP×PP 网格

令DP=2、TP=2、PP=2，rank编号为 $r=4p+2d+t$，三个坐标均为0或1：

| group类型 | 成员 | 共享什么 |
|---|---|---|
| TP | [0,1]、[2,3]、[4,5]、[6,7] | 同一stage、同一数据分片的算子计算 |
| DP | [0,2]、[1,3]、[4,6]、[5,7] | 同一stage、同一TP参数切片 |
| PP链 | [0,4]、[1,5]、[2,6]、[3,7] | 同一数据/TP坐标的前后stage |

rank1=(p0,d0,t1)：与rank0做TP，与rank3做DP，与rank5交换流水边界数据。不同组可能重叠成员，但它们的参数身份与操作语义不同。

例如逻辑权重的t1切片在rank1得到梯度[1,3]、在rank3得到[5,7]，DP平均为[3,5]；和rank0的t0切片求平均没有这个意义。应先确认参数身份，再构造通信组。

## 从表到真实配置

依次检查每个参数的TP/EP布局、复制组、PP所属stage、batch归属和物理位置。专家可能有自己的rank生成和并行映射，不能由上述dense网格自动推出。接口依据：[Megatron并行策略](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)。验证：[EP与网格脚本](../labs/parallel_runtime_walkthroughs.py)。

---

[所属领域](README.md) · [手工推演总入口](../手工推演索引.md)
