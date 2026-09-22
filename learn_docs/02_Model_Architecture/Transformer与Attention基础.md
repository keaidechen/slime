# Transformer 与 Attention：从 token 到一层完整计算

> 类型：基础概念。前置：[张量形状](../01_Math_Numerics/张量形状与复杂度.md)、[数值精度](../01_Math_Numerics/浮点数与低精度.md)。以下以 decoder-only、Pre-Norm 的示意结构说明，其他模型可能改变顺序和算子。

## 输入与参数分别是什么

Tokenizer 把输入映射为 token ID；Embedding 用 ID 查找向量。对 batch B、长度 S、hidden size H，hidden state 可表示为 $X\in\mathbb R^{B\times S\times H}$。参数在训练时更新；hidden state 随本次输入变化。

Attention 混合序列位置的信息，FFN 对每个位置做通道变换。残差保留并更新 hidden state，归一化控制数值尺度。这些组成部分共同决定模型功能和执行成本。

## Q、K、V 与可见性

投影得到 Q、K、V。对单个 head，$Q\in\mathbb R^{S_q\times d}$，$K\in\mathbb R^{S_k\times d}$，$V\in\mathbb R^{S_k\times d_v}$：

$$P=\operatorname{softmax}(QK^T/\sqrt d+M),\qquad O=PV.$$

M 表示 mask。Causal mask 禁止访问未来位置；padding mask 排除填充；packed 样本需要隔离边界。mask 改变允许使用的信息，属于模型计算语义。

例如 Q=[1,0]，两个 Key 分别 [1,0] 与 [0,1]，Value 为 [2,0] 与 [0,4]。缩放后 logits 约 [0.707,0]，概率约 [0.670,0.330]，输出约 [1.340,1.321]。若第二个位置被 mask，则概率为 [1,0]，输出变成 [2,0]。

## 多头与 KV 表达

多头让不同投影子空间同时参与信息混合。MHA 为各 query head 配备相应 K/V；MQA 共享一组 K/V；GQA 在若干 query heads 间共享 K/V。它们改变参数化和缓存大小，不能在任意已训练模型上随意切换而假设质量不变。

标准 K/V 缓存字节数可按 $2LTS_{kv}d b$ 核算，其中 L 是层数、T 是驻留 token 总数、$S_{kv}$ 是 KV head 数、d 是每头维度、b 是每元素字节。这里的 T 已汇总 batch，不再额外乘 batch；特殊压缩结构另算。

MLA 改变缓存表示，稀疏 Attention 改变访问集合，线性/递推结构改变历史状态组织。它们应在这些设计轴上比较，见[Attention 设计空间](Attention设计空间与演进.md)。

## FFN、归一化与残差

普通 FFN 可写为 $\phi(XW_1)W_2$；门控 FFN 使用一条变换调制另一条，再投影回 H。中间宽度决定参数、激活和 GEMM 形状。MoE 将一部分 FFN 计算替换为条件选择的专家计算，并增加路由与数据交换。

一个示意 Pre-Norm block 为：

$$U=X+\operatorname{Attention}(\operatorname{Norm}(X)),$$
$$Y=U+\operatorname{FFN}(\operatorname{Norm}(U)).$$

Norm 的归约维和精度必须说明。位置编码使模型获得顺序信息；RoPE 作用于特定 Q/K 分量，和简单把位置向量加到输入不是同一计算。

## 训练、Prefill、Decode

训练通常并行计算多个位置并使用 causal mask，通过后续 token 的预测目标构造 loss。自回归生成需要前一步生成的 token，因此输出步骤之间有依赖。

Prefill 处理已知前缀；decode 对新 token 计算查询与新 K/V，并复用可保留的历史状态。Prefill 常具有较多矩阵并行工作，decode 的形状、批量和历史读取不同；不能无条件断言所有 prefill 都计算受限、所有 decode 都带宽受限。

## 演进的核心问题

Transformer 提供可并行的序列表示计算，但长序列 Attention、参数规模和自回归依赖带来不同瓶颈。IO 优化、KV 压缩、稀疏访问、条件计算和递推状态分别处理其中一部分，并引入新的质量或系统约束。

相关：[KV 与执行阶段](../10_Inference_Serving/Transformer执行与KV基础.md)、[残差专题](residual/README.md)、[MoE 专题](moe/README.md)。

参考：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[GQA](https://arxiv.org/abs/2305.13245)。

---

[所属专题](README.md) · [百科总览](../README.md)
