# 模型结构与计算过程

从 Transformer 的完整计算进入 Attention、MoE 与残差专题，以状态表示、访问方式和执行成本比较结构。

## 主题地图

| 条目 | 定位 |
|---|---|
| [Transformer与Attention基础](Transformer与Attention基础.md) | Transformer 与 Attention：从 token 到一层完整计算 |
| [Attention设计空间与演进](Attention设计空间与演进.md) | Attention 的设计空间与问题演进 |

## 结构专题与模型案例

- [MLA：压缩 KV 表示与执行变换](mla/README.md)
- [MoE：条件计算、路由与专家系统](moe/README.md)
- [线性注意力：递推状态与混合记忆](linear_attention/README.md)
- [稀疏注意力：访问集合与选择成本](sparse_attention/README.md)
- [残差、归一化与深度方向的状态](residual/README.md)
- [模型架构案例与历史资料](architecture_cases/README.md)

## 读完本专题之后

压缩状态、减少访问、条件计算和递推记忆是可组合的设计轴。模型质量、计算与通信需要共同评估。



## 查阅与关联

[系列总览](../README.md) · [术语索引](../术语索引.md) · [问题索引](../问题索引.md)


## 内容深化与验证

[微型 Transformer 完整推演](微型Transformer完整推演.md)：两个 token 的前向、全部参数梯度和状态生命周期。


## 多模态与生成计算

[多模态与扩散计算](多模态与扩散计算.md)从视觉 patch、LLM KV 和去噪目标推导工作量，并连接现有 SGLang 源码专题。

## Omni：理解与媒体生成

沿 [能力与架构边界](Omni能力与架构边界.md) → [音视频表示与时间对齐](音视频表示与时间对齐.md) → [理解与生成计算图](Omni理解与生成计算图.md) 阅读，再进入 [八篇完整路线](../guides/Omni扩展阅读路线.md)。这里补齐语音生成、时间依赖和连续 latent 状态；原有多模态与扩散篇保留视觉/KV 和噪声计算基础。
