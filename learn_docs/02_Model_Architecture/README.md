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

## 问题与演进

压缩状态、减少访问、条件计算和递推记忆是可组合的设计轴。模型质量、计算与通信需要共同评估。

历史与方法的跨领域关系见[技术演进索引](../技术演进索引.md)。

## 查阅与关联

[百科总览](../README.md) · [概念关系](../概念关系与正文归属.md) · [术语索引](../术语索引.md) · [问题索引](../问题索引.md)

<a id="模型结构与-infra-专项"></a>
