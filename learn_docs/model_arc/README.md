# 模型结构与 Infra 专项

长文保留完整推导、算例与资料。首次先理解改变了哪些 tensor、状态和通信，再深入数学与实现，不要求读完整目录才开始 RL 实验。

| 主题 | 首读 | 二读 |
|---|---|---|
| [MLA](MLA.md) | MHA/GQA 到 KV 表达、容量与 shape | 投影吸收、backend 与训练/推理 |
| [MoE](MoE-Infra.md) | router→dispatch→GEMM→combine | EP、倾斜、Grouped GEMM、通信 |
| [Sparse Attention](Sparse-Attention.md) | 保留哪些 query/key pair | 稀疏访问、索引与 kernel |
| [Linear Attention](Linear-Attention.md) | 递归状态和 KV 的差别 | 扫描、混合结构与精度 |
| [Residual Evolution](Residual-Evolution.md) | 残差、执行依赖与稳定性 | 推导与具体模型 |
| [架构演进资料](AI_Infra_Model_Architecture_Evolution_DeepSeek_Kimi_Qwen_GLM_2024_2026.md) | 按模型查差异 | 型号/版本/性能条件另行核验 |

前置：[Transformer/KV](../00_Foundations/05_Transformer执行与KV基础.md)、[通信](../02_Distributed_Communication_Memory/00_专题总览.md)、[并行](../03_Parallelism/00_专题总览.md)。用[动态负载分析](../../docs/performance_analysis_guide/09_dynamic_workloads.md)连接实验。
