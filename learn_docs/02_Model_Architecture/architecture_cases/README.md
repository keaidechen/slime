# 模型架构案例与历史资料

保留原长文的跨模型比较与推导，按问题域拆分。这里的型号、发布日期、实现与性能陈述属于待逐项核验的历史资料，不作为百科基础定义或当前版本支持矩阵。

> 前置：[模型基础](../Attention设计空间与演进.md)。事实边界：[版本与验证范围](../../版本与验证范围.md)。

## 专题正文

- [模型路线与条件计算](模型路线与条件计算.md)
- [KV压缩与稀疏访问](KV压缩与稀疏访问.md)
- [线性状态与混合架构](线性状态与混合架构.md)
- [残差与深度方向](残差与深度方向.md)
- [数值优化器与预测](数值优化器与预测.md)
- [系统综合与资料](系统综合与资料.md)

## 与相邻领域的关系

- [模型设计空间](../Attention设计空间与演进.md)：区分表示、访问集合与物理管理。
- [相邻专题](../mla/README.md)：比较可以组合或竞争的方法。
- [并行布局](../../08_Parallelism/统一切分语言.md)：解释状态归属与通信。
- [成本与测量](../../12_Performance_Reliability/成本模型与测量方法.md)：约束性能结论。

<details>
<summary>本专题章节索引</summary>

- [DeepSeek / Kimi / Qwen / GLM：模型结构为什么这样变，它们具体如何实现，以及瓶颈如何迁移](#read-01)
- [0. 核心结论：大模型结构演化，本质上是瓶颈不断迁移](模型路线与条件计算.md#read-02)
- [1. 版本路线：先把四家的时间线放在同一张图里](模型路线与条件计算.md#read-03)
- [1.1 DeepSeek](模型路线与条件计算.md#read-04)
- [1.2 Kimi](模型路线与条件计算.md#read-05)
- [1.3 Qwen](模型路线与条件计算.md#read-06)
- [1.4 GLM](模型路线与条件计算.md#read-07)
- [2. 第一场革命：从 Dense FFN 到 Sparse MoE](模型路线与条件计算.md#read-08)
- [2.1 Dense Transformer 真正的问题是什么？](模型路线与条件计算.md#read-09)
- [3. MoE 到底怎么工作？](模型路线与条件计算.md#read-10)
- [4. DeepSeekMoE：为什么不是简单 Top-K MoE？](模型路线与条件计算.md#read-11)
- [4.1 Fine-grained Expert Segmentation](模型路线与条件计算.md#read-12)
- [4.2 Shared Expert 为什么存在？](模型路线与条件计算.md#read-13)
- [5. MoE 为什么把模型问题变成 Infra 问题？](模型路线与条件计算.md#read-14)
- [6. 为什么 Ultra-Sparse MoE 会让 Grouped GEMM 越来越重要？](模型路线与条件计算.md#read-15)
- [7. DeepSeek-V3：为什么会出现 Auxiliary-Loss-Free Load Balancing？](模型路线与条件计算.md#read-16)
- [8. 为什么 DeepSeek-V3 的 DualPipe 与 MoE 是一套设计？](模型路线与条件计算.md#read-17)
- [9. 第二场革命：为什么 MLA 会出现？](KV压缩与稀疏访问.md#read-18)
- [KV Cache](KV压缩与稀疏访问.md#read-19)
- [9.1 标准 MHA 的 KV Cache](KV压缩与稀疏访问.md#read-20)
- [10. GQA/MQA 先解决了一部分 KV 问题](KV压缩与稀疏访问.md#read-21)
- [11. MLA：Multi-head Latent Attention 到底具体怎么实现？](KV压缩与稀疏访问.md#read-22)
- [12. 为什么 MLA 是一个典型 Infra trade-off？](KV压缩与稀疏访问.md#read-23)
- [13. MLA 解决了 KV bytes，但没有解决 sequence length](KV压缩与稀疏访问.md#read-24)
- [Sparse Attention](KV压缩与稀疏访问.md#read-25)
- [14. DeepSeek Sparse Attention：DSA 到底怎么流？](KV压缩与稀疏访问.md#read-26)
- [15. DSA 的 Infra Pipeline](KV压缩与稀疏访问.md#read-27)
- [16. Sparse Attention 优化完之后，为什么 Indexer 自己会成为瓶颈？](KV压缩与稀疏访问.md#read-28)
- [17. DeepSeek-V4：CSA 是怎么实现的？](KV压缩与稀疏访问.md#read-29)
- [Compressed Sparse Attention](KV压缩与稀疏访问.md#read-30)
- [17.1 为什么 CSA 还保留 Sliding Window？](KV压缩与稀疏访问.md#read-31)
- [18. DeepSeek-V4：HCA 又是什么？](KV压缩与稀疏访问.md#read-32)
- [19. 从 DSA → CSA/HCA 的路线说明了什么？](KV压缩与稀疏访问.md#read-33)
- [20. Linear Attention：为什么另一批模型干脆不保存全部历史？](线性状态与混合架构.md#read-34)
- [21. Linear Attention 的基本数学心智模型](线性状态与混合架构.md#read-35)
- [22. Gated DeltaNet / Delta Attention 解决什么？](线性状态与混合架构.md#read-36)
- [23. Kimi Delta Attention：KDA 具体做了什么？](线性状态与混合架构.md#read-37)
- [24. KDA 为什么训练时还能并行？](线性状态与混合架构.md#read-38)
- [25. 为什么 Kimi K3 不是纯 KDA，而是 3:1 KDA + Gated MLA？](线性状态与混合架构.md#read-39)
- [26. Gated MLA 中的 Gate 是什么意义？](线性状态与混合架构.md#read-40)
- [27. Qwen3-Next：为什么 Qwen 也走 GDN + Attention Hybrid？](线性状态与混合架构.md#read-41)
- [28. Qwen3.8-Flash-Next：QSA 具体比普通 Sparse Attention多做了什么？](线性状态与混合架构.md#read-42)
- [28.1 Compressed Lightweight Indexer](线性状态与混合架构.md#read-43)
- [28.2 Micro-block Granularity](线性状态与混合架构.md#read-44)
- [29. GLM-5.2 IndexShare：为什么“共享 Indexer”也很重要？](线性状态与混合架构.md#read-45)
- [30. GLM-5.3-Flash：为什么又走 Linear + Sparse？](线性状态与混合架构.md#read-46)
- [31. 现代 Attention 其实正在变成“多级 Cache”](线性状态与混合架构.md#read-47)
- [32. 第三场革命：Residual Stream 为什么突然开始被重新设计？](残差与深度方向.md#read-48)
- [33. Kimi AttnRes：把层深方向也变成 Attention](残差与深度方向.md#read-49)
- [34. Full AttnRes 为什么很贵？](残差与深度方向.md#read-50)
- [35. Block AttnRes 如何把它工程化？](残差与深度方向.md#read-51)
- [36. DeepSeek mHC：为什么与 AttnRes 路线不同但目标相同？](残差与深度方向.md#read-52)
- [37. mHC 的 Infra 代价是什么？](残差与深度方向.md#read-53)
- [38. Qwen Gated Residual：4 路 residual stream 是怎么工作的？](残差与深度方向.md#read-54)
- [39. AttnRes / mHC / Gated Residual 应该如何统一理解？](残差与深度方向.md#read-55)
- [40. Kimi K3：Stable LatentMoE 到底新在哪里？](数值优化器与预测.md#read-56)
- [41. Kimi K3 的 Quantile Balancing 在解决什么？](数值优化器与预测.md#read-57)
- [42. SiTU / activation 为什么在极端 MoE 下也重要？](数值优化器与预测.md#read-58)
- [43. FP8 / FP4：为什么越来越重要？](数值优化器与预测.md#read-59)
- [44. 为什么 Quantization-Aware Training 比部署后量化更重要？](数值优化器与预测.md#read-60)
- [45. Qwen 的 N-gram Embedding：为什么是非常“系统型”的架构创新？](数值优化器与预测.md#read-61)
- [46. 为什么 N-gram Embedding 可以放 CPU？](数值优化器与预测.md#read-62)
- [47. 这为什么很像推荐系统？](数值优化器与预测.md#read-63)
- [48. Muon：为什么 optimizer 也是 Infra 话题？](数值优化器与预测.md#read-64)
- [49. 为什么更快收敛就是 Infra 优化？](数值优化器与预测.md#read-65)
- [50. Kimi 的 MuonClip：为什么 optimizer 改了又会制造新的稳定性问题？](数值优化器与预测.md#read-66)
- [51. MTP：为什么 Multi-Token Prediction 越来越常见？](数值优化器与预测.md#read-67)
- [52. 为什么 Autoregressive Decode 天生是 Infra 的痛点？](数值优化器与预测.md#read-68)
- [53. 为什么 GLM-5.2 还继续优化 MTP？](数值优化器与预测.md#read-69)
- [54. 现在把 Kimi K3 的整个 block 流程串起来](系统综合与资料.md#read-70)
- [55. Qwen3.8-Next 也可以按三个维度拆](系统综合与资料.md#read-71)
- [56. DeepSeek-V4 同样可以这样拆](系统综合与资料.md#read-72)
- [57. 四家路线的“风格”差异](系统综合与资料.md#read-73)
- [57.1 DeepSeek：Hardware-aware Full-stack Co-design](系统综合与资料.md#read-74)
- [57.2 Kimi：把 Scaling 分解为 Sequence / Depth / Width](系统综合与资料.md#read-75)
- [57.3 Qwen：高度 Serving-Oriented 的异构架构](系统综合与资料.md#read-76)
- [57.4 GLM：快速吸收主线结构并优化实际瓶颈](系统综合与资料.md#read-77)
- [58. 为什么 Dense 不会彻底被 MoE 淘汰？](系统综合与资料.md#read-78)
- [59. 一条最重要的 Infra 规律：Arithmetic Intensity 在改变模型设计](系统综合与资料.md#read-79)
- [60. 把你最近学习的 Infra 知识全部串起来](系统综合与资料.md#read-80)
- [61. 现代模型的“六层调度系统”](系统综合与资料.md#read-81)
- [第 1 层：Token / Sequence Routing](系统综合与资料.md#read-82)
- [第 2 层：Parameter Routing](系统综合与资料.md#read-83)
- [第 3 层：Depth Routing](系统综合与资料.md#read-84)
- [第 4 层：GPU Kernel Scheduling](系统综合与资料.md#read-85)
- [第 5 层：GPU/Network Scheduling](系统综合与资料.md#read-86)
- [第 6 层：Memory Tier Scheduling](系统综合与资料.md#read-87)
- [62. 未来 LLM 越来越像一个“操作系统 + 数据库 + HPC 程序”](系统综合与资料.md#read-88)
- [63. 一张表总结“创新 → 解决瓶颈 → 新瓶颈”](系统综合与资料.md#read-89)
- [64. 我认为真正的“总纲”](系统综合与资料.md#read-90)
- [65. 最终 Insight：未来不是“哪种 Attention 胜出”，而是 Hierarchical Memory](系统综合与资料.md#read-91)
- [66. 对 AI Infra 学习最重要的方法论](系统综合与资料.md#read-92)
- [67. 参考资料](系统综合与资料.md#read-93)
- [DeepSeek](系统综合与资料.md#read-94)
- [Kimi / Moonshot](系统综合与资料.md#read-95)
- [Qwen](系统综合与资料.md#read-96)
- [GLM / Z.ai](系统综合与资料.md#read-97)
- [68. 后续适合继续展开的专题](系统综合与资料.md#read-98)

</details>

<a id="deepseek--kimi--qwen--glm模型结构为什么这样变它们具体如何实现以及瓶颈如何迁移"></a>
<a id="read-01"></a>
<a id="从-ai-infra-视角理解-20242026-大模型结构演化"></a>
<a id="分主题正文"></a>
