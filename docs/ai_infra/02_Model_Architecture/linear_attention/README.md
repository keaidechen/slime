# 线性注意力：递推状态与混合记忆

从 token 列表转向可更新状态，比较记忆容量、遗忘、覆写、训练并行与精确检索；把 Kernel、运行时和质量边界放在同一条问题链上。

> 前置：[模型基础](../Attention设计空间与演进.md)。事实边界：[版本与证据范围](../../reference/README.md)。

## 专题正文

- [基本定义与状态记忆](基本定义与状态记忆.md)
- [递推结构与演进](递推结构与演进.md)
- [并行执行与反向传播](并行执行与反向传播.md)
- [检索边界与混合模型](检索边界与混合模型.md)
- [运行时并行与数值](运行时并行与数值.md)
- [线性 Attention：递推状态的数值、质量与成本边界](递推状态的数值与质量边界.md)

## 与相邻领域的关系

- [模型设计空间](../Attention设计空间与演进.md)：区分表示、访问集合与物理管理。
- [相邻专题](../sparse_attention/README.md)：比较可以组合或竞争的方法。
- [并行布局](../../08_Parallelism/统一切分语言.md)：解释状态归属与通信。
- [成本与测量](../../12_Performance_Reliability/00_concepts.md#6-成本模型的适用范围)：约束性能结论。

<details>
<summary>本专题章节索引</summary>

- [缩写与术语](基本定义与状态记忆.md#read-01)
- [0. 先给结论：Linear Attention 把历史从 token 列表变成了一个程序状态](基本定义与状态记忆.md#read-02)
- [1. 先划清边界：哪些结构可以叫 Linear Attention](基本定义与状态记忆.md#read-03)
- [2. 为什么 Softmax Attention 不能直接交换乘法顺序](基本定义与状态记忆.md#read-04)
- [3. Kernelized Linear Attention 的完整推导](基本定义与状态记忆.md#read-05)
- [4. Feature Map：线性化 Softmax 的代价在哪里](基本定义与状态记忆.md#read-06)
- [5. 从 Attention 到 Fast Weight Programmer](基本定义与状态记忆.md#read-07)
- [6. Vanilla additive update 为什么不够](基本定义与状态记忆.md#read-08)
- [7. RetNet：把 decay、并行、递推和 chunkwise 统一](递推结构与演进.md#read-09)
- [8. RWKV：Transformer 训练形态与 RNN 推理形态的另一条路线](递推结构与演进.md#read-10)
- [9. S4、Mamba 与 Mamba-2：为什么必须写进 Linear Attention 演化史](递推结构与演进.md#read-11)
- [10. Hyena：别把所有线性时间模型都叫 Attention](递推结构与演进.md#read-12)
- [11. GLA：让忘记多少由当前数据决定](递推结构与演进.md#read-13)
- [12. Delta Rule：从“追加记忆”变成“纠错写入”](递推结构与演进.md#read-14)
- [13. DeltaNet：质量提高后，训练并行成为新瓶颈](递推结构与演进.md#read-15)
- [14. Gated DeltaNet：全局忘记与定向覆写结合](递推结构与演进.md#read-16)
- [15. KDA：把 forget gate 从每个 Head 细化到每个 Channel](递推结构与演进.md#read-17)
- [16. 三种执行模式：Recurrent、Parallel、Chunkwise](并行执行与反向传播.md#read-18)
- [17. Vanilla Linear Attention 的 Chunk 公式](并行执行与反向传播.md#read-19)
- [18. Chunk size 怎么选](并行执行与反向传播.md#read-20)
- [19. Gated/Delta Chunk 算法为什么更难](并行执行与反向传播.md#read-21)
- [20. Prefill 的真正目标：用 FLOPs 换掉长序列状态 IO](并行执行与反向传播.md#read-22)
- [21. Decode：复杂度不再随 Context 增长，但 State 本身成为带宽瓶颈](并行执行与反向传播.md#read-23)
- [22. State Layout：一个转置就能让长上下文静默损坏](并行执行与反向传播.md#read-24)
- [23. Variable-length batching 与 `cu_seqlens`](并行执行与反向传播.md#read-25)
- [24. Backward：为什么训练显存不会自动变成 $O(1)$](并行执行与反向传播.md#read-26)
- [25. Position：没有显式 Token Cache 后，顺序从哪里来](并行执行与反向传播.md#read-27)
- [26. 为什么 Fixed State 天然不擅长 Exact Retrieval](检索边界与混合模型.md#read-28)
- [27. Hybrid Attention：不是过渡方案，而是 Memory Hierarchy](检索边界与混合模型.md#read-29)
- [28. Hybrid 的三种组织方式](检索边界与混合模型.md#read-30)
- [29. 现代开源模型路线对比](检索边界与混合模型.md#read-31)
- [30. 其他具有里程碑意义的 Hybrid 模型](检索边界与混合模型.md#read-32)
- [31. 里程碑时间线](检索边界与混合模型.md#read-33)
- [32. Serving Runtime：Linear State 不是普通 KV Cache](运行时并行与数值.md#read-34)
- [33. Speculative Decoding：回滚 Recurrent State 为什么困难](运行时并行与数值.md#read-35)
- [34. Context Parallelism：状态转移的组合比 KV All-Gather 更微妙](运行时并行与数值.md#read-36)
- [35. Tensor/Head Parallelism：State 应该怎么切](运行时并行与数值.md#read-37)
- [36. Kernel Fusion 清单](运行时并行与数值.md#read-38)
- [37. FlashKDA 与 FLA：从研究公式到可部署算子](运行时并行与数值.md#read-39)
- [38. 数值稳定性](运行时并行与数值.md#read-40)
- [39. 性能模型](运行时并行与数值.md#read-41)
- [40. 正确的 Benchmark 方法](递推状态的数值与质量边界.md#read-42)
- [41. Quality Evaluation](递推状态的数值与质量边界.md#read-43)
- [42. Troubleshooting：按症状定位](递推状态的数值与质量边界.md#read-44)
- [43. 常见误区](递推状态的数值与质量边界.md#read-45)
- [44. 实现检查表](递推状态的数值与质量边界.md#read-46)
- [45. 一个最小 Reference 实现](递推状态的数值与质量边界.md#read-47)
- [46. 一个端到端数量级算例](递推状态的数值与质量边界.md#read-48)
- [47. 如何选择架构](递推状态的数值与质量边界.md#read-49)
- [48. Infra 学习路线](递推状态的数值与质量边界.md#read-50)
- [49. 最终心智模型](递推状态的数值与质量边界.md#read-51)
- [50. 参考资料与推荐阅读顺序](递推状态的数值与质量边界.md#read-52)
- [51. 与下一专题的接口](递推状态的数值与质量边界.md#read-53)

</details>


## 原稿资料时间与论证口径

以下保留原稿的时间范围；博客化重构没有将其改成当前版本的全量核验。

本文面向模型架构、训练系统和推理 Infra 学习者。公式首先解释结构本身，再落到 prefill、decode、GPU kernel、状态管理和 Serving runtime。论文或模型报告中的速度数字只作为作者特定实验的结果，不直接外推到其他模型、硬件和工作负载。

**资料版本：截至 2026-09-02。** 其中 GDN-2、DART、Qwen3.8-Flash-Next、GLM-5.3-Flash 等属于快速演进中的近期工作；文中会区分论文结论、作者报告和本文推导，具体 kernel 支持范围以对应仓库当前版本为准。

本专题保留原稿的模型实例和历史支持描述；它们尚未逐项完成当前版本核验，不能当作实时支持矩阵。通用推导按文中假设阅读，具体模型和性能结论应回到各节一手资料及其版本。[证据范围](../../reference/README.md)。
