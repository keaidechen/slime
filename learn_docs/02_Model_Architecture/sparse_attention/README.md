# 稀疏注意力：访问集合与选择成本

把扫描全部历史转为访问选定位置，解释稀疏模式、索引器、Top-K、物理布局与质量；同时核算省下的工作和新增的选择成本。

> 前置：[模型基础](../Attention设计空间与演进.md)。事实边界：[版本与验证范围](../../版本与验证范围.md)。

## 专题正文

- [访问集合与稀疏模式](访问集合与稀疏模式.md)
- [索引器与Kernel数据流](索引器与Kernel数据流.md)
- [方法比较与成本边界](方法比较与成本边界.md)
- [参考实现验证与资料](参考实现验证与资料.md)

## 与相邻领域的关系

- [模型设计空间](../Attention设计空间与演进.md)：区分表示、访问集合与物理管理。
- [相邻专题](../linear_attention/README.md)：比较可以组合或竞争的方法。
- [并行布局](../../08_Parallelism/统一切分语言.md)：解释状态归属与通信。
- [成本与测量](../../12_Performance_Reliability/成本模型与测量方法.md)：约束性能结论。

<details>
<summary>本专题章节索引</summary>

- [0. 先给结论：Sparse Attention 优化的不是 Cache 宽度，而是访问集合](访问集合与稀疏模式.md#read-01)
- [1. Dense Attention 的真实成本：Prefill 和 Decode 不是同一个问题](访问集合与稀疏模式.md#read-02)
- [2. “Sparse Attention”不是一种结构，而是一组设计轴](访问集合与稀疏模式.md#read-03)
- [3. 稀疏模式的“工具箱”](访问集合与稀疏模式.md#read-04)
- [4. 从图算法理解 Sparse Attention](访问集合与稀疏模式.md#read-05)
- [5. 动态 Sparse Attention 的通用数学](访问集合与稀疏模式.md#read-06)
- [6. DeepSeek Sparse Attention（DSA）：结构与张量流](索引器与Kernel数据流.md#read-07)
- [7. DSA 如何训练：不能把离散 Top-K 直接扔给模型自己摸索](索引器与Kernel数据流.md#read-08)
- [8. Indexer 为什么会成为新瓶颈](索引器与Kernel数据流.md#read-09)
- [9. Top-K 不是一个小算子：为什么 logits 物化会爆](索引器与Kernel数据流.md#read-10)
- [10. 从逻辑 token 到物理 KV page](索引器与Kernel数据流.md#read-11)
- [11. Prefill Sparse Attention：二维 ragged 问题](索引器与Kernel数据流.md#read-12)
- [12. Decode Sparse Attention：低算术强度与大量小任务](索引器与Kernel数据流.md#read-13)
- [13. Continuous Batching：每个 Query 都有自己的稀疏世界](索引器与Kernel数据流.md#read-14)
- [14. FP8 Indexer 与 FP8 KV：省带宽，但 scale 也属于 layout](索引器与Kernel数据流.md#read-15)
- [15. Sparse MLA kernel：不只是把 Dense MLA 的 L 改成 K](索引器与Kernel数据流.md#read-16)
- [16. NSA：为什么 block sparse 对硬件更自然](方法比较与成本边界.md#read-17)
- [17. MInference：不重训模型时，怎样稀疏化长 Prefill](方法比较与成本边界.md#read-18)
- [18. 稀疏选择的质量问题：不是平均分高就够了](方法比较与成本边界.md#read-19)
- [19. Head、Layer 与 Query 之间能否共享选择结果](方法比较与成本边界.md#read-20)
- [20. 并行与分布式：Sparse Attention 并不会自动消除通信](方法比较与成本边界.md#read-21)
- [21. 性能模型：什么时候 Sparse 反而更慢](方法比较与成本边界.md#read-22)
- [22. 一个可实现的 Reference Pipeline](参考实现验证与资料.md#read-23)
- [23. Kernel 设计检查表](参考实现验证与资料.md#read-24)
- [24. Profiling：把端到端耗时拆对](参考实现验证与资料.md#read-25)
- [25. Troubleshooting：按症状找根因](参考实现验证与资料.md#read-26)
- [26. 常见误区](参考实现验证与资料.md#read-27)
- [27. Sparse Attention 与 Linear Attention：最终为什么常走向 Hybrid](参考实现验证与资料.md#read-28)
- [28. 一个数值算例：从 Dense MLA 到 Sparse MLA](参考实现验证与资料.md#read-29)
- [29. 如何选择方案：一个工程决策表](参考实现验证与资料.md#read-30)
- [30. 学习与实现路线](参考实现验证与资料.md#read-31)
- [31. 最终心智模型](参考实现验证与资料.md#read-32)
- [32. 参考资料与阅读顺序](参考实现验证与资料.md#read-33)
- [33. 与下一专题的接口](参考实现验证与资料.md#read-34)

</details>

<a id="sparse-attention从扫描全部历史到检索少量有效记忆"></a>
<a id="分主题正文"></a>
<a id="缩写与术语"></a>
