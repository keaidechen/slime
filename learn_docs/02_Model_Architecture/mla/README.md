# MLA：压缩 KV 表示与执行变换

标准 KV 状态随历史 token 增长。MLA 从潜变量表示、位置处理与矩阵吸收解释如何降低状态宽度，再讨论新增计算、Kernel 和并行约束。

> 前置：[模型基础](../Transformer与Attention基础.md)。事实边界：[版本与验证范围](../../版本与验证范围.md)。

## 专题正文

- [基础与KV表示](基础与KV表示.md)
- [低秩投影与矩阵吸收](低秩投影与矩阵吸收.md)
- [执行模式与容量](执行模式与容量.md)
- [内核并行与运行时](内核并行与运行时.md)
- [边界验证与资料](边界验证与资料.md)

## 与相邻领域的关系

- [模型设计空间](../Attention设计空间与演进.md)：区分表示、访问集合与物理管理。
- [相邻专题](../sparse_attention/README.md)：比较可以组合或竞争的方法。
- [并行布局](../../08_Parallelism/统一切分语言.md)：解释状态归属与通信。
- [成本与测量](../../12_Performance_Reliability/成本模型与测量方法.md)：约束性能结论。

<details>
<summary>本专题章节索引</summary>

- [0. 先给结论](基础与KV表示.md#read-01)
- [1. 为什么标准 Attention 在 decode 阶段需要 KV Cache？](基础与KV表示.md#read-02)
- [2. 先把 MHA、MQA、GQA、MLA 放在同一个坐标系里](基础与KV表示.md#read-03)
- [3. 从一个“天真的 Low-rank KV”开始](低秩投影与矩阵吸收.md#read-04)
- [4. 完整 MLA：Content 分支与 RoPE 分支](低秩投影与矩阵吸收.md#read-05)
- [5. 为什么 RoPE 会破坏矩阵吸收？](低秩投影与矩阵吸收.md#read-06)
- [6. 矩阵吸收：为什么不用把历史 K/V 解压出来？](低秩投影与矩阵吸收.md#read-07)
- [7. MHA mode 与 MQA mode：同一个 MLA 为什么有两套执行图？](执行模式与容量.md#read-08)
- [8. Training、Prefill、Decode 三条完整数据流](执行模式与容量.md#read-09)
- [9. KV Cache 账本：MLA 到底省了多少？](执行模式与容量.md#read-10)
- [10. Paged Latent Cache：MLA 如何进入 vLLM / SGLang 一类运行时？](内核并行与运行时.md#read-11)
- [11. Roofline 视角：MLA 为什么是“以算换存”而不只是“少算”？](内核并行与运行时.md#read-12)
- [12. FlashMLA：算法变成高性能 kernel 时发生了什么？](内核并行与运行时.md#read-13)
- [13. 并行策略：为什么 MLA 与 TP 的关系很微妙？](内核并行与运行时.md#read-14)
- [14. MLA 与 Prefill–Decode Disaggregation](内核并行与运行时.md#read-15)
- [15. 一个接近官方实现的简化伪代码](内核并行与运行时.md#read-16)
- [16. 训练侧需要注意什么？](内核并行与运行时.md#read-17)
- [17. MLA 的收益边界与新瓶颈](边界验证与资料.md#read-18)
- [18. MLA 与相邻技术的区别](边界验证与资料.md#read-19)
- [19. 常见误区](边界验证与资料.md#read-20)
- [20. Profiling：如何判断 MLA 真的帮到了你的服务？](边界验证与资料.md#read-21)
- [21. 实现与排障清单](边界验证与资料.md#read-22)
- [22. FAQ](边界验证与资料.md#read-23)
- [23. 最终心智模型](边界验证与资料.md#read-24)
- [24. 参考资料与阅读顺序](边界验证与资料.md#read-25)
- [25. 与后续专题的接口](边界验证与资料.md#read-26)

</details>

<a id="mla从-kv-cache-压缩到推理内核的完整-infra-解析"></a>
<a id="分主题正文"></a>
