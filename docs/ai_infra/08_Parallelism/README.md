# 并行计算：状态由谁持有，工作怎样组合

单卡显存不足、单层太慢和样本吞吐不足是不同问题。[八类并行阅读主线](八类并行阅读主线.md)先比较 DP、ZeRO/FSDP、TP、PP、SP、CP、EP 与 ETP，再链接到各篇的完整推导和 Megatron 实现。

## 首读与深入

首次沿 **[DP/DDP](DP与DDP.md) → [TP](TensorParallelism.md) → [ZeRO/FSDP](ZeRO与FSDP.md) → [PP](PipelineParallelism.md)** 理解副本、算子、状态与层四种切分；逐项数值和时间线已在这些正文中。

| 进一步的问题 | 文章 |
|---|---|
| 副本、分片和部分和怎样统一描述？ | [设计空间](并行设计空间与演进.md) · [统一切分语言](统一切分语言.md) |
| 序列太长，哪些状态和计算能够切分？ | [SP](SequenceParallelism.md) · [CP](ContextParallelism.md) |
| 专家分散后，token 和参数各归谁？ | [EP](ExpertParallelism.md) · [ETP](ExpertTensorParallelism.md) |
| 大词表和多维组合怎样落到设备？ | [词表并行](Embedding与词表并行.md) · [DeviceMesh](多维并行与DeviceMesh.md) |
| 为什么不能只把所有并行度相乘？ | [方案的约束与取舍](并行方案的约束与取舍.md) |
| 能否自动找到更好的切分？ | [自动并行与代价模型](自动并行与代价模型.md) |
| 具体模型怎样组合这些方法？ | [DeepSeek 与 DualPipe](案例_DeepSeek与DualPipe.md) |
| 这些方案为什么先后出现？ | [训练并行演进史](训练并行演进史.md) |

布局决定哪些数据必须交换，但重叠是否成功取决于实际到达时间和依赖。接着可读[训练等待与通信重叠案例](../12_Performance_Reliability/贯穿案例_分布式训练等待与通信重叠.md)，或进入[Megatron 源码](../13_Source_Analysis/megatron/README.md)。

[默认主线](../guides/系统阅读路线.md) · [完整专题](../guides/专题导航.md)
