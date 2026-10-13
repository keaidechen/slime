# 附录与查询索引

本书正文按照[七篇、32 章目录](../README.md)连续阅读。这里集中保存特殊结构、实现细节、论文与工具材料，供核对正文中的具体问题，不另设必读顺序。

## 按正文主题查阅

| 正文位置 | 补充资料 |
|---|---|
| 导论与共同设计语言 | [负载与成本](design/README.md) |
| 模型计算 | [数学与张量](mathematics/README.md)、[模型结构与多模态](models/README.md) |
| 设备执行 | [物理基础](foundations/README.md)、[硬件](hardware/README.md)、[内存](memory/README.md)、[算子](kernels/README.md)、[编译器](compilers/README.md)、[通信](communication/README.md) |
| 训练系统 | [并行](parallelism/README.md)、[训练](training/README.md) |
| 推理系统 | [服务专题](serving/README.md)、[运行时对照](runtime/README.md) |
| 后训练与 RL | [RL 演进资料](rl/README.md) |
| 性能与综合实践 | [性能工具与诊断](performance/README.md)、[CUDA 实验材料](../cuda/README.md) |

## 论文与固定源码

[论文详解](papers/README.md)保留原始问题、推导和实验条件；[源码资料](source/README.md)解释固定版本的实际分支。正文负责通用概念，不要求先读论文或源码才能理解后续章节。

源码资料分别覆盖 [Megatron](source/megatron/README.md)、[SGLang](source/sglang/README.md)、[slime](source/slime/README.md)和 [FlashAttention](source/flash_attention/README.md)。它们是实现查询目录，不构成另一套全书章序。

## 索引与证据

[术语、问题、算例与历史索引](indexes/README.md)按查询对象定位；[数值程序](../examples/README.md)用于复算；[证据与维护](../reference/README.md)说明实验和实现的适用范围。
