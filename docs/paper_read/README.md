# 论文与设计取舍

按工程问题读论文。Runtime 导读与共享硬件基础已接入对应课程，不需要重复读旧入口。

| 问题 | 论文 | 首次掌握 |
|---|---|---|
| Attention 的 IO | [FlashAttention](01_FlashAttention论文详解.md) | tiling、减少中间结果搬运 |
| KV 分页 | [PagedAttention](02_PagedAttention-vLLM论文详解.md) | 逻辑/物理块、碎片、共享/COW |
| 前缀复用 | [RadixAttention](03_SGLang-RadixAttention论文详解.md) | prefix tree、锁、驱逐 |
| 动态 batch | [Orca](06_Orca与Continuous-Batching论文详解.md) | iteration-level scheduling |
| PD 分离 | [DistServe](07_DistServe与Prefill-Decode分离论文详解.md) | 干扰与 KV 交接成本 |
| Kernel 深入 | [FA2](04_FlashAttention-2论文详解.md) · [FA3](05_FlashAttention-3论文详解.md) | 分工、流水与资源 |
| 不规则 KV | [FlashInfer](<FlashInfer 论文详解.md>) | backend、paged/ragged layout |

随 SGLang 主线先读 PagedAttention、Orca、RadixAttention 的首读部分；FA1 建立 IO 直觉，其余按瓶颈深入。[技术演化总览](11_LLM推理Runtime技术演化总览.md)作为地图。

配套：[GPU 基础](../../learn_docs/00_Foundations/04_GPU执行与内存基础.md)、[KV 基础](../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md)、[SGLang 概念](../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md)、[Runtime 对比](../runtime_comparisons/README.md)。论文结论按原工作负载理解；当前实现以本仓库代码为准。
