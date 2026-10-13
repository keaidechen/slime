# 论文详解资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../README.md>)。

- [01｜FlashAttention 论文详解：为什么少算不一定更快，而少搬数据可以更快](<01_FlashAttention论文详解.md>)
- [02｜PagedAttention / vLLM 论文详解：把 KV Cache 当成操作系统的虚拟内存](<02_PagedAttention-vLLM论文详解.md>)
- [03｜SGLang / RadixAttention 论文详解：从 Paged KV 到“可自动复用的前缀树缓存”](<03_SGLang-RadixAttention论文详解.md>)
- [04｜FlashAttention-2 论文详解：当“减少 HBM IO”之后，为什么还要重新设计并行方式？](<04_FlashAttention-2论文详解.md>)
- [05｜FlashAttention-3 论文详解：为什么 H100 需要“异步 Attention”？](<05_FlashAttention-3论文详解.md>)
- [06｜Orca 与 Continuous Batching：LLM Serving 为什么不能使用传统“静态 Batch”？](<06_Orca与Continuous-Batching论文详解.md>)
- [07｜DistServe 与 Prefill-Decode Disaggregation：为什么要把一次 LLM 推理拆到两组 GPU？](<07_DistServe与Prefill-Decode分离论文详解.md>)
- [11｜LLM 推理 Runtime 技术演化总览：从 FlashAttention 到 vLLM / SGLang / TensorRT-LLM](<11_LLM推理Runtime技术演化总览.md>)
- [FlashInfer：动态 Attention 如何统一表示、编译与调度](<FlashInfer 论文详解.md>)
