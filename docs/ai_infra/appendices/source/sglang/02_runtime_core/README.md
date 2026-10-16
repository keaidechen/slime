# 02_runtime_core资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../../README.md>)。

先从 Scheduler 的两本预算入手，再看它依赖的 KV 映射和引用锁；随后跟着 `ForwardBatch` 把计划送入设备，最后理解一轮产生多个 token 或施加语法约束时发生了什么。

本组以 page size=1、短序列、单卡为起点，页大小、并行和异步再逐项增加。读完应能手工写出两请求 batch 的 `input_ids`、`seq_lens`、`out_cache_loc`，算出每 token KV 字节数，并解释采样的接受/拒绝为什么保持目标概率。

- [2.1 Scheduler、请求状态机与连续批处理](<01_scheduler_and_batch.md>)
- [2.2 KV Cache、RadixAttention 与引用锁](<02_kv_cache_and_radix_attention.md>)
- [2.3 ModelRunner、Attention Backend 与 CUDA Graph](<03_model_runner_attention_cuda_graph.md>)
- [2.4 推测解码、约束输出与采样正确性](<04_speculative_structured_sampling.md>)
