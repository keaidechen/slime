# 第二部分：Runtime 核心

这一部分从调度、内存、执行、生成控制四条主线解释 `sglang.srt`。建议按顺序阅读，因为后一篇会使用前一篇建立的 request/batch/KV 状态。

1. [Scheduler、请求状态机与连续批处理](01_scheduler_and_batch.md)
2. [KV Cache、RadixAttention 与引用锁](02_kv_cache_and_radix_attention.md)
3. [ModelRunner、Attention Backend 与 CUDA Graph](03_model_runner_attention_cuda_graph.md)
4. [推测解码、约束输出与采样正确性](04_speculative_structured_sampling.md)

