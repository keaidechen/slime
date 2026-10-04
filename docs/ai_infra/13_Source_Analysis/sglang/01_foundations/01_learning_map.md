# SGLang 学习地图

相关基础：[Transformer 与 KV](<../../../10_Inference_Serving/Transformer执行与KV基础.md>)。


先读[Runtime 概念课](00_runtime_concepts.md)，再跟一条普通文本请求。通用 GPU、KV、并行与指标分别在基础课和性能教材维护，本页只负责源码阅读顺序。

| 次序 | 主章节 | 产物 |
|---|---|---|
| 1 | [请求链](02_process_topology_and_request_path.md) | 进程、消息、请求对象 |
| 2 | [Scheduler](../02_runtime_core/01_scheduler_and_batch.md) | waiting/running 与 token/KV 预算 |
| 3 | [KV](../02_runtime_core/02_kv_cache_and_radix_attention.md) | 页、pool、radix 锁与回收 |
| 4 | [ModelRunner](../02_runtime_core/03_model_runner_attention_cuda_graph.md) | backend、shape 与 CUDA Graph |
| 5 | [采样接口](../04_interfaces_and_models/01_api_request_and_sampling.md) | token/logprob 与配置 |
| 6 | [控制面](../04_interfaces_and_models/03_control_plane_and_post_training.md) | 换权、暂停与失败边界 |

原学习地图中的计算模型和读码方法已并入[概念课的计算模型](00_runtime_concepts.md#learning-models)。首次只开启普通文本路径；逐项加入 speculative、LoRA、PD、多模态，并记录新增状态与释放者。

## 4. 源码地图

```text
sglang/python/sglang/srt/
  entrypoints/       HTTP、OpenAI protocol、Engine API
  managers/          tokenizer、scheduler、batch、detokenizer、TP worker
  mem_cache/         request/token pool、radix/HiCache、allocator
  model_executor/    ModelRunner、ForwardBatch、CUDA graph
  layers/            attention、logits、quantization、MoE 等
  sampling/          sampling metadata 与算子
  speculative/       EAGLE/MTP/ngram 等
  constrained/       grammar backend
  disaggregation/    prefill/decode 与 KV transfer
  observability/     metrics、tracing、request time
```


## 四种请求差异怎样改变状态

相同与不同前缀改变 prefix match 和共享 KV，长度与并发改变预算及排队，正常完成与取消改变清理路径。Speculative、LoRA、PD 与多模态进一步增加缓存、版本和跨进程所有权。每次增加一种机制，都需要说明新增状态由谁持有、最后由谁释放。

[推理性能](../../../12_Performance_Reliability/05_inference.md)把这些差异连接到容量与延迟，[系统原理](../../../00_Physical_Foundations/README.md)解释共同的计算和移动成本。

<!-- series-nav -->

[上一篇](<00_runtime_concepts.md>) · [本系列目录](<../README.md>) · [下一篇](<02_process_topology_and_request_path.md>)

<!-- /series-nav -->
