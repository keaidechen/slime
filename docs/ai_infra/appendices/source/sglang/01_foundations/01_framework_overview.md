# SGLang 框架与状态概览

相关基础：[Transformer 与 KV](<../../../../05_Inference/19_生成执行与采样.md#s1>)。


SGLang 将一条请求分散到入口、调度器、缓存管理、模型执行和输出处理等职责中。本页按组件定位实现；生成、KV 与服务机制由正文第 19—23 章解释。

| 组件 | 实现资料 | 负责的状态 |
|---|---|---|
| 请求入口 | [请求链](02_process_topology_and_request_path.md) | 进程、消息、请求身份 |
| Scheduler | [调度实现](../02_runtime_core/01_scheduler_and_batch.md) | 队列、批次与预算 |
| KV 管理 | [缓存实现](../02_runtime_core/02_kv_cache_and_radix_attention.md) | 页、pool、引用与回收 |
| ModelRunner | [执行实现](../02_runtime_core/03_model_runner_attention_cuda_graph.md) | backend、形状与执行图 |
| 采样接口 | [接口实现](../04_interfaces_and_models/01_api_request_and_sampling.md) | 输出 token、概率与配置 |
| 控制面 | [发布实现](../04_interfaces_and_models/03_control_plane_and_post_training.md) | 换权、暂停与失败状态 |

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

[推理性能](<../../../performance/05_inference.md>)把这些差异连接到容量与延迟，[系统原理](<../../../foundations/README.md>)解释共同的计算和移动成本。
