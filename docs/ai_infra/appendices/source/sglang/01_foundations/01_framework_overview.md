# SGLang 框架与状态概览

相关基础：[Transformer 与 KV](<../../../../05_Inference/19_生成执行与采样.md#s1>)。


SGLang 将一条请求分散到入口、调度器、缓存管理、模型执行和输出处理等职责中。本页按组件定位实现；生成、KV 与服务机制由正文第 19—23 章解释。

## 1. 先分开模型与服务

模型定义“输入这些 token，应输出什么分布”；框架还需要决定“哪些输入现在执行、哪些资源必须保留”。以两条请求 A=`[10,11,12]`、B=`[10,11,99]` 为例：API 保存各自的连接，Scheduler 决定两条是否同轮执行，KV 管理器识别共享的 `[10,11]`，ModelRunner 把决定变成 GPU 张量，输出层再把各自的 token 交还各自连接。不能让模型只输出两个 token，却丢失它们属于 A 还是 B 的对应关系。

## 2. 一条请求与一批执行是两种尺度

请求的生命周期可能持续几十轮，batch（批次）只描述一轮执行。A 完成后，下一批可以由 B 和新来的 C 组成。`rid` 是贯穿生命周期的请求标识；batch 中的第 0 行只是本轮的位置。把两者混用会在批次压缩后把 B 的结果发给 C。

下面的表按“持有哪种状态”划分职责。进程是操作系统隔离的执行单位，组件是逻辑职责；一个进程可以包含多个组件，一项模型执行也可以横跨多个 GPU 进程。

| 组件 | 实现资料 | 负责的状态 |
|---|---|---|
| 请求入口 | [请求链](02_process_topology_and_request_path.md) | 进程、消息、请求身份 |
| Scheduler | [调度实现](../02_runtime_core/01_scheduler_and_batch.md) | 队列、批次与预算 |
| KV 管理 | [缓存实现](../02_runtime_core/02_kv_cache_and_radix_attention.md) | 页、pool、引用与回收 |
| ModelRunner | [执行实现](../02_runtime_core/03_model_runner_attention_cuda_graph.md) | backend、形状与执行图 |
| 采样接口 | [接口实现](../04_interfaces_and_models/01_api_request_and_sampling.md) | 输出 token、概率与配置 |
| 控制面 | [发布实现](../04_interfaces_and_models/03_control_plane_and_post_training.md) | 换权、暂停与失败状态 |

## 3. 用一次交接判断所有权

| 状态 | 何时建立 | 谁消费 | 何时可清理或复用 |
|---|---|---|---|
| API 等待记录 | 收到请求并分配 `rid` | 输出接收循环、HTTP 协程 | 最终结果或取消路径处理完成 |
| 请求映射表行 | 调度准入成功 | Attention 的地址查找 | GPU/异步消费者不再引用后 |
| 历史 KV | 每轮 forward 写入 | 后续 decode、相容前缀请求 | 活跃引用归零，且缓存决定驱逐后 |
| 本轮输入张量 | 构造 batch | 当前设备 forward | 设备读完后；复用须有同步依赖 |

“发送一条消息”只说明跨了控制边界，不说明下游完成。例如 API 把取消消息发出后，Scheduler 仍要在安全边界终止请求，不能立即把 GPU 正在读的 slot 分给别人。后续章节就是把这张表中的“完成条件”落实到字段、事件和函数。

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
