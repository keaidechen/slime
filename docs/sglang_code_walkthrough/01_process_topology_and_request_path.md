# 01. 进程拓扑与端到端请求路径

## 1. 为什么是多进程

典型 SGLang server 将高延迟/高吞吐环节拆开：

```text
HTTP server / TokenizerManager
          │ ZMQ/IPC
          ▼
Scheduler + TP ModelWorker(s)
          │ outputs
          ▼
DetokenizerManager
          │ text chunks
          └────────→ TokenizerManager → HTTP stream
```

拆分让 tokenization/detokenization 的 CPU 工作不阻塞 GPU scheduler，也隔离 TP worker。具体拓扑会随 DP、PP、multi-node、disaggregation 和 grpc/ray 模式变化。

## 2. 服务入口

- CLI：`sglang/launch_server.py`
- HTTP 主入口：`srt/entrypoints/http_server.py:2648` 的 `launch_server`
- Engine API：`srt/entrypoints/engine.py`
- scheduler 子进程入口：`srt/managers/scheduler.py:4662` 的 `run_scheduler_process`

启动阶段做的事情比“load model”多：

1. 解析并 normalize `ServerArgs`；
2. 分配端口和 IPC endpoint；
3. spawn tokenizer/scheduler/detokenizer 等进程；
4. 初始化 distributed group、GPU、memory pool、model；
5. 收集各 scheduler 的 init result；
6. 完成 warmup/readiness 后才接流量。

生产中 liveness 与 readiness 必须分开：进程活着不表示权重加载、graph capture、跨 rank rendezvous 已完成。

## 3. `TokenizerManager`

主类位于 `srt/managers/tokenizer_manager.py:265`。职责包括：

- 校验/规范化请求；
- chat template、tokenization、多模态输入处理；
- 为请求建立 `ReqState`（约 172 行）；
- 向 scheduler 发送 tokenized request；
- 按 request id 聚合异步/流式输出；
- abort/control request；
- 将 token 输出与 detokenizer 结果关联。

一个 HTTP coroutine 不能同步等待 GPU；它通常注册状态后异步迭代 output queue。请求 id 是跨进程关联键，重复、泄漏或生命周期错误会造成串流/内存泄漏。

## 4. `Scheduler`

主类位于 `srt/managers/scheduler.py:326`。初始化包含：

- model worker/runner；
- waiting queue 与 running batch；
- request pool、KV allocator、prefix cache；
- schedule policy；
- sampler/grammar/speculative worker；
- metrics、watchdog、weight updater；
- IPC channels。

它是资源所有权中心：决定哪个请求进入 batch、拿多少 KV slot、何时释放、forward 后状态如何推进。

## 5. `DetokenizerManager`

位于 `srt/managers/detokenizer_manager.py:91`。增量 detokenize 不能每次简单 decode 全序列，否则 CPU 开销随输出长度二次增长。它需处理：

- tokenizer 的不完整 byte/Unicode 边界；
- stop string；
- skip special tokens；
- streaming chunk 的稳定前缀；
- logprob/token metadata；
- abort/finish 后清理。

“生成 token 已完成”和“可安全向用户发出文本”不是同一时刻。

## 6. 一次请求的时序

```text
POST /v1/chat/completions
  → protocol validation / chat template
  → tokenize → GenerateReqInput
  → IPC send to Scheduler
  → Req created, enters waiting queue
  → prefix match + KV budget
  → prefill/extend batch → first sampled token
  → repeated decode batches
  → token results → Detokenizer
  → streaming chunks → HTTP
  → EOS/stop/max_tokens/abort
  → cache finished prefix + release req slot/locks
```

## 7. IPC 排障方法

为每类 message 记录：

```text
request_id, message_type, source, destination,
monotonic_timestamp, input/output token count, finish_reason
```

若 HTTP 卡住：

1. 请求是否成功发给 scheduler；
2. scheduler 是否创建 `Req`；
3. 是否一直 waiting（为什么）；
4. 是否 forward 完成；
5. output 是否送到 detokenizer；
6. detokenized result 是否回到正确 `ReqState`；
7. HTTP coroutine 是否被 client disconnect/timeout 取消。

不要只看 GPU utilization；控制面丢一条消息时 GPU 可能完全正常。

## 8. `TokenizerManager.__init__` 源码精读

固定快照 `tokenizer_manager.py:265` 的初始化顺序：

```python
self.init_model_config()
self.init_tokenizer_and_processor()
self.init_ipc_channels(port_args)
self.init_running_status()
self.init_request_logging_and_dumping()
self.init_weight_update()
self.init_lora()
self.init_disaggregation(...)
self.init_metric_collector_watchdog()
self.init_request_dispatcher()
```

顺序表达了依赖：

- model config 决定 tokenizer、context length、是否 generation/VLM；
- IPC 建立后才能注册 running request 的收发状态；
- LoRA、PD 会改变 cache namespace 与路由；
- dispatcher 最后初始化，避免准备未完成时接收请求。

`init_model_config()` 对 EAGLE 还会预留输出 token slots：

```python
self.num_reserved_tokens = max(
    topk * num_steps,
    max_speculative_num_draft_tokens,
)
```

所以 API 层的最大长度校验不能只看用户 `max_new_tokens`，还需给 draft 临时 token 留空间。

## 9. 启动握手

阅读 `entrypoints/http_server.py:launch_server` 与 `managers/scheduler.py:run_scheduler_process`，把启动拆成：

```text
parent 分配 PortArgs
  → spawn 子进程
  → scheduler 设置 GPU/distributed
  → ModelRunner load weights / profile KV / capture graph
  → SchedulerInitResult 返回 capacity 和状态
  → TokenizerManager 得到 max_req_input_len 等运行结果
  → HTTP readiness
```

模型 config 中的 theoretical context length 不等于当前服务器能接收的最大输入；实际还受 KV capacity、最大 running requests、spec reserved tokens 和 multimodal padding 影响。

## 10. 请求 id 的生命周期

一个 `rid` 同时存在于：

```text
HTTP coroutine / ReqState
IPC request
Scheduler Req
Detokenizer state
metrics/trace
```

必须满足：

- 创建前无同名 active state；
- streaming chunk 单调增加；
- finish/abort 只提交一次；
- late message 不能命中新复用状态；
- client disconnect 触发 scheduler abort，而非只取消 HTTP task。

建议 IPC/trace 中加入 request generation 或唯一 UUID，而不是用户可重复的业务 id。

## 11. streaming 细节

detokenize 一个 token id 并不一定立刻产生稳定字符：

```text
UTF-8 一个字符可能跨多个 tokenizer token
SentencePiece/BPE token 可能依赖前后空格
stop string 可能跨 chunk
```

因此 detokenizer 保留 read offset 与 decoded offset，只发送确认稳定的新 suffix。若为了“低延迟”每次直接 `tokenizer.decode([new_id])`，拼接文本可能与整序列 decode 不同。

## 12. 单请求观测实验

以 native `/generate` 发 `max_new_tokens=3`，记录：

| 时刻 | 预期对象 |
|---|---|
| HTTP 接收 | text request |
| tokenize 后 | input ids |
| scheduler receive | `Req` 尚无 pool slot |
| admission | `req_pool_idx`、prefix match |
| prefill result | 首 sampled id |
| detokenizer | 首稳定文本 |
| 两轮 decode | output ids 追加 |
| finish | cache/free/unlock |

关闭 streaming 再跑一次：GPU 路径应基本相同，差别主要在 HTTP response 聚合。

## 13. 本章延伸阅读

- [SGLang Basic Usage](https://docs.sglang.io/docs/basic_usage/send_request)：把 API payload 与内部请求字段对应。
- [OpenAI-compatible API 文档](https://docs.sglang.io/docs/basic_usage/openai_api)：理解 protocol adapter 的边界。
- [Mini-SGLang 博客](https://www.lmsys.org/blog/2025-12-17-minisgl/)：包含 frontend/tokenizer/scheduler 的精简架构图。
- [SGLang Runtime 性能博客](https://www.lmsys.org/blog/2024-07-25-sglang-llama3/)：理解为何 Python scheduler 仍可达到高吞吐。
