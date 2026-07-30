# SGLang Runtime 系统学习与代码走读

这套文档面向对 SGLang 代码库零基础、但希望达到推理 infra 工程师水平的读者。这里的核心对象是高性能 serving runtime（`sglang.srt`），不是只学习 OpenAI API 的使用方法。

完成后，你应能从一个 HTTP 请求追到 GPU kernel，再从 token 输出追回应答流；能解释 continuous batching、RadixAttention、paged KV、chunked prefill、overlap scheduling、CUDA Graph、TP/DP/EP、PD disaggregation 和 speculative decoding 的正确性边界。

## 阅读基线

- 本仓库 SGLang 快照：`f5155d960286db25952217f343ee0d3c358f7f77`
- 源码演进很快，文中优先给稳定符号名；行号对应上述快照。
- slime 集成视角另见 `docs/code_walkthrough/11_engine_internals_sglang.md`；本系列聚焦 SGLang Runtime 本身。

## 推荐顺序

| 顺序 | 文档 | 学完应能做到 |
|---|---|---|
| 0 | [00_learning_map.md](00_learning_map.md) | 建立推理 infra 能力地图和性能模型 |
| 1 | [01_process_topology_and_request_path.md](01_process_topology_and_request_path.md) | 跟踪 HTTP→Tokenizer→Scheduler→Detokenizer |
| 2 | [02_scheduler_and_batch.md](02_scheduler_and_batch.md) | 理解请求状态机、prefill/decode 与 continuous batching |
| 3 | [03_kv_cache_and_radix_attention.md](03_kv_cache_and_radix_attention.md) | 读懂 KV 两级映射、Radix tree、锁与驱逐 |
| 4 | [04_model_runner_attention_cuda_graph.md](04_model_runner_attention_cuda_graph.md) | 跟踪 ForwardBatch、attention backend、CUDA Graph |
| 5 | [05_distributed_and_pd_disaggregation.md](05_distributed_and_pd_disaggregation.md) | 设计 TP/DP/EP/PP 与 prefill-decode 解耦部署 |
| 6 | [06_speculative_structured_sampling.md](06_speculative_structured_sampling.md) | 理解推测解码验证、采样和约束输出 |
| 7 | [07_production_performance_and_extension.md](07_production_performance_and_extension.md) | 做可靠 benchmark、观测、排障与新模型接入 |

## 合格标准

- 能准确区分 TTFT、TPOT、ITL、E2E latency、goodput 与 throughput；
- 能解释为什么 prefill compute-bound、decode memory/latency-bound，以及混批的干扰；
- 能画出 request pool slot→token KV index→实际 K/V storage 的映射；
- 能证明 radix cache 命中、节点 split、lock ref、eviction 不会释放在用 KV；
- 能解释 overlap scheduler 为何有一批结果延迟，以及何时必须禁用 overlap；
- 能针对共享前缀、高并发短请求、长上下文、MoE、RL rollout 选择不同配置；
- 能用 workload-aware benchmark 和 trace 判断瓶颈，而非只跑一个峰值 tokens/s。

## 一手资料

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang 官方文档](https://docs.sglang.io/)
- [SGLang 论文](https://arxiv.org/abs/2312.07104)
- [RadixAttention 介绍](https://lmsys.org/blog/2024-01-17-sglang/)
- [SGLang v0.4 调度器文章](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- [PD Disaggregation 文档](https://docs.sglang.ai/backend/pd_disaggregation.html)
- [官方学习资料与 slides](https://github.com/sgl-project/sgl-learning-materials)

## 如何使用这套文档

SGLang 主仓库功能很多，直接从 `Scheduler.__init__` 顺序读很容易淹没。每章采用同一种方法：

1. **只跟一个普通文本生成请求**，暂时关闭 speculative、LoRA、PD、VLM；
2. 建立 request、batch、KV 三本账；
3. 再逐个打开高级功能，看它在哪本账上增加状态；
4. 最后用 trace 验证 CPU scheduler 与 GPU forward 的时间关系。

建议个人维护：

| 账本 | 最关键字段 | 何时创建 | 何时释放 |
|---|---|---|---|
| Request | rid、input/output、finish、grammar | API/scheduler 接收 | finish/abort |
| Batch | forward mode、seq lens、sampling info | 每 scheduler iteration | result process 后 |
| KV | req slot、physical indices、radix lock | admission/prefix match | cache/evict/free |

### 源码阅读工具

```bash
rg -n "class Scheduler|def event_loop_overlap" \
  sglang/python/sglang/srt

rg -n "req_pool_idx|out_cache_loc|lock_ref" \
  sglang/python/sglang/srt

# benchmark 一定同时保留请求长度分布和原始 JSONL
python -m sglang.bench_serving ... --output-details
```

SGLang 演进很快，看到文档与代码不一致时，以本仓库固定 commit 为本教材基线，以在线官方文档确认最新版行为。
