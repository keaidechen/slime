# 2.3 ModelRunner、Attention Backend 与 CUDA Graph

## 1. 调用链

```text
Scheduler.run_batch
  → TpModelWorker.forward_batch_generation
  → ForwardBatch.from_schedule_batch(...)
  → ModelRunner.forward
  → model.forward
  → attention backend / MoE / kernels
  → logits processor
  → sampler
  → GenerationBatchResult
```

关键符号：

- `TpModelWorker`：`srt/managers/tp_worker.py:273`
- `ForwardBatch`：`srt/model_executor/forward_batch_info.py:372`
- `ModelRunner`：`srt/model_executor/model_runner.py:241`

Scheduler 拥有请求对象；ModelRunner 应消费张量化执行描述。这个边界避免模型代码依赖 Python `Req`。

## 2. `ForwardBatch` 携带什么

常见字段包括：

- forward mode（extend/decode/idle/target verify 等）；
- input ids、positions；
- batch size、sequence lengths；
- request pool indices；
- KV pool locations；
- prefix/extend lengths；
- attention backend metadata；
- sampling/logprob 信息；
- multimodal、LoRA、speculative、DP/PP metadata。

它不是普通 `(input_ids, attention_mask)`。paged attention 不构造巨大 dense mask，而依靠每请求长度与物理 KV mapping 找历史。

## 3. attention backend 的职责

不同 backend（FlashInfer、FlashAttention、Triton、MLA 专用实现等）通常分两步：

1. `init_forward_metadata(...)`：根据 batch 形状、长度和 KV mapping 构造 workspace/index；
2. layer forward：复用 metadata 执行每层 attention。

把 metadata 预计算放在 layer 外，避免每层重复 CPU/GPU 工作。新增 backend 必须覆盖：

- extend 与 decode；
- causal/window/prefix 语义；
- MHA/GQA/MLA；
- page size、KV dtype；
- CUDA graph capture；
- speculative verify；
- TP/DP attention；
- logprob/空 batch 边界。

## 4. ModelRunner 初始化为何重

通常包括：

- 选择 device/dtype；
- 初始化 distributed group；
- load/quantize weights；
- 分配 KV pool 与 workspace；
- 选择 attention/sampling backend；
- memory profiling 计算可用 KV token；
- warmup kernels；
- capture CUDA graphs。

`--mem-fraction-static` 影响静态池预算。设置过高会让运行时临时 workspace/graph capture OOM，过低则并发 token 容量不足。

## 5. CUDA Graph 的收益与约束

decode 每轮 kernel 多而小，Python/CPU launch 开销显著。CUDA Graph 把固定执行 DAG capture 后 replay，减少 launch overhead。

约束：

- tensor 地址稳定；
- 控制流和大部分 shape 固定；
- capture 时不能做不支持的动态分配/同步；
- replay 输入通过预分配 buffer 更新。

在线 batch size 动态，常用预定义 capture sizes。例如真实 batch=5，选择 graph batch=8，补 dummy/padding slots，再用 metadata mask 掉。

权衡：

- capture sizes 密：内存和启动时间高、padding 少；
- capture sizes 稀：内存低、padding/额外计算多；
- 超过最大 capture size：eager fallback。

## 6. piecewise/breakable graph

模型中某些部分动态（MoE routing、structured/spec 状态），完整 graph 难 capture。piecewise graph 只捕获稳定片段；breakable graph 在条件处切开。读 trace 时不要以为“启用 CUDA Graph”意味着整个 forward 都无 CPU launch。

## 7. logits 与 sampling

最后 hidden state 经 LM head 得到 logits。为节省带宽，通常只计算需要采样位置的 logits；请求若要 prompt logprob 则需更多位置。

sampler 必须处理每请求不同的：

- temperature/top-k/top-p/min-p；
- repetition/frequency/presence penalty；
- allowed token mask/grammar；
- RNG state；
- greedy；
- NaN/Inf fallback。

sampling metadata 的动态性会影响 CUDA graph 与 overlap。DP attention 场景还要明确 logits/sampling 在哪个 rank 执行、结果如何广播。

## 8. kernel 性能判断

不要凭“用了 FlashAttention”断定 attention 已最优。至少看：

- prefill/decode 分开；
- head_dim、GQA ratio、page size；
- batch 与实际 token 数；
- KV dtype/layout；
- graph/eager；
- kernel occupancy、HBM bandwidth；
- metadata 准备和 D2H/H2D 同步；
- TP collective 是否暴露。

一次请求慢可能根本不在模型 forward，而在排队、tokenizer、CPU scheduler 或 detokenizer。

## 9. `ForwardBatch` 源码精读

`forward_batch_info.py:372` 把字段按 ownership 分组：

```python
forward_mode: ForwardMode
batch_size: int
input_ids: torch.Tensor
req_pool_indices: torch.Tensor
seq_lens: torch.Tensor
out_cache_loc: torch.Tensor
seq_lens_sum: int
```

关键映射：

```text
req_pool_indices[b]
  → ReqToTokenPool 第几行
seq_lens[b]
  → 该请求有效历史长度
out_cache_loc
  → 本轮新 token 的 physical KV 写入位置
```

源码注释指出这些 tensor 当前从 `ScheduleBatch` **borrowed by reference**，跨 stream 时存在 alias 风险；未来希望在边界 clone 成 ForwardBatch-owned snapshot。这正是 overlap WAR barrier 的数据来源之一。

`ForwardBatch` 还带：

```python
is_extend_in_batch
global_forward_mode
can_run_dp_cuda_graph
tbo_split_seq_index
sampling_info
spec_info
```

说明“执行 batch”不仅是模型输入，也携带分布式 ranks 必须一致的控制决策。

## 10. `ModelRunner.__init__` 顺序

`model_runner.py:241` 先保存：

```python
self.mem_fraction_static = mem_fraction_static
self.page_size = server_args.page_size
self.req_to_token_pool = req_to_token_pool
self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
self.use_mla_backend = model_config.attention_arch == AttentionArch.MLA
self.spec_algorithm = SpeculativeAlgorithm.from_string(...)
```

随后在设置 device 后才初始化 distributed/backend。源码特别把 Mooncake TransferEngine 放在 torch distributed 前，以便作为 PG backend 共享。初始化顺序改变可能不是风格问题，而会影响通信 backend 的资源复用。

### 静态显存预算

逻辑近似：

```text
total GPU memory
- model weights
- non-KV runtime peak/workspace
- CUDA graph capture pool
× mem_fraction policy
= KV pool
```

系统通常通过一次 profile forward 估计非 KV 峰值，再把剩余换算成可缓存 token 数。profile shape 若不能代表最坏路径（VLM、spec verify、MoE），运行时仍可 OOM。

## 11. attention metadata 例子

两个 decode 请求：

```text
seq_lens=[3,5]
req_pool_indices=[7,9]
req_to_token[7,:3]=[20,4,11]
req_to_token[9,:5]=[2,8,6,30,31]
out_cache_loc=[12,32]
```

attention backend 根据 block/page table 读取各自历史 KV，并把新 K/V 写到 12、32。逻辑序列连续不要求 physical indices 连续。

若把 `seq_lens` 写成 `[4,6]` 但新 slot 尚未写完，attention 会读取未初始化/别的请求 KV。

## 12. CUDA Graph buffer 安全

capture size=8、真实 batch=5：

```text
slots 0..4: real req indices
slots 5..7: req_pool_idx=0 dummy
```

这解释了 `ReqToTokenPool` 第 0 padding row。除了 req index，seq len、out cache loc、sampling mask 都需为 dummy 给安全值。

replay 前通常把动态输入 copy 到固定地址。若 CPU/GPU overlap 在 copy 完成前修改 source，或 graph 仍读 buffer 时下一轮覆盖，就产生跨请求错误；需要 stream event 而不是 Python 执行顺序。

## 13. backend 选择不能只看 GPU 型号

当前官方指南的选择还依赖：

- MHA/MLA/稀疏/线性 attention；
- prefill 与 decode phase；
- page size 原生支持；
- speculative top-k；
- KV dtype；
- CUDA Graph；
- Hopper/Blackwell/AMD/NPU。

同一模型可用 hybrid attention：prefill 用大吞吐 backend，decode 用低延迟 backend。验证必须分别跑 prefill、decode、target verify。

## 14. logits 计算范围

普通 generation prefill 只需最后有效 token 的 logits；prompt logprob 要计算更多位置。若 `return_logprob` 大量开启：

- LM head tokens 增多；
- vocab-parallel gather/top-k 增多；
- GPU→CPU metadata 增多；
- streaming payload 增大。

因此 logprob API 不应与普通 generation 用同一容量假设。

## 15. 本章延伸阅读

- [Attention Backend 官方指南](https://docs.sglang.io/docs/advanced_features/attention_backend)：当前硬件、page、spec 与 backend 矩阵。
- [SGLang v0.3 博客](https://www.lmsys.org/blog/2024-09-04-sglang-v0-3/)：MLA、torch.compile 与小 batch 优化案例。
- [FlashAttention](https://arxiv.org/abs/2205.14135)：理解 IO-aware attention。
- [CUDA Graph 官方文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)：capture/replay、固定地址与同步语义。
- [FlashInfer 论文](https://arxiv.org/abs/2501.01005)：serving attention、sampling 和调度友好的 kernel 设计。
