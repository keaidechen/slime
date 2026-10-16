# 2.3 ModelRunner、Attention Backend 与 CUDA Graph

相关基础：[Transformer 与 KV](<../../../../05_Inference/19_生成执行与采样.md#s1>)。


调度器得到的还是“请求 A 计算一个 token、B 计算两个 token”的逻辑计划。GPU kernel 需要的是数组地址、形状、长度和读写位置；它不会查 Python 请求对象或 HTTP 连接。本章解释这次转换，以及为什么同一组张量还需要 Attention metadata 和执行图。

Backend（后端）是实现同一执行契约的具体算子方案；eager 指每次沿程序调用逐个提交算子；CUDA Graph 指先捕获一组设备操作及依赖，再重放这张图。二者应实现相同逻辑输入到输出的映射，差异主要在提交方式和适用形状。

## 1. 调用链

```text
Scheduler.run_batch
  → TpModelWorker.forward_batch_generation
  → ForwardBatch.init_new(...)
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

例如 A 本轮算 1 个 token、B 算 2 个，`input_ids` 是 `[3]`，而长度、请求索引是 `[2]`。若每个 token 有 2 个 Query 头、1 个 KV 头、每头维度 2，则本轮 `Q:[3,2,2]`、`K/V:[3,1,2]`；Attention 输出为 `[3,2,2]`，展平为 `[3,4]` 再进入输出投影。GQA 中两个 Query 头都访问同一个 KV 头；并不是把 K/V 多复制一份才算“形状匹配”。

`positions` 属于逻辑序列；`out_cache_loc` 属于物理池。把 token 搬到另一个 slot 不改变它的位置编码，反之位置增加也不意味着物理地址连续。后台索引结构正是把这两套编号连接起来。

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

### 用启动成本推导何时值得捕获

DAG 是 Directed Acyclic Graph（有向无环图）：节点是设备操作，有向边表示“前者完成后后者才能执行”。设一次 eager forward 有 K 个小 kernel，每次提交造成的有效开销为 `t_l`，设备工作总耗时为 `T_gpu(B)`，B 是真实 batch 大小。无重叠的简化模型为：

```text
T_eager(B) ≈ K × t_l + T_gpu(B)
T_graph(B) ≈ t_replay + t_copy + T_gpu(B_cap)
```

所有 T/t 的单位是微秒或毫秒，必须统一；`B_cap≥B` 是选中的捕获尺寸，`t_copy` 是把新输入放到固定 buffer 的开销。假设 K=20、`t_l=3 μs`、`T_gpu(5)=100 μs`，eager 约 160 μs。若 batch=5 用捕获尺寸 8，`T_gpu(8)=125 μs`、重放 5 μs、拷贝 4 μs，总计 134 μs；节省 26 μs，而非把模型计算省掉。若 padding 后计算升到 180 μs，则图路径反而更慢。

捕获本身还有一次性成本 `T_capture`。若每次净省 `ΔT>0`，至少重放约 `T_capture/ΔT` 次才摊回；capture 的显存也要常驻。现实中提交与计算可重叠，不能把 trace 上所有 launch 时间直接相加预测收益，要以关键路径验证这个模型。

## 6. piecewise/breakable graph

模型中某些部分动态（MoE routing、structured/spec 状态），完整 graph 难 capture。piecewise graph 只捕获稳定片段；breakable graph 在条件处切开。读 trace 时不要以为“启用 CUDA Graph”意味着整个 forward 都无 CPU launch。

## 7. logits 与 sampling

最后 hidden state 经 LM head 得到 logits。为节省带宽，通常只计算需要采样位置的 logits；请求若要 prompt logprob 则需更多位置。

sampler 必须处理每请求不同的：

RNG（Random Number Generator，随机数生成器）为概率采样提供随机数；请求重排后，随机状态仍应与正确的请求对应。

- temperature/top-k/top-p/min-p；
- repetition/frequency/presence penalty；
- allowed token mask/grammar；
- RNG state；
- greedy；
- NaN/Inf fallback。

sampling metadata 的动态性会影响 CUDA graph 与 overlap。DP attention 场景还要明确 logits/sampling 在哪个 rank 执行、结果如何广播。

## 8. kernel 性能判断

不要凭“用了 FlashAttention”断定 attention 已最优。至少看：

occupancy（驻留率）是一个 SM 上驻留 warp 数与硬件支持的最大驻留 warp 数之比。寄存器、共享内存和线程配置都会限制它；驻留线程可能在等待，所以它不等于有效计算利用率。

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

先把所有量统一为字节。设设备可用预算为 M、权重 W、运行时非 KV 峰值 A、图与通信等额外保留 G、安全余量 H，则规划层面的上界是 `M_KV≤M-W-A-G-H`。这是一份容量账本，不是 `mem_fraction_static` 的逐字源码公式；实际预算还取决于模型布局、预留策略以及是否在 graph capture 后调整 KV pool。

例如 M=16 GiB，W=8、A=2、G=1、H=1 GiB，剩下 4 GiB 给 KV。若每 token 128 KiB，则上界为 `4×2^30/(128×2^10)=32768` 个物理 token 位置，再扣除页取整和其他元数据。本例没有说“可同时接 32768 条请求”：每条请求占用许多位置。

本快照 `model_runner_components/kv_pool_runtime.py:compute_post_capture_kv_resize()` 的一种路径会先同步设备，再读取剩余显存、扣除 headroom，并计入已有 KV backing 后重新配置池。其 headroom 包含 `pre_model_load_memory×(1-mem_fraction_static)`；因此不能先随意减完所有项再统一乘一次比例。应观察启动日志中最终的 token 容量。若预算未覆盖 VLM（视觉语言模型）、spec verify 或 MoE（Mixture of Experts，混合专家模型）的峰值，运行时仍可 OOM（Out Of Memory，内存不足）。

## 11. attention metadata 例子

两个请求进入本轮前各已有 3/5 个历史 KV；本轮各输入一个新 token。令 page size=1，分配位置 12/32，准备完成后的 metadata 应是：

```text
历史长度（本轮前）=[3,5]
seq_lens（本轮可见总长）=[4,6]
positions=[3,5]
req_pool_indices=[7,9]
req_to_token[7,:4]=[20,4,11,12]
req_to_token[9,:6]=[2,8,6,30,31,32]
out_cache_loc=[12,32]
```

`input_ids`、`positions`、`out_cache_loc` 均为 `[2]`。每层先计算本轮 K/V，并由 backend 的写入与 Attention 依赖保证新值在被读取前有效；有的实现将相关操作融合。`seq_lens=[4,6]` 在准备阶段是正确的，它描述本轮执行范围，不能误写成只含旧 KV 的 `[3,5]`。

错误发生在没有依赖保证就读取 12/32，或长度增长却没有更新映射，而不是“CPU 提前写了本轮长度”本身。逻辑位置与物理 slot 分离让碎片和共享变得可管理，也把正确性责任转移到了 metadata 与执行顺序。

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

设隐藏维度 d=4、词表大小 V=6，两条请求只需要各自末位置的分布，则 `H_last:[2,4]` 乘 `W_head:[4,6]` 得 `logits:[2,6]`，12 个数。若输入分别有 3/2 个位置，并为全部位置计算 logits，中间结果会是 `[5,6]`，30 个数；实际 logprob 还要按“位置 i 预测位置 i+1”的位移取值。长 prompt、大词表时差距很大，即使最终只返回每个位置一个概率，也可能增加 LM head 的工作。

## 15. 本章延伸阅读

- [Attention Backend 官方指南](https://docs.sglang.io/docs/advanced_features/attention_backend)：当前硬件、page、spec 与 backend 矩阵。
- [SGLang v0.3 博客](https://www.lmsys.org/blog/2024-09-04-sglang-v0-3/)：MLA、torch.compile 与小 batch 优化案例。
- [FlashAttention](https://arxiv.org/abs/2205.14135)：理解 IO-aware attention。
- [CUDA Graph 官方文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)：capture/replay、固定地址与同步语义。
- [FlashInfer 论文](https://arxiv.org/abs/2501.01005)：serving attention、sampling 和调度友好的 kernel 设计。
