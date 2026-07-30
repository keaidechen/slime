# 05. 分布式执行与 Prefill-Decode 解耦

## 1. 推理并行维度

| 维度 | 切什么 | 主要通信 | 使用场景 |
|---|---|---|---|
| TP | attention heads / hidden / weights | all-reduce/AG/RS | 单模型跨 GPU |
| PP | layers | stage P2P | 模型放不下、特定部署 |
| DP | 请求/batch | 调度协调，模型副本独立 | 扩请求吞吐 |
| DP attention | attention 按 DP，本体/MoE 特殊布局 | gather/scatter + MoE 通信 | 大 MoE |
| EP | experts | token all-to-all | MoE |

在线推理的 DP 与训练 DP 不同：通常不做 gradient all-reduce，而是把不同请求路由到模型副本。cache-aware router 很重要，因为请求迁移会丢失 prefix 命中。

## 2. Tensor Parallel

TP 与 Megatron 类似切 Linear/attention。decode 每轮工作小，collective latency 比训练更敏感，因此：

- 优先让 TP group 落在高速互联域；
- TP 太大可能让本地 GEMM 太小；
- batch 增大可摊薄通信，但会提高 ITL；
- custom all-reduce 只在拓扑、消息大小和实现匹配时占优。

multi-node TP 是最后手段之一；先评估 PP/DP/EP 或量化能否避免。

## 3. DP attention 与 EP

大 MoE 模型中 attention 参数/计算较小但 dense，experts 巨大且稀疏。系统可让 attention 在 DP ranks 处理不同 token，再将 token dispatch 到 EP experts。关键问题：

- 各 DP rank 本轮 token 数不同；
- collective 参与顺序必须一致；
- 空 batch rank 也可能必须参与；
- prefill 与 decode 混合会导致极端负载差异；
- DeepEP normal/low-latency 模式适合不同 phase。

源码中 scheduler 对 DP attention 的许多判断必须做全局同步。若某 rank 本地认为是 extend、另一个认为 decode 并走不同 collective，会直接 deadlock。

## 4. 为什么做 PD disaggregation

prefill：

- 一次处理长 prompt；
- 大 GEMM，compute-heavy；
- 决定 TTFT；
- 产生大量 KV。

decode：

- 每步少量 token；
- memory/latency-heavy；
- 决定 ITL/TPOT；
- 需要持有长期 KV。

统一调度时，长 prefill 会打断 decode，制造 ITL spike；DP attention 中不同 ranks phase 不一致还会失衡。PD 解耦使用独立 prefill 与 decode 实例，各自优化 batch、并行和 kernel。

## 5. 请求与 KV 迁移时序

简化时序：

```text
router → decode instance：预留 request/KV destination slots
decode → prefill：bootstrap metadata（目标地址、传输句柄）
router → prefill：执行 prompt
prefill：产生 KV
prefill ==RDMA/NIXL/Mooncake==> decode KV pool
decode：确认 KV ready，开始生成
后续 token 从 decode 流回 client
```

先在 decode 分配目标 slot 是为了让 prefill 知道 KV 写到哪里。控制 metadata 与数据传输必须用 request id/version 关联，防止超时后旧 transfer 写入已复用 slot。

## 6. ownership 与失败清理

必须定义每阶段谁拥有：

- request record；
- source KV；
- destination KV reservation；
- radix lock；
- transfer handle；
- abort/finish notification。

失败场景：

1. decode 分配后 prefill 不可达；
2. prefill 完成但 transfer timeout；
3. transfer 完成，decode 在 ack 前崩溃；
4. client abort 正在传输；
5. router 重试造成重复 request；
6. prefill/decode 版本或 KV layout 不一致。

正确实现需要幂等 cleanup、generation/version token、heartbeat 和超时。仅“捕获 exception”不够，因为 GPU/远端内存资源可能仍被占用。

## 7. 传输何时值得

PD 收益必须覆盖：

- 路由与额外排队；
- destination allocation；
- KV 网络传输；
- bootstrap/ack latency；
- cache locality 变化。

粗略 KV transfer bytes 与 token 数成正比：

```text
bytes ≈ prompt_tokens × layers × 2(K,V)
        × local_kv_heads × head_dim × kv_dtype_bytes
```

长 prompt 的 KV 可达 GB 级。需要 RDMA/高速网络并与 prefill 计算 overlap。短 prompt 或低负载下统一部署可能更简单更快。

## 8. 容量规划

prefill pool 按 arrival rate、prompt token/s、TTFT SLO 规划；decode pool按活跃序列、output token/s、ITL SLO 和 KV 容量规划。两边扩缩容信号不同，不能只用 GPU utilization。

监控至少包含：

- prefill/decode queue time；
- KV transfer bytes/latency/timeout；
- destination reservation；
- bootstrap failure；
- decode active tokens 与 KV occupancy；
- phase 各自 p99；
- router retry/affinity。

官方当前文档列出的传输 backend 和参数会快速变化，部署前以仓库 `docs_new/docs/advanced_features/pd_disaggregation.mdx` 与当前版本为准。

## 9. TP rank 上一次 decode 的通信账

本章建议同步打开这些源码：

- `sglang/python/sglang/srt/distributed/parallel_state.py`：TP/PP/DP 等 process group；
- `sglang/python/sglang/srt/managers/scheduler_components/dp_attn.py`：DP attention 调度适配；
- `sglang/python/sglang/srt/managers/scheduler_pp_mixin.py`：PP scheduler 与 P2P；
- `sglang/python/sglang/srt/layers/moe/token_dispatcher/base.py`：MoE dispatcher 接口；
- `sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py`：DeepEP dispatch/combine；
- `sglang/python/sglang/srt/layers/moe/moe_runner/runner.py`：grouped GEMM runner；
- `sglang/python/sglang/srt/disaggregation/prefill.py` 与 `decode.py`：PD 两侧状态机。

Llama 类 TP=4，一层近似：

```text
QKV Column Parallel：各 rank 计算本地 heads
attention：读取本地 KV shard
output Row Parallel：partial output → all-reduce/RS
MLP gate/up Column Parallel：本地 FFN shard
down Row Parallel：partial output → all-reduce/RS
```

每 token、每层可能有两次 TP reduction。模型 80 层就有大量小 collective，所以 decode 对 NCCL latency 极敏感。跨节点 TP 即使总带宽够，也可能被每层往返延迟限制。

评估 TP 需抓：

```text
collective message bytes
collective duration
前后 GEMM duration
是否 custom all-reduce
是否跨 NVLink/IB
```

## 10. DP attention 的一致决策

假设 DP-attention rank0 有长 prefill，rank1 只有 decode。MoE/TP 后续可能要求两 rank 共同 collective。SGLang 需形成 `global_forward_mode` 或同步 `is_extend_in_batch`：

```text
rank0 local: EXTEND
rank1 local: DECODE
global decision: MIXED/EXTEND-aware
```

两 rank必须进入同一代码路径和 collective 序列，即使其中一个本地 token 数为 0。空 tensor 也要用协议允许的 shape 参与，不能直接 `if no_tokens: return`。

## 11. EP dispatch 的数据结构

对每个 local token，router 产生：

```text
topk expert ids
topk weights
destination EP ranks
send splits
```

all-to-all 后接收端按 local expert 分组做 grouped GEMM，再按 inverse mapping 返回。prefill token 多，DeepEP normal 模式追求吞吐；decode token 少，low-latency 模式更重视固定 buffer/CUDA graph 与低 launch latency。

当前官方 EP 文档把 MoE forward 拆成：

```text
TopK
 → Dispatcher.dispatch
 → pre-permute
 → grouped GEMM runner
 → post-permute
 → Dispatcher.combine
```

新增 backend 应只替换相应接口，而不是把通信特例写入模型层。

## 12. PD transfer 的 generation 防护

建议 transfer descriptor 至少含：

```text
request_id
request_generation
model/cache_version
source rank / destination rank
layer range
source/destination KV indices
token range
dtype/layout/page_size
checksum or completion token
```

decode slot 12 被请求 A 预留后，A timeout 释放；请求 B 复用 slot 12。A 的迟到 RDMA 若无 generation/version 检查，会覆盖 B，而且传输本身可能“成功”。这是必须在协议层防止的 silent corruption。

## 13. PD 容量计算例

到达率 10 req/s，平均 prompt 4000 tokens，平均 output 500 tokens：

```text
prefill offered load = 40k input tok/s
decode offered load  = 5k output tok/s
```

如果单 prefill replica 在目标 TTFT 下可处理 15k tok/s，至少需 3，再留尾延迟余量。decode replica 若在目标 ITL 下 1.5k tok/s，至少需 4。

同时算 Little's Law：

```text
平均活跃 decode 请求
≈ arrival_rate × 平均 decode duration
```

它决定长期 KV 容量，不能只按 output tok/s 规划。

## 14. 故障注入矩阵

| 注入 | 应观察到 |
|---|---|
| prefill 在 transfer 前崩溃 | decode reservation 超时释放 |
| transfer 中 client abort | source/destination 都幂等取消 |
| decode heartbeat 丢失 | router 停止新流量，prefill 清理 |
| 重复 bootstrap | 同 generation 幂等，不重复分配 |
| layout/version 不同 | transfer 前拒绝，不允许写入 |
| 一条 TP rank transfer 慢 | 整请求超时原因指出具体 rank |

## 15. 本章延伸阅读

- [PD Disaggregation 官方文档](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)：当前启动方式、Mooncake/NIXL 等 backend。
- [SGLang 大规模 Expert Parallelism 博客](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)：DeepSeek 部署、DP attention、DeepEP 与负载均衡。
- [Expert Parallelism 官方指南](https://docs.sglang.io/docs/advanced_features/expert_parallelism)：dispatcher/runner backend、TBO/SBO 与 EPLB。
- [DistServe 论文](https://arxiv.org/abs/2401.09670)：prefill/decode 分离的 goodput 与资源配置背景。
- [SGLang Pipeline Parallel 博客](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/)：长上下文 chunk、async P2P 与多 stream。
