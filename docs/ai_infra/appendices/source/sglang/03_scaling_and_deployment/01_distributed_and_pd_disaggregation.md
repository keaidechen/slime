# 3.1 分布式执行与 Prefill-Decode 解耦

相关基础：[Transformer 与 KV](<../../../../05_Inference/19_生成执行与采样.md#s1>)。


## 1. 推理并行维度

单卡遇到两种不同困难：模型及活跃状态放不下，或者虽然放得下却不能满足请求量。前者需要把同一模型执行拆开，后者可以复制完整服务。这两种方案对应不同通信与故障边界，不能仅用“用了几张卡”比较。

Rank 是通信组内的进程编号，world size 是组成员数。TP（Tensor Parallelism，张量并行）切一层的计算，PP（Pipeline Parallelism，流水线并行）切层，DP（Data Parallelism，数据并行）分请求，EP（Expert Parallelism，专家并行）分专家。表中的 all-reduce 将各 rank 的部分和求和并交给各 rank；AG 是 all-gather（全收集），RS 是 reduce-scatter（归约后分片），P2P 是 point-to-point（点对点传输），all-to-all 则让每个 rank 向其他 rank 交换各自的数据块。

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

ITL（Inter-Token Latency，token 间延迟）是相邻输出 token 的到达时间差；一条回答会有多个 ITL，用于观察流式输出的停顿和抖动。

- 优先让 TP group 落在高速互联域；
- TP 太大可能让本地 GEMM 太小；
- batch 增大可摊薄通信，但会提高 ITL；
- custom all-reduce 只在拓扑、消息大小和实现匹配时占优。

multi-node TP 是最后手段之一；先评估 PP/DP/EP 或量化能否避免。

### 两张卡怎样共同算一个线性层

用行向量约定 `Y=XW`。取 `X=[1,2,3,4]`，形状 `[1,4]`；`W=[[1,0],[0,1],[1,0],[0,1]]`，形状 `[4,2]`。按输入维拆成两份：rank 0 用 `X0=[1,2]` 和 W 的前两行，得到 `Y0=[1,2]`；rank 1 用 `X1=[3,4]` 和后两行，得到 `Y1=[3,4]`。两份输出均为 `[1,2]`，但只是部分和，all-reduce 后才得到完整 `Y=[4,6]`。把两份直接拼成 `[1,2,3,4]` 不是这个层的输出。

如果改为按 W 的输出列切分，各 rank 得到 `[1,1]` 的不同输出片段，此时需要拼接而不是相加；后续层若能直接消费分片，也可以延后收集。权重分片规则决定通信语义，张量形状本身不足以判断。

设一次通信暴露在关键路径上的固定延迟为 α 秒、数据量为 n 字节、有效带宽为 β 字节/秒，粗略代价为 `α+n/β`。增大 TP 会减少每卡计算，却增加通信与同步；即使 n 很小，α 也不会消失。假设每层两次归约、80 层、每次暴露 10 μs，仅固定成本就是 `2×80×10=1600 μs=1.6 ms`。这是解释小 batch decode 对延迟敏感的教学估算，不是 NCCL 或某种网络的实测。

PP 则把层分给不同 stage（阶段），传的是 `[本轮 token 数,隐藏维度]` 激活。假设两阶段各用 3 ms，一个批次依次经过两阶段需 6 ms；三个独立批次理想流水需 `(2+3-1)×3=12 ms`，而逐批串行需 18 ms。末尾空闲区称 bubble（流水线气泡）。同一请求的下一 token 依赖最后阶段的采样，不能像三个独立批次一样任意提前，因此 PP 的吞吐收益不直接等于单请求延迟收益。

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

RDMA（Remote Direct Memory Access，远程直接内存访问）让网络设备对已授权的远端内存执行传输，减少 CPU 中转。内存注册、权限以及远端消费者何时可读取，仍属于协议的一部分。

长 prompt 的 KV 可达 GB 级。需要 RDMA/高速网络并与 prefill 计算 overlap。短 prompt 或低负载下统一部署可能更简单更快。

将公式展开为可核算的时延：设一个待传 shard 含 `T=1024` 个 token、`L=2` 层、`H=1` 个 KV 头、`d=4`、每元素 `s=2` 字节。一个 token 的 K/V 共 `2×2×1×4×2=32` 字节，总计 `32768 字节=32 KiB`。有效网络带宽若为 32 MiB/s，纯传输下界为 `32 KiB/(32 MiB/s)=1/1024 秒≈0.977 ms`；加上 bootstrap（传输双方建立联系和交换描述符）及等待，真实耗时更长。

令统一部署因相互干扰多花的时间为 `T_interference`，拆分新增排队、传输和协调的关键路径耗时为 `T_extra`，只从延迟看需 `T_interference>T_extra` 才值得。能与 prefill 重叠的传输不能再次全额相加，但末层 KV 完成后仍可能有一段传输尾巴。不同 TP/PP 布局还可能要求重分片；上述字节式只适用于相容布局的一个实际传输 shard，不应把单 rank 数据量直接当成全实例网络流量。

## 8. 容量规划

SLO（Service Level Objective，服务等级目标）是服务承诺的可量化目标，例如“至少 99% 的请求首 token 延迟小于 1 秒”；阈值、统计窗口、失败请求和分母都属于定义。

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

用 2 个 token、每 token 选 2 个专家的例子追踪数据。隐藏向量 `H:[2,4]`，expert IDs 为 `[[0,2],[1,2]]`，权重为 `[[0.75,0.25],[0.4,0.6]]`；expert 0/1 在 rank 0，expert 2 在 rank 1。dispatch 产生 4 条 token-expert 工作项，形状 `[4,4]`，两个 rank 各接收两条。专家计算后必须按原 token 和 expert 配对归还，最终第一个 token 的输出是 `0.75×expert0(h0)+0.25×expert2(h0)`，仍为 `[4]`。

如果只按接收顺序拼回，会混淆原 token；inverse mapping（逆映射）保存的就是这个归属。总工作项有 4 条并不代表四个请求，也不代表各专家负载平均：expert 2 接了两条。源/目标通信 buffer 在 dispatch 或 combine 完成前不能覆盖，空闲 rank 也要遵守通信组的 collective 顺序。

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

特别注意：仅在收到完成通知时检查 generation，无法撤销已经发生的远端写。安全协议还必须让传输后端保证旧操作停止，或隔离/撤销旧写入能力，确认不再有在途写后才允许目标内存复用。发送提交、源 buffer 可释放、目标写入完成、decode 可读取这几个完成语义可能不同，应按具体 backend 的事件/确认契约串联；不能把某个名为 `done` 的标记默认解释为全部完成。

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

Little 定律来自“请求在系统里停留的时间面积”：观察足够长的稳定窗口，每条请求贡献自己的驻留秒数，除以窗口秒数就得到平均在途数。因此 `N=λW`，λ 是进入该阶段的请求率（req/s），W 是同一阶段的平均驻留时长（s），N 是平均请求数。边界必须一致，不能用全链路时长乘 decode 到达率来解释纯 running 数。

延续本例，若平均 decode 驻留为 20 s，则约有 `10×20=200` 条活跃请求。假设固定 prompt 4000、输出 500、生成速度均匀且无共享，每条在途请求平均已积累约 250 个输出位置，总逻辑 KV 约 `200×4250=850000` 个位置。每 token 若占 128 KiB，则全池合计约 103.76 GiB。实际输出长短、到达突发、prefix 共享和分片会改变结果；平均量只用于规划，不能替代每个实例的峰值检查。

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
