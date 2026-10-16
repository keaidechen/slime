# 3.2 部署拓扑、多节点与路由

相关基础：[Transformer 与 KV](<../../../../05_Inference/19_生成执行与采样.md#s1>)。


## 入门例子：路由是一种状态取舍

两个副本都能生成相同模型的答案，但它们的 prefix cache 不一定相同。把后续轮次送回原副本有机会复用 KV；若该副本排队严重，坚持亲和也可能增加延迟。路由需要同时考虑工作量和缓存收益。

把这个取舍变成一个小账本。请求长 6 个 token，实例 A 已缓存前 4 个但预计排队 8 ms，实例 B 无命中且不排队。暂设每个未命中 token 的 prefill 平均 1 ms、其他项相同，则 A 的首 token 前半程约 `8+(6-4)×1=10 ms`，B 约 `0+6×1=6 ms`。A 的命中率更高，却不是此刻延迟更低的选择。

概念性路由代价可写为 `Q_j+C_j(L-H_j)+X_j`：j 是候选实例，Q 是预计等待 ms，C 是该实例对未命中后缀的执行时间函数，L/H 是总输入/可复用 token 数，X 是额外传输或协调 ms。这里 C 通常非线性，还依赖历史长度和 batch；它是决策模型，不是声称 Router 内置了这个固定评分式。估计过时、热点集中或新副本冷缓存都会使决策偏离预期。

记录请求 id/session、目标实例、输入输出长度、排队、cache hit 与完成状态。实例 ready 表示能接工作，不能仅用进程存活代替；扩容后新实例的加载/预热与缓存冷启动要单独观察。

TTFT（Time To First Token，首 token 延迟）从约定的请求起点计到首个输出 token，通常包含排队、输入处理和 Prefill。客户端与服务端的起点可能不同，报告时应明确。

相同总卡数和输入分布下，单实例、多副本与缓存亲和策略改变排队与前缀复用的不同部分。TTFT 改善可能来自更短队列，也可能来自较少 Prefill 工作；这两种收益需要分别解释。

## 机制与实现

本章讨论实例如何组成服务，不重复 3.1 中单次 forward 的 TP/DP/EP 通信细节。

## 1. 从 workload 和 SLO 开始

SLO（Service Level Objective，服务等级目标）是服务承诺的可量化目标，例如“至少 99% 的请求首 token 延迟小于 1 秒”；阈值、统计窗口、失败请求和分母都属于定义。

部署前固定模型 revision、精度、最大上下文、输入/输出长度分布、并发、TTFT/ITL SLO 和目标任务。显存要分两本账：

```text
启动/常驻预算：权重、预分配 KV pool、graph pool、部分通信/workspace
随工作负载变化的峰值：活跃 KV 占用、临时激活、多模态中间结果等
```

峰值 tokens/s 不能代表线上容量。容量点必须同时满足显存安全边界、延迟分位数和错误率。

“逻辑占用动态”与“底层显存动态申请”是不同维度。KV 活跃 token 数随请求变化，但池本身可以在启动时就分配好；workspace 也可能按形状临时扩张。做设备容量账时应避免把预分配池和其中活跃 KV 再加一次。

## 2. 从单实例到服务拓扑

| 层级 | 主要选择 | 失败边界 |
|---|---|---|
| Rank | TP/EP/PP group | collective 或单 rank 失败 |
| Instance | 单个模型服务副本 | 整组 rank 一致启停 |
| Replica pool | DP 副本 | 路由、版本和缓存亲和性 |
| Disaggregated pool | prefill/decode/encoder 等资源池 | 中间状态传输和所有权 |
| Gateway | 协议、发现、限流、重试 | 客户端可见的服务边界 |

先在单节点验证相同并行逻辑，再扩到多节点；否则网络、环境和模型正确性问题会叠在一起。

例如有 4 张 GPU：TP=4 的一个副本让四卡共同执行一批请求；两个 TP=2 副本则可各自维护队列与缓存；四个单卡副本要求每张卡都能容纳完整模型及状态。前者减少单卡权重容量要求，后者增加独立请求处理能力，却复制更多权重并分散缓存。不能把 TP rank 数当作可独立路由的 replica 数。

## 3. 多节点启动不变量

- 所有 rank 使用相同的权重、tokenizer、配置、代码与 kernel 版本；
- rank/world size 与 TP、DP、EP、PP 的维度一致；
- 控制面和数据面端口、网卡与防火墙明确；
- NCCL/RCCL/HCCL 等 backend 的网络选择一致；
- 任一 rank 初始化失败时，整个实例不能 ready；
- worker 丢失后停止接流量，并清理请求、KV 和传输状态；
- request id、trace id 和时间戳可以跨节点关联。

Kubernetes Deployment、LWS、RBG 等是实现手段，不会改变这些不变量。

## 4. 网关与路由

纯 round-robin 简单，但会破坏 prefix cache 亲和性；纯 cache-aware routing 又可能制造热点。路由决策至少考虑：

- 实例健康、排队长度和 KV 压力；
- prompt 前缀亲和性；
- 模型、LoRA 和 policy 版本；
- context 长度及预计输出长度；
- 租户优先级、配额和 admission control；
- PD 拓扑中的 prefill/decode 配对。

生成请求的重试不是天然幂等。客户端可能已经收到部分 token，重新采样还可能产生另一条序列。网关必须区分连接重试、请求重放和服务端 failover。

幂等表示重复执行与执行一次具有相同的可观察效果。一个反例：第一次请求已向客户端输出 `A B` 后连接中断，重发相同 prompt 得到 `A C D`；把第二次结果直接接上会成为 `A B A C D`。仅复用请求 id 不会自动提供恢复能力。路由层需要记录是否已输出、是否支持按已提交 token 恢复、重复请求如何去重；无恢复协议时应明确失败边界。

## 5. Readiness 与发布

“进程活着”“模型加载完成”“完成必要预热”“能够接收新请求”是四种状态。readiness 只能在模型、通信组和必要 graph capture 完成后成功。

```text
固定 artifacts
 -> 离线准确率
 -> 单实例功能/压力测试
 -> canary
 -> 分批扩容
 -> 观察 SLO、错误、缓存和输出漂移
 -> 完成或回滚
```

回滚要同时恢复模型版本、路由、LoRA/cache 状态和监控基线，而不只是重启进程。

发布中的 drain（排空）是先从新请求路由集合移除实例，再等待旧请求自然结束；它和杀进程不同。即使新实例已 ready，也不能据此立即回收旧实例：旧连接、KV 和可能的 PD 传输还由旧实例持有。Canary（灰度实例）先承接少量代表性流量，用于比较新版本延迟与质量；缓存冷热要单独控制，否则冷启动惩罚会混进版本差异。

## 6. 延伸阅读

- `sglang/docs_new/docs/references/multi_node_deployment/`
- `sglang/docs_new/docs/advanced_features/sgl_model_gateway.mdx`
- `sglang/docs_new/docs/advanced_features/pd_disaggregation.mdx`
- `sglang/docs_new/docs/advanced_features/llm-d.mdx`
