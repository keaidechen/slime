# 3.2 部署拓扑、多节点与路由

本章讨论实例如何组成服务，不重复 3.1 中单次 forward 的 TP/DP/EP 通信细节。

## 1. 从 workload 和 SLO 开始

部署前固定模型 revision、精度、最大上下文、输入/输出长度分布、并发、TTFT/ITL SLO 和目标任务。显存要分两本账：

```text
静态显存 = 权重 + runtime workspace + CUDA Graph + 通信 buffer
动态显存 = KV cache + request metadata + multimodal intermediate
```

峰值 tokens/s 不能代表线上容量。容量点必须同时满足显存安全边界、延迟分位数和错误率。

## 2. 从单实例到服务拓扑

| 层级 | 主要选择 | 失败边界 |
|---|---|---|
| Rank | TP/EP/PP group | collective 或单 rank 失败 |
| Instance | 单个模型服务副本 | 整组 rank 一致启停 |
| Replica pool | DP 副本 | 路由、版本和缓存亲和性 |
| Disaggregated pool | prefill/decode/encoder 等资源池 | 中间状态传输和所有权 |
| Gateway | 协议、发现、限流、重试 | 客户端可见的服务边界 |

先在单节点验证相同并行逻辑，再扩到多节点；否则网络、环境和模型正确性问题会叠在一起。

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

## 6. 延伸阅读

- `sglang/docs_new/docs/references/multi_node_deployment/`
- `sglang/docs_new/docs/advanced_features/sgl_model_gateway.mdx`
- `sglang/docs_new/docs/advanced_features/pd_disaggregation.mdx`
- `sglang/docs_new/docs/advanced_features/llm-d.mdx`

