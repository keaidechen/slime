# MoE、MLA、MTP：高级架构如何进入训练系统

## 1. MoE 是一条分布式数据流

```text
hidden states -> router logits -> top-k/score
 -> permutation/dispatch -> local experts
 -> combine/unpermute -> output
```

EP 把 experts 分到不同 rank，token dispatcher 通过 all-to-all 或其他 backend 重排 token。性能由 expert GEMM 大小、token 不均衡、capacity/drop 策略、通信 overlap 与网络共同决定。Router 的 load-balancing loss、z-loss、dtype 和 replay/recompute 还会影响数值行为。当前实现通过 `MoEAuxLossAutoScaler` 把 auxiliary loss 自动接入 autograd；任务 loss 不应再手工重复相加，详见[源码问题详解第 15 节](../06_reference/03_source_questions.md)。

关键入口：`megatron/core/transformer/moe/` 下的 `router.py`、`token_dispatcher.py`、`experts.py` 与 `moe_layer.py`。

## 2. Router replay

Router replay 记录 top-k expert indices 并在后续 forward/recompute 使用，目的是固定离散 expert 选择。replay 时 probability 仍从当前 `scores` 对记录的 indices 做 gather，因此 router score 仍可有梯度；它既不冻结 router，也不保存完整 logits。需要区分 record、forward replay、backward replay，并保证记录与 microbatch/layer 对应。详见[源码问题详解第 16 节](../06_reference/03_source_questions.md)。

## 3. MLA

Multi-Latent Attention 用低维 latent 表示改变 Q/K/V 路径，目标之一是降低 KV 表示成本。启用时不仅设置开关，还需要 MLA 专用 config，并核对 RoPE 维度、Q/kv latent 维度、head 划分和 checkpoint mapping。代码从 `core/transformer/multi_latent_attention.py` 与相关 config 开始。

## 4. MTP

Multi-Token Prediction 在每个位置增加未来多个 token 的预测深度。MTP module 通常含共享 embedding、projection、Transformer block 与共享 output head；额外 loss 按 scaling factor 合入主目标。

PP 下 MTP 默认靠近最后 stage，也可用自定义 layout 中的 `m` 指定，但所有 MTP 层需满足集中放置等约束。当前支持边界（例如与 CP、mask、position embedding 的组合）必须查当前版本，不应依据旧博客推断。

## 5. 组合验证

对 MoE/MLA/MTP 分别验证 dense/reference 或单 rank 等价、TP/EP shard shape、PP placement、checkpoint round-trip、recompute 前后一致性，再测性能。高级结构最常见错误是“forward 能跑，但 pipeline loss、共享参数同步或恢复边界错了”。
