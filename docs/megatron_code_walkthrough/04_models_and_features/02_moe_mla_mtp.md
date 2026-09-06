# MoE、MLA、MTP：高级架构如何进入训练系统

<!-- learning-position -->
> **学习定位**：A8 · 专项。
> **前置**：[GPU、tensor 与通信基础](<../../../learn_docs/00_Foundations/README.md>)。
> **首读/二读**：MoE/MLA/MTP 的具体模块与 router replay，承接模型基础。
> **进度与实验**：[学习清单](<../../../learn_docs/学习清单.md>) · [总入口](<../../../learn_docs/README.md>)。
<!-- /learning-position -->

<a id="beginner-example"></a>

## 入门例子：把新架构翻译为状态、算子和通信

MoE 增加 router、token dispatch/combine 与不规则 expert GEMM；MLA 改变 attention 的投影和缓存表达；MTP 增加额外预测计算及其训练/推理使用方式。理解模型名之后，必须分别追这三类工程变化。

例如 MoE 的 router replay 可以固定某些路由决策用于对齐，但不等于消除所有数值或采样差异。MLA 的缓存优势也取决于实际 backend 保存了什么，而不能仅从论文参数量推断进程显存。

练习：选一种特性，列新增 tensor 的 shape、dtype、owner、lifetime 与通信；先完成单层正确性，再测整步性能。一次只接入一种新结构，保留 Dense 对照。

## 机制与实现

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
