# 02. 模型、Transformer 组件与 ModuleSpec

## 1. 从 `GPTModel` 看组合关系

主类位于 `megatron/core/models/gpt/gpt_model.py:47`。逻辑结构是：

```text
GPTModel
  ├─ embedding（仅 pre_process stage）
  ├─ rotary position embedding（按配置）
  ├─ decoder: TransformerBlock
  │    └─ N × TransformerLayer
  └─ output_layer（仅 post_process stage）
```

`pre_process/post_process` 不是普通模型开关，而是 PP 切层的边界。非首 stage 不创建 embedding，非末 stage 不创建 output/loss 相关部分。共享 embedding/output weight 时还要跨首尾 stage 同步对应梯度。

forward 时需追踪两个 shape：

- 用户常见 batch-first：`[batch, seq, hidden]`；
- Megatron 内部很多路径使用 sequence-first：`[seq, batch, hidden]`。

packed sequence 又可能压成 `[total_tokens, ...]`。阅读 bug 时先确认 layout，不能只看维数。

## 2. `TransformerBlock` 与 `TransformerLayer`

- `TransformerBlock`：`megatron/core/transformer/transformer_block.py:261`
- `TransformerLayer`：`megatron/core/transformer/transformer_layer.py:302`

Block 负责 layer 列表、pipeline offset、final norm、recompute/offload；Layer 负责一个完整层的 submodule 数据流。典型 pre-norm layer：

```text
hidden
  → input_layernorm → self_attention → bias/dropout/residual
  → pre_mlp_layernorm → MLP or MoE → bias/dropout/residual
```

代码中的 `hidden_states`、`context`、`context_mask`、`packed_seq_params`、`rotary_pos_emb` 是跨模型族的统一接口。扩展模型时，优先维护接口契约，不要在训练主循环里塞模型特例。

## 3. `ModuleSpec` 是什么

Megatron Core 用 `megatron/core/transformer/spec_utils.py` 的 spec 描述“实例化哪个类、传哪些 submodules/params”。它解决：

- 同一逻辑层可选 local PyTorch 或 Transformer Engine 实现；
- attention、MLP、norm、MoE 可局部替换；
- pipeline stage 可从同一配置构建不同 layer；
- 模型结构声明与构造时机解耦。

简化理解：

```python
ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=...,
        self_attention=ModuleSpec(...),
        mlp=ModuleSpec(...),
    ),
)
```

`build_module(spec, ...)` 递归实例化。关键点是 spec 自身不是 module，也没有参数；真正的参数注册发生在 build 后。

### 常见扩展错误

- 把一个已实例化 module 放进 spec，导致 rank 间构造次序或参数共享异常；
- 新 submodule 的 forward 签名不符合 Layer 的调用约定；
- local/TE 两套 spec 只改了一套；
- pipeline offset 变了，但 checkpoint key mapping 没处理；
- 条件构建依赖 global rank，而不是 TP/PP 语义 rank。

## 4. Transformer Engine 的边界

TE 提供优化 Linear、LayerNorm/RMSNorm、attention、FP8 autocast、融合路径等。Megatron Core 负责并行语义和模型组合，TE 负责硬件优化实现。两者的边界可从：

- `megatron/core/extensions/transformer_engine.py`
- `megatron/core/extensions/transformer_engine_spec_provider.py`

观察。

判断性能问题时先问：

1. 实际构建的是 local 还是 TE module？
2. dtype/shape/mask 是否进入期望 fused backend？
3. 当前 GPU/TE 版本是否支持该路径？
4. fallback 是正确性 fallback 还是配置错误？

## 5. attention 的关键细节

`DotProductAttention` 在 `megatron/core/transformer/dot_product_attention.py:26`。TP 通常切 attention heads；GQA 下 query heads 与 KV groups 的可整除关系决定 shard 合法性。核心计算：

```text
scores = Q Kᵀ / sqrt(head_dim)
scores += mask
probs = softmax(scores)
context = probs V
```

但工程实现还受以下因素影响：

- causal/padding mask；
- RoPE 施加位置；
- packed sequence 的 `cu_seqlens`；
- CP 的 KV 交换；
- FlashAttention/TE backend；
- softmax 是否 fp32；
- attention dropout RNG 是否跨 TP/CP 正确管理。

## 6. MLP 的列并行与行并行

SwiGLU MLP 通常：

```text
x --ColumnParallelLinear--> [gate, up] shard
  --SiLU(gate) * up-->
  --RowParallelLinear--> output
```

第一层按输出维切，GPU 各算中间维的一部分；第二层按输入维切并汇总输出。这使两个大 GEMM 都本地化，仅在边界通信。

## 7. 验证一个模型改动

至少做四层验证：

1. 单卡 forward 与 HF/reference 对齐；
2. 单卡 forward+backward，检查关键参数 gradient；
3. TP=2/PP=2 与单卡 loss、更新后权重在容差内一致；
4. checkpoint save/load 与不同 DP 规模恢复。

性能验证必须报告实际 module 类型、kernel 名、shape 和 dtype，不能只报告总 tokens/s。

## 8. `GPTModel.__init__` 源码精读

`megatron/core/models/gpt/gpt_model.py:47` 的构造参数直接暴露了 PP 与 TP 的语义：

```python
class GPTModel(LanguageModule):
    def __init__(
        self,
        config,
        transformer_layer_spec,
        vocab_size,
        max_sequence_length,
        pre_process=True,
        post_process=True,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        pg_collection=None,
        vp_stage=None,
    ):
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.vp_stage = vp_stage
```

- `pre_process=False` 表示输入来自前一个 PP stage，而不是 token embedding。
- `post_process=False` 表示输出还要发给下一 stage，不构建最终 logits/loss 路径。
- `parallel_output=True` 让词表 logits 保持 TP shard，避免 `[tokens,vocab]` all-gather。
- `pg_collection` 是当前模型所需通信组的显式载体；新代码应尽量避免在模块深处读全局 `parallel_state`。
- `vp_stage` 区分同一物理 rank 上的 virtual chunks。

### 条件构建为什么影响 checkpoint

PP=2 时，rank0 可能拥有：

```text
embedding + layers[0:N/2]
```

rank1 拥有：

```text
layers[N/2:N] + final_norm + output_layer
```

因此单 rank `state_dict()` 从来不是完整模型。checkpoint 框架必须知道逻辑层号和 global offset，不能把本地 key 顺序当作全局顺序。

## 9. `ModuleSpec` 的实例化路径

建议从 `megatron/core/transformer/spec_utils.py` 追：

```text
ModuleSpec
  → build_module(spec, ...)
      → spec.module(...)
          → 把 spec.submodules 传给父模块
              → 父模块继续 build 子 spec
```

一个典型 attention spec 的逻辑形态：

```python
ModuleSpec(
    module=SelfAttention,
    params={"attn_mask_type": AttnMaskType.causal},
    submodules=SelfAttentionSubmodules(
        linear_qkv=ColumnParallelLinear,
        core_attention=DotProductAttention,
        linear_proj=RowParallelLinear,
    ),
)
```

这里 QKV 用 Column Parallel：每个 TP rank 得到部分 heads；输出投影用 Row Parallel：消费本地 heads 后跨 TP 汇总。这不是随意搭配，而是通信最少的成对布局。

### local spec 与 TE spec 对照法

同时打开：

- `megatron/core/models/gpt/gpt_layer_specs.py`
- `megatron/core/extensions/transformer_engine_spec_provider.py`

逐项比较 norm、QKV、core attention、projection、MLP。逻辑行为应一致，但 TE 可能把 bias/dropout/add 或 norm+linear 融合。若只在 local path 加新参数，生产默认 TE path 可能根本没运行你的代码。

## 10. forward 的 shape 推演

以 `S=4096, B=2, H=8192, heads=64, TP=4`：

```text
hidden_states 全局逻辑 [S,B,H]
每 TP rank Q heads = 64/4 = 16
local Q shape ≈ [S,B,16,128]
若 GQA kv_heads=8，每 rank local KV heads 要按实现/分组合法切分
attention context local hidden = H/4 = 2048
RowParallel projection 后恢复逻辑 H，物理上可能进入 SP shard
```

检查新模型最先验证：

```text
num_attention_heads % TP == 0
num_query_groups 与 TP 的复制/切分规则合法
ffn_hidden_size 的 TP shard 满足 kernel alignment
vocab_size padding 后能被 TP 整除
```

## 11. 参数共享的通信

`share_embeddings_and_output_weights=True` 时，首末 PP stage 可能持有同一个逻辑权重的副本。它们不在相邻 P2P 语义里，所以 Megatron 创建 embedding group，在参数初始化/gradient finalize 时同步。

一个常见 bug 是只在 DP group reduction 后认为 embedding gradient 已一致；实际上首尾 stage 是不同 PP rank，必须通过 embedding group 对齐。

## 12. 本章延伸阅读

- [Megatron Core Transformer API](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/transformer.html)：按类核对 `TransformerBlock/Layer/Config`。
- [GPT Model API](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/models/models.gpt.html)：理解模型层与 Core 组件的接口。
- [Transformer Engine 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)：理解 TE module、FP8 autocast 和融合算子。
- [Megatron Core 新功能介绍](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-megatron-core-functionalities/)：从框架使用者角度理解 composable Core。
