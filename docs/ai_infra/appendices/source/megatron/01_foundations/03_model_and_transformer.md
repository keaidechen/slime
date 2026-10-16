# 模型、Transformer 组件与 ModuleSpec

相关基础：[通信与 tensor 基础](<../../../communication/两卡通信与torchrun.md>)。


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

### 从 token 到预测：先用小形状连起来

设词表大小 `V=8`、`B=1,S=4,H=4`，输入 ID 为 `[1,3,2,5]`。embedding 参数 `E:[8,4]` 保存每个 ID 对应的向量，查表得到 `[1,4,4]`，转置后是内部常见的 `[4,1,4]`。decoder 保持这个形状，但每个位置的向量已融入可见前缀信息。输出投影用 `W_out:[4,8]` 得到逻辑 logits `[4,1,8]`；logit 是尚未归一化的实数分数，不是概率。

以一个位置的四类简化输出 `[0,ln2,0,0]` 为例，指数是 `[1,2,1,1]`，softmax 除总和 5 得 `[0.2,0.4,0.2,0.2]`。若目标是第二类，交叉熵 `-ln0.4≈0.9163 nat`。训练一般只需目标类的负对数概率，不必把全词表概率长期保存给 Python。真实词表 V=8 的过程相同；词表并行还可避免完整 logits 跨卡拼接。

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

### norm 与 residual 分别解决什么

层叠加后，特征尺度会变化，直接把越来越不稳定的输入交给下一层不利于训练。Layer Normalization（LayerNorm，层归一化）对每个 token 的 H 个特征计算：

```text
μ = (x₁+...+x_H)/H
σ² = Σⱼ(xⱼ-μ)²/H
x̂ⱼ = (xⱼ-μ)/sqrt(σ²+ε)
yⱼ = γⱼ x̂ⱼ + βⱼ
```

μ 是特征均值，σ² 是总体方差，ε 是防止除零的小常数，γ/β 是可学习的缩放/偏移向量 `[H]`。这些模型数值无物理单位；ε 的尺度应与 σ² 相容。取 `x=[1,3]`，忽略 ε 的微小影响，μ=2、σ²=1，因此归一化为 `[-1,1]`。RMSNorm 则除以 `sqrt(Σxⱼ²/H+ε)`，不先减均值；同一输入得到约 `[0.447,1.342]`（令 γ=1），因此它不是 LayerNorm 的另一种名字。

residual（残差连接）把子层输出加回输入，例如 `h'=h+Attention(Norm(h))`。它保留一条不经过子层变换的传播路径，要求两项 shape 相同。pre-norm 指归一化放在子层之前。Dropout 则按概率 p 随机将元素置零，保留元素通常乘 `1/(1-p)` 以保持期望；重计算时需要复现相同的随机掩码。

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

这里 Q、K、V 分别是 query（查询）、key（键）、value（值），由 hidden states 的可学习线性投影产生。先只看单个 batch、一个 head：`Q:[S,d]`、`K:[S,d]`、`V:[S,d_v]`，于是 `QKᵀ:[S,S]`，每行是一个 query 对各 key 的匹配分数。softmax 对每行的 key 轴归一化，乘 V 后输出 `[S,d_v]`。多 head 是多套这样的计算；普通多头注意力 MHA 将 `H=n_heads×d`，各 head 的结果拼回 H。

为什么除以 `sqrt(d)`？在 Q/K 各分量近似独立、零均值、方差为 1 的启发式假设下，点积 `Σqᵢkᵢ` 的方差约为 d；除以 `sqrt(d)` 后方差约为 1，避免 d 增大时 softmax 过于尖锐。这是尺度控制的动机，不要求训练中的 Q/K 永远严格满足这个分布。

取 `S=2,d=1,Q=[[1],[1]],K=[[0],[ln3]],V=[[2],[6]]`。causal（因果）mask 禁止第一个位置看未来：第一行分数 `[0,-∞]` 给权重 `[1,0]`，输出 2；第二行分数 `[0,ln3]` 给 `[1/4,3/4]`，输出 `2/4+18/4=5`。mask 的 `-∞` 经指数变成 0。若不加 mask，第一个位置也输出 5，模型会在训练时偷看未来。

GQA（Grouped-Query Attention，分组查询注意力）让多个 Q heads 共享同组 K/V heads，例如 4 个 Q heads、2 个 KV heads，每两个 Q heads 共用一组 K/V，减少 K/V 的投影和缓存量。RoPE（Rotary Position Embedding，旋转位置编码）把 Q/K 的成对分量按位置旋转，使点积携带相对位置信息；它不改变这些张量的维度，但要求使用正确的全局位置。

RoPE 可以单独用一个二维特征对手算。角度 φ 的旋转矩阵为 `R(φ)=[[cosφ,-sinφ],[sinφ,cosφ]]`，将位置 i 的这对 Q 特征变成 `R(iθ)q`，K 同理；θ 是这一对特征的频率，iθ 是以弧度表示的角度。旋转后点积满足 `(R(iθ)q)ᵀ(R(jθ)k)=qᵀR((j-i)θ)k`，因为转置旋转等于逆旋转，两个旋转角度相减。因此同一特征对中的位置信号依赖 j-i。

例如 q=k=`[1,0]`、θ=π/2，位置相同的点积为 1，位置相差 1 的点积为 0。实际模型的不同特征对采用不同频率，这个例子仅展示旋转关系。前面的 d=1 attention 算例没有加 RoPE；旋转成对特征时至少要有相应的偶数维子空间。CP 切分后若把位置 6 错重置为 0，旋转角度就改变，单看 `[local_S,B,H]` 的 shape 无法发现。

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

MLP 是多层感知机，在这里是每个 token 独立执行的前馈网络。令展平后的 `X:[T,H]`，`T=S×B`。SwiGLU 使用两个投影 `G=XW_g`、`U=XW_u`，其中 `W_g,W_u:[H,F]`，再计算 `Z=SiLU(G)⊙U:[T,F]`，最终 `Y=ZW_d:[T,H]`，`W_d:[F,H]`。`⊙` 是逐元素相乘，`SiLU(a)=a/(1+exp(-a))`；gate 决定 up 分支每个特征通过多少。

例如单 token 的 gate 为 `[0,ln3]`、up 为 `[2,4]`，则 `SiLU(gate)=[0,3ln3/4]`，Z=`[0,3ln3]`；若 `W_d` 两行为 `[1,0]`、`[0,1]`，输出仍为 `[0,3ln3]`。合并实现会把 gate/up 一次投影成 `[T,2F]`，但 F 仍指激活后的中间宽度，不能误算两倍权重或只为一半中间值分配空间。

两张 TP 卡可以各持有 F 的一半：非线性只作用本地对应的 gate/up，FC2 结果再相加。这要求切分时保留 gate 与 up 的配对，而不是对拼接后的存储随便切一刀。

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

共享参数也可以用一行导数理解：同一 E 同时用于 embedding 和输出，损失对它的梯度为 `dE=dE_embedding+dE_output`。如果两个使用点在不同 PP stage，两边各自产生一部分贡献，必须在更新前合成同一个逻辑参数的梯度。仅仅让初始权重数值相等，并不意味着它们之后会自动保持共享。

## 12. 本章延伸阅读

- [Megatron Core Transformer API](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/transformer.html)：按类核对 `TransformerBlock/Layer/Config`。
- [GPT Model API](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/models/models.gpt.html)：理解模型层与 Core 组件的接口。
- [Transformer Engine 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)：理解 TE module、FP8 autocast 和融合算子。
- [Megatron Core 新功能介绍](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-megatron-core-functionalities/)：从框架使用者角度理解 composable Core。
