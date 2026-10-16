# MoE、MLA、MTP：高级架构如何进入训练系统

相关基础：[GPU、tensor 与通信基础](<../../../foundations/README.md>)。


## 入门例子：把新架构翻译为状态、算子和通信

MoE（Mixture of Experts，混合专家）由路由器为每个 token 选择少量专家子网络，只执行被选中的分支。总参数量决定存储，激活专家数决定部分计算，还需计入路由和交换成本。

MLA（Multi-head Latent Attention，多头潜在注意力）用压缩的历史表示及相关投影组织多头注意力，以减少 KV 状态；缓存容量、展开计算与位置编码必须按具体结构核算。

MTP（Multi-Token Prediction，多 token 预测）在训练中增加对后续多个位置的预测目标或模块。它改变训练计算与状态；是否用于推理草稿生成，是另一个实现选择。

MoE 增加 router、token dispatch/combine 与不规则 expert GEMM；MLA 改变 attention 的投影和缓存表达；MTP 增加额外预测计算及其训练/推理使用方式。理解模型名之后，必须分别追这三类工程变化。

例如 MoE 的 router replay 可以固定某些路由决策用于对齐，但不等于消除所有数值或采样差异。MLA 的缓存优势也取决于实际 backend 保存了什么，而不能仅从论文参数量推断进程显存。

每种结构都会增加或改变 tensor 的 shape、dtype、owner、lifetime 与通信。单层数学解释提供基本语义，整步性能还依赖组合与调度；Dense 路径可以作为理解新增状态的参照。

## 1. MoE 是一条分布式数据流

```text
hidden states -> router logits -> top-k/score
 -> permutation/dispatch -> local experts
 -> combine/unpermute -> output
```

EP 把 experts 分到不同 rank，token dispatcher 通过 all-to-all 或其他 backend 重排 token。性能由 expert GEMM 大小、token 不均衡、capacity/drop 策略、通信 overlap 与网络共同决定。Router 的 load-balancing loss、z-loss、dtype 和 replay/recompute 还会影响数值行为。当前实现通过 `MoEAuxLossAutoScaler` 把 auxiliary loss 自动接入 autograd；任务 loss 不应再手工重复相加，详见[源码问题详解第 15 节](<../06_reference/03_source_questions.md>)。

关键入口：`megatron/core/transformer/moe/` 下的 `router.py`、`token_dispatcher.py`、`experts.py` 与 `moe_layer.py`。

设展平后的输入 `X:[T,H]`，T 是本批 token 数；router 权重 `W_r:[H,E]` 把它投影为 `[T,E]` 的专家分数。选择 K 个专家后，每个 token 有 K 条计算路径，最终 `y_t=Σ_(e∈选择集合) p_te f_e(x_t)`，其中 f_e 是专家 MLP、p_te 是按当前路由规则得到的权重。两专家输出 `[2,0]`、`[0,4]`，权重 0.7、0.3 时，输出为 `[1.4,1.2]`，不是拼成四个特征。

若有 E=4 个同规模专家，每 token 只激活 K=2 个，其专家矩阵乘法工作约是全部执行的 1/2，但四个专家的参数都要保存；router、dispatch、inverse permutation 和网络还有额外成本。训练还需保存路由与反向状态，不能把“激活参数量”当作总显存占用。执行与梯度细节见[EP/ETP](<../02_parallelism/05_ep_etp_execution.md>)。

## 2. Router replay

Router replay 记录 top-k expert indices 并在后续 forward/recompute 使用，目的是固定离散 expert 选择。replay 时 probability 仍从当前 `scores` 对记录的 indices 做 gather，因此 router score 仍可有梯度；它既不冻结 router，也不保存完整 logits。需要区分 record、forward replay、backward replay，并保证记录与 microbatch/layer 对应。详见[源码问题详解第 16 节](<../06_reference/03_source_questions.md>)。

## 3. MLA

MLA（Multi-head Latent Attention，多头潜在注意力）用低维 latent 表示改变 Q/K/V 路径，目标之一是降低 KV 表示成本。启用时不仅设置开关，还需要 MLA 专用 config，并核对 RoPE 维度、Q/kv latent 维度、head 划分和 checkpoint mapping。代码从 `core/transformer/multi_latent_attention.py` 与相关 config 开始。

### 用形状理解压缩的对象

普通多头注意力在逐 token 推理时保存历史 K/V，避免每次重新投影全部前缀。若 n=4 个 KV heads、每 head 的 key/value 宽都为 d=2，一个位置需要 `2nd=16` 个数。MLA 的简化表示先将 hidden `h_t:[H]` 压成 `c_t=h_t W_down:[r]`，再以 `W_up_K:[r,n·d_k]`、`W_up_V:[r,n·d_v]` 提供所需 K/V 表示。c_t 是跨 head 共享的低维 latent。

只考虑非位置部分，若 r=3，则每位置保存 3 个数即可表达后续展开来源；还需单独计入实际结构中的 RoPE key 等内容。假设另存 d_R=2 个数，则每位置是 `r+d_R=5` 个数，长度 S=4、BF16 时缓存 40 bytes；前述普通多头缓存则为 `4×16×2=128 bytes`。这是结构账本，不包括投影权重、workspace 和训练 saved activations，也不是任意 MLA backend 的实际峰值。

为什么压缩后仍能算分数？对不含 RoPE 的一项，`q(c W_up_K)ᵀ=(q W_up_Kᵀ)cᵀ`，矩阵结合律允许先投影 query，再与历史 c 做内积，避免每次完整展开 K。位置旋转和其他子路径不一定可同样吸收，实际实现需专门处理。新增投影、算子支持和计算效率是交换条件；推理缓存节省不能直接外推为训练显存按同样比例下降。

## 4. MTP

Multi-Token Prediction 在每个位置增加未来多个 token 的预测深度。MTP module 通常含共享 embedding、projection、Transformer block 与共享 output head；额外 loss 按 scaling factor 合入主目标。

PP 下 MTP 默认靠近最后 stage，也可用自定义 layout 中的 `m` 指定，但所有 MTP 层需满足集中放置等约束。当前支持边界（例如与 CP、mask、position embedding 的组合）必须查当前版本，不应依据旧博客推断。

### 预测更远的标签怎样移位

取一条有五个 token 的文档 `[A,B,C,D,E]`。普通 next-token 目标在位置 0..3 是 `[B,C,D,E]`；向前再预测一位的辅助深度，其目标为 `[C,D,E,无效]`。主 loss 有 4 个有效位置，辅助 loss 只有 3 个，末尾不能卷绕成 A；packed documents 也不能从下一篇借标签。

用 D 个辅助深度、总权重 λ 的概念式表示：`L=L_main+(λ/D)Σ_(d=1)^D L_mtp,d`。每个 `L_mtp,d` 需要说明有效位置分母，λ 是无单位系数。例：主目标均值 2，两个辅助均值 3、4，λ=0.2，则总目标为 `2+(0.2/2)(3+4)=2.7`。实际 autograd 接入可能由 autoscaler 完成，不应在任务 closure 再加一次。

MTP 不是让普通 head 同时输出几个互不相关的猜测：具体模块还可能使用移位后的 embedding 与上一深度 hidden，增加投影、Transformer 计算与共享参数梯度。源码 `multi_token_prediction.py` 的 `roll_tensor` 和 loss mask 移位共同保护边界。推理是否把这些模块用作 speculative decoding（先生成草稿再验证），需要额外推理实现，不能由训练目标直接推断。

## 5. 组合验证

对 MoE/MLA/MTP 分别验证 dense/reference 或单 rank 等价、TP/EP shard shape、PP placement、checkpoint round-trip、recompute 前后一致性，再测性能。高级结构最常见错误是“forward 能跑，但 pipeline loss、共享参数同步或恢复边界错了”。
