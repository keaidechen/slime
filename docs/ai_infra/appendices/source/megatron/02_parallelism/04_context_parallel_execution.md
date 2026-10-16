# CP 执行走读：从 batch 坐标到 attention backend

基线：本仓库 Megatron commit `21fe0fe1597932421f1bb0efd93f469376a7a255`。数学前置：[CP 概念](<../../../../04_Training/15_模型并行与执行日程.md#s4>)、[在线 softmax 与远端梯度推演](<../../../../04_Training/15_模型并行与执行日程.md#s4-数值展开-sp-与-cp布局转换在线-softmax-与远端梯度>)。本章以 GPT、causal attention、固定 CP 和 TE attention 为主线。

## 1. 出发点：SP 无法独自解决长上下文的全部激活

SP 只改变 TP 区间内部分激活的保存布局；CP 让不同 ranks 分别处理一条长序列中的 query/token，非 attention 层可在本地 token 上计算，attention 必须访问所需远端信息。

CP 切 activation/token 工作，不直接切参数矩阵。相同参数跨 CP ranks 复制，其梯度包含不同 token 的贡献，必须最终按正确的数据/上下文域汇总。

以 `S=8,B=1,H=4,CP=2,TP=1` 为例，每卡持有四个 query 位置的 hidden states `[4,1,4]`，同一层的权重仍各有完整一份。若是一个 head、head 维 d=4，则本地 Q/K/V 可写为 `[4,1,1,4]`。本地 Q 只产生本地四个位置的输出，但其中后部位置必须使用远端 K/V。CP 不能把两半序列当成两条独立短样本，否则训练目标就变了。

causal attention 中全局位置 i 有 `i+1` 个合法 key（编号从 0 开始）。全序列的合法 query-key 对数是 `1+...+S=S(S+1)/2`；它解释后半段比前半段更贵，也解释下一节为什么按首尾配对，而不只按 token 数平均。

## 2. 从 batch 到 attention 的完整责任链

| 层次 | 当前源码入口 | 需要核对的对象 |
|---|---|---|
| 组网 | [parallel_state.initialize_model_parallel](<../../../../../../Megatron-LM/megatron/core/parallel_state.py>) | `cp`、`dp-cp`、`tp` groups；哪些 ranks 合作处理同一序列 |
| 输入 | [utils.get_batch_on_this_cp_rank](<../../../../../../Megatron-LM/megatron/core/utils.py>) | per-sequence、per-document、hybrid 分支；token、label、position、mask 的一致切分 |
| QKV | [transformer/attention.py](<../../../../../../Megatron-LM/megatron/core/transformer/attention.py>) | CP-local sequence 与 TP-local heads；Q/K/V shape 和 packed metadata |
| backend 适配 | [TEDotProductAttention](<../../../../../../Megatron-LM/megatron/core/extensions/transformer_engine.py>) | `cp_group`、global ranks、`cp_stream`、`cp_comm_type` 的传递 |
| attention 通信与反向 | 已安装版本的 `transformer_engine.pytorch.attention` 及其 CP 实现 | KV 布局、mask、在线合并、远端 dK/dV 返回、stream 依赖 |
| 参数梯度收尾 | [DDP](<../../../../../../Megatron-LM/megatron/core/distributed/distributed_data_parallel.py>) 与 [finalize_model_grads](<../../../../../../Megatron-LM/megatron/core/distributed/finalize_model_grads.py>) | dense 参数的数据/上下文归约域、有效 token 分母、梯度缩放 |

Megatron 的 adapter 是一个真实实现边界：不能在该文件里看到 `cp_comm_type='p2p'`，就宣称已经走读了 ring attention kernel。继续深入必须记录 TE 版本和相应源码；本章不把外部版本的 kernel 行号伪装为本仓库代码。

## 3. 第一个具体结果：每 rank 究竟拿到哪些 token

`get_batch_on_this_cp_rank` 按 metadata 分派：

- 没有 `cu_seqlens`，或显式选择 per-sequence：走 `_get_batch_on_this_cp_rank_per_sequence_balancing`。
- 有 `cu_seqlens` 且非 hybrid：走 per-document 分支。
- hybrid：从样本的 `local_cp_size` 取得所用 group；不能继续按全局固定 CP size 解释这一批。

默认 per-sequence helper 先分 `2×CP` 段，再取首尾配对。`S=8,CP=2`：

| rank | chunk IDs | 全局 token IDs | causal 可见 pair 数 |
|---:|---|---|---:|
| 0 | 0,3 | 0,1,6,7 | 1+2+7+8=18 |
| 1 | 1,2 | 2,3,4,5 | 3+4+5+6=18 |

连续切半会得到 10 与 26，长度均衡而计算不均衡。此 helper 要求长度满足 `2×CP` 整除；`attention_mask` 的 sequence 轴和普通 batch tensor 不同，metadata 不按 token tensor 一起乱切。

Packed 场景的 per-document 路径还使用 TE 的 partition-index 逻辑，需保留 `cu_seqlens` 等边界。改变 placement 不能改变文档隔离、position 或 causal 关系。

## 4. 第二个具体结果：为什么交换 KV 后仍需正确合并

本地 query 为 Qr，分块 KV 为 `(Kj,Vj)`。每块产生未归一化统计 `(mj,lj,oj)`。合并时：

```text
m = max(m_old, m_new_block)
l = exp(m_old-m) l_old + exp(m_new_block-m) l_new_block
o = exp(m_old-m) o_old + exp(m_new_block-m) o_new_block
最终 output = o/l
```

只使用本地 KV 会漏掉远端项，两个块分别 softmax 再平均又会给块错误的权重。必须在公共归一化基准上合并。全 masked 块和空状态需要专门处理，不能计算未定义的 `-inf - -inf`。下面用两个 key 展开所有中间值。

### 把在线合并逐项算一次

对单个 query，设两块各有一个合法 key，分数分别为 `s₀=0,s₁=ln3`，value 分别为 `v₀=2,v₁=6`。这里分数已包含 `QKᵀ/sqrt(d)`；m、l 是每 query/head 的标量，o 是长度 d_v 的 value 加权和；多个 query 时可将 m/l 看作 `[query数,head数]`，o 多一个 d_v 轴。

第一块：`m₀=0,l₀=exp(0-0)=1,o₀=1×2=2`。第二块单独计算：`m₁=ln3,l₁=1,o₁=6`。合并取 m=ln3，把旧块的统计换到新基准：`exp(m₀-m)=1/3`，新块倍率为 1。因此 `l=1/3+1=4/3`，`o=2/3+6=20/3`，结果 `o/l=5`。两块各自 softmax 后平均却是 `(2+6)/2=4`，因为它把“块”当成等权，而真实 key 概率是 1/4 与 3/4。

一般推导只用指数恒等式：`exp(s-m_new)=exp(s-m_old)·exp(m_old-m_new)`。把旧块里每个分子和分母同时乘这个倍率，不改变其原有信息，也就能和新块相加。初始空状态 l=0、o=0，m 的哨兵处理和全 masked 块需要 backend 显式分支。

这解释数学契约，不限定 backend 必须以未归一化 o 保存：实际 FlashAttention/TE 可以使用 output 与 log-sum-exp 等等价形式。

## 5. 反向与 loss 分母：两个独立问题

**attention 内部**：每个 query shard 对远端 K/V 也有梯度。`dQ` 归 query owner；`dK/dV` 需要返回 KV owner 并累加其他 query shard 的贡献。只验证 forward 无法发现这类训练错误。

上面两 key 例子还能手算远端梯度。令最终输出的上游梯度为 1，概率 `p=[1/4,3/4]`，则 `dV=[1/4,3/4]`；对分数，`ds_j=p_j(v_j-output)`，所以 `ds=[-3/4,3/4]`。若 d=1、q=1，则 `dK=[-3/4,3/4]`，`dQ=Σ_j ds_j k_j=(3/4)ln3`。远端 key 即使不在 query owner 的设备上，也得到非零梯度；丢掉它会让 forward 正确、训练错误。多个 query 对同一 K/V 的贡献还必须相加后交回 owner。

ring 中通信 stream 接收下一块，计算 stream 消费当前块，二者可以交叠；但接收未完成不可读取，计算尚未完成不可用下一轮覆盖该块。反向所需的本地 Q、输出/归一化统计、K/V 来源关系也必须保存或重建。双缓冲增加 overlap 空间，也增加峰值显存。

**模型参数与目标函数**：CP ranks 持有同一参数并处理同一序列的不同部分。dense DDP 路径需要考虑包含 CP 的数据归约域；有效 token mean 则需要整条序列和各数据副本的正确总分母。已有[源码问题第 5 节](<../06_reference/03_source_questions.md>)解释 `calculate_per_token_loss` 与 legacy microbatch mean 的区别。不能在任务 loss 中自行除 CP，再让 schedule/finalize 缩放一次。

## 6. 通信选择、收益与代价

| 通信思路 | 怎样完成上下文访问 | 主要权衡 |
|---|---|---|
| KV all-gather | 先拿到所需完整 KV，再计算本地 queries | 容易理解，但完整 KV buffer 峰值大 |
| P2P/ring | 逐块接收并消费 KV，合并统计 | 可做通信计算重叠；有多轮消息与依赖，短序列可能不划算 |
| all-to-all | sequence/head 之间转换布局 | 受 head/GQA、可整除性和 backend 限制 |
| hierarchical | 在不同网络域组合协议 | 拓扑匹配可能获益；分组与实现约束更多 |

当前 adapter 明确依据 TE 版本设置 CP 能力和通信参数。版本最低要求不等于任意模型、mask、dtype、head layout 都支持；先检查实际 module 和配置。

收益是 local token/activation 规模下降，而不是把参数也除 CP；FlashAttention 已避免完整存储 S² scores，不能把 CP 的显存收益建立在“原来一定存 S²”这个假设上。KV 临时 buffer、通信重叠和 TP/SP 的布局还会改变真实峰值。

## 7. 怎么运行和判断结果

数学检查：`python3 docs/ai_infra/examples/parallel_runtime_walkthroughs.py` 的 `sequence_context` 会核对 attention 输出、online merge 和有限差分 dQ/dK/dV。

CP 的分布式等价性至少依赖以下条件，证据层次见[并行边界](<../05_design_and_failures/03_parallelism_validation.md>)：

1. CP=1 参考与 CP=2 zigzag，在全局 token 顺序下对齐输出及梯度。
2. 两个 packed documents，核对 document 边界不发生跨样本 attention。
3. 固定总有效 tokens，核对 `calculate_per_token_loss` 对齐目标分母；梯度异常成倍时先查归约和缩放。
4. 不同序列长度下记录峰值、KV 通信时间和 step time；长序列有收益不意味着短序列也有收益。

观测必须同时保存局部与全局 token IDs；只有 `[S/CP,B,H]` shape 还不能证明索引正确。当前文档没有附真实 TE 多卡 trace，数学 PASS 不代表已覆盖外部 kernel。
