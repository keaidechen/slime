# EP / ETP：从 token 路由走到专家梯度

本章固定 `Megatron-LM` commit `21fe0fe1597932421f1bb0efd93f469376a7a255`，主线选择 **普通 `MoEAlltoAllTokenDispatcher`、dropless、无 shared-expert overlap、无低精度 padding**。AllGather、Flex/DeepEP 等路径不能套用全部细节。


## 1. 出发点与实际入口

EP 分摊 experts，但单 expert 可能仍太大；ETP 切单 expert。代价分别是路由 A2A 和 expert TP 的布局变换。构模入口为 [MoELayer](<../../../../../../Megatron-LM/megatron/core/transformer/moe/moe_layer.py>)，沿 router、dispatcher、experts 三个成员往下读。

[MoETokenDispatcher.__init__](<../../../../../../Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py>) 中最容易误读的赋值是：

```python
self.ep_group = pg_collection.ep
self.tp_group = pg_collection.expt_tp
self.tp_ep_group = pg_collection.tp_ep
```

因此本文件中的 `tp_size` 是 **ETP**，不一定是 attention TP。`tp_ep` 是 expert TP 与 EP 的联合域，用于汇总 token 分布等信息。

把一个 expert 看成独立的 MLP，router 看成给每个 token 选专家的函数。EP（Expert Parallelism）决定“专家 e 住在哪张卡”；ETP（Expert Tensor Parallelism）决定“专家 e 的一个矩阵由哪几张卡合算”；EDP（Expert Data Parallelism）决定“同一专家分片还有几份处理其他数据的副本”。三个组不是三个同义词。

dropless 指所有选中的 token-expert 路径都保留，不因容量上限丢弃；top-k 的 k 是每 token 选择的专家数。即使某个 rank 本地没有分到 token，它也仍可能是通信组成员，不能直接跳过所有 collective。路由 metadata 和 payload 的顺序必须由所有参与者共同遵守。

## 2. 参数和网格如何确定

[parallel_state.initialize_model_parallel](<../../../../../../Megatron-LM/megatron/core/parallel_state.py>) 先建 decoder generator，再建 expert generator：

```text
decoder: tp=TP, cp=CP, ep=1, dp=DP, pp=PP
expert:  tp=ETP, cp=1, ep=EP, dp=EDP, pp=PP
EDP = W / (ETP × EP × PP)
```

未指定 ETP 时采用 TP；还要验证整除、rank order 和两套 PP groups 一致。expert generator 的 `get_ranks('tp')` 创建 expert TP 组，`get_ranks('ep')` 创建 EP 组，`get_ranks('dp')` 创建 EDP 组。

从 [experts.py](<../../../../../../Megatron-LM/megatron/core/transformer/moe/experts.py>) 的实际构造类继续确认 FC1/FC2 使用 `pg_collection.expt_tp`。不要仅查看普通 `MLP` 的 TP 参数就断言专家切分相同。

## 3. 六个阶段的源码与张量账本

记本 rank 输入展平后为 `[T,H]`，总 experts 为 E，top-k 为 K，当前 EP owner 的专家数为 `L=E/EP`。

| 方法（均在 token_dispatcher.py） | 输入/输出和动作 | 所属组/不变量 |
|---|---|---|
| `dispatch_preprocess` → `preprocess` | `[T,H]`、`routing_map:[T,E]`、`probs:[T,E]`；计数，按 expert 排 token；无 drop/pad 时得到 `[T×K,H]` | 保存逆置换；每 token 可以出现 K 次 |
| `token_dispatch` | 用 `input_splits/output_splits` 对 token 和 probability 分别 A2A | **EP**；接收行数等于 `sum(output_splits)` |
| `dispatch_postprocess` | ETP>1 时 gather token/probability；再从“来源 rank 块”排成“本地 expert 块” | **ETP**；同一 expert 的各矩阵 shard 看到匹配 token 顺序 |
| expert forward | FC1 shard → activation/加权 → FC2 shard，产出 `[接收 token 数,H]` partial output | `tokens_per_expert` 决定 grouped GEMM 分段 |
| `combine_preprocess` | 撤销第二次排序，再 reduce-scatter partial output | **ETP**；同时求和并恢复 token 分布 |
| `token_combine` → `combine_postprocess` | 反向 EP A2A、撤销第一次置换、合并重复 token、恢复原 hidden shape | **EP**；回到 residual 分支需要的 token owner |

这里的六阶段指 dispatcher 的六个公开方法，expert forward 插在 dispatch 与 combine 中间。展平和重排通常需要额外 storage，不能只核算 experts 的参数。

## 4. 两个 EP ranks 的 split 手算

先令 ETP=1、E=4、EP=2，每 rank 两个 tokens，top-2，无 drop/pad：

```text
rank0 输入：t0→e0,e3；t1→e1,e0
rank1 输入：t2→e2,e3；t3→e3,e1
EP owner0 管 e0,e1；owner1 管 e2,e3
```

发送计数矩阵 `S[source,destination]`：

```text
           dest0 dest1
source0      3     1
source1      1     3
```

rank0 `input_splits=[3,1]`，`output_splits=[3,1]`；rank1 分别为 `[1,3]` 与 `[1,3]`。本例矩阵恰好对称，不能由此推断 send/recv splits 总是相同。普遍关系是接收 split 取矩阵的一列，发送 split 取一行。

一个不对称例子是 `S=[[3,1],[2,2]]`：rank0 send `[3,1]`、recv `[3,2]`；rank1 send `[2,2]`、recv `[1,2]`。此时 rank0 接收 5 行，rank1 接收 3 行。

`preprocess` 对本 rank 的 expert 计数先按 `[EP,L]` 求和得到 `input_splits`；在 `tp_ep_group` 汇总所有计数后，取出本地 experts 的计数，得到 `output_splits` 和 `output_splits_tp`。后者描述 ETP gather/RS 的变长分段。`token_combine` 交换 A2A 的输入/输出 split 角色，完成逆向传输。

把第一个例子的记录写出来，才能看懂两次置换。rank0 发向 owner0 的三条记录为 `(t0,e0)、(t1,e1)、(t1,e0)`，rank1 发来 `(t3,e1)`；接收时可能按来源排列，计算时则要排成 e0 的 `[t0,t1]` 与 e1 的 `[t1,t3]`。这里 t1 出现两次是有意的 top-2 复制，不是重复读取样本。第二次排序让同一专家的矩阵乘法拥有连续行，逆排序则把结果还给正确的来源。

例如 t0 的两条路径权重是 0.7、0.3，两个专家输出分别为 `[2,0]`、`[0,4]`，则合并结果为 `[1.4,1.2]`，shape 仍是 `[H=2]`。若上游梯度为 `[1,1]`，专家输出收到的梯度分别为 `[0.7,0.7]`、`[0.3,0.3]`；两个概率的直接梯度分别是点积 2 和 4，再通过 router 的归一化规则回传。不能在 expert 内乘一次概率，combine 时再乘一次，否则结果变成 `[0.98,0.36]`。

逆置换索引、split 计数和概率不仅供 forward combine 使用；反向还要沿同一路由送回梯度，因此应存活到相应 autograd 节点完成。A2A 的 send buffer 在传输完成前也不能当下一批的 permutation workspace 使用。

## 5. 再叠 ETP=2，会多什么

对固定 EP owner，两个 ETP ranks 各收到一部分来源 tokens。`dispatch_postprocess` 沿 `self.tp_group` gather，使两个 ranks 都能用自己的 FC1 shard 处理该 owner 的完整 token 集合。

对于一个 expert，令其收到 N 个 tokens：

```text
gather 后输入：两个 ETP ranks 均为 [N,H]
FC1：各产生 [N,F/2]，SwiGLU 的 FC1 输出另含 gate/up 两路
FC2：各产生 [N,H] partial sum
combine_preprocess：ETP SUM reduce-scatter
```

输出 `[N,H]` 形状完整不代表数值完整：FC2 结果仍可能只是某个 intermediate shard 的贡献。要看 communication ownership，不能凭 shape 决定是否还需 reduction。

可在 [TEGroupedLinear](<../../../../../../Megatron-LM/megatron/core/extensions/transformer_engine.py>) 直接核对这条边界：`explicit_expert_comm` 生效时先按真实 ETP 切 `input_size/output_size`，再给底层 TE 传 `parallel_mode=None`、`tp_size=1` 和空 TP group。这表示通信交给 Megatron 外层处理，不表示 expert 权重没有分片。随后还要保留正确的分片 metadata，供重分片和权重转换使用。

概率加权也有 ownership。[TEGroupedMLP.forward](<../../../../../../Megatron-LM/megatron/core/transformer/moe/experts.py>) 接收 `permuted_probs`，可能在 activation 路径融合乘法；普通 AlltoAll 的 `combine_postprocess` 调用 `unpermute` 时没有再传 probs。手工追踪时给每条 top-k 路径写一次权重，防止重复乘。

## 6. 反向：三种“合并”不能混成一次 all-reduce

[tensor_parallel/mappings.py](<../../../../../../Megatron-LM/megatron/core/tensor_parallel/mappings.py>) 的 `_AllToAll.backward` 交换 split 方向。结合 autograd，逻辑上：

1. 原 token 的输出梯度按原路由送回各 expert 分支，传播到 probability 和 expert 计算。
2. ETP forward RS 的反向是 AG，让各 FC2 shard 拿到所需输出梯度。
3. expert FC2/FC1 求本地参数梯度与输入贡献；ETP forward AG 的反向通常 RS，把输入贡献相加并送回 shard owner。
4. dispatch A2A 的反向再将贡献送回原 token owner；top-k 重复路径在逆置换反向中正确累加。
5. **EDP** 同步同一 expert shard 在不同数据副本上得到的参数梯度，之后才做一致的 optimizer update。

ETP 是一个函数的部分和；EP 是 token/expert 目的地与逆路由；EDP 是同一参数的样本贡献。三者混用会在 shape 完全合法时仍得到错误梯度。

## 7. 最终结果、代价与性能观察

预期结果是重建原 token 顺序后，输出、输入梯度和每 expert shard 的梯度与参考相符。参数容量按 `EP×ETP` 分摊，但输入 gather、routing metadata 和 permutation buffer 会占用峰值。

重点观测：

- `tokens_per_expert` 直方图与最大值；A2A 等待可能来自热点，未必是均匀带宽不足。
- `input_splits/output_splits` 从 GPU 复制到 CPU 的同步点；不能只看 NCCL kernel 时间。
- ETP gather 后的 token 数与 `F/ETP`，判断 GEMM 是否已经太小。
- 两次 permutation、两种通信与 expert GEMM 的 exposed tail；吞吐结论看整个 step。

dropless、drop-and-pad、FP8 padding、shared-expert overlap、AllGather 和 Flex dispatcher 会改变上述分段或同步。切换路径时先重新列账本，再复用性能结论。

令某组共有 T 个唯一 token、每 token 选 K 个专家，则总路由行数是 T×K，均匀时每个专家平均 `T×K/E` 行。例如 T=8、K=2、E=4，平均是 4；若计数为 `[10,2,2,2]`，热点专家需要处理平均量的 2.5 倍。增加专家数减少每个专家的 token 行数，可能让 GEMM 更小，不能只以参数分摊比例推算加速。

grouped GEMM 把多组不同专家矩阵乘法统一提交，主要减少小算子的调度开销；它不会把热点的 10 行自动均分给其他具有不同权重的专家。capacity/drop、路由平衡和网络布局解决的是不同问题，启用其中一个不能替代另两个。

## 8. 不对称路由如何暴露等价性的边界

[ETP 数学参考](<../../../../examples/etp_reference.py>)把分片前反向与完整矩阵联系起来；真实运行还增加不对称 splits、空 expert、多 expert/token 和通信顺序。平均路由的例子可能掩盖这些边界，因此它不能代表任意路由分布。

ETP=1 与 ETP=2 的区别也不只是一层 GEMM 的宽度。固定 world size 时，EP/EDP 的分解受网格约束；如果改变它们，状态复制、token 分配与梯度归约可能一起变化。参数和路由相同是比较一步更新的前提，稳态性能还取决于通信尾部和矩阵尺寸。各层证据的关系见[并行等价性](<../05_design_and_failures/03_parallelism_validation.md>)。

[并行总览](<00_strategy_and_topology.md>) · [MoE 与低精度](<../04_models_and_features/04_context_moe_precision_deep_dive.md>)
