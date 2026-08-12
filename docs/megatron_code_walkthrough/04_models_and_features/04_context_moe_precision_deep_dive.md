# Context Parallel、MoE/EP 与低精度训练深读

## 1. Context Parallel（CP）

TP 切 head/hidden，CP 切 sequence。每个 CP rank 只持有一段 query 和相关 activation，但 attention 需要访问全上下文的 K/V，因此必须交换 K/V 或中间统计量。

官方支持的通信类型随版本变化，常见思路包括：

- all-gather K/V 后本地 attention：简单但峰值内存高；
- ring P2P：分块旋转 K/V，逐块累积；
- all-to-all：在 sequence/head layout 间变换；
- hierarchical 组合：匹配节点内 NVLink 与节点间 IB。

### ring attention 的在线 softmax

分块处理 scores 时不能分别 softmax 再相加。需维护每行：

```text
m = 已见分块最大值
l = Σ exp(score - m)
o = Σ exp(score - m) V
```

新分块最大值为 `m_b`，令 `m_new=max(m,m_b)`，旧统计要乘 `exp(m-m_new)` 后与新统计合并。最终输出 `o/l`。这与分布式 log-sum-exp 同源，是 CP 数值正确性的核心。

排查 CP 错误重点检查 causal mask 在“全局 token 坐标”上是否正确，而不是本地 shard 坐标。

## 2. MoE 数据流

`MoELayer` 位于 `megatron/core/transformer/moe/moe_layer.py:213`，`TopKRouter` 位于 `moe/router.py:141`。典型流程：

```text
tokens
  → router logits
  → top-k expert + probability
  → token permutation / dispatch
  → all-to-all 到 expert owner
  → grouped expert GEMM
  → all-to-all 返回
  → unpermute + weighted combine
```

MoE 参数量大但每 token 只激活少数 expert。瓶颈常不是 FLOPs，而是 token dispatch、负载不均、小 expert batch 和 all-to-all。

## 3. router 的关键语义

router 不只是 `topk(softmax(xW))`：

- score function：softmax/sigmoid 等；
- pre/post-softmax top-k；
- load balancing auxiliary loss；
- z-loss；
- expert bias 动态平衡；
- capacity factor、token drop/pad；
- expert choice 或 node-limited routing；
- deterministic routing 与 tie-breaking。

配置中关闭或错误缩放 auxiliary loss 可能让 expert 负载塌缩。当前实现通过 `MoEAuxLossAutoScaler` 把 auxiliary loss 自动挂到 activation 的 autograd 边上，并由 schedule 设置 accumulation scale；任务 closure 再手工相加会双计。完整调用链见[源码问题详解第 15 节](../06_reference/03_source_questions.md)。

### token permutation 为什么难

需要同时保存：

- 原 token index；
- 目标 expert；
- top-k slot；
- routing probability；
- 各 expert token count；
- 跨 EP rank send/recv split。

返回后必须精确逆置换并按概率合并。空 expert、重复 expert、capacity drop、packed sequence 都是边界条件。

## 4. EP、TP、SP 的组合约束

EP 把 experts 分给 ranks；TP 还可切单个 expert 的矩阵。TP+EP 时通常要求 SP，以保证进入 token dispatcher 的布局和并行组语义一致，并避免复制/错误通信。

性能上：

- EP 太小：单 rank expert 参数多、显存高；
- EP 太大：每 expert token 太少，GEMM 变碎，A2A 更贵；
- TP 太大：expert GEMM shard 变小、TP collective 频繁；
- global batch/seq 太小：router 无法给每 expert 足够 token。

调参应同时看每 expert token 直方图、A2A 时间、grouped GEMM shape，而不是只看总 step time。

## 5. FP8/FP4 不只是改 dtype

低精度训练需要缩放：

```text
x_fp8 = quantize(x / scale)
近似恢复 x ≈ dequantize(x_fp8) × scale
```

scale 通常来自 amax 历史或 current scaling recipe。工程关键：

- 哪些 tensor 用 FP8，哪些累加仍是高精度；
- amax 如何收集、延迟、跨 rank 归约；
- forward/backward recipe；
- stochastic rounding；
- layer 是否因 shape/backend fallback；
- checkpoint 是否保存 scale/amax state。

只比较最终 loss 不够。应监控 activation/gradient amax、overflow/underflow、层级误差，并用短程 BF16 baseline 做权重或 logits 对齐。

## 6. recompute 与 activation offload

- recompute：少存 activation，backward 前重算；交换 compute；
- offload：把 activation 搬到 CPU，backward 前搬回；交换 PCIe/NVLink-C2C 带宽；
- fine-grained 策略可对 attention/MLP/norm 分别选择。

理想 overlap 条件是搬运时间小于可遮蔽的计算窗口。否则显存下降但 step time 明显上升。还要计入 pinned host memory 与 NUMA 位置。

## 7. 组合功能的验证矩阵

新增模型/特性至少覆盖：

| 维度 | 最小组合 |
|---|---|
| 精度 | BF16 baseline、目标 FP8/FP4 |
| 并行 | 单卡、TP、PP、目标 CP/EP |
| 序列 | 短序列、目标长序列、packed |
| checkpoint | 保存/恢复、必要时重分片 |
| 性能 | 无 overlap baseline、目标 overlap |

功能各自能跑不代表组合能跑；Megatron 最多的问题出现在 feature cross-product。

## 8. CP 的局部与全局坐标

先用“连续切半”只解释全局 mask 语义。假设 sequence=8、CP=2，逻辑上 rank A 处理 query `[0,1,2,3]`、rank B 处理 `[4,5,6,7]`，对 causal attention：

```text
rank0 query 0..3 只能看全局 key <= query
rank1 query 4..7 可看 rank0 全部 KV + rank1 对应前缀
```

如果 rank B 把本地 query 0 当作全局 0，它会错误 mask 掉前半 KV。packed sequence 时还必须防止一个 sample 的 token 看见相邻 sample。

但这不是当前 batch helper 的实际 placement。`_get_batch_on_this_cp_rank_per_sequence_balancing` 会把序列切成 `2×CP` 个 chunk 并 zigzag 分配；`S=8, CP=2` 时 rank0 得 `[0,1,6,7]`，rank1 得 `[2,3,4,5]`，用首尾配对均衡 causal work。实现与 mask 坐标的完整区分见[源码问题详解第 8 节](../06_reference/03_source_questions.md)。

官方 CP 图指出：CP 会切所有 activation；非 attention module 不做跨 token 运算，可直接处理 local sequence，只有 attention 必须交换 KV。

### 在线 softmax合并代码

两个 KV block 的等价合并：

```python
m_new = torch.maximum(m_old, m_block)
alpha = torch.exp(m_old - m_new)
beta = torch.exp(m_block - m_new)
l_new = alpha * l_old + beta * l_block
o_new = alpha * o_old + beta * o_block
```

这里 `o_block` 已是 `Σ exp(score-m_block)V`。最后 `o_new/l_new`。m、l 的累积精度影响长上下文稳定性。

## 9. `MoELayer` 构造源码精读

`moe_layer.py:213`：

```python
self.router = self.submodules.router(
    config=self.config,
    pg_collection=pg_collection,
)

if config.moe_token_dispatcher_type == "allgather":
    self.token_dispatcher = MoEAllGatherTokenDispatcher(...)
elif config.moe_token_dispatcher_type == "alltoall":
    self.token_dispatcher = MoEAlltoAllTokenDispatcher(...)
```

router 决定逻辑 token→expert，dispatcher 决定物理 token 如何到 expert owner。分开后可以保持 routing 算法不变，仅替换 AllGather、AllToAll、DeepEP 等数据面。

### `TopKRouter` 状态

`router.py:141` 初始化：

```python
self.topk = config.moe_router_topk
self.routing_type = config.moe_router_load_balancing_type
self.score_function = config.moe_router_score_function

if config.moe_router_enable_expert_bias:
    self.register_buffer("local_tokens_per_expert", ...)
    self.register_buffer("expert_bias", ...)
```

`expert_bias` 是持久 buffer，会进入 checkpoint；`local_tokens_per_expert` 标为 `persistent=False`，是运行统计。恢复训练时二者语义不同。

## 10. token dispatch 的数值例子

4 tokens、4 experts、top-2、EP=2：

```text
t0 → e0(.7), e3(.3)
t1 → e1(.6), e0(.4)
t2 → e2(.9), e3(.1)
t3 → e3(.8), e1(.2)

rank0 owns e0,e1: 接收 t0,t1,t1,t3
rank1 owns e2,e3: 接收 t2,t0,t2,t3
```

注意一个 token 出现 top-k 次。dispatch metadata 需保存 `(source_rank, original_token, topk_slot, probability)`。expert 输出返回后：

```text
out[t0] = .7*out_e0(t0) + .3*out_e3(t0)
```

若 capacity 丢掉某一路，要按配置重新归一化或保持明确定义。

## 11. FP8 缩放状态

TE 的 delayed scaling 可维护 amax history：

```text
step t: 用历史 amax 计算 scale_t
       quantize/compute
       记录当前 amax_t
step t+1: history 滑动，再算 scale
```

因此 checkpoint 不保存 scaling metadata 可能让恢复后最初若干 step 数值不同。跨 DP/TP 是否归约 amax、何时归约也影响一致性。

排查时同时记录：

- layer/module 名；
- input/weight/grad amax；
- scale 与 inverse scale；
- saturation/zero 比例；
- 是否实际使用 FP8 kernel。

## 12. offload 的带宽预算

假设每层要 offload 256 MiB activation，forward 到对应 backward 之间有 20 ms 可隐藏，单向最低有效带宽：

```text
256 MiB / 20 ms ≈ 12.8 GiB/s
```

还要完成 D2H 和之后 H2D，并与其他 microbatch 竞争 PCIe。若 NUMA 远端或 pinned pool 不足，理论窗口无法实现。

当前官方实现由 `PipelineOffloadManager`、每 microbatch `ChunkOffloadHandler` 和 module interface 三层组成；这与 PP/VPP 的 chunk 生命周期直接耦合。

## 13. 本章延伸阅读

- [Context Parallel 官方指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)：含 TP2CP2 通信图与适用原因。
- [Dynamic Context Parallelism 技术博客](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)：理解变长样本为何需要每 microbatch 动态 CP。
- [MoE 官方指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)：dispatcher、capacity、overlap、parallel folding 的当前配置。
- [Switch Transformer](https://arxiv.org/abs/2101.03961)：router、capacity 与 load balancing 的基础。
- [Transformer Engine FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)：理解 scaling recipe。
- [Fine-Grained Activation Offloading](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/fine_grained_activation_offloading.html)：模块粒度、组件和限制。
