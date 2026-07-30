# 12 训练侧内部实现：Megatron-LM 篇

> 衔接 [06_megatron_backend_and_mbridge.md](06_megatron_backend_and_mbridge.md)。
> 06 篇讲 slime 如何调用 Megatron；本篇深入仓库根目录 vendored 的 `Megatron-LM/` 源码，回答"slime 调的那几个入口背后发生了什么"。姊妹篇 [13_megatron_bridge_internals.md](13_megatron_bridge_internals.md) 讲格式转换层 Megatron-Bridge——两个库体量和主题差异较大（Megatron-LM 是训练执行引擎，Megatron-Bridge 是 HF↔Megatron 的转换/建模层），故拆成两篇分别讲透，避免一篇文档被迫在"引擎"和"转换器"两个话题间来回切换。

---

## 0. slime 的调用入口回顾

slime 的全部 Megatron 调用集中在四个入口（`slime/backends/megatron_utils/model.py:270-318`）：`get_model` → `OptimizerConfig` → `get_megatron_optimizer` → `get_optimizer_param_scheduler`；训练时经 `get_forward_backward_func()` 拿调度函数。下面逐个拆。

## 1. `get_model`：模型是怎么被"切"出来的

`megatron/training/training.py:1691-1885`。slime 传一个 `model_provider_func`（06 篇 §3），Megatron 负责切分与包装：

**（a）PP / VPP 切分**（build_model，1715-1750）：

- 每个 rank 只实例化**自己那段层**；`pre_process`（embedding）只给第一个 PP stage 的第一个 virtual chunk，`post_process`（输出头/loss）只给最后一个——这就是为什么 05 篇里 advantage 计算要判断"只有 pipeline 最后一段有 logits"；
- 开 VPP（interleaving）时一个 rank 持有 `vp_size` 个 model chunk，轮流参与调度。

**（b）TP 不在此函数**：TP 切分发生在模型内部——`ColumnParallelLinear/RowParallelLinear` 等层构造时读全局 mpu 的 TP 组自行切权重。

**（c）参数属性补全**（1775-1777）：给每个参数打 `tensor_model_parallel / partition_dim` 等属性——优化器靠这些属性判断"这是 TP 参数还是 expert 参数"，决定分片与通信策略。

**（d）精度包装**：bf16 时包 `Float16Module`（1804-1807）；FP8 时 `correct_amax_history_if_needed`（1813-1818）修正一个隐蔽坑——半精度转换的 in-place copy 会把当前 amax 误写进 TE 的 `amax_history`。

**（e）DDP 包装**（1820-1883）：slime 默认走 `DDP`（Megatron 的 LocalDDP，`megatron/core/distributed/distributed_data_parallel.py`）。内部机制：

- 梯度存进连续 buffer，按 bucket 切小异步 reduce，与反向计算重叠（bucket 默认 `max(40M, 1M×dp_size)` 元素）；
- **梯度缩放**（distributed_data_parallel.py:169-204）：目标是把梯度缩到 `1/dp_size`——`average_in_collective=True` 用 AVG 集合通信（expert 参数要预乘 `edp_size/dp_size` 修正），否则先乘 `1/dp_size` 再 SUM。**05 篇 §6 的"loss 缩放抵消 Megatron 内部平均"就是在这条链上对账的**；
- expert 与非 expert 参数分不同通信组（206-241 行）：expert 用 `intra_expt_dp_group`，普通参数用 `intra_dp_cp_group`——EP 与 DP×CP 的梯度归约各走各的。

## 2. `parallel_state.initialize_model_parallel`：通信组是怎么排的

`megatron/core/parallel_state.py:547`。给定 world size 和 TP/CP/EP/PP/DP 尺寸，按 **order**（默认 `tp-cp-ep-dp-pp`）把全局 rank 网格切成各维度的通信组。直觉：

```
world_size = TP × CP × EP × DP × PP
组内 rank 的选取 = 按 order 从高维到低维切分 rank 编号
```

例如 `TP=2, PP=2, DP=2`（world=8）：TP 组 = 相邻 2 个 rank（0,1）、(2,3)…；PP 组 = 跨最远的 rank（0,4）、(1,5)…。slime 04 篇的"只有 DP=0 且 TP=0 的 rank 当权重源"能成立，正是因为这套确定性排布保证了"每组取 rank 0 的成员恰好覆盖完整模型"。RL 侧常用的查询函数：`get_tensor_model_parallel_group / get_data_parallel_group(with_context_parallel=...) / get_expert_model_parallel_group / get_pipeline_model_parallel_group` 等。

### 2.1 深入拆解：`RankGenerator` 与 `generate_masked_orthogonal_rank_groups` 的精确公式

`parallel_state.py:446-521` 的 `RankGenerator` 是这套确定性排布的核心实现，公式其实很简单——只是"多维数组下标与一维 rank 编号互相换算"：

```
global_rank = tp_rank + cp_rank·TP + ep_rank·(TP·CP) + dp_rank·(TP·CP·EP) + pp_rank·(TP·CP·EP·DP)
```

（这是 `order="tp-cp-ep-dp-pp"` 时的展开式——**排在 order 里越靠前的维度，在 rank 编号里就是越低位、跨度越小、越"相邻"**；越靠后的维度跨度越大、越"分散"。）`generate_masked_orthogonal_rank_groups` 的源码注释给了一个可以直接验证的例子（`order='tp-pp-dp'`，`TP=2,PP=2,DP=2`，world=8）：

```
TP 组（mask=[T,F,F]）：[[0,1],[2,3],[4,5],[6,7]]     相邻两个 rank 一组（TP 是最内层）
DP 组（mask=[F,F,T]）：[[0,4],[1,5],[2,6],[3,7]]     跨度为 TP×PP=4 的两个 rank 一组
PP 组（mask=[F,T,F]）：[[0,2],[1,3],[4,6],[5,7]]     跨度为 TP=2 的两个 rank 一组
```

**为什么这个排布决定了"DP=0,TP=0 的 rank 覆盖完整模型"**：每个 PP stage 只装了模型的一部分层，一个 PP stage 内如果把 TP 维度 all-gather 完，就拿到了这一 stage 的完整层参数——只要**恰好取每个 PP stage 里 TP=0、DP=0 的那一个 rank**，横着扫完所有 PP stage，正好覆盖模型全部层、且不重复。这正是 04 篇 §5.6 里"NCCL 广播组数 = PP stage 数，每次发送方是该 stage 内 TP=0/DP=0 的 rank"这个结论的严格数学依据。

如果改变 `order`（比如把 `pp` 挪到最前变成 `pp-tp-cp-ep-dp`），rank 编号与物理拓扑的对应关系会完全不同（相邻 rank 变成了不同 PP stage 而非同 TP 组），这也是为什么"改 order 会让所有基于 rank 算术的假设失效"——slime 完全依赖 Megatron 的默认 order 语义，从不自己重新计算 rank 分组，这正是"原生透传"哲学在通信层的体现。

## 3. `get_megatron_optimizer` 与 DistributedOptimizer

`megatron/core/optimizer/__init__.py:989`。要点：

- **DistributedOptimizer**：把优化器状态（Adam 的 m/v）按 DP 组分片——每个 DP rank 只存 `1/dp_size` 的优化器状态和自己的参数分片；step 时先 all-gather 参数再各自更新。这是 ZeRO-1 式省显存；
- 参数分桶与 grad buffer 配合：反向完成的梯度先进连续 buffer，reduce 完后 optimizer 只对自己那一片做更新；
- **`init_state_fn`**：分布式优化器惰性初始化状态的钩子。slime 的 stateless Adam（06 篇 §2.2）正是**置空这个钩子**（`_disable_distributed_optimizer_state_initialization`）——不分配 m/v，每步临时算，优化器显存接近归零。理解了 DistributedOptimizer 的正常路径，就明白这个 hack 省掉的是什么。

**一个数字例子**：7B 模型（约 70 亿参数），标准 Adam 每个参数需要 `4(参数FP32主权重)+4(m)+4(v)=12` 字节的优化器状态，共约 84GB；如果 `DP=8`，`DistributedOptimizer` 把这 84GB 均分成 8 份，每个 rank 只需要存 `84/8≈10.5GB`——这就是为什么 RL colocate 场景（训练要跟推理共享显存）几乎总是默认开 `--use-distributed-optimizer`。而 slime 的 `--use-stateless-adam` 更进一步：连这 10.5GB 都不要了，用"当前梯度即时算出等效更新量"replace 掉"维护跨步矩估计"——两者可以叠加使用（stateless 时 `DistributedOptimizer` 仍负责参数分片和 all-gather，只是不再分配 m/v 存储）。

## 4. 配置与 checkpoint

- **`core_transformer_config_from_args`**（`megatron/training/arguments.py`）：把命令行 args 映射成 `TransformerConfig`（`megatron/core/transformer/transformer_config.py`）——slime 原生 GPTModel 路径用；bridge 路径则由 Megatron-Bridge 生成 provider（13 篇）；
- **checkpointing**（`megatron/training/checkpointing.py`）：dist-ckpt 格式（每个 rank 存自己分片 + 全局元数据），目录下 `latest_checkpointed_iteration.txt` 记录最新 iteration——slime 的 `checkpoint.py:97-152` 靠这个文件判断"是 Megatron ckpt 还是 HF 目录"（06 篇 §3）。

## 5. MoE 与流水线调度（训练一步的执行面）

- **MoE**（`megatron/core/transformer/moe/`）：router（topk 打分 + 分组路由，对应 `moe_router_topk / moe_router_num_groups`）→ token dispatcher（默认 alltoall：各 EP rank 交换 token，算完再换回）。routing replay（06 篇 §5）就是在 router 决策处注入回放数据；
- **流水线调度**（`megatron/core/pipeline_parallel/schedules.py`）：`forward_backward_no_pipelining`（672）/ `forward_backward_pipelining_without_interleaving`（2127）/ `forward_backward_pipelining_with_interleaving`（984）。它们内部按 `num_microbatches` 循环 micro-batch 做 1F1B（one-forward-one-backward）调度——**这就是 Megatron"梯度累积"的所在**：slime 传 `num_microbatches`，调度器自动把 loss 平均到每个 micro-batch（05 篇 §6 的缩放对账点）；
- `get_forward_backward_func()` 按配置选择上面三者之一返回。

### 5.1 深入拆解：1F1B 调度到底在"调度"什么

三个调度函数虽然名字不同，处理的都是同一个问题：**一个 PP stage 在一步训练里要跑多个 micro-batch 的前向和反向，前向和反向的顺序如何安排才能让"流水线气泡"（GPU 空等）最小**。

- `forward_backward_no_pipelining`（PP=1）：没有流水线问题，`num_microbatches` 个 micro-batch **依次**做完整的 forward+backward，梯度在 DDP 层累积（不立刻更新参数），最后统一 `optimizer.step()`；
- `forward_backward_pipelining_without_interleaving`（PP>1，无 VPP）：经典 1F1B——启动阶段（warmup）先做若干个纯前向（数量等于"到最后一个 stage 的距离"），进入稳态后**每做一次前向立刻做一次前一个 micro-batch 的反向**（这就是"1F1B"的字面意思），让前向和反向交替把流水线填满，尾部再排空剩余反向；
- `forward_backward_pipelining_with_interleaving`（PP>1，开 VPP）：每个 rank 持有多个 model chunk，调度顺序在 chunk 之间也要交织，进一步缩短气泡但调度逻辑更复杂。

**为什么这对 RL 训练重要**：`05_rl_algorithms.md` 里 loss 的归一化必须与"梯度累积多少个 micro-batch"对齐——1F1B 调度器在内部隐式做了这件事（它知道自己总共处理了多少个 micro-batch），slime 只需要保证自己传的 `num_microbatches` 与实际切分的数据量一致，剩下的累积、平均、参数更新时机全部交给调度器，不需要在 slime 代码里重新实现。

---

## 6. 小结

- `get_model` = PP/VPP 切层 + TP 属性补全 + 精度包装 + LocalDDP 分桶；梯度缩放与 expert/非 expert 分组归约是 05 篇 loss 缩放的对账基础；
- `parallel_state` 的确定性 rank 排布是 slime"DP0/TP0 当权重源"等约定的前提；
- DistributedOptimizer 的 `init_state_fn` 钩子是 stateless Adam 的落点；
- 三种流水线调度函数的核心是安排 micro-batch 前向/反向的执行顺序以压缩气泡，同时天然承担了"梯度累积"的职责；
- 下一篇（13）看 Megatron-Bridge 如何在 HF 与 Megatron 之间做双向转换。
