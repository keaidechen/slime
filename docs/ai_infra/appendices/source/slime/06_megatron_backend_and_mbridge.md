# 06 Megatron 训练后端与 HF↔Megatron 格式转换

相关基础：[Ray、队列与前置系统](<../../../06_Post_Training/27_权重发布异步与资源调度.md#s4>)。


> 对应综述（`00_rl_infra_survey.md`）§2.7「训练后端」。
> slime 的训练侧只有 Megatron 一路后端，设计哲学是"**原生透传**"：不包一层自己的抽象，直接复用 Megatron 的 `get_model`、optimizer、checkpoint，并把 Megatron 参数原样传入。本篇解读这条链路。
>
> 文件名中的 `mbridge` 是历史遗留：slime v0.3.1 已删除 Megatron-Bridge 依赖、`--megatron-to-hf-mode bridge` 和相关 iterator。当前代码使用原生 Megatron model provider，并由 slime 内建转换器负责 HF checkpoint 加载与导出。

---

### 从一次单卡更新理解后端的职责

模型 forward（前向）把 tokens 变成 logits，loss 把它们变成一个标量，backward（反向）计算每个参数的梯度，optimizer（优化器）据此更新参数。分布式后端需要把这四步分给多进程，同时保持等价的数学目标。slime 负责提供轨迹与 RL loss，Megatron 负责模型执行、通信和优化器调度；两者不是各训练一遍模型。

最小例子是 4 条轨迹、2 个 DP 副本、每副本一次能处理 1 条：副本 0 处理 `[a,b]`，副本 1 处理 `[c,d]`，每个副本做 2 个 micro-batch 的前反向，再同步梯度并更新一次。若每个副本还使用 TP=2，就需要 4 张卡；同一副本的两个 TP rank 协作计算 a 或 c，而不是额外增加两条样本。

在固定样本数、无 padding 且可整除时，梯度累积次数为

$$
N_{micro}=\frac{B_{global}}{D\,B_{micro}},
$$

其中 $B_{global}$ 是一次参数更新的样本数，D 是 DP 副本数，$B_{micro}$ 是每副本每次前反向的样本数。动态 token packing 下各 micro-batch 的样本数不同，这个固定公式就应让位于 `build_dp_schedule` 产生的实际计划。

SP（sequence parallelism，序列并行）在 TP 协作中分摊部分激活；CP（context parallelism，上下文并行）切长序列的上下文计算。它们都不能当作额外的 DP 副本计入独立样本数。

## 1. 训练 actor 的初始化

每个训练 rank 是一个 `MegatronTrainRayActor`（`slime/backends/megatron_utils/actor.py:51`），其 `init`（actor.py:53-201）做：

1. 分布式初始化（`slime/backends/megatron_utils/initialize.py`）：`torch.distributed`、Megatron 的 `mpu`（TP/PP/DP/EP/CP 通信组）、种子、tokenizer；
2. **模型与优化器构建**（`initialize_model_and_optimizer`，`slime/backends/megatron_utils/model.py:968-1007`）；
3. checkpoint 加载；
4. 按角色创建附属模型：`with_ref`（KL 参考模型）、`with_opd_teacher`（Megatron 版 teacher）、`keep_old_actor`（保留旧策略用于 off-policy 场景，见 actor.py:614-623 的"队列式更新"：`rollout_actor → old_actor, actor → rollout_actor`）；
5. 构建 `weight_updater`（04 篇）。

值得注意：**一个进程内可以有多个模型"tag"**（actor / ref / old_actor / rollout_actor / teacher），`_switch_model(target_tag)`（actor.py:291）在它们之间切换——ref 前向、old_actor 备份等操作共用同一套分布式上下文。

### 1.1 深入拆解：`TensorBackuper` ——用"pinned CPU 影子权重"模拟多个模型，而不是真的多份 GPU 权重

pinned memory（页锁定主机内存）在注册期间保持物理页稳定，可供某些异步 DMA 路径使用。它消耗主机资源，是否真正与计算重叠还取决于设备、stream 依赖和缓冲区生命周期。

如果给 actor/ref/old_actor 各开一份独立的 GPU 权重，显存直接翻 3 倍——colocate 场景下这是不可接受的。slime 的解法在 `slime/utils/tensor_backper.py`：**永远只有一份"活跃"的 GPU 权重，其余 tag 的权重被搬到 pinned CPU 内存里当"备份"，切换时整段拷回 GPU 覆盖当前权重**。

```python
class _TensorBackuperNormal(TensorBackuper):
    def backup(self, tag):                      # 把当前 GPU 权重存成名为 tag 的备份
        for name, param in self._source_getter():
            backup_dict[name] = torch.empty_like(param, device="cpu", pin_memory=True)  # 首次分配 pinned 内存
            backup_dict[name].copy_(param.detach(), non_blocking=True)                  # 异步 D2H
        torch.cuda.synchronize()

    def restore(self, tag):                      # 把名为 tag 的备份权重拷回，覆盖当前 GPU 权重
        for name, param in self._source_getter():
            param.copy_(backup_dict[name], non_blocking=True)                           # 异步 H2D
        torch.cuda.synchronize()
```

DMA（Direct Memory Access，直接内存访问）让设备在 CPU 配置任务后自行搬运数据，减少 CPU 逐字节复制的工作；数据依赖和完成同步仍需显式管理。

**`pin_memory=True` 是关键**：普通 CPU 内存（page-able memory）做 D2H/H2D 拷贝时 CUDA 驱动要先把数据搬到一段临时的锁页缓冲区再传输，pinned memory 可减少中转并支持异步传输，但带宽仍受总线、NUMA、并发和拷贝大小限制，不能保证跑满或给出固定时延。

**`copy()` 方法**（`copy(src_tag, dst_tag)`）：不经过 GPU，直接在两份 CPU 备份之间互拷（例如把 `actor` 备份复制成 `old_actor` 备份的初值），用于"某个 tag 需要先有一份初始快照，之后再逐步被覆盖"的场景。

**`_TensorBackuperNoop` 是预留实现，不是当前 actor 的实际分支。** `TensorBackuper.create(source_getter, single_tag)` 在 `single_tag` 非空时才选它，并用廉价 checksum 验证“权重没有被改动”；但当前 `MegatronTrainRayActor.init` 明确传入 `single_tag=None`，因此总是创建 `_TensorBackuperNormal`。不要因为类存在，就推断 `kl_coef=0` 时运行中会自动跳过 CPU backup。

```python
def backup(self, tag):
    self._backup_hash_dict = _compute_hash_dict(dict(self._source_getter()))   # 只存哈希，不存数据
def restore(self, tag):
    assert _compute_hash_dict(...) == self._backup_hash_dict                    # 用哈希校验"权重确实没变过"
```

`_compute_hash_tensor` 故意写了注释 `# Not a real/good hash, but pretty fast`——把 tensor 按位重解释成 `uint32` 后求和，不是密码学哈希，会漏检某些碰撞；它只适合做开发期不变量检查，不能声称与真实数据备份具有相同安全性。当前 actor 不走这条分支，阅读它应当把它当成一个尚未启用的轻量优化接口。

一次 ref 评分的所有权顺序是：保存 actor 快照 → 把 ref 从 CPU 复制到当前 GPU 参数 → 等复制完成 → 无梯度 forward，保存 `ref_log_probs` → 恢复 actor → 再做可求导训练。ref 概率是小得多的 response 长度数组，不必为它常驻整套模型。

例如参数主体共 8 GiB，actor/ref/old 三份 CPU tag 就可能需要约 24 GiB 主机存储，GPU 活跃参数仍约 8 GiB；这不包括优化器、梯度和 workspace。省 GPU 容量的代价是 CPU 容量与切换时间。`non_blocking=True` 只表示允许异步提交，代码末尾的设备同步保证拷贝完成前不复用源/目标；不能在上一套参数的 backward 尚未消费它时覆盖参数。

### 1.2 深入拆解：`StatelessAdam`（`stateless_adam.py`）——数学上等价于什么？

`--use-stateless-adam` 的动机是省掉 Adam 一阶/二阶矩（`exp_avg`/`exp_avg_sq`）的显存（每参数两份 FP32 矩状态，共 8 字节；相对于每参数 2 字节的 BF16 权重是 4 倍，而非 2 倍）。但它不是"不用 Adam"，而是**每一步都假装 Adam 的历史矩全是 0，重新算一遍"如果只有当前这一个梯度，Adam 会怎么更新"**：

```python
group["step"] = 1                                   # 永远当作 Adam 的第 1 步
numerator_scale, denominator_scale = (1.0, 1.0) if bias_correction else (1-beta1, sqrt(1-beta2))
denom = grad.abs() * denominator_scale + eps         # 二阶矩的"第一步"估计就是 |grad| 本身
param.addcdiv_(grad, denom, value=-lr * numerator_scale)   # param -= lr * grad / (|grad| + eps)
```

当 `bias_correction=True`（默认）时，`denom ≈ |grad| + eps`，于是更新量约为 `-lr * grad / |grad| ≈ -lr * sign(grad)`——**这在数学上非常接近 signSGD**（只看梯度符号、不看梯度大小的更新规则），只是保留了 Adam 的 per-parameter eps 平滑和 weight decay 处理。这解释了它为什么被称为"stateless"：它不是 Adam 的近似简化版，而是**精确复现"矩估计每一步都被重置为 0"这一假设下 Adam 该做的更新**，代价是失去了 Adam 最有价值的"跨步矩估计带来的自适应学习率"能力，换来零优化器状态显存。

`load_state_dict` 里 `state_dict["state"] = {}`（清空 state）与 `_disable_distributed_optimizer_state_initialization`（把 Megatron `DistributedOptimizer.init_state_fn` 替成空函数）配合，确保 Megatron 分布式优化器框架**从不尝试为这个优化器分配矩状态的存储空间**——如果不做这一步，即使用了 `StatelessAdam`，Megatron 外层框架仍会按标准 Adam 的假设预分配状态张量，显存节省就落空了。

---

把 stateless 的公式从 Adam 第一步推出来。普通 Adam 的矩递推为

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

$g_t$ 是梯度，与参数同形状；$m_t$ 与梯度同单位，$v_t$ 是梯度平方的单位。若每步都重置历史，就有 $m=(1-\beta_1)g_t$、$v=(1-\beta_2)g_t^2$；再用 step=1 的偏差校正，$\hat m=m/(1-\beta_1)=g_t$、$\hat v=v/(1-\beta_2)=g_t^2$。学习率 $\eta$ 控制参数步长，$\epsilon$ 是防止分母为零的小正数，与分母中的梯度尺度相同，所以更新为

$$
\Delta\theta=-\eta\frac{g_t}{\sqrt{g_t^2}+\epsilon}
=-\eta\frac{g_t}{|g_t|+\epsilon}.
$$

取参数 `[1,1]`、梯度 `[2,-0.5]`、学习率 $\eta=0.1$、暂忽略 eps 和 weight decay，更新后为 `[0.9,1.1]`；普通 SGD 会得到 `[0.8,1.05]`。若下一步梯度翻转，stateless 立即跟着翻转，普通 Adam 的历史一阶矩会缓冲这种变化。它节省状态存储，但改变优化轨迹，不能以“省显存”替代质量评估。

## 2. 复用 Megatron 的模型构建

### 2.1 `setup_model_and_optimizer`（model.py:270-318）

核心三行逻辑：

1. **`get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)`**（model.py:294）——直接调 `megatron.training.training.get_model`。slime 只提供一个 `model_provider` 函数，Megatron 负责按 TP/PP 把模型切好、包装 DDP；
2. **OptimizerConfig 透传**（model.py:297-301）：用 `dataclasses.fields(OptimizerConfig)` 枚举 Megatron 优化器配置的所有字段，把 args 里的同名值拷过去——slime 不需要为每个 Megatron 新参数加适配代码；
3. `get_megatron_optimizer` 建优化器。

**LR scheduler 的步数估算**（model.py:182-235）：Megatron 按 iteration 调度学习率，而 RL 的"一步"是 rollout——slime 用 `num_rollout × rollout_batch_size × n_samples_per_prompt / global_batch_size` 换算出 `train_iters`（model.py:204），让 cosine/linear/WSD 调度在 RL 语义下正确展开。

用最小批次核对调度单位：计划 10 轮 rollout，每轮 2 个 prompt×2 个回答，共 40 条；若一次更新使用全局 batch=2，则预计 20 次 optimizer step。学习率 warmup 若写 2 个更新步，只覆盖第一轮的两次更新，不是前两轮 rollout。动态过滤通常会增加生成开销而保持接受 batch 大小；fan-out 或不完整批次则可能改变实际更新计数，最终应检查执行计划。

### 2.2 Stateless Adam（省显存技巧）

`--use-stateless-adam` 时（model.py:242-267），`_patch_megatron_adam` 临时把 `megatron.core.optimizer.Adam` 替换为 `StatelessAdam`（`stateless_adam.py`），并用 `_disable_distributed_optimizer_state_initialization` 置空 `DistributedOptimizer.init_state_fn`——**不分配 Adam 的一阶/二阶 moment**，每步临时计算。它改变了跨步矩估计和更新规则，不能视为与标准 Adam 等价的纯性能优化；数学语义见上文 §1.2。显存收益必须与训练质量一起验证。

### 2.3 critic 的 value head

`model_provider.py` 在 `role == "critic"` 时给 GPTModel 加输出维度为 1 的 value head。从纯 LM checkpoint 加载时 value head 形状对不上，`_critic_output_layer_needs_reinit`（model.py:125-168）检测后重新以 `N(0, init_method_std)` 初始化（model.py:171-179）。

---

## 3. 当前模型构建只有两条路径

`_get_model_provider_func`（`slime/backends/megatron_utils/model_provider.py`）按语义优先级选择：

| 路径 | 触发 | 谁负责模型结构 |
|---|---|---|
| 自定义 provider | `--custom-model-provider-path` | 用户函数返回 Megatron `GPTModel` 兼容对象；slime 只补 critic head 与冻结规则 |
| 原生 Megatron provider | 默认 | `core_transformer_config_from_args(args)` 构造 `TransformerConfig`，再按 `--spec`、MoE、Transformer Engine、MLA、MTP 等参数选择 layer/block spec 并实例化 `GPTModel` |

这里没有“HF config 自动生成 Megatron 模型结构”的 bridge 分支。**支持某个新模型需要同时满足两个条件**：

1. Megatron 侧能按当前参数/`--spec` 构造出正确结构；
2. slime 的内建 checkpoint loader 和 exporter 有该模型族的参数映射。

因此“HF 上一发布新模型，slime 无需任何适配即可 day-0 训练”并不成立。自定义 provider 解决结构扩展，`hf_to_megatron/` 与 `megatron_to_hf/` 解决权重扩展；两条边界要分别实现和测试。

---

## 4. 为什么需要两个方向的内建转换器

### 4.1 启动/恢复：`hf_to_megatron/`

`checkpoint.load_checkpoint` 先用 `_is_megatron_checkpoint` 判断目录：存在 `latest_checkpointed_iteration.txt`（或路径本身是 `iter_XXXXXXX`）时，直接调用 Megatron 原生 loader；否则走 `_load_checkpoint_hf` → `hf_to_megatron.load_hf_weights`。

HF 加载链路是：

```text
AutoConfig.model_type
  -> _LOADERS 选择模型族 get_hf_tensor
  -> SafetensorReader 按参数名从 shard 读取 CPU tensor
  -> QKV / gate-up / MoE 等布局合并
  -> shard_mcore_tensor 按 Megatron 参数属性切 TP/ETP shard
  -> copy_ 到当前 rank 的 parameter/buffer
```

RNG（Random Number Generator，随机数生成器）具有随抽样推进的内部状态。仅保存初始 seed 不能恢复已推进的位置，重计算和恢复训练通常还需保留状态及对应的消费顺序。

`SafetensorReader.get_tensor` 的 `lru_cache(maxsize=1)` 缓存的是最近一次按名称返回的 **tensor 结果**，不是 shard handle。访问过的 `safe_open` 对象会保存在 `self._files` 中，以便同一 shard 后续复用；它依靠 safetensors 的 mmap/lazy read 避免一次性把所有 shard 内容读入内存，但当前实现并不把“已打开文件数”限制为 1。FP8 checkpoint 如果带 `<name>_scale_inv`，读取时会按 128×128 block 反量化为 BF16。词表 embedding/output layer 会按 `padded_vocab_size` 补齐。HF checkpoint 不包含 Megatron optimizer、RNG 和训练迭代状态，所以 loader 返回 iteration 0，参数校验阶段也会设置 `no_load_optim/no_load_rng/finetune`。

当前 `_LOADERS` 显式注册 DeepSeek、GLM、Llama/Qwen、Qwen3.5/Next、MiMo、MiniMax 等 `model_type`。未注册类型会抛 `Unsupported HuggingFace model type`，这是刻意的 fail-fast，而不是静默猜映射。

### 4.2 每轮热更新/导出：`megatron_to_hf/`

反方向不能简单复用上述函数，因为输入不再是一个完整 HF 文件，而是**分散在 PP/TP/EP rank 上、可能还位于 CPU backup 的 Megatron 参数**。`HfWeightIteratorDirect` 负责把分布式状态收敛成稳定的 chunk 接口：

```text
收集全局 ParamInfo 并按 --update-weight-buffer-size 分桶
  -> 从 actor CPU backup 取本 rank 参数
  -> PP/EP broadcast + TP/ETP all-gather 得到完整 Megatron tensor
  -> convert_to_hf 按模型族拆 QKV/gate-up、改名、去 padding、可选量化
  -> yield list[(hf_name, tensor)]
```

direct iterator 被 tensor/文件两类消费者复用：

- `UpdateWeightFromTensor`：每轮生成 chunk；同机 engine 走 CUDA IPC，混合拓扑中的远端 engine 仍可走 NCCL；
- `UpdateWeightFromDisk`：通过 HF checkpoint writer 消费同一 iterator，写完整 checkpoint 后再通知 engine 重载；
- `hf_checkpoint_saver.save_hf_model_direct_to_path`：写 safetensors shard 和 index，作为真正的 HF 导出。

`UpdateWeightFromDistributed` 是另一条专用流式路径：它自己遍历参数，普通参数逐个 TP gather，expert 参数先按 buffer 聚合再做 EP gather，随后调用同一个 `convert_to_hf` 并通过 NCCL 广播。`UpdateWeightFromDiskDelta` 继承的也是这套 iterator，而不是 `HfWeightIteratorDirect`。

`HfWeightIteratorBase.create` 当前只在 tensor 路径创建 `HfWeightIteratorDirect`，不存在 bridge/direct 多实现选择。两条路径共享模型族转换器和 HF 命名契约，但分布式 gather 的所有权不同：colocate/hybrid 与完整文件导出使用 direct iterator；专用 NCCL/disk-delta 路径在 updater 内流式 gather。

### 4.3 为什么两个方向不是一个“自动可逆”的函数

- HF→Megatron 在**加载时**按目标 parameter 的 `tensor_model_parallel/partition_dim/partition_stride` 属性切片，目标结构已经存在；
- Megatron→HF 在**热路径**必须先跨 PP/TP/EP 找齐来源、处理重复参数和 MTP virtual PP，再做拆分与量化；
- 某些转换有额外状态，例如 Q-LoRA 的成对参数可能跨 chunk 才能一起发给 SGLang；HF 保存还要管理 shard/index/资产文件。

所以两边共享的是模型布局知识，不共享相同的 I/O 和分布式控制流。强行包成一个“万能 converter”会隐藏所有权，反而更难验证。

---

## 5. 训练一步的内部

当前 `train_actor` 的依赖顺序为：

1. 若启用 routing replay（专家路由回放），载入 rollout 记录的专家选择，让训练按所记录的路由执行。这减少路由不同造成的评分偏差，不消除精度、上下文或采样处理的其它差异。
2. 在需要构造优势时，按配置切到 ref/teacher tag，分别无梯度评分并保存概率，再恢复 actor tag；这些结果必须先于使用它们的优势或正则计算就绪。
3. 按配置缓存更新前的 actor 评分，或在满足 `can_reuse_log_probs_in_loss` 时复用训练 forward 的 detached 评分；PPO 路径还接收 critic 的 values。不能认为每种配置必定多算同样次数的 forward。
4. 构造 advantages/returns，执行数据后处理与指标记录，之后才调用 Megatron `train`，按计划完成各 micro-batch 的可求导 forward、loss、backward 与梯度累积。
5. 到一个全局 batch 的更新边界再调用优化器并推进学习率。训练结束后推进 profiler；外层训练方法保存更新后的 actor CPU 备份，供换权使用。

`train_critic` 先无梯度计算当前 values，供 value loss 的旧值和 actor 优势使用，再构造自己的 returns 并训练价值模型。返回 actor 的是这批**更新前已算好的 values**，不是训练后重新计算的一批值；相关编排见 01 篇 §2.1。

---

## 6. 参数透传机制

三层参数（01 篇 §6）的实现：

- **Megatron 参数**：`slime/utils/arguments.py` 先解析 slime 自有参数，剩余未知参数交给 `slime/backends/megatron_utils/arguments.py` 按 Megatron 的参数定义解析校验——所以 Megatron 新加的参数通常无需改动即可用；
- **SGLang 参数**：以 `--sglang-` 前缀收集，启动 server 时剥前缀透传（`slime/backends/sglang_utils/arguments.py`）；
- **分角色覆盖**：`--megatron-config-path` 指向 YAML，可为 actor/critic 设不同的 Megatron 配置（`parse_megatron_role_args`，01 篇 placement_group.py:165-168）——例如 critic 用更小的并行度。

---

## 7. 与其他框架的对照

| 维度 | slime | verl | NeMo-RL |
|---|---|---|---|
| Megatron 接入 | 直接复用 `megatron.training` API + 原生/自定义 provider | 经 mbridge/Pai-Megatron-Patch | 经 Megatron Bridge |
| 参数透传 | 全量原样 | 封装后映射 | 封装后映射 |
| 另一后端 | 无 | FSDP2 | FSDP2（AutoModel/DTensor） |
| checkpoint | Megatron 原生 + slime 内建 HF 双向转换 | 封装格式 | HF/Megatron 双格式 |

slime 的取舍（README「轻量且有明确取舍」）：只深度优化 Megatron+SGLang 一路，把抽象层压到最薄——复用上游实现可减少重复开发，但新特性仍需验证版本、模型和 RL 数据路径的兼容性。

---

## 8. 小结

> 本篇讲的是 slime 的调用方式；Megatron-LM 内部分别见[训练主线](<../megatron/01_foundations/02_training_mainline.md>)、[并行状态与 TP](<../megatron/02_parallelism/01_parallel_state_and_tensor_parallel.md>)、[分布式优化器](<../megatron/03_state_and_memory/01_data_parallel_optimizer_checkpoint.md>)和[流水线调度](<../megatron/02_parallelism/02_pipeline_parallel.md>)。slime 内建转换器的模型映射、分片和导出细节见 [13_megatron_bridge_internals.md](<13_megatron_bridge_internals.md>)。

- 训练 actor = Megatron 原生 `get_model/optimizer` + RL 所需的 ref/old/teacher 多 tag 管理；
- 模型结构由原生/自定义 Megatron provider 构造；`hf_to_megatron` 负责启动加载，`megatron_to_hf` 负责每步权重同步和 HF 导出；
- routing replay、stateless Adam、critic head 重初始化是 RL 场景的三个特色增量。
- 相关资料（07）看如何在这些机制之上写自己的 agentic RL 应用。
