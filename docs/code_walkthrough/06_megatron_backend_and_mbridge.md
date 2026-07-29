# 06 Megatron 训练后端与 HF↔Megatron 格式转换

> 对应综述（`00_rl_infra_survey.md`）§2.7「训练后端」。
> slime 的训练侧只有 Megatron 一路后端，设计哲学是"**原生透传**"：不包一层自己的抽象，直接复用 Megatron 的 `get_model`、optimizer、checkpoint，并把 Megatron 参数原样传入。本篇解读这条链路。

---

## 1. 训练 actor 的初始化

每个训练 rank 是一个 `MegatronTrainRayActor`（`slime/backends/megatron_utils/actor.py:51`），其 `init`（actor.py:53-201）做：

1. 分布式初始化（`slime/backends/megatron_utils/initialize.py`）：`torch.distributed`、Megatron 的 `mpu`（TP/PP/DP/EP/CP 通信组）、种子、tokenizer；
2. **模型与优化器构建**（`initialize_model_and_optimizer`，`slime/backends/megatron_utils/model.py:968-1007`）；
3. checkpoint 加载；
4. 按角色创建附属模型：`with_ref`（KL 参考模型）、`with_opd_teacher`（Megatron 版 teacher）、`keep_old_actor`（保留旧策略用于 off-policy 场景，见 actor.py:614-623 的"队列式更新"：`rollout_actor → old_actor, actor → rollout_actor`）；
5. 构建 `weight_updater`（04 篇）。

值得注意：**一个进程内可以有多个模型"tag"**（actor / ref / old_actor / rollout_actor / teacher），`_switch_model(target_tag)`（actor.py:291）在它们之间切换——ref 前向、old_actor 备份等操作共用同一套分布式上下文。

---

## 2. 复用 Megatron 的模型构建

### 2.1 `setup_model_and_optimizer`（model.py:270-318）

核心三行逻辑：

1. **`get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)`**（model.py:294）——直接调 `megatron.training.training.get_model`。slime 只提供一个 `model_provider` 函数，Megatron 负责按 TP/PP 把模型切好、包装 DDP；
2. **OptimizerConfig 透传**（model.py:297-301）：用 `dataclasses.fields(OptimizerConfig)` 枚举 Megatron 优化器配置的所有字段，把 args 里的同名值拷过去——slime 不需要为每个 Megatron 新参数加适配代码；
3. `get_megatron_optimizer` 建优化器。

**LR scheduler 的步数估算**（model.py:182-235）：Megatron 按 iteration 调度学习率，而 RL 的"一步"是 rollout——slime 用 `num_rollout × rollout_batch_size × n_samples_per_prompt / global_batch_size` 换算出 `train_iters`（model.py:204），让 cosine/linear/WSD 调度在 RL 语义下正确展开。

### 2.2 Stateless Adam（省显存技巧）

`--use-stateless-adam` 时（model.py:242-267），`_patch_megatron_adam` 临时把 `megatron.core.optimizer.Adam` 替换为 `StatelessAdam`（`stateless_adam.py`），并用 `_disable_distributed_optimizer_state_initialization` 置空 `DistributedOptimizer.init_state_fn`——**不分配 Adam 的一阶/二阶 moment**，每步临时计算。代价是每步多一点计算，换来优化器状态显存接近归零（colocate 显存紧张时的救命特性）。

### 2.3 critic 的 value head

`model_provider.py` 在 `role == "critic"` 时给 GPTModel 加输出维度为 1 的 value head。从纯 LM checkpoint 加载时 value head 形状对不上，`_critic_output_layer_needs_reinit`（model.py:125-168）检测后重新以 `N(0, init_method_std)` 初始化（model.py:171-179）。

---

## 3. 三条模型构建路径

`_get_model_provider_func`（model_provider.py:61-242）：

| 路径 | 触发 | 说明 |
|---|---|---|
| 自定义 | `--custom-model-provider-path` | 用户全控 |
| **bridge** | `--megatron-to-hf-mode bridge`（默认推荐） | 用 `megatron.bridge.AutoBridge.from_hf_pretrained(...)` 从 HF 配置生成 Megatron provider，再把 slime args 里的并行参数（TP/PP/EP/SP/CP）拷到 provider（model_provider.py:95-107） |
| 原生 GPTModel | 传统路径 | `core_transformer_config_from_args(args)` 生成 `TransformerConfig`，按 `num_experts`/`transformer_impl` 选 layer spec，支持 MTP |

**bridge 模式的意义**：HF 上新发布的模型（新架构、新配置项）可以 day-0 被 Megatron 训练，不必等人工写 Megatron 版模型定义。checkpoint 加载同理（`slime/backends/megatron_utils/checkpoint.py:97-152`）：路径是 Megatron ckpt（有 `latest_checkpointed_iteration.txt`）走原生加载，否则用 `AutoBridge.load_hf_weights` 把 HF 权重灌进 Megatron 模型。

---

## 4. 权重转换层：`megatron_to_hf/`

训练侧存的是 Megatron 分片格式，推理引擎要 HF 格式——04 篇的 `convert_to_hf` 就来自这里：

- **每模型族一个转换模块**：`megatron_to_hf/deepseekv3.py`、`qwen3moe.py`、`glm4moe.py` 等，负责命名映射（`module.decoder.layers.N.self_attention.linear_qkv` → `q/k/v_proj`）、融合算子拆分（QKV、gate-up）、MoE expert 重排；
- **processors/**：`padding_remover`（去 TP padding）、`quantizer_fp8`（BF16→FP8 cast，04 篇）、`quantizer_compressed_tensors`（int4/fp4）；
- **`hf_weight_iterator_base/bridge/direct.py`**：`update_weight_from_tensor`（IPC 模式）与 disk 模式按 HF 语义**逐参数**迭代训练侧权重时的三种实现——bridge 模式走 Megatron-Bridge 的转换 API，direct 模式走 `megatron_to_hf` 的本地转换。

这一层是 RL 特有的高频需求催生的：SFT 只需"启动时转一次"，RL 是"每步都要转"，所以转换必须与权重同步流水线深度集成（gather→转换→分桶，见 04 篇 §2.2）。

---

## 5. 训练一步的内部

`train_actor`（actor.py:414-542）的大致流程：

1. **（可选）routing replay 填充**（`fill_routing_replay`，actor.py:297-345）：MoE 模型把 rollout 时记录的 `rollout_routed_experts` 灌进 forward，强制训练侧走相同专家路由——消除训推 router 数值差异引起的 logprob 不一致（MoE RL 稳定性的关键技巧，`ENABLE_ROUTING_REPLAY=1`，actor_group.py:96-97）；
2. **多轮前向**：对数据迭代器的每个 micro-batch 跑 forward（packing 后的序列），`loss_function` 计算（05 篇）；
3. **ref/old 模型的 logprob**：需要 KL 时先用 ref tag 前向一遍存 `ref_log_probs`（`compute_log_prob`，actor.py:346）；
4. Megatron 的 `optimizer.step()` 与 LR 调度；
5. 指标聚合返回。

`train_critic`（actor.py:386-414）同构，区别是 value head + `value_loss_function`，并把 values 通过 `external_data` 回传（critic 先训、actor 后训的编排见 01 篇 §2.1）。

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
| Megatron 接入 | 直接复用 `megatron.training` API + bridge | 经 mbridge/Pai-Megatron-Patch | 经 Megatron Bridge |
| 参数透传 | 全量原样 | 封装后映射 | 封装后映射 |
| 另一后端 | 无 | FSDP2 | FSDP2（AutoModel/DTensor） |
| checkpoint | Megatron 原生 + HF bridge | 封装格式 | HF/Megatron 双格式 |

slime 的取舍（README「轻量且有明确取舍」）：只深度优化 Megatron+SGLang 一路，把抽象层压到最薄——上游 Megatron 的并行、优化器、checkpoint 新特性可以零成本继承。

---

## 8. 小结

> 本篇讲的是 slime 的调用方式；Megatron-LM 内部（get_model/parallel_state/DistributedOptimizer/流水线调度）与 Megatron-Bridge 内部（AutoBridge/CONFIG_MAPPING/ParamMapping）的实现细节见 [11_megatron_and_bridge_internals.md](11_megatron_and_bridge_internals.md)。

- 训练 actor = Megatron 原生 `get_model/optimizer` + RL 所需的 ref/old/teacher 多 tag 管理；
- bridge 模式让 HF 新模型 day-0 可训；`megatron_to_hf` 让每步权重同步可转；
- routing replay、stateless Adam、critic head 重初始化是 RL 场景的三个特色增量。
- 下一篇（07）看如何在这些机制之上写自己的 agentic RL 应用。
