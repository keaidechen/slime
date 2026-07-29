# 11 训练侧内部实现：Megatron-LM 与 Megatron-Bridge

> 衔接 [06_megatron_backend_and_mbridge.md](06_megatron_backend_and_mbridge.md)。
> 06 篇讲 slime 如何调用 Megatron；本篇深入仓库根目录 vendored 的 `Megatron-LM/` 与 `Megatron-Bridge/` 源码，回答"slime 调的那几个入口背后发生了什么"。

---

## 第一部分：Megatron-LM 内部

slime 的全部 Megatron 调用集中在四个入口（`slime/backends/megatron_utils/model.py:270-318`）：`get_model` → `OptimizerConfig` → `get_megatron_optimizer` → `get_optimizer_param_scheduler`；训练时经 `get_forward_backward_func()` 拿调度函数。下面逐个拆。

### 1. `get_model`：模型是怎么被"切"出来的

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

### 2. `parallel_state.initialize_model_parallel`：通信组是怎么排的

`megatron/core/parallel_state.py:547`。给定 world size 和 TP/CP/EP/PP/DP 尺寸，按 **order**（默认 `tp-cp-ep-dp-pp`）把全局 rank 网格切成各维度的通信组。直觉：

```
world_size = TP × CP × EP × DP × PP
组内 rank 的选取 = 按 order 从高维到低维切分 rank 编号
```

例如 `TP=2, PP=2, DP=2`（world=8）：TP 组 = 相邻 2 个 rank（0,1）、(2,3)…；PP 组 = 跨最远的 rank（0,4）、(1,5)…。slime 04 篇的"只有 DP=0 且 TP=0 的 rank 当权重源"能成立，正是因为这套确定性排布保证了"每组取 rank 0 的成员恰好覆盖完整模型"。RL 侧常用的查询函数：`get_tensor_model_parallel_group / get_data_parallel_group(with_context_parallel=...) / get_expert_model_parallel_group / get_pipeline_model_parallel_group` 等。

### 3. `get_megatron_optimizer` 与 DistributedOptimizer

`megatron/core/optimizer/__init__.py:989`。要点：

- **DistributedOptimizer**：把优化器状态（Adam 的 m/v）按 DP 组分片——每个 DP rank 只存 `1/dp_size` 的优化器状态和自己的参数分片；step 时先 all-gather 参数再各自更新。这是 ZeRO-1 式省显存；
- 参数分桶与 grad buffer 配合：反向完成的梯度先进连续 buffer，reduce 完后 optimizer 只对自己那一片做更新；
- **`init_state_fn`**：分布式优化器惰性初始化状态的钩子。slime 的 stateless Adam（06 篇 §2.2）正是**置空这个钩子**（`_disable_distributed_optimizer_state_initialization`）——不分配 m/v，每步临时算，优化器显存接近归零。理解了 DistributedOptimizer 的正常路径，就明白这个 hack 省掉的是什么。

### 4. 配置与 checkpoint

- **`core_transformer_config_from_args`**（`megatron/training/arguments.py`）：把命令行 args 映射成 `TransformerConfig`（`megatron/core/transformer/transformer_config.py`）——slime 原生 GPTModel 路径用；bridge 路径则由 Megatron-Bridge 生成 provider（第二部分）；
- **checkpointing**（`megatron/training/checkpointing.py`）：dist-ckpt 格式（每个 rank 存自己分片 + 全局元数据），目录下 `latest_checkpointed_iteration.txt` 记录最新 iteration——slime 的 `checkpoint.py:97-152` 靠这个文件判断"是 Megatron ckpt 还是 HF 目录"（06 篇 §3）。

### 5. MoE 与流水线调度（训练一步的执行面）

- **MoE**（`megatron/core/transformer/moe/`）：router（topk 打分 + 分组路由，对应 `moe_router_topk / moe_router_num_groups`）→ token dispatcher（默认 alltoall：各 EP rank 交换 token，算完再换回）。routing replay（06 篇 §5）就是在 router 决策处注入回放数据；
- **流水线调度**（`megatron/core/pipeline_parallel/schedules.py`）：`forward_backward_no_pipelining`（672）/ `forward_backward_pipelining_without_interleaving`（2127）/ `forward_backward_pipelining_with_interleaving`（984）。它们内部按 `num_microbatches` 循环 micro-batch 做 1F1B（one-forward-one-backward）调度——**这就是 Megatron"梯度累积"的所在**：slime 传 `num_microbatches`，调度器自动把 loss 平均到每个 micro-batch（05 篇 §6 的缩放对账点）；
- `get_forward_backward_func()` 按配置选择上面三者之一返回。

---

## 第二部分：Megatron-Bridge 内部

slime 的 `--megatron-to-hf-mode bridge`（06 篇 §3）依赖它。源码在 `Megatron-Bridge/src/megatron/bridge/`。

### 6. 三层职责分离

`models/conversion/model_bridge.py:352-361` 的注释把架构讲得很清楚：

| 层 | 类 | 职责 |
|---|---|---|
| 编排 | `MegatronModelBridge`（model_bridge.py） | 建转换任务、进度、错误处理 |
| 映射表 | `MegatronMappingRegistry`（mapping_registry.py） | 参数名映射，通配符匹配（`*.layers.{i}.`） |
| 张量变换 | `MegatronParamMapping`（param_mapping.py:56） | 真正做拼接/拆分 + TP/PP/EP 分布式通信 |

### 7. `AutoBridge`（auto_bridge.py:229）

**`from_hf_pretrained`（auto_bridge.py:459 起）**：

1. **只读 config**（508 行）：线程安全地读 `config.json`，不下载权重；
2. **架构校验与路由**（`_validate_config`，1919-1995）：检查 `architectures` 以 `ForCausalLM` 等后缀结尾，解析出 dispatch key（优先 `auto_map` 类名），查注册表找到具体 bridge（如 `LlamaBridge/Qwen3Bridge/DeepseekV3Bridge/Glm4MoeBridge`——`models/` 下 55 个模型族 bridge 文件）；
3. **懒加载权重**：权重**不立即读入内存**，`PreTrainedCausalLM` + `SafeTensorsStateSource`（`models/hf_pretrained/`）按需从 safetensors 分片流式读取——大模型转换的内存友好关键；
4. 兼容处理：transformers 5.0+ 的 `rope_scaling` property 坑（512-526 行）。

**`to_megatron_provider`（auto_bridge.py:1638-1714）**：HF config → Megatron provider 的翻译：

- 核心是 **`CONFIG_MAPPING`**（model_bridge.py:437）：HF→Megatron 字段翻译表，例如
  - `num_hidden_layers→num_layers`、`intermediate_size→ffn_hidden_size`、`num_key_value_heads→num_query_groups`、`tie_word_embeddings→share_embeddings_and_output_weights`；
  - MoE：`num_local_experts/n_routed_experts→num_moe_experts`、`num_experts_per_tok→moe_router_topk`、`n_group→moe_router_num_groups`；
  - MLA：`q_lora_rank/kv_lora_rank→qk_head_dim/qk_pos_emb_head_dim`；MTP：`num_nextn_predict_layers→mtp_num_layers`；
- 产出 `GPTModelProvider`（MLA 模型为 `MLAModelProvider`）——它继承 `TransformerConfig` 全部字段，**同时也是一个 model provider**（可调用返回 GPTModel）。slime 的 `model_provider.py:95-107` 再往它身上拷 TP/PP/EP 等并行参数（06 篇 §3）；
- 细节：`make_vocab_size_divisible_by`（找 ≤128 的 2 的幂整除词表，供 TP padding）、YaRN rope scaling 的 GPT/MLA 两种字段风格。

**`load_hf_weights`（auto_bridge.py:573）/ `save_hf_weights`**：按映射表逐参数转换 + 分布式通信。`MegatronParamMapping` 的子类体系（param_mapping.py，20+ 种）覆盖了所有典型变换：

- `ReplicatedMapping`（param_mapping.py:1266）：各 TP rank 复制；
- **QKV 融合/拆分**：HF 的 `q/k/v_proj` ↔ Megatron 的 `linear_qkv`（拼接维度按 GQA 分组）；
- **gate_up 融合/拆分**：`gate_proj+up_proj` ↔ `linear_fc1`；
- **MoE**：HF 的 `experts.{i}.` ↔ Megatron 的 `experts.local_experts.{i}.`，并处理 EP 分片（每个 EP rank 只持有部分 expert）；
- TP 分布式原语：转换时按本 rank 的 TP/PP/EP 位置只处理自己的分片（或做必要的 gather/scatter）。

**量化**（`models/conversion/quantization_utils.py`）：FP8（ue8m0、blockwise scale）、MXFP4、INT4 的量化/反量化——与 slime 04 篇 §5 的 FP8 传输链衔接。

### 8. 与 slime 本地转换层的关系

| | slime `megatron_to_hf/` | Megatron-Bridge |
|---|---|---|
| 覆盖 | 每模型族一个手写精简模块（deepseekv3/qwen3moe/glm4moe…） | 通用注册表 + 55 个模型 bridge |
| 优化目标 | **每步权重同步**的热路径（与 gather/分桶流水线内联，04 篇 §2.2） | 正确性与覆盖率（启动加载/最终导出） |
| 在 slime 中的角色 | NCCL/IPC 权重同步时的逐参数转换 | `--megatron-to-hf-mode bridge` 时的模型构建与 HF 权重加载 |

即：**bridge 管"进"（HF→Megatron 建模型、加载权重），slime 本地层管"出"（Megatron→推理引擎的高频转换）**。两者在同一套命名约定上工作，互相验证。

---

## 9. 小结

- `get_model` = PP/VPP 切层 + TP 属性补全 + 精度包装 + LocalDDP 分桶；梯度缩放与 expert/非 expert 分组归约是 05 篇 loss 缩放的对账基础；
- `parallel_state` 的确定性 rank 排布是 slime"DP0/TP0 当权重源"等约定的前提；
- DistributedOptimizer 的 `init_state_fn` 钩子是 stateless Adam 的落点；
- Megatron-Bridge = 编排层 + 通配符映射表 + 20 余种张量变换/分布式原语；懒加载 safetensors 让大模型转换不占内存；
- slime 的取舍因此清晰：训练栈全复用 Megatron，转换热路径自己写精简版，加载/导出交给 Bridge。
