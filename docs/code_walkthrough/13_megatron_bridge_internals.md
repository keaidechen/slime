# 13 训练侧内部实现：Megatron-Bridge 篇

> 衔接 [06_megatron_backend_and_mbridge.md](06_megatron_backend_and_mbridge.md) 与 [12_megatron_lm_internals.md](12_megatron_lm_internals.md)。
> 12 篇讲 Megatron-LM 这个训练执行引擎本身；本篇讲 slime `--megatron-to-hf-mode bridge` 依赖的转换/建模层 Megatron-Bridge——它负责"HF 格式 ↔ Megatron 格式"的双向翻译，是与 Megatron-LM 完全不同的一套代码（源码在 `Megatron-Bridge/src/megatron/bridge/`），值得单独理解其设计。

---

## 1. 为什么需要一个独立的 Bridge 层

Megatron 原生只认自己的 `TransformerConfig` + 内部命名（`self_attention.linear_qkv` 等），HuggingFace 生态的模型全部是 `AutoModelForCausalLM` + `config.json` + safetensors，两套命名/切分方式完全不同（HF 的 `q_proj/k_proj/v_proj` 三个独立权重 vs Megatron 融合成一个 `linear_qkv`；HF 无并行概念 vs Megatron 权重按 TP/PP/EP 切片存储）。RL 训练又需要频繁在两种格式间转换（读 HF 预训练权重初始化、导出 HF 格式做评测/部署），转换逻辑如果散落在各处会随模型族数量爆炸——Megatron-Bridge 把这件事收敛成一套**注册表驱动**的通用框架，一次实现新模型族的映射规则，之后加载/导出全自动复用。

## 2. 三层职责分离

`models/conversion/model_bridge.py:352-361` 的注释把架构讲得很清楚：

| 层 | 类 | 职责 |
|---|---|---|
| 编排 | `MegatronModelBridge`（model_bridge.py） | 建转换任务、进度、错误处理 |
| 映射表 | `MegatronMappingRegistry`（mapping_registry.py） | 参数名映射，通配符匹配（`*.layers.{i}.`） |
| 张量变换 | `MegatronParamMapping`（param_mapping.py:56） | 真正做拼接/拆分 + TP/PP/EP 分布式通信 |

这个三层拆分本身是一个值得学习的设计模式：**"知道转换什么"（映射表）与"知道怎么转换"（张量变换）与"负责流程调度"（编排）分离**——新增一个模型族只需要写一份映射表 + 复用已有的张量变换类（QKV融合、gate_up融合等变换是通用的，绝大多数 transformer 模型族都能复用），而不需要重新实现整套转换逻辑。

## 3. `AutoBridge`（auto_bridge.py:229）

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

### 3.1 深入拆解：一个具体例子——Qwen3 的 QKV 从 HF 到 Megatron 怎么拼

假设 Qwen3（GQA，`num_attention_heads=32`，`num_key_value_heads=8`，`head_dim=128`）从 HF checkpoint 加载：HF 侧是三个独立张量 `q_proj.weight [4096, hidden]`、`k_proj.weight [1024, hidden]`、`v_proj.weight [1024, hidden]`（`1024 = 8×128`）。Megatron 的 `linear_qkv` 需要把它们**按 GQA 分组交织**拼成一个大张量——不是简单地 `cat([q, k, v])`，而是按"每个 KV head 对应 4 个 Q head（32/8=4）"的分组规律，把 `[q0,q1,q2,q3,k0,v0]`、`[q4,q5,q6,q7,k1,v1]`……这样交织排列（Megatron 内部实现按 GQA repeat 的方式组织 QKV 存储布局，以便前向时能连续切片取出各注意力头）。对应的 `QKVMapping` 类知道这套排列规则，加载时按此重排 HF 权重，导出时执行逆操作还原成 HF 的三个独立张量。

如果模型是 `TP=2`，这个融合后的 `linear_qkv` 张量还要在**注意力头维度**上再切一刀分给两个 TP rank（每个 rank 拿 16 个 Q head + 4 个 KV head 对应的权重行），映射类需要同时处理"HF→Megatron 命名/融合"和"按当前 rank 的 TP 位置切片"两件事——这也是为什么"张量变换"要单独抽出一层类体系，而不是写成一次性脚本：融合逻辑和并行切片逻辑必须能够独立组合（同一套融合规则要能在 TP=1/2/4/8 等不同并行度下正确工作）。

## 4. 与 slime 本地转换层的关系

| | slime `megatron_to_hf/` | Megatron-Bridge |
|---|---|---|
| 覆盖 | 每模型族一个手写精简模块（deepseekv3/qwen3moe/glm4moe…） | 通用注册表 + 55 个模型 bridge |
| 优化目标 | **每步权重同步**的热路径（与 gather/分桶流水线内联，04 篇 §2.2） | 正确性与覆盖率（启动加载/最终导出） |
| 在 slime 中的角色 | NCCL/IPC 权重同步时的逐参数转换 | `--megatron-to-hf-mode bridge` 时的模型构建与 HF 权重加载 |

即：**bridge 管"进"（HF→Megatron 建模型、加载权重），slime 本地层管"出"（Megatron→推理引擎的高频转换）**。两者在同一套命名约定上工作，互相验证。

**为什么 slime 不干脆全用 Megatron-Bridge 做权重同步**：权重同步是**每一轮 rollout 都要执行一次**的热路径，追求的是"这一批已知模型族、已知并行配置下尽可能快"，可以把转换规则和分桶/NCCL 广播流水线写在一起做深度优化（04 篇）；而 Bridge 追求的是"任意 HF 模型、任意并行配置都要转换正确"，必然要为通用性付出一些性能开销（通配符匹配、注册表查找、更保守的通信模式）。两者的取舍方向不同，所以 slime 选择"启动/评测用 Bridge 保正确性，热路径用手写精简模块保性能"。

---

## 5. 小结

- Megatron-Bridge = 编排层（`MegatronModelBridge`）+ 通配符映射表（`MegatronMappingRegistry`）+ 20 余种张量变换/分布式原语（`MegatronParamMapping` 子类）；
- `AutoBridge.from_hf_pretrained` 三步：只读 config → 架构路由到具体模型 bridge → 懒加载 safetensors，大模型转换不占内存；
- `CONFIG_MAPPING` 是 HF↔Megatron 字段翻译的核心表，覆盖普通/MoE/MLA/MTP 各类字段；
- QKV/gate_up 融合与 TP/PP/EP 切片是两件必须能独立组合的事，这是张量变换类体系存在的原因；
- slime 的取舍清晰：训练栈全复用 Megatron-LM（12 篇），转换热路径自己写精简版，加载/导出交给 Bridge（本篇）。
