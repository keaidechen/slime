# 03 数据流：Sample、Data Buffer、Partial Rollout 与异步

> 对应综述（`00_rl_infra_survey.md`）§2.4「长尾治理」与 §2.5「异步与 off-policy」。
> RL 系统的数据不是静态数据集，而是"由模型实时产生、带状态、可续写"的样本流。本篇解读 slime 的数据抽象与异步设计。

---

## 1. 核心数据结构：`Sample`

`Sample`（`slime/utils/types.py:93-244`）是贯穿 rollout→训练全链路的"一等公民"。逐组字段解读：

### 1.1 身份与归属

```97:106:slime/utils/types.py
    group_index: int | None = None
    index: int | None = None
    # Id of the rollout this sample came from. ...
    rollout_id: int | None = None
```

- `index`：全局递增的样本编号（排序、确定性都靠它）；
- `group_index`：GRPO 的"组"编号——同一 prompt 的 N 个采样共享一个 `group_index`；
- `rollout_id`：标注"一次 rollout 执行"的 ID。注释里说明了一个精妙点：compact/subagent 路径会把一次执行拆成多个兄弟样本，它们共享 `rollout_id`，使 loss 聚合时**在 rollout 内求平均**而不是重复计数。

### 1.2 内容字段

| 字段 | 含义 |
|---|---|
| `prompt` / `tokens` | prompt 文本（或 chat 消息列表）/ 完整 token 序列（prompt+response） |
| `response` / `response_length` | 生成的文本与长度 |
| `loss_mask` | 与 response 等长的 0/1 列表：1=模型生成可训练，0=环境注入（工具返回）不可训练 |
| `reward` | float 或 dict（多指标时配合 `--reward-key` 取用） |
| `rollout_log_probs` | 推理引擎返回的逐 token logprob（TIS、重要性比值的原料） |
| `rollout_top_p_token_ids/offsets` | top-p 采样核的回放数据（见 02 篇 §3.2） |
| `rollout_routed_experts` | MoE 每层每 token 的专家路由回放（routing replay，训推一致性） |
| `teacher_log_probs` | on-policy distillation 的 teacher logprob |
| `weight_versions` | 生成该样本时引擎的权重版本（off-policy 诊断） |
| `session_id` | 会话亲和路由 key（02 篇 §3.3） |
| `non_generation_time` | 非生成耗时（工具执行等），用于性能归因 |

### 1.3 状态机

```130:140:slime/utils/types.py
    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"
    status: Status = Status.PENDING
```

由 SGLang 返回的 `finish_reason` 映射而来（types.py:410-416）：`stop→COMPLETED`、`length→TRUNCATED`、`abort→ABORTED`。`FAILED` 留给工具调用失败等可恢复错误。**partial rollout 的全部魔法都建立在 ABORTED 状态上**（§3）。

### 1.4 严格的不变量校验

`append_response_tokens`（types.py:253-314）是追加生成内容的唯一入口，它强制了一批不变量：

- `loss_mask` 长度必须等于 `response_length`（types.py:418-420）；
- `rollout_log_probs` 长度必须等于 `response_length`（422-425）；
- 可训练 token（`trainable=True`）**必须**带 logprob；环境 token（`trainable=False`）**禁止**带 logprob，自动补 0 并打 mask 0（276-281 行）；
- top-p replay 的 offsets 长度必须等于 `response_length + 1`（432-437）。

**举例（multi-turn）**：模型生成 50 token（trainable），工具返回拼入 30 token（trainable=False），再续写 40 token → `loss_mask = [1]*50 + [0]*30 + [1]*40`，`rollout_log_probs` 中工具段为 0 占位。训练时 0 掩码段不参与 loss。这种"全量保序 + 掩码"的设计让任意多轮轨迹都能压进一条扁平 token 序列。

---

## 2. 数据源与 Buffer：`data_source.py`

### 2.1 类层次

```
DataSource (abc)                     data_source.py:17-46
 └─ RolloutDataSource                data_source.py:50-165   只读 prompt 源 + 游标
     └─ RolloutDataSourceWithBuffer  data_source.py:168-222  加可回写 buffer（默认实现）
```

### 2.2 游标式无限采样

`RolloutDataSource` 维护 4 个游标（data_source.py:54-57）：`epoch_id / sample_group_index / sample_index / sample_offset`。取数逻辑（90-118）：

```92:103:slime/rollout/data_source.py
        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
```

即顺序切片、到尾就 `epoch_id+1`、以新 epoch 为种子重新 shuffle、绕回开头——无限多 epoch 的确定性采样。每个 prompt 再 `deepcopy` 成 `n_samples_per_prompt` 份组成 group，打 `group_index/index`（107-117）。

**举例**：数据集 1000 条、`rollout_batch_size=128`、每 prompt 采 8 条 → 一个 epoch 约 ⌈1000/128⌉=8 个 rollout；第 9 个 rollout 开始用 epoch 1 的重排数据。`save/load`（123-160）把游标持久化到 `global_dataset_state_dict_{rollout_id}.pt`，故障恢复后按 `epoch_id` 重 shuffle 即可精确续跑。

### 2.3 Buffer：半成品与补采的蓄水池

`RolloutDataSourceWithBuffer` 在只读源上加 `self.buffer`：

```177:196:slime/rollout/data_source.py
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples
```

**buffer 优先、新数据补齐**。谁往 buffer 里放东西？两条路：(1) partial rollout 的 ABORTED 样本（02 篇 §3.4）；(2) fully async 被截断的 group（§4.2）。取出策略默认 FIFO 的 `pop_first`（225-229），可用 `--buffer-filter-path` 换成任意策略函数（如"优先取最旧的""优先取剩余量小的"），签名 `(args, rollout_id, buffer, num_samples)`，要求自行从 buffer 删除返回项。

---

## 3. Partial Rollout：长尾的解决方案

### 3.1 问题

同步 rollout 一步的耗时 = 最长样本耗时。long CoT 下，1000 条里 999 条 2k token、1 条 32k token，整步被拖慢数倍。

### 3.2 slime 的解法：截断-缓存-续写

一轮流程（对照 02 篇代码）：

1. 主循环收够 `rollout_batch_size` 组完整样本后，调 `abort`（sglang_rollout.py:448）；
2. `abort` 对所有 server 发 `/abort_request`，未完成请求中断并返回**已生成前缀**；这些样本状态置 ABORTED，打上 `start_rollout_id` 元数据（364-365 行）；
3. `generate_rollout` 把它们 `data_source.add_samples(aborted_samples)` 回写 buffer（sglang_rollout.py:636-639）；
4. 下一轮 `get_samples` 优先从 buffer 取出这些半成品，**带着已有 tokens 重新提交**——`generate` 里 `sample.tokens` 非空时直接把已有 token 作为输入续写（`_prepare_prompt_ids` 的 `reuse_existing_input_ids` 分支，sglang_rollout.py:46-60）。

### 3.3 正确性：旧权重生成的部分算不算 off-policy？

续写意味着一条样本的前半段由旧权重生成、后半段由新权重生成——严格说是混合策略。slime 给出开关 `--mask-offpolicy-in-partial-rollout`：

```230:232:slime/rollout/sglang_rollout.py
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length
```

续写前把**已有部分的 loss mask 清零**——旧权重的产出只做上下文、不参与梯度，只有新权重续写的部分产生训练信号。这是"吞吐"与"严格 on-policy"之间由用户选择的旋钮。配合 `weight_versions` 字段还能诊断每条样本经历了几代权重。

### 3.4 相关旋钮

| 参数 | 作用 |
|---|---|
| `--partial-rollout` | 开关 |
| `--over-sampling-batch-size` | 每批超投的组数（02 篇 §3.1） |
| `--mask-offpolicy-in-partial-rollout` | 旧段 mask 清零 |
| `--buffer-filter-path` | buffer 取出策略 |
| `--dynamic-sampling-filter-path` | 组级过滤（DAPO dynamic sampling） |
| `--rollout-sample-filter-path` / `--rollout-all-samples-process-path` | 收工后的全量样本后处理钩子（sglang_rollout.py:458-465） |

---

## 4. 异步：从 one-step-async 到 fully async

### 4.1 光谱

| 方案 | 文件 | off-policy 程度 | 吞吐 |
|---|---|---|---|
| 同步 | `train.py` | 0 | 基线 |
| one-step-async | `train_async.py` | ≤1 步 | rollout 与训练重叠 |
| fully async | `examples/fully_async` + `slime/rollout/fully_async_rollout.py` | 可控多步 | 最高，抗 long-tail 差异最大的 agentic 场景 |

### 4.2 fully async 的实现

接入方式只是一个参数：`--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async`。其核心是 `AsyncRolloutWorker`（fully_async_rollout.py:76-191）：

- **常驻后台线程 + 独立 asyncio loop**（91-116 行），跨 rollout 边界存活（全局单例 `_global_worker`，48-62 行）；
- **固定并发池**：`_loop` 里只要 `active_tasks < max_concurrent` 就从 `data_buffer.get_samples(1)` 补任务（136-152 行）——"在飞轨迹数"与 `rollout_batch_size` 解耦，这是注释中强调的"Decouples max_concurrent_tasks from rollout_batch_size"（1-8 行）；
- **完成回调分流**（`_make_done_cb`，169-191）：含 ABORTED 的 group **不送训练，重新入 buffer**（183-187 行）——权重更新时被中断的样本下一轮自动用新权重重跑；正常完成的进 `output_queue`；
- 每轮 rollout 调用（`_generate_rollout_async`，194-248）只是从 `output_queue` 收够 `rollout_batch_size` 组，按 `sample.index` 排序保证确定性。

**直觉**：同步模式是"集合点式"（barrier）——所有人到齐才训练；fully async 是"流水线式"——推理引擎永远在满负荷产数据，训练侧按到达顺序取货。代价是数据由新旧不一的权重产生（每条样本带 `weight_versions` 可观测），需要 TIS 等修正配合（05 篇）。

### 4.3 外部 buffer 服务

`slime_plugins/rollout_buffer/buffer.py` 提供了另一种形态：**独立的 FastAPI buffer 服务**（端口 8889），面向"外部 agent 系统持续产生轨迹、训练侧按需拉取"的彻底解耦架构（如 OpenClaw-RL 这类"边服务边学习"的产品形态）。它自动发现 `generator/` 目录下声明了 `TASK_TYPE` 与 `run_rollout` 的模块，经 `/buffer/write` 写入、`/get_rollout_data` 拉取。这与进程内 buffer 是互补的两套体系：进程内 buffer 服务"训练循环自己产数据"，外部 buffer 服务"数据由外部世界产生"。

---

## 5. 样本 → 训练 batch 的最后一公里

`RolloutManager._convert_samples_to_train_data`（`slime/ray/rollout.py:709`）把 `list[Sample]` 转成 dict-of-lists 的 `RolloutBatch`（types.py:459 定义为 dict），再由 `_split_train_data_by_dp`（rollout.py:828）按训练 DP 数切分，经 Ray object store 分发给各训练 rank；训练 rank 侧 `MegatronTrainRayActor._get_rollout_data`（`slime/backends/megatron_utils/actor.py:235`）把字段转成 GPU 上的 tensor list，供 data iterator 消费。这条"样本对象 → 列式 batch → DP 分片 → GPU tensor"的转换链，是 rollout 世界与训练世界的海关。

---

## 5.5 深入拆解：`Sample.append_response_tokens` ——为什么它是整条 rollout 链路里最"婆婆妈妈"的函数

`slime/utils/types.py:253-314` 是 `Sample` 类里逻辑最密的方法，因为它要同时维护 **5 组必须始终对齐的并行数组**：`tokens`、`loss_mask`、`rollout_log_probs`、`rollout_top_p_token_ids/offsets`（ragged）、`rollout_routed_experts`。凡是"多轮工具调用/agent 场景"下逐段拼接响应（模型输出一段、工具返回一段、模型再输出一段……），都要走这个函数，而不是手写 `sample.tokens += xxx`。

**核心设计：`trainable` 参数把"模型生成的 token"和"环境/工具塞进来的 token"区分开**：

```python
if tokens and trainable and log_probs is None:
    raise ValueError("trainable response tokens require rollout log probabilities.")   # 模型 token 必须带 logprob，否则后续 loss 计算无源可依
if tokens and not trainable:
    if log_probs is not None:
        raise ValueError("non-trainable response tokens should not pass rollout log probabilities.")  # 工具 token 不该有 logprob
    log_probs = [0.0] * len(tokens)   # 用 0 占位，保证数组长度对齐；真正是否算 loss 靠 loss_mask=0 而不是这个 0
```

`loss_mask` 的追加逻辑很关键：`self.loss_mask += [1 if trainable else 0] * len(tokens)`——**工具返回的文本会被塞进 `tokens`（模型需要"看到"工具结果才能继续推理），但对应位置的 `loss_mask` 是 0**，训练时这些 token 只贡献 context（参与 attention），不贡献梯度。这就是"Agentic RL 中如何把多轮工具调用拼成单条训练序列"的标准答案——不是过滤掉工具文本，而是保留它但屏蔽其 loss。

**top-p ragged 数组为什么要用 `offsets` 而不是定长数组**：SGLang 返回的"每个生成 token 对应的 top-p 候选集合"，候选数量因 token 而异（有的 token 分布集中只需 1-2 个候选，有的分布平坦需要几十个）。`rollout_top_p_token_ids` 是所有 token 候选集拼接后的一维数组，`rollout_top_p_token_offsets` 长度恒为 `response_length+1`，第 `i` 个 token 的候选区间是 `token_ids[offsets[i]:offsets[i+1]]`（经典 CSR/ragged tensor 表示法，节省存储且天然支持变长）。`_merge_rollout_top_p_token_data`（types.py:39-51）在续写（partial rollout）时把新一段的 offsets **整体平移 `base_offset`** 后拼到旧数组尾部——这是"分段生成如何合并成一条完整轨迹的 ragged 元数据"的具体实现，比直接理解"续写"更底层一层。

`_apply_meta_info` 里 `rollout_routed_experts` 的 reshape（types.py:361-376）是另一个"不常见但关键"的地方：SGLang 侧只返回**从某个起点 `routed_experts_start_len` 开始新增**的路由记录（增量传输，不重复传已确认的部分），`expected_rows = len(self.tokens) - 1 - routed_experts_start_len`（`-1` 是因为路由发生在"预测下一个 token"时，最后一个 token 之后没有下一步路由）；reshape 成 `(rows, num_layers, moe_router_topk)` 后，若 `routed_experts_start_len==0`（首次写入）直接赋值，否则**保留旧数组的前 `routed_experts_start_len` 行、丢弃可能被重新计算的尾部、拼上新数据**——这是为 MoE 模型做 routing replay（训练时强制走 rollout 时同样的专家路径，避免因权重更新导致路由漂移带来的偏差）设计的增量式校验+拼接协议。

## 5.6 `Sample` 字段速查（按用途分组）

| 分组 | 字段 | 说明 |
|---|---|---|
| 身份 | `group_index`, `index`, `rollout_id` | `rollout_id` 默认回退为 `index`；agent 拆分单次 rollout 为多训练样本时，同一 rollout 产生的样本共享 `rollout_id`，loss 归一化按 rollout 而非样本计数，避免"一次探索拆成 5 条训练样本导致它被过采样 5 倍" |
| 输入 | `prompt`, `tokens`, `multimodal_inputs`, `multimodal_train_inputs`, `multimodal_train_input_id` | `multimodal_train_inputs` 是 processor 处理后可直接喂模型的张量（如 `pixel_values`），续写时靠它避免重复跑视觉预处理 |
| 输出 | `response`, `response_length`, `loss_mask`, `rollout_log_probs`, `rollout_top_p_token_ids/offsets`, `rollout_routed_experts` | 见上 |
| 评价 | `label`, `reward`, `remove_sample` | `reward` 可为标量或 dict（多目标奖励，靠 `--reward-key` 选取），`remove_sample=True` 会在后处理阶段被整条丢弃 |
| 状态 | `status`（PENDING/COMPLETED/TRUNCATED/ABORTED/FAILED）, `weight_versions` | `FAILED` 专指"工具报错/解析失败等可恢复错误"，区别于 `ABORTED`（被主动打断）——上层重试逻辑据此决定是否重投 |
| 蒸馏/回放 | `teacher_log_probs` | OPD（on-policy distillation）专用，05 篇 |
| 调试/路由 | `metadata`, `train_metadata`, `session_id`, `generate_function_path`, `custom_rm_path` | `session_id` 用于 router 的一致性哈希策略（同一会话尽量落在同一引擎，提高 KV cache 复用率），`generate_function_path`/`custom_rm_path` 支持按样本级别覆盖生成函数/奖励函数（比全局参数更细粒度） |
| 性能统计 | `spec_info`, `prefix_cache_info`, `non_generation_time` | `SpecInfo` 统计投机采样接受率/接受长度；`PrefixCacheInfo` 统计 prefix cache 命中率；`non_generation_time` 记录 agent 场景下"等工具/等环境"占用的墙钟时间，与"生成时间"分开统计，避免 agentic 场景吞吐指标被工具延迟污染 |

---

## 6. 小结

- `Sample` 用"扁平 token + loss_mask + 状态机 + 严格不变量"统一了单轮、多轮、多模态、续写一切轨迹；
- `DataSource` 游标保证确定性可恢复，`buffer` 承接一切"未完成/被回收"的样本；
- partial rollout = abort 截断 + buffer 缓存 + 续写，配 mask-offpolicy 旋钮；
- fully async = 常驻 worker + 固定并发池 + ABORTED 回炉，把长尾彻底从关键路径上移除。
- 下一篇（04）看权重如何秒级同步、显存如何错峰。
