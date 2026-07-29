# tau-bench + Qwen3-4B 训练全流程详解

> 本文档基于 `examples/tau-bench/run_qwen3_4B.sh` 与 slime 源码逐行梳理，重点讲解 **rollout（生成/采样）**、**训练（Megatron actor 更新）** 以及 **与 sglang 训练引擎的对接** 三部分，并顺着代码逻辑盘点其余模块。最后用「一条数据」贯穿所有阶段，展示其字段如何逐步变化。

---

## 0. 一句话总结与总体架构

这是一个 **GRPO（组相对策略优化）** 强化学习训练任务：

- **环境**：tau-bench（`retail` 电商场景），agent 通过多轮「调用工具 → 环境反馈」完成任务，最终得到一个成败奖励。
- **生成引擎（rollout）**：sglang，提供 OpenAI 兼容的 `/generate` 接口，agent 每轮把整段对话发给它取下一个动作。
- **训练引擎（train）**：Megatron-LM（TP=2），对 rollout 收集到的轨迹做 GRPO 策略梯度更新。
- **资源形态**：`--colocate`，2 张 GPU 上 **训练与推理分时复用**（通过 sglang 的 memory-saver 机制在 GPU 上释放/恢复权重），权重通过 `UpdateWeightFromTensor` 从训练进程直接推送到 sglang。

### 0.1 进程/角色架构图

```
                          Ray 集群 (本机 2 GPU, colocate)
 ============================================================================
  Driver (train.py, 主循环)  ── ray 调度 ──┐
                                          │
                       ┌──────────────────┴───────────────────┐
                       ▼                                       ▼
              RolloutManager (Actor)                    MegatronTrainRayActor (Actor)
              ├─ 启动 2 个 sglang engine                  ├─ actor 模型 (TP=2)
              │   engine0 → GPU0                          ├─ 可选 ref 模型 (kl_loss_coef=0 仍会加载)
              │   engine1 → GPU1                          └─ weight_updater = UpdateWeightFromTensor
              ├─ 启动 sglang_router (端口写回 args)                │
              ├─ 调度 generate_with_tau.generate           权重同步 │ (colocate: GPU→CPU→Ray→sglang)
              │   (agent↔env↔sglang 多轮交互)  ◄──HTTP──┐    │
              └─ 收集/转换样本 ────────────────────────┘    ▼
                                                        sglang engine0/1 (GPU0/1)
                                                        (colocate 时与 actor 同卡, 分时占用)
 ============================================================================
```

### 0.2 一次迭代（`train.py` 主循环）的时序

```
rollout_id:
  1) rollout_manager.generate(rollout_id)
        └─ 调 sglang 生成 32 任务×8 轨迹 → 收集 reward/loss_mask/tokens
        └─ 转成 train_data, 按 DP 切分, ray.put → 返回 rollout_data_ref (list[Box])
  2) (offload_rollout) rollout_manager.offload()        # sglang 释放 GPU (release_memory_occupation)
  3) actor_model.async_train(rollout_id, rollout_data_ref)
        └─ 重算 old/ref log_probs → 计算 GRPO advantage → 16 步梯度累积 → optimizer.step
  4) actor_model.save_model(rollout_id)
  5) (offload_train) actor 释放 GPU
  6) rollout_manager.onload_weights()                   # sglang 重新加载(旧)权重
  7) actor_model.update_weights()                       # 把「刚训完」的新权重推给 sglang  ← 权重同步
  8) rollout_manager.onload_kv()                        # sglang 重新加载 KV cache
  9) (eval_interval) rollout_manager.eval(rollout_id)   # 用 dev 集评估 (n_samples_per_eval_prompt=1, top_k=1)
```

> **时序要点**：权重同步 `actor_model.update_weights()` 发生在**本次迭代末尾、下一次 `generate` 之前**（步骤 7），所以它推送的就是「步骤 3 刚训练出来的最新权重」。下一轮 rollout 用的正是这组新权重，**不存在滞后一拍**（这也是 `--colocate` 下的标准做法）。脚本未启用 rollout logprobs（`--use-rollout-log-probs` 并未设置，见 §4.4），但本例 on-policy 训练无需 off-policy 修正。

---

## 1. 启动脚本逐段解读（`run_qwen3_4B.sh`）

### 1.1 清理与 Ray 启动

```bash
pkill -9 sglang / ray / python   # 清掉旧进程, 保证重跑干净
ray start --head --num-gpus 2 ... # 在本机起一个 2 GPU 的 Ray head
```

脚本通过 `ray job submit` 把 `train.py` 提交为 Ray 任务运行（driver 在 Ray 集群内）。`NUM_GPUS=2` 即本机两张卡。

### 1.2 环境变量

```bash
export PYTHONPATH="/root/Megatron-LM/:${SCRIPT_DIR}"   # Megatron-LM 源码 + 本例目录(generate_with_tau 可 import)
export CUDA_DEVICE_MAX_CONNECTIONS=1                    # 限制每个设备的 CUDA 连接数, 配合 Megatron 的 deterministic 执行
```

### 1.3 模型配置（来自 `qwen3-4B-Instruct-2507.sh`）

`source .../qwen3-4B-Instruct-2507.sh` 定义了 `MODEL_ARGS` 数组，通常包含：
- `--tokenizer-name-or-path` / `--tokenizer-type`：Qwen3 的分词器与类型（`Qwen2Tokenizer` 一类）。
- `--model-type`：如 `Qwen3ForCausalLM`。
- `set_qwen3_lora` / `set_qwen3_deterministic` 之类函数：设置 RoPE/scaling、关闭 dropout、固定随机种子等，保证可复现。
- `--seq-length` / `--max-position-embeddings`：序列长度（rollout 的 prompt+response 总长需 ≤ 此值）。
- `--padded-vocab-size`：词表 pad 到便于 TP 切分的大小。

> 这些具体值在被 source 的 `scripts/models/qwen3-4B-Instruct-2507.sh` 里，本文档聚焦运行时流程，不再展开模型结构本身。

### 1.4 检查点参数（`CKPT_ARGS`）

```bash
--hf-checkpoint /root/Qwen3-4B-Instruct-2507/          # sglang 用的 HF 格式权重
--ref-load      /root/Qwen3-4B-Instruct-2507_torch_dist/ # 参考策略(Megatron 格式)加载路径
--load          /root/Qwen3-4B-Instruct-2507_slime/      # actor 初始权重(slime 格式)
--save          /root/Qwen3-4B-Instruct-2507_slime/      # 训练后保存路径
--save-interval 20
```

要点：
- **三套权重、三种格式**：sglang 用 HF 格式（`--hf-checkpoint`）；训练/参考用 Megatron 格式（`--load`/`--ref-load`）。`UpdateWeightFromTensor` 的核心工作就是把「Megatron 格式 actor 权重」转成「HF 格式张量」再喂给 sglang（§5.3）。
- `--ref-load` 指向 `_torch_dist`（Megatron/torch.distributed 格式）。虽然本脚本 `kl-loss-coef=0.00`，但 `--use-kl-loss` 会让 slime 仍然加载一个 **冻结的参考模型**（来自该 checkpoint），用于计算 `ref_log_probs`（§4.4）。

### 1.5 Rollout / 数据集（`ROLLOUT_ARGS`）

```bash
--prompt-data /root/tau-bench/retail_train_tasks.jsonl   # tau-bench 训练任务(jsonl)
--input-key index                                        # 每行用 "index" 字段作为任务索引
--rollout-shuffle                                        # 每个 rollout 打乱任务顺序
--num-rollout 500                                        # 总共 500 次 rollout(迭代)
--rollout-batch-size 32                                  # 每次 rollout 采样 32 个任务
--n-samples-per-prompt 8                                 # 每个任务采样 8 条轨迹 → GRPO 组大小=8
--rollout-max-response-len 1024                          # 单条轨迹(response)最大长度
--rollout-temperature 1                                  # 采样温度
--global-batch-size 256                                  # 训练全局 batch = 32×8 = 256
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
--balance-data
```

GRPO 组的大小推导：`rollout_batch_size(32) × n_samples_per_prompt(8) = 256` 条样本 = **32 个任务组 × 每组 8 条轨迹**。组内 8 条轨迹的奖励方差必须非零（否则该组 advantage 全 0，被 `check_reward_nonzero_std` 过滤并重采），这是 GRPO 能算出相对优势的前提。

### 1.6 GRPO 算法（`GRPO_ARGS`）

```bash
--advantage-estimator grpo        # 用 GRPO 组内归一化 advantage
--use-kl-loss                     # 开启对参考策略的 KL 项
--kl-loss-coef 0.00               # 但 KL 权重设为 0 → 实际不起作用(ref_log_probs 仍会被计算)
--kl-loss-type low_var_kl         # KL 估计器类型(低方差 KL)
--entropy-coef 0.00               # 不加熵正则
--eps-clip 0.2                    # PPO 裁剪下界
--eps-clip-high 0.28              # PPO 裁剪上界(非对称裁剪)
```

注意 `eps_clip_high=0.28 > eps_clip=0.2`：ratio 上限被放得更宽，意味着允许新策略比旧策略概率高出更多而不被裁剪（对探索略友好）。

### 1.7 性能/并行（`PERF_ARGS`）

```bash
--tensor-model-parallel-size 2   # TP=2(模型切到 2 卡)
--sequence-parallel              # 序列并行(配合 TP)
--pipeline-model-parallel-size 1 # PP=1
--recompute-granularity full ... # 重计算以省显存
--use-dynamic-batch-size         # 动态 batch(按 token 数打包 micro-batch)
--max-tokens-per-gpu 9216        # 每卡 micro-batch 的 token 上限
```

### 1.8 sglang（`SGLANG_ARGS`）

```bash
--rollout-num-gpus-per-engine 1   # 每个 sglang engine 占 1 卡
--sglang-mem-fraction-static 0.7  # sglang 静态显存占比(其余留给 KV cache)
```

`rollout_num_gpus=2`、`num_gpus_per_engine=1` → 启动 **2 个 sglang engine**（各一份完整模型，TP=1），前挂一个 router 做负载均衡。因为 `--colocate`，这 2 个 engine 与 actor(TP=2) 共用同一对 GPU，靠 memory occupation 分时。

### 1.9 自定义生成（`CUSTOM_ARGS`）

```bash
--custom-generate-function-path generate_with_tau.generate
```

这是 tau-bench 的「灵魂」：它把每个任务样本的生成替换为 `generate_with_tau.generate`（agent 多轮交互），而非默认的「一条直白续写」。注意 **没有** 设置 `--rollout-function-path`，所以顶层 rollout 函数仍是默认的 `slime.rollout.sglang_rollout.generate_rollout`；`--custom-generate-function-path` 只替换「单条样本的生成」这一步（见 §3.3）。

### 1.10 启动命令

```bash
ray job submit ... -- python3 train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 2 --colocate \
  ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ... ${CUSTOM_ARGS[@]}
```

`--colocate` 开启训练/推理同卡分时；`--actor-num-gpus-per-node 2` 与 `--rollout-num-gpus 2` 一致，说明 actor 与 rollout 共享这两张卡。

---

## 2. 入口：`train.py` 主循环

`train.py` 的 `train(args)` 是真正的主循环（不是某个 `async_run_actor`）：

```python
# slime/train.py (精简)
args, _, _ = parse_args()                 # 解析全部参数 + 校验(slime_validate_args)
pgs = create_placement_groups(args)       # 分配 actor/ref/rollout 的 GPU 资源
rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
...
# 训练前先把 actor 权重推给 sglang 一次
actor_model.update_weights()
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    if eval_interval and rollout_id == 0 and not skip_eval_before_train:
        ray.get(rollout_manager.eval.remote(rollout_id))
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))   # 单返回值: list[Box(ref)]
    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())              # sglang 释放 GPU
    actor_trains = (not use_critic) or rollout_id >= num_critic_only_steps
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
    if should_save:
        actor_model.save_model(rollout_id, force_sync=...)
    offload_train(actor_trains)                                # actor 释放 GPU
    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())       # sglang 重新加载(旧)权重
    actor_model.update_weights()                               # ← 把新权重推给 sglang (权重同步)
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())            # sglang 重新加载 KV cache
    if should_eval:
        ray.get(rollout_manager.eval.remote(rollout_id))
```

关键函数：

- **`parse_args()`**（`slime/utils/arguments.py`）：聚合 Megatron + slime + sglang 的所有参数，末尾调用 `slime_validate_args` 做一致性校验。其中与本例强相关的校验：`if args.kl_coef != 0 or args.use_kl_loss:` 要求 `--ref-load` 路径必须存在（本例满足，因为开了 `--use-kl-loss`）；`assert not (args.kl_coef != 0 and args.kl_loss_coef != 0)` 保证两种 KL 不共存。
- **`create_placement_groups(args)`**（`slime/ray/placement_group.py`）：按 `with_ref = kl_coef != 0 or use_kl_loss` 决定是否给 actor 额外预留一个「参考模型」的 GPU 资源（本例 `with_ref=True`）。它返回各角色的 placement group，actor 与 rollout 据此拿到各自的 GPU。
- **`create_rollout_manager` / `create_training_models`**：都返回 Ray Actor 的「未来句柄」，真正的初始化在各自 `create()` 里异步完成（sglang 启动慢，故异步等待）。
- **`actor_model.update_weights()`**（`slime/backends/megatron_utils/actor.py`）：训练侧触发权重同步（不是 `rollout_manager.update_weights`）。它调用 `weight_updater.update_weights(...)`，把 actor 的 Megatron 权重转 HF 后推给所有 sglang engine（见 §5.3）。在 `train.py` 里每轮迭代末尾调用一次，并把最初一次放在训练循环之前，保证 sglang 冷启动就有正确权重。

> 注意：`rollout_manager.generate` **只返回** `rollout_data_ref`（一个 `list[Box]`，每个 DP rank 一个 `ray.put` 出来的引用），`metrics` 在 `generate` 内部通过 `_log_rollout_data` 打印，并不作为返回值。`train.py` 也只接收这一个值。这与早期 slime 版本「返回 `(ref, metrics)` 元组」的写法不同。

---

## 3. Rollout 阶段（生成 / 采样）

### 3.1 `RolloutManager.generate` 总流程

`RolloutManager.generate(rollout_id)` 是 rollout 的总入口：

```python
# slime/ray/rollout.py (精简)
def generate(self, rollout_id):
    self.health_monitoring_resume()
    data, metrics = self._get_rollout_data(rollout_id=rollout_id)   # 调生成函数 + 展平
    self._save_debug_rollout_data(data, rollout_id, evaluation=False)
    _log_rollout_data(rollout_id, self.args, data, metrics, ...)    # metrics 在此打印
    if self.args.debug_rollout_only:
        return
    data = self._convert_samples_to_train_data(data)   # Sample → 训练张量字典
    return self._split_train_data_by_dp(data)           # 切分+张量化+ray.put → 返回 list[Box(ref)]
```

- **`_get_rollout_data(rollout_id)`**：内部 `data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)`，得到 `RolloutFnTrainOutput`，取 `.samples`（即 `list[list[Sample]]`，32 组 × 8），再用 `itertools.chain.from_iterable` **展平**成 256 条 `Sample`，并强制给每条 `Sample` 打上 `rollout_id`（供后续 loss reducer 按「一次 rollout」聚合，而不是按 256 条样本计数）。返回 `(data, metrics)`。
- **`call_rollout_fn(func, args, rollout_id, data_source, evaluation)`**（`slime/rollout/base_types.py`）：统一封装「调用 rollout 函数」。非评估签名 `func(args, rollout_id, data_source)`；评估签名 `func(args, rollout_id)`。这里 `func = self.generate_rollout`（默认 `slime.rollout.sglang_rollout.generate_rollout`）。
- `generate` 的返回值就是 `_split_train_data_by_dp` 的结果：**一个 `list[Box]`**（每个 DP rank 一个 `ray.put` 出来的引用）。`train.py` 直接 `ray.get` 得到它，作为 `async_train` 的输入。**`generate` 内部并不做 offload/update_weights/resume**——这些由 `train.py` 主循环在 `generate` 之后驱动（见 §0.2 / §5.5）。

### 3.2 数据采样：`data_source` 返回分组样本

`generate_rollout_async` 把「要样本 + 提交生成 + 动态过滤」放进一个循环，按 `rollout_batch_size` 个**组**收集：

```python
# slime/rollout/sglang_rollout.py (精简)
state = GenerateState(args)
target_data_size = args.rollout_batch_size                         # 32 个任务组
while len(data) < target_data_size:
    while state.remaining_batch_size < target_data_size:
        samples = data_source(args.over_sampling_batch_size)       # 取 grouped 样本 list[list[Sample]]
        state.submit_generate_tasks(samples)                       # 每组提交一个 generate_and_rm_group 任务
    done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        group = task.result()                                      # 一个 GRPO 组(n_samples_per_prompt 条)
        dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
        if not dynamic_filter_output.keep:                         # 组内 reward 方差为 0 → 丢弃重采
            state.remaining_batch_size -= 1; continue
        data.append(group)
```

- **`over_sampling_batch_size`** = `rollout_batch_size × n_samples_per_prompt`（=32×8=256）。`data_source` 是一个可调用对象：`data_source(over_sampling_batch_size)` 返回 `list[list[Sample]]`（32 组 × 8 条），每组内 8 条共享同一任务（GRPO 组）。
- 数据集底层读取 `retail_train_tasks.jsonl`，`--input-key index` 决定每行用哪个字段作为任务索引。此刻每条 `Sample` 的初始字段：`index`（全局序号）、`prompt`（任务索引，如 `"42"`）、`group_index`、`metadata`，但**还没有** `tokens/reward/loss_mask`。
- **动态采样过滤 `check_reward_nonzero_std`**：这是 `--dynamic-sampling-filter-path` 指定的过滤器。GRPO 要求组内 8 条轨迹的奖励有方差；若某组 8 条奖励全相同（std=0），则该组 advantage 全为 0、无学习信号，`call_dynamic_filter` 判定 `keep=False`，该组被丢弃并触发一次重采样（`remaining_batch_size -= 1`）。这是 GRPO 能学到「组内相对好坏」的关键保障。

### 3.3 生成函数链：`submit_generate_tasks → generate_and_rm_group → generate_and_rm`

- **`GenerateState.submit_generate_tasks(samples)`**：对 `data_source` 返回的每组样本，各起一个 `asyncio.create_task(generate_and_rm_group(...))`，组内 8 条样本在 `generate_and_rm_group` 内**并发**生成（每组一个异步任务，组内用 `asyncio.create_task` 并发）。
- **`generate_and_rm_group(args, group, sampling_params, evaluation=False)`**：为组内每条样本 `asyncio.create_task(generate_and_rm(...))`，`asyncio.gather` 等待整组完成。若开启 `group_rm`，还会对整个组做一次群体奖励模型（本例未用）。
- **`generate_and_rm(args, sample, sampling_params, evaluation=False)`**：单条轨迹生成的核心调度点：

```python
# slime/rollout/sglang_rollout.py (精简)
custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path
if custom_func_path is not None:
    custom_generate_func = load_function(custom_func_path)         # ← 注入 tau-bench 的 generate
    sample = await custom_generate_func(args, sample, sampling_params)   # 调 generate_with_tau.generate
if sample.reward is None:                              # tau-bench 已在 generate 里填好 reward, 跳过
    sample.reward = await async_rm(args, sample)
```

  - **`load_function(path)`**（`slime/utils/misc.py`）：按 `"module.func"` 字符串动态 `importlib` 导入。`--custom-generate-function-path generate_with_tau.generate` → 导入本例目录下的 `generate_with_tau.generate`。
  - tau-bench 的 `generate` **自行把 reward 写进 Sample**，所以 `if sample.reward is None` 分支被跳过（不会走通用 reward-model `async_rm`）。
  - `generate_rollout_async` 在 `rollout_id == 0` 时会 `await asyncio.gather(*[engine.initialize_weight_update_group.remote(...) for engine in sglang_engines])`，建立首次 actor↔sglang 的权重同步 NCCL 组（见 §5.4）。

> 注意区分两个「generate」：
> - `generate_rollout`（来自默认 `slime.rollout.sglang_rollout.generate_rollout`，因为脚本未设 `--rollout-function-path`）—— 顶层 rollout 函数，负责「要样本 + 提交各组 + 收集 + 动态过滤」。
> - `generate`（来自 `--custom-generate-function-path`）—— 单条样本的生成，被 tau-bench 替换成 agent 多轮交互。
> 两者通过 `generate_and_rm` 里的 `load_function(args.custom_generate_function_path)` 衔接。

### 3.4 tau-bench agent 与环境的多轮交互（`generate_with_tau.py` + `trainable_agents.py`）

`generate_with_tau.generate(args, sample, sampling_params)` 的流程：

```python
# examples/tau-bench/generate_with_tau.py
task_index = int(sample.prompt)                       # 任务索引(来自数据集)
env = get_env(env_name="retail", user_strategy="llm", user_model="gemini-2.5-flash-lite",
              task_split="train", task_index=task_index)   # 构造 tau-bench 环境
agent = agent_factory(tools_info=env.tools_info, wiki=env.wiki, config=tau_config,
                      rollout_args=args, sampling_params=sampling_params)
interaction_result = await agent.asolve(env, agent.rollout_args, agent.sampling_params, task_index)
return res_to_sample(interaction_result, task_index)
```

重点在 `agent.asolve`（`examples/tau-bench/trainable_agents.py`）的多轮循环：

```python
# examples/tau-bench/trainable_agents.py :: asolve (精简)
state = GenerateState(rollout_args)                   # 持有 tokenizer(从 hf_checkpoint 加载)
url = f"http://{rollout_args.sglang_router_ip}:{rollout_args.sglang_router_port}/generate"  # sglang router
obs, info = env.reset(task_index=task_index)          # 环境给出初始用户请求
messages = [{"role":"system","content":wiki}, {"role":"user","content":obs}]
loss_masks, response_token_ids = [], []
for _ in range(max_num_steps):                        # 最多 30 轮
    text_input = state.tokenizer.apply_chat_template(messages, ..., tools=tools_info)  # 全对话→文本
    output = await self._call_llm(url, {"text": text_input, "sampling_params": sampling_params})
    if output["meta_info"]["finish_reason"]["type"] == "abort":
        return ABORTED
    response = output["text"]
    openai_result = self._parse_tool(response)         # 解析 tool call(OpenAI adapter)
    if not openai_result["success"]:
        return ABORTED
    messages.append({"role":"assistant","content":response})
    a_tok, a_mask = self._get_token_delta(state.tokenizer, messages)
    response_token_ids += a_tok; loss_masks += a_mask          # assistant token: mask=1
    action = call_to_action_sglang(parsed["calls"], parsed["normal_text"])
    env_response = await env.step(action)                      # 环境执行工具
    if action.name != RESPOND_ACTION_NAME:
        messages.append({"role":"tool","name":...,"content":env_response.observation})
    else:
        messages.append({"role":"user","content":env_response.observation})
    e_tok, e_mask = self._get_token_delta(state.tokenizer, messages)
    response_token_ids += e_tok; loss_masks += e_mask          # 环境/工具 token: mask=0
    total_reward = env_response.reward
    if env_response.done:
        status = COMPLETED; break
return _build_final_result(res, total_reward, info, messages, loss_masks, prompt_token_ids, response_token_ids)
```

关键函数解释（这些是不太常见的点）：

- **`GenerateState(rollout_args)`**（`slime/rollout/sglang_rollout.py`）：一个轻量容器，持有 `tokenizer`（从 `--hf-checkpoint` 加载，与 sglang 用的同一套分词器）以及一些生成状态。`trainable_agents` 借此在 agent 侧自己完成 tokenize，而不依赖 sglang 返回 token id。
- **`http://{sglang_router_ip}:{sglang_router_port}/generate`**：agent 直接打 sglang 原生的 `/generate` 端点（不是 `/v1/chat/completions`）。`sglang_router_ip/port` 由 `RolloutManager` 启动 router 后**写回 `args`**（见 §5.1），所以 `generate_and_rm` 透传的 `args` 里已带这个值。router 再把请求负载均衡到 2 个 engine。
- **`_get_token_delta(tokenizer, messages)`**：多轮对话下计算「新增了哪些 token」的技巧（思路来自 verl 的 sglang 多轮实现）。它把「到上一条消息为止」和「到当前消息为止」的模板分别编码，取差值，得到本轮新增 token id。这样能精确区分哪些 token 是 assistant 生成的（`loss_mask=1`）、哪些是工具/环境返回的（`loss_mask=0`）。这是 GRPO 只优化「模型自己输出」的关键。
- **`_parse_tool` / `create_openai_adapter`**：把 sglang 返回的文本（含 `<tools>` XML 标签的函数调用）解析成 OpenAI 格式的结构化 tool call。`call_to_action_sglang` 再转成 tau-bench 的 `Action`。
- **`res_to_sample`**：把 `InteractionResult` 映射成 slime 的 `Sample`：`prompt=res.prompt`（对话模板文本）、`tokens=res.tokens`（prompt+全部轮次的 token id）、`reward=res.reward`（**标量**，最终任务成败）、`loss_mask=res.loss_mask`、`response_length=len(loss_masks)`、`status`、`metadata=res.info`。

> 至此，一条 `Sample` 的字段为：`tokens`（长度 L）、`reward`（标量 0/1 之类）、`loss_mask`（长度 L，仅 assistant 位为 1）、`response_length`（=L_response）、`status`、`metadata`、`index`、`group_index`。注意 **`rollout_log_probs` 为 None**（本例 agent 没有逐 token 记录 logprob）。

### 3.5 收集与转换：`_convert_samples_to_train_data`

`RolloutManager` 把 256 条 `Sample` 转成训练所需的「字典」（未张量化）：

```python
# slime/ray/rollout.py (精简)
def _convert_samples_to_train_data(self, samples):   # samples: 256 条 Sample(已展平)
    raw_rewards, rewards = self._post_process_rewards(samples)   # rewards 为标量列表
    train_data = defaultdict(list)
    for sample in samples:
        train_data["tokens"].append(sample.tokens)              # 已 pad 到统一长度
        train_data["response_lengths"].append(sample.response_length)
        train_data["loss_masks"].append(sample.loss_mask)
        if sample.rollout_log_probs is not None:                # 本例为 None → 不加入
            train_data["rollout_log_probs"].append(sample.rollout_log_probs)
        if sample.metadata is not None:
            train_data["source_names"].append(get_source(sample))
        ...
    train_data["rewards"] = rewards
    # 注意: total_lengths / advantages 不在这里算(见 §3.6 / §4.4)
    return train_data
```

字段含义：
- `tokens`：每条轨迹的完整 token id 序列（prompt + 所有轮次）。
- `rewards`：**标量**列表（256 个），每组 8 个共享同一任务的最终 reward（`_post_process_rewards` 可能做归一化/裁剪，本例是原始成败值）。
- `loss_masks`：与 `tokens` 等长，1=模型生成、0=环境/工具/提示。
- `response_lengths`：每条轨迹的 response 长度，用于打包 micro-batch。
- `rollout_log_probs`：本例为 None（tau-bench 未记录），故不加入；若记录则会成为 off-policy 旧策略概率。
- `total_lengths`：**不在本函数计算**，而是在下一步 `_split_train_data_by_dp` 里由 `len(tokens)` 现算。

### 3.6 切分：`_split_train_data_by_dp`

```python
# slime/ray/rollout.py (精简)
def _split_train_data_by_dp(self, data):
    data["total_lengths"] = [len(t) for t in data["tokens"]]     # 现算每条总长
    partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
        self.args, self.train_parallel_config, data["total_lengths"],
        global_batch_size=self.args.global_batch_size, rollout_indices=data["rollout_ids"],
    )
    rollout_data_refs = []
    for r in range(dp_size):                                      # 本例 dp_size=1
        partition = partitions[r]
        rollout_data = {"partition": partition,
                        "global_batch_sizes": global_batch_sizes,
                        "num_microbatches": num_microbatches,
                        "micro_batch_indices": micro_batch_indices[r]}
        for key in ["tokens", "rewards", "loss_masks", "response_lengths", "rollout_log_probs", ...]:
            if key in data:
                rollout_data[key] = [data[key][j] for j in partition]
        _tensorize_rollout_data_for_training(rollout_data)        # 张量化
        rollout_data_refs.append(Box(ray.put(rollout_data)))      # 每个 DP rank 一个引用
    return rollout_data_refs                                      # list[Box]
```

- **`build_dp_schedule(args, train_parallel_config, lengths, global_batch_size, rollout_indices)`**（`slime/utils/data.py`）：把 256 条样本按 `total_length` 打包成若干个 micro-batch，使每个 micro-batch 的 token 数尽量不超过 `max_tokens_per_gpu`(=9216)、且序列数均衡；同时返回 `micro_batch_indices`（每个 DP rank 内 micro-batch 的下标）与 `num_microbatches`。本例 `global_batch_size=256, micro_batch_size=16, dp_size=1` → **16 个 micro-batch，每批 16 条**。
- **`ray.put`**：把张量字典放入 Ray 的共享内存 object store，返回一个 `ObjectRef`；外层再包一个 `Box`（slime 用于标记「这是一段 rollout 训练数据」的轻量容器）。训练 actor 之后用 `ray.get(ref)` 取回，避免跨进程大拷贝。
- 返回 `list[Box]`（每个 DP rank 一个；本例 `dp_size=1` 故长度为 1）。`train.py` 把这个列表直接作为 `async_train` 的输入。

---

## 4. 训练阶段（Megatron actor 更新）

### 4.1 `MegatronTrainRayActor` 初始化

`create_actor_model` → `MegatronTrainRayActor.create()`（异步）里：

- `init(args, ...)`：建立分布式进程组、加载 tokenizer、`initialize_model_and_optimizer` 构建 actor（TP=2）与可选 ref 模型。因为 `use_kl_loss=True`，会加载 `--ref-load` 指定的冻结参考模型。
- `weight_updater = UpdateWeightFromTensor(...)`（colocate 默认）：负责后续把训练权重同步给 sglang（§5.3）。
- `weights_backuper = TensorBackuper()`：保存训练前权重，便于可能的回滚/对比。

### 4.2 colocate 显存分时：`sleep` / `wake_up`

因为 actor 与 sglang 共用 GPU，训练前要让 sglang 让出显存：

```python
# slime/backends/megatron_utils/actor.py (精简)
def sleep(self):
    if self.colocate:
        torch.cuda.synchronize()
        self.optimizer.offload()                       # 优化器状态移到 CPU
        self.memory_saver.pause()                       # 释放训练显存(activation/params 到 CPU)
        destroy_model_parallel()                        # 释放进程组(让出 NVLink/通信资源)
        self.saved_param = False

def wake_up(self):
    if self.colocate and not self.saved_param:
        self.memory_saver.resume()                      # 把参数搬回 GPU
        self.optimizer.load()                           # 优化器状态回 GPU
        reinitialize_model_parallel()                   # 重建进程组
```

`torch_memory_saver` 是 slime 自带的显存暂存器：`pause()` 把参数/优化器状态/激活搬到 CPU 并释放 GPU 显存；`resume()` 再搬回。配合 `destroy/reinitialize_model_parallel`，保证训练与推理在**同一时刻只有一方占用 GPU**。

### 4.3 `train().train_actor` 主流程

```python
# slime/backends/megatron_utils/actor.py :: train() (精简)
def train(self, rollout_id, rollout_data):
    self.wake_up()                                      # 训练前唤醒(占用 GPU)
    data_iterator = self._get_rollout_data(rollout_data)  # ray.get(ref) → 拆出 partition
    async def train_actor():
        self.global_step += 1
        rollout_data = self._maybe_compute_logprob(data_iterator)  # ← 重算 old/ref log_probs
        advantages, returns, _ = self.compute_advantages_and_returns(rollout_data)  # GRPO advantage
        self.train_step += 1
        loss = self.train(rollout_data, advantages, returns)        # 前向/反向/优化
        ...
    ...
    self.sleep()                                        # 训练完释放 GPU(给 sglang 用)
```

- **`_get_rollout_data`**：`ray.get(rollout_data)` 拿到张量字典，弹出 `partition`（micro-batch 划分），并做长度对齐（pad/截断到 `max_model_len`）。
- **`_maybe_compute_logprob` / `compute_log_prob`**：本例 **`use_rollout_logprobs=False`**（脚本没设 `--use-rollout-log-probs`），所以会调用训练引擎对 rollout 的 `tokens` 做一次前向，**重算** `log_probs`（作为「旧策略」概率）和 `ref_log_probs`（参考策略）。这就是 on-policy 的来源——old logprobs 来自「当前模型对 rollout 序列的重新打分」，而非 rollout 当时的概率。

  ```python
  # actor.py (精简)
  can_reuse_log_probs_in_loss = (len(num_microbatches)==1 and loss_type=="policy_loss"
                                 and kl_coef==0 and not use_rollout_logprobs and ...)
  if (not use_rollout_logprobs or get_mismatch_metrics) and not can_reuse_log_probs_in_loss:
      rollout_data.update(self.compute_log_prob(rollout_data))   # 写入 "log_probs","ref_log_probs"
  ```

- **`compute_advantages_and_returns`** → 见 §4.4。

### 4.4 GRPO 损失函数（`loss.py`）

#### 4.4.1 advantage 计算（`compute_advantages_and_returns`）

```python
# slime/backends/megatron_utils/loss.py
def compute_advantages_and_returns(args, rollout_data, log_probs, old_log_probs, ...):
    rewards = torch.tensor(rollout_data["rewards"], ...).to(device).reshape(-1, 1)  # [256,1] 标量
    if args.kl_coef == 0 or not log_probs:        # 本例 kl_coef=0(脚本未设 --kl-coef) → 走此分支
        kl = [torch.zeros_like(lp) for lp in log_probs]     # kl 全 0
    else:
        kl = [compute_approx_kl(log_probs[i], ref_log_probs[i], kl_loss_type) for i ...]
    returns = get_grpo_returns(rewards, kl)                 # 标量 reward 广播到每条 token
    returns = distributed_masked_whiten(returns, loss_masks, group=group_sizes, ...)  # 组内 whitening
    advantages = returns
    return advantages, returns, full_kl
```

- **`get_grpo_returns(rewards, kl)`**（`slime/utils/ppo_utils.py`）：对第 i 条样本，`returns[i] = torch.ones_like(kl[i]) * rewards[i]`。因为 `kl` 形状是 `[L_i]`（与 token 对齐），`ones_like` 就把**标量 reward 复制成长度 L_i 的向量**——这就是 reward 从「标量」变「逐 token」的地方。
- **`distributed_masked_whiten`**：在**同一 GRPO 组（8 条轨迹）内**做 whitening：`advantage = (r - mean(r)) / std(r)`，且只在 `loss_mask=1` 的 token 上算、其余置 0。`mean/std` 用 `dist.all_reduce` 跨 DP rank 聚合，保证组内统计量一致。
  - 例：某任务 8 条轨迹 reward = `[1,0,1,0,1,0,0,1]`，均值 0.5、标准差≈0.5 → advantage ≈ `[+1,-1,+1,-1,+1,-1,-1,+1]`（组内相对优劣）。

#### 4.4.2 策略损失（`policy_loss_function` / `compute_grpo_loss`）

```python
# slime/backends/megatron_utils/loss.py
def policy_loss_function(args, batch, logits, sum_of_sample_mean, ...):
    log_probs_and_entropy = get_log_probs_and_entropy(args, batch["unconcat_tokens"], logits, ...)
    log_probs = log_probs_and_entropy["log_probs"]           # 当前策略对 rollout 序列的 logprob
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")
    # 本例 use_rollout_logprobs=False → old_log_probs = 重算的 log_probs(来自 §4.3)
    advantages = torch.cat(batch["advantages"], dim=0)
    ppo_kl = old_log_probs - log_probs                        # log-ratio 的负号形式
    pg_loss, clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high, args.eps_clip_c)
    loss = pg_loss - args.entropy_coef * entropy_loss         # entropy_coef=0 → 无熵项
    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        kl = compute_approx_kl(log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type)
        loss = loss + args.kl_loss_coef * sum_of_sample_mean(kl)   # kl_loss_coef=0 → 不起作用
    return loss, metrics
```

- **`get_log_probs_and_entropy`**：对当前策略（正在更新的 actor）在 rollout 的 `tokens` 上做前向，取对应位置的 logprob 与 entropy。只对 `loss_mask=1` 的位置生效——所以环境/工具 token 不参与梯度。
- **`compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c)`**：标准 PPO clipped surrogate。

  ```python
  # slime/utils/ppo_utils.py
  def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c):
      # ppo_kl = old_log_probs - new_log_probs, 故 exp(-ppo_kl) = new/old 概率比(ratio)
      if eps_clip_c is not None:                               # 中心化裁剪(可选, 本例未用)
          ...
      else:
          coef_1 = torch.exp(-ppo_kl)                          # ratio = π_new/π_old
          coef_2 = torch.clamp(coef_1, 1 - eps_clip, 1 + eps_clip_high)  # 非对称裁剪 [0.8, 1.28]
          loss = -torch.min(coef_1 * advantages, coef_2 * advantages).mean()
      clipfrac = (coef_1 > 1 + eps_clip_high).float().mean() + (coef_1 < 1 - eps_clip).float().mean()
      return loss, clipfrac
  ```

  - `coef_1 = exp(old - new) = π_old/π_new`；而 PPO 通常用 `π_new/π_old`，二者互为倒数，但配合 `advantages` 与 `torch.min` 的写法等价（slime 采用 `old-new` 的约定）。
  - 裁剪区间 `[1-eps_clip, 1+eps_clip_high] = [0.8, 1.28]`：概率比超过 1.28 或低于 0.8 的部分被截掉，防止单步更新过大。`eps_clip_high=0.28` 让「概率升高」更容易被接受。
- **`compute_approx_kl(new, ref, kl_loss_type)`**（`slime/utils/ppo_utils.py`）：估计 `KL(π_new ‖ π_ref)`。`kl_loss_type="low_var_kl"` 用 `k3/low_var_kl` 公式（非负、低方差的 KL 估计）。本例因 `kl_loss_coef=0`，该项乘 0，不影响梯度，但 `ref_log_probs` 仍被计算（因为 `use_kl_loss=True` 要求 ref 模型存在）。

> **本例旧策略概率的来源澄清**：脚本**没有** `--use-rollout-log-probs`（真正的参数名是 `--use-rollout-logprobs`，脚本里并不存在），因此 `use_rollout_logprobs=False`。`old_log_probs` 来自训练引擎对 rollout 序列的**重新前向计算**（on-policy），而不是 sglang 生成时记录的概率。tau-bench 的 `generate` 也确实没有在 `Sample` 里记录 `rollout_log_probs`（见 §3.4 末尾）。

### 4.5 优化与梯度累积

- `global_batch_size=256`，`micro_batch_size=16`，`dp_size=1`（`--actor-num-gpus-per-node 2` 全用于 TP，无 DP）→ **梯度累积步数 = 256 / (16×1) = 16**。即 16 个 micro-batch 前向/反向后做一次 `optimizer.step()`。
- `model.forward_backward(...)` 走 Megatron 的 pipeline 调度（`pipeline-model-parallel-size=1` 退化为普通 TP 前向），每个 micro-batch 累加梯度，累积满 16 步后：`opt_param_scheduler.step()`（本例 `lr-decay-style constant` → 学习率恒为 `1e-6`）+ `optimizer.step()`。
- 优化器：`adam`，`weight-decay 0.1`，`adam-beta1 0.9 / beta2 0.98`。

### 4.6 保存与权重备份

- `weights_backuper.backup("actor")`：训练前把权重暂存（用于指标对比/可能回滚）。
- `save_model(rollout_id)`：按 `--save-interval 20` 把 actor 的 Megatron 格式权重写入 `--save` 路径。
- `save_hf`（`actor.py` 中由 `save_model` 触发）：把 Megatron 权重转成 HF 格式落到磁盘（供 sglang 下次冷启动或离线使用）。

---

## 5. 与 sglang 训练引擎的对接（重点）

### 5.1 启动与注册

`RolloutManager` 初始化时（异步）启动 sglang：

```python
# slime/ray/rollout.py :: _launch_servers (精简)
router_ip, router_port = _start_router(args)            # 启动 sglang_router
args.sglang_router_ip = router_ip                       # ← 写回 args(agent 靠它连 sglang)
args.sglang_router_port = router_port
...
for group_cfg in model_cfg.server_groups:
    group = _make_group(group_cfg, router_ip, router_port)
    handles, _ = group.start_engines(port_cursors)       # 启动 2 个 engine
    ray.get(handles)
```

- **`_start_router`**：用 `sglang_router` 起一个负载均衡 router，返回 `(router_ip, router_port)`，并**写回 `args`**。这是 agent 能连上 sglang 的前提（§3.4 里的 `url` 就是用它拼的）。
- **`start_engines` → `launch_server_process`**：每个 engine 起一个 `python -m sglang.launch_server ...` 子进程，`_register_to_router` 把 engine 的 `http://host:port` 注册到 router 的 `/workers`。
- **colocate GPU 分配**：`get_base_gpu_id(rank)` 把 engine rank 映射到 actor 占用的 GPU。`needs_offload=True`（colocate）时，engine0→GPU0、engine1→GPU1，与 actor(TP=2 占 GPU0/GPU1) 同卡。

### 5.2 colocate 显存互斥：memory occupation

sglang 支持 `enable_memory_saver`：可以把权重/KV cache 从 GPU 临时搬到 CPU，腾出显存给训练。

- **`release_memory_occupation()`**（`sglang_engine.py`）：调用 sglang 的 `/release_memory_occupation`，释放 GPU 上的模型权重与 KV cache。训练前（`offload_rollout`）调用。
- **`resume_memory_occupation(rollout_id)`**：调用 `/resume_memory_occupation`，把权重搬回 GPU，sglang 重新可服务。训练后、下一轮 rollout 前调用。
- 配合 `flush_cache`：释放旧请求的 KV cache，避免显存碎片。

### 5.3 权重同步：`UpdateWeightFromTensor`（colocate 默认）

这是「训练 → 推理」权重传递的核心。流程（colocate 且 `update_weights_method` 默认 `from_tensor`）：

```text
actor 训练完一步
   │
   ▼ weight_updater.update_weights()
   │  weights = weights_getter()                       # 取 actor 当前 Megatron 权重(分片在各 rank)
   │  hf_weights = convert_to_hf(weights, ...)         # Megatron 分片 → 完整 HF 张量
   │  bucket = FlattenedTensorBucket(hf_weights)       # 扁平化成一段连续缓冲区
   │  all_weights = dist.gather_object(bucket, group=gloo_group)  # Gloo 收集全量 HF 权重到各 rank
   │  for engine in sglang_engines:
   │      engine.update_weights_from_tensor.remote(all_weights, ...)   # Ray 调用推送到 engine
   ▼
sglang engine.update_weights_from_tensor(...)
   │  self.resume_memory_occupation(model_name)         # 先把权重搬回 GPU
   │  # 通过 NCCL 组(_distribute_weights)把 HF 张量加载进 sglang 模型
   │  self.release_memory_occupation()                 # 同步完再释放(回 colocate 状态)
```

关键函数解释：

- **`convert_to_hf(weights, ...)`**（`slime/backends/megatron_utils/weight_convert.py` 一类）：Megatron 为了 TP 把权重切成了多块（如 `column/row parallel` 的 `weight` 分片），`convert_to_hf` 按层名把它们聚合成 HuggingFace 格式的完整张量（如 `model.layers.{i}.self_attention.qkv.weight`）。这是「训练格式→推理格式」的关键转换。
- **`FlattenedTensorBucket`**：把一个「嵌套张量字典」压平成「一段连续内存 + 形状/偏移元数据」，方便通过 IPC/NCCL 高效传输，接收端再按元数据还原。
- **`dist.gather_object(bucket, group=gloo_group)`**：用 **Gloo**（CPU 通信后端）把所有 actor rank 的权重片段收集到各 rank，使每个 rank 都持有**完整** HF 权重（因为接下来要各自推给对应的 engine，且 engine 是 TP=1 的完整副本）。
- **`update_weights_from_tensor`**（`sglang_engine.py`）：sglang 侧真正实现加载的接口。它先用 `resume_memory_occupation` 让权重回到 GPU，再通过初始化时建立的 **NCCL 组**（`connect_rollout_engines` / `initialize_weight_update_group`）把张量广播/分发到 sglang 进程内的模型，最后 `release_memory_occupation` 释放多余显存。

> 与「落盘同步」(`update_weights_from_disk`) 的区别：`from_tensor` 全程在 GPU/CPU/Ray object store 内存里流转，**不经过磁盘**，延迟低，是 colocate 的默认选择。

### 5.4 NCCL 同步与 TP 组

- `initialize_weight_update_group(backend, group_ranks, ...)`：在 actor 与 sglang engine 之间建立用于权重同步的 NCCL 通信组。rollout_id=0 时 `generate_rollout_async` 会 `await asyncio.gather(*[engine.initialize_weight_update_group.remote(...) for engine in sglang_engines])` 完成首次建组。
- `connect_rollout_engines`：actor 侧据此拿到与各 engine 通信的句柄。`_distribute_weights` 利用该 NCCL 组把 HF 张量下发到每个 engine（本例每个 engine TP=1，所以下发是「一份完整权重给一个 engine」）。

### 5.5 触发时机与一次迭代的权重流

权重同步由 **`train.py` 主循环**驱动，而不是 `RolloutManager.generate` 内部。真实时序：

```text
# 训练循环前(只一次):
actor_model.update_weights()      # 把 actor 初始权重推给 sglang, 保证冷启动 sglang 权重正确

# 每个 rollout_id:
generate(rollout_id)              # sglang 用「当前(上一轮已同步好的)权重」生成 32 任务×8 轨迹
   └─ 返回 rollout_data_ref
offload()                         # sglang 释放 GPU(release_memory_occupation)
async_train(rollout_id)           # actor 在 GPU 上训练 → 权重更新为 W_{rollout_id}
save_model(rollout_id)
offload_train()                   # actor 释放 GPU
onload_weights()                  # sglang 把(旧)权重搬回 GPU, 进入可服务状态
actor_model.update_weights()      # ★ 把刚训完的 W_{rollout_id} 推给 sglang(from_tensor 同步)
onload_kv()                       # sglang 重新加载 KV cache
eval(rollout_id)
```

关键点：

- **没有「滞后一拍」**。`actor_model.update_weights()` 在 `async_train` **之后、下一次 `generate` 之前**执行，推送的就是本轮刚训练出的权重。因此 `generate(rollout_id+1)` 使用的 sglang 权重 = `W_{rollout_id}`，即最新策略。这正是 colocate 的标准做法。
- `update_weights` 内部（`UpdateWeightFromTensor`）会先 `resume_memory_occupation`（把 sglang 权重搬回 GPU）再同步、最后 `release_memory_occupation`（释放多余显存），与 `train.py` 里 `onload_weights`（line 84）/`onload_kv`（line 88）配合，保证 sglang 在「被推新权重」和「对外服务」两个状态间正确切换。
- 之所以先 `onload_weights` 再 `update_weights`：前者让 sglang 先恢复成可服务(旧权重)状态，`update_weights` 再用新权重覆盖它。两者都涉及显存 occupation 的切换，但顺序保证了下一步 `generate` 时 sglang 既在 GPU 上、又是新权重。

---

## 6. 一条数据的完整旅程（字段演变）

以 `task_index = 42`（retail 训练集第 42 号任务）为例，追踪它在一次 rollout 中的字段变化。

### 阶段 0：数据集采样（`get_samples`）

假设 42 号任务被抽中，且组内排在第 5 组、组内第 3 条：

```
Sample:
  index      = 42*8 + 2 = 338        # 全局样本序号(0..255)
  prompt     = "42"                  # 任务索引(字符串)
  group_index= 5                     # 第 5 个 GRPO 组
  metadata   = {...}
  tokens/reward/loss_mask = 尚未生成
```

### 阶段 1：agent 多轮交互（`generate_with_tau.generate` → `asolve`）

agent 与环境交互（假设 4 轮：assistant 调用工具 → 环境返回 → assistant 再调用 → 最终回复）。累计得到：

```
InteractionResult:
  tokens      = [prompt ids(120)] + [asst(30) | tool_obs(50) | asst(25) | tool_obs(40) | asst(20)]   # 总长 L=285
  loss_mask   = [0]*120 + [1]*30 + [0]*50 + [1]*25 + [0]*40 + [1]*20   # 仅 3 段 assistant=1, 共 75 个 1
  reward      = 1.0                  # 任务成功(标量)
  response_length = 75               # = len(loss_mask)
  status      = COMPLETED
```

`res_to_sample` 后变成 slime 的 `Sample`：

```
Sample:
  index=338, prompt="42", group_index=5
  tokens=[285 个 int]
  reward=1.0                      # 仍是标量
  loss_mask=[285 个 0/1]
  response_length=75
  status="completed", metadata={...}
  rollout_log_probs=None          # 本例未记录
```

### 阶段 2：收集转换（`_convert_samples_to_train_data`）

256 条 Sample 汇总成 `train_data` 字典（本例条目）：

```
train_data["tokens"]          = [tensor(285), ...]     # 256 条, 已 pad 到统一长度
train_data["rewards"]         = [1.0, 0.0, 1.0, ...]  # 256 个标量 (同组 8 个共享任务 reward)
train_data["loss_masks"]      = [tensor(285), ...]
train_data["response_lengths"]= [75, ...]
# 注意: 无 "rollout_log_probs"(为 None 未加入); 无 "advantages"(训练期才算)
# 注意: "total_lengths" 不在本步算, 而是在下一步 _split_train_data_by_dp 里由 len(tokens) 现算
```

### 阶段 3：切分（`_split_train_data_by_dp`）

```
schedule = build_dp_schedule(...)   # 把 256 条按长度打包成 16 个 micro-batch
train_data["partition"]        = [[0..15],[16..31],...,[240..255]]   # 16 组, 每组 16 条
train_data["num_microbatches"] = [16]
train_data["global_batch_sizes"]=[256]
tensorized = {k: tensor(v) for k,v in train_data.items()}
ref = ray.put(tensorized)           # 放入 Ray object store
返回 list[Box]（dp_size=1，故只含一个 Box）
```

### 阶段 4：训练（`async_train`）

1. `_get_rollout_data`：`ray.get(ref)` → 按 `partition` 构造 micro-batch 迭代器。
2. `compute_log_prob`：对 `tokens` 做一次前向，写入
   ```
   rollout_data["log_probs"]    = [tensor(285, logp) × 256]   # 重算的旧策略 logprob(只在 loss_mask=1 处有效)
   rollout_data["ref_log_probs"]= [tensor(285, logp) × 256]   # 参考模型 logprob
   ```
3. `compute_advantages_and_returns`：
   ```
   rewards = [[1.0]] × 256
   returns = get_grpo_returns(rewards, kl=zeros)   # 每条样本: ones(L)*1.0 → 长度 285 的向量
   advantages = distributed_masked_whiten(returns, loss_masks, group=同组8条)
             # 例: 若同组 reward=[1,0,1,0,1,0,0,1] → advantage≈[+1,-1,+1,-1,+1,-1,-1,+1](组内 whitening)
             # 仅 loss_mask=1 处非零
   ```
4. `policy_loss_function`（每个 micro-batch）：
   ```
   log_probs     = 当前正在更新的 actor 对 tokens 的 logprob
   old_log_probs = rollout_data["log_probs"]          # 重算值(on-policy)
   ppo_kl        = old_log_probs - log_probs
   ratio         = exp(-ppo_kl) = π_old/π_new
   pg_loss       = -min(ratio*adv, clamp(ratio,0.8,1.28)*adv).mean()   # 非对称裁剪
   kl_loss       = kl_loss_coef(=0) * KL(new‖ref)     # 0, 不影响
   loss          = pg_loss (+ 0)
   ```
   16 个 micro-batch 累积梯度后 `optimizer.step()`，actor 权重更新。

### 阶段 5：权重同步（`update_weights`）

**本迭代末尾**（`async_train` 之后，`train.py` 调用 `actor_model.update_weights()`），`UpdateWeightFromTensor` 把刚训练完的 actor 权重（经 `convert_to_hf` → `FlattenedTensorBucket` → Gloo `gather_object` → `update_weights_from_tensor`）推送到 2 个 sglang engine；下一轮 `generate` 即用这组新权重（见 §5.5，无滞后）。

### 字段演变总表

| 阶段 | 对象 | `tokens` | `reward` | `loss_mask` | 新增/变化字段 |
|------|------|----------|----------|-------------|----------------|
| 0 采样 | `Sample` | — | — | — | `index, prompt(="42"), group_index` |
| 1 生成 | `Sample` | `[285 int]` | `1.0`(标量) | `[285 个 0/1]` | `tokens, reward, loss_mask, response_length, status`；`rollout_log_probs=None` |
| 2 转换 | `train_data` | list[tensor] | list[标量×256] | list[tensor] | 汇总为字典；`response_lengths`（total_lengths 在下一步算） |
| 3 切分 | `train_data` | tensorized | 同 | tensorized | `partition`, `num_microbatches=16`, `global_batch_sizes=256`；`ray.put` → `Box([ref])` |
| 4 训练 | `rollout_data` | 同 | 标量→广播 | 同 | `log_probs`, `ref_log_probs`(重算)；`advantages/returns`(组内 whitening) |
| 4 训练 | 反向 | — | — | — | 由 `log_probs/old_log_probs/advantages` 算 `pg_loss` → 更新权重 |
| 5 同步 | sglang | — | — | — | 新权重经 `convert_to_hf`+`update_weights_from_tensor` 推入 sglang |

---

## 7. 关键参数对照表（本脚本实际取值）

| 参数 | 取值 | 含义 / 备注 |
|------|------|-------------|
| `--colocate` | 开 | 训练/推理同卡分时 |
| `--actor-num-gpus-per-node` / `--rollout-num-gpus` | 2 / 2 | 2 卡；actor(TP=2) 与 2 个 sglang engine 同卡 |
| `--tensor-model-parallel-size` | 2 | 模型切 2 卡训练 |
| `--rollout-num-gpus-per-engine` | 1 | 每个 sglang engine 占 1 卡 → 共 2 个 engine(完整副本)+router |
| `--hf-checkpoint` | `.../Qwen3-4B-Instruct-2507/` | sglang 用(HF 格式) |
| `--ref-load` | `.../_torch_dist/` | 参考策略(Megatron 格式)，因 `use-kl-loss` 被加载 |
| `--load` / `--save` | `.../_slime/` | actor 初始/保存权重(slime 格式) |
| `--prompt-data` | `retail_train_tasks.jsonl` | tau-bench 训练任务，`--input-key index` |
| `--rollout-batch-size` | 32 | 每次 rollout 的任务数 |
| `--n-samples-per-prompt` | 8 | GRPO 组大小；总样本 32×8=256 |
| `--rollout-max-response-len` | 1024 | 单轨迹最大长度 |
| `--rollout-temperature` | 1 | 采样温度 |
| `--global-batch-size` | 256 | 训练全局 batch(=256 条样本) |
| `--dynamic-sampling-filter-path` | `...check_reward_nonzero_std` | 过滤组内奖励方差为 0 的任务组 |
| `--advantage-estimator` | `grpo` | 组内归一化 advantage |
| `--use-kl-loss` / `--kl-loss-coef` | 开 / `0.00` | 开启 KL 项但权重 0(仍加载 ref 计算 `ref_log_probs`) |
| `--kl-loss-type` | `low_var_kl` | KL 估计器类型 |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.28` | 非对称 PPO 裁剪 `[0.8, 1.28]` |
| `--entropy-coef` | `0.00` | 无熵正则 |
| `--lr` / `--lr-decay-style` | `1e-6` / `constant` | 恒定学习率 |
| `--max-tokens-per-gpu` | 9216 | 动态 micro-batch 的 token 上限 |
| `--custom-generate-function-path` | `generate_with_tau.generate` | 单样本生成替换为 agent 多轮交互 |
| `--num-rollout` | 500 | 总迭代次数 |
| `--save-interval` | 20 | 每 20 次迭代保存 |

---

## 8. 关键函数速查（不常见函数）

| 函数 / 类 | 位置 | 作用 |
|-----------|------|------|
| `RolloutManager.generate` | `slime/ray/rollout.py` | rollout 总入口：调生成函数→展平→转换→切分→(offload/update/resume) |
| `call_rollout_fn` | `slime/rollout/base_types.py` | 统一封装 rollout 函数调用(区分训练/评估签名) |
| `generate_rollout_async` | `slime/rollout/sglang_rollout.py` | 顶层 rollout：要样本→分组→并发生成→收集 metrics→返回 `RolloutFnTrainOutput` |
| `generate_and_rm_group` / `generate_and_rm` | `slime/rollout/sglang_rollout.py` | 组内并发生成；`generate_and_rm` 通过 `load_function(custom_generate_function_path)` 注入 tau-bench 的 `generate` |
| `load_function(path)` | `slime/utils/load.py` | 按 `"module.func"` 动态导入函数(插件式扩展点) |
| `GenerateState(rollout_args)` | `slime/rollout/sglang_rollout.py` | 持有 tokenizer，供 agent 侧自行 tokenize |
| `_get_token_delta(tokenizer, messages)` | `examples/tau-bench/trainable_agents.py` | 多轮对话下计算「本轮新增 token」与对应 `loss_mask`(assistant=1, 环境=0) |
| `_build_final_result` | `examples/tau-bench/trainable_agents.py` | 拼出最终 `InteractionResult`(tokens/loss_mask/reward/response_length) |
| `res_to_sample` | `examples/tau-bench/generate_with_tau.py` | `InteractionResult` → slime `Sample` |
| `_convert_samples_to_train_data` | `slime/ray/rollout.py` | `list[Sample]` → 训练张量字典(tokens/rewards/loss_masks/...) |
| `build_dp_schedule` | `slime/utils/data.py` | 按长度把样本打包成 micro-batch(返回每个 DP rank 的样本下标) |
| `_maybe_compute_logprob` / `compute_log_prob` | `slime/backends/megatron_utils/actor.py` | 对 rollout tokens 重算 `log_probs`/`ref_log_probs`(on-policy 旧策略) |
| `compute_advantages_and_returns` | `slime/backends/megatron_utils/loss.py` | 算 GRPO returns + 组内 whitening advantage |
| `get_grpo_returns` | `slime/utils/ppo_utils.py` | 把标量 reward 广播成逐 token 向量(`ones_like(kl)*reward`) |
| `distributed_masked_whiten` | `slime/utils/ppo_utils.py` | 组内 whitening，仅 `loss_mask=1` 处有效，跨 DP `all_reduce` 聚合统计量 |
| `get_log_probs_and_entropy` | `slime/backends/megatron_utils/loss.py` | 当前策略前向，取 logprob/entropy(只在 loss_mask=1 处) |
| `compute_policy_loss` | `slime/utils/ppo_utils.py` | PPO clipped surrogate，`ratio=exp(old-new)`，非对称裁剪 `[1-eps_clip,1+eps_clip_high]` |
| `compute_approx_kl` | `slime/utils/ppo_utils.py` | 估计 `KL(π_new‖π_ref)`，`low_var_kl` 为低方差估计 |
| `sleep` / `wake_up` | `slime/backends/megatron_utils/actor.py` | colocate 显存分时：`torch_memory_saver.pause/resume` + 进程组 destroy/reinit |
| `UpdateWeightFromTensor` | `slime/backends/megatron_utils/update_weight/` | 训练权重→HF 格式→Gloo 收集→推送到 sglang(colocate 默认) |
| `convert_to_hf` | `slime/backends/megatron_utils/weight_convert.py` | Megatron 分片权重 → 完整 HF 张量 |
| `FlattenedTensorBucket` | `slime/backends/megatron_utils/update_weight/` | 把嵌套张量字典压成连续缓冲区以便 IPC/NCCL 传输 |
| `release/resume_memory_occupation` | `slime/backends/sglang_utils/sglang_engine.py` | sglang 显存占用切换(配合 colocate) |
| `initialize_weight_update_group` / `connect_rollout_engines` | `slime/backends/sglang_utils/` | 建立 actor↔sglang 的 NCCL 权重同步组 |
| `_start_router` | `slime/ray/rollout.py` | 启动 sglang_router 并把 `sglang_router_ip/port` 写回 `args`(agent 连接用) |

---

## 附：常见疑问

- **Q：为什么 reward 直到训练期才变成逐 token？** A：rollout 期 tau-bench 只产出「整条轨迹的成败标量」，存进 `Sample.reward`（标量）。真正广播到每个 token 发生在训练的 `get_grpo_returns`（用与 token 对齐的 `kl` 形状做 `ones_like*k`）。
- **Q：old_log_probs 来自哪里？** A：脚本未启用 `--use-rollout-logprobs`，所以由训练引擎对 rollout 序列重新前向计算得到（on-policy）。tau-bench 的 `generate` 也没有记录逐 token logprob。
- **Q：开了 `--use-kl-loss` 却 `kl-loss-coef 0.00` 有何意义？** A：保证参考模型被加载、`ref_log_probs` 被计算（代码结构要求），但 KL 对梯度的实际权重为 0，等价于「有参考模型但不加 KL 惩罚」。
- **Q：rollout 用的权重是不是最新的？** A：不是，比 actor 落后一次迭代（§5.5）。这是 colocate 下权重同步时序决定的标准行为。
