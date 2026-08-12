# 08 奖励模型与评估体系：RM Hub、Dynamic Filter 与 Eval Pipeline

> 对应综述（`00_rl_infra_survey.md`）§2.7「RL 算法」的输入端——advantage 计算前，reward 从哪来、怎么打分、怎么过滤、评测集怎么单独跑一套采样参数，是常被忽视但直接决定训练信号质量的一环。承接 02 篇（生成后如何调用 RM）与 03 篇（`Sample.reward` 字段），本篇专门讲透这条链路。

---

## 1. reward 从哪来：`rm_hub` 的分发架构

`slime/rollout/rm_hub/` 是内置规则奖励的集合：

```
__init__.py          # 分发入口：async_rm / batched_async_rm / remote_rm
math_utils.py         # MATH 数据集答案抽取 + sympy 等价性判断（\boxed{}）
math_dapo_utils.py     # DAPO 风格 math 评分（minerva/strict_box，reward=1.0/-1.0）
deepscaler.py          # deepscaler 规则奖励（复用 math_utils）
f1.py                  # F1/EM 文本匹配（QA 类任务）
gpqa.py                # GPQA 多选题规则打分
ifbench.py             # IFBench 指令遵循打分（依赖外部 IFBench repo）
```

### 1.1 单样本分发：`async_rm`（`rm_hub/__init__.py`）

一条样本生成完成后，reward 按下面的优先级链条被计算出来：

```python
async def async_rm(args, sample: Sample, **kwargs):
    # 单样本级 custom_rm_path（常来自 eval dataset 配置）优先级最高
    if sample.custom_rm_path:
        rm_function = load_function(sample.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    if args.custom_rm_path is not None:                 # 全局 CLI 级自定义 RM
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()   # 单样本可覆盖全局默认
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""    # 先提取 \boxed{} 内容再走对应类型
        rm_type = rm_type[len("boxed_") :]

    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    elif rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    elif rm_type == "ifbench":
        return compute_ifbench_reward(response, label, metadata=metadata)
    elif rm_type == "random":
        return random.randint(0, 1)
    else:
        raise NotImplementedError(...)
```

**优先级链**：`sample.custom_rm_path`（单样本级）→ `args.custom_rm_path`（全局 CLI）→ `sample.metadata["rm_type"]`（单样本可覆盖）→ `args.rm_type`（全局默认）。这个"从细到粗"的覆盖顺序，让同一次训练里可以混合多个数据源、每个数据源用不同的打分方式（例如训练集用 `dapo` 规则打分，某个 eval 数据集用 `gpqa`），只需要在数据源配置里设 `rm_type`/`custom_rm_path` 字段，不需要改全局参数。

`boxed_` 前缀（如 `rm_type="boxed_gpqa"`）是一个组合技巧：先用 `extract_boxed_answer` 从 response 里挖出 `\boxed{}` 内的内容当作"净答案"，再按去掉前缀后的类型走正常打分——**复用同一份提取逻辑，避免每个 RM 类型都各自实现一遍"怎么从模型输出里找到最终答案"**。

### 1.2 批量分发：`batched_async_rm`

```python
async def batched_async_rm(args, samples: list[Sample], **kwargs) -> list[int | float]:
    if args.custom_rm_path is not None:
        # 契约：自定义函数必须实现"批量模式"
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
```

设了全局 `custom_rm_path` 时，**整批** samples 一次性传给自定义函数（契约：签名须是 `async def fn(args, samples: list[Sample]) -> list[float]`）——这是给"需要看到整批样本才能打分"的场景留的接口（比如需要跨样本比较、批量调用外部服务减少 RPC 次数）。否则退化为对每条样本并发调用 `async_rm`。

### 1.3 `remote_rm`：外部奖励服务

```python
async def remote_rm(args, sample: Sample, max_retries: int = 10):
    payload = {"prompt": sample.prompt, "response": sample.response, "label": sample.label}
    session = _get_shared_session()
    for attempt in range(max_retries):
        try:
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception:
            if attempt + 1 >= max_retries:
                raise
            backoff = min(2**attempt, 30) + random.random()
            await asyncio.sleep(backoff)
```

走 HTTP POST 到 `args.rm_url`，指数退避重试（带随机抖动避免"惊群"）——这是接一个**独立部署的奖励模型服务**（比如一个真正的神经网络 RM，而不是规则打分）的标准做法：奖励服务与 rollout 引擎完全解耦，可以独立扩缩容、独立更新。

### 1.4 深入拆解：数学答案等价性判断（`math_utils.py`）

规则奖励里最不简单的一类是"判断两个数学表达式是否相等"——`\frac{1}{2}` 和 `0.5` 和 `2/4` 字面上完全不同，但数学上相等。`grade_answer`（改编自 deepscaler/Hendrycks MATH 的判分代码）分两层：

1. **字符串归一化**（`_strip_string` / `mathd_normalize_answer`）：处理 `\frac` 的各种写法变体、`a/b` 转 `\frac{a}{b}`、去掉单位文本（`\text{ ... }`）、统一空格等——这一层解决"格式不同但字面等价"的情况；
2. **sympy 语义等价性判断**：归一化后仍不能直接字符串比较的（如 `\sqrt{2}` vs `1.41421356...` 或代数表达式化简后相等），用 `sympy.parsing.sympy_parser` 把 LaTeX 转成 sympy 表达式对象，调 `sympy.simplify(a - b) == 0` 一类的方法判断数学等价——这一层解决"表达式形式不同但数值/代数等价"的情况。

**为什么需要两层而不是直接上 sympy**：sympy 解析 LaTeX 字符串本身有失败率（复杂的嵌套结构、非标准写法容易解析出错或超时），先做一遍规则化的字符串归一化能过滤掉大部分"简单格式差异"，只把真正需要语义判断的少数情况交给较慢且可能出错的 sympy 路径，兼顾了准确率与稳定性/速度。

### 1.5 `last_boxed_only_string`：括号计数而非正则

`math_dapo_utils.py` 里提取 `\boxed{...}` 用的不是正则表达式，而是手写的括号计数：

```python
def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed{")
    ...
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{": num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx : right_brace_idx + 1]
```

**为什么不用正则**：`\boxed{}` 内部可能嵌套花括号（如 `\boxed{\frac{1}{2}}`），正则的贪婪/非贪婪匹配都无法正确处理任意深度的嵌套括号（这是正则表达式的理论局限——正则匹配不了"括号配对"这种上下文无关语法），必须手写一个计数器逐字符扫描，遇到 `{` 加 1、`}` 减 1，减到 0 说明找到了与最外层 `{` 配对的 `}`。这是一个"看似可以用正则解决，但实际不行，必须写状态机"的典型例子，也是本系列反复强调的"复杂且不常见的函数需要重点理解"的一例。

---

## 2. 生成、hook、RM、filter 的准确顺序

这几类扩展点都能改 `Sample`，但调用时机不同。默认 rollout 的真实顺序是：

```text
generate_and_rm
  1. 默认 generate 或 custom generate
  2. apply_rollout_sample_hooks（每个 Sample 叶子）
  3. reward 已由 generate/hook 填好？是：保留；否：调用 RM
  4. --group-rm 时跳过单样本 RM，等同 prompt 的所有采样完成后批量打分

generate_rollout_async
  5. dynamic sampling filter 看完整 group，决定接收还是重采
  6. 收齐 rollout_batch_size 个 group 后执行 rollout-sample-filter
  7. all-samples-process 同时拿到全部已完成组和最终采用组
```

由此可得几个实用结论：

- hook 可以修正 response/metadata 后再让 RM 评分，也可以直接填 `sample.reward` 跳过单样本 RM；
- dynamic filter 看到的一定是已经打完 reward 的完整组；
- `rollout-sample-filter-path` 执行时目标 batch 已经收齐，不会因为它 mask 某条样本而自动再生成一条补位；
- 当前 `--group-rm` 路径在拿到 rewards 后执行 `for sample, reward in zip(group, rewards): sample.reward = reward`，因此实际要求外层 `group` 的元素就是 `Sample`。custom generate 的 fan-out 会形成 `list[list[Sample]]`；虽然生成与 hook 保留了该形状，但这里会对 list 写 `.reward`，二者目前不能直接组合。需要先由自定义 rollout 展平/重组，或在框架侧补齐嵌套 reward 回填逻辑。

这也是为什么逐样本 hook、RM、dynamic filter 和最终 sample filter 不能合并成一个“万能后处理函数”：它们拥有的数据视野和是否会触发重采都不同。

---

## 3. Dynamic Filter：整组丢弃机制

02/03 篇提到"dynamic filter 丢弃全对/全错组"，具体实现在 `slime/rollout/filter_hub/`：

```python
# slime/rollout/filter_hub/dynamic_sampling_filters.py
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
```

`DynamicFilterOutput`（`filter_hub/base_types.py`）是一个简单的 `{keep: bool, reason: str|None}` dataclass。这是 GRPO 训练里最常用的过滤器——**如果一组（同一 prompt 的 N 个采样）reward 标准差为 0（全对或全错），这组样本的 advantage 会全部算成 0，对训练没有任何梯度贡献**，不如提前丢弃、把生成算力省下来采一组"有区分度"的样本。

**为什么用 `std() > 1e-6` 而不是 `!= 0`**：reward 有可能是浮点数（比如自定义 RM 打的连续分数），直接判断浮点数是否为 0 会因为精度问题不可靠，用一个很小的阈值（`1e-6`）判断"实质上没有区分度"更稳健——这是"浮点数比较不能用相等判断"这条通用原则在 RL 场景的具体应用。

被这个过滤器拒绝的组，会导致 02 篇 §6.3 讲的 `remaining_batch_size` 减少，触发新一轮过采样重新补组——**dynamic filter 与 over-sampling 主循环是紧耦合的两个机制**：过滤器决定"哪些组不要"，over-sampling 循环决定"不够了就再采"。

同目录下还有 `--buffer-filter-path`（默认 `pop_first`，见 03 篇）和 `--rollout-sample-filter-path`（决定单个样本是否参与 loss 计算，直接修改 `sample.remove_sample`）——三个过滤器分别作用在"要不要保留这组刚生成的样本"（dynamic filter，发生在生成阶段）、"partial rollout buffer 里先取哪些"（buffer filter，发生在取数阶段）、"训练前单样本级别是否计入 loss"（sample filter，发生在训练前）三个不同阶段，职责边界很清晰，不要混淆。

---

## 4. 评测（Eval）Pipeline：与训练共用生成代码，但采样参数独立

### 4.1 `EvalDatasetConfig`：每个评测集的独立配置

`slime/utils/eval_config.py:120` 起，字段远比想象中丰富：

```python
@dataclass
class EvalDatasetConfig:
    name: str
    path: str
    rm_type: str | None = None
    custom_rm_path: str | None = None            # 该数据集专属的打分方式（覆盖全局 args）

    input_key: str | None = None                  # 数据集特有的字段名覆盖
    label_key: str | None = None
    multimodal_keys: dict[str, str] | None = None
    apply_chat_template: bool | None = None

    n_samples_per_eval_prompt: int | None = None   # 评测时每个 prompt 采样几次（可与训练不同，如 pass@k）
    temperature: float | None = None               # 评测专属采样参数，覆盖训练时的 rollout_temperature
    top_p: float | None = None
    max_response_len: int | None = None
    custom_generate_function_path: str | None = None   # 该数据集专属的生成函数（如需要特殊工具调用）
    app_service: str | None = None                       # 走独立 AppServer 生成（agent 场景）

    eval_task_timeout: int | None = None
    min_eval_samples: int | None = None
    eval_early_stop_remaining: int | None = None         # 提前终止：剩余样本数 < 此值
    eval_early_stop_idle_timeout: float | None = None    # 且这么久没收到新结果，才提前终止

    metadata_overrides: dict[str, Any] = field(default_factory=dict)
```

**设计要点**：每个字段默认 `None`，`pick_from_args` 在实际使用时"数据集配置有值就用数据集的，没有就回退到全局 `args`"——这是一套"逐字段可覆盖的配置分层"模式，训练可以同时挂多个评测集（如 MATH500 用 `temperature=0` 贪心解码测准确率，另一个创造性写作评测集用 `temperature=0.8` 多样采样测多样性），互不干扰。

**Early stop 机制**（`eval_early_stop_remaining` + `eval_early_stop_idle_timeout`）很值得注意：评测和训练一样会遇到长尾问题（某几条评测样本特别难/特别长，卡住整个评测的收尾），这两个参数**必须同时设置才生效**——"剩余样本数已经很少（比如 <5）"且"已经有一段时间（比如 30 秒）没有新结果进来"，才判定为"大概是卡住了/没必要等了"，提前结束评测并用已收集到的结果计算指标。两个条件同时要求是为了避免误判：只看"剩余数量少"可能是正常的收尾阶段（马上就出结果），只看"空闲时间长"可能是评测集本身就很大还有很多在跑——两者叠加才能较可靠地识别"真正卡住的长尾"。

### 4.2 `eval` 与训练共用生成代码，靠参数覆盖区分

`eval_rollout_single_dataset`（`sglang_rollout.py:486` 起）与训练路径共用同一个 `generate_and_rm`/`generate_and_rm_group`，不是重新写一套生成逻辑——这是"评测只是换了一组采样参数和数据源的特殊 rollout"这一设计理念的体现，好处是训练路径里的所有机制（partial rollout 之外，因为评测不需要）、abort、trace 埋点、多模态支持等，评测路径全部免费获得，不需要重复实现和重复测试。

`assert not args.group_rm, "Group RM is not supported for eval rollout"`（474/496 行）是一处明确的能力边界声明：group RM（需要整组比较打分，如竞技型 RM）在评测场景语义不清晰（评测通常是逐样本独立评分，不存在"组内比较"的概念），代码选择直接断言拒绝而不是静默按错误方式处理。

`EVAL_PROMPT_DATASET` 全局缓存（按 `cache_key` = 数据集路径 + tokenizer + chat template 配置等组合）避免每个 rollout 都重新加载和 tokenize 一遍评测数据集——评测数据集在整个训练过程中内容不变，只需要加载一次。

### 4.3 结果聚合

`RolloutManager.eval(rollout_id)`（`slime/ray/rollout.py`）调用 `eval_rollout`，按数据集名字聚合 rewards/truncated 等指标，交给 `_log_eval_rollout_data` 记录到 wandb/tensorboard——每个评测集的指标独立打点（如 `eval/math500/reward`、`eval/gpqa/reward`），可以在训练曲线里分别观察不同能力维度的变化趋势，而不会被混在一起看不出哪个能力在退化。

---

## 5. 一个具体例子：从生成到 reward=1 的完整判定

设某数学题 prompt，`rm_type="boxed_math"`，模型生成的 response 里包含 `...因此答案是 \boxed{42}。`：

1. `generate` 阶段：SGLang 返回完整 response 文本，写入 `sample.response`；
2. `generate_and_rm` 检测 `sample.reward is None`，调 `async_rm(args, sample)`；
3. `async_rm` 里 `rm_type = "boxed_math"` 命中 `startswith("boxed_")` 分支：`response = extract_boxed_answer(sample.response)`——扫描找到 `\boxed{42}`，用括号计数法提取出内容 `"42"`，`rm_type` 变成 `"math"`；
4. 走 `rm_type == "math"` 分支：`grade_answer_verl("42", sample.label)`——假设 `sample.label = "42"`，字符串归一化后两者完全一致，直接判定相等，返回 `True`；
5. `1 if True else 0` → `reward = 1`，写回 `sample.reward`；
6. 若这条样本所在的组（同 prompt 的 N 个采样）reward 有 0 也有 1（不是全对或全错），`check_reward_nonzero_std` 判定 `keep=True`，这组样本进入训练数据；若整组全是 1（模型对这道题总是能对），则被过滤丢弃、不进入这一批训练数据，转而触发新一轮采样。

---

## 6. 小结

- reward 的产生是一条"细粒度覆盖优先级链"：单样本 `custom_rm_path` > 全局 `custom_rm_path` > 单样本 `metadata["rm_type"]` > 全局 `rm_type`，这套设计让混合多数据源、多打分方式训练成为可能；
- 内置规则奖励覆盖数学（sympy 语义等价）、QA（F1/EM）、多选题（GPQA）、指令遵循（IFBench）、外部服务（remote_rm 带指数退避重试）；`last_boxed_only_string` 的手写括号计数是"正则表达式解决不了嵌套结构"的典型反例；
- Dynamic Filter（组内 reward 方差为 0 即丢弃）与 over-sampling 主循环紧耦合，是 GRPO 训练效率的关键一环；三种过滤器（dynamic/buffer/sample）分别作用在生成、取数、训练前三个不同阶段；
- 评测复用训练的生成代码，靠 `EvalDatasetConfig` 的逐字段覆盖机制实现"每个数据集独立采样参数"，Early stop 的双条件设计是应对评测长尾问题的稳健方案；
- 下一篇（09）回到工程化视角，看这套系统如何被调试、监控与验证正确性。
