# 07 自定义接口与 Agentic RL：把任意数据生成流程接入训练闭环

> 对应综述（`00_rl_infra_survey.md`）§2.8「Agentic RL」。
> slime 的核心承诺是"最大化的数据生成自由度"（README）：math、code、search、tool、sandbox、multi-agent 都能接入同一条 training / rollout / Data Buffer 路径。本篇讲清接口体系，并给出 examples 的完整地图。

---

## 1. 接口分层：从"换一个函数"到"换一套数据源"

按侵入程度从低到高（`slime/utils/arguments.py` 中定义）：

| 层级 | 参数 | 替换对象 | 签名 |
|---|---|---|---|
| 单样本生成 | `--custom-generate-function-path` | 一条样本如何产生（可含多轮/工具） | `async (args, sample, sampling_params[, evaluation]) -> Sample \| list[Sample]` |
| 整轮 rollout | `--rollout-function-path` | 一轮 `generate(rollout_id)` 的全部逻辑 | `(args, rollout_id, data_source[, evaluation]) -> RolloutFnTrainOutput` |
| 数据源 | `--data-source-path` | prompt 从哪来、buffer 怎么管 | 实现 `DataSource` 抽象类（03 篇） |
| RM | `--custom-rm-path` / `--group-rm` | reward 计算 | 见 `slime/rollout/rm_hub/` |
| buffer 策略 | `--buffer-filter-path` | 半成品取出策略 | 03 篇 §2.3 |
| 算法钩子 | `--custom-loss-function-path`、`--custom-advantage-function-path`、`--custom-tis-function-path`、`--custom-pg-loss-reducer-function-path` | loss/advantage/TIS/归约 | 05 篇 |
| 样本后处理 | `--rollout-sample-filter-path`、`--rollout-all-samples-process-path` | 收工后过滤/加工 | 02 篇 §3.1 |

加载机制统一是 `load_function(path)`（`slime/utils/misc.py`）：`path` 形如 `my_module.my_file.my_func`，动态 import。这意味着**自定义代码不需要放进 slime 仓库**——你自己的项目目录加进 `PYTHONPATH` 即可。

### 1.1 分发的关键细节

自定义生成函数的分发点在 `generate_and_rm`（02 篇）：

```250:261:slime/rollout/sglang_rollout.py
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
```

两个设计：(1) **样本级覆盖**——`sample.generate_function_path` 优先于全局参数，eval 数据集可以给每条样本指定不同生成函数（sglang_rollout.py:569）；(2) 函数可以返回 `list[Sample]`——一次 agent 执行 fan-out 成多条前缀链式样本（共享 `rollout_id`，03 篇 §1.1），框架在 `generate_and_rm_group` 的 gather 中原样保留形状（sglang_rollout.py:297-303 的注释专门说明）。

### 1.2 最小自定义生成函数示例

```python
# my_project/multi_turn_generate.py
from slime.utils.types import Sample
from slime.utils.http_utils import post

async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    for turn in range(3):
        output = await post(url, {
            "input_ids": sample.tokens,
            "sampling_params": sampling_params,
            "return_logprob": True,
        })
        new_tokens = [t[1] for t in output["meta_info"]["output_token_logprobs"]]
        new_logps  = [t[0] for t in output["meta_info"]["output_token_logprobs"]]
        sample.append_response_tokens(args, tokens=new_tokens, log_probs=new_logps,
                                      trainable=True, meta_info=output["meta_info"], text=output["text"])
        tool_result = run_my_tool(sample.response)          # 你的工具逻辑
        if tool_result is None:                             # 没有更多工具调用则结束
            break
        tool_ids = tokenizer_encode(tool_result)            # 环境注入的 token
        sample.append_response_tokens(args, tokens=tool_ids, trainable=False)
    sample.reward = my_reward_fn(sample)                    # 或交给 --custom-rm-path
    return sample
```

启动时加 `--custom-generate-function-path my_project.multi_turn_generate.generate` 即可。要点回顾（03 篇）：模型 token `trainable=True` 且必须带 logprob；环境 token `trainable=False` 自动 mask；`append_response_tokens` 会替你维护全部不变量。

---

## 2. `slime/agent/`：生产级 agentic 支撑层

当 agent 复杂到"真实 agent harness（如 Claude Code/Codex）直接对接 RL 训练"的程度时，手写上面的循环就不够了。`slime/agent/` 提供 TITO（token-in-token-out）范式的基础设施：

| 文件 | 职责 |
|---|---|
| `trajectory.py` | `TrajectoryManager`：按 session 把多轮对话组织成消息树（`MessageNode`），`record_turn` 记录每轮的 prompt/output token 快照（`TurnRecord`），`get_trajectory` 把树线性化成带 loss mask 的 `list[Sample]`。**核心难题是 re-tokenization drift**——同一文本在不同轮次被 tokenize 的结果可能不同，代码用 `_common_prefix_len` 检测漂移并分类（`DriftKind`），以 fork/replace 策略容错（trajectory.py:116-230） |
| `adapters/openai.py` | OpenAI 兼容 API 适配层：外部 agent 框架像调普通 LLM API 一样调用，适配层把请求翻译给 SGLang、把响应翻译回 OpenAI 格式，同时暗中记录 token 级轨迹 |
| `parsing.py` | 模型输出解析：`parse_tool_uses`、XML 格式 tool call 解析 |
| `sandbox.py` | 沙盒抽象（`Sandbox` Protocol）+ E2B 实现：代码执行环境，带 RPC 重试与 transient error 识别 |
| `harness/` | 真实 agent harness 的对接：`codex.py`、`claude_code.py`——复用生产环境的 agent 循环，RL 侧只负责"记录 + 打分" |
| `aiohttp_threaded.py` | 在线程里跑 aiohttp 服务（适配层需要与 asyncio 主循环共存） |

这套设计的思想（README 引用的「Agent-Oriented Design」博客）：**agent 循环属于用户代码，slime 只提供 token 捕获与训练闭环**——agent 用什么框架、什么 prompt 工程、什么工具，都与 RL 基础设施解耦。

---

## 3. Examples 全景地图

`examples/` 下每个目录演示一种接入方式，建议按此顺序阅读：

| 示例 | 演示的接口 | 场景 |
|---|---|---|
| `search-r1/` | `--custom-generate-function-path` | 复现 Search-R1：模型生成 `<search>query</search>` → 调检索服务（`local_search_server.py`/`google_search_server.py`）→ 结果拼回继续生成；`qa_em_format.py` 是 EM+格式 reward |
| `retool/` | 同上 | ReTool 式代码工具多轮 RL |
| `multi_agent/` | `--rollout-function-path` | 多 agent 交互的 rollout（替换整轮逻辑） |
| `fully_async/` | `--rollout-function-path slime.rollout.fully_async_rollout...` | 全异步（03 篇 §4.2），适合耗时差异大的 long-tail agentic |
| `coding_agent_rl/` | `--custom-generate-function-path` + 沙盒 | 端到端 SWE agent：容器化工具执行、test-based reward、token 对齐的轨迹段导出 |
| `on_policy_distillation/` | OPD 参数组 | teacher server 打分 + reverse KL 惩罚（05 篇 §3.4） |
| `tau-bench/` | agent harness | tau-bench 客服 agent benchmark（本目录已有 `tau-bench_qwen3_4B.md` 走读） |
| `delta_weight_sync/` | `--update-weight-*` 参数组 | 训推分离 + 磁盘增量权重同步（04 篇 §3） |
| `train_infer_mismatch_helper/` | 诊断工具 | 训推 logprob 偏差的测量与可视化（05 篇 §5 的配套） |
| `eval_multi_task/` | `--eval-*` 配置 | 多任务评测 |
| `geo3k_vlm/`、`geo3k_vlm_multi_turn/` | 多模态 | VLM 单轮/多轮 RL |
| `strands_sglang/` | 外部 agent 框架 | strands agent 与 slime 的集成 |

仓库根还有 `slime/rollout/` 下的"内置特殊 rollout"可参考：`sft_rollout.py`（伪 rollout 跑 SFT）、`on_policy_distillation.py`（teacher 打分 reward）、`sglang_streaming_rollout.py`（SSE 流式生成，abort 不丢已生成内容）、`forge_load.py`（落盘数据回放做显存测试）、`sleep_rollout.py`（rollout 侧占位，纯训练调试）。

---

## 4. 如何选择接口（决策树）

1. 只是"单条样本的生成过程不同"（多轮、工具、检索）→ `--custom-generate-function-path`；
2. "一轮 rollout 的采集逻辑不同"（多 agent、全异步、自定义过滤/聚合）→ `--rollout-function-path`；
3. "数据来自外部世界"（线上服务回流、外部 agent 系统）→ `--data-source-path` 或 `slime_plugins/rollout_buffer`（03 篇 §4.3）；
4. "reward 需要整组比较或外部服务" → `--custom-rm-path` / `--group-rm`；
5. "算法本身不同" → custom loss/advantage/TIS 钩子。

选择原则（README「为什么这个设计重要」）：**不要 fork 训练 kernel**。以上任何一层都不需要改动 slime 本体，升级上游不冲突。

---

## 5. 小结

- 三层数据接口 + 若干算法钩子覆盖了从"改一个函数"到"换一套世界"的全部需求；
- `append_response_tokens` 的不变量体系是自定义 multi-turn 时最重要的合约；
- `slime/agent/` 为真实 agent harness 提供 token 级轨迹捕获（TITO）与沙盒；
- examples 是按接口组织的活教材，建议边跑边读。
- 下一篇（08）收尾：调试、CI、容错、profiling 等工程化设施。
