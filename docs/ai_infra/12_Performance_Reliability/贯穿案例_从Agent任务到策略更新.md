# 从 Agent 任务到策略更新：代码、轨迹、判分与训练之间的边界

> 前置：[Harness 边界](../11_RL_Systems/Agent系统与Harness边界.md)、[训练的数据与反馈闭环](../11_RL_Systems/Agent训练的数据与反馈闭环.md)。本篇核对当前仓库的 coding-agent 接入路径；已有的 [Rollout 到参数更新](贯穿案例_从Rollout到参数更新.md)继续维护训练端完整推演。

本文在 2026-10-13 对照 slime HEAD `ed63d2c0fa4fea96a18ef7d4115e3e0205462cb8` 的工作区源码。独立的[源码清单](../reference/agent_source_manifest.json)保存文件 SHA-256、是否与该 HEAD 相同以及函数位置，不覆盖旧的[贯穿案例快照](../reference/贯穿案例源码证据.json)。教学轨迹和建议记录协议不等于运行结果；本次没有启动 CLI、E2B 沙盒、模型服务或 GPU 训练。

## 固定一个任务和三种计数

假设任务是修复一个解析函数：仓库与依赖固定，题目要求处理一种边界输入，独立测试决定是否通过。整个例子只考虑一个 agent；不同 harness 的调用协议可以不同，训练接口必须明确记录其真实输入输出。

教学任务组包含两次独立尝试 A、B：A 未修好，reward=0；B 先遇到文件路径错误，随后修复，并在历史压缩后提交，reward=1。B 导出两个训练片段。于是这里有 **1 个任务组、2 次逻辑 rollout、3 个训练行**。环境错误、上下文摘要和测试成功均是假设，用来核对数据流，不是对当前模型的测量。

实际任务入口是 [generate.py](../../../examples/coding_agent_rl/generate.py) 的 `generate`，通过 `--custom-generate-function-path examples.coding_agent_rl.generate.generate` 接入。它根据 `evaluation` 选择训练或评估协议，读取数据行的镜像、工作目录和判分 metadata。具体 schema 见 [coding-agent 示例说明](../../../examples/coding_agent_rl/README.md)，不能只给一道题目就假定环境可执行。

这里的 `Sample.rollout_id` 标识同一逻辑尝试导出的兄弟片段；顶层 `train.py` 循环里的 `rollout_id` 则是采集批次编号，不能仅因同名就视为同一层身份。建议实验记录另保留明确的 `task_id`、episode/session、request 与 segment 映射。

## 从题目到模型请求的真实调用链

```text
sglang_rollout.generate_and_rm
  → coding_agent_rl.generate.generate
    → swe.get_metadata / evaluability_check
    → adapter.open_session
    → boot_agent_sandbox → swe.prepare_workspace
    → harness.run → CLI 自己执行模型/工具循环
      → API adapter._run_turn
        → 模板渲染 prompt_ids → SGLang /generate
        → output_ids / logprob / finish_reason
        → 协议响应 → TrajectoryManager.record_turn
    → swe.git_diff → 退出 agent 沙盒
    → swe.run_evaluation：新沙盒应用 diff 并判分
    → adapter.finish_session → list[Sample]
  → sample hooks / 必要时 RM
  → rollout manager → 训练与更新
```

源码入口与职责如下；符号定位优先于容易漂移的行号。

| 阶段 | 文件与符号 | 输出或检查点 |
|---|---|---|
| 任务与生成编排 | [generate.py](../../../examples/coding_agent_rl/generate.py)：`generate`、`boot_agent_sandbox` | Session、任务 metadata、运行结束码、评分、训练行 |
| 环境初态与判分 | [swe.py](../../../examples/coding_agent_rl/swe.py)：`prepare_workspace`、`git_diff`、`run_evaluation` | 题面、补丁、判分结果与补丁是否成功应用 |
| Harness 生命周期 | [harness/common.py](../../../slime/agent/harness/common.py)：`BaseHarness.run`、`run_agent` | 安装后的 CLI、配置、执行与结束码 |
| 模型接口与原始 token | [adapters/common.py](../../../slime/agent/adapters/common.py)：`BaseAdapter._run_turn`、`call_sglang_generate` | 真实 prompt/output IDs、输出概率、停止原因 |
| 轨迹到训练行 | [trajectory.py](../../../slime/agent/trajectory.py)：`TrajectoryManager.get_trajectory`、`_SampleBuilder.to_sample` | `tokens`、response 范围、mask、reward 和片段身份 |
| 样本消费与调度 | [ray/rollout.py](../../../slime/ray/rollout.py)：`_post_process_rewards`、`_convert_samples_to_train_data`、`_split_train_data_by_dp` | 奖励统计、rollout 分母与训练分片 |

`_AGENTS` 当前选择 harness 与协议 adapter 的配对。它表达本仓库的接入关系，不是外部 CLI 任意版本均兼容的保证。真正运行的 CLI 包版本、模板、工具解析器和环境镜像需要作为实验输入保存。

`BaseHarness.run` 负责用户准备、写配置和启动 CLI；它不解释任务判分。任务初始化和最终补丁评价由示例层负责。因此“更换 harness”不应被解释为更换 optimizer 或模型 kernel，但它可能改变每轮输入、工具动作和数据分布。

## 记录的是实际服务 token，仍须区分概率语义

`BaseAdapter._run_turn` 将协议请求翻译为消息与工具定义，用 tokenizer 模板得到 `prompt_ids`。`call_sglang_generate` 以 `input_ids` 调用 `/generate`，请求 `return_logprob=True`，再从 `meta_info.output_token_logprobs` 取得输出 IDs 和数值，构造 `TurnRecord`。

这避免了从最终文本日志重新猜测训练 token，但不单凭字段名证明这些数值等于经过所有采样变换后的行为分布。Temperature、top-p、服务端概率口径和训练端重算条件仍需按[行为概率与异步校正](../11_RL_Systems/行为概率与异步校正.md)核对。该 `TurnRecord` 的字段也不包含每轮策略版本；本例以同步采集期间模型固定为前提。若改成跨版本异步采集，需要另补版本来源和关联，不能从最后一次换权编号倒推。

Adapter 在协议响应写出后才调用 `record_turn`；发送时发生断连或取消的路径不会照常记录这一轮。这是源码中的记录边界，不能升级为客户端已经消费结果的持久确认。模型接口记录也不完整代表工具副作用：动作是否落盘、是否重试、最终产物是什么，仍需环境证据。

## 三轮教学轨迹：一次工具失败和一次历史压缩

为显示对齐，使用虚构 token ID，忽略协议分隔符。每个数组都是本例声明的实际输入或输出；短数组不对应真实命令的 tokenization。

| 轮次 | 实际输入 IDs | 输出 IDs | 之后的环境事件 |
|---|---|---|---|
| 1 | `[10,11]` | `[20,21]` | 动作引用错误路径，工具返回错误观察 `[30,31]` |
| 2 | `[10,11,20,21,30,31]` | `[40,41]` | Agent 根据错误改用正确路径；本轮教学动作完成检查与修复，环境返回结果 |
| 3 | `[10,11,50,51]` | `[70,71]` | Harness 已用摘要 `[50,51]` 替换旧交互；模型在此上下文下提交结果 |

摘要由教学 harness 提供，不是这次被训练策略的新动作。第 2 轮代表单次模型输出驱动的一段工具执行；真实任务通常需要更多读取、编辑和测试轮次，处理原则相同。第 1 轮工具错误已明确返回，模型第 2 轮重新选择动作，所以这是新的策略动作，不能当作同一动作的传输重试抹掉。

前两轮的 token 前缀完全延伸，可得到一行：

```text
B1.tokens            = [10,11,20,21,30,31,40,41]
B1.response_length   = 6
B1.loss_mask         = [1,1,0,0,1,1]
B1.rollout_log_probs = [l20,l21,0,0,l40,l41]
```

第三轮输入的公共前缀只有 `[10,11]`，无法继续拼在 B1 后面，同时保持第三轮实际条件。单独一行是：

```text
B2.tokens            = [10,11,50,51,70,71]
B2.response_length   = 2
B2.loss_mask         = [1,1]
B2.rollout_log_probs = [l70,l71]
```

`l20` 等符号指对应实际采样 token 的已记录概率值；观察位置的零只是 mask=0 的占位。两行的 `rollout_id` 相同，reward 都可以保存 B 的任务结果 1；有效动作数分别为 4、2，总数为 6。摘要作为 B2 的输入保留，不能通过把 B1 塞回 B2 来“修复历史”。

这组数组用于推演表示，并不承诺任意消息树一定导出恰好两行。当前 `TrajectoryManager` 先按消息组织树，再按生成节点构建片段：

- `CLEAN`：已持有的 token 是下一 prompt 的完整前缀，追加观察和新输出。
- `REALIGN`：差异落在最近响应范围内，且新输出长度小于阈值；覆盖旧响应区域并将被替换部分置为上下文，舍弃该部分原训练信号。
- `FORK`：其他不能吸收的 token 差异，关闭当前 builder，开启新片段。

消息层的 `_try_merge_assistant_rewrite` 还可能把短的、被重写的 assistant 节点降为仅路由节点。共享生成节点通过 `response_trained` 避免在多个叶子路径中重复训练。因此要核对原始 `TurnRecord` 到导出行的映射，不能只检查文本相似度。这里没有运行真实 harness 的压缩逻辑；源码具备处理变化后输入的路径，不等于本层主动生成摘要。

## 补丁评分与结束状态是不同的字段

CLI 返回后，`generate` 提取 `git_diff`，离开 agent 沙盒，再调用 `swe.run_evaluation`。例如 scaleswe 路径会创建新评估沙盒，准备环境、应用补丁，并按 `swepro`、`eval_cmd`、`f2p_script` 的优先级选择判分入口。具体 reward 解释仍由所选协议决定。

新沙盒有助于排除 agent 运行过程中的未提交环境状态，但不自动保证判分不可操纵。当前 `git_diff` 仅显式排除题面文件和 `.harness/`；它没有统一排除所有测试文件。若任务要求测试不可修改，还需在可信判分流程中保护测试或限制补丁范围，不能把 fresh sandbox 本身当作该保证。

| 分支 | 当前实现行为 | 消费者还需注意什么 |
|---|---|---|
| CLI 非零或返回负的预算超限码 | 仍提取补丁、评分并尝试导出轨迹；将 `agent_exit_code` 放入 metadata | 结束码本身既不决定 reward，也不自动剔除轨迹 |
| 异常、外围墙钟超时或会话无可导出样本 | `_abort_result` 返回 ABORTED、reward=0、`remove_sample=True` 的零 mask 占位 | 占位不是有效策略动作；是否进组统计要核对下游 |
| 正常轨迹导出 | `_SampleBuilder.to_sample` 将状态设为 COMPLETED | `truncated`、`ill_formed` 另在 metadata，COMPLETED 不等于测试成功 |
| `evaluation=True` | 评分后返回 `_eval_result` 的零 mask 占位，不走完整训练轨迹导出 | 此分支结果不能用于声称已验证训练 token 或 loss 对齐 |

默认训练转换器的 `truncated` 列从 `Sample.Status.TRUNCATED` 生成，不从上述 metadata 推导。因此不能假定 metadata 中的长度结束已经自动进入所有截断处理。是否需要桥接该字段应按采用的算法定义；本篇记录这个实现边界，不修改训练行为。

时间预算也需要分层解释。[sandbox.py](../../../slime/agent/sandbox.py) 的 `exec_and_wait` 启动 detached 命令后轮询退出标记；等待预算耗尽时返回 `-1`，该函数本身没有显式终止 detached 进程。随后提取 diff、沙盒退出、外围超时和 session 清理另有执行边界，不能把一个预算参数当作精确停机时刻。若要严格冻结补丁，应验证停止与产物读取的顺序，这是需要额外实验或实现的协议。

## 训练前必须分别核对奖励分组和 loss 分母

`generate_and_rm` 接收 `list[Sample]` 后执行 sample hooks；非 group-RM 路径只为缺失 reward 的行调用 RM。这里 `finish_session` 已将任务 reward 赋给导出的每行，不会因为采用自定义 generate 就自动再调用一次正确的任务级判分器。

`RolloutManager._get_rollout_data` 检查嵌套片段的 `rollout_id` 后展平。后面有两套不同逻辑：

**第一套是 loss 与训练调度。** `_convert_samples_to_train_data` 根据同一 `rollout_id` 汇总所有片段的有效 mask 数，生成 `rollout_mask_sums`。`_split_train_data_by_dp` 将身份交给 [build_dp_schedule](../../../slime/utils/dp_schedule.py)，把同一 rollout 的行放在同一个训练 step，再分配 microbatch 与 DP rank。默认非 per-token loss 路径中，[get_sum_of_sample_mean](../../../slime/backends/megatron_utils/cp_utils.py)使用完整 rollout 分母，[loss_function](../../../slime/backends/megatron_utils/loss.py)再用该 step 的 rollout 数与并行因子缩放。启用 per-token loss 或自定义 reducer 时需重新核对目标。

对教学 B，B1、B2 都得到分母 6；若它们的有效 token 损失和分别为 $u$、$v$，合计贡献是 $(u+v)/6$。按各行自己的长度平均再平均，会变成另一个目标。完整数值反例见[训练归约推演](../11_RL_Systems/Agent训练的数据与反馈闭环.md#数值推演相同轨迹怎样因归约方式得到不同权重)。

**第二套是 reward 的组统计。** 当前默认 `_post_process_rewards` 在相应算法且开启奖励归一化时，先收集展平后的每行 reward；若总行数等于 `n_samples_per_prompt * rollout_batch_size`，按固定候选数 reshape，否则把所有行作为一组。该分支不根据 `group_index` 或 `rollout_id` 还原逻辑任务组。

在本例假定 `rollout_batch_size=1`、`n_samples_per_prompt=2`，展平后是 A、B1、B2 三行，reward 为 `[0,1,1]`。默认回退组的均值为 $2/3$，而两次逻辑尝试 `[0,1]` 的均值应为 $1/2$。若再启用标准差归一化，尺度也会随行重复变化。多个任务混入回退组时，还会跨任务混合统计。**Loss 端识别同一 rollout，不代表 reward 端已经按逻辑 rollout 统计。**

希望维持任务级组目标时，需要在明确的后处理或转换接口中，按任务组对逻辑尝试去重统计，再向对应片段广播。当前存在 `custom_reward_post_process_func` 和自定义训练数据转换入口，但本篇没有安装这套处理，也不宣称示例默认已完成它。只是碰巧片段数相等，也不能代替逐组身份核对。

失败占位还有顺序问题：默认转换器先 `_post_process_rewards`，再将 `remove_sample` 的 mask 清零，不直接删除该行。因此零 mask 行若到达这一分支，仍可能影响奖励均值或训练 step 的计数。上游过滤、组统计和最终分母需要联合验证；“置零梯度”与“从统计总体移除”不是一件事。

## 接到参数更新，再回到独立评估

[train.py](../../../train.py) 的同步路径先调用 rollout manager 采集，然后等待 actor 训练；需要 critic 的分支另有 value 数据依赖。保存与清理后，`actor_model.update_weights()` 将更新后的模型交给生成端，再按周期评估。训练端由 [MegatronTrainRayActor.train_actor](../../../slime/backends/megatron_utils/actor.py)进入概率、优势和训练路径，继续沿用[样本到更新案例](贯穿案例_从Rollout到参数更新.md)解释梯度、归约和优化器。

这一步的输入不应只剩一个 reward 标量：实际 tokens、mask、行为概率、逻辑身份和目标归约必须仍然相容。损失能够计算并不证明任务组统计正确，也不证明更新后 coding 能力改善。

独立评估固定保留任务、初始环境、harness revision、工具与上下文配置、判分版本，以及 token/工具/墙钟预算，比较更新前后模型。更换 harness 时，另做固定模型的对照，见[四组对照设计](../11_RL_Systems/Agent系统与Harness边界.md#训练部署与评估需要对齐的是执行条件)。记录所有计划尝试的结果，区分未启动、执行失败、判分失败和真实任务失败，再报告成功率及资源成本。

## 本篇能核对的证据与下一次运行要采什么

| 层次 | 本次证据 | 尚不能证明的事 |
|---|---|---|
| 源码与分支 | 本篇符号与独立文件 hash | 实际 CLI/模型/服务的依赖兼容性 |
| 教学记录与算术 | 三轮输入、两个片段、mask=4+2、均值 $1/2$ 与 $2/3$ | 真实 tokenizer、概率与 GPU 梯度一致性 |
| 外部执行 | 未运行沙盒或模型请求 | 超时停止、工具重试、清理和测试隔离的实际效果 |
| 训练与质量 | 未进行参数更新或评估测量 | 收敛、任务成功率、吞吐或成本改善 |

实际运行至少保存以下相互关联的产物：任务/环境/harness manifest；每次模型请求的 token 与概率记录；工具动作、attempt、结果与结束原因；提交的补丁与 grader 原始输出；导出训练行到原始动作的映射；分组、分母和更新前后的独立评估结果。当前示例中的 metadata 和日志只覆盖其中一部分，不能将这份采集清单当作已存在的统一接口。

在仓库根目录可以用标准库检查本次源码是否漂移；这只比对文件身份，不导入框架或运行模型：

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("docs/ai_infra/reference/agent_source_manifest.json").read_text())
for item in manifest["files"]:
    path = Path(item["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
print(f"checked {len(manifest['files'])} source files")
PY
```

下一步若要验证系统行为，先采集一条固定任务的完整证据，检查实际 prompt、片段映射与组统计，再扩大并发和训练规模。吞吐与故障实验的方法分别见 [RL 性能](06_slime.md)与[故障模型与恢复](故障模型与恢复.md)。

[Agent 阅读路线](../guides/Agent训练与Harness阅读路线.md) · [贯穿案例索引](贯穿案例索引.md) · [性能与可靠性](README.md)
