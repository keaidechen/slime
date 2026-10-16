# tau-bench + Qwen3-4B：从工具交互到一次参数更新

相关基础：[Agent 环境与多轮训练](<../../../06_Post_Training/28_Agent环境与多轮训练.md>)；算法推导见 [05](<05_rl_algorithms.md>)，数据契约见 [03](<03_data_buffer_partial_rollout_async.md>)。

本篇逐项对照当前 `examples/tau-bench/run_qwen3_4B.sh`、`generate_with_tau.py`、`trainable_agents.py` 和 slime 主循环。先解释脚本希望建立的闭环，再区分源码已经实现的行为与仍需核对的适配边界。这里没有执行 GPU 训练；文中手算数字用于验证机制，不能当作一次成功运行记录。

## 0. 为什么这个案例不能只当“问答生成”

普通数学题只需生成一段回答再判分。tau-bench 的 retail（零售）环境则让 agent 查询信息、调用工具、与模拟用户交谈，环境可能随动作改变。policy（策略模型）决定下一步说什么或调用什么工具，environment（环境）执行动作并返回 observation（观察），reward（奖励）评价任务结果。

例如任务是“查询订单状态”：模型先输出查询工具调用，工具返回“已发货”，模型再回复用户。最终成功并不只取决于最后一句话，也取决于调用是否正确、此前观察是否被正确使用。因此训练数据必须保留整段上下文，同时只对模型产生的动作设置直接损失。

脚本采用 GRPO（Group Relative Policy Optimization，组相对策略优化）：同一个任务独立执行多次，用奖励的相对差异提供信号。脚本每轮接受 32 个任务组、每组 8 条轨迹，共 256 条。后文先用 2 组×2 条手算，再映射回这些实际配置。

### 0.1 谁在执行，状态在哪里

```text
Driver：train.py，驱动生成 → 训练 → 权重发布
  ├─ RolloutManager：CPU 上维护任务组、轨迹与训练 batch
  │    ├─ 每条轨迹：tau-bench 环境 + agent 循环
  │    │    ├─ 工具执行/用户模拟：环境与外部服务
  │    │    └─ HTTP /generate → router → SGLang engine 0 或 1
  │    └─ CPU tensor → 对象存储引用 → 训练进程
  └─ 两个 Megatron 训练 rank：TP=2，一起完成一份策略模型的训练
       └─ 重建完整权重 → HF 布局 → CUDA IPC → 两个 SGLang 副本
```

TP（tensor parallelism，张量并行）把同一模型的矩阵切给多卡。训练 TP=2 需要两个 rank 协作完成一次前向；推理每个 engine 用 1 卡，因此两卡上是两个独立完整副本，router 把不同请求分给它们。`--colocate` 让这两种角色预约同一对物理 GPU，靠显存阶段切换复用。

SGLang 在这里是**推理引擎**，Megatron 是训练引擎。agent 使用 SGLang 原生 `/generate`，并不是 OpenAI 兼容 `/v1/chat/completions`；示例里的 OpenAI adapter 是解析工具调用结果的适配层，不能据此混淆 HTTP 端点。

## 1. 读启动脚本：先看每个数字的单位

### 1.1 启动环境与资源

脚本启动本机 2-GPU Ray head，再经 `ray job submit` 提交 `train.py`，runtime env 的 `PYTHONPATH` 包含 Megatron 源码和 tau-bench 示例目录，使远端进程可以导入 `generate_with_tau`。

脚本开头使用进程名级别的强制清理命令，这是独占实验环境的启动约定，不能当作任意共享机器上的必要初始化步骤。`CUDA_DEVICE_MAX_CONNECTIONS=1` 影响 CUDA 工作队列/连接调度，不能单独保证数值确定性；dropout、种子、并行归约和采样也会影响复现。

### 1.2 模型结构不是从文件名猜出来的

`qwen3-4B-Instruct-2507.sh` 设置 rotary base 后引用 `qwen3-4B.sh`。当前基础配置明确指定 36 层、hidden size=2560、FFN hidden size=9728、32 个 attention head、8 个 query group、`kv_channels=128`，以及 RMSNorm、SwiGLU、QK layernorm 等结构参数。

这里 head dimension 是显式的 128，不能用 `hidden_size/num_heads=80` 代替。Q 投影输出维数是 `32×128=4096`，K/V 各是 `8×128=1024`，输入维数仍是 2560；所以逻辑权重形状分别为 Q `[4096,2560]`、K/V `[1024,2560]`。这也解释了为什么权重转换必须核对配置而不是只按模型名字拼张量，详见 [13](<13_megatron_bridge_internals.md>)。

### 1.3 checkpoint 与 reference

| 参数 | 脚本路径用途 | 需要分清的语义 |
|---|---|---|
| `--hf-checkpoint` | HF 模型、tokenizer/config 资产 | 初始化推理和提供模型信息 |
| `--load` | actor 恢复/初始化目录 | 实际加载格式由 loader 检查目录内容 |
| `--ref-load` | reference 策略目录 | 为 KL 评分提供冻结权重 |
| `--save` | 训练 checkpoint 输出目录 | 恢复训练需要可用的优化器、迭代等状态 |

`_slime`、`_torch_dist` 是示例目录命名，不代表又发明了两种不同的数学权重格式。HF 权重目录与可恢复的 Megatron 训练 checkpoint 也不能互换：仅有权重不够恢复优化器历史与随机数状态。

本例 `--use-kl-loss` 为真、`--kl-loss-coef 0.00`，当前 `with_ref = kl_coef != 0 or use_kl_loss` 因而仍加载 ref tag，训练侧也会做 reference 评分。系数为 0 只让该项不影响 loss，不会自动省掉加载和 forward。ref 保存在 pinned CPU（页锁定主机内存）备份里，评分时换入同一套 GPU module，没有额外常驻一组 ref GPU。

### 1.4 rollout 与训练批次

```bash
--rollout-batch-size 32
--n-samples-per-prompt 8
--global-batch-size 256
--rollout-max-response-len 1024
--rollout-temperature 1
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

32 的单位是 prompt/任务组，8 是每组独立轨迹数，256 是名义训练全局 batch。若每次生成都被接受，`32×8=256`；有动态过滤时，为收齐这 256 条，实际执行的任务数可能更多。`--rollout-shuffle` 在数据源的 epoch 顺序上起作用，不表示每个 rollout 都重新独立洗牌。

`over_sampling_batch_size` 未显式设置，参数校验默认对齐 rollout batch size；`data_source(32)` 取的是 32 组，不是 32 条轨迹。

**1024 的边界尤其重要**：自定义 `asolve` 每轮把同一份 `sampling_params` 传给 `/generate`，所以这里限制的是单次模型调用的新增 token 预算。当前 agent 循环默认最多 30 次环境交互；没有因此自动把整条轨迹限制为 1024 token。总上下文还包含初始提示、历次输出和工具观察，应另行核对模型上下文与训练组批限制。

### 1.5 训练与推理的容量预算

训练设置 `TP=2, PP=1, CP=1, EP=1`，所以两张训练卡只有 1 个 DP（数据并行）副本。`--sequence-parallel` 在 TP 配合下切部分激活，不能把它再计为额外副本。`--recompute-granularity full` 用反向时重算激活换取显存节约，会增加计算。

`--use-dynamic-batch-size --max-tokens-per-gpu 9216` 按真实长度规划 micro-batch（一次前反向的子批），不是固定每 16 条一批。`--sglang-mem-fraction-static 0.7` 则是推理侧配置：静态预算包含模型权重和 KV cache pool，其余留给激活、图缓冲等运行空间；并不是“70% 放权重，剩余 30% 才放 KV”。也不能据此直接推断训练必然还有 30% 可用，因为共置依赖 offload 时序。

## 2. 一轮训练：把权重版本和完成条件标出来

用 $W_0$ 表示首次同步完成的策略，$D_0$ 表示由它产生的轨迹。主循环先建立资源、启动 rollout、初始化训练模型，并发布一次初始权重，然后按下列依赖执行：

| 阶段 | 执行者与数据 | 下一阶段的前提 |
|---|---|---|
| 生成 $D_0\leftarrow W_0$ | RolloutManager、agent 环境、SGLang | 收齐有效组，清理剩余任务，完成 batch 转换 |
| rollout offload | SGLang 释放相应 GPU allocation | 相关请求/设备工作已达到安全边界 |
| 训练 $D_0$，得到 $W_1$ | Megatron rank 上 forward/backward/optimizer | 所有 micro-batch 和需要的通信结束 |
| 按周期保存、清理 rollout 临时数据 | 训练 rank、manager | 必须先完成相应消费者 |
| 训练侧 offload | 训练状态按协议挂起，CPU 保存必要数据 | 推理恢复时显存足够 |
| onload weights、发布 $W_1$ | updater 与两个 SGLang engine | 所有目标参数更新完成且缓存失效 |
| onload KV、按周期 eval | 推理服务 | 新权重和 KV pool 都已可用 |

`ray.get` 等待远程方法返回；真正 GPU 完成条件还由方法内部的同步与通信协议负责。`async_train` 这个方法名不意味着这里自动和下一轮生成重叠：driver 紧接着等待结果，脚本是同步阶段循环。

下一轮 $D_1\leftarrow W_1$，因此没有额外一轮的版本发布滞后。但“权重版本新鲜”不等于“推理与训练概率逐位相同”，也不等于“重新编码后的轨迹精确等于采样动作”。后面的概率与 token 核对仍然必要。

当前 `sleep/wake_up` 使用 `torch_memory_saver.pause/resume` 和进程组销毁/重载；它是被集成的显存管理库，不是示例手写的 `optimizer.offload/load` 方法。CPU backup、CUDA context、通信和临时 buffer 各有生命周期，不能把 offload 描述为这一方完全没有任何驻留状态。

## 3. 从任务索引到多轮轨迹

### 3.1 两种 generate 的职责

脚本只设置 `--custom-generate-function-path generate_with_tau.generate`，未替换整轮 rollout。因此外层仍是：

```text
RolloutManager.generate
  → 默认 generate_rollout
  → data_source 取任务组
  → generate_and_rm_group：并发执行同题的多个采样，等组完成
  → generate_and_rm：调用 tau-bench 的单样本 generate
  → sample hooks → 缺奖励才调用 RM
  → dynamic filter → 收齐目标组 → 转训练 batch
```

对于 2 任务×2 采样，每组要等两条结果才能算奖励方差。一条很快答完不代表该组已经可以训练。若奖励分别 `[1,0]`、`[1,1]`，第一组保留，第二组被丢弃并补采；这省的是无相对信号组的训练预算，已执行的生成与外部用户模拟调用并不会退还。

### 3.2 环境与模型交替执行

`generate_with_tau.generate` 从 `sample.prompt` 读任务索引，调用 `get_env` 创建 retail 环境，构造 agent，再 `await agent.asolve(...)`。本实现显式断言不支持 partial rollout，因为 token 前缀本身不足以恢复整个工具环境。

一次 `asolve` 的逻辑顺序是：

1. `env.reset` 返回初始观察；把 wiki 和观察组织成 system/user 消息。
2. `apply_chat_template(..., tools=tools_info)` 渲染模型输入，示例还会重写工具调用说明。
3. 发 `/generate`，等待文本响应；HTTP 等待期间事件循环可推进其它任务。
4. 解析工具调用或普通回复，增加 assistant 消息和训练 token 片段。
5. 环境执行动作；工具调用产生 tool 消息，直接答复可能触发模拟用户产生 user 消息。
6. 追加观察对应的非训练 token，记录 reward/info；环境 done 时结束，否则下一轮。

奖励取自 `env_response.reward`，代码每轮覆盖 `total_reward`，不是对所有轮次 reward 求和。最终值是该示例约定的轨迹反馈。解析失败、模型请求 abort、部分环境异常都可能返回 ABORTED；达到交互次数上限则 TRUNCATED，它们不能自动解释为任务成功。

### 3.3 用户模拟服务也是工作负载的一部分

本地配置选择 Gemini 用户模拟模型，`generate_with_tau.py` 还把 API key 设成占位字符串 `NONE`。这说明示例接入需要配置真实运行环境；这里只阅读源码，不发起外部调用。GPU 服务空闲时，瓶颈可能在用户模拟、工具执行或额度/并发限制，而非 SGLang。

每条轨迹应有独立环境状态。多个同题采样若共用可变订单状态，会互相影响，不再是从同一初态出发的独立尝试。取消模型请求也不会自动回滚已经执行的工具动作；若要支持续跑，必须额外设计环境快照与幂等处理。

## 4. token 与 mask：先用 9 个整数算清楚

### 4.1 三种长度不能混用

以下是省略模板细节的教学序列，整数是 token ID：

```text
初始 prompt： [10,11,12]    P=3
assistant 1： [20,21]       2 个模型 token
工具观察：     [30,31,32]    3 个环境 token
assistant 2： [22]          1 个模型 token

tokens = [10,11,12,20,21,30,31,32,22]   shape [L=9]
loss_mask = [1,1,0,0,0,1]              shape [R=6]
response_length = 6
有效训练 token 数 = sum(loss_mask) = 3
```

`L=P+R=9`；response 区间包含工具观察，因此 R=6，不是 3；mask 不包含初始 prompt，因此长度也不是 9。最后输出 token 22 的 logits 由其前面整个上下文计算而来，所以工具观察不能从 tokens 删除。

`_build_final_result` 实际使用 `tokens=prompt_token_ids+response_token_ids`、`loss_mask=loss_masks`、`response_length=len(loss_masks)`。`response` 文本则拼接 assistant 消息，不包含全部工具文本；不能重新 tokenize 这个字符串来替换完整训练轨迹。

### 4.2 token delta 做了什么，没证明什么

`examples/tau-bench/token_delta.py::get_token_delta` 比较前后渲染文本，截取新增部分后编码。assistant 的 generation prompt（提示模型开始回复的模板头）若位于 response 区间，会被 mask 为 0；assistant 内容记 1，tool/user 增量记 0。检测到不满足预期的模板前缀/后缀关系时会报错，避免盲目按旧长度截取。

文本前缀一致仍不自动证明 token 前缀一致。例如整个文本编码为 `[7,8]`，把两段文本分开编码再拼接却可能成为 `[9,10,8]`。本例向服务提交文本，并在客户端重建 token，且移除了某些末尾标记；因此应对照**真实请求 input token 和真实生成 output token**验证记录。不能把本地 token delta 无条件称为严格 TITO（token-in-token-out）。07 篇解释了重编码漂移与停止对无法对齐片段训练的取舍。

### 4.3 当前 `res_to_sample` 的适配边界

源码直接构造一个新 `Sample`，需要读者特别注意它与通用契约的差异：

| 字段 | 当前示例行为 | 对后续机制的影响 |
|---|---|---|
| `index` | 写成 `task_index` | 同题多采样会重复，不是数据源分配的唯一执行编号 |
| `group_index/rollout_id` | 新对象没有复制原值 | 原身份丢失；普通批次仍有嵌套分组，但追踪与 sibling 语义不能靠它自动保留 |
| `status` | 映射成字符串 `"completed"/"aborted"` 等 | 通用路径与 `Sample.Status` 枚举比较，字符串不会自动等于枚举 |
| `rollout_log_probs` | 未填 | 无法直接做依赖行为概率的 TIS 或启用 `--use-rollout-logprobs` |
| `weight_versions` | 未记录 | 不能从该字段诊断策略年龄 |

这些是当前源码的可见行为，不是推荐复制的接口设计。本篇保留其真实性：想把案例接成可靠的自定义生成器，应在对应转换处保留身份、使用统一状态类型，并明确是否记录原始采样概率。字段长度正确只是第一层，不足以证明任务状态和策略来源正确。

## 5. 四条轨迹如何变成训练 batch

### 5.1 先评分和组内标准化，再分发

`RolloutManager._post_process_rewards` 先得到原始标量 reward。正常固定组大小场景按 `[B,G]` reshape。两组分别是 `[1,0]`、`[0,1]`，shape `[2,2]`；每组均值为 0.5、样本标准差为

$$
s=\sqrt{\frac{(1-0.5)^2+(0-0.5)^2}{2-1}}=\sqrt{0.5}.
$$

标准化结果为 `[+0.7071,-0.7071]` 与 `[-0.7071,+0.7071]`。标准差分母是 G−1，若误用 G，会算成 ±1。实际分母还加 `1e-6`。

真实脚本 G=8。若奖励 `[1,0,1,0,1,0,0,1]`，平方偏差和为 `8×0.25=2`，样本方差 `2/7`，标准差约 0.5345，优势约 ±0.9354。已经标准化的 `rewards` 与原始 `raw_reward` 应分开保存，否则日志看不到成功率。

自定义 fan-out 若改变每组条数，默认“按固定数量 reshape”就未必适用；不能把列表长度凑对当作语义正确，需按任务身份定义奖励聚合。

### 5.2 从 Python 列表到 GPU tensor

四条轨迹可能有不同长度 `[9,6,8,5]`，因此转换后是列式 dict-of-lists：

```text
tokens           : 四个列表，长度 [9,6,8,5]
response_lengths : 四个整数，例如 [6,3,5,2]
loss_masks       : 四个列表，长度与各自 response_lengths 相同
raw_reward       : 四个原始标量
rewards          : 四个标准化标量
```

`_split_train_data_by_dp` 算 `total_lengths`，由 `build_dp_schedule` 生成每个 DP 副本的分区和 micro-batch 下标；`tensorize_rollout_data_for_training` 再把 token/mask 等变成 CPU tensor。默认 `ray.put` 返回 ObjectRef，用 `Box` 包住后传给训练侧，避免在控制流中直接物化大数据；跨节点传输、反序列化和 H2D 拷贝仍有成本。

本例 DP=1，所以只有一份逻辑分区，两张 TP 卡协作使用它。actor 的 `_get_rollout_data` 经 `process_rollout_data` 取回数据，把 token/mask 转到 GPU；它不是任意截断长序列的清洗函数。超预算序列应在相应生成/调度契约处处理，不能假定 loader 会悄悄修好。

### 5.3 动态 packing 的手算

把 token 上限缩小为 14，按上述长度顺序 first-fit：9 先入桶 A；6 放不下 A，开桶 B；8 放入 B，得到 14；5 放入 A，得到 14。因此两桶的样本下标分别为 `[0,3]`、`[1,2]`。输入总量 28 token，恰好不需要为了样本长度统一补到 `4×9=36`。

这里只演示容量分配，实际还有对齐 padding、并行调度与 rollout 边界。packing 后 attention 必须隔离不同样本，position 和 shifted target 也须按各自边界解释；简单将 token 拼成一句长文本会泄漏上下文。

真实上限为 9216，256 条轨迹的 micro-batch 数取决于实际长度，不能预先写死。micro-batch 数改变的是梯度分几次累积，不应无意改变这 256 条样本的总归一化权重。

## 6. 训练端：从概率到 GRPO 更新

### 6.1 当前概率、old 概率、reference 概率

训练当前 forward 输出 logits，产生可求导的 `log_probs`；PPO 需要一个不求导的旧策略分母。脚本未设 `--use-rollout-logprobs`，agent 也没记录逐 token 行为概率，所以不能从推理端直接取它。

当前 `train_actor` 有两类路径：满足 `can_reuse_log_probs_in_loss` 条件时省去额外 actor 评分 forward，loss 内用当前 logprob 的 detached 值作分母；否则先 `compute_log_prob` 缓存更新前的评分。本例名义上一轮一次更新，可能满足复用条件，所以“必定单独重算一次 old”并不准确。`len(num_microbatches)==1` 在这里表示只有一个训练 step 的计划项，不是整个 batch 只有一个 micro-batch。

还须核对优势构造的形状来源：当前 `compute_advantages_and_returns` 的零 KL 分支从 `log_probs or rollout_log_probs or values` 取数组形状；若本例省略独立评分、又没有 rollout 概率和 critic value，不能假定这个前置分支自动有数据可用。这是现有示例与主训练路径需要实际验证的兼容点，本篇不把它描述为已验证可直接运行。

无论哪条路径，**训练侧重新评分不使数据自动变成 on-policy**。on-policy 描述样本产生分布与所用策略的关系；重算只能回答当前模型如何评价这段序列，不能改变当时的采样行为。

### 6.2 用一个 token 算 PPO loss

沿用成功轨迹优势 $A\approx0.7071$。假设某 token 旧评分概率为 0.5，当前概率为 0.6：

$$
\ell_{old}=\ln0.5,\quad \ell_{new}=\ln0.6,\quad
\rho=e^{\ell_{new}-\ell_{old}}=0.6/0.5=1.2.
$$

脚本区间为 `[1−0.2,1+0.28]=[0.8,1.28]`，没有裁剪，loss 为 `−1.2×0.7071≈−0.8485`。最小化它会鼓励这次动作。若成功轨迹该位置比率升到 1.5，PPO 选择裁剪项 `−1.28×0.7071≈−0.9051`，继续增大该比率不再从这一项获得收益。

对失败轨迹 A<0，若比率为 1.5，实际选择的是未裁剪项以抑制坏动作概率上升；不能说“所有超界 token 都停止梯度”。完整公式为

$$
L_t=\max(-\rho_tA_t,-\operatorname{clip}(\rho_t,0.8,1.28)A_t).
$$

`compute_policy_loss` 返回逐 token loss 和裁剪指标，外层才用 mask 与归约器汇总，不是在该函数里不加区分地 `.mean()`。工具位置 mask=0 取消自身直接 loss，但工具上下文仍影响后续动作的计算和梯度。

### 6.3 标量奖励怎样覆盖轨迹

`get_grpo_returns` 把每条标准化奖励复制到它的 response 概率形状。纸上例子的 R=6，因此成功轨迹 advantage 为 shape `[6]` 的常数 0.7071，再用 `[1,1,0,0,0,1]` 选择三个模型动作。不是 shape `[9]`，也不是仅长为 3 的数组；保留完整区间让对齐关系清楚。

本脚本没有 `--normalize-advantages`，不会额外做跨 DP 的 masked 白化。组内 reward normalization 与全局 advantage whitening 是不同步骤。KL loss 和 entropy 系数都为 0，不增加这两项梯度；ref 的计算成本仍可能存在。

### 6.4 梯度累积与更新时机

Megatron 对计划中的每个 micro-batch 执行前向和反向，在一次全局 batch 的梯度累积结束后做 optimizer.step。中间 forward 只是读取同一版本参数，不能每过一个 micro-batch 就把它算成一次模型更新。

脚本 Adam 参数为学习率 `1e-6`、一阶矩系数 0.9、二阶矩系数 0.98、weight decay 0.1。常量学习率不代表更新幅度恒定，Adam 还根据梯度历史调整坐标尺度。训练后 `weights_backuper.backup("actor")` 将**更新后的**参数存成 CPU 快照，后续换权消费者从它读取；不能将这个调用误解为只有“训练前回滚备份”。

## 7. 换权：两张训练分片如何变成两个推理副本

### 7.1 先重建与转换，再共享

默认 `full + nccl` 配置遇到 colocate 时会选择 `UpdateWeightFromTensor`，IPC 是按拓扑推导的内部路径，没有单独的 `--update-weight-transport=ipc` 选项。

```text
actor CPU backup
  → HfWeightIteratorDirect 统一参数计划
  → 训练侧 TP all-gather 重建完整矩阵
  → convert_to_hf 拆 QKV/gate-up、改名、处理 dtype/padding
  → FlattenedTensorBucket 连续 GPU buffer + 名称/shape/offset
  → 序列化 CUDA IPC handle
  → Gloo gather_object 汇集小元数据
  → Ray/HTTP 控制请求交给 SGLang
  → 引擎打开共享 buffer，加载进自己的模型参数
```

NCCL 用于训练 TP 的 GPU collective；Gloo 是这里用于小对象协调的通信后端。`gather_object` 收的是句柄，不是用它把几 GB 参数重建起来。CUDA IPC（进程间 CUDA 内存共享）避免将本体经 CPU 对象存储中转，但引擎通常仍需复制到自己的参数空间，不能叫完全零拷贝。

### 7.2 两种拓扑不要混淆

本例训练 TP=2，推理 engine0/1 各 TP=1。训练侧先重建完整权重，再分别更新两个 engine。最后一跳使用共置 IPC；这不否认训练内部的 all-gather 使用 NCCL。

若额外存在训练 GPU 范围之外的 remote engine，tensor updater 还能对那部分走 NCCL。因此“用了 tensor updater”不代表整个拓扑中绝不会有远端权重广播。

### 7.3 何时能释放 bucket，何时能继续生成

发送端创建的 GPU allocation 必须活到接收端完成读取/加载；仅把 IPC handle 发出去就复用内存，会让接收端读到已被覆盖的内容。updater 等相关调用完成后才能清理 producer buffer。

权重更新前暂停生成、清理旧缓存；所有目标桶安装完成后再继续。KV cache 是旧权重对历史上下文的计算结果，换权后一般不能直接沿用。`onload_weights` 只是恢复权重存储，为覆盖新参数做准备；不是一调用它就保证推理已能安全服务。`onload_kv` 在发布后恢复 KV pool，下一轮才具备完整执行条件。

## 8. 将较长轨迹的字段重新算一遍

初始 prompt 120 token，三段 assistant 分别 30、25、20 token，两个工具观察分别 50、40 token。省略额外模板头后：

$$
P=120,\quad R=30+50+25+40+20=165,\quad
L=120+165=285,\quad N_{valid}=30+25+20=75.
$$

| 字段 | 内容/形状 | 生存阶段 |
|---|---|---|
| `tokens` | `[285]`，完整上下文 | 生成记录 → CPU batch → GPU forward |
| `loss_mask` | `[1]*30+[0]*50+[1]*25+[0]*40+[1]*20`，shape `[165]` | 对齐 response，直到归约/反向结束 |
| `response_length` | 165 | 切 response 的位置依据 |
| `reward` | 一个标量，如 1.0 | 生成结束时评分 |
| `rewards` | 每条一个标准化标量 | rollout 转换后供优势计算 |
| old/ref/current logprob | 完整 response 视角 `[165]`；CP 时可能局部切分 | 相应评分或训练阶段产生 |
| `advantages` | `[165]` | loss 前构造，工具位置由 mask 排除 |

`sum(loss_mask)=75` 是有效动作数，绝不能写成 `response_length=75`。如果每个有效位置 loss 都为 2，该轨迹按有效 token 平均为 `150/75=2`；错除以 165 会得到约 0.909，凭空削弱这条工具轨迹的学习权重。

数据源的唯一执行编号也不能从任务 ID 直接推导。任务 42 可以出现在任意采样位置，同题八条应有八个可区分执行身份；任务号用于找环境，执行号用于排序、追踪和归约。当前示例重写 index 的边界见 §4.3。

## 9. 评测与复现：配置名字不等于实际任务分区

脚本每 5 轮请求 `retail-dev` 数据集、每题采样一次、top-k=1。top-k=1 限制模型只选最高分候选，但外部模拟用户仍可能随机，整个 agent 任务不会因此自动确定。

当前 `generate_with_tau.generate` 没有 `evaluation` 参数，`tau_config.task_split` 固定为 `"train"`，所以仅把 eval JSONL 路径命名为 dev，不能保证 `get_env` 真正在 dev 分区读取对应任务。应分别核对“数据文件里的索引”和“环境按哪个 split 解释这些索引”。这是当前适配层的实际边界，不能把脚本意图写成已经完成隔离的 dev 评测。

调试可先选两题、每题两次，保存实际请求、工具观察、token/mask 和环境最终状态，再回放固定轨迹检查 loss。若连样本身份、状态枚举、评分形状和 eval split 都未核实，扩大到 32×8 只会放大问题；若固定数据训练正确而在线不同，再检查采样记录、环境状态、换权和并发选择。

源码阅读入口依次为 [启动脚本](<../../../../../examples/tau-bench/run_qwen3_4B.sh>)、[单样本适配](<../../../../../examples/tau-bench/generate_with_tau.py>)、[agent 循环](<../../../../../examples/tau-bench/trainable_agents.py>)、[token delta](<../../../../../examples/tau-bench/token_delta.py>)、[主循环](<../../../../../train.py>)、[训练 actor](<../../../../../slime/backends/megatron_utils/actor.py>)。用这些具体边界逐项核对，才能判断观察到的是模型学习、任务变化，还是数据与执行协议的偏差。
