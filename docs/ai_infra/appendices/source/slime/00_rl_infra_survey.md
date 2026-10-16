# RL Infra 的设计空间：编排、数据流与策略版本

<details>
<summary>文章导航</summary>

- [1. 为什么 RL infra 是一个独立的工程领域](<#1-为什么-rl-infra-是一个独立的工程领域>)
- [2. 知识点地图（Taxonomy）](<#2-知识点地图taxonomy>)
- [3. 从框架实例理解不同设计边界](<#3-从框架实例理解不同设计边界>)
- [4. 比较系统时的四个维度](<#4-比较系统时的四个维度>)
- [5. 设计改变后，瓶颈如何转移](<#5-设计改变后瓶颈如何转移>)
- [6. 知识点 → slime 代码索引](<#6-知识点--slime-代码索引>)

</details>

相关基础：[系统概览](<../../../06_Post_Training/25_RL目标与系统角色.md#s1>)。


不同 RL 框架都要连接生成与参数更新，差异主要在资源如何复用、样本如何流动，以及允许多大的策略版本差。本文保留这些设计问题，具体功能与发布状态由各项目的一手文档维护。

---


## 1. 为什么 RL infra 是一个独立的工程领域

### 1.1 RL post-training 与传统 SFT 的本质区别

infra 是 infrastructure（基础设施）的简称，指让数据生成、评分和参数更新能可靠执行的程序、资源与通信系统。模型的 forward（前向）从输入计算输出，backward（反向）沿计算图求参数梯度，optimizer（优化器）再按梯度改变参数。

监督微调（SFT）是"数据静止、模型动"：数据提前准备好，训练框架只需要做 forward/backward。

RL post-training（RLHF / RLVR / Agentic RL）是"数据由模型自己产生"：

```
循环每一步（rollout step）：
  1. rollout：用当前 policy 模型对一批 prompt 生成回答（推理 workload）
  2. scoring：计算 reward（规则验证器 / reward model / 环境反馈）
  3. advantage：由 reward 构造每个 token 的训练信号
  4. train：用 PPO/GRPO 等算法更新模型（训练 workload）
  5. weight sync：把更新后的权重同步回推理引擎，进入下一轮
```

这一个循环把**两套异构系统**（训练框架 + 推理引擎）耦合在一起，由此增加了动态数据、策略版本和资源阶段切换之间的协调问题。

### 1.1.1 从四条回答理解训练闭环

先给缩写落到具体对象上：RL 是 reinforcement learning（强化学习），SFT 是 supervised fine-tuning（监督微调），policy（策略）是模型“在当前上下文下选下一个 token”的概率分布。RLHF 用人类偏好提供反馈，RLVR 用可验证的结果提供反馈；它们描述反馈来源，不是资源调度方案。actor 在算法里指策略模型，在 Ray 里指远程有状态对象，两种含义要根据上下文区分。critic（价值模型）预测未来奖励，reference（参考策略）提供稳定的概率参照，reward model（RM，奖励模型或规则评分器）评价已经完成的回答。

现在有两道题，每题采样两次：

| prompt 组 | 两条输出 | 规则奖励 | 相对比较告诉我们什么 |
|---|---|---|---|
| “1+1=?” | “2”“3” | `[1,0]` | 第一条优于本组平均，第二条劣于平均 |
| “2+2=?” | “4”“4” | `[1,1]` | 两条都好，但组内无法区分谁更好 |

奖励不是正确答案的标签序列：评分器只给整条结果一个数，训练还必须把它变成对模型动作的信号。GRPO 的组内比较解决这个问题，05 篇会手算标准差、优势和梯度方向。生成时保存 token 和概率，是为了以后能回答“模型当时究竟做了什么”；训练后发布新权重，是为了下一轮回答分布真的发生变化。漏掉发布，即使训练 loss 正常下降，推理端也会一直用旧模型出题给自己练。

同一轮可以有两个 prompt、四条轨迹、多个 micro-batch（一次前反向处理的子批）和一个或多个 optimizer step（参数真正更新一次）。它们是不同计数单位，不能把日志中的所有 step 都理解为同一件事。

### 1.2 生成长尾为何可能支配端到端时间

生成占比取决于模型、回答长度、奖励与资源配置，不能把论文某个工作负载的比例当成 RL 的固定常数。rollout 容易成为瓶颈的原因包括：

- 推理是自回归的，长思维链（CoT，chain of thought，模型生成的多步推理文本）动辄几千到几万 token；
- 同一批样本生成长度差异极大（**长尾 long-tail**），同步系统里最快的样本要等最慢的；
- 训练集群的并行切分方式（TP 是张量并行，把层内矩阵切开；PP 是流水线并行，把层分开；EP 是专家并行，把混合专家模型的专家分开）与推理引擎的最优切分不同，两边都不能简单复用对方的最优配置。

因此 RL infra 的主线几乎全部围绕一件事：**让 rollout 更快、让训练与 rollout 之间的衔接更紧密**。下文的知识分类全部由这条主线派生。

先用墙钟时间建立性能尺度。设生成和评分共花 $T_g$ 秒，训练花 $T_t$ 秒，换权花 $T_w$ 秒；假定三段没有重叠，其他开销已忽略，则

$$
T_{\mathrm{sync}}=T_g+T_t+T_w.
$$

例如四条回答形成一批，生成评分 6 秒、训练 3 秒、换权 1 秒，每批用 10 秒，吞吐为 $4/10=0.4$ 条/秒。如果训练和下一批生成有独立资源、能充分重叠，稳态每批的理想下界变为 $\max(6,3)+1=7$ 秒，吞吐约 0.571 条/秒；加速比是 $10/7\approx1.43$，不是两段相加就能省掉全部训练时间。启动、收尾、队列和资源争用还会使实际时间更长。

还要比较资源成本：两卡运行 10 秒是 20 GPU·秒，四卡运行 7 秒是 28 GPU·秒。后者延迟更短，单位样本成本却可能更高。若更快完成的样本恰好都是短而简单的题，接受样本分布也会改变，因此“每秒 token 更多”需要与有效训练样本数、任务质量一起报告。

### 1.3 RL infra 的十大工程问题

PPO（Proximal Policy Optimization，近端策略优化）比较当前与旧策略对同一动作的概率比，并通过裁剪目标等方式抑制某些过大的更新收益；裁剪不是对最终参数距离的硬保证。

GRPO（Group Relative Policy Optimization，组相对策略优化）用同一任务多条候选的奖励比较构造优势，通常不依赖独立价值模型。组定义、标准差、概率比与损失归约仍需具体说明。

| # | 问题 | 一句话描述 |
|---|---|---|
| 1 | 编排（orchestration） | 训练进程、推理进程、数据进程如何组织与通信（Ray / 单控制器 / 服务化） |
| 2 | rollout 架构 | 推理引擎跑在训练进程里（engine 模式）还是独立服务（server 模式） |
| 3 | 权重同步 | 每轮把新权重从训练侧搬进推理引擎，如何减少传输、布局转换和停顿 |
| 4 | 显存管理 | colocate 时训练与推理共享 GPU 显存，如何错峰、offload、释放 KV cache |
| 5 | 长尾治理 | partial rollout、over-sampling、中断续写，避免一步被最长样本拖死 |
| 6 | 异步与 off-policy | 训练/推理解耦提升吞吐，代价是数据变"旧"（staleness），如何算法补偿 |
| 7 | RL 算法 | PPO / GRPO / GSPO / DAPO 等的 loss、advantage、KL、clip 的正确高效实现 |
| 8 | 训练后端 | Megatron 还是 FSDP2；TP/PP/EP/CP/SP 并行；sequence packing；optimizer 显存 |
| 9 | agentic 支持 | multi-turn、tool call、sandbox 环境交互的轨迹生成与 loss mask |
| 10 | 工程化 | 可复现、容错、调试、trace、profiling、CI——RL bug 往往不报错只降智 |

---


## 2. 知识点地图（Taxonomy）

这一节把领域知识组织成 10 个子领域，每个子领域给出：核心问题、主流方案、代表性工作。后续 slime 走读的 01–09 篇分别跟踪相关实现，其他专题继续分析数据平面与格式转换。

### 2.1 编排与资源管理 → 走读文档 01

**核心问题**：一个 RL 任务里有训练 worker 池、推理 server 池、数据 buffer、reward 服务，谁来调度谁？

三种主流组织方式：

- **单控制器 + 多进程 worker**（verl / HybridFlow）：driver 进程用单进程 Python 写控制流，计算以 RPC 形式 fan-out 到 worker 组。优点是控制流可读，数据用张量中心的 `DataProto` 协议传递。
- **按角色组织 Ray actor**（slime / OpenRLHF）：训练与 rollout 通过远程对象调用协作。actor 描述执行边界，不代表系统没有控制器；slime 的 `train.py:train()` 仍负责驱动生成、训练与换权的顺序。独立重启还需要状态恢复协议。
- **独立服务与其他 actor 运行时**：PrimeRL 等系统分离训练、推理与环境生命周期，TorchForge 使用 Monarch 组织分布式角色。文件、消息或张量服务都能承载交换，关键在于身份、所有权与完成条件。

资源分配上的关键概念：

- **colocate（训推共置）**：训练与推理放同一批 GPU，靠"先推理、释放显存、再训练"错峰复用。省卡、省跨机带宽，但显存要精打细算。
- **disaggregate（训推分离）**：训练与推理各占独立 GPU 池，可独立扩缩容、容忍异构硬件，代价是权重同步要走网络。
- **placement group**（Ray 概念）：把一组 GPU/CPU 资源请求组织成多个 bundle，作为资源组预约；actor 选择 bundle，具体拓扑仍由放置策略与 rank/device 映射共同决定。

### 2.2 Rollout 架构：Engine 模式 vs Server 模式 → 走读文档 02

| 维度 | Engine 模式（同进程） | Server 模式（独立服务） |
|---|---|---|
| 代表 | verl HybridEngine、OpenRLHF | slime 默认 SGLang 路径、verl AgentLoop/AsyncServer、PrimeRL |
| 通信 | 内存/NCCL 直传 | HTTP/RPC |
| 优点 | 可减少进程间序列化、支持原地重分片的实现 | 故障隔离、独立扩缩、多实例负载均衡、天然支持 multi-turn 与外部环境 |
| 缺点 | 角色耦合更紧；多轮等待与资源复用需要额外调度 | 有网络与序列化成本 |

server 模式下的关键组件：

- **router（推理路由）**：多个推理 server 前的负载均衡层。RL 场景有特殊需求，如 multi-turn agent 要求同一 session 落到同一台 server（**session affinity**，因为 KV cache 在 server 本地）。
- **PD 分离（prefill/decode disaggregation）**：把 prefill 与 decode 拆到不同资源组，应对 multi-turn  workload 中两者资源需求不同的问题。
- **控制接口**：RL 需要普通 serving 没有的能力——热更新权重、释放/恢复显存（`release_memory_occupation`）、中断请求（`abort_request`）续写、按 token 返回 logprob。SGLang 与 vLLM 都为 RL 专门提供这类端点（如 SGLang 的 `update_weights_from_distributed`、vLLM 的 weight transfer NCCL engine）。

### 2.3 权重同步（weight sync）→ 走读文档 04

先区分几种传输词汇：NCCL 是 NVIDIA Collective Communications Library，提供多 GPU 的广播、归约等集合通信；CUDA 是 GPU 执行与内存管理平台，IPC 是 inter-process communication（进程间通信）；P2P 指设备之间直接传输，RDMA 指 remote direct memory access（远程直接内存访问）。它们解决数据怎样搬的问题，不能替代参数布局和版本协议。

每轮训练后要把新权重送进推理引擎。模型大小、布局差异和更新频率决定这一步是否进入关键路径，常见优化方向包括：

1. **resharding（重分片）**：训练侧与推理侧并行切分不同（如训练 TP=8 PP=2、推理 TP=2），权重需要先变换分片布局。verl 的 3D-HybridEngine 原地 resharding 近乎零拷贝（仅 engine 模式可行）。
2. **NCCL 广播**：训练 rank 与推理 rank 组一个临时 NCCL 通信组，逐桶广播。slime、OpenRLHF、NeMo-RL、torchrl（`SGLangWeightSyncScheme`）都用此路。
3. **CUDA IPC**：同机进程共享显存句柄，避免把大张量序列化到 CPU；引擎通常仍需将共享 buffer 的内容复制到自己的参数存储。slime 的 colocate 路径使用这一机制。
4. **P2P / RDMA**：跨机绕过 CPU，将张量传输与 host 调度开销分开处理；具体收益依赖规模与网络拓扑。
5. **分桶 + flatten**：把几百个参数张量合并成大桶一次传输，减少启动开销。slime 的 `FlattenedTensorBucket` 即此类。
6. **FP8 量化传输**：传输前把 BF16 权重主体从每元素 2 字节量化为 FP8 的 1 字节，另需 scale 与元数据；slime 支持 FP8 rollout（DeepSeek-V3 系原生 FP8 checkpoint）。
7. **delta weight sync**：训练/推理完全分离（甚至异构 GPU）时，通过共享文件系统做 full/delta checkpoint 更新。
8. **中间件化**：MoonshotAI 的 checkpoint-engine 把权重同步做成引擎无关的独立组件。

### 2.4 长尾治理与 rollout 吞吐 → 走读文档 03

同步 rollout 的一步时间 = 最长那条样本的时间。治理手段：

- **partial rollout（部分 rollout）**：一步结束时未完成的样本不丢弃，把已生成前缀缓存，下一步用新权重续写。slime 原生支持（配 `/abort_request`）。
- **over-sampling / APRIL**：主动多投请求，凑够一批后中断或缓存未完成请求，以减少长尾等待；收益取决于长度分布、缓存复用及额外生成成本。
- **dynamic sampling**（DAPO 提出）：过滤掉全对/全错的 prompt 组（组内相对奖励项的 advantage 为 0，额外 KL/熵项仍可能有信号），补采新样本。
- **speculative decoding for rollout**：由草稿模型提议、目标模型验证 token；收益依赖接受率和验证成本，在线训练草稿模型还会增加状态同步。
- **流式生成 + 中断续写**：SSE 流式返回，中断时部分结果已在客户端，不依赖引擎返回已生成文本（slime 的 `sglang_streaming_rollout.py`）。

### 2.5 异步与 off-policy → 走读文档 03

两个"异步"要分清：

1. **async rollout（请求级）**：一批样本内部可乱序推进；若最后仍用 `gather` 等整批，它照样等待最慢者。需要超投、收够即停等机制才能缩短批尾等待。
2. **async train / fully async（系统级）**：训练与 rollout 完全解耦，推理用"稍旧"的权重持续产数据，训练消费缓冲区数据（AReaL、slime fully_async、NeMo-RL async GRPO、TRL v1.0 async GRPO）。

异步的代价是 **off-policy**：生成数据的策略 ≠ 当前训练策略。主流补偿手段：

- **staleness 控制**：限制样本最大滞后步数（AReaL 的 max staleness、slime 的 off-policy 步数参数）。
- **重要性采样修正**：TIS（token-level importance sampling，用 rollout_log_probs 与 train logprobs 的比值加权）；AReaL 的 decoupled PPO objective（行为策略与目标策略解耦）。
- **算法护栏**：KL 正则、clip-higher（DAPO）、GSPO 的序列级重要性比率（将裁剪粒度改为序列，具体收益依赖模型与任务）。

AReaL 的解耦生成、Magistral 的权重切换与 LlamaRL 的 GPU 直连同步从不同层面缩短等待；吞吐结论应回到各自论文的模型、资源和质量约束。

### 2.6 RL 算法 → 走读文档 05

GAE（Generalized Advantage Estimation，广义优势估计）把多步时序差分残差按 (γλ) 的幂加权合并，用来估计动作相对价值基线的优势。终止与时间截断会决定是否继续递推和 bootstrap。

- **PPO**：actor-critic，GAE 估计 advantage，clip 目标 + KL 惩罚。RLHF 经典配方（InstructGPT）。
- **GRPO**（DeepSeekMath）：去掉 critic，同一 prompt 采样一组（group）回答，组内 reward 归一化当 advantage。省一个 value model 的显存与训练成本，是当前开源 RLVR 的主流。
- **DAPO**（Decoupled Clip and Dynamic Sampling Policy Optimization，解耦裁剪与动态采样策略优化）：组合非对称裁剪、动态补采、token 级归约和超长回答奖励处理。各项解决不同问题，不是只改一个裁剪阈值。
- **GSPO**（Group Sequence Policy Optimization，组序列策略优化）：用长度归一化的序列比率统一裁剪，同一轨迹的 token 变化会共同影响结果，推导见 05 篇。
- **CISPO**（Clipped Importance Sampling Policy Optimization，裁剪重要性采样策略优化）：将裁剪后的重要性比率作为停止梯度的权重乘在 log-probability 上，区别于 PPO 的 clipped surrogate。
- **KL 估计器**：k1（直接 log 比）、k2（平方）、k3（在相应采样分布、支持集合且未截断时无偏）——数值稳定性各不相同。
- **entropy / clip 统计**：训练健康度观测的必需品。

**奖励与评估（RM Hub / Eval Pipeline） → 走读文档 08**：advantage 计算的输入端——规则奖励（数学答案等价性判断、F1/EM、多选题）、外部 RM 服务、group RM、dynamic sampling filter（丢弃全对/全错组）、评测集独立采样参数与 early stop，这些常被视为"业务代码"而非"infra"，但直接决定训练信号质量，slime 把它们做成了统一的可插拔 hub。

### 2.7 训练后端 → 走读文档 06

- **Megatron-LM**：NVIDIA 的大规模训练框架，支持 TP/PP/EP/CP/SP 五维并行，大模型（百亿→万亿）标配。slime、NeMo-RL（经 Megatron Bridge）走此路。
- **FSDP2 / DTensor**：FSDP 是 fully sharded data parallel（全分片数据并行），在数据并行副本间分摊参数、梯度和优化器状态；DTensor 描述分布式张量的切分/复制布局。PyTorch 原生分片，HF 生态亲和，中小规模与快速迭代友好。verl、NeMo-RL（AutoModel 路径）、TorchForge、PrimeRL 走此路。
- **DeepSpeed ZeRO-3**：OpenRLHF 的经典后端（其新后端 Molt 基于 Automodel 进一步扩展）。
- **格式转换层**：Megatron 参数命名/切分与 HuggingFace 完全不同，RL 场景需要高频互转 → 转换层成为独立组件。slime v0.3.1 起使用仓库内建的 `hf_to_megatron/` 与 `megatron_to_hf/`；NeMo-RL 使用 NVIDIA Megatron Bridge，Pai-Megatron-Patch 服务 verl。不要把它们误认为同一条实现路径。
CP（context parallelism，上下文并行）将长序列的上下文计算分给多个 rank；SP（sequence parallelism，序列并行）在 TP 内分摊部分激活，不能把二者当作增加独立样本数的 DP。MoE（mixture of experts，混合专家）则让不同 token 选择部分专家网络计算，EP 分摊这些专家。

- **训练侧优化**：sequence packing、stateless Adam（省优化器状态显存）、CPU offload、梯度 checkpointing。

### 2.8 Agentic RL → 走读文档 07

从"单轮问答 RL"到"多轮智能体 RL"是 2025 下半年以来各框架的主战场：

- **multi-turn 生成**：模型生成 ↔ 工具执行（search/code/sandbox）交替，轨迹 token 来自模型与环境的混合。
- **loss mask**：环境产生的 token（工具返回）不能计入 loss，需要精确的 token 级掩码——coding agent 场景还要求"token-correct trajectory segments"（多段续写的 token 对齐）。
- **环境/沙盒集成**：SWE 类任务需要容器化执行环境（如 AWS AgentCore、RAY 沙盒池）。
- **session 管理**：多轮请求的 KV cache 亲和（router 的 session affinity）。
- **代表工作**：verl AgentLoop、slime 的 search-r1 / coding_agent_rl / multi_agent 示例、ROLL 的 GEM 环境、NeMo-Gym、qqr 的 ArenaRL+MCP。

### 2.9 精度与数值一致性 → 走读文档 04/05

- **FP8 rollout**：推理用 FP8 省显存提速度，但训练是 BF16 → **train-infer 数值不一致**，logprob 有系统性偏差，严格说不再是 on-policy。
- **训练与推理分布校正**：TIS 用行为策略与训练策略的概率关系减轻分布偏差；截断、有限样本和数值误差仍会影响估计，不能据此保证严格无偏或恢复完全相同的 on-policy 数据。
- **rollout_log_probs 回传**：修正的前提是推理引擎把每个 token 的 logprob 随样本一起返回。

### 2.10 工程化与可观测性 → 走读文档 09

- **可复现**：种子、数据顺序、rollout 重放（rollout-only / train-only 分离调试，把 rollout 数据落盘再回放训练）。
- **容错**：rollout server 健康监控、超时重试、故障隔离（server 模式天然优势）。
- **观测**：timer、wandb/metric 记录、Chrome trace viewer、nsys profiling。
- **CI**：RL bug 不报错只降智，因此需要 CPU 单测 + GPU e2e 测试覆盖 dense/MoE、数值精度、async rollout 等关键路径。

---


## 3. 从框架实例理解不同设计边界

下表提炼各项目中值得对照的机制，不是一份实时功能排名。API、模型名单和后端组合会变动，具体支持范围以链接中的发布版本为准。

| 项目 | 值得理解的机制 | 连到哪个系统问题 |
|---|---|---|
| [slime](<../../../../../README.md>) | Ray 编排、SGLang server、Data Buffer、分桶换权、partial/fully async、三层定制接口 | 生成、训练与权重发布怎样形成闭环 |
| [verl](https://github.com/volcengine/verl) | HybridFlow 的单控制器与 worker 组；DataProto；HybridEngine 与 AgentLoop 的不同边界 | 角色复用、布局重分配与多轮对话如何组织 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 按角色拆分 Ray actor，训练与生成各自消费状态 | 模型、奖励、reference 与推理实例如何协作 |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) | AutoModel/DTensor 与 Megatron 路径，生成 backend、replay buffer 和异步执行 | 后端组合与状态格式怎样约束流程 |
| [AReaL](https://github.com/inclusionAI/AReaL) | 解耦生成/训练，可中断 rollout，陈旧度约束与行为/目标策略区分 | 提高吞吐后如何控制 off-policy 偏差 |
| [ROLL](https://github.com/alibaba/ROLL) | 异步执行、算法目标与环境交互共同设计 | 系统时序与 agent 轨迹为何相互影响 |
| [TRL](https://github.com/huggingface/trl) | HF 训练接口与在线生成组件衔接 | 算法代码如何消费生成样本与行为概率 |
| [TorchForge](https://github.com/meta-pytorch/torchforge) | Monarch actor、服务化 RL 组件和分布式状态交换 | 编排与模型实现如何分离 |
| [RLinf](https://github.com/RLinf/RLinf) | 高层工作流到执行任务的映射，仿真、生成和训练资源协作 | 具身任务为何增加环境与渲染状态 |

旧稿将 NeMo RL 的 SGLang、Muon 与投机解码列为统一“规划中”，这个描述已经失效：[v0.6.0 发布说明](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.6.0)已经列出这些能力；其中 SGLang 的训练后端限制必须单独理解，不能推出所有组合都可用。因此本文不再复制滚动 Roadmap。

权重与样本传输还可以成为独立组件。checkpoint-engine、Monarch tensor store 关注状态交换；[TransferQueue](<10_transferqueue.md>)把样本与字段的生产/消费账本同存储后端分开；SpecForge 面向推测生成相关模型，verifiers/PrimeRL 等工作则把环境或跨集群问题纳入系统。slime 的内建 HF↔Megatron 转换与 NVIDIA Megatron Bridge 是独立实现，详见[格式转换](<13_megatron_bridge_internals.md>)。

## 4. 比较系统时的四个维度

| 维度 | 区分的设计 | 代价与边界 |
|---|---|---|
| 编排与资源 | 角色共置、分离部署、actor 或单控制器 | 共置节省容量，但需要显存与执行时序切换；分离增加传输 |
| 生成边界 | 进程内 engine、独立 server、外部服务 | 隔离与伸缩更清楚，也增加网络、请求身份与故障处理 |
| 数据与权重 | 张量协议、独立数据平面、全量/增量发布 | 必须定义所有权、完成条件与版本相容性 |
| 同步程度 | 同步、partial rollout、流水、fully async | 等待减少可能换来陈旧度、背压与校正成本 |

这些维度可以组合，没有哪一个项目名称自动保证更快。比较性能需要相同目标、数据分布、资源预算与质量边界。

## 5. 设计改变后，瓶颈如何转移

独立生成服务让故障隔离和伸缩更直接，同时把权重发布、请求取消和跨进程状态变成显式协议。异步执行减少阶段等待，却需要记录行为策略、样本年龄和消费顺序。partial rollout 与 fully async 分别处理样本长尾和角色耦合，二者并不互相替代。

多轮工具调用、sandbox 和长周期任务又增加 session affinity、token 级轨迹对齐与环境生命周期。低精度生成需要区分数值偏差和真正的 off-policy；扩展到具身或多模态时，渲染、传感与模型执行共同形成关键路径。分析的终点是这些具体约束，而不是“异步必然带来数量级加速”的概括。

## 6. 知识点 → slime 代码索引

| 子领域 | slime 代码落点 | 走读文档 |
|---|---|---|
| 编排与资源管理 | `train.py`、`slime/ray/`（one-step async 见 01 篇历史时序）（placement_group.py / actor_group.py / rollout.py） | 01 |
| Rollout 架构（server 模式） | `slime/rollout/sglang_rollout.py`、`slime/backends/sglang_utils/`、`slime/ray/rollout.py` | 02 |
| 数据流 / Data Buffer / 长尾 / 异步 | `slime/rollout/data_source.py`、`fully_async_rollout.py`、`sglang_streaming_rollout.py`、`slime_plugins/rollout_buffer/` | 03 |
| 权重同步与显存管理 | `slime/backends/megatron_utils/update_weight/`、`sglang.py`（FlattenedTensorBucket）、`kernels/fp8_kernel.py` | 04 |
| RL 算法 | `slime/backends/megatron_utils/loss.py`、`slime/utils/ppo_utils.py` | 05 |
| 训练后端与格式转换 | `slime/backends/megatron_utils/`（`model.py` / `model_provider.py` / `hf_to_megatron/` / `megatron_to_hf/`） | 06 / 13 |
| Agentic RL 与自定义接口 | `slime/agent/`、`examples/`（search-r1 / fully_async / multi_agent / coding_agent_rl） | 07 |
| 奖励模型与评估 | `slime/rollout/rm_hub/`、`slime/rollout/filter_hub/`、`slime/utils/eval_config.py` | 08 |
| 工程化、可观测与 Profiling | `slime/observability/` 与 `slime/utils/`（timer / tracking）、`slime/ray/rollout.py` 健康监控、`tests/` | 09 |
| 独立数据平面（第三方） | `TransferQueue/`（仓库根目录下的 Ascend 开源源码，非 slime 组件） | 10 |
| 推理引擎内部（第三方） | `sglang/`（仓库根目录 vendored 源码）：RL 端点服务端实现 | [SGLang 系列](<../sglang/README.md>) |
| 训练引擎内部（第三方，Megatron-LM） | `Megatron-LM/`（仓库根目录 vendored 源码） | [Megatron 系列](<../megatron/README.md>) |
| HF↔Megatron 转换内部（slime 自身） | `slime/backends/megatron_utils/hf_to_megatron/`、`megatron_to_hf/`、`update_weight/hf_weight_iterator_direct.py` | 13 |

> 注：行号引用以写作时仓库快照为准，上游演进后可能漂移；阅读时建议用符号名搜索定位。
