# RL Infra 全景调研：知名开源 RL 代码库、近期工作与 Roadmap

> 本文是"从 0 系统学习 RL infra"系列的第一篇：先建立领域地图，再逐篇深入本仓库（slime）的代码。
>
> - 调研对象：slime、verl、OpenRLHF、NeMo-RL、AReaL、ROLL、TRL、TorchForge、RLinf，以及若干支撑性项目（checkpoint-engine、APRIL、SGLang/vLLM 的 RL 能力等）。
> - 阅读建议：先读 §1 建立问题直觉，读 §2 建立知识分类，再按需翻 §3 的各框架细节。§6 给出每个知识点在 slime 代码中的落点索引。

---

## 1. 为什么 RL infra 是一个独立的工程领域

### 1.1 RL post-training 与传统 SFT 的本质区别

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

这一个循环把**两套异构系统**（训练框架 + 推理引擎）耦合在一起，由此产生一系列 SFT 不存在的工程问题。

### 1.2 核心矛盾：rollout 占了 80–90% 的时间

verl（HybridFlow 论文）与 APRIL 论文都指出：**RL 训练 80% 以上的时间花在样本生成上**。原因是：

- 推理是自回归的，长思维链（long CoT）动辄几千到几万 token；
- 同一批样本生成长度差异极大（**长尾 long-tail**），同步系统里最快的样本要等最慢的；
- 训练集群的并行切分方式（TP/PP/EP）与推理引擎的最优切分不同，两边都不能简单复用对方的最优配置。

因此 RL infra 的主线几乎全部围绕一件事：**让 rollout 更快、让训练与 rollout 之间的衔接更紧密**。下文的知识分类全部由这条主线派生。

### 1.3 RL infra 的十大工程问题

| # | 问题 | 一句话描述 |
|---|---|---|
| 1 | 编排（orchestration） | 训练进程、推理进程、数据进程如何组织与通信（Ray / 单控制器 / 服务化） |
| 2 | rollout 架构 | 推理引擎跑在训练进程里（engine 模式）还是独立服务（server 模式） |
| 3 | 权重同步 | 每轮把新权重从训练侧搬进推理引擎，TB 级参数如何秒级完成 |
| 4 | 显存管理 | colocate 时训练与推理共享 GPU 显存，如何错峰、offload、释放 KV cache |
| 5 | 长尾治理 | partial rollout、over-sampling、中断续写，避免一步被最长样本拖死 |
| 6 | 异步与 off-policy | 训练/推理解耦提升吞吐，代价是数据变"旧"（staleness），如何算法补偿 |
| 7 | RL 算法 | PPO / GRPO / GSPO / DAPO 等的 loss、advantage、KL、clip 的正确高效实现 |
| 8 | 训练后端 | Megatron 还是 FSDP2；TP/PP/EP/CP/SP 并行；sequence packing；optimizer 显存 |
| 9 | agentic 支持 | multi-turn、tool call、sandbox 环境交互的轨迹生成与 loss mask |
| 10 | 工程化 | 可复现、容错、调试、trace、profiling、CI——RL bug 往往不报错只降智 |

---

## 2. 知识点地图（Taxonomy）

这一节把领域知识组织成 10 个子领域，每个子领域给出：核心问题、主流方案、代表性工作。后续每一篇 slime 代码走读文档（01–09，另有 08 篇专讲奖励模型与评估）对应其中一个子领域。

### 2.1 编排与资源管理 → 走读文档 01

**核心问题**：一个 RL 任务里有训练 worker 池、推理 server 池、数据 buffer、reward 服务，谁来调度谁？

三种主流组织方式：

- **单控制器 + 多进程 worker**（verl / HybridFlow）：driver 进程用单进程 Python 写控制流，计算以 RPC 形式 fan-out 到 worker 组。优点是控制流可读，数据用张量中心的 `DataProto` 协议传递。
- **Ray actor 对等模型**（slime / OpenRLHF）：训练、rollout、buffer 都是 Ray actor，互相持有 handle 直接调用，没有上帝视角的控制器。优点是模块边界干净、可以独立重启。
- **去 Ray 的服务化**（PrimeRL + verifiers、TorchForge/Monarch）：训练与推理是完全独立的可执行程序，用文件系统或消息总线握手，面向跨集群甚至跨供应商训练。

资源分配上的关键概念：

- **colocate（训推共置）**：训练与推理放同一批 GPU，靠"先推理、释放显存、再训练"错峰复用。省卡、省跨机带宽，但显存要精打细算。
- **disaggregate（训推分离）**：训练与推理各占独立 GPU 池，可独立扩缩容、容忍异构硬件，代价是权重同步要走网络。
- **placement group**（Ray 概念）：把一组 GPU/CPU 资源"捆"成一个原子单位（bundle），actor 按 bundle 放置，保证拓扑（如同一 actor 的各 rank 落在指定卡上）。

### 2.2 Rollout 架构：Engine 模式 vs Server 模式 → 走读文档 02

| 维度 | Engine 模式（同进程） | Server 模式（独立服务） |
|---|---|---|
| 代表 | verl HybridEngine、OpenRLHF | slime（唯一模式）、verl AgentLoop/AsyncServer、PrimeRL |
| 通信 | 内存/NCCL 直传 | HTTP/RPC |
| 优点 | 零序列化开销、权重可原地 resharding | 故障隔离、独立扩缩、多实例负载均衡、天然支持 multi-turn 与外部环境 |
| 缺点 | 同步批处理导致队头阻塞；multi-turn agent 场景几乎不可用 | 有网络与序列化成本 |

server 模式下的关键组件：

- **router（推理路由）**：多个推理 server 前的负载均衡层。RL 场景有特殊需求，如 multi-turn agent 要求同一 session 落到同一台 server（**session affinity**，因为 KV cache 在 server 本地）。
- **PD 分离（prefill/decode disaggregation）**：把 prefill 与 decode 拆到不同资源组，应对 multi-turn  workload 中两者资源需求不同的问题。
- **控制接口**：RL 需要普通 serving 没有的能力——热更新权重、释放/恢复显存（`release_memory_occupation`）、中断请求（`abort_request`）续写、按 token 返回 logprob。SGLang 与 vLLM 都为 RL 专门提供这类端点（如 SGLang 的 `update_weights_from_distributed`、vLLM 的 weight transfer NCCL engine）。

### 2.3 权重同步（weight sync）→ 走读文档 04

每轮训练后要把新权重送进推理引擎，大模型下这是**头号瓶颈**，各家的优化方向：

1. **resharding（重分片）**：训练侧与推理侧并行切分不同（如训练 TP=8 PP=2、推理 TP=2），权重需要先变换分片布局。verl 的 3D-HybridEngine 原地 resharding 近乎零拷贝（仅 engine 模式可行）。
2. **NCCL 广播**：训练 rank 与推理 rank 组一个临时 NCCL 通信组，逐桶广播。slime、OpenRLHF、NeMo-RL、torchrl（`SGLangWeightSyncScheme`）都用此路。
3. **CUDA IPC**：同机场景直接共享显存句柄，免拷贝。slime colocate 模式使用。
4. **P2P / RDMA**：跨机绕过 CPU，如 LMSys 为 Kimi K2 做的 P2P 权重传输，把 1T 参数同步从 53s 优化到约 7s。
5. **分桶 + flatten**：把几百个参数张量合并成大桶一次传输，减少启动开销。slime 的 `FlattenedTensorBucket` 即此类。
6. **FP8 量化传输**：传输前把权重 cast 成 FP8，带宽减半；slime 支持 FP8 rollout（DeepSeek-V3 系原生 FP8 checkpoint）。
7. **delta weight sync**：训练/推理完全分离（甚至异构 GPU）时，通过共享文件系统做 full/delta checkpoint 更新。
8. **中间件化**：MoonshotAI 的 checkpoint-engine 把权重同步做成引擎无关的独立组件，约 20 秒更新万亿参数到数千 GPU。

### 2.4 长尾治理与 rollout 吞吐 → 走读文档 03

同步 rollout 的一步时间 = 最长那条样本的时间。治理手段：

- **partial rollout（部分 rollout）**：一步结束时未完成的样本不丢弃，把已生成前缀缓存，下一步用新权重续写。slime 原生支持（配 `/abort_request`）。
- **over-sampling / APRIL**（RLsys 基金会，2025.9）：主动多投请求（如 130%），凑够一批就"主动截断"多余请求进缓存池，rollout 吞吐平均提升 22.5%（最高 44%）。
- **dynamic sampling**（DAPO 提出）：过滤掉全对/全错的 prompt 组（advantage 为 0、无训练信号），补采新样本。
- **speculative decoding for rollout**：用草稿模型加速生成（NeMo-RL roadmap 中；SGLang 侧 SpecForge）。
- **流式生成 + 中断续写**：SSE 流式返回，中断时部分结果已在客户端，不依赖引擎返回已生成文本（slime 的 `sglang_streaming_rollout.py`）。

### 2.5 异步与 off-policy → 走读文档 03

两个"异步"要分清：

1. **async rollout（请求级）**：一批样本内部乱序完成、不等最长样本（解决长尾）。
2. **async train / fully async（系统级）**：训练与 rollout 完全解耦，推理用"稍旧"的权重持续产数据，训练消费缓冲区数据（AReaL、slime fully_async、NeMo-RL async GRPO、TRL v1.0 async GRPO）。

异步的代价是 **off-policy**：生成数据的策略 ≠ 当前训练策略。主流补偿手段：

- **staleness 控制**：限制样本最大滞后步数（AReaL 的 max staleness、slime 的 off-policy 步数参数）。
- **重要性采样修正**：TIS（token-level importance sampling，用 rollout_log_probs 与 train logprobs 的比值加权）；AReaL 的 decoupled PPO objective（行为策略与目标策略解耦）。
- **算法护栏**：KL 正则、clip-higher（DAPO）、GSPO 的序列级重要性比率（对 MoE 更稳）。

里程碑工作：AReaL（蚂蚁+清华，全异步，约 2.77× 吞吐，NeurIPS 2025）、Magistral（Mistral，热替换权重不打断生成）、LlamaRL（Meta，DDMA GPU 直连权重同步，405B 规模 10.7× 加速）。

### 2.6 RL 算法 → 走读文档 05

- **PPO**：actor-critic，GAE 估计 advantage，clip 目标 + KL 惩罚。RLHF 经典配方（InstructGPT）。
- **GRPO**（DeepSeekMath）：去掉 critic，同一 prompt 采样一组（group）回答，组内 reward 归一化当 advantage。省一个 value model 的显存与训练成本，是当前开源 RLVR 的主流。
- **DAPO**（字节）：clip-higher、dynamic sampling、token-level loss、overlong reward shaping。
- **GSPO**（Qwen）：序列级（而非 token 级）重要性比率，MoE 训练稳定性显著更好；Qwen3 系标配。
- **CISPO**（MiniMax）：clip 重要性采样权重而非比值。
- **KL 估计器**：k1（直接 log 比）、k2（平方）、k3（Schulman 无偏估计）——数值稳定性各不相同。
- **entropy / clip 统计**：训练健康度观测的必需品。

**奖励与评估（RM Hub / Eval Pipeline） → 走读文档 08**：advantage 计算的输入端——规则奖励（数学答案等价性判断、F1/EM、多选题）、外部 RM 服务、group RM、dynamic sampling filter（丢弃全对/全错组）、评测集独立采样参数与 early stop，这些常被视为"业务代码"而非"infra"，但直接决定训练信号质量，slime 把它们做成了统一的可插拔 hub。

### 2.7 训练后端 → 走读文档 06

- **Megatron-LM**：NVIDIA 的大规模训练框架，支持 TP/PP/EP/CP/SP 五维并行，大模型（百亿→万亿）标配。slime、NeMo-RL（经 Megatron Bridge）走此路。
- **FSDP2 / DTensor**：PyTorch 原生分片，HF 生态亲和，中小规模与快速迭代友好。verl、NeMo-RL（AutoModel 路径）、TorchForge、PrimeRL 走此路。
- **DeepSpeed ZeRO-3**：OpenRLHF 的经典后端（其新后端 Molt 基于 Automodel 进一步扩展）。
- **格式转换层**：Megatron 参数命名/切分与 HuggingFace 完全不同，RL 场景需要高频互转 → 转换层成为独立组件。slime v0.3.1 起使用仓库内建的 `hf_to_megatron/` 与 `megatron_to_hf/`；NeMo-RL 使用 NVIDIA Megatron Bridge，Pai-Megatron-Patch 服务 verl。不要把它们误认为同一条实现路径。
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
- **true on-policy**：slime 提出并实践的修正路线——通过 TIS 重要性采样修正 FP8 rollout 引入的偏差，恢复 on-policy 语义。
- **rollout_log_probs 回传**：修正的前提是推理引擎把每个 token 的 logprob 随样本一起返回。

### 2.10 工程化与可观测性 → 走读文档 09

- **可复现**：种子、数据顺序、rollout 重放（rollout-only / train-only 分离调试，把 rollout 数据落盘再回放训练）。
- **容错**：rollout server 健康监控、超时重试、故障隔离（server 模式天然优势）。
- **观测**：timer、wandb/metric 记录、Chrome trace viewer、nsys profiling。
- **CI**：RL bug 不报错只降智，因此需要 CPU 单测 + GPU e2e 测试覆盖 dense/MoE、数值精度、async rollout 等关键路径。

---

## 3. 各框架详细调研

### 3.1 slime（本仓库，THUDM / 智谱）

- **定位**：SGLang-native 的 RL scaling post-training 框架，Megatron 训练 + SGLang server 推理 + Data Buffer 数据流的三分结构。
- **战绩**：GLM-4.5 → GLM-5.2 全系模型的 RL 训练框架；支持 Qwen3 系（含 MoE）、DeepSeek-V3/R1、Llama-3。
- **近期工作主线**（从其文档、README 与生态项目归纳）：
  1. **server-based engine 深化**：SGLang 参数全量透传（`--sglang-*`）、SGLang Config YAML（异构 server group、EPD 部署、multi-model serving）、router session affinity、PD 分离。
  2. **权重同步**：NCCL 广播 + CUDA IPC + 分桶 flatten（Qwen3-30B-A3B 8×H100 约 7s）；**Delta Weight Sync** 支持训推完全分离（可异构 GPU，经共享文件系统做 full/delta 更新）；External Rollout Engines。
  3. **FP8 与 true on-policy**：FP8 rollout（Triton blockwise cast kernel）+ TIS 修正 train-infer 偏差。
  4. **fully async**：`examples/fully_async` 的全异步流水线（独立 asyncio loop + 固定 in-flight 任务池 + ABORTED 样本回收）。
  5. **agentic 接口体系**：三层自定义接口（`--rollout-function-path` / `--custom-generate-function-path` / `--data-source-path`），示例覆盖 search-r1（搜索多轮）、multi_agent、coding_agent_rl（SWE 沙盒）、on-policy distillation。
  6. **工程化**：rollout-only/train-only 分离调试、rollout 数据落盘回放、健康监控、trace/profiling、CPU+GPU 双层 CI。
- **生态**：Miles（RadixArk 企业级 fork）、vime（vLLM 官方基于 slime 的 vLLM-native 版）、Relax（omni-modal 异步）、OpenClaw-RL、P1（物理奥赛）、RLVE、TritonForge、APRIL、qqr、ART（AWS）。
- **Roadmap 信号**：从 README 与博客可见方向是——更大规模 agentic RL（长周期任务）、训推分离形态（delta sync / external engine）、omni-modal（经 Relax）、以及"avoid 割裂的 trainer/rollout/agent framework"的统一数据流。

### 3.2 verl（字节跳动 → 社区）

- **定位**：HybridFlow 论文的官方实现，单控制器（single-controller）+ 多 worker 组的混合编程模型，目前社区最活跃的 RL 框架之一。
- **架构特点**：控制流单进程 Python（可读性好），计算以 RPC fan-out；数据协议 `DataProto`（张量中心）；`ActorRolloutRefWorker` 单体 worker 承载多角色；3D-HybridEngine 在 engine 模式下做原地 resharding。
- **近期工作主线**：
  1. **AgentLoop / AsyncServer**：从 HybridEngine 走向 server 模式的多轮 agentic RL——每个对话独立推进、乱序返回，是 tool-call RL 的必需项；reward 前置（把 RM 打分放进 rollout 阶段）。
  2. **SGLang 深度合作**：SGLang RL Group 在 verl 上构建 multi-turn agentic RL、VLM RLHF、server-based RL、partial rollout。
  3. **训练后端多极化**：FSDP2 + Megatron 双后端（经 Pai-Megatron-Patch/mbridge），sequence packing。
  4. **算法库**：PPO/GRPO/DAPO/GSPO/SPPO 及 entropy 机制等持续跟进。
- **Roadmap 信号**：one-step-off / 异步流水、更强的 agentic 抽象（sandbox、MCP）、vLLM/SGLang 双推理后端对齐。

### 3.3 OpenRLHF（社区）

- **定位**："第一個高性能、生产可用的开源 RLHF 框架"，Ray + vLLM + HF Transformers 的务实组合，易用性优先。
- **架构特点**：按角色拆分 Ray actor（Actor/Critic/Reward/Reference 独立）；Actor 用 vLLM（AutoTP + IPC 权重更新），Critic/Reward 用 DeepSpeed ZeRO-3；混合引擎指多组件共享 Ray placement group。
- **近期工作主线**：
  1. **异步与智能体 RL**：`--async_train` + `--agent_func_path` 一套接口支持 async RLHF 与 agent-based RLHF；`LLMRayActorAsync` 异步 vLLM 推理与环境交互。
  2. **新后端 Molt**：基于 HF Automodel 的新训练后端，官方称比 DeepSpeed 更强，可把 RL 训练扩展到千亿参数级。
  3. **算法覆盖**：PPO/GRPO/RLOO/Reinforce++/DPO/KTO 等广覆盖。
- **Roadmap 信号**：继续降低使用门槛（notebook 级上手）、agent RL 模板化、与 vLLM 新特性（weight transfer engine）对齐。

### 3.4 NeMo-RL（NVIDIA）

- **定位**：NVIDIA 官方后训练库，单 GPU 原型到数千 GPU，HF 生态（DTensor/FSDP2 via AutoModel）与 Megatron Core（经 Megatron Bridge）双路径。
- **已支持**（官方 features 页）：GRPO/GSPO/DAPO、SFT(含 LoRA)、DPO、on-policy distillation；多轮 RL（工具调用/游戏）；FSDP2/TP/CP/SP（DTensor 路径）、Megatron Core 全家桶并行（Bridge 路径）；sequence packing；vLLM 生成；端到端 FP8 训练+生成；VLM SFT/GRPO；异步 RL（异步 rollout + replay buffer + 全异步 GRPO）；NeMo-Gym 环境集成；GB200 容器。
- **官方 Roadmap（进行中/规划）**：
  1. **Muon 优化器**（SFT/RL 引入新优化器）；
  2. **SGLang 推理后端**（目前是 vLLM only）；
  3. 原生 PyTorch 模型训练提速；
  4. 大型 MoE 的训练/生成性能改进；
  5. 新模型（Qwen3-Next、Nemotron-Super）；
  6. 算法扩展（GDPO、GRPO/DPO 的 LoRA 覆盖）；
  7. **弹性/容错**（fault tolerance + auto-scaling）；
  8. **投机解码加速 rollout**。

### 3.5 AReaL（蚂蚁 + 清华 IIIS）

- **定位**：全异步 RL 训练系统的代表作（NeurIPS 2025），system-algorithm co-design 的标杆。
- **核心设计**：
  - 生成与训练完全解耦（无固定配对的生产者-消费者池），吞吐提升约 2.77×；
  - **可中断 rollout**：权重更新后丢弃旧 KV、用新权重重算；
  - **staleness 控制 + staleness-aware 目标**：decoupled PPO objective（行为/目标策略解耦）+ KL 正则 + staleness 过滤，流式队列中每个样本带 staleness 时间戳。
- **近期工作**：AReaL-lite（algorithm-first API，降低算法开发门槛）、AReaL-boba²（效果无损的稳定高吞吐）、与 SGLang/vLLM 深度集成、agentic 场景扩展。
- **Roadmap 信号**：把"异步 + off-policy 修正"做成通用基础设施，向 agentic / 多模态扩展。

### 3.6 ROLL（阿里巴巴）

- **定位**：面向 agentic RL 的大模型 RL 框架（淘天未来生活实验室 + 智能引擎），强调"基建-算法-机理"全栈协同。
- **近期工作**：「3A」协同框架——**Async 异步训练架构**、**AsyPPO 非对称 PPO**、**Attention-based Reasoning Rhythm**（推理节奏机理研究）；GEM 环境系统（游戏化/程序化环境）；多轮工具调用模板。
- **Roadmap 信号**：agentic 场景的系统-算法联合优化（与 slime 生态的 APRIL、RLVE 思路相近）。

### 3.7 TRL（HuggingFace）

- **定位**：HF 生态的 RL 训练库，算法覆盖最广、上手门槛最低，研究原型首选。
- **近期工作主线**：
  1. **vLLM 集成**：GRPO/Online DPO 的在线生成瓶颈 → vLLM server 模式与 **colocate 模式**（"不让任何 GPU 掉队"，2025.6 官方博客）；
  2. **TRL v1.0**：**Async GRPO**（生成与训练重叠）是 v1.0 最重要新特性；
  3. 算法持续扩充（GRPO 变体、Online DPO、KTO、ORPO 等）。
- **Roadmap 信号**：继续"算法超市 + 推理加速"，与 vLLM weight transfer 对齐。

### 3.8 TorchForge（Meta / PyTorch）

- **定位**：PyTorch-native 的 RL post-training 与 agentic 开发库（2025.10 发布），理念是"研究者写算法，不写基础设施"。
- **架构特点**：构建于 **Monarch**（PyTorch-native 分布式 actor 消息框架）之上；清晰的 RL 抽象（Policy、Critic、ReplayBuffer 等服务化组件）；**分布式张量 KV store**（基于 Monarch，自动 DTensor resharding）专门解决权重同步问题；与 vLLM、torchtitan、torchrl 生态联动。
- **Roadmap 信号**：把 RL 组件服务化（weight store、replay buffer、sandboxed tool executor），强化 agentic 开发体验。

### 3.9 RLinf（清华 + 中关村学院 + 无问芯穹）

- **定位**：面向**具身智能**（embodied AI）的大规模 RL 基础设施（"渲训推一体化"），2025.9 发布。
- **核心设计**：**M2Flow（宏-微流变换）**——开发者用过程式接口写高层 RL 工作流（宏逻辑），系统自动映射为底层微流执行；统一支持 VLA 模型、仿真器（渲染）与推理/训练的资源调度。
- **Roadmap 信号**：从 LLM RL 扩展到 embodied / omni-modal agent，是"RL infra 泛化"的代表方向。

### 3.10 支撑性项目与单点工作

| 项目 | 组织 | 解决的问题 |
|---|---|---|
| checkpoint-engine | MoonshotAI | 引擎无关的权重同步中间件，万亿参数约 20s 更新到数千 GPU（Kimi K2 实战） |
| P2P weight transfer | LMSys/SGLang | RDMA P2P 把 1T 参数同步 53s→约 7s |
| APRIL | RLsys Foundation | 主动 partial rollout，rollout 吞吐 +22.5%（最高 44%） |
| SpecForge | SGLang | 投机解码训练框架（EAGLE3），用于 rollout 加速 |
| TransferQueue | Ascend | RL 系统独立数据平面（AsyncFlow 论文）：控制面 Ray actor 维护样本×字段生产状态与样本×任务消费状态，数据面可插拔存储（Mooncake/元戎/RayRDT）；verl 集成（PR #5401）后 e2e 吞吐 +49.1%，被 ROLL（RemoteBatch）、UniRL（KV 接口）、Relax 采用。源码已随本仓库提供，走读见 [10_transferqueue.md](10_transferqueue.md) |
| slime native converter / Megatron Bridge | THUDM / NVIDIA | 两套独立的 HF ↔ Megatron 权重/配置转换方案；slime 当前使用前者 |
| verifiers + PrimeRL | Prime Intellect | 去中心化跨集群 RL（SHARDCAST 权重分发 + TOPLOC 计算验证） |
| Miles / vime / Relax | RadixArk / vLLM / RedAI | 基于 slime 的企业级 / vLLM-native / omni-modal 衍生框架 |

---

## 4. 横向对比

| 框架 | 训练后端 | 推理后端 | 编排 | 异步能力 | 强项 | 适用场景 |
|---|---|---|---|---|---|---|
| **slime** | Megatron | SGLang（server 唯一模式） | Ray actor 对等 | partial rollout、fully async | 大模型生产验证、SGLang 特性全量可用、FP8/true-on-policy | 百亿~万亿参数生产级 RLVR/agentic |
| **verl** | FSDP2 / Megatron | vLLM + SGLang（engine+server 双模式） | 单控制器 HybridFlow | partial rollout、one-step-off（开发中） | 社区最大、后端最全、AgentLoop | 研究+生产通用 |
| **OpenRLHF** | DeepSpeed ZeRO-3 / Molt | vLLM（engine） | Ray actor 按角色拆分 | `--async_train`、agent func | 易用、算法广 | 快速复现 RLHF 实验 |
| **NeMo-RL** | FSDP2 / Megatron Core | vLLM（SGLang 规划中） | Ray | 异步 rollout + replay buffer + 全异步 GRPO | NVIDIA 全栈、FP8 端到端、官方 roadmap 清晰 | NVIDIA 硬件上的生产 |
| **AReaL** | Megatron/FSDP | SGLang/vLLM | 全异步生产者-消费者 | **业界最完整的 staleness-aware 异步** | 系统-算法 co-design | 极致吞吐的研究与生产 |
| **ROLL** | 未公开细节 | 未公开细节 | Async 架构 | Async + AsyPPO | agentic 全栈协同 | agentic RL 研究 |
| **TRL** | HF Trainer/Accelerate | vLLM（server+colocate） | 单进程 | Async GRPO（v1.0） | 算法超市、生态 | 原型与教学 |
| **TorchForge** | torchtitan/FSDP2 | vLLM | Monarch actor | 组件服务化 | PyTorch 原生抽象 | PyTorch 系研究 |
| **RLinf** | 多后端 | 多后端 | M2Flow 宏微流 | — | 具身智能一体化 | VLA/机器人 |

---

## 5. 趋势总结（2025 → 2026）

1. **server 模式成为主流**：engine 模式的性能优势被 SGLang/vLLM 的 RL 专用控制端点（热更新、显存释放、abort）追平，而 server 模式在故障隔离、扩缩容、multi-turn 上的优势是结构性的。连 engine 模式起家的 verl/OpenRLHF 都在向 server/agent-loop 演进。
2. **权重同步中间件化**：checkpoint-engine、Monarch tensor store、TransferQueue 把权重/数据传输做成独立基础设施。
3. **异步是吞吐的下一个数量级**：fully async + staleness-aware 算法（AReaL 路线）与 partial rollout（slime/APRIL 路线）是两大互补方向；TRL v1.0、NeMo-RL 跟进说明已进入大众视野。
4. **agentic 是主战场**：multi-turn tool-call、sandbox、MCP、长周期任务；对应 infra 需求是 session affinity、PD 分离、token 级轨迹对齐、环境服务化。
5. **精度工程**：FP8 rollout 普及后，train-infer 数值一致性（true on-policy / TIS）成为正确性新议题。
6. **RL infra 泛化**：从 LLM 扩展到 omni-modal（Relax）、具身（RLinf）、kernel 生成（TritonForge）等"任意可验证环境"。

---

## 6. 知识点 → slime 代码索引

| 子领域 | slime 代码落点 | 走读文档 |
|---|---|---|
| 编排与资源管理 | `train.py`、`train_async.py`、`slime/ray/`（placement_group.py / actor_group.py / rollout.py） | 01 |
| Rollout 架构（server 模式） | `slime/rollout/sglang_rollout.py`、`slime/backends/sglang_utils/`、`slime/ray/rollout.py` | 02 |
| 数据流 / Data Buffer / 长尾 / 异步 | `slime/rollout/data_source.py`、`fully_async_rollout.py`、`sglang_streaming_rollout.py`、`slime_plugins/rollout_buffer/` | 03 |
| 权重同步与显存管理 | `slime/backends/megatron_utils/update_weight/`、`sglang.py`（FlattenedTensorBucket）、`kernels/fp8_kernel.py` | 04 |
| RL 算法 | `slime/backends/megatron_utils/loss.py`、`slime/utils/ppo_utils.py`（如存在） | 05 |
| 训练后端与格式转换 | `slime/backends/megatron_utils/`（`model.py` / `model_provider.py` / `hf_to_megatron/` / `megatron_to_hf/`） | 06 / 13 |
| Agentic RL 与自定义接口 | `slime/agent/`、`examples/`（search-r1 / fully_async / multi_agent / coding_agent_rl） | 07 |
| 奖励模型与评估 | `slime/rollout/rm_hub/`、`slime/rollout/filter_hub/`、`slime/utils/eval_config.py` | 08 |
| 工程化、可观测与 Profiling | `slime/utils/`（timer / trace_utils / profile_utils / tracking）、`slime/ray/rollout.py` 健康监控、`tests/` | 09 |
| 独立数据平面（第三方） | `TransferQueue/`（仓库根目录下的 Ascend 开源源码，非 slime 组件） | 10 |
| 推理引擎内部（第三方） | `sglang/`（仓库根目录 vendored 源码）：RL 端点服务端实现 | 11 |
| 训练引擎内部（第三方，Megatron-LM） | `Megatron-LM/`（仓库根目录 vendored 源码） | 12 |
| HF↔Megatron 转换内部（slime 自身） | `slime/backends/megatron_utils/hf_to_megatron/`、`megatron_to_hf/`、`update_weight/hf_weight_iterator_direct.py` | 13 |

> 注：行号引用以写作时仓库快照为准，上游演进后可能漂移；阅读时建议用符号名搜索定位。
