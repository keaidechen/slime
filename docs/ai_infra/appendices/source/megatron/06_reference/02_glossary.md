# 术语表

相关基础：[通信与 tensor 基础](<../../../communication/两卡通信与torchrun.md>)。


把术语分成三类会更好记：DP/TP 等描述“谁拥有哪部分工作”，AR/AG 等描述“怎样交换数值”，activation/master weight 等描述“什么状态活多久”。同一个 tensor 可以同时有这三类属性，例如“TP 列分片的权重，跨 DP 分片保存主副本，在 forward 前收集”。术语描述的是不同层面，不能互相替代。

下表中的 shard 指一个逻辑张量的切片；replica 指同一逻辑张量或切片的副本。看见两卡形状相同，不能据此判断是哪一种：可能是不同元素的 shard，也可能是相同元素的 replica，还可能是需要求和的 partial sum。

| 缩写/术语 | 中文解释 | 不要混淆 |
|---|---|---|
| MCore | Megatron Core 可组合训练库 | 不等于端到端 Megatron-LM 应用 |
| DP | Data Parallelism，数据并行，切 batch | Distributed Optimizer/FSDP 是 DP 状态管理方式 |
| TP | Tensor Parallelism，张量并行，切单层 tensor | 高频通信，通常局限高速互联域 |
| SP | Sequence Parallelism，在 TP 的部分区域分片序列激活 | 不等于长上下文 CP |
| PP/VPP | Pipeline / Virtual Pipeline Parallelism，流水线/虚拟流水线 | VPP 不增加物理 rank |
| CP | Context Parallelism，上下文并行，切序列工作 | 通信发生在 attention 语义内 |
| EP | Expert Parallelism，专家并行，切 MoE experts | 通常伴随 token dispatch/all-to-all |
| ETP | Expert Tensor Parallelism，切同一 expert 的矩阵 | 使用 expert TP 组，degree 不必等于 attention TP |
| EDP | Expert Data Parallelism，复制同一 expert shard | 同步相同参数的样本梯度，不能用 ETP 组替代 |
| MBS/GBS | Micro/Global Batch Size，单批/全局样本批量 | `GBS=MBS×DP×microbatches`（常规情形） |
| ModuleSpec | 模块结构与实现选择的描述 | 不是序列化 checkpoint schema |
| Distributed Optimizer | 在 DP 组分片 optimizer/主参数等状态 | 不等同完整 FSDP 生命周期 |
| FSDP | Fully Sharded Data Parallel，可配置模型状态分片 | 参数也分片时，unit 粒度决定 materialize 峰值和通信 |
| DCP | Distributed Checkpointing | 重点是 global state 到 shard 的映射 |
| TE | NVIDIA Transformer Engine | 提供 kernel/FP8 等实现，不负责完整训练循环 |
| recompute | backward 前重算 activation | 与把 activation 搬到 CPU 的 offload 不同 |
| CUDA Graph | 捕获并 replay 稳定 CUDA 工作图 | 不自动优化 GPU kernel 本身 |
| MLA | Multi-head Latent Attention，多头潜在注意力，用联合压缩的历史表示组织 K/V | 与 MoE/MTP 是不同架构维度 |
| MTP | Multi-Token Prediction | 训练目标/额外模块，不是推理 speculative decoding 本身 |
| critical path | 决定 step 完成时间的暴露路径 | profiler 中累计耗时最大者未必在关键路径 |

## 通信与执行：同一个数到哪里去

| 术语 | 输入、操作、结果 | 容易出错的理解 |
|---|---|---|
| rank / world size | 进程编号 / 全作业进程数 | rank 不是物理节点号；local rank 另指本机编号 |
| process group | 共同通信的进程集合 | 同样大小的 TP/DP groups 不一定含相同成员 |
| AR / all-reduce | 对齐位置归约，所有成员获得结果 | SUM 与 AVG 的缩放不同 |
| RS / reduce-scatter | 先归约，再给各成员一片结果 | 不等于只 scatter 原输入 |
| AG / all-gather | 将各成员分片拼接到所有成员 | 不求和、不去重、不自动平均 |
| A2A / all-to-all | 每个成员向各目的成员发不同片段 | 各目的地流量可以不均匀 |
| P2P / point-to-point | 匹配的发送与接收 | PP 反向发送的是边界 activation 的梯度 |
| stream / event | 有序设备任务队列 / 依赖与完成标记 | CPU 返回不等于设备完成，跨 stream 需建立依赖 |
| materialize / reshard | 临时重建完整参数 / 回到分片常态 | “完整”可能仍是 TP-local 参数 |
| permutation / unpermute | 为专家计算重排行 / 撤销重排并合并路径 | top-k 复制的 token 必须按原身份加权返回 |

例如 rank0=`[1,2]`、rank1=`[3,4]`：AG 得 `[1,2,3,4]`，SUM AR 得 `[4,6]`，AVG AR 得 `[2,3]`；三个操作都有意义，但维护的是三个不同数学关系。代码中遇到 collective，先写出想要哪一个结果，再核对 group 和算子。

## 张量、模型和状态

| 术语 | 定义与形状示例 | 生命周期或前提 |
|---|---|---|
| token / tokenizer | 离散词表 ID / 文本与 ID 的映射规则 | ID 的含义必须与 embedding 行一致 |
| embedding / logits | ID 查表后的向量 / 未归一化类别分数 | `[B,S,H]` 与 `[B,S,V]` 不同，V 是词表大小 |
| Q/K/V | Query/Key/Value，注意力的查询/键/值投影 | 单 head 为 `[S,d]`，K/V 不等于参数本身 |
| GQA / RoPE | Grouped-Query Attention / Rotary Position Embedding | 分别改变 Q-KV 共享关系和位置表示 |
| MLP / FFN | Multi-Layer Perceptron / Feed-Forward Network，前馈子层 | 按 token 计算，常从 H 扩到 F 再回 H |
| MoE | Mixture of Experts，由 router 选部分专家计算 | 总参数与每 token 激活参数要分开算 |
| activation / saved tensor | 前向中间值 / 反向明确保留的数值 | 不等于全部 forward 输出都须保留 |
| gradient / main_grad | 对目标的导数 / Megatron 的主梯度缓冲视图 | shape 同参数，dtype 可不同，完成归约后才可更新 |
| master weight / m、v | 高精度更新参数 / Adam 一阶和二阶历史 | 属于跨 step 状态，不应每批清零 |
| loss scaling / clipping | 放大 loss 再还原梯度 / 限制梯度范数 | 前者控制动态范围，后者改变过大梯度的幅度 |
| amax / scale | 最大绝对值 / 量化尺度 | scale 可能指乘法系数或其倒数，先查约定 |
| RNG / seed / state | 随机数生成器 / 初始种子 / 当前内部状态 | 恢复中途位置需要 state 与消费顺序 |
| checkpoint / resume | 状态快照 / 从快照接续训练 | 只加载模型参数是 warm start，不保证续训轨迹 |

梯度裁剪的一个例子：g=`[3,4]`，L2 范数为 5，阈值 c=2 时乘 `min(1,c/||g||)=0.4` 得 `[1.2,1.6]`。范数应覆盖一次逻辑梯度，复制参数不能重复计数，不同 shard 也不能遗漏。它与 loss scaling 的“乘 s 后除 s”不是同一种操作。

## 性能量的单位

元素数乘每元素字节数才是内存。FP32 通常 4 bytes/元素，BF16/FP16 通常 2 bytes/元素；临时 workspace 和对齐另计。`1 MiB=2²⁰ bytes`，`1 GiB=2³⁰ bytes`，`1 GB=10⁹ bytes`。

FLOPs 是运算总数，FLOP/s 是速率；latency 是完成一次工作的时间，throughput 是单位时间完成的工作量。microbatch 更大可能让单批 latency 增加却提升 tokens/s，二者不矛盾。HBM 是 GPU 高带宽显存，pinned memory 是便于 DMA（直接内存访问）搬运的页锁定主机内存，不能把它们当成同一容量池。
