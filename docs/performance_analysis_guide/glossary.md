# 大模型性能分析缩写与术语词典

这不是需要背诵的单词表。第一次阅读其他章节时，遇到缩写就回到这里查。每个词条都回答三个问题：全称是什么、中文是什么意思、在性能分析中应该想到什么。

## 1. 最先掌握的 30 个词

| 缩写/术语 | 全称 | 最简单的理解 |
|---|---|---|
| CPU | Central Processing Unit | 运行 Python、调度和数据处理的通用处理器 |
| GPU | Graphics Processing Unit | 执行大规模并行计算的加速器 |
| CUDA | Compute Unified Device Architecture | NVIDIA GPU 编程与运行平台 |
| kernel | CUDA kernel | 在 GPU 上执行的一段函数，不是操作系统内核 |
| op/operator | operation/operator | 框架中的一个算子，如 `aten::mm` |
| trace | execution trace | 带时间戳的执行事件记录 |
| profiler | performance profiler | 收集时间、调用栈或硬件计数器的工具 |
| latency | latency | 完成一次请求/step 所需时间 |
| throughput | throughput | 单位时间完成的请求、样本或 token 数 |
| utilization | utilization | 采样窗口内资源处于忙碌状态的比例 |
| FLOPS | Floating-point Operations Per Second | 每秒浮点运算次数，计算吞吐单位 |
| HBM | High Bandwidth Memory | GPU 上的高带宽显存 |
| SM | Streaming Multiprocessor | NVIDIA GPU 的主要并行计算单元 |
| GEMM | General Matrix Multiplication | 通用矩阵乘法，大模型核心计算 |
| OOM | Out Of Memory | 内存或显存不足 |
| rank | process rank | 分布式任务中一个进程的编号 |
| DP | Data Parallelism | 数据并行 |
| TP | Tensor Parallelism | 张量并行 |
| PP | Pipeline Parallelism | 流水线并行 |
| EP | Expert Parallelism | 专家并行 |
| NCCL | NVIDIA Collective Communications Library | NVIDIA GPU 集合通信库 |
| all-reduce | collective operation | 多 rank 聚合后让每个 rank 得到结果 |
| LLM | Large Language Model | 大语言模型 |
| KV Cache | Key-Value Cache | 推理时保存 attention 历史状态的显存 |
| prefill | prompt processing | 一次处理输入 prompt 的阶段 |
| decode | token generation | 逐步生成输出 token 的阶段 |
| TTFT | Time To First Token | 请求到首 token 的时间 |
| TPOT | Time Per Output Token | 首 token 后平均每个输出 token 的时间 |
| rollout | rollout | 用当前策略生成训练样本的过程 |
| KL | Kullback–Leibler divergence | 衡量两个概率分布差异的量 |

掌握这 30 个词后，就可以开始第 0～2 章；其余词按需查阅。

## 2. 单位与数字

### 时间

| 单位 | 含义 | 换算 |
|---|---|---|
| s | second，秒 | 1 s |
| ms | millisecond，毫秒 | `10^-3` s |
| μs/us | microsecond，微秒 | `10^-6` s |
| ns | nanosecond，纳秒 | `10^-9` s |

比较前必须统一单位。`200 μs` 是 `0.2 ms`，不是 `200 ms`。

### 数据量与带宽

- **B**：byte，字节；**b**：bit，比特。网络常用 Gb/s，内存常用 GB/s，两者不能直接混用。
- **KB/MB/GB/TB**：常被软件按十进制或二进制近似显示；精确报告应注明口径。
- **GB/s**：每秒传输的 GB 数，常用于 HBM、PCIe、NVLink 带宽。
- **Gbps**：gigabits per second，常用于网络链路。

### 运算量

- **FLOP**：一次浮点运算；是运算“数量”。
- **FLOPs**：有些资料也用它表示复数运算数量，容易与下面的 FLOPS 混淆。
- **FLOPS**：floating-point operations per second；是运算“速度”。
- **TFLOPS/PFLOPS**：每秒 `10^12`/`10^15` 次浮点运算。
- **MFU**：Model FLOPs Utilization，模型 FLOPs 利用率；通常是实际模型计算吞吐与理论峰值的比例。不同项目的 FLOPs 公式可能不同，不能只比百分比。
- **HFU**：Hardware FLOPs Utilization，硬件 FLOPs 利用率；可能把 recompute 等实际执行计算计入。必须先看项目定义。

## 3. 计算机与 GPU 硬件

| 缩写 | 全称 | 中文与性能含义 |
|---|---|---|
| CPU | Central Processing Unit | 通用处理器。负责 Python、tokenizer、调度、数据加载、kernel launch 等。 |
| GPU | Graphics Processing Unit | 高并行加速器。训练和推理的主要矩阵/attention 计算发生在这里。 |
| RAM | Random Access Memory | 主机内存，不等于 GPU 显存。 |
| DRAM | Dynamic Random Access Memory | 动态内存；在 GPU 语境中常泛指 device/global memory。 |
| HBM | High Bandwidth Memory | GPU 板载高带宽显存，容量和带宽都是性能约束。 |
| SRAM | Static Random Access Memory | 更靠近计算单元、容量小但快；寄存器/shared memory/cache 属于相关层级。 |
| SM | Streaming Multiprocessor | NVIDIA GPU 调度 thread block/warp、执行指令的核心单元。 |
| Tensor Core | Tensor Core | 加速矩阵乘加的专用单元，支持 BF16、FP16、FP8、TF32 等。 |
| ALU | Arithmetic Logic Unit | 算术逻辑单元。 |
| PCIe | Peripheral Component Interconnect Express | CPU、GPU、NIC 等设备连接总线；权重或数据传输可能经过它。 |
| NVLink | NVIDIA high-speed interconnect | NVIDIA GPU 间高速链路。 |
| NVSwitch | NVIDIA switch fabric | 连接多张 GPU 的交换芯片/系统。 |
| NUMA | Non-Uniform Memory Access | CPU 访问不同 socket 内存的代价不同；数据线程和 GPU/NIC 亲和性会受影响。 |
| NIC | Network Interface Card | 网卡。多机通信要看 NIC、GPU 和 NUMA 拓扑。 |
| IB | InfiniBand | HPC/AI 集群常用低延迟网络。 |
| RoCE | RDMA over Converged Ethernet | 在以太网上承载 RDMA。 |
| RDMA | Remote Direct Memory Access | 绕过较多 CPU 路径的远程直接内存访问。 |
| DMA | Direct Memory Access | 设备直接搬运内存，减少 CPU 参与。 |
| GPUDirect RDMA | GPU Direct RDMA | NIC 与 GPU memory 直接传输数据的技术路径。 |

看到“通信慢”，要先问走的是 NVLink、PCIe 还是跨机 NIC，而不是直接调整 NCCL 参数。

## 4. CUDA 与底层算子

### 4.1 执行层次

| 术语 | 含义 |
|---|---|
| kernel | 在 GPU 上执行的函数。一个 PyTorch operator 可能启动多个 kernel。 |
| thread | GPU 最小逻辑执行线程。 |
| warp | NVIDIA GPU 通常以 32 个 thread 为一组调度。分支不一致会造成 divergence。 |
| block / CTA | 一组可共享 shared memory、可同步的 thread。CTA 是 Cooperative Thread Array。 |
| grid | 一次 kernel launch 的所有 block。 |
| stream | CUDA 操作的有序队列；不同 stream 可能重叠。 |
| event | CUDA 时间戳/同步对象；可用于正确测量 GPU 时间。 |
| launch | CPU 把 kernel 提交给 GPU 的动作。大量极短 kernel 可能 launch-bound。 |

### 4.2 编译与二进制

| 缩写 | 全称 | 含义 |
|---|---|---|
| PTX | Parallel Thread Execution | NVIDIA 的虚拟指令表示，不是最终硬件机器码。 |
| SASS | Shader Assembly | NVIDIA GPU 实际执行的机器指令表示。 |
| cubin | CUDA binary | 包含 GPU 机器代码的二进制对象。 |
| fatbin | fat binary | 可包含多个架构 cubin/PTX 的 CUDA 二进制容器。 |
| IR | Intermediate Representation | 中间表示；Triton/编译器会经历多级 IR。 |
| JIT | Just-In-Time compilation | 运行时编译；首次执行可能包含编译开销。 |
| AOT | Ahead-Of-Time compilation | 运行前提前编译。 |
| NVRTC | NVIDIA Runtime Compilation | CUDA 运行时编译库。 |

### 4.3 性能概念

| 缩写/术语 | 含义 |
|---|---|
| occupancy | 一个 SM 上活跃 warp 相对硬件上限的比例；高 occupancy 不保证高性能。 |
| IPC | Instructions Per Cycle | 每周期执行指令数。 |
| ILP | Instruction-Level Parallelism | 指令级并行。 |
| TLP | Thread-Level Parallelism | 线程级并行。 |
| arithmetic intensity | 算术强度，计算量/内存流量；Roofline 的关键横轴。 |
| compute-bound | 主要受计算单元吞吐限制。 |
| memory-bound | 主要受显存/cache 带宽或访问效率限制。 |
| latency-bound | 主要受依赖链或访问延迟限制，未必达到带宽峰值。 |
| launch-bound | 工作太碎，CPU launch/调度开销占比较高。 |
| coalescing | 相邻 thread 的内存访问能否合并成高效事务。 |
| bank conflict | shared memory 多线程访问同一 bank 产生串行化。 |
| divergence | 一个 warp 的线程走不同分支，路径被分批执行。 |
| Roofline | 用算术强度和性能上界判断 memory/compute 限制的模型。 |

### 4.4 库与工具

| 缩写 | 全称 | 含义 |
|---|---|---|
| CUDA | Compute Unified Device Architecture | NVIDIA GPU 软件与编程平台。 |
| cuBLAS | CUDA Basic Linear Algebra Subprograms | NVIDIA GPU 线性代数库。 |
| cuDNN | CUDA Deep Neural Network library | NVIDIA 深度学习算子库。 |
| CUTLASS | CUDA Templates for Linear Algebra Subroutines | NVIDIA 模板化 GEMM/张量计算库。 |
| GEMM | General Matrix Multiplication | 通用矩阵乘法，通常表示 `C = A × B + C` 类操作。 |
| NVTX | NVIDIA Tools Extension | 给 timeline 添加命名区间/标记。 |
| CUPTI | CUDA Profiling Tools Interface | PyTorch Profiler/Nsight 等收集 CUDA 活动的底层接口之一。 |
| NVML | NVIDIA Management Library | `nvidia-smi` 等使用的设备管理接口。 |
| DCGM | Data Center GPU Manager | NVIDIA 数据中心 GPU 监控、健康与管理组件。 |
| nsys | Nsight Systems CLI | 系统时间线采集/分析命令。 |
| ncu | Nsight Compute CLI | 单 kernel 硬件指标采集命令。 |

## 5. PyTorch 与编译器

| 缩写/术语 | 全称 | 含义 |
|---|---|---|
| PyTorch eager | eager execution | Python 语句执行到算子时立即运行的默认模式。 |
| op/operator | operation/operator | 框架语义算子，例如 `aten::mm`。它不是必然对应一个 kernel。 |
| ATen | A Tensor Library | PyTorch 核心 tensor/operator 库和算子命名空间。 |
| dispatcher | dispatcher | 根据 device、dtype、autograd 等 dispatch key 选择 operator 实现。 |
| autograd | automatic differentiation | 自动微分系统，构建并执行 backward。 |
| forward/fwd | forward pass | 前向计算。 |
| backward/bwd | backward pass | 反向传播，计算梯度。 |
| optimizer | optimizer | 根据梯度更新参数，如 Adam。 |
| AMP | Automatic Mixed Precision | 自动混合精度。 |
| dtype | data type | 数据类型，如 FP32、BF16、FP8。 |
| shape | tensor shape | 张量各维大小；决定 kernel 选择和计算量。 |
| contiguous | contiguous memory layout | tensor 数据是否按当前逻辑顺序连续存储。 |
| fusion | operator fusion | 把多个算子融合，减少内存往返和 launch。 |
| graph break | graph break | 编译器无法继续捕获图，退回 Python/eager 的边界。 |
| recompile | recompilation | 输入条件变化导致编译图重新编译。 |
| FX | PyTorch FX | PyTorch 图表示和变换工具。 |
| Dynamo | TorchDynamo | `torch.compile` 前端，捕获 Python/PyTorch 程序。 |
| AOTAutograd | Ahead-Of-Time Autograd | 提前生成/编译 forward 和 backward 图。 |
| Inductor | TorchInductor | PyTorch 默认编译后端之一，可生成 Triton/C++ kernel。 |
| Triton | Triton language/compiler | 编写和生成 GPU kernel 的语言与编译器；不是 NVIDIA Triton Inference Server。 |
| Kineto | PyTorch profiler backend | PyTorch Profiler 使用的性能采集基础设施。 |
| CUDA Graph | CUDA Graphs | 预先捕获一组 CUDA 工作并低开销 replay；shape/address 通常需稳定。 |

### 常见数值类型

| 缩写 | 含义 | 初学者注意 |
|---|---|---|
| FP32 | 32-bit floating point | 精度较高、显存和计算成本较大。 |
| TF32 | TensorFloat-32 | NVIDIA Tensor Core 的 FP32 输入加速路径之一，不等同普通 IEEE FP32 运算。 |
| FP16 | 16-bit floating point | 范围较小，训练常配 loss scaling。 |
| BF16 | Brain Floating Point 16 | 与 FP32 有相近 exponent 范围，训练常用。 |
| FP8 | 8-bit floating point | 更低精度和更高潜在吞吐，需要 scaling/格式管理。 |
| INT8/INT4 | 8/4-bit integer | 常用于量化推理或权重压缩。 |

理论峰值会随 dtype、是否使用 Tensor Core、稀疏条件而变化，因此“这张 GPU 有多少 TFLOPS”必须同时问是哪种 dtype。

## 6. 分布式训练与 Megatron

### 6.1 进程与组

| 术语 | 含义 |
|---|---|
| rank | 全局进程编号。 |
| local rank | 当前节点内的进程/GPU 编号。 |
| world size | 分布式任务总进程数。 |
| process group / PG | 参与一组 collective 的进程集合。 |
| rendezvous | 分布式进程发现并建立通信的过程。 |
| straggler | 比其他 rank 慢、让大家等待的进程/节点。 |

### 6.2 并行方式

| 缩写 | 全称 | 切分什么 | 常见代价 |
|---|---|---|---|
| DP | Data Parallelism | 每个副本处理不同数据 | 梯度同步 |
| DDP | Distributed Data Parallel | PyTorch 复制模型的数据并行 | all-reduce/reduce-scatter |
| FSDP | Fully Sharded Data Parallel | 参数、梯度、optimizer state 分片 | all-gather/reduce-scatter、额外调度 |
| TP | Tensor Parallelism | 一个层内 tensor/matmul | 高频 collective |
| PP | Pipeline Parallelism | 不同层/stage | pipeline bubble、send/recv |
| CP | Context Parallelism | sequence/context 维 | attention 相关通信 |
| SP | Sequence Parallelism | sequence 维上的部分 activation | 与 TP 配套通信 |
| EP | Expert Parallelism | MoE experts | token all-to-all、不均衡 |

这些缩写描述的是“切分维度”，不是某种神奇加速开关。并行度增大通常降低单卡内存，却增加通信或 bubble。

### 6.3 Batch 与流水线

- **micro-batch**：一次 forward/backward 处理的最小 batch。
- **global batch**：一次参数更新覆盖的总样本数，通常跨 DP 和 gradient accumulation。
- **gradient accumulation**：累积多个 micro-batch 梯度再更新参数。
- **pipeline stage**：PP 中负责一段模型层的 rank 组。
- **bubble**：某个 PP stage 因流水线填充、排空或不平衡而空闲。
- **recompute / activation checkpointing**：少存 activation，在 backward 重新算 forward，省显存但增加计算。
- **overlap**：让通信与有用计算同时进行。时间线有重叠不代表关键路径一定缩短。

### 6.4 通信 collective

| 操作 | 直观含义 | 常见场景 |
|---|---|---|
| broadcast | 一个 rank 发给所有 rank | 初始化参数 |
| all-reduce | 所有 rank 聚合，所有 rank 得结果 | DDP 梯度同步 |
| reduce-scatter | 聚合后每个 rank 只保留一片 | FSDP/ZeRO 梯度分片 |
| all-gather | 每个 rank 的分片收集为完整数据 | 参数/activation 恢复 |
| all-to-all | 每个 rank 向每个 rank 发送不同部分 | MoE token dispatch/combine |
| send/recv | 点对点发送/接收 | PP stage 通信 |
| barrier | 所有 rank 到齐后继续 | 同步；会暴露 straggler |

| 缩写 | 全称 | 含义 |
|---|---|---|
| NCCL | NVIDIA Collective Communications Library | NVIDIA GPU collective 通信库。 |
| Gloo | Gloo collective library | 常用于 CPU collective/control path。 |
| UCX | Unified Communication X | HPC 通信抽象层。 |
| SHARP | Scalable Hierarchical Aggregation and Reduction Protocol | 网络侧 collective 加速技术。 |
| ZeRO | Zero Redundancy Optimizer | 分片 optimizer/gradient/parameter 以减少冗余。 |

## 7. Transformer 与 MoE

| 缩写/术语 | 全称 | 含义 |
|---|---|---|
| Transformer | Transformer architecture | 以 attention 和 MLP 为核心的模型架构。 |
| attention | attention | 根据 query/key/value 计算上下文关联。 |
| Q/K/V | Query/Key/Value | attention 的查询、键和值 tensor。 |
| MHA | Multi-Head Attention | 多头注意力。 |
| MQA | Multi-Query Attention | 多 query head 共享较少 key/value head。 |
| GQA | Grouped-Query Attention | query head 分组共享 key/value head。 |
| MLA | Multi-head Latent Attention | 使用低维 latent 表示的一类注意力结构。 |
| FFN | Feed-Forward Network | Transformer block 中的前馈网络。 |
| MLP | Multi-Layer Perceptron | 在大模型语境常指 FFN 子层。 |
| RMSNorm | Root Mean Square Normalization | 常见归一化层。 |
| RoPE | Rotary Position Embedding | 旋转位置编码。 |
| MoE | Mixture of Experts | 每个 token 只路由到部分 expert 的稀疏模型。 |
| expert | expert network | MoE 中的一个 FFN/子网络。 |
| router/gate | routing/gating network | 决定 token 去哪些 expert。 |
| top-k | top-k routing | 每个 token 选择分数最高的 k 个 expert。 |
| load balance | load balancing | expert 间 token 数是否均衡。 |

## 8. 推理、SGLang 与 vLLM

### 8.1 请求阶段

- **prompt**：输入给模型的 token 序列。
- **prefill**：一次处理输入 token 并建立 KV Cache；也叫 prompt phase。
- **decode**：每个 step 生成下一 token，并读取已有 KV Cache。
- **detokenization**：把 token ID 转回文本。
- **streaming**：生成 token 后逐步返回客户端。
- **scheduler**：选择哪些请求组成下一 batch。
- **continuous batching**：请求动态加入/离开 batch，而非整批一起完成。
- **chunked prefill**：把长 prompt 分块处理，平衡 prefill 与 decode。
- **prefix cache**：复用相同前缀已有的 KV 状态。
- **PagedAttention**：以分页方式管理 KV Cache 的 attention/内存方案。
- **retract/preemption**：资源不足时暂时撤回或抢占请求。
- **speculative decoding**：先由 draft 模型/方法提出多个 token，再由 target model 验证。
- **PD disaggregation**：Prefill-Decode 分离，把两个阶段放在不同 worker/资源。

### 8.2 延迟、吞吐与服务指标

| 缩写 | 全称 | 定义 |
|---|---|---|
| E2E | End-to-End latency | 客户端发请求到完整响应结束。 |
| TTFT | Time To First Token | 请求发出到首 token。 |
| TPOT | Time Per Output Token | 首 token 之后每个输出 token 的平均时间。 |
| ITL | Inter-Token Latency | 相邻流式 token 的间隔分布。 |
| TBT | Time Between Tokens | 常与 ITL 接近，但要看工具定义。 |
| TPS | Tokens Per Second | 每秒 token 数；必须注明 input/output/total、全局还是单卡。 |
| QPS/RPS | Queries/Requests Per Second | 每秒请求数。 |
| concurrency | concurrency | 同时在途请求数。 |
| request rate | arrival/request rate | 每秒到达请求数，不等于并发数。 |
| p50/p95/p99 | percentile | 50/95/99 分位延迟；p99 表示 99% 样本不超过该值。 |
| SLO | Service Level Objective | 内部设定的服务目标，例如 p99 TTFT。 |
| SLA | Service Level Agreement | 对外承诺的服务协议，通常包含处罚或责任。 |

不要只写“TPS=1000”。应写成“8 GPU、并发 32、输入 1024、输出 256 时，全服务 output throughput 1000 tokens/s”。

### 8.3 框架名称

- **SGLang**：面向大模型/多模态模型的高性能 serving 与编程框架；名称本身在使用中通常不展开。
- **vLLM**：高吞吐、内存高效的 LLM serving 引擎；通常作为项目名使用，不必强行展开。
- **Ray**：分布式任务/actor 执行框架，Slime 用它编排训练和 rollout 进程。

## 9. 强化学习与 Slime

### 9.1 角色与数据

| 缩写/术语 | 全称 | 含义 |
|---|---|---|
| RL | Reinforcement Learning | 强化学习，通过奖励优化策略。 |
| RLHF | Reinforcement Learning from Human Feedback | 从人类反馈训练 reward/preference，再优化策略。 |
| RLAIF | Reinforcement Learning from AI Feedback | 使用 AI 反馈的一类 RL 方法。 |
| SFT | Supervised Fine-Tuning | 监督微调，常作为 RL 前的初始化。 |
| policy/actor | policy model | 产生 action/token 的待训练模型。 |
| reference/ref | reference model | 固定或较慢更新的参考模型，用于 KL 等约束。 |
| RM | Reward Model | 给生成结果打分的奖励模型。 |
| critic/value | value model/function | 估计未来 return，用于 advantage。 |
| rollout | rollout | 用当前 policy 与环境交互/生成样本。 |
| trajectory | trajectory | 一段状态、动作、奖励序列。 |
| episode | episode | 从起点到终止的一次完整交互。 |
| reward | reward | 环境或模型给出的标量/分项反馈。 |
| return | return | 从当前位置起累积的折扣奖励。 |
| advantage | advantage | 某动作相对基线有多好。 |
| on-policy | on-policy | 数据由当前或足够接近的 policy 产生。 |
| off-policy | off-policy | 使用其他/旧 policy 产生的数据训练。 |

### 9.2 常见算法和量

| 缩写 | 全称 | 初学者理解 |
|---|---|---|
| PPO | Proximal Policy Optimization | 用 clipped objective 限制一次 policy 更新幅度。 |
| GRPO | Group Relative Policy Optimization | 在同一 prompt 的一组 response 内构造相对 advantage 的方法。 |
| DPO | Direct Preference Optimization | 直接用偏好对优化策略；通常不需要在线 rollout/critic。 |
| GAE | Generalized Advantage Estimation | 在偏差与方差之间折中的 advantage 估计。 |
| KL | Kullback–Leibler divergence | 当前 policy 与 reference 分布差异。 |
| entropy | entropy | policy 分布的不确定性；常用于观察探索/塌缩。 |
| clip ratio/range | clipping ratio/range | PPO/GRPO 限制概率比变化的范围或被截断比例。 |
| log-prob | log probability | token 概率的对数，乘法可转为加法且数值更稳定。 |

### 9.3 Slime 系统词汇

- **train side**：Megatron actor/reference 等训练计算一侧。
- **rollout side**：SGLang 生成、router、reward/tool/environment 一侧。
- **weight update/sync**：把训练后的权重转换并加载到推理 engine。
- **colocate**：训练和推理在同一组 GPU 上分时复用资源。
- **offload/onload**：把模型状态移出/移入 GPU，以便另一侧使用显存。
- **partial rollout**：保留未完成样本并在后续继续的一类策略。
- **fully async**：训练和生成更彻底解耦的异步流程。
- **effective token**：算法定义下真正参与有效目标/统计的 token；以项目代码为准。
- **non-generation time**：tool、环境、reward 等非模型生成时间。

## 10. 性能分析与可观测性

| 缩写/术语 | 含义 |
|---|---|
| benchmark | 用固定 workload 比较性能的实验。 |
| microbenchmark | 只测一个很小组件/算子的 benchmark。 |
| baseline | 改动前可复现的对照结果。 |
| A/B | 除一个变量外保持相同的对比实验。 |
| warmup | 正式计时前运行若干次，排除初始化、编译、cache 建立。 |
| synchronization | 等待异步工作完成；CUDA 正确计时通常需要。 |
| profiler | 收集函数、operator、kernel 或计数器的工具。 |
| instrumentation | 在程序中插桩记录事件，信息精确但有开销。 |
| sampling | 周期性采样调用栈/计数器，开销通常较低但不是完整事件。 |
| trace | 按时间记录的事件集合。 |
| timeline | 把 trace 画在时间轴上的视图。 |
| flame graph | 按采样或分配权重显示调用栈宽度的图，不是时间线。 |
| critical path | 决定端到端完成时间的依赖链。 |
| bottleneck | 限制整体性能继续提升的资源或阶段。 |
| utilization | 资源处于忙状态的比例，不直接等于有效效率。 |
| saturation | 继续增加负载时资源/队列已接近上限。 |
| overhead | 为调度、通信、profile 等付出的额外成本。 |
| regression | 新版本相对基线的性能退化。 |
| observability | 用 metrics、logs、traces 了解系统内部状态。 |

### 常见软件名

| 名称 | 用途 |
|---|---|
| PyTorch Profiler | Python/ATen/CUDA operator trace。 |
| Perfetto | 浏览器中的 trace timeline 查看器。 |
| TensorBoard | 训练 scalar、图和 profiler 数据可视化。 |
| Nsight Systems | CPU、GPU、CUDA、NCCL 和系统级时间线。 |
| Nsight Compute | 单个 CUDA kernel 的硬件计数器分析。 |
| Compute Sanitizer | CUDA 内存错误、race、未初始化访问检查。 |
| py-spy | Python CPU 调用栈采样。 |
| Memray | Python/native 主机内存分配分析。 |
| Prometheus | 抓取并存储 metrics 时间序列。 |
| PromQL | Prometheus Query Language，Prometheus 查询语言。 |
| Grafana | dashboard 和告警可视化。 |
| W&B | Weights & Biases，实验跟踪平台。 |

### 可观测性的三根支柱

- **Metrics**：可聚合的数字时间序列，如 tokens/s、queue length、GPU memory。
- **Logs**：离散文本/结构化事件，如错误、配置、每 step 统计。
- **Traces**：带父子关系或时间顺序的事件，如一次请求跨 router、engine、reward 的路径。

Metrics 告诉你“何时异常”，trace 帮你解释“时间花在哪里”，logs 提供“当时发生了什么和配置是什么”。

## 11. 一词多义和高频误区

### `kernel`

- CUDA 文档中：GPU kernel；
- Linux 文档中：操作系统 kernel；
- PyTorch dispatcher 中：某个 dispatch key 对应的 operator implementation，也可能不是单一 CUDA kernel。

### `step`

- training iteration；
- optimizer step；
- gradient accumulation micro-step；
- SGLang forward/decode step；
- Slime rollout step。

报告必须说明是哪一种 step。

### `batch size`

可能指 micro-batch、local batch、global batch、running decode batch 或 batch tokens。只写“batch=32”不够。

### `throughput`

可能是 samples/s、requests/s、input tokens/s、output tokens/s、total tokens/s、每卡或全局。必须写全口径。

### `memory`

可能是 CPU RAM、pinned host memory、GPU allocated、GPU reserved、KV Cache、CUDA Graph pool、NCCL buffer。OOM 前先确认是哪一种。

### `GPU utilization`

采样窗口内忙不等于 Tensor Core 满载，也不等于有效业务吞吐高。要结合 workload、功耗、memory/compute throughput 和端到端指标。

### `overlap`

时间线上两个事件重叠，只说明同时发生。只有关键路径缩短且正确性不变，才能说 overlap 带来了有效优化。

## 12. 查词后的固定动作

遇到陌生缩写时按这个顺序：

1. 在本文查全称；
2. 确定它属于硬件、CUDA、框架、分布式、推理还是 RL；
3. 查当前项目里这个指标/参数的具体定义；
4. 核对单位、聚合范围和 step 语义；
5. 再阅读 trace 或做 A/B。

不要只根据缩写的字面全称猜行为。性能领域很多指标在不同工具中同名但边界不同。
