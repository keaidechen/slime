# 并行等价性：数学、运行时与性能的三层边界

同一个分布式计算需要满足数学目标、运行时协议和资源约束。CPU 参考说明公式与布局，真实 backend 决定 hook、通信与数值，性能证据还需要固定 workload；这三层不能互相替代。

## 1. 数学参考解释哪些不变量

[并行参考](<../../../../examples/parallel_runtime_walkthroughs.py>)与 [ETP 参考](<../../../../examples/etp_reference.py>)只依赖 Python 标准库，分别表达基本并行和专家张量并行的数学关系。

第一份的断言覆盖 `ddp_fsdp`、`tensor_parallel`、`pipeline`、`sequence_context`、`experts_mesh`、`kv_runtime` 六项 PASS。第二份的断言命名说明其覆盖范围：

```text
PASS etp_forward_backward
PASS etp_finite_difference
PASS ep_etp_dispatch_combine
PASS dense_expert_mesh
CPU mathematical references passed; no distributed backend was executed.
```

| 数学对象 | 参考关系 | 边界 |
|---|---|---|
| DDP/FSDP | 不等 token 数的梯度分母、分片更新、状态内存算术 | 没有实际 hooks 或 optimizer backend |
| TP | 分片前反向与未分片参考/有限差分 | 小矩阵，未测 GPU GEMM |
| PP | 前反向依赖、任务数量、在途激活峰值 | 离散时隙模型，不含真实 P2P 延迟 |
| SP/CP | norm 数值、CP 输出、online merge、远端梯度 | 没有调用 TE attention |
| EP/ETP | degree=1/2/4 的输出/梯度、非均匀路由、空 expert、逆置换、网格 | probability 固定；没有验证 router 参数梯度或真实 A2A |

这些程序的 PASS 只表示相应参考模型中的断言成立；脚本没有实际执行 NCCL、TE 或 GPU GEMM。


## 2. 仓库已有真实 GPU 入口

[tests/test_qwen3_0.6B_parallel_check.py](<../../../../../../tests/test_qwen3_0.6B_parallel_check.py>) 是已有 dense 集成验证入口：

它要求仓库 GPU 测试环境及 8 GPUs，会准备模型/数据并执行多次训练，不是本地 CPU smoke test。脚本保存/重放 rollout 数据，对照梯度范数，并覆盖 DP、TP/PP/CP 组合以及 `calculate_per_token_loss` 两种模式；组合运行包含 SP。它不等于 SP off/on 的逐 tensor 对照，也没有因此覆盖 MoE/ETP/FSDP。

MoE 可从 [Qwen3-30B-A3B 启动脚本](<../../../../../../scripts/run-qwen3-30B-A3B.sh>) 阅读配置入口，它显式使用 EP 和 ETP=1。该脚本是有环境、模型和资源前提的训练作业，不能当作两卡 ETP 教学测试。Megatron-FSDP 的实验应使用支持它的 Megatron 训练入口；不能假设 slime 的任意模型脚本接受全部 FSDP 配置。

已有数学校验的证据仅在 CPU 参考层；本文不据此给出 GPU trace 或吞吐数字。

## 3. 真实执行增加了哪些条件

实际执行的比较需要同时固定或解释以下状态：

```text
代码 commit、容器、PyTorch/TE/CUDA/NCCL、GPU 和物理拓扑
完整启动命令、展开后的参数、所有 group members
初始化后的全局参数、输入 token IDs、mask/position、有效 token 数
GBS/MBS、microbatch 数、DP/CP/EP/ETP/EDP、精度与 RNG 策略
固定输入的一步 outputs/loss/gradients/updated weights
```

不只固定 seed：并行初始化、dropout stream、归约次序可能不同。先用 dropout=0 和可比较的全局初始参数，明确绝对/相对误差容限，再引入随机性或低精度。


## 4. 八类并行的配置与失效边界

下表把配置映射到状态和失效信号。它不表示这些特性可以任意组合；固定 world size 后，DP/EDP 常随其他 degree 改变。

| 方式 | 对照变量/配置入口 | 关键观测点 | 预期与失败信号 |
|---|---|---|---|
| DP | 数据副本数；固定全局样本与目标分母 | grad hook、bucket start/finish、更新前 main_grad | 全局梯度对齐；误差随 DP 成倍变化常提示归一化错误 |
| FSDP | `--use-megatron-fsdp`、`--data-parallel-sharding-strategy` | unit pre-forward/pre-backward、param gather、grad reduce | 参数读取前已 materialize；收益看持久状态和瞬时峰值；checkpoint 格式核对 `fsdp_dtensor` |
| TP | `--tensor-model-parallel-size` 1/2 | FC1/FC2、dX reduction、重建 dW | shard 合起来等于全局矩阵；只 scatter 未 reduce 会丢贡献 |
| PP | `--pipeline-model-parallel-size`；VPP layout | 每个 `(microbatch,chunk,F/B)`、send/recv、激活队列 | 每项一次 F/B，更新前完成 accumulation；不能只看单 rank 调用数 |
| SP | 固定 TP=2，`--sequence-parallel` off/on | norm 边界、AG/RS、norm grad finalize | 激活布局变，数学输出不变；norm grad 漏同步会少 token 贡献 |
| CP | `--context-parallel-size`、`--cp-comm-type` | 全局 token IDs、KV 协议、dK/dV、token denominator | 重组后输出/梯度对齐；causal/packed 泄漏或缩放错误需分开定位 |
| EP | `--expert-model-parallel-size`；固定路由做对照 | routing map、两次 permutation、EP splits | 同 token 恢复原位，top-k 加权一次；空 expert 也不能破坏 collective 顺序 |
| ETP | `--expert-tensor-parallel-size` 1/2，显式记录 EP/EDP | expert TP group、FC1/FC2 shard、AG/RS、EDP gradients | 拼回的权重梯度与基线一致；不是把不同 shard 的 dW 相加 |

EP/ETP 的普通 AlltoAll、AllGather、Flex 后端需分别验收。检查固定版本的配置校验，不把八种技术的存在理解为任意组合均受支持。

## 5. 为什么需要 rank、布局与操作序号

跨 rank 日志会交错。单 rank 的记录只有同时携带操作身份和张量归属，才能重建分布式顺序：

```text
iteration, microbatch, layer/chunk, global_rank
group_name, group_members, op_sequence, collective
global_shape, local_shape, dtype, token_ids 或 shard_offset
expert_id, input_splits, output_splits, tokens_per_expert
参数/梯度摘要、同步提交与完成边界
```

不要为每个生产操作加 barrier。barrier 会改变 schedule 和 overlap；调试 instrumentation 仅在隔离小运行中开启。对异步通信区分提交、真正完成、消费者等待，不用 CPU launch 时间冒充设备通信耗时。

## 6. 数值等价之后，性能收益仍需独立解释

先完成一步数值等价，再 warmup，再采少量稳定 step。比较相同模型、目标 batch 和 workload 分布，报告 max-rank step time 的 p50/p95、tokens/s/GPU、allocated/reserved 峰值及观察到的通信尾部。

| 假设 | 支持它需要看到什么 | 不能当作结论的替代指标 |
|---|---|---|
| SP/FSDP 解决 OOM | 相同工作负载下真实峰值下降，完成一步更新 | 只计算持久参数字节 |
| PP/VPP 降低 bubble | 时间线上空闲区减少，端到端 step 改善 | F/B 次数相同 |
| ETP 值得开启 | 容量获益或总 step 改善，且不是变更 batch 的效果 | 单独一个 GEMM 耗时变小 |
| EP 网络是瓶颈 | splits/热点与关键路径 A2A 尾部对应 | 累计 NCCL 时间很大 |
| overlap 有效 | 消费者等待缩短且额外 buffer/launch 未抵消收益 | 两个 kernel 在图上有重叠 |

采集路径见[通信等待与重叠案例](<../../../../07_Practice/30_训练综合案例.md#s1>)。最终结果写成“模型/输入/硬件/配置 + 数值误差 + 峰值 + step time + 证据范围”；缺测量项保留为未测。
