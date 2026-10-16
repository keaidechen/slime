# 性能、排障与工程实战

相关基础：[通信与 tensor 基础](<../../../communication/两卡通信与torchrun.md>)。


## 1. 先定义指标

训练性能至少记录：

- step time；
- samples/s 与 tokens/s；
- tokens/s/GPU；
- model FLOPs utilization（说明 FLOPs 口径）；
- peak allocated/reserved memory；
- data、forward、backward、optimizer、checkpoint 分段；
- p50/p95/p99 step time 和各 rank 最大值。

straggler（慢参与者）是明显晚于其他参与者到达同步点或完成工作的进程／设备。同步操作受最晚到达者影响，原因可能是数据、计算、资源配额或网络。

只看 rank0 平均值会掩盖 straggler；分布式 step 由最慢 rank 决定。

### 先固定“完成了多少工作”再算吞吐

设一次 update 消费 8 个样本，每样本 padded length=4，其中有效预测 token 共 28 个，用时 0.1 s，共 W=8 张 GPU。样本吞吐是 `8/0.1=80 samples/s`，有效 token 吞吐是 `28/0.1=280 tokens/s`，每卡为 `280/8=35 tokens/s/GPU`。若改按所有位置计算，则是 `8×4/0.1=320 positions/s`。两种口径都能报告，但不能把补齐比例不同的实验混在一起比较。

MFU（Model FLOPs Utilization）=`F_model/(t_step×W×P_peak)`。F_model 是约定的本步有效模型运算次数，P_peak 是单卡对应精度的峰值 FLOP/s，分子分母单位相消。假设 F_model=1600 FLOPs、t=0.1 s、W=8、每卡峰值 4000 FLOP/s，则 MFU=0.5。这个缩小的数值例子不代表真实硬件。dense Transformer 常用 `6×参数量×token数` 近似主体训练 FLOPs，但长序列 attention、MoE 激活比例、词表等需另算，不能无条件套用。

重计算执行的额外 FLOPs 通常不计入 F_model，否则“做更多重复工作”也会看起来提高模型有效利用率。硬件忙碌率与 MFU 同样不同：忙着通信或重复计算，未必在推进更多有效训练工作。

## 2. 容量估算

先拆五项：

```text
参数 + gradient + optimizer states
+ activation
+ 通信/融合 workspace
+ CUDA graph/allocator 碎片
+ dataloader/pinned host memory
```

MBS（Micro Batch Size）是每个数据并行参与者一次前反向处理的样本数；GBS（Global Batch Size）是一次逻辑更新的全局样本数。固定均匀批次时 GBS=MBS×DP 大小×累积次数。

参数状态受 DP shard、TP/PP/EP shard 和 dtype 影响；activation 受 MBS、seq、hidden、layers/stage、SP/CP、recompute 影响。估算只是上界起点，最终以 allocator snapshot 和 profiler 验证。

峰值应写成时间函数：`M_peak=max_t Σ_j M_j(t)`，M_j(t) 是时刻 t 第 j 类实际存活的设备存储字节数。各类别各自的最大值之和可作为保守上界，但通常不是实际峰值；共享 storage 的 view 又不能重复计数。pinned host memory 需要单独的主机预算，不能加到 HBM 使用量里。

例如常驻状态 10 GiB，forward 的激活 6 GiB、workspace 2 GiB，阶段峰值为 18 GiB；optimizer 时激活已释放，新增临时空间 5 GiB，则为 15 GiB，整步峰值是 18 而不是 `10+6+2+5=23 GiB`。若异步通信让旧 activation 尚未释放，optimizer 的峰值可能重新上升；生命周期决定哪些项可以相加。

## 3. 从现象到层次

| 现象 | 优先检查 |
|---|---|
| GPU 周期性空洞 | PP bubble、dataloader、同步日志/checkpoint |
| collective 尾部很长 | rank imbalance、拓扑、消息大小、NIC/GPU mapping |
| 小 kernel 极多 | fusion fallback、shape 过碎、MoE token 少 |
| OOM 发生在 optimizer step | master/moment materialization、param AG、临时 buffer |
| OOM 只在首几个 PP rank | warmup activation、多层分配不均 |
| loss 从第一步不同 | 初始化/RNG、数据、mask、loss normalization、精度 |
| 若干步后发散 | optimizer state、scaler、FP8 scale、通信 race |
| hang 无报错 | 更早 rank exception、collective 次序/shape、P2P 不匹配 |

## 4. profiling 方法

### 粗分段

先用 Megatron timers 与日志找 data/forward/backward/optimizer/checkpoint 大类。不要一上来抓超长 trace。

### PyTorch profiler

截取 warmup 后 3–5 个稳定 step，带：

NVTX 是为 NVIDIA 分析工具标记事件和区间的接口。它让时间线上的操作更容易对应业务阶段，标记区间本身不会自动等待 GPU 完成。

- CPU/CUDA activity；
- shape（短 trace 才开）；
- memory；
- NVTX ranges；
- rank 与并行配置元数据。

计时还要区分 CPU 与 GPU：在异步 kernel 调用前后读主机时钟，可能只测到提交耗时。设备区间应使用正确 stream 上的 CUDA events，读取 elapsed time 前确保终点完成；跨 stream、跨 rank 的整步则需要相容的边界。为测量而在每个算子后全局同步，会破坏原有 overlap，应只在实验边界或专门测量中使用。

### Nsight Systems

用于看：

- CUDA stream 并行；
- NCCL 与 GEMM overlap；
- CPU launch gap；
- rank 间时间线；
- P2P 与 pipeline bubble。

Nsight Compute 只在已锁定单个热点 kernel 后使用。

## 5. 配置差异怎样帮助区分原因

单卡 BF16、无 fusion/overlap 的参考路径减少并行与精度变量。DP 增加样本分配和梯度归约，TP 增加层内分片与通信，PP 增加跨阶段依赖，CP/EP 分别增加上下文与专家路由。FP8、overlap、recompute/offload 和 CUDA Graph 又各自改变数值、时序或内存生命周期。

这些层次提供一种因果分解：如果基础路径成立而某个组合失效，新增边界就是优先解释对象。但模型、全局 batch 和输入必须相容；同时改变 batch、并行度与精度会产生多个竞争解释。命令、commit、环境、初期 loss 和 trace 共同限定比较，不能把排障顺序当作所有配置都受支持的保证。

## 6. 一个合格的并行方案评审

给定方案必须写清：

1. GPU 拓扑与 rank placement；
2. TP/PP/CP/EP/DP 及进程组；
3. 每 rank 参数、optimizer、activation 估算；
4. microbatch 数和 PP bubble；
5. 各 collective 的消息大小/频率/网络层级；
6. checkpoint 大小、频率、恢复目标；
7. accuracy baseline 与性能验收门槛；
8. 失败回滚配置。


## 7. 三种代码变化怎样影响系统


### Transformer submodule：局部替换也有分布式契约

通过 `ModuleSpec` 改变 MLP 激活，会影响单卡数值、TP shard 与梯度、PP 保存恢复，也可能改变 fusion。局部输出对齐只是第一层证据，分片和状态字典仍需保持模型语义。


### Pipeline 重平衡：算得均匀不等于峰值更低

非均匀 layout 可以按每层耗时分配 stage，减少最慢 stage 的等待；同时每 stage 在途激活、参数和通信可能变化。气泡、峰值显存与 checkpoint key 的对应关系需要一起解释。


### Collective 挂起：最后停住的调用未必是第一因

如果一个 DP rank 跳过一次 collective，其后操作序列会与其他 rank 分歧。op sequence、distributed/NCCL 日志和最早异常控制流提供定位依据；只看最后等待的 NCCL kernel，会把协议不一致误判成通信实现故障。

## 8. 上线前 checklist

- 固定代码、容器、驱动、CUDA/NCCL/TE 版本；
- 至少一次保存—恢复连续性测试；
- 参数 hash/关键 logits 或短程 loss 对齐；
- 目标规模 soak test；
- rank/host/GPU/NIC 映射记录；
- OOM、节点失败、存储变慢的处置预案；
- 性能 dashboard 同时有吞吐、延迟分位数、网络、显存、错误和 checkpoint 指标。

## 9. 把 step time 分解成可行动项

先把源码入口固定下来：

- `Megatron-LM/megatron/core/timers.py`：分布式 timer 与 barrier/log 语义；
- `Megatron-LM/megatron/training/training.py:training_log`：训练日志如何汇总 loss、scale、grad norm、timer；
- `Megatron-LM/megatron/training/theoretical_memory_usage.py`：模型状态与 activation 理论估算；
- `Megatron-LM/megatron/core/rerun_state_machine.py`：异常数值的检测/重放；
- `Megatron-LM/megatron/core/fault_injector.py`：故障注入入口。

建议为每个 PP rank 建：

```text
T_step =
  T_data_wait
  + T_exposed_forward_compute
  + T_exposed_backward_compute
  + T_exposed_TP/CP/EP_comm
  + T_pipeline_bubble
  + T_optimizer_exposed
  + T_logging/checkpoint
```

“exposed”很重要：一个 NCCL kernel 用了 5 ms，但完全藏在 8 ms GEMM 后面，对 step time 的边际影响接近 0。优化应针对 critical path，而不是 profiler 中累计时间最大的类别。

例如 GEMM 在 `[0,8] ms` 执行，通信在 `[2,7] ms` 执行，两者累计 13 ms，但这段墙钟只有 8 ms；把通信优化到 3 ms，不一定改变完成时刻。若通信是 `[6,11] ms`，消费者在 8 ms 后等到 11 ms，则暴露尾部为 3 ms。此时减少尾部才直接缩短关键路径。若争用带宽使 GEMM 从 8 变成 10 ms，重叠图形更漂亮也不保证更快。

这也是 Amdahl 定律的直观来源：总时间中只有比例 f 可以加速 s 倍，新的相对时间为 `(1-f)+f/s`，加速比为它的倒数。f=0.2、s=2 时，总体只有 `1/(0.8+0.1)≈1.11` 倍，而不是两倍。所有比例必须从相同端到端时间口径得到。

## 10. rank straggler 定位

每 step 收集各 rank timer 后，不要立刻 all-reduce average；至少保留 max 与 max-rank：

```python
local = torch.tensor([elapsed_ms], device="cuda")
all_values = [torch.empty_like(local) for _ in range(world)]
dist.all_gather(all_values, local)
```

生产不应每 step all-gather，可低频采样。发现 rank 17 慢后继续对齐：

- host/NUMA/GPU/NIC mapping；
- GPU clocks/ECC/Xid；
- dataloader shard；
- NCCL channel/topology；
- 该 PP stage layer/MoE token 数；
- storage/checkpoint writer。

“先逐步取最慢 rank，再求分位数”与“先对每个 rank 求均值，再取最慢”并不等价。两个 step 的 rank 耗时分别为 `[10,14] ms`、`[16,10] ms`，逐步完成时间是 `[14,16]`，均值 15 ms；各 rank 均值为 `[13,12]`，再取最大却只有 13 ms，漏掉了轮流出现的慢参与者。

p50 表示一半观测不超过该值，p95 表示 95% 不超过该值；样本很少时尾部分位数不稳定。3–5 步 trace 用来观察机制，较长的无 profiler 运行才适合测吞吐与尾延迟，二者的职责不同。

## 11. Nsight 时间线阅读例

理想 overlap：

```text
compute stream: [GEMM bucket n backward][GEMM bucket n-1 ...]
comm stream:              [RS bucket n][RS bucket n-1]
```

如果 RS 只在所有 backward 结束后排成一串，检查：

1. `overlap_grad_reduce` 是否真正启用；
2. bucket 是否被设为 `None`；
3. gradient hooks 是否及时触发；
4. CUDA stream 是否因同步 API 串行；
5. bucket 太大，直到末尾才 ready。

如果 overlap 存在但 step 不变，通信可能原本就不是 critical path，或新增 buffer/分桶开销抵消。

## 12. OOM 快照推理

观察四个时间点：

```text
模型构建后
第一个 forward 峰值
第一个 backward 峰值
optimizer step / param all-gather 峰值
```

- forward 峰值：activation、attention workspace、CUDA graph；
- backward 峰值：saved tensors、recompute 临时、grad bucket；
- optimizer 峰值：master state、all-gather、cast buffer；
- 第二步才 OOM：异步 handle/graph/cache 泄漏或 allocator 碎片。

记录 `memory_allocated` 与 `memory_reserved`。reserved 很高但 allocated 较低通常是 allocator cache/碎片，不应误判为模型 tensor。

reserved 是分配器向设备保留的内存，allocated 是其中当前活跃 tensor 分配，后者通常包含在前者内，不能相加。`reserved-allocated` 较大可能只是可复用缓存，也可能包含难以复用的碎片，要结合 snapshot 判断。`empty_cache` 只能归还未被活跃 tensor 占用的缓存，不能释放 autograd 仍引用的激活，也不能替代修复保存列表或异步队列不断增长的问题。

## 13. 配置变更的 A/B 模板

```text
Hypothesis:
  TP=8 的 GEMM 太小且跨节点 collective 暴露；
Change:
  TP=4, PP×2，保持 DP/GBS/token budget 不变；
Correctness guard:
  前 20 step loss、grad norm、参数 hash；
Performance guard:
  warmup 10 step，测 50 step，报告 max-rank p50/p95；
Artifacts:
  args.yaml, env, nsys, logs, checkpoint metadata；
Rollback:
  恢复原布局，checkpoint 是否可重分片。
```

若改变 TP 的同时改变 MBS，就无法归因。

## 14. 故障注入清单

- 一个 rank 在 collective 前抛异常；
- 一个节点数据读取延迟 5 秒；
- checkpoint 存储短暂不可写；
- FP8 输入注入 NaN/Inf；
- MoE 极端路由到单 expert；
- 重启时改变 DP；
- SIGTERM 发生在异步 checkpoint；
- NCCL interface 配错。

验收不是“任务最终失败”，而是错误被及时检测、日志指出第一因、资源可清理、checkpoint 不被标记为完整。

## 15. 本章延伸阅读

- [Megatron Core 并行策略与性能优化](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)：官方配置基线。
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：CUDA/NCCL 时间线分析。
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)：短窗口 operator/shape/memory 分析。
- [NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)：拓扑、共享内存、网络与调试变量。
- [Megatron Core On-call Guide](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/developer/oncall.md)：跟随上游维护者的故障处理入口。
