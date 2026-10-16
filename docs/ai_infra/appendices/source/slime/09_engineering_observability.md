# 09 工程化与可观测性：从现象定位到可验证的原因

相关基础：[权重发布、异步与资源调度](<../../../06_Post_Training/27_权重发布异步与资源调度.md>)。

训练可以不报错却逐渐变差：模型在正常反向，输入的 reward、token 对齐或权重版本却不正确。可观测性要回答“哪条数据、哪个版本、哪个阶段出了什么问题”，而不是仅展示 GPU 利用率。本篇按隔离问题、核对数值、理解容错、分析时间与内存的顺序展开。

实现以当前工作区为准：计时、trace、指标与 profiler 位于 `slime/observability/`。当前 `TrainProfiler` 只有整体训练 profiler，**没有旧稿中的 `--profile-target train_actor/train_log_probs` 分支**，不要将历史参数直接加入启动命令。

## 1. 先固定输入：把随机闭环拆成两个实验

如果每次排查都重新生成，输出内容、长度和奖励一起变化，很难判断 loss 改变来自代码还是样本。把 rollout 落盘后重复训练同一批，能固定一半变量。

| 入口 | 执行什么 | 适合验证 | 不能据此证明什么 |
|---|---|---|---|
| `--debug-rollout-only` | 生成、评分、样本整理 | 任务数据、token、mask、reward | backward 和优化器正确 |
| `--save-debug-rollout-data` | 保存 rollout 数据 | 固定输入，之后重放 | 外部环境/RNG 全部已快照 |
| `--debug-train-only` 与 `--load-debug-rollout-data` | 用已保存的数据训练 | loss、归约、梯度、checkpoint | 线上换权、生成和训练接口全部正常 |
| `slime.rollout.forge_load.generate_rollout` | 重放数据且保留 SGLang 生命周期 | 共置显存、换权、offload | 真实工具/生成长尾已覆盖 |
| `--save-debug-train-data` | 训练侧调试快照 | 核对实际消费字段与评分 | 完整训练恢复状态已保存 |

文件路径模板按参数帮助配置，rollout dump 与 train dump 应使用不同路径，当前参数校验会拒绝两者相同。`--dump-details` 是另一类详细记录开关，不能替代明确的可回放数据路径。实现入口是 `slime/observability/rollout_data_utils.py` 和 `train_data_utils.py`。

一个最小实验使用 2 个 prompt×2 条回答。先逐条确认 `len(tokens)=P+R`、`len(loss_mask)=R`、reward 与任务对应，再重放同一批算旧概率、当前概率和 loss。最后恢复完整系统，只增加换权与生成这两个变量。这样“固定数据能训对、在线却出问题”会把调查范围缩到生成记录、版本发布、数值对齐与动态调度。

## 2. 数值正确性：先对齐语义，再比较误差

### 2.1 权重检查与版本检查各证明什么

`--check-weight-update-equal` 在启动过程中请求引擎 snapshot/reset，完成首次同步后再 compare，目标是验证加载与传输路径。磁盘发布还有 `get_weight_version` 对账。它们并不等价：版本号一致说明两端声称同一次发布；张量一致性才能进一步检查实际内容。即使张量相同，训练和推理的采样处理、算子精度也可能不同。

对量化权重不应无条件要求与 BF16 原值逐位相等；应比较约定格式下的数值、反量化误差或同输入输出。检查对象、精度与容差要明确，不能只写“权重一致”。

### 2.2 用三项误差定位概率不一致

设同一条轨迹三个可训练 token 的训练 logprob 为 `[-0.7,-1.1,-0.2]`，推理记录为 `[-0.69,-1.15,-0.2]`。逐项绝对差为 `[0.01,0.05,0]`，平均为 0.02 nat，最大为 0.05 nat。若中间 token 的上下文或 token ID 不同，这组误差没有可比意义，应先修对齐。

`train_rollout_logprob_abs_diff` 提供这种偏差观测，但“同参数”不是由指标名字自动保证的前提。排查顺序是：

1. 对齐 token ID、shifted target、mask 和实际上下文。
2. 对齐行为权重版本、训练评分快照、tokenizer、temperature 和 top-p 支持集合。
3. 再对照精度、kernel、packing、MoE 路由和并行归约。

若概率比率为 `[1,1.1,3]`，TIS 上限为 2，则截断后 `[1,1.1,2]`，有效三项中截断率为 `1/3`。它提示修正强度变大，不直接说明模型退化；若拒采后只在剩余 token 上算“被拒比例”，分母会漏掉被拒项而得到误导性的 0。当前 loss 为 mismatch 指标保留修正前的 reducer，正是在维护统计口径。

### 2.3 长度相同也可能语义错误

两条回答的概率数组都是长度 3，把它们交换后长度断言仍会通过。需要同时核对 `index/group_index/rollout_id`、版本与字段归属。全局排序也只能固定已选集合的顺序，不能修复采样集合本身改变。形状断言、数值对账和端到端任务结果提供不同层次的证据。

## 3. 容错：进程恢复只是第一步

推理引擎有权重、KV、在飞请求和通信组状态。健康检查发现引擎异常后，不能仅重启 HTTP 端口便宣布恢复；新进程必须重新加入部署、加载目标权重并恢复通信拓扑，才能参与下一轮发布和采样。

当前主要链路是 `RolloutHealthMonitor` 检测/标记故障，部署层 `RolloutServer.recover` 重建空位，训练侧换权时重新连接新引擎。一个逻辑 TP 引擎跨多个 actor 时，要按逻辑引擎处理故障，不能留下半套 rank 继续参与 collective。

### 3.1 为什么探活必须理解维护窗口

`slime/utils/health_monitor.py` 用两个 `threading.Event`：`_stop_event` 表示退出，`_pause_event` 表示暂停检查。正常主循环为：暂停时等待 → 恢复后先等待预热 → 发健康请求 → 等下一检查周期。`Event.wait(timeout)` 可被 stop 唤醒，比固定 `sleep` 更便于停止；正在执行的网络请求仍受自身超时约束，不能保证任意时刻瞬间退出。

假设探活超时为 3 秒，正常权重加载要 8 秒。若换权期间照常探活，系统可能每次都把正常加载误判成崩溃，触发反复重启。`health_monitoring_pause/resume` 将计划内不可用与真正故障分开；恢复后的 first-wait 给引擎预热留出时间。增加超时减少误杀，也延长发现真实故障的时间，是检测延迟与误报之间的取舍。

### 3.2 三份状态如何一起恢复

训练 checkpoint 保存模型、优化器等可用状态，数据源保存取数游标，推理端需要匹配的新权重发布。若训练恢复到版本 7，数据游标却恢复到版本 5 对应位置，就可能重复训练旧样本；若引擎仍服务版本 6，接下来又产生策略滞后。

外部工具的订单状态、沙盒文件和未完成会话还可能在这三份状态之外。一次请求超时并不证明远端没执行：重试前要按任务协议识别重复操作。框架能重建引擎，不意味着任意 agent 轨迹都具有 exactly-once（恰好执行一次）的保证。CI 中的故障注入只能验证所覆盖的故障窗口。

## 4. 先看墙钟，再看 GPU 时间线

### 4.1 timer、trace、profiler 分别回答什么

- timer 记录命名区间的耗时，例如 `train`、`data_preprocess`、`update_weights`。
- trace 记录请求/阶段的起止与父子关系，适合追踪某个慢样本。
- profiler 记录 CPU 算子、CUDA kernel、内存复制等执行活动，适合解释某区间为何慢。

`slime/observability/timer.py` 使用 host `time()`，本身不插入 CUDA 同步。CPU 提交 kernel 可能仅用 1 毫秒，GPU 5 毫秒后才完成；若区间内没有等待，timer 可能更接近提交时间。已有 Ray 等待、张量回 CPU 或同步点会改变测量边界，所以读数必须连着代码解释。不要为了每个小区间计时都插全局同步，那会破坏本来存在的重叠。

### 4.2 等待比例不是瓶颈判决书

`train_metric_utils.log_perf_data` 由 `train_wait_time` 和 `train_time` 形成：

$$
T_{step}=T_{wait}+T_{train},\qquad f_{wait}=T_{wait}/T_{step}.
$$

两项单位为秒，比例无量纲。若分别为 6 秒、4 秒，则等待比例 60%。这表明训练区间之外的时间较大；它可能包含生成、奖励、换权、checkpoint、offload 和 driver 调度，不能直接断言“GPU decode 慢”。也不能把嵌套的 `actor_train`、`log_probs` 再与整个 `train` 相加，否则会重复计时。

| 观察 | 仍需排除的原因 | 下一条证据 |
|---|---|---|
| 等待比例高 | 工具等待、RM 慢、换权或保存，而非纯生成 | rollout 请求 trace 与换权区间 |
| `actor_train_time` 高 | token 变多、重计算、DP 不均、通信等待 | 长度分布及训练 profiler |
| `data_preprocess_time` 高 | 主机反序列化、H2D、外部路由文件读取 | CPU/拷贝时间线与数据大小 |
| 同步耗时上升 | 桶大小、拓扑、序列化、接收端处理 | 各阶段字节数、调用次数与事件 |

`--max-tokens-per-gpu` 和 `--use-dynamic-batch-size` 控制训练组批，不是直接加速 SGLang decode 的开关。指标归因应先找到对应进程和配置层。

### 4.3 吞吐的分子与分母

若 2 张推理卡在 4 秒内产生 80 个计数 token，吞吐为

$$
80/(2\times4)=10\ \text{token/(GPU·秒)}.
$$

但 response 区间可能含工具 token，计数函数采用哪些字段必须看 `rollout_metrics.compute_perf_metrics_from_samples`；它不总等于模型实际生成数。单条最长样本的 token/秒与全池 token/(GPU·秒) 的分母也不同，不能直接用二者的数值差断言性能差距。

假设总共生成 100 token，只有 60 token 留在有效训练样本，实际用了 10 秒，则生成吞吐为 10 token/s、有效 token 吞吐为 6 token/s。超投让生成吞吐增加而接受率下降时，成本未必改善。应将质量、长度、接受率、总 GPU 数一起记录。

### 4.4 时间线上的重叠要按区间并集算

两条工具请求分别占 `[0,3]` 秒和 `[1,4]` 秒，单条耗时之和为 6 秒，墙钟跨度只有 4 秒。把 `non_generation_time` 相加后从总 rollout 时间中扣除，可能得到负数，不能据此解释 GPU 工作时间。

同理，一次训练耗时 5 秒，其中通信 3 秒、计算 4 秒且重叠 2 秒，总计是 `3+4-2=5` 秒。优化掉已经完全被计算覆盖的 1 秒通信未必缩短 step；优化关键路径上暴露的 1 秒等待才会直接改变端到端时间。

## 5. 逐层采集：当前可用的入口

### 5.1 PyTorch Profiler：训练进程内的证据

当前 `slime/observability/profile_utils.py::TrainProfiler` 在初始化结束后启动 profiler，在训练完成时推进 schedule。可给现有训练命令加入：

```bash
--use-pytorch-profiler \
--profile-step-start 3 \
--profile-step-end 5 \
--tensorboard-dir /root/slime_profile_out
```

这些是命令参数片段，需要放入已配置好的训练命令。当前 schedule 的 `wait=max(start-1,0)`、`warmup=1 if start>0 else 0`、`active=end-start`，因此该例 wait=2、warmup=1、active=2。单位是 profiler 的 `step()` 推进次数；从 checkpoint 恢复时，不要未经检查把它等同于日志中的绝对 rollout ID。

当前实现记录 shape、stack、memory、FLOPs，所有参与的 rank 各自产生 gzip trace，没有本地 `profile_ranks` 过滤分支。它带来额外开销，适合在已复现问题的小规模运行中截取短窗口。输出可解压后用 [Perfetto](https://ui.perfetto.dev/) 读取，或用兼容的 TensorBoard profiler 插件。

看 trace 时，把 GPU 空白区与 CPU 调度、H2D、NCCL（NVIDIA Collective Communications Library，集合通信库）和其它 rank 对齐。一个 `ncclAllReduce` 区间很长，可能在等迟到的 rank，也可能是真正传输慢；仅按 kernel 时间排序无法区分两者。

### 5.2 显存与主机内存：两种账本

```bash
--record-memory-history \
--memory-recorder torch \
--memory-snapshot-dir /root/slime_mem_snapshot \
--memory-snapshot-num-steps 5
```

当前 `record_memory_history` 独立于 PyTorch profiler 开关，不依赖已删除的 `profile-target`。torch 后端在支持的设备 API 上记录分配历史，按配置步数 dump；CUDA OOM observer 可在异常时留快照，不支持该 API 的设备会跳过并记录说明。

假设 PyTorch 活跃 tensor 占 12 GiB、allocator reserved 16 GiB，设备层看到 20 GiB：`16-12=4` GiB 是 PyTorch 保留而未被活跃 tensor 使用的部分；其余 4 GiB 可能来自 NCCL、CUDA context 或其它进程，不能仅靠 PyTorch 快照全部归因。`empty_cache` 释放的是可回收缓存，不会删除活跃 tensor。

torch 快照可用 [PyTorch memory visualizer](https://docs.pytorch.org/memory_viz) 查看。`memray` 后端主要分析主机 Python/native 内存分配，不能把它当 CUDA 显存分配器快照；当前实现要求指定 `--memory-snapshot-num-steps` 来主动结束记录。CPU backup、队列与 Python 对象膨胀适合在主机账本中检查。

### 5.3 SGLang：生成进程要单独抓取

训练 profiler 不会自动覆盖独立 SGLang 进程。`tools/profile_rollout.py` 从 router 查询 worker，再向实际服务发送 `/start_profile` 或 `/stop_profile`：

```bash
python tools/profile_rollout.py \
  --router-url http://127.0.0.1:30000 \
  --action start \
  --output-dir /root/sglang_profile_out \
  --num-steps 5 \
  --activities GPU
```

地址应换成实际 router 地址，输出目录位于引擎所在主机。这里的 step 是推理引擎 forward 步数，既可能是 prefill，也可能是 decode，和训练 rollout 轮数不同。若使用自动停止参数，应确认有足够请求推进这几个 forward；没有负载时不会凭空产生完整采集窗口。

### 5.4 Nsight Systems：先确认采到了哪个进程

NVIDIA Nsight Systems 的 `nsys` 可从系统视角观察 CUDA、NVTX（NVIDIA Tools Extension，语义区间标记）及操作系统线程。需检查运行环境中工具是否已安装，不能假定任意 CUDA 环境都有它。

```bash
nsys profile -t cuda,nvtx,osrt -s none \
  -o /root/nsys_out/train_trace \
  python3 YOUR_MINIMAL_GPU_REPRO.py
```

这里明确要求一个**实际执行 GPU 工作的最小复现程序**。只包裹 `python train.py` 可能仅捕获 driver；Ray worker 可能由已有 Ray 服务另行启动，即使 `--debug-train-only` 也不会自动变成单进程训练。多进程分析必须通过受当前环境支持的 Ray runtime profiling/进程启动配置覆盖实际 worker，并检查报告是否包含目标 PID、CUDA context 与 GPU。

同机时间线容易对齐，多机还需时钟与事件对齐依据。NVTX 标记本身不等待 GPU，区间末尾不能直接当最后一个 kernel 完成时刻。

## 6. 可复现：固定种子、固定顺序、固定集合是三件事

设种子固定后四个任务仍因网络抖动以不同顺序完成，一次超投收下 A/B，另一次收下 A/C。最后按 index 排序只能把 `[B,A]` 变成 `[A,B]`，不能把 `[A,C]` 变成 `[A,B]`。fully async、动态过滤和 early stop 都会增加这种时间相关选择。

更强的复现需固定模型/依赖版本、数据与模板、随机数状态、采样配置，以及必要时重放实际轨迹和环境响应。初始 seed 不等于已推进到某一步的 RNG（随机数生成器）状态；相同 seed 也不保证不同并行拓扑/精度的逐位一致。

当前 `slime/utils/staleness.py` 报告的 age 是 `current_weight_version + 1 - min(sample.weight_versions)`，所以同版本新样本的 age 为 1。文中通常说的版本差 d 则从 0 起算，age=d+1。无法解析或包含未来版本时计为 unknown，不能当作最新样本静默填 0。

## 7. 将证据变成回归检查

CPU 检查适合字段形状、token delta、分组归约、参数映射；GPU 集成检查覆盖训练/推理同输入评分、换权、并行通信、offload 和故障恢复。每种检查应对应一个可解释的不变量。例如改变 micro-batch 划分后，固定数据的归一化梯度应在数值容差内一致；若只是验证函数返回形状，无法发现重复平均导致的学习率变化。

排查时先保存最小输入与预期值，再逐层引入并行、异步和低精度，记录第一次偏离的位置。这样测试、指标与 profiler 才能互相印证：测试说明应当发生什么，指标说明哪里偏离，时间线解释偏离发生在什么执行依赖上。
