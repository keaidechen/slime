# 09 工程化与可观测性：调试、容错、Profiling、CI 与可复现

> 对应综述（`00_rl_infra_survey.md`）§2.10「工程化」。
> README 里有句纲领："RL bug 往往不会立刻报错"——权重没同步、logprob 对不上、数据错位，训练照跑，只是模型悄悄变差。slime 把正确性设施当一等公民。本篇盘点这些设施及对应代码，并把 profiling 部分写得足够细（这是训练慢时最需要动手排查的部分）。

---

## 1. 分离调试：把 RL 拆成两个可独立运行的半系统

最常用的调试手段（`slime/ray/placement_group.py:100-117` 的资源布局分支）：

| 参数 | 效果 |
|---|---|
| `--debug-rollout-only` | 只起 rollout 侧（SGLang + 数据流），不建训练 actor——验证数据生成、reward、partial rollout |
| `--debug-train-only` | 只起训练侧，rollout 数据从落盘的 dump 读——验证 loss、梯度、checkpoint |

配套机制：

- **rollout 数据落盘**：`_save_debug_rollout_data`（`slime/ray/rollout.py:663`）把每轮 rollout 数据 dump 到磁盘（`--dump-details` 等参数控制）；train-only 模式回放这份数据，实现"rollout 一次、训练调一百遍"；
- **`forge_load.py`**（`slime/rollout/forge_load.py`）：加载落盘数据但**不跳过 SGLang**——server、router、权重更新、offload 照常运行，专门用来测真实显存占用；
- **`sleep_rollout.py`**：rollout 侧彻底占位（死循环打日志），纯训练侧联调；
- **save/load 对称**：`MegatronTrainRayActor.save_model`（`slime/backends/megatron_utils/actor.py:542-564`）在 `debug_rollout_only` 时直接跳过；`update_weights`（actor.py:567-569）在两种 debug 模式下都短路。

**工作流建议**：新任务先 `--debug-rollout-only` 把数据和 reward 跑对并落盘 → 再 `--debug-train-only` 回放数据把训练跑对 → 最后全链路小规模 → 放量。

---

## 2. 正确性校验设施

### 2.1 权重同步对账

`--check-weight-update-equal`（train.py:29-30）：

- 启动时 `check_weights(action="snapshot")` + `reset_tensors`（placement_group.py:246-248）在引擎侧记录基准；
- 首轮权重同步后 `check_weights(action="compare")` 逐张量比对训练侧与引擎侧；
- disk 模式还有**版本号对账**（`slime/ray/actor_group.py:254-265`）：CI 中校验每个引擎的 `get_weight_version` 必须等于训练侧发布的版本，不匹配直接 `RuntimeError`。

### 2.2 训推数值监控（持续）

不是一次性校验，而是每步记录（05 篇 §5）：

- `train_rollout_logprob_abs_diff`：同参数下训练侧与推理侧 logprob 的绝对差，是 FP8、kernel 差异、routing 差异的"体温计"（loss.py:1073-1077）；
- `tis / tis_clipfrac / tis_abs / ois`：TIS 修正的强度与截断比例；
- `examples/train_infer_mismatch_helper/`：专门的离线诊断工具。

### 2.3 不变量断言

`Sample` 的长度校验（03 篇）、`add_samples` 的组长度断言（data_source.py:206-209）、`start_rollout_id` 全组一致断言（placement_group.py:216）——把"数据错位"这类 bug 尽量变成启动即报的错误。

---

## 3. 容错

- **健康监控**：`RolloutManager.__init__` 可选启动 `RolloutHealthMonitor`（`slime/ray/rollout.py:470-477`），定期检查 SGLang server 存活；`health_monitoring_pause/resume`（rollout.py:620-627）在权重同步等敏感窗口暂停监控避免误判；
- **故障恢复**：`RolloutServer.recover`（rollout.py:346）重建挂掉的引擎；训练侧下次 `update_weights` 时 `recover_updatable_engines`（rollout.py:601）+ 检测到 `num_new_engines > 0` 后把新引擎加进 NCCL 组（`slime/backends/megatron_utils/actor.py:597-607`）——**引擎热替换不断训**；
- **故障注入测试**：`_try_ci_fault_injection`（rollout.py:479）在 CI 里故意制造故障，验证恢复路径真的可用——"未经测试的容错等于没有容错"的实践；
- **checkpoint 恢复**：训练侧 `start_rollout_id` 对齐 + 数据源游标恢复（03 篇）+ `rollout_manager.load.remote(start_rollout_id - 1)`（placement_group.py:221-222），三层状态一起回到断点。

详细运维文档见 `docs/zh/advanced/fault-tolerance.md` 与 `docs/zh/advanced/reproducibility.md`。

### 3.1 深入拆解：`RolloutHealthMonitor` 的双 `Event` 状态机（`slime/utils/health_monitor.py`）

这是一个教科书式的"用两个 `threading.Event` 实现可暂停后台线程"的实现，值得拆开看：

```python
self._stop_event = threading.Event()    # 一旦 set，线程彻底退出
self._pause_event = threading.Event()   # set=暂停巡检，clear=正常巡检
self._pause_event.set()                 # 初始状态：先暂停（引擎还没就绪，不该立刻探活）
```

`_health_monitor_loop`（105-135 行）的主循环结构：

```python
while not self._stop_event.is_set():
    while self._pause_event.is_set() and not self._stop_event.is_set():
        self._stop_event.wait(timeout=0.5)     # 暂停期：0.5s 轮询一次是否该醒来/退出
    ...
    if self._need_first_wait:                   # 每次从暂停恢复后，先等一段"预热期"
        if self._stop_event.wait(self._check_first_wait):
            break
        ...
    if not self._pause_event.is_set() and not self._stop_event.is_set():
        self._run_health_checks()
    if self._stop_event.wait(self._check_interval):   # 用 wait() 代替 sleep()，可被 stop 立即打断
        break
```

三个值得注意的细节：

1. **用 `Event.wait(timeout)` 代替 `time.sleep(timeout)`**：`wait()` 在等待期间如果 event 被其他线程 `set()`，会立刻返回 `True` 并结束等待；`sleep()` 则必须等满整个时长。这让 `stop()` 能让巡检线程在 5 秒内（`_check_timeout + _check_interval + 5`）就退出，而不必等到下一个巡检周期。
2. **`_need_first_wait`**：每次 `resume()` 后先等 `rollout_health_check_first_wait` 秒（注释"for large MoE models to be ready"）——大 MoE 模型刚从 sleep 恢复、权重刚广播完时，SGLang 引擎需要一点时间才能真正响应推理请求；如果不等这段"预热期"直接探活，会把"正常的启动延迟"误判成"引擎挂了"，进而触发不必要的 `_kill_engine`。
3. **`pause()`/`resume()`**：由 `RolloutManager` 在权重同步、offload/onload 这些"引擎注定短暂不可用"的窗口主动调用（rollout.py 里 555/571/585/607 行的调用点）——**巡检窗口与"引擎正常状态变化窗口"必须错开**，否则健康监控会把"我们自己主动让引擎休眠"误诊为"故障"，反而触发误杀。这是分布式系统里"维护窗口"与"故障检测"必须协调的典型例子。

**`_check_engine_health` → `_kill_engine` 的连锁**：一旦某个引擎的 `health_generate` 请求超时或抛异常，直接 `ray.kill(engine)` 杀掉整组（`nodes_per_engine` 决定一个"引擎"横跨几个 Ray actor/节点，TP>1 的引擎需要连坐杀掉所有 TP rank，否则残留进程会占着 GPU 且状态不完整）；被杀掉的 slot 在 `all_engines` 数组里置 `None`，等下一次 `update_weights` 时由 `recover_updatable_engines`（§3 提到的 rollout.py:601）检测到空位并重建——**"杀死->留空->下次同步时重建"是一套完全解耦的三段式容错**，`RolloutHealthMonitor` 只负责第一步，完全不关心如何恢复。

---

## 4. Trace 与 Profiling

### 4.1 内置埋点：trace 与 timer

- **trace**：`slime/utils/trace_utils.py` 提供 `trace_function` / `trace_span` 装饰器与上下文管理器。rollout 链路的关键节点都埋了点——`generate_and_rm`（sglang_rollout.py:223）、`generate_and_rm_group`（289）、`sglang_generate`（201）、`reward_model`（273/283）——导出后可用 Chrome trace viewer 这类工具可视化"一步里每个样本的时间线"，定位长尾的利器；`build_sglang_meta_trace_attrs` 把 SGLang 返回的 meta（排队时间、prefill/decode 耗时）也纳入 trace。文档：`docs/zh/developer_guide/trace.md`；
- **timer**：`@timer` 装饰器（如 actor.py:566 的 `update_weights`）自动记录各阶段耗时；
- **性能指标**：`_log_rollout_data` / `compute_perf_metrics_from_samples`（rollout.py:1292-1359）计算 token 吞吐、按 `non_generation_time` 拆分"生成耗时 vs 环境耗时"（agentic 场景归因工具执行开销），以及 SGLang 请求级性能指标（`_compute_sglang_request_perf_metrics`）。

下面 §4.2-§4.6 是这套内置埋点之外，定位性能问题的完整手把手流程（从零成本指标逐级放大到系统级 profiler）。

### 4.2 第一步（零成本）：内置的 `perf/*` 计时器

slime 有一个极简的计时器单例 `Timer`（`slime/utils/timer.py:15`），用法是：

```python
with timer("train"):
    ...  # 会被计时并累加到 Timer().timers["train"]
```

它在训练侧（`slime/backends/megatron_utils/actor.py`）的关键位置都埋了点：

| 计时名 | 代码位置 | 覆盖范围 |
|---|---|---|
| `train_wait` | `actor.py:52, 423` | 训练进程等待下一次 rollout 数据到达的时间（**越大说明训练在等 rollout，即 rollout 是瓶颈**） |
| `data_preprocess` | `actor.py:371` | 把 Ray object store 里的 rollout 数据反序列化/搬到 GPU 的时间 |
| `train` | `actor.py:423` | 整个训练阶段（log_probs + 反向 + 记录）的总时间 |
| `log_probs` / `ref_log_probs` / `teacher_log_probs` | `actor.py:353`（`compute_log_prob` 内） | actor/ref/teacher 模型算 log_probs 的前向时间 |
| `actor_train` | `actor.py:507` | 真正的前向+反向+optimizer step（`train()` 函数）时间 |
| `ref_model_update` | `actor.py:534` | 定期把 actor 权重同步给 ref 模型的时间 |
| `save_model` | `actor.py:541`（`@timer` 装饰器） | 存 checkpoint 的时间 |
| `update_weights` | `actor.py:566`（`@timer` 装饰器） | 训练权重同步到 SGLang 推理引擎的时间 |

每个 rollout step 结束时，`log_perf_data()`（`slime/backends/megatron_utils/data.py:497`）会：
1. 把 `Timer().log_dict()` 里累计的所有耗时取出来，`reset()` 清零（下一轮重新累计）；
2. 加上 `perf/` 前缀（例如变成 `perf/actor_train_time`），派生出更多指标：
   - `perf/actor_train_tflops`、`perf/actor_train_tok_per_s`（用 `calculate_fwd_flops` 估算算力/吞吐）
   - `perf/log_probs_tflops`、`perf/ref_log_probs_tflops`
   - `perf/step_time` = `train_wait_time + train_time`（一整个 rollout 的墙钟时间）
   - `perf/wait_time_ratio` = `train_wait_time / step_time`（**训练侧空等占比，非常关键的一个指标**）
3. 通过 `logging_utils.log()`（`slime/utils/logging_utils.py:45`）写到 wandb / tensorboard，同时也会 `logger.info(f"perf {rollout_id}: {log_dict}")` 直接打印到 stdout/日志文件（哪怕不开 wandb 也能看到）。

Rollout（生成）侧的耗时在 `slime/ray/rollout.py:1292` 的 `_log_rollout_data` 里统计，核心函数 `compute_perf_metrics_from_samples`（`rollout.py:1325`），产出的指标带 `perf/` 前缀，例如：

| 指标 | 含义 |
|---|---|
| `perf/rollout_time` | 一次 rollout 生成阶段的总墙钟时间 |
| `perf/tokens_per_gpu_per_sec` | 平均每张推理 GPU 每秒吐出的 token 数（吞吐） |
| `perf/longest_sample_tokens_per_sec` | 最长样本的生成速度（长尾请求的瓶颈参考） |
| `perf/non_generation_time/*` | 生成过程中"非生成"时间（例如工具调用、等待调度）的统计 |

**怎么看**：不需要额外加任何 profiling 开关，只要正常起训练——直接 `grep perf` 训练日志（stdout），每个 rollout 都会打一行 `perf {rollout_id}: {...}` 的 dict；或加 `--use-wandb`/`--use-tensorboard` 在面板里搜索 `perf/` 前缀曲线看趋势。

**怎么判断"卡点"（举例）**：

- `perf/wait_time_ratio` ≈ 0.6，训练进程 60% 时间在等 rollout 数据 → **瓶颈在 Rollout 生成侧**，去看 `perf/rollout_time`、`perf/tokens_per_gpu_per_sec` 是否偏低，考虑加大 `--rollout-num-gpus`、调大 `--max-tokens-per-gpu`、检查是否开了 `--use-dynamic-batch-size`、response 长度是否有极端长尾；
- `perf/wait_time_ratio` 很低（<0.1），但 `perf/actor_train_time` 本身很大且 `perf/actor_train_tflops` 远低于该 GPU 型号的理论算力 → **瓶颈在训练侧的计算效率**，需要用 §4.3 的 PyTorch Profiler 细看；
- `perf/log_probs_time` 占 `perf/train_time` 比例异常高 → 可能是 `--log-probs-max-tokens-per-gpu` 设置不合理（micro-batch 切太碎，launch overhead 主导）；
- `perf/data_preprocess_time` 偏高 → Ray object store 序列化/搬数据慢，考虑 `--rollout-data-transport nixl`。

这一步不需要装任何额外依赖，是**永远第一个该看的东西**。

### 4.3 第二步：PyTorch Profiler —— 抓训练侧 kernel 级别 trace

`torch.profiler` 是 PyTorch 官方自带的 profiler，记录每个 CPU 算子/CUDA kernel 的起止时间、Python 调用栈（`with_stack=True`）、显存分配时间线（`profile_memory=True`）、FLOPs 估算（`with_flops=True`）。产出是一份 trace 文件，可用 TensorBoard 插件或 Chrome/Perfetto 打开，图形化看"时间轴上每个 GPU stream 在做什么"。

**涉及的开关**：

| 参数 | 来源 | 默认值 | 作用 |
|---|---|---|---|
| `--use-pytorch-profiler` | Megatron 原生（`common_config.py:41` 的 `ProfilingConfig.use_pytorch_profiler`） | `False` | **总开关**，不开则下面所有 profile-target 都不生效 |
| `--profile-step-start` | Megatron 原生（`profile_step_start`） | `10` | 从第几个 rollout step 开始抓（之前是 warmup，不计入 trace） |
| `--profile-step-end` | Megatron 原生（`profile_step_end`） | `12` | 到第几个 rollout step 结束抓；`active = end - start` 步会被真正记录 |
| `--profile-target` | slime 自定义（`slime/utils/arguments.py:1296-1302`） | `["train_overall"]`，可多选 `train_overall / train_actor / train_log_probs` | 决定"在哪个粒度"抓 trace |
| `--tensorboard-dir` | Megatron 原生 | 无默认，需要显式传 | trace 文件的输出目录（同时也是 tensorboard 日志目录） |

`--profile-target` 三个可选值对应三种不同粒度（实现在 `slime/utils/profile_utils.py`）：

- `train_overall`：以 **rollout step** 为最小单位推进 profiler 的 schedule。适合看"训练侧在多个 rollout 之间"的整体波动（比如某次权重同步特别慢、某次显存 GC 卡顿）。代码：`TrainProfiler.__init__`（`profile_utils.py:19-24`）创建 profiler，`TrainProfiler.on_init_end()`（26-28，在 `actor.py:199` 模型初始化完后调用）启动它，`TrainProfiler.step()`（30-39，在 `actor.py:518` 每次 `train_actor` 结束后调用）推进一步；
- `train_actor`：以 **micro-batch** 为最小单位，只抓 `train_actor` 里真正跑 forward+backward 的那部分循环（`TrainProfiler.iterate_train_actor()`，41-42），更细，适合看单次训练反向内部各 micro-batch 之间的 kernel 情况；
- `train_log_probs`：同理，但抓的是算 log_probs 的前向循环（`iterate_train_log_probs()`）。

三者共享同一套 schedule 参数（`wait/warmup/active`），由 `_create_torch_profiler()`（`profile_utils.py:60-78`）统一构造：

```python
torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=max(args.profile_step_start - 1, 0),
        warmup=1 if args.profile_step_start > 0 else 0,
        active=args.profile_step_end - args.profile_step_start,
        repeat=1,
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        args.tensorboard_dir,
        worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
        use_gzip=True,
    ),
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
    with_flops=True,
)
```

**重要坑点**：

1. 这里**没有**读取 Megatron 原生的 `--profile-ranks`（那是 Megatron 自己 `training.py` 里训练循环用的过滤，slime 用的是自己的训练循环 `actor.py`，没有复用这个过滤）。也就是说 **一旦开启，所有 rank 都会各自生成一份 trace 文件**——大规模 TP/PP/DP 训练下目录文件数=rank 数，单个文件可能几百 MB～几 GB，且 `with_stack=True`+`profile_memory=True` 有不小的额外开销，会拖慢被 profile 的那几步。**建议**：先在小规模（单机 1~2 张卡、小模型）复现问题再开 profiler；或只跑很短的区间（3~5 步）。
2. `profile_step_start=0` 时 `warmup=0`，意味着从第 0 步就是"正式记录"，没有 warmup 步，第一步数据可能包含 CUDA context 初始化等噪声，一般建议 `profile-step-start` 至少设成 2~3。

**依赖安装**：`pip install torch-tb-profiler`（`torch.profiler` 本身随 PyTorch 自带；`tensorboard` 已在 `requirements.txt`）。

**手把手**：以 `scripts/run-qwen3-4B.sh` 为模板，加一组参数：

```bash
PROFILE_ARGS=(
   --use-pytorch-profiler
   --profile-step-start 3
   --profile-step-end 5          # 只抓第 3~4 个 rollout（active = 5-3 = 2 步）
   --profile-target train_overall train_actor
   --tensorboard-dir /root/slime_profile_out
)
```

正常提交训练，训练跑到第 5 个 rollout后，每个 rank 会在 `/root/slime_profile_out/` 下生成形如 `train_overall_rank_0.<hostname>_<pid>.<timestamp>.pt.trace.json.gz` 的文件。

**查看方式一（推荐，图形化）**：`tensorboard --logdir /root/slime_profile_out --port 6006`，浏览器打开后选 "PYTORCH_PROFILER" 标签页——Overview（GPU利用率/Kernel/Memcpy/Communication 占比饼图，一眼看出算子慢/通信慢/GPU空闲）、Kernel（耗时排序的 CUDA kernel 列表）、Trace（完整时间轴，"卡点"就是出现大段 GPU 空白的地方）。

**查看方式二（无需装插件）**：`gunzip xxx.pt.trace.json.gz` 后，在 `chrome://tracing` 或 https://ui.perfetto.dev/ 打开解压后的 `.json`。

**怎么定位卡点（举例）**：某 rank 的 GPU 轨道长时间空白、同一时刻其他 rank 却在跑——典型 **PP/DP 负载不均衡**（某 micro-batch 序列长度特别长），配合 `--balance-data` / `--balance-by-flops` 解决；大量 `ncclAllReduce`/`ncclBroadcast` kernel 占比很高——通信瓶颈，检查网络拓扑（`nvidia-smi topo -m`，是否有 NVLink）、TP size 是否过大；CPU 侧 launch 间隔很大、GPU 却空——**launch overhead 主导**，常见于 micro-batch 切太碎，考虑调大。

### 4.4 第三步：显存分析（定位 OOM / 显存碎片 / offload 卡顿）

**涉及的开关**：

| 参数 | 来源 | 默认值 | 作用 |
|---|---|---|---|
| `--record-memory-history` | Megatron 原生，slime 用 `reset_arg` 改成 `action="store_true"`（`arguments.py:1309`） | `False` | 总开关 |
| `--memory-recorder` | slime 自定义（`arguments.py:1303-1308`） | `torch`，可选 `torch`/`memray` | 选择显存记录后端 |
| `--memory-snapshot-dir` | slime 自定义（`arguments.py:1286-1290`） | `.`（当前目录） | dump 文件存放目录 |
| `--memory-snapshot-path` | Megatron 原生（`ProfilingConfig.memory_snapshot_path`） | `snapshot.pickle` | dump 文件名后缀 |
| `--memory-snapshot-num-steps` | slime 自定义（`arguments.py:1291-1295`） | `None` | 跑够这么多个 rollout 后主动 dump 一次（不设的话只在 OOM 时自动 dump） |

只在 `--profile-target` 包含 `train_overall` 时才会启用（`TrainProfiler.__init__` 里 `if args.record_memory_history and ("train_overall" in args.profile_target)`，这是默认值，一般不用改）。

**两种后端的区别**（`slime/utils/profile_utils.py:81-147`）：

- **`torch`（`_TorchMemoryProfiler`）**：调用 PyTorch 原生 `torch.cuda.memory._record_memory_history(...)`，同时用 `torch._C._cuda_attach_out_of_memory_observer` 挂一个 **OOM 自动 dump 回调**——一旦这张卡真的 OOM，会自动把当前的显存分配历史 dump 成 pickle 文件，并打印堆栈。这是**排查 OOM 最有效的手段**，几乎零心智负担：开着它训练，等它自己 OOM 的时候自动留证据。如果设置了 `--memory-snapshot-num-steps`，跑到第 `N-1` 个 rollout 时也会主动 dump 一次（即使没有 OOM）；
- **`memray`（`_MemrayMemoryProfiler`）**：Python 级别的内存分析器（不止 GPU，也能看 CPU 端 Python 对象的内存），`native_traces=True` 表示同时记录 C/C++ 层调用栈。**必须**设置 `--memory-snapshot-num-steps`（代码里有 `assert`），因为 memray 没有 OOM 回调机制，只能按步数主动停止记录。

**依赖安装**：`pip install memray`（已写在 `requirements.txt`）；`torch` 后端不需要额外依赖。

**手把手**：

场景 A（怀疑 OOM，想留证据）：

```bash
DEBUG_ARGS=(
   --record-memory-history
   --memory-recorder torch
   --memory-snapshot-dir /root/slime_mem_snapshot
)
```

某 rank OOM 时日志会打印 dump 路径，文件已自动生成。打开 https://docs.pytorch.org/memory_viz 把 `.pickle` 拖进去，可以看显存占用曲线，鼠标悬停能看到当前时刻每块已分配显存对应的 **Python 调用栈**（`stacks="all"`），直接定位是哪一行代码申请的、是否有"越攒越多"的泄漏。

场景 B（不 OOM，看正常训练时的显存分布）：加 `--memory-snapshot-num-steps 5`，跑完第 5 个 rollout 自动 dump，同样用 memory_viz 查看。

场景 C（memray 看 Python/C 层内存）：`--memory-recorder memray` + `--memory-snapshot-num-steps`，生成的文件用 `memray flamegraph xxx.pickle` 生成 html 火焰图，或 `memray tree xxx.pickle` 看命令行调用树。

### 4.5 第四步：Rollout（SGLang 推理引擎）侧 profiling

**为什么单独拿出来**：训练侧 profiling 走 `train.py` 的 CLI 参数，但 Rollout 生成是**独立的 HTTP 服务进程**，它的 profiling 不走 CLI 参数，而是**训练过程中实时通过 HTTP 请求控制**。

调用链：`slime/backends/sglang_utils/sglang_engine.py:485` 的 `SGLangEngine.start_profile()` / `sglang_engine.py:516` 的 `stop_profile()`，本质是往每个 SGLang 引擎发 `POST /start_profile` / `POST /stop_profile`（引擎自己实现了这两个 admin 接口，见 11 篇 §2 的服务端调用链）。slime **没有**把这两个方法接到 `train.py` 的任何 CLI 开关上，需要训练跑起来后另开终端手动触发。仓库提供了现成工具 `tools/profile_rollout.py`：先请求 router 的 `/workers` 拿所有推理 worker 地址，再对每个 worker 分别发 `/start_profile`/`/stop_profile`。

**手把手**：

```bash
python tools/profile_rollout.py \
    --router-url http://127.0.0.1:30000 \
    --action start \
    --output-dir /root/sglang_profile_out \
    --num-steps 5 \
    --activities GPU \
    --with-stack \
    --record-shapes
```

`--num-steps 5` 表示自动抓 5 个 forward step（prefill/decode 各算一步）后自动停止（源码注释："If it is set, profiling is automatically stopped after this step"）。不传的话需要再手动执行一次 `--action stop`。每个 worker 各自把 trace dump 到配置的目录，格式同样是 `.pt.trace.json.gz`，**查看方式跟 §4.3 完全一样**。

**进阶**：`--profile-by-stage` 让 SGLang 按"prefill 阶段"和"decode 阶段"分别落盘 trace，适合区分"长 prompt 处理慢"还是"逐 token 生成慢"。`--sglang-enable-layerwise-nvtx-marker`（转发给 SGLang 引擎）让每层 Transformer layer 打一个 NVTX marker，配合 §4.6 的 `nsys` 能精确看到"第几层"、"attention 还是 MLP"耗时多少。

### 4.6 第五步（全局视角）：NVIDIA Nsight Systems（`nsys`）

`torch.profiler` 只能看到"PyTorch 能感知到"的东西；`nsys` 是**系统级**的 profiler，可以同时看到多进程/多 GPU 时间线对齐、CPU 侧系统调用/线程调度/锁等待、CUDA driver API/NCCL 通信/显存拷贝、NVTX marker 语义化区间。**什么时候用它**：怀疑瓶颈不是某个算子本身慢，而是"进程之间互相等"（colocate 模式下训练等推理 offload 显存、多机 NCCL 通信被网络打满、CPU 侧调度导致 GPU 空闲）。

**重要说明**：Megatron 原生的 nsys 开关（`--profile`/`use_nsys_profiler`、`--nvtx-ranges`、`--record-shapes`）在 slime 里**不生效**（slime 用自己写的训练循环 `actor.py`，没有调用 Megatron 那套 nsys 相关代码）。因此对 slime 做 `nsys` 分析要用**整进程包裹**方式。

**依赖**：`nsys` 是 CUDA Toolkit 附带的命令行工具（`which nsys` 检查是否已装），推荐装桌面版 Nsight Systems GUI 在本地机器打开 `.nsys-rep` 文件。

**手把手（方式一，推荐先用）**：脱离 Ray，用最小复现脚本单进程运行：

```bash
nsys profile \
    -t cuda,nvtx,osrt \
    -s none \
    -o /root/nsys_out/train_trace \
    --force-overwrite true \
    python3 train.py --debug-train-only --load-debug-rollout-data /root/dump/{rollout_id}.pt ...
```

`-t cuda,nvtx,osrt` 采集 CUDA API/NVTX 区间/系统调用调度；`-s none` 不用 CPU 抽样，产出文件更小；`--debug-train-only --load-debug-rollout-data` 跳过真实 SGLang 推理，专注训练侧。

**方式二**（Ray 多进程场景）：通过 `RAY_worker_process_setup_hook` 或把 Ray runtime env 里的 Python 可执行文件替换成 `nsys profile ... python3 "$@"` 的 wrapper 脚本，让每个 Ray worker 自动被 nsys 包裹（配置略复杂，建议先用方式一定位大致范围）。

**怎么看结果**：`nsys stats /root/nsys_out/train_trace.nsys-rep` 命令行看摘要统计（CUDA kernel/NVTX 区间/CUDA API 调用耗时 Top N）；真正定位卡点建议用 Nsight Systems GUI 打开 `.nsys-rep`——每个 CUDA stream/进程一条时间轴纵向对比，一眼看出哪些 GPU 在空闲，放大空白区间看紧邻的是什么 API/区间。

### 4.7 手把手完整示例：一次完整的"定位卡点"流程

假设 Qwen3-4B GRPO 训练感觉比预期慢，按下面顺序排查：

**Step 1 —— 先看免费的 perf 指标（5 分钟）**：跑几十个 rollout，观察 `perf/wait_time_ratio`：持续 >0.4~0.5 说明训练大部分时间在等 rollout，下一步分析 rollout 侧（跳到 Step 3）；很低说明训练本身计算量大，下一步分析训练侧（Step 2）。

**Step 2 —— 训练侧慢：开 PyTorch Profiler**：

```bash
PROFILE_ARGS=(
   --use-pytorch-profiler
   --profile-step-start 3
   --profile-step-end 5
   --profile-target train_actor
   --tensorboard-dir /root/slime_profile_out
)
```

跑到第 5 个 rollout 后 `tensorboard --logdir /root/slime_profile_out`，看 Kernel 面板定位耗时最长的算子/通信，针对性调整并行策略（TP/PP/CP/EP）、`--recompute-*`、`--use-dynamic-batch-size`+`--max-tokens-per-gpu`。

**Step 3 —— Rollout 侧慢：开 SGLang profiler**：

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:<router端口> \
    --action start --output-dir /root/sglang_profile_out --num-steps 5 --activities GPU
```

用 tensorboard/perfetto 打开产出的 trace，看是 prefill 慢还是 decode 慢（配合 `--profile-by-stage`），检查显存是否够（`--sglang-mem-fraction-static`）、是否长尾请求拖慢了整批（`perf/longest_sample_tokens_per_sec` vs `perf/tokens_per_gpu_per_sec` 的差距）。

**Step 4 —— 怀疑显存问题：开显存 snapshot**：加 `--record-memory-history --memory-recorder torch --memory-snapshot-dir /root/slime_mem_snapshot`，跑着等它 OOM 自动落盘，用 memory_viz 打开分析。

**Step 5 —— 想看更全局的"进程间互相等待"：上 `nsys`**：先在最小复现场景（`--debug-train-only`+dump 数据）跑一遍 `nsys profile`，GUI 打开看整体空闲情况，尤其关注 colocate 模式下 offload/onload 的那几段时间。

### 4.8 参数速查表

| 参数 | 来源 | 类型/默认值 | 作用简述 |
|---|---|---|---|
| `--use-pytorch-profiler` | Megatron 原生 | bool, `False` | 训练侧 PyTorch Profiler 总开关 |
| `--profile-step-start` / `--profile-step-end` | Megatron 原生 | int, `10`/`12` | 训练侧 profiler 起止 rollout step |
| `--profile-target` | slime (`arguments.py:1296`) | list, `[train_overall]` | 训练侧 profiler 抓取粒度 |
| `--tensorboard-dir` | Megatron 原生 | 需显式指定 | trace / tensorboard 日志输出目录 |
| `--record-memory-history` | Megatron 原生（slime reset 为 flag） | bool, `False` | 显存历史记录总开关（含 OOM 自动 dump） |
| `--memory-recorder` | slime (`arguments.py:1303`) | str, `torch`/`memray` | 显存记录后端 |
| `--memory-snapshot-dir` / `--memory-snapshot-path` | slime / Megatron 原生 | str | 显存 dump 文件目录/文件名 |
| `--memory-snapshot-num-steps` | slime (`arguments.py:1291`) | int, `None` | 跑够 N 个 rollout 后主动 dump 一次（memray 模式下必填） |
| （无 CLI 开关，运行时 HTTP 控制） | `sglang_engine.py:485` `start_profile()` / `tools/profile_rollout.py` | - | Rollout（SGLang）侧 PyTorch Profiler，训练跑起来后手动触发 |
| `--sglang-enable-layerwise-nvtx-marker` | 转发给 SGLang 引擎 | bool | 逐层打 NVTX marker，配合 nsys 看层级耗时 |
| `--profile`（Megatron `use_nsys_profiler`）、`--nvtx-ranges`、`--record-shapes` | Megatron 原生 | - | **在 slime 里目前不生效**，需用整进程包裹方式（§4.6） |
| `--debug-train-only` + `--load-debug-rollout-data` / `--debug-rollout-only` | slime | - | 分离调试，方便单独 profile 一侧 |

依赖：`pip install torch-tb-profiler memray`；`nsys` 是 CUDA Toolkit 自带工具，需单独装 Nsight Systems。

### 4.9 常见问题 / 踩坑提示

1. **开了 `--use-pytorch-profiler` 但目录下什么都没有？** 检查是否跑到 `--profile-step-end`（`on_trace_ready` 只在 `repeat` 完成后触发写盘），或进程是否异常退出导致 profiler 没走完 schedule。
2. **trace 文件特别大，tensorboard 打开很卡？** 调小 `profile-step-end - profile-step-start`，或 `--profile-target` 只留一个值；`with_stack=True` 是主要体积来源，目前硬编码打开（`profile_utils.py:75`）。
3. **多 rank 一起开 profiler 会不会互相干扰或拖慢训练？** 会有一定开销，且因为没有 `profile_ranks` 过滤，所有 rank 都会各自产生开销和文件。生产训练 debug 时建议先在小规模复现问题。
4. **SGLang 侧的 `--num-steps` 是"多少个 rollout"还是"多少个 forward step"？** 是 SGLang 引擎自己的 forward step（prefill 一次算一步，decode 每个 token 也算一步），跟训练侧"rollout step"是不同的计数单位。
5. **想同时对齐训练侧和 rollout 侧的 trace 时间轴？** 目前两边分别产出各自的 trace 文件，没有自动合并工具；需要严格对齐时序建议用 `nsys` 全局采样（前提是各进程时钟同步，同机训练天然满足）。
6. **colocate 模式下怎么区分"这段时间是训练在跑"还是"推理在跑"？** 看 `perf/wait_time_ratio` 结合训练日志里 `offload/onload` 相关的 log，或在 `nsys` timeline 上直接看某一时刻是哪个进程的 CUDA context 活跃。

---

## 5. CI：把"正确性"变成回归测试

README 概括了 CI 的三层：

1. **CPU 单测**：数据流、参数、转换逻辑的纯 CPU 测试（`tests/` 下大量 `test_*.py`）；
2. **customization hook contract test**：保证 07 篇那些 `--*-path` 钩子的签名契约不被破坏；
3. **GPU e2e 测试**：真实 Megatron + SGLang 跑通 dense/MoE、checkpoint、数值精度、async rollout、OPD、PPO workflow、debug rollout-then-train replay。

CI 里的特色设施：`--ci-test` 参数触发严格断言（如 generate 里 `assert isinstance(sample.prompt, str)`，sglang_rollout.py:155-156）、磁盘权重更新的版本对账（04 篇）、故障注入（§3）。文档：`docs/zh/developer_guide/ci.md`。

---

## 6. 可复现性

- **种子链路**：`--rollout-seed` 驱动数据集 shuffle（`Dataset.shuffle` 用 `seed + epoch_id`，保证同 epoch 同排列，03 篇）；`sglang_enable_deterministic_inference` 时组内每个采样用 `rollout_seed + i` 的固定采样种子（sglang_rollout.py:110-112、317-319）；
- **确定性排序**：rollout 收集结果按 `sample.index` 排序（sglang_rollout.py:451-454），fully async 同样按 index 排序（fully_async_rollout.py:233-241）——乱序完成 ≠ 乱序训练；
- **调试文档**：`docs/zh/developer_guide/debug.md`、`docs/zh/advanced/reproducibility.md`。

---

## 7. 系列收尾：学习路线建议

至此本系列核心 9 篇（00-09）已覆盖 slime 自身架构的全部主干，08 篇补充了奖励/评估这一常被忽视的环节，10-13 篇继续深入第三方依赖库源码。推荐的动手顺序：

1. 跑通 `docs/zh/get_started/quick_start.md` 的最小示例，对照 01/02 篇理解主循环；
2. 用 `--debug-rollout-only` + `--dump-details` 观察 `Sample` 的真实内容（03 篇）；
3. 改 `--advantage-estimator` 与 TIS 开关，观察 wandb 指标差异（05 篇）；
4. 写一个 `--custom-generate-function-path` 的 two-turn 玩具示例（07 篇 §1.2）；
5. 读 `examples/search-r1`，然后读 `examples/fully_async`，最后挑战 `examples/coding_agent_rl`；
6. 有余力可对照综述 §3 读 verl（HybridEngine/AgentLoop）与 AReaL（staleness-aware）源码，体会不同架构取舍；
7. 训练变慢时按本篇 §4.7 的五步法排查，而不是凭感觉猜。
