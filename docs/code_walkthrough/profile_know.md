# RL 训练性能分析（Profiling）手把手教程

> 面向零经验读者。目标：教你如何在 slime 的 RL 训练流程（Rollout 生成 + Megatron 训练 + 权重同步）里，
> 用最小成本先找到"大概是哪一段慢"，再用 `torch.profiler` / 显存 snapshot / SGLang profiler / `nsys`
> 逐步放大镜头，看清楚具体卡在哪个算子/哪次通信/哪次显存分配上。
>
> 全文所有开关都能在 `slime/utils/arguments.py` 里找到定义（部分是复用 Megatron 原生参数，会特别标注来源）。

---

## 0. 先搞懂：一次 RL 训练循环里，"慢"可能发生在哪几个地方

slime 的训练主循环大致是（细节可参考 `docs/code_walkthrough/rollout_know.md`）：

```
for rollout_id in range(num_rollout):
    1. RolloutManager.generate()          # SGLang 引擎做推理采样，生成 response（"rollout 阶段"）
    2. 计算 reward / advantage
    3. TrainRayActor.train()              # Megatron 侧：log_probs 前向 + actor 训练反向（"train 阶段"）
    4. update_weights()                   # 把训练好的新权重同步回 SGLang 推理引擎
    5. (可选) save_model() / eval()
```

四个阶段（生成、算 log_probs、训练反向、权重同步）分别跑在不同的代码/不同的硬件资源利用模式上，
"卡点"可能是其中任何一个：

- Rollout 生成慢：prompt/response 太长、SGLang 引擎显存不够频繁抢占、DP 之间长尾请求不均衡……
- log_probs / 训练反向慢：micro-batch 切分不合理、TP/PP/CP 配置不当、某个 kernel 没走高效路径……
- 权重同步慢：NCCL 广播/磁盘 IO 慢、MoE 模型 all-gather 权重量太大……
- GPU 之间互相等待：colocate 模式下训练要等推理引擎 offload 完才能拿到显存，反之亦然。

**分析思路（本文的主线）**：

1. 先看 slime **内置的、几乎零成本**的 `perf/*` 计时指标，定位是"生成慢"还是"训练慢"还是"互相等待多"。
2. 如果是训练侧慢 → 开 **PyTorch Profiler**，看 Megatron 训练循环里具体是哪个 kernel/通信慢。
3. 如果怀疑是显存不够导致的 OOM / 频繁 offload → 开 **显存 snapshot（torch memory history / memray）**。
4. 如果是 Rollout 生成慢 → 用 **SGLang 自带的 `/start_profile`**（也是 PyTorch Profiler，但抓的是推理引擎）。
5. 如果想看整机（CPU+GPU+NCCL 通信+进程调度）的全貌 → 用 **NVIDIA Nsight Systems (`nsys`)**。

---

## 1. 第一步（零成本）：内置的 `perf/*` 计时器

### 1.1 原理和代码位置

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
| `log_probs` / `ref_log_probs` / `teacher_log_probs` | `actor.py:353` (`compute_log_prob` 内) | actor/ref/teacher 模型算 log_probs 的前向时间 |
| `actor_train` | `actor.py:507` | 真正的前向+反向+optimizer step（`train()` 函数）时间 |
| `ref_model_update` | `actor.py:534` | 定期把 actor 权重同步给 ref 模型的时间 |
| `save_model` | `actor.py:541` (`@timer` 装饰器) | 存 checkpoint 的时间 |
| `update_weights` | `actor.py:566` (`@timer` 装饰器) | 训练权重同步到 SGLang 推理引擎的时间 |

每个 rollout step 结束时，`log_perf_data()`（`slime/backends/megatron_utils/data.py:497`）会：
1. 把 `Timer().log_dict()` 里累计的所有耗时取出来，`reset()` 清零（下一轮重新累计）；
2. 加上 `perf/` 前缀（例如变成 `perf/actor_train_time`），派生出更多指标：
   - `perf/actor_train_tflops`、`perf/actor_train_tok_per_s`（用 `calculate_fwd_flops` 估算算力/吞吐）
   - `perf/log_probs_tflops`、`perf/ref_log_probs_tflops`
   - `perf/step_time` = `train_wait_time + train_time`（一整个 rollout 的墙钟时间）
   - `perf/wait_time_ratio` = `train_wait_time / step_time`（**训练侧空等占比，非常关键的一个指标**）
3. 通过 `logging_utils.log()`（`slime/utils/logging_utils.py:45`）写到 wandb / tensorboard，同时也会
   `logger.info(f"perf {rollout_id}: {log_dict}")` 直接打印到 stdout/日志文件（哪怕不开 wandb 也能看到）。

Rollout（生成）侧的耗时在 `slime/ray/rollout.py:1292` 的 `_log_rollout_data` 里统计，核心函数
`compute_perf_metrics_from_samples`（`rollout.py:1325`），产出的指标带 `perf/` 前缀，例如：

| 指标 | 含义 |
|---|---|
| `perf/rollout_time` | 一次 rollout 生成阶段的总墙钟时间 |
| `perf/tokens_per_gpu_per_sec` | 平均每张推理 GPU 每秒吐出的 token 数（吞吐） |
| `perf/longest_sample_tokens_per_sec` | 最长样本的生成速度（长尾请求的瓶颈参考） |
| `perf/non_generation_time/*` | 生成过程中"非生成"时间（例如工具调用、等待调度）的统计 |

### 1.2 怎么看

不需要额外加任何 profiling 开关，只要正常起训练：
- **最简单**：直接 `grep perf` 训练日志（stdout），每个 rollout 都会打一行 `perf {rollout_id}: {...}` 的 dict。
- **可视化**：加上 `--use-wandb --wandb-project xxx` 或 `--use-tensorboard --tb-project-name xxx`，
  在 wandb/tensorboard 里搜索 `perf/` 前缀的曲线，随 rollout step 变化看趋势。

### 1.3 怎么判断"卡点"（举例）

假设你在 wandb 里看到某次训练：

- `perf/wait_time_ratio` ≈ 0.6，也就是训练进程 60% 的时间在等 rollout 数据 → **瓶颈在 Rollout 生成侧**，
  应该去看 `perf/rollout_time`、`perf/tokens_per_gpu_per_sec` 是否偏低，再考虑加大 `--rollout-num-gpus`、
  调大 `--max-tokens-per-gpu`、检查是否开了 `--use-dynamic-batch-size`、response 长度是否有极端长尾等。
- `perf/wait_time_ratio` 很低（比如 <0.1），但 `perf/actor_train_time` 本身很大且 `perf/actor_train_tflops`
  远低于该 GPU 型号的理论算力 → **瓶颈在训练侧的计算效率**，需要用第 2 步的 PyTorch Profiler 细看是哪个
  kernel/通信慢。
- `perf/log_probs_time` 占 `perf/train_time` 比例异常高（比如超过 actor_train 本身）→ 可能是
  `--log-probs-max-tokens-per-gpu` 设置不合理（切太碎的 micro-batch，前向开销被 launch overhead 主导）。
- `perf/data_preprocess_time` 偏高 → Ray object store 序列化/搬数据慢，可以看是否是 `--rollout-data-transport`
  用了 `object-store` 而数据量很大，考虑 `nixl`。

这一步不需要装任何额外依赖，是**永远第一个该看的东西**。

---

## 2. 第二步：PyTorch Profiler —— 抓训练侧 kernel 级别 trace

### 2.1 这是什么

`torch.profiler` 是 PyTorch 官方自带的 profiler，可以记录：
- 每个 CPU 算子（aten op）、每个 CUDA kernel 的起止时间和耗时；
- Python 调用栈（`with_stack=True`），方便定位到具体是哪一行代码触发的；
- 显存分配时间线（`profile_memory=True`）；
- FLOPs 估算（`with_flops=True`）。

产出是一份 trace 文件，可以用 TensorBoard 插件或 Chrome/Perfetto 打开，图形化看"时间轴上每个 GPU stream
在做什么"。这是分析"训练一个 rollout 具体慢在哪个算子/哪次 NCCL 通信"最直接的工具。

### 2.2 涉及的开关

| 参数 | 来源 | 默认值 | 作用 |
|---|---|---|---|
| `--use-pytorch-profiler` | Megatron 原生（`common_config.py:41` 的 `ProfilingConfig.use_pytorch_profiler`） | `False` | **总开关**，不开则下面所有 profile-target 都不生效 |
| `--profile-step-start` | Megatron 原生（同上，`profile_step_start`） | `10` | 从第几个 rollout step 开始抓（之前是 warmup，不计入 trace） |
| `--profile-step-end` | Megatron 原生（同上，`profile_step_end`） | `12` | 到第几个 rollout step 结束抓；`active = end - start` 步会被真正记录 |
| `--profile-target` | slime 自定义（`slime/utils/arguments.py:1296-1302`） | `["train_overall"]`，可多选 `train_overall / train_actor / train_log_probs` | 决定"在哪个粒度"抓 trace（见下） |
| `--tensorboard-dir` | Megatron 原生 | 无默认，需要显式传 | trace 文件的输出目录（同时也是 tensorboard 日志目录） |

`--profile-target` 三个可选值对应三种不同粒度（实现在 `slime/utils/profile_utils.py`）：

- `train_overall`：以 **rollout step** 为最小单位推进 profiler 的 schedule（每个 rollout 算一"步"）。
  适合看"训练侧在多个 rollout 之间"的整体波动（比如某次权重同步特别慢、某次显存 GC 卡顿）。
  代码：`TrainProfiler.__init__`（`profile_utils.py:19-24`）创建 profiler，
  `TrainProfiler.on_init_end()`（`profile_utils.py:26-28`，在 `actor.py:199` 模型初始化完后调用）启动它，
  `TrainProfiler.step()`（`profile_utils.py:30-39`，在 `actor.py:518` 每次 `train_actor` 结束后调用）推进一步。
- `train_actor`：以 **micro-batch** 为最小单位，只抓 `train_actor` 里真正跑 forward+backward 的那部分循环
  （`TrainProfiler.iterate_train_actor()` → `profile_utils.py:41-42`），更细，适合看单次训练反向内部
  各 micro-batch 之间的 kernel 情况。
- `train_log_probs`：同理，但抓的是算 log_probs 的前向循环（`iterate_train_log_probs()`）。

三者共享同一套 schedule 参数（`wait/warmup/active`），由 `_create_torch_profiler()`
（`profile_utils.py:60-78`）统一构造：

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
1. 这里**没有**读取 Megatron 原生的 `--profile-ranks`（那是 Megatron 自己 `training.py` 里训练循环用的过滤，
   slime 用的是自己的训练循环 `actor.py`，没有复用这个过滤）。也就是说 **一旦开启，所有 rank 都会各自生成一份
   trace 文件**。在大规模 TP/PP/DP 训练下，这会导致：
   - `--tensorboard-dir` 目录下文件数 = rank 数，单个 trace 文件可能几百 MB～几 GB；
   - profiler 本身（`with_stack=True` + `profile_memory=True`）有不小的额外开销，会拖慢被 profile 的那几步。
   - **建议**：先在小规模（比如单机 1~2 张卡、小模型）复现问题再开 profiler；或者只跑很短的
     `--profile-step-start/--profile-step-end` 区间（比如 3~5 步）。
2. `profile_step_start=0` 时代码里 `warmup=0`（见上面 `1 if args.profile_step_start > 0 else 0`），意味着
   从第 0 步就是"正式记录"，没有 warmup 步，第一步的数据可能包含 CUDA context 初始化等噪声，一般建议
   `profile-step-start` 至少设成 2~3，跳过模型刚起来的抖动。

### 2.3 依赖安装

```bash
pip install torch-tb-profiler   # tensorboard 里查看 PyTorch Profiler 面板需要的插件
```

（`torch.profiler` 本身随 PyTorch 自带，不需要单独装；`tensorboard` 已经在 `requirements.txt` 里。）

### 2.4 手把手：怎么开、怎么看

以 `scripts/run-qwen3-4B.sh` 为模板，在启动命令里加一组 `PROFILE_ARGS`：

```bash
PROFILE_ARGS=(
   --use-pytorch-profiler
   --profile-step-start 3
   --profile-step-end 5          # 只抓第 3~4 个 rollout（active = 5-3 = 2 步）
   --profile-target train_overall train_actor   # 同时看整体 rollout 粒度 和 micro-batch 粒度
   --tensorboard-dir /root/slime_profile_out
)
```

然后正常提交训练：

```bash
ray job submit --address="http://127.0.0.1:8265" \
   -- python3 train.py \
   ... \
   ${PROFILE_ARGS[@]}
```

训练跑到第 5 个 rollout（`profile_step_end`）之后，每个 rank 会在 `/root/slime_profile_out/` 下生成形如
`train_overall_rank_0.<hostname>_<pid>.<timestamp>.pt.trace.json.gz` 的文件。

**查看方式一（推荐，图形化）**：

```bash
tensorboard --logdir /root/slime_profile_out --port 6006
```

浏览器打开 `http://<机器IP>:6006`，顶部选 "PYTORCH_PROFILER" 标签页，可以看到：
- **Overview**：GPU 利用率、Kernel/Memcpy/Communication 各自占比的饼图 —— 一眼看出是"算子慢"还是
  "通信（NCCL）慢"还是"GPU 空闲（等 CPU 调度）"。
- **Kernel**：按耗时排序的 CUDA kernel 列表，找耗时最长的那几个。
- **Trace**：完整时间轴（Timeline），可以放大到微秒级看每个 stream 在做什么，"卡点"就是时间轴上出现大段
  GPU 空白（gap）的地方。

**查看方式二（无需装插件，纯文本/浏览器打开单个 trace）**：

如果不想装 `torch-tb-profiler`，可以把 `.pt.trace.json.gz` 解压后（`gunzip xxx.pt.trace.json.gz`），
在 Chrome 地址栏输入 `chrome://tracing`，或者打开 https://ui.perfetto.dev/ ，把解压后的 `.json` 文件拖进去，
同样能看时间轴，Perfetto 的界面通常比 chrome://tracing 更好用（支持按名字搜索、按线程/stream 过滤）。

**怎么定位"卡点"（举例）**：
- 如果 Timeline 上看到某个 rank 的 GPU 轨道长时间空白，同一时刻其他 rank 却在跑 —— 典型的 **PP/DP 负载不均衡**
  （某个 micro-batch 序列长度特别长），可以配合 `--balance-data` / `--balance-by-flops` 解决。
- 如果看到大量 `ncclAllReduce` / `ncclBroadcast` kernel 占比很高 —— 通信瓶颈，检查网络拓扑
  （`nvidia-smi topo -m`，是否有 NVLink）、`--tensor-model-parallel-size` 是否设置过大导致 TP 通信量太大。
- 如果 CPU 侧（Python/aten op launch）之间的间隔很大，GPU 却是空的 —— **launch overhead 主导**，常见于
  micro-batch 切得太碎（`--use-dynamic-batch-size` 配合过小的 `--max-tokens-per-gpu`），考虑调大。

---

## 3. 第三步：显存分析（定位 OOM / 显存碎片 / offload 卡顿）

### 3.1 涉及的开关

| 参数 | 来源 | 默认值 | 作用 |
|---|---|---|---|
| `--record-memory-history` | Megatron 原生，slime 用 `reset_arg` 改成 `action="store_true"`（`arguments.py:1309`） | `False` | 总开关 |
| `--memory-recorder` | slime 自定义（`arguments.py:1303-1308`） | `torch`，可选 `torch` / `memray` | 选择显存记录后端 |
| `--memory-snapshot-dir` | slime 自定义（`arguments.py:1286-1290`） | `.`（当前目录） | dump 文件存放目录 |
| `--memory-snapshot-path` | Megatron 原生（`ProfilingConfig.memory_snapshot_path`） | `snapshot.pickle` | dump 文件名后缀 |
| `--memory-snapshot-num-steps` | slime 自定义（`arguments.py:1291-1295`） | `None` | 跑够这么多个 rollout 后主动 dump 一次（不设的话只在 OOM 时自动 dump） |

只在 `--profile-target` 包含 `train_overall` 时才会启用（见 `TrainProfiler.__init__` 里
`if args.record_memory_history and ("train_overall" in args.profile_target)`），这是默认值，一般不用改。

### 3.2 两种后端的区别（`slime/utils/profile_utils.py:81-147`）

- **`torch`（`_TorchMemoryProfiler`）**：调用 PyTorch 原生 `torch.cuda.memory._record_memory_history(...)`，
  同时用 `torch._C._cuda_attach_out_of_memory_observer` 挂一个 **OOM 自动 dump 回调**——一旦这张卡真的 OOM，
  会自动把当前的显存分配历史 dump 成 pickle 文件，并打印堆栈（`traceback.print_stack()`）。这是**排查 OOM
  最有效的手段**，几乎零心智负担：开着它训练，等它自己 OOM 的时候自动留证据。
  另外如果设置了 `--memory-snapshot-num-steps`，跑到第 `N-1` 个 rollout（`rollout_id == N-1`）时也会主动
  dump 一次（即使没有 OOM），方便你看"正常训练过程中"的显存分布。
- **`memray`（`_MemrayMemoryProfiler`）**：Python 级别的内存分析器（不止 GPU，也能看 CPU 端 Python 对象的
  内存），`native_traces=True` 表示同时记录 C/C++ 层的调用栈。**必须**设置 `--memory-snapshot-num-steps`
  （代码里有 `assert`），因为 memray 没有 OOM 回调机制，只能按步数主动停止记录。

### 3.3 依赖安装

```bash
pip install memray   # 已经写在 requirements.txt，正常装好 slime 环境应该已经有；如果没有单独装一下
```

`torch` 后端不需要额外依赖。

### 3.4 手把手：怎么开、怎么看

**场景 A：怀疑训练会 OOM，想留证据**

```bash
DEBUG_ARGS=(
   --record-memory-history
   --memory-recorder torch
   --memory-snapshot-dir /root/slime_mem_snapshot
)
```

正常跑训练，如果某个 rank OOM 了，日志里会打印
`Observe OOM, will dump snapshot to /root/slime_mem_snapshot/memory_snapshot_time..._rank{N}_snapshot.pickle`，
文件已经自动生成，不需要你手动干预。

**查看方式**：打开 https://docs.pytorch.org/memory_viz （PyTorch 官方的 Memory Visualizer），把生成的
`.pickle` 文件拖进网页，可以看到：
- 一条随时间变化的显存占用曲线；
- 鼠标悬停在曲线的任意一点，能看到当前时刻每一块已分配显存对应的 **Python 调用栈**（得益于
  `stacks="all"`），直接定位到是哪一行代码申请了这块显存、有没有异常的"越攒越多"（说明有显存泄漏，
  比如某个 list 一直在往里 append tensor 没释放）。

**场景 B：不 OOM，但想看正常训练时的显存分布/是否有碎片**

```bash
DEBUG_ARGS=(
   --record-memory-history
   --memory-recorder torch
   --memory-snapshot-dir /root/slime_mem_snapshot
   --memory-snapshot-num-steps 5    # 跑完第 5 个 rollout(rollout_id=4) 后主动 dump 一次
)
```

跑到第 5 个 rollout 结束会自动 dump，同样用上面的 memory_viz 网页查看。

**场景 C：用 memray 看 Python/C 层的内存（不只是 GPU 显存）**

```bash
DEBUG_ARGS=(
   --record-memory-history
   --memory-recorder memray
   --memory-snapshot-dir /root/slime_mem_snapshot
   --memory-snapshot-num-steps 5
)
```

生成的文件用 memray 自带 CLI 分析：

```bash
memray flamegraph /root/slime_mem_snapshot/memory_snapshot_time..._rank0_snapshot.pickle
# 会生成一份 html 火焰图，用浏览器打开即可交互式下钻
memray tree /root/slime_mem_snapshot/....pickle    # 或者看命令行版本的调用树
```

---

## 4. 第四步：Rollout（SGLang 推理引擎）侧 profiling

### 4.1 为什么这一步单独拿出来

上面第 2、3 步都是**训练侧（Megatron/Ray Train Actor）**的开关，定义在 `slime/utils/arguments.py` 里，
跟着 `train.py` 的启动命令一起传。但 Rollout 生成（SGLang 推理引擎）是**独立的 HTTP 服务进程**，它的
profiling 不走 `train.py` 的 CLI 参数，而是**训练过程中，实时地通过 HTTP 请求去控制**。

调用链：
- `slime/backends/sglang_utils/sglang_engine.py:485` 的 `SGLangEngine.start_profile()` /
  `sglang_engine.py:516` 的 `stop_profile()`，本质是往每个 SGLang 引擎发 `POST /start_profile` /
  `POST /stop_profile` 的 HTTP 请求（引擎自己实现了这两个 admin 接口，代码在 `sglang/` 里，
  可以参考它自带的文档 `sglang/docs_new/docs/developer_guide/benchmark_and_profiling.mdx`）。
- slime 仓库里没有把这两个方法接到 `train.py` 的任何 CLI 开关上（也就是说：**没有一个 `--rollout-profile`
  之类的参数能自动帮你抓 rollout trace**），需要你在训练**跑起来之后**，另开一个终端手动触发。
- 仓库已经提供了现成的小工具：`tools/profile_rollout.py`，它会先请求 router 的 `/workers` 接口拿到所有
  推理 worker 的地址，然后对每一个 worker 分别发 `/start_profile` 或 `/stop_profile`。

### 4.2 手把手：怎么用

1. 正常启动训练（参考 `scripts/run-qwen3-4B.sh`），等 SGLang 引擎和 router 起来（一般日志里能看到
   router 监听的端口，slime 里通常是 sglang-router，默认端口可以在启动日志 grep `router` 找到，
   常见形如 `http://127.0.0.1:<port>`）。
2. 另开一个终端，训练跑到你想观察的某个 rollout 附近时，执行：

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

   `--num-steps 5` 表示自动抓 5 个 forward step（prefill/decode 各算一步）后自动停止，不需要你再手动调用
   `--action stop`（源码见 `sglang_engine.py:485-499` 注释：`If it is set, profiling is automatically
   stopped after this step`）。如果不传 `--num-steps`，则需要你再手动执行一次：

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:30000 --action stop
```

3. 每个 worker 会各自把 trace dump 到它自己配置的目录（`--output-dir` 传给引擎，如果为空则用引擎自己的
   环境变量 `SGLANG_TORCH_PROFILER_DIR`，默认 `/tmp`）。产出格式同样是 `torch.profiler` 的
   `.pt.trace.json.gz`，**查看方式跟第 2 步完全一样**（tensorboard / chrome://tracing / perfetto）。

### 4.3 进阶：只看 prefill 或只看 decode

`tools/profile_rollout.py --profile-by-stage` 会让 SGLang 按 "prefill 阶段" 和 "decode 阶段"分别落盘
trace（而不是混在一起），适合区分"长 prompt 处理慢"还是"逐 token 生成慢"这两种完全不同的瓶颈。

### 4.4 进阶：结合 NVTX + Nsight Systems 看逐层耗时

SGLang 引擎支持 `--sglang-enable-layerwise-nvtx-marker`（启动 SGLang 引擎时的参数，slime 会把
`--sglang-*` 开头的参数转发给 SGLang 引擎，具体转发逻辑见 `slime/backends/sglang_utils/arguments.py`），
开了之后每一层 Transformer layer 都会打一个 NVTX marker，配合下面第 5 步的 `nsys` 使用，可以在 Nsight
Systems 的 timeline 里精确看到"第几层"、"attention 还是 MLP"具体耗时多少。用法可参考
`sglang/docs_new/docs/developer_guide/benchmark_and_profiling.mdx` 里 "Using `--enable-layerwise-nvtx-marker`
with Nsight Systems" 一节，思路是：nsys 全程录制 + `/start_profile` 传 `activities: ["CUDA_PROFILER"]`
来精确控制 `cudaProfilerStart/Stop` 的起止点（配合 `nsys ... --capture-range=cudaProfilerApi` 只抓这段区间，
省文件大小）。

---

## 5. 第五步（全局视角）：NVIDIA Nsight Systems（`nsys`）

### 5.1 这是什么，什么时候该用它

`torch.profiler` 只能看到"PyTorch 能感知到"的东西（Python 进程内的算子/kernel）。而 `nsys` 是
**系统级**的 profiler，可以同时看到：
- 多进程/多 GPU 的时间线对齐（Ray 起的多个 actor 进程、多张卡）；
- CPU 侧系统调用、线程调度、锁等待；
- CUDA driver API、NCCL 通信、显存拷贝；
- 如果代码里打了 NVTX marker（`torch.cuda.nvtx.range_push/pop`，或 SGLang 的 layerwise NVTX），
  能在时间轴上看到语义化的区间名字。

**什么时候用它，而不是 torch.profiler**：怀疑瓶颈不是某个算子本身慢，而是"进程之间互相等"（比如 colocate
模式下训练等推理 offload 显存、多机之间 NCCL 通信被网络打满、CPU 侧 Python GIL/调度导致 GPU 空闲）。

### 5.2 重要说明：slime 训练侧目前**没有**接入 Megatron 原生的 nsys 开关

Megatron 原生还定义了这些 profiling 参数（`common_config.py:28-67` 的 `ProfilingConfig`）：

- `--profile`（对应字段 `use_nsys_profiler`）
- `--nvtx-ranges`
- `--record-shapes`（配合 `torch.autograd.profiler.emit_nvtx` 用）

这些参数在 Megatron **自己的** `megatron/training/training.py` 训练主循环里会被用来调用
`torch.cuda.cudart().cudaProfilerStart()/Stop()` 和 `torch.autograd.profiler.emit_nvtx()`。但是
**slime 用的是自己写的训练循环**（`slime/backends/megatron_utils/actor.py` 里的 `train_actor` /
`model.py` 里的 `train()` 函数），并没有调用 Megatron 那套 nsys 相关的代码，所以这几个参数目前
**传给 slime 的 `train.py` 是不生效的**（不会报错，但也没有实际效果）。

因此，在 slime 里对 Megatron 训练进程做 `nsys` 分析，实践上要用**整进程包裹**的方式：直接用 `nsys profile`
包一层去启动会跑到 Megatron 训练代码的那个 Python 进程。

### 5.3 依赖安装

`nsys` 是 NVIDIA CUDA Toolkit 附带的命令行工具（不是 pip 包）：

```bash
# 大概率机器上已经有（跟 nvidia-smi 一起装的 CUDA Toolkit 通常带 nsys）
which nsys
nsys --version

# 如果没有，去 https://developer.nvidia.com/nsight-systems 下载对应版本安装
# 或者容器里：apt-get install nsight-systems-<version>（视基础镜像而定）
```

查看结果推荐装桌面版 **Nsight Systems GUI**（在你自己的电脑/Mac 上装，不需要在训练机上），把生成的
`.nsys-rep` 文件拷回本地用 GUI 打开，图形化体验最好。命令行也能看摘要（见下）。

### 5.4 手把手：怎么用

因为 slime 用 Ray 拉起子进程（每个 GPU 一个 Megatron worker 进程，由 Ray 管理），直接在最外层
`nsys profile python3 train.py ...` 只能抓到 Ray driver 进程本身，抓不到真正跑训练的 worker 进程。
两种可行方式：

**方式一：只想快速看单卡/小规模复现（推荐先用这个入门）**

先脱离 Ray，直接用最小复现脚本（例如仓库里的 debug 模式 `--debug-rollout-only` / `--debug-train-only`，
或者干脆写一个只跑几步 forward+backward 的最小 Megatron 脚本），单进程运行：

```bash
nsys profile \
    -t cuda,nvtx,osrt \
    -s none \
    -o /root/nsys_out/train_trace \
    --force-overwrite true \
    python3 train.py --debug-train-only --load-debug-rollout-data /root/dump/{rollout_id}.pt ...
```

- `-t cuda,nvtx,osrt`：采集 CUDA API、NVTX 区间、操作系统调用（线程/系统调用调度）。
- `-s none`：不用 CPU 抽样（sampling），只用 trace，产出文件更小更聚焦。
- `--debug-train-only --load-debug-rollout-data ...` 是 slime 自带的调试开关（见
  `slime/utils/arguments.py:1247-1263` 附近注释），可以跳过真实 SGLang 推理，只跑训练部分，方便你专注分析
  训练侧而不用等 rollout。

**方式二：Ray 多进程场景，通过环境变量让 Ray 起的每个 worker 自动被 nsys 包裹**

Ray 的每个 actor/worker 都是单独的 `python` 子进程，可以通过设置环境变量
`RAY_worker_process_setup_hook` 或者更简单粗暴地——在 `ray start`/`ray job submit` 的 runtime env 里，
把 Python 可执行文件替换成一个 wrapper 脚本，例如：

```bash
# nsys_wrapper.sh
#!/bin/bash
exec nsys profile -t cuda,nvtx,osrt -s none \
     -o /root/nsys_out/worker_$$ --force-overwrite true \
     python3 "$@"
```

然后把 `RUNTIME_ENV_JSON` 里的 `PYTHONPATH` 那一层，改成让 Ray 用这个 wrapper 启动 worker
（不同 Ray 版本配置方式略有差异，具体可参考 Ray 官方文档 "How to profile Ray workers with Nsight
Systems"）。这个方式配置略复杂，**建议先用方式一在小规模上定位到大致问题范围**，再考虑要不要上大规模。

**如果只是想看整机全局情况（不追求精确到某个 rank）**：也可以简单地对着已经在跑的训练进程，用
`nsys profile -p <pid>` 附加式采样一段时间（不是所有 nsys 版本/权限配置都支持附加采样，视机器环境而定）。

### 5.5 怎么看结果

```bash
# 命令行看摘要统计（不需要 GUI）
nsys stats /root/nsys_out/train_trace.nsys-rep
```

会输出各类耗时排行榜（CUDA kernel 耗时 Top N、NVTX 区间耗时 Top N、CUDA API 调用耗时 Top N 等），
文本形式，适合先扫一眼。

真正定位"卡点"建议用 **Nsight Systems GUI** 打开 `.nsys-rep` 文件：
- 每个 CUDA stream/每个进程一条时间轴，纵向对比，一眼看出"这段时间哪些 GPU 在空闲"；
- 放大到具体某一段空白区间，看紧邻它的是什么 CUDA API / NVTX 区间，往往就是"在等这个东西"；
- 如果用了 NVTX（比如 SGLang 的 layerwise marker，或者你自己在代码里加
  `torch.cuda.nvtx.range_push("xxx")` / `range_pop()`），区间会用你起的名字标注，非常直观。

---

## 6. 手把手完整示例：一次完整的"定位卡点"流程

假设你的 Qwen3-4B GRPO 训练（`scripts/run-qwen3-4B.sh`）感觉比预期慢，按下面顺序排查：

**Step 1 —— 先看免费的 perf 指标（5 分钟）**

加上 `--use-wandb ...`（或者不加，直接看 stdout 日志 grep `perf`），跑几十个 rollout，观察：
- `perf/wait_time_ratio`：如果持续 > 0.4~0.5，说明训练大部分时间在等 rollout，**下一步该分析 rollout 侧**（跳到
  Step 3）。
- 如果这个比例很低，说明训练本身计算量就很大，**下一步该分析训练侧**（继续 Step 2）。

**Step 2 —— 训练侧慢：开 PyTorch Profiler**

```bash
PROFILE_ARGS=(
   --use-pytorch-profiler
   --profile-step-start 3
   --profile-step-end 5
   --profile-target train_actor
   --tensorboard-dir /root/slime_profile_out
)
```

加进启动命令，跑到第 5 个 rollout 后，`tensorboard --logdir /root/slime_profile_out`，看 Kernel 面板，
定位耗时最长的算子/通信，针对性调整并行策略（TP/PP/CP/EP 大小）、`--recompute-*` 重计算策略、
`--use-dynamic-batch-size` + `--max-tokens-per-gpu`。

**Step 3 —— Rollout 侧慢：开 SGLang profiler**

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:<router端口> \
    --action start --output-dir /root/sglang_profile_out --num-steps 5 --activities GPU
```

同样用 tensorboard/perfetto 打开产出的 trace，看是 prefill 慢还是 decode 慢（配合
`--profile-by-stage`），检查显存是否够（`--sglang-mem-fraction-static` 是否太保守/太激进导致频繁 KV cache
换出）、是否长尾请求拖慢了整批（看 `perf/longest_sample_tokens_per_sec` vs
`perf/tokens_per_gpu_per_sec` 的差距）。

**Step 4 —— 怀疑显存问题（OOM / offload 卡顿）：开显存 snapshot**

```bash
DEBUG_ARGS=(
   --record-memory-history
   --memory-recorder torch
   --memory-snapshot-dir /root/slime_mem_snapshot
)
```

跑着，等它 OOM 自动落盘，或主动加 `--memory-snapshot-num-steps` 定期看显存曲线，用
https://docs.pytorch.org/memory_viz 打开分析。

**Step 5 —— 想看更全局的"进程间互相等待"：上 `nsys`**

用第 5 节的方式一，先在最小复现场景（`--debug-train-only` + dump 好的 rollout 数据）下跑一遍 `nsys profile`，
GUI 打开看 CPU/GPU 时间轴整体的空闲情况，尤其关注 colocate 模式下 offload/onload 的那几段时间。

---

## 7. 参数速查表

| 参数 | 来源 | 类型/默认值 | 作用简述 |
|---|---|---|---|
| `--use-pytorch-profiler` | Megatron 原生 | bool, `False` | 训练侧 PyTorch Profiler 总开关 |
| `--profile-step-start` | Megatron 原生 | int, `10` | 训练侧 profiler 起始 rollout step |
| `--profile-step-end` | Megatron 原生 | int, `12` | 训练侧 profiler 结束 rollout step |
| `--profile-target` | slime (`arguments.py:1296`) | list, `[train_overall]`，可选 `train_overall/train_actor/train_log_probs` | 训练侧 profiler 抓取粒度 |
| `--tensorboard-dir` | Megatron 原生 | str，需显式指定 | 训练侧 trace / tensorboard 日志输出目录 |
| `--record-memory-history` | Megatron 原生（slime reset 为 flag） | bool, `False` | 显存历史记录总开关（含 OOM 自动 dump） |
| `--memory-recorder` | slime (`arguments.py:1303`) | str, `torch`，可选 `torch/memray` | 显存记录后端 |
| `--memory-snapshot-dir` | slime (`arguments.py:1286`) | str, `.` | 显存 dump 文件目录 |
| `--memory-snapshot-path` | Megatron 原生 | str, `snapshot.pickle` | 显存 dump 文件名后缀 |
| `--memory-snapshot-num-steps` | slime (`arguments.py:1291`) | int, `None` | 跑够 N 个 rollout 后主动 dump 一次显存快照（memray 模式下必填） |
| （无 CLI 开关，运行时 HTTP 控制） | `sglang_engine.py:485` `start_profile()` / `tools/profile_rollout.py` | - | Rollout（SGLang）侧 PyTorch Profiler，训练跑起来后手动触发 |
| `--sglang-enable-layerwise-nvtx-marker` | 转发给 SGLang 引擎 | bool | 逐层打 NVTX marker，配合 nsys 看层级耗时 |
| `--profile`（Megatron 的 `use_nsys_profiler`）、`--nvtx-ranges`、`--record-shapes` | Megatron 原生 | - | **在 slime 里目前不生效**（slime 训练循环没接入 Megatron 原生的 nsys 逻辑），如需 nsys 请用整进程包裹方式（第 5 节） |
| `--debug-train-only` + `--load-debug-rollout-data` | slime (`arguments.py:1247` 附近) | - | 跳过真实 SGLang 推理，只跑训练部分，方便单独 profile 训练侧 |
| `--debug-rollout-only` | slime | bool | 跳过训练，只跑 rollout，方便单独 profile 推理侧 |

配套需要安装的依赖：

```bash
pip install torch-tb-profiler   # tensorboard 里看 PyTorch Profiler 面板
pip install memray              # 已在 requirements.txt；memray 后端的显存/内存分析
# nsys 是 CUDA Toolkit 自带的命令行工具，不是 pip 包，需要单独安装 Nsight Systems
```

---

## 8. 常见问题 / 踩坑提示

1. **开了 `--use-pytorch-profiler` 但 `--tensorboard-dir` 目录下什么都没有？**
   检查是不是训练还没跑到 `--profile-step-end` 那一步（schedule 是按 rollout 累积推进的，`on_trace_ready`
   只在 `repeat` 完成后才触发写盘）；也检查是不是被 kill/异常退出导致 profiler 没走完 schedule。

2. **trace 文件特别大，tensorboard 打开很卡？**
   把 `--profile-step-end - --profile-step-start` 调小（比如只抓 1~2 步），或者把 `--profile-target`
   只留一个最关心的值，减少同时产出的 trace 数量；`with_stack=True` 是主要的体积/开销来源，目前代码里是硬编码
   打开的（`profile_utils.py:75`），如果需要关闭需要改代码。

3. **多 rank 一起开 profiler，会不会互相干扰或者拖慢训练？**
   会有一定开销（stack 记录 + 显存 profile 都不是免费的），且因为没有 `profile_ranks` 过滤，所有 rank
   都会各自产生开销和文件。生产训练 debug 时**建议先在小规模复现问题**，不要在大规模正式训练任务上直接开。

4. **SGLang 侧的 `--num-steps` 到底是"多少个 rollout"还是"多少个 forward step"？**
   是 SGLang 引擎自己的 forward step（prefill 一次算一步，decode 每个 token 也算一步），跟训练侧的
   "rollout step"是完全不同的计数单位，不要混用理解。

5. **想同时对齐"训练侧 trace"和"rollout 侧 trace"的时间轴做整体分析？**
   目前两边是分别产出各自的 trace 文件（一个由 `--tensorboard-dir` 控制，一个由
   `tools/profile_rollout.py --output-dir` 控制），没有自动合并的工具；如果需要严格对齐时间轴分析
   "训练在等 rollout" 这种跨进程时序问题，建议用第 5 节的 `nsys` 全局采样方式，它能把多进程放到同一条
   系统时间轴上比较（前提是各进程时钟同步，同机训练天然满足）。

6. **colocate 模式（训练和推理共享 GPU）下，怎么区分"这段时间是训练在跑"还是"推理在跑"？**
   看 `perf/wait_time_ratio` 结合训练日志里 `offload/onload` 相关的 log（`self.wake_up()` /
   `self.sleep()` 附近，`megatron_utils/actor.py` 里能看到 `offload_train` 分支），或者在 `nsys` 的
   timeline 上直接看某一时刻是训练进程的 CUDA context 活跃，还是 SGLang 引擎进程的活跃。
