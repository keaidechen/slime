# 06 Slime 强化学习全链路性能分析

Slime 的性能问题不是“训练性能 + 推理性能”的简单相加。一个 rollout step 里还包含数据准备、reward、优势计算、权重同步、显存 offload/onload，以及训练侧和推理侧互相等待。真正要优化的是端到端关键路径，而不是某个局部 kernel 的峰值。

本章以当前仓库代码为准。先照着做，不需要一开始就读懂 Ray、Megatron 或 SGLang 源码。

## 1. 先画出正在测量的系统

一个简化的 Slime 循环是：

```text
prompt/data source
  -> SGLang rollout
  -> reward / tools / environment
  -> filter + advantage/return
  -> Megatron log-prob/reference/actor train
  -> Megatron -> SGLang weight update
  -> 下一轮
```

Colocate 模式还会插入模型或 KV Cache 的 release、offload、onload。异步模式中若训练与 rollout 能重叠，端到端耗时取决于较慢的一侧和同步边界；不能把所有阶段耗时直接相加。

第一次分析前写下：

- 训练 GPU 数和 rollout GPU 数；
- 是否 colocate；
- 每轮样本数、每 prompt 采样数、输入/输出长度；
- 模型、精度、TP/PP/CP/EP/DP；
- reward 是本地计算、远程服务还是 agent/tool 环境；
- 同步、部分异步还是 fully async；
- 每次 rollout 后训练多少 step，多久更新一次权重。

这些是 workload，不是无关配置。

## 2. 第一次运行只看 `perf/*`

先用一个已经能正确运行的小配置跑 10～20 个 rollout。不要立即开 profiler。保存完整日志，并在日志或 TensorBoard/W&B 中观察：

| 指标 | 它回答的问题 |
|---|---|
| `perf/rollout_time` | 一轮样本生成总共用了多久？ |
| `perf/tokens_per_gpu_per_sec` | rollout 每张 GPU 的原始 response token 吞吐是多少？ |
| `perf/effective_tokens_per_gpu_per_sec` | 去除无效 response 部分后，真正有用的吞吐是多少？ |
| `perf/longest_sample_tokens_per_sec` | 最长样本是否控制了整轮尾部？ |
| `perf/non_generation_time/*` | tool、环境、reward 等非生成时间是否显著？ |
| `perf/log_probs_time` / `ref_log_probs_time` | actor/reference log-prob 前向用了多久？ |
| `perf/actor_train_time` | actor 训练主体用了多久？ |
| `perf/update_weights_time` | 权重更新是否成为同步墙？ |
| `perf/actor_train_tflops` | 训练侧估算计算效率是否变化？ |
| `perf/actor_train_tok_per_s` | 训练侧 token 吞吐是多少？ |
| `perf/step_time` | 训练侧工作加等待的完整 step 时间 |
| `perf/wait_time_ratio` | 训练侧有多少比例在等待另一侧？ |

具体 key 取决于本次启用的算法和阶段。当前实现可从 [train_metric_utils.py](../../slime/utils/train_metric_utils.py) 和 [rollout.py](../../slime/ray/rollout.py) 核对。

### 2.1 先做三个判断

1. **是否已经稳态**：前几轮包含模型初始化、compile、CUDA Graph capture 或 cache 建立，应从稳态统计中排除。
2. **方差是否可接受**：至少看中位数、p95 和每轮曲线；只看均值会隐藏长尾。
3. **瓶颈在哪一侧**：`wait_time_ratio` 高说明训练侧大量等待，但它不自动证明 SGLang kernel 慢，reward、长尾 agent、数据处理和同步同样可能是原因。

一个实用的第一层归因是：

```text
rollout_time 高
  -> 看生成 token、最长样本和 non_generation_time
train_time 高
  -> 看 log_probs / ref_log_probs / actor_train
update_weights_time 高
  -> 看权重转换、传输、量化、collective 和 onload
wait_time_ratio 高
  -> 对齐训练和 rollout 时间线，找真正阻塞训练的事件
```

## 3. 先拆开 rollout 和训练再 profile

全链路的输入和并发都在变化，不适合直接做第一次下钻。Slime 提供了可复现的分离调试方式。

### 3.1 固定 rollout 并保存数据

在现有启动命令后加入：

```bash
--debug-rollout-only \
--save-debug-rollout-data /tmp/slime_debug/rollout_{rollout_id}.pt
```

这时不初始化 Megatron，只运行 rollout。先检查：

- 输出是不是正常文本；
- response 长度、truncated ratio、reward 是否符合预期；
- 文件 `/tmp/slime_debug/rollout_0.pt` 等是否生成；
- 多跑几轮时，输入分布和性能是否稳定。

`{rollout_id}` 是 Python format 占位符，保留花括号，不要先被 shell 展开。

### 3.2 固定输入，只跑训练

复用刚才的数据，在原启动命令后加入：

```bash
--debug-train-only \
--load-debug-rollout-data /tmp/slime_debug/rollout_{rollout_id}.pt
```

`--load-debug-rollout-data` 会跳过 SGLang 初始化。现在每次实验输入相同，可以公平比较 TP/PP/CP、micro-batch、activation checkpoint、kernel backend 等训练变量。

先验证 loss、KL、grad norm 和有效 token 数与基线一致，再比较性能。固定数据并不代表可以跳过正确性检查。

完整参数语义见 [Debug 指南](../zh/developer_guide/debug.md)。

## 4. 训练侧 PyTorch Profiler

### 4.1 只抓最短且稳定的窗口

在训练启动参数中加入：

```bash
--use-pytorch-profiler \
--profile-target train_actor \
--profile-step-start 3 \
--profile-step-end 5 \
--tensorboard-dir /tmp/slime_train_profile
```

当前 `--profile-target` 可取一个或多个：

- `train_overall`：从 Slime 训练生命周期整体观察；
- `train_actor`：只看 actor 训练迭代；
- `train_log_probs`：只看 log-prob 计算迭代。

`profile-step-start/end` 的 step 含义会随 target 改变：整体 target 对应 rollout 级推进，局部 target 对应相应 iterator。先抓 2 个 active step，并检查 trace 中到底包含了什么，再扩大窗口。

当前 [profile_utils.py](../../slime/utils/profile_utils.py) 会为每个分布式 rank 输出 trace，并启用 shape、stack、memory 和 FLOPs，开销和文件都较大。第一次应使用小模型、少量 rank、固定 debug 数据；不要在生产任务上长期开启。

用 TensorBoard 打开：

```bash
tensorboard --logdir /tmp/slime_train_profile
```

也可以解压 `.trace.json.gz` 后上传到 [Perfetto](https://ui.perfetto.dev/)。按第 4 章的方法检查 pipeline bubble、collective、rank 间长尾和 GPU gap，再按第 3 章只下钻关键路径中的 kernel。

### 4.2 训练 trace 的阅读顺序

1. 找完整且稳态的 train iteration；
2. 对比所有 rank 的 iteration 起止时间；
3. 分出 forward、backward、optimizer、log-prob、collective；
4. 找等待最长的 rank，而不只是 kernel 最长的 rank；
5. 判断 GPU gap 前 CPU 在做什么；
6. 判断 collective 是关键路径，还是被计算成功覆盖；
7. 最后才对热点 CUDA kernel 用 Nsight Compute。

## 5. 训练侧显存和主机内存

### 5.1 PyTorch CUDA allocator 快照

```bash
--record-memory-history \
--memory-recorder torch \
--profile-target train_overall \
--memory-snapshot-dir /tmp/slime_memory \
--memory-snapshot-path snapshot.pickle \
--memory-snapshot-num-steps 5
```

当前代码只在 `profile-target` 包含 `train_overall` 时创建 memory recorder。`memory-snapshot-path` 是文件名后缀；实际文件还包含时间和 rank。

两种用法：

- 不设置 `--memory-snapshot-num-steps`：持续记录，发生 CUDA OOM 时自动 dump；
- 设置为 `N`：在第 `N` 个 rollout 附近主动 dump，适合观察正常峰值和碎片。

将 pickle 拖入 [PyTorch Memory Viz](https://pytorch.org/memory_viz)。按时间找峰值，再检查大块存活 tensor、频繁分配释放和 reserved/allocated 差距。快照主要解释 PyTorch allocator 可见的显存，NCCL、CUDA context 或第三方库分配可能不可见。

### 5.2 Memray 看 Python/native 分配

```bash
--record-memory-history \
--memory-recorder memray \
--profile-target train_overall \
--memory-snapshot-dir /tmp/slime_memray \
--memory-snapshot-path train.bin \
--memory-snapshot-num-steps 3
```

Memray 模式必须设置步数。生成后：

```bash
memray flamegraph /path/to/generated_file
memray tree /path/to/generated_file
```

它适合查 Python/C/C++ 主机内存，不等价于 CUDA allocator snapshot。

## 6. Rollout 侧 SGLang Profiler

### 6.1 让 Slime 初始化后等待

要手工构造压力而不让正常 rollout 立刻运行，在启动参数中替换 rollout function：

```bash
--rollout-function-path slime.rollout.sleep_rollout.sleep
```

日志会打印 router 地址，例如 `http://127.0.0.1:3000`。检查 worker：

```bash
curl http://127.0.0.1:3000/workers
```

### 6.2 同时启动所有 worker 的 profile

```bash
python tools/profile_rollout.py \
  --router-url http://127.0.0.1:3000 \
  --action start \
  --output-dir /tmp/slime_rollout_profile \
  --num-steps 10 \
  --activities GPU \
  --profile-by-stage
```

然后向 router 发送一组受控请求。如果要看 CPU 调度，再把 `CPU` 加入 activities；只有在必须找 Python 调用源时才加 `--with-stack --record-shapes`。

这里的 `num-steps` 是 SGLang engine 的 forward/profile step，不是 Slime rollout 轮数。工具会向 `/workers` 返回的每个 worker 发送请求，所以输出目录必须对对应 worker 可写；多机时尤其要确认目录语义。

需要提前停止：

```bash
python tools/profile_rollout.py \
  --router-url http://127.0.0.1:3000 \
  --action stop
```

### 6.3 阅读与汇总 rollout trace

在 Perfetto 中先按 prefill/decode 分开，再看 scheduler gap、CUDA Graph、attention/GEMM、DeepEP/NCCL 和 rank 长尾。当前仓库还提供面向 SGLang decode trace 的汇总脚本：

```bash
python tools/analyze_profile.py \
  --profile-dir /tmp/slime_rollout_profile \
  --rank 0

python tools/analyze_profile.py \
  --profile-dir /tmp/slime_rollout_profile \
  --all-ranks
```

脚本输出是线索，不是最终结论。回到原始 trace 核对其分类是否适配当前 backend 和 kernel 命名。

## 7. 看 sample 级端到端时间线

开启 rollout debug dump 后，每个 sample 可携带 trace。生成可视化：

```bash
python tools/trace_timeline_viewer.py \
  /tmp/slime_debug/rollout_0.pt \
  --no-serve
```

打开生成的 HTML，逐层回答：

1. prompt 何时进入队列？
2. prefill 和 decode 在哪段？
3. tool/reward 等非生成阶段在哪里？
4. 是所有样本慢，还是少数长尾控制整轮？
5. PD 分离时，prefill/decode lane 是否存在空洞或 handoff 等待？

这是分析 agentic、多轮工具调用和 reward 服务长尾的首选入口。详细字段见 [rollout trace 文档](../zh/developer_guide/trace.md)。

## 8. 用 Nsight Systems 看跨进程关键路径

PyTorch trace 更擅长算子；Nsight Systems 更适合 CPU、GPU、CUDA、NCCL 和系统调度的统一时间线。第一次先用固定数据和 `--debug-train-only` 缩小范围：

```bash
nsys profile \
  -t cuda,nvtx,osrt \
  -s none \
  -o /tmp/slime_train_nsys \
  python train.py \
    --debug-train-only \
    --load-debug-rollout-data /tmp/slime_debug/rollout_{rollout_id}.pt \
    ...其余原配置参数
```

真实命令通常由项目 shell 脚本、Ray 和多进程启动器包裹。先确认 `nsys` 是否捕获到实际 worker，而不只是 driver。Slime 使用自己的训练循环，不能假设 Megatron 原生的按 iteration capture 参数一定能直接控制这里的范围；以 trace 内容为准。

全链路 trace 文件会很大。只有在两侧独立分析后仍无法解释等待时，才抓一个极短全链路窗口，重点看：

- rollout 完成到训练开始之间；
- colocate release/offload/onload；
- 权重转换和 update；
- 训练结束到下一轮请求真正进入 engine 之间；
- Ray task、对象传输或 CPU serialization 的空洞。

## 9. 权重同步怎么分析

当 `perf/update_weights_time` 高时，依次做：

1. 固定模型、并行和 rollout 数据，确认多轮都慢；
2. 区分等待、权重转换/量化、通信、SGLang load 和 cache/graph 重建；
3. 看各 rank 是否同时开始、哪个 rank 最后结束；
4. 比较传输字节、实际改动密度和网络带宽；
5. 检查 TP/PP/EP 参数映射是否产生串行小传输；
6. 修改协议或频率前先验证权重更新后的数值一致性。

需要排除正确性问题时，可使用仓库的 `--check-weight-update-equal`，但它本身有额外开销，不用于正式基线。

## 10. 系统平衡实验

若训练和 rollout 使用独立 GPU，做一个小 sweep：

```text
方案 A：更多训练 GPU，较少 rollout GPU
方案 B：当前分配
方案 C：较少训练 GPU，更多 rollout GPU
```

每个方案固定总 GPU 数、样本和 token 分布，记录：

- 端到端有效 token/s 或每小时完成的有效样本；
- `rollout_time`、`train_time`、`update_weights_time`；
- `wait_time_ratio`；
- GPU 利用率和显存余量；
- reward、KL、有效 token 数等正确性指标。

目标不是让每张卡始终 100%，而是减少关键路径上的空闲，并提高最终有效产出。局部 actor TFLOPS 上升但端到端 step 变慢，不是优化成功。

## 11. 症状到下一步

| 症状 | 先收集 | 下一步 |
|---|---|---|
| `rollout_time` 高、GPU 也高 | 长度分布、prefill/decode trace | 按第 5 章查 batching、KV、kernel |
| `rollout_time` 高、GPU 低 | sample trace、CPU/queue、non-generation | 查 scheduler、tool/reward、长尾 |
| actor TFLOPS 低 | 固定 dump 的训练 trace | 按第 4 章查 shape、bubble、通信 |
| `wait_time_ratio` 高 | 两侧按同一时间轴的阶段数据 | 找训练真正等待的最后一个事件 |
| `update_weights_time` 高 | rank 时间、传输量、网络、转换阶段 | 拆权重同步，不先调训练 kernel |
| Colocate 周期性空洞 | nsys、显存曲线 | 查 release/offload/onload 和 graph 重建 |
| 偶发超慢轮次 | p95/p99、最长 sample、各 rank | 查长尾 agent、straggler、重试、抖动 |
| OOM | allocator snapshot + 非 PyTorch 显存 | 区分活跃 tensor、碎片和外部分配 |

## 12. 本章完成标准

你应该交付一个最小性能报告，明确写出：

- 一张端到端阶段表；
- rollout 与训练各自的稳定基线；
- 一个固定的 rollout dump；
- 至少一个训练或 rollout trace；
- 一个可证伪瓶颈假设；
- 一组只改变一个变量的 A/B 结果；
- 正确性未退化的证据。

不要以“GPU 利用率提高了”作为最终结论。Slime 的最终指标应是固定资源和质量约束下，端到端有效 token、有效样本或训练进展的提升。
