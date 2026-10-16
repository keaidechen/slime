# 04 权重同步与显存管理：RL 系统的"头号瓶颈"是如何被攻克的

相关基础：[Ray、队列与前置系统](<../../../06_Post_Training/27_权重发布异步与资源调度.md#s4>)。


> 对应综述（`00_rl_infra_survey.md`）§2.3「权重同步」与 §2.9「精度」。
> 每步训练后，数百 GB 权重要从 Megatron 训练栈搬进 SGLang 推理引擎——训练侧是 TP/PP/EP 切分的 Megatron 格式，推理侧是 HF 命名、另一种 TP 切分。本篇解读 slime 的完整链路。

---

### 阅读起点：传的是数值，还是布局？

参数是模型学到的数值；张量布局说明这些数值的名称、形状、切片方向和所在进程。TP（tensor parallelism，张量并行）把同一层的矩阵切开，PP（pipeline parallelism，流水线并行）把不同层放到不同进程，DP（data parallelism，数据并行）复制模型并处理不同样本，EP（expert parallelism，专家并行）把混合专家模型的专家分散保存。

假设线性层 $y=xW^T$，输入 $x=[1,2]$ 形状 `[2]`，完整权重为

```text
W = [[1,0], [0,1], [1,1], [2,1]]   shape [4,2]
y = [1,2,3,4]                       shape [4]
```

训练 TP=2 时 rank 0 可持有前两行 `[2,2]`，rank 1 持有后两行 `[2,2]`；推理 TP=1 需要完整 `[4,2]`。只改参数名而不找齐两片，推理无法得到相同的 y；把两片反序拼接仍是 `[4,2]`，却输出 `[3,4,1,2]`。换权正确性因此同时包含**值正确、布局正确、版本完整**。

以下字节估算采用 1 GB = $10^9$ 字节，1 GiB = $2^{30}$ 字节；BF16（16 位浮点）每元素 2 字节，FP32 每元素 4 字节。不要把参数个数直接当显存字节数。

## 1. 先分清：CLI 是两个维度，运行时有四个 updater

`slime/backends/megatron_utils/update_weight/` 目录：

| 运行时类 / 文件 | 实际数据路径 | 适用场景 |
|---|---|---|
| `UpdateWeightFromDistributed` | 全量参数经 NCCL 广播 | 训练/推理分卡（disaggregate） |
| `UpdateWeightFromTensor` | colocated engine 经 CUDA IPC；若同一拓扑还有远端 engine，则远端部分经 NCCL | colocate 或 colocate + remote 混合拓扑 |
| `UpdateWeightFromDisk` | 写完整 HF checkpoint，engine 从磁盘重载 | 共享存储、external engine、`release-train` |
| `UpdateWeightFromDiskDelta` | 写字节级 delta，各 rollout host patch 本地 checkpoint 后重载 | 非 colocate 的低带宽磁盘同步 |

用户只配置两个正交参数：

- `--update-weight-mode={full,delta}` 决定发完整参数还是字节增量；
- `--update-weight-transport={nccl,disk}` 决定走通信域还是文件系统。

**没有 `--update-weight-transport=ipc` 这个选项。** 当配置是默认的 `full + nccl` 且开启 `--colocate` 时，`MegatronTrainRayActor.init` 自动选择 `UpdateWeightFromTensor`，再按每个 engine 的 GPU offset 判断哪些 engine 与训练 rank 共置：共置部分用 CUDA IPC，超出 actor GPU 范围的远端部分仍用 NCCL。因此 IPC 是拓扑推导出的内部执行计划，不是第三个用户可选 transport。

精确选择顺序在 `MegatronTrainRayActor.init`：

```text
mode=delta                         -> UpdateWeightFromDiskDelta（且 transport 必须是 disk）
mode=full, transport=disk          -> UpdateWeightFromDisk
mode=full, transport=nccl, colocate-> UpdateWeightFromTensor（可同时覆盖 colocated + remote engines）
mode=full, transport=nccl          -> UpdateWeightFromDistributed
```

这也回答了“为什么目录里有四个类、CLI 却只有两个 transport”的问题：**类是具体执行策略，CLI 描述的是用户意图；colocate 拓扑负责把意图进一步解析成 IPC/NCCL 组合。** 训练侧统一入口仍是 `MegatronTrainRayActor.update_weights` → `self.weight_updater.update_weights()`。

---

## 2. NCCL 模式逐行解读（最常用）

### 2.1 建组：训练 rank 与推理引擎组成临时通信域

```75:92:slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"

        if self._is_pp_src_rank:
            ...
            self._model_update_groups = connect_rollout_engines_from_distributed(
```

设计要点：

PD 分离（Prefill–Decode disaggregation）把输入处理与逐步生成放到不同执行资源。它允许分别调度两阶段，但增加 KV 传输、交接等待和资源预留。

- **只有 DP=0 且 TP=0 的 rank 当"源"**：因为 DP 是纯副本、TP 切片会被 all-gather 补全（见 §2.2），所以每个 PP 段只需一个源 rank；
- **每个 PP rank 一个 NCCL 组**（`slime-pp_0`、`slime-pp_1`…）：流水线并行下不同 PP 段持有不同层，各自并行推权重；
- `connect_rollout_engines_from_distributed`（268-314）：训练源 rank 当 rank 0、所有引擎的全部 GPU 依次排 rank，world_size = `sum(engine_gpu_counts) + 1`。注意它支持**异构 TP 的引擎**（注释里举例 prefill TP=2 / decode TP=4 各占不同 rank 数）——这正对应 PD 分离部署。

### 2.2 发送：gather → 转换 → 分桶 → 广播

`update_weights`（102-134）定义了总流程：**pause_generation → flush_cache → 发送 → continue_generation**。暂停生成是必须的——权重被替换到一半时采样会产生"混合权重"的脏数据。

`_send_weights`（136-146）分两遍：**非专家参数（TP 切分）一遍，MoE 专家参数（EP 切分）一遍**，中间以 barrier 隔开。非专家参数的迭代器：

```161:176:slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            param = all_gather_param(name, param)
            if not self._is_pp_src_rank:
                continue
            hf_chunk = convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
            chunk_bytes = sum(t.numel() * t.element_size() for _, t in hf_chunk)
            if buffer and buffer_size + chunk_bytes > self.args.update_weight_buffer_size:
                yield buffer
```

三步流水线：

1. **`all_gather_param`**（common.py）：把 TP 切片的参数在 TP 组内 all-gather 成完整参数（所有 TP rank 参与计算，但只有源 rank 保留结果）；
2. **`convert_to_hf`**（`slime/backends/megatron_utils/megatron_to_hf/`）：Megatron 命名/布局 → HF 命名/布局。例如 Megatron 的 `module.decoder.layers.0.self_attention.linear_qkv.weight`（QKV 融合）要拆成 HF 的 `q_proj/k_proj/v_proj`；每个模型族一个转换模块（`deepseekv3.py`、`qwen3moe.py`、`glm4moe.py`…），FP8 量化由 `processors/quantizer_fp8.py` 等处理（06/13 篇详解）；
3. **分桶（bucketing）**：累积到 `--update-weight-buffer-size` 就 yield 一桶，以限制在途参数量并批量提交控制信息。它与 tensor 路径的 `FlattenedTensorBucket` 要区分，具体差异和成本见下一小节。

专家参数的 EP all-gather（`_ep_gather_and_convert`，204-238）更精巧：先用 `all_gather_object` 对齐各 EP rank 的参数名清单，再异步 `all_gather` 收齐所有 rank 的专家分片，最后源 rank 把"哪个 expert 的哪层"映射回 HF 的 `experts.{i}` 命名。

### 2.2.1 分桶为何更快，以及它不能保证什么

设共有 $K$ 次传输、总数据量 $S$ 字节，每次固定启动成本 $\alpha$ 秒，有效带宽 $\beta$ 字节/秒。只估算串行传输，则

$$
T\approx K\alpha+S/\beta.
$$

100 个小张量总共 1 MB，假设启动成本 20 微秒、带宽 10 GB/s：逐张量启动要 $100\times20+100=2100$ 微秒；若协议允许打成 4 次，约为 $4\times20+100=180$ 微秒。这里没有减少总权重，只减少了启动次数。

但“把参数装进 Python 列表的一桶”不必然等于“一次 NCCL 操作”。专用 distributed 路径会在一批控制请求后逐 tensor 广播；flattened bucket 路径把 tensor 拼成连续存储，才可能进一步减少数据面调用。必须沿具体 updater 确认。

桶越大，临时完整参数和转换缓冲区的峰值越高，第一桶就绪也可能更晚。同步总时间还包括 gather、格式转换、接收端加载和暂停窗口，不能仅用网络带宽除出端到端时间。单个大参数超过桶阈值时仍可能独占超大桶，阈值也不是硬显存上限。

### 2.3 广播与防死锁

```240:265:slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py
        # lock the rollout engines to prevent dead lock on broadcast.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        refs = update_weights_from_distributed(...)
        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
```

- **全局锁**（`Lock` actor，`slime/ray/utils.py`）：多个 PP 源 rank 同时向同一批引擎广播会争抢 NCCL 资源导致死锁，锁把广播串行化；
- **元数据走 Ray、张量走 NCCL**（326-355）：先把 `names/dtypes/shapes` 经 `engine.update_weights_from_distributed.remote()` 发给引擎（引擎据此分配接收 buffer），再 `dist.broadcast(param.data, src=0, async_op=True)` 推张量本体。控制面与数据面分离，是这类系统的惯用手法。

把锁、暂停和通信完成分开看：

| 条件 | 解决的问题 | 不解决的问题 |
|---|---|---|
| `pause_generation` | 换权期间不启动不相容的生成计算 | 不会替发送方决定 collective 的调用顺序 |
| `rollout_engine_lock` | 不同 PP 源向共享接收者发起操作的次序协调 | 不表示所有参数已安装完成 |
| broadcast 的工作完成与接收端返回 | 相关 buffer 不再被此次操作使用 | 不表示其它桶、其它层也已更新 |
| 全轮发布完成再继续生成 | 所有目标层形成同一版本 | 不会让旧版本 KV 自动适配新权重 |

一次 GPU 异步提交返回的是工作句柄；只有满足对应等待/stream 依赖之后才能覆盖发送 buffer。CUDA IPC 还要求生产进程保持共享 allocation 存活，直到消费进程加载完成。`converted_named_tensors.clear()` 放在请求完成后，目的正是避免接收端还在读、发送端已经复用内存。

KV 是旧权重对上下文的计算结果。例如某层 key 为 $K=XW_K^T$，即使 X（token 对应输入）相同，换成 $W_K'$ 后通常有 $XW_K'^T\ne K$。因此换权清缓存是在维护数值语义，并不仅是整理显存碎片。

### 2.4 入口处的容错与重连

`actor.py:567-628` 的 `update_weights` 在调用 `weight_updater` 之前还做了：

- **故障恢复**（571-574）：`use_fault_tolerance` 时先 `recover_updatable_engines`——健康监控发现挂掉的引擎会被重建（新引擎 = `num_new_engines > 0`），需要 `connect_rollout_engines` 把它们加进 NCCL 组（597-607）；
- **offload_train 的配合**（592-628）：训练显存被 torch_memory_saver 挂起期间，进程组要先 `reload_process_groups()`、传完再 `destroy_process_groups()`；权重传输本身在 `torch_memory_saver.disable()` 上下文里执行（609 行），避免传输内存被挂起。

---

## 3. IPC 与 Disk 模式（机制对比）

- **CUDA IPC（`update_weight_from_tensor.py`）**：每组 colocated 训练 rank 先用 Gloo `gather_object` 把序列化后的 bucket handle 汇到与 engine 对应的源 rank；`MultiprocessingSerializer` 传的是 CUDA IPC handle 和元数据，真正的大张量不进入 Ray object store。engine 打开 handle 后拷入模型权重。这里仍有小体积控制面通信，并非“完全零通信”；省掉的是跨进程复制整份权重和远端 NCCL 传输。
- **Disk 全量/增量（`update_weight_from_disk*.py`）**：训练侧把权重写成 HF checkpoint 目录（带版本号 `weight_v000123/`），引擎 `update_weights_from_disk` 重新加载。配套逻辑在 `slime/ray/actor_group.py:161-268`：

```226:253:slime/ray/actor_group.py
    def _reload_rollout_weights_from_disk(self, disk_weight_dir, weight_version):
        ...
        if self.args.update_weight_local_checkpoint_dir:
            # each host pulls the published checkpoint onto local disk (e.g. NVMe) and
            # the engines reload from there; the pull is disk-only, so it runs before
            # pause and overlaps generation
            ray.get([engine.pull_weights.remote(int(weight_version)) for engine in engines])
            ...
        ray.get([engine.pause_generation.remote() for engine in engines])
        ray.get([engine.flush_cache.remote() for engine in engines])
        ray.get([engine.update_weights_from_disk.remote(model_path=model_path, weight_version=weight_version) ...])
```

精妙点：`pull_weights`（把共享文件系统的新 checkpoint 拉到各机本地 NVMe）**在暂停生成之前**执行，与正在进行的推理重叠；只有最后的本地加载需要 pause。版本号机制（`_disk_weight_version`，actor_group.py:54/166-172）让引擎与训练侧能对账——CI 里会逐一校验 `get_weight_version`（actor_group.py:254-265）。Delta 模式对相邻版本做字节级差量编码；权重数值改变量小，不代表变化字节一定少，实际 IO 节省取决于可压缩性和差量链的维护成本。

---

## 4. 显存管理：colocate 的错峰艺术

一张 80GB 卡要轮流装"推理引擎（权重+KV cache）"和"训练栈（权重+梯度+优化器状态+激活）"，靠三层机制：

### 4.1 rollout 侧：`--offload-rollout`

主循环（train.py:23-33/55-56/83-88）在训练前调 `rollout_manager.offload.remote()`，其内部走 SGLang 的 `release_memory_occupation`：释放 KV cache 池（权重可保留或按 tag 释放）；训练完 `onload_weights` + `onload_kv` 恢复。拆成 weights/kv 两个动作（rollout.py:403-425 的 `onload_weights/onload_kv`）是因为：权重恢复后即可开始权重同步，KV 池可以等到权重同步完再重建——把串行步骤尽量重叠。

### 4.2 训练侧：`--offload-train`

基于 **torch_memory_saver**（一个通过 `LD_PRELOAD` hook CUDA 显存分配的库）：

```73:93:slime/ray/actor_group.py
        if self.args.offload_train and self.args.train_backend == "megatron":
            import torch_memory_saver
            ...
            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
```

`MegatronTrainRayActor.sleep/wake_up`（actor.py:204-234）把训练显存整体挂起到 CPU（`TMS_INIT_ENABLE_CPU_BACKUP` 表示挂起时留 CPU 副本，唤醒免重算）。因此主循环里 `offload_train` 为真时甚至不需要 `clear_memory()`（train.py:39-46 的注释说明）。

### 4.2.1 用显存账本检查错峰是否真的装得下

设卡容量 $C=24$ GiB，推理阶段权重 6 GiB、KV 8 GiB、其它 workspace 2 GiB，共 16 GiB；训练阶段参数/梯度/优化器/激活合计 20 GiB。两套同时常驻会超过容量，错峰后的基础峰值是 $\max(16,20)=20$ GiB。

这只是基础值。换权时若目标权重 6 GiB 与临时桶 4 GiB 同时存在，外加未释放的 KV 8 GiB 和其它工作区，就应重新按**同一时刻存活的对象**求和，不能只对阶段总量取 max。CPU backup 也需要主机内存；若换出 12 GiB、有效搬运速度假设 12 GiB/s，单方向理想时间就是 1 秒，来回至少约 2 秒，尚未计同步与分配。

因此 offload 将 GPU 容量压力换成主机内存和传输时间。`clear_memory()` 只能回收已无用/缓存的分配，不能让仍被活对象持有的参数凭空消失；`sleep()` 保留可恢复状态；`release()` 销毁进程后则需靠 checkpoint 重建，成本和恢复契约各不相同。

### 4.3 更激进：`--release-train`

每步训练后 `RayTrainGroup.release()`（actor_group.py:180-185）直接 `ray.kill` 全部训练 actor，GPU 完全交还；下步训练前 `create()` 重建并从 checkpoint 重载（`save_model` 里同步调整 `args.load` 等恢复参数，actor_group.py:150-159）。适合 rollout 极慢、训练极快的极端长尾 agentic 场景。

---

## 5. FP8 与量化链路

TIS（Truncated Importance Sampling，截断重要性采样）对目标策略与行为策略的概率比进行截断或限制，以控制大权重。它可降低权重波动，但截断可能引入偏差，不能自动修复动作或上下文错位。

- **FP8 cast**：`slime/backends/megatron_utils/sglang.py` 从 SGLang 引入 `quant_weight_ue8m0 / transform_scale_ue8m0`（DeepSeek 系 FP8 checkpoint 的 ue8m0 scale 格式）；`megatron_to_hf/processors/quantizer_fp8.py` 在 `convert_to_hf` 时把 BF16 权重 cast 成 FP8 再传输——权重主体从每元素 2 字节降到 1 字节；还要传 scale 和元数据，端到端耗时不保证减半，推理侧按支持的格式执行。
- **自研 Triton kernel**：`slime/backends/megatron_utils/kernels/fp8_kernel.py` 提供 blockwise FP8 cast。
- **int4/fp4（compressed-tensors）**：NCCL 传输前后各有一次 `post_process_weights`（update_weight_from_distributed.py:113-132），先在引擎侧反量化恢复、传完再重新量化。
- **数值一致性**：FP8 rollout 使推理 logprob 与 BF16 训练侧产生系统性偏差——`rollout_log_probs` 随样本回传后，训练侧用 TIS 修正（05 篇 §5），并用 `train_rollout_logprob_abs_diff` 指标持续监控偏差（loss.py:1073-1077）。`examples/train_infer_mismatch_helper/` 是配套的诊断工具。
- **权重同步对账**：`--check-weight-update-equal`（train.py:29-30）在启动时对训练侧与引擎侧权重做 snapshot + 逐张量 compare，把"权重没同步上"这类静默错误变成显式报错。

---

## 5.5 深入拆解：异构 PD 分离下 NCCL 组的 rank 分配

`connect_rollout_engines_from_distributed`（`update_weight/update_weight_from_distributed.py:268-310`）建组时，**训练侧只有 rank 0（`_is_pp_src_rank` 为真的那些 PP stage 的 DP=TP=0 rank）参与这个 NCCL 组**，其余训练 rank 完全不知道这个组的存在——权重广播是"训练侧一个代表 + 全部推理引擎"的星型拓扑，不是"全部训练 rank + 全部引擎"。

```python
if engine_gpu_counts is None:
    engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)  # 同构默认：每引擎 TP 大小一致
world_size = sum(engine_gpu_counts) + 1        # +1 = 训练侧那个代表
cumulative = [0]
for c in engine_gpu_counts:
    cumulative.append(cumulative[-1] + c)      # 前缀和，用于算每个引擎在组内的起始 rank
refs = [
    engine.init_weights_update_group.remote(
        rank_offset=cumulative[i] + 1,          # engine i 占据 [cumulative[i]+1, cumulative[i+1]] 这一段 rank
        world_size=world_size, group_name=group_name, backend="nccl",
    ) for i, engine in enumerate(rollout_engines)
]
model_update_groups = init_process_group(..., rank=0, world_size=world_size)  # 训练代表自己是 rank 0
```

**具体数字例子**：PD 分离场景，2 个 prefill 引擎（各 TP=2）+ 1 个 decode 引擎（TP=4），`rollout_engines = [prefill0, prefill1, decode0]`，`engine_gpu_counts = [2, 2, 4]`：

```
cumulative = [0, 2, 4, 8]
world_size = 2+2+4+1 = 9
训练代表 rank : 0
prefill0 占据 rank : [1, 2]     (cumulative[0]+1=1 起，共2个)
prefill1 占据 rank : [3, 4]     (cumulative[1]+1=3 起，共2个)
decode0  占据 rank : [5, 6,7,8] (cumulative[2]+1=5 起，共4个)
```

每个引擎内部再各自决定组内 rank 到"自己 TP rank"的映射（各引擎自己的 `init_weights_update_group` 实现，[SGLang 控制面](<../sglang/04_interfaces_and_models/03_control_plane_and_post_training.md>) 有对应服务端解读）。这段代码同时说明了**为什么改变引擎数量/TP 配置必须重建整个 NCCL 组**（`disconnect_rollout_engines_from_distributed` 先销毁旧组）——rank 分配完全依赖 `engine_gpu_counts` 这个静态列表，一旦引擎拓扑变化，之前分配的 rank 全部失效，容错重连（09 篇）本质上就是"销毁重建"这一套逻辑的重复调用。

## 5.6 一个完整的多并行维度广播例子

设训练侧 `TP=4, PP=2, EP=1, DP=2`（共 16 卡），rollout 侧 2 个同构引擎各 `TP=2`（共 4 卡）：

1. **PP 维度**：训练侧有 2 个 PP stage，每个 stage 只负责模型的一部分层，所以**每个 PP stage 都要单独建一个 NCCL 组**并广播自己那部分层——`self._group_name = f"slime-pp_{pp_rank}"`（代码 80 行），即整个流程要建 `slime-pp_0` 和 `slime-pp_1` 两个独立的组，分两次广播才能把完整模型送到引擎侧；
2. **TP 维度**：每个 PP stage 内 `_is_pp_src_rank` 要求 `DP=0 且 TP=0`，即每个 stage 只有 1 个训练 rank 作为代表——但代表只是"发起广播的那个进程"，它发出去的权重必须是**先在 TP 组内 all-gather 完整（把 4 份 TP 切片拼成完整 tensor）之后**的完整权重（04 篇正文 §2 提到的"AllGather 到 rank0"步骤），否则引擎侧收到的就是不完整的切片；
3. **DP 维度**：`DP=2` 意味着同一份权重训练时被复制了两份（两组 TP×PP 网格各自算梯度、优化器更新后理论上权重相同），选 `DP rank==0` 的那一份广播即可，另一份 DP 组完全不参与传输；
4. **引擎侧**：`world_size = 4(prefill/decode GPU) + 1 = 5`，两次广播（对应 2 个 PP stage）各自的组大小都是 5，只是 payload 不同（stage 0 广播 embedding+前半层，stage 1 广播后半层+lm_head）。

**此处描述的是按 PP stage 广播的专用 distributed 路径。** 每段模型由该 PP stage 的源 rank 汇集并发送；TP 分片先按参数布局重建，专家参数则需相应 EP 处理。不能把这一过程推广为所有更新器都必须聚齐全部专家：当前 tensor 更新器还包含按目标引擎路由专家权重的分支，见 `_prepare_expert_weight_batch`。通信组数、发送 rank 和临时显存峰值都应由所选路径的拓扑来推导。

---

## 6. 小结

> 本篇讲的是训练侧视角；引擎侧（SGLang）的接收实现（NCCL 组管理、读写锁、torch_memory_saver、IPC 还原）见 [SGLang 控制面与在线换权](<../sglang/04_interfaces_and_models/03_control_plane_and_post_training.md>)；slime 内建 HF 转换实现见 [13_megatron_bridge_internals.md](<13_megatron_bridge_internals.md>)。

- 权重同步 = 分片 gather（TP/EP）→ HF 转换 → 分桶 → NCCL/IPC/磁盘传输 → 引擎落盘或广播加载；pause/flush/continue 保证一致性，全局锁防 NCCL 死锁；
- 异构引擎（PD 分离）、新引擎热加入（容错）、版本对账都被一等支持；
- 显存错峰三档：rollout release（KV/权重）→ train sleep（torch_memory_saver）→ release-train（杀进程）；
- FP8 减少权重主体字节数，TIS 可缓解概率偏差，但截断和支持集合差异使其不能保证恢复严格 on-policy。
- 相关资料（05）进入训练侧内部：PPO/GRPO/GSPO 的 loss 是怎么算的。
