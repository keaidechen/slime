# 04 权重同步与显存管理：RL 系统的"头号瓶颈"是如何被攻克的

> 对应综述（`00_rl_infra_survey.md`）§2.3「权重同步」与 §2.9「精度」。
> 每步训练后，数百 GB 权重要从 Megatron 训练栈搬进 SGLang 推理引擎——训练侧是 TP/PP/EP 切分的 Megatron 格式，推理侧是 HF 命名、另一种 TP 切分。本篇解读 slime 的完整链路。

---

## 1. 四种传输模式总览

`slime/backends/megatron_utils/update_weight/` 目录：

| 文件 | 模式 | 适用场景 |
|---|---|---|
| `update_weight_from_distributed.py` | NCCL 广播 | 训练/推理分卡（disaggregate），跨机 |
| `update_weight_from_tensor.py` | CUDA IPC 直传 | colocate 同卡，零网络开销 |
| `update_weight_from_disk.py` | 共享文件系统全量 | 训推完全分离、异构 GPU（External Engines） |
| `update_weight_from_disk_delta.py` | 共享文件系统增量 | 同上，但只传变化分片（Delta Weight Sync） |

选择由 `--update-weight-mode`（full/delta）与 `--update-weight-transport`（nccl/ipc/disk 等）组合决定。训练侧统一入口是 `MegatronTrainRayActor.update_weights`（`slime/backends/megatron_utils/actor.py:567-628`）→ `self.weight_updater.update_weights()`，`weight_updater` 在 init 时按模式实例化。

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
2. **`convert_to_hf`**（`megatron_to_hf/`）：Megatron 命名/布局 → HF 命名/布局。例如 Megatron 的 `module.decoder.layers.0.self_attention.linear_qkv.weight`（QKV 融合）要拆成 HF 的 `q_proj/k_proj/v_proj`；每个模型族一个转换模块（`deepseekv3.py`、`qwen3moe.py`、`glm4moe.py`…），FP8 量化由 `processors/quantizer_fp8.py` 等处理（06 篇详解）；
3. **分桶（bucketing）**：累积到 `--update-weight-buffer-size` 就 yield 一桶。几百个小张量逐张广播会被 NCCL 启动开销拖死，合并成大桶摊薄开销——这与 SGLang 侧的 `FlattenedTensorBucket`（`slime/backends/megatron_utils/sglang.py:20-22` 引入）配合，把一桶张量 flatten 后一次传输、对端按元数据 reshape。

专家参数的 EP all-gather（`_ep_gather_and_convert`，204-238）更精巧：先用 `all_gather_object` 对齐各 EP rank 的参数名清单，再异步 `all_gather` 收齐所有 rank 的专家分片，最后源 rank 把"哪个 expert 的哪层"映射回 HF 的 `experts.{i}` 命名。

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

### 2.4 入口处的容错与重连

`actor.py:567-628` 的 `update_weights` 在调用 `weight_updater` 之前还做了：

- **故障恢复**（571-574）：`use_fault_tolerance` 时先 `recover_updatable_engines`——健康监控发现挂掉的引擎会被重建（新引擎 = `num_new_engines > 0`），需要 `connect_rollout_engines` 把它们加进 NCCL 组（597-607）；
- **offload_train 的配合**（592-628）：训练显存被 torch_memory_saver 挂起期间，进程组要先 `reload_process_groups()`、传完再 `destroy_process_groups()`；权重传输本身在 `torch_memory_saver.disable()` 上下文里执行（609 行），避免传输内存被挂起。

---

## 3. IPC 与 Disk 模式（机制对比）

- **CUDA IPC（`update_weight_from_tensor.py`）**：colocate 时训练与推理在同一物理机上，直接传递显存指针（`MultiprocessingSerializer` 序列化 CUDA IPC handle，配合 `FlattenedTensorBucket` 整桶共享），引擎端"按指针取用"，省掉 NCCL 拷贝。这是 slime 在 colocate 下 8×H100 同步 Qwen3-30B-A3B 约 7 秒的关键之一。
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

精妙点：`pull_weights`（把共享文件系统的新 checkpoint 拉到各机本地 NVMe）**在暂停生成之前**执行，与正在进行的推理重叠；只有最后的本地加载需要 pause。版本号机制（`_disk_weight_version`，actor_group.py:54/166-172）让引擎与训练侧能对账——CI 里会逐一校验 `get_weight_version`（actor_group.py:254-265）。Delta 模式则只写发生变化的权重分片，进一步缩减 IO（对应官方文档「Delta Weight Sync」）。

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

### 4.3 更激进：`--release-train`

每步训练后 `RayTrainGroup.release()`（actor_group.py:180-185）直接 `ray.kill` 全部训练 actor，GPU 完全交还；下步训练前 `create()` 重建并从 checkpoint 重载（`save_model` 里同步调整 `args.load` 等恢复参数，actor_group.py:150-159）。适合 rollout 极慢、训练极快的极端长尾 agentic 场景。

---

## 5. FP8 与量化链路

- **FP8 cast**：`slime/backends/megatron_utils/sglang.py:1-8` 从 SGLang 引入 `quant_weight_ue8m0 / transform_scale_ue8m0`（DeepSeek 系 FP8 checkpoint 的 ue8m0 scale 格式）；`update_weight/megatron_to_hf/processors/quantizer_fp8.py` 在 `convert_to_hf` 时把 BF16 权重 cast 成 FP8 再传输——**带宽减半**，推理侧直接跑 FP8。
- **自研 Triton kernel**：`slime/backends/megatron_utils/kernels/fp8_kernel.py` 提供 blockwise FP8 cast。
- **int4/fp4（compressed-tensors）**：NCCL 传输前后各有一次 `post_process_weights`（update_weight_from_distributed.py:113-132），先在引擎侧反量化恢复、传完再重新量化。
- **数值一致性**：FP8 rollout 使推理 logprob 与 BF16 训练侧产生系统性偏差——`rollout_log_probs` 随样本回传后，训练侧用 TIS 修正（05 篇 §5），并用 `train_rollout_logprob_abs_diff` 指标持续监控偏差（loss.py:1073-1077）。`examples/train_infer_mismatch_helper/` 是配套的诊断工具。
- **权重同步对账**：`--check-weight-update-equal`（train.py:29-30）在启动时对训练侧与引擎侧权重做 snapshot + 逐张量 compare，把"权重没同步上"这类静默错误变成显式报错。

---

## 6. 小结

> 本篇讲的是训练侧视角；引擎侧（SGLang）的接收实现（NCCL 组管理、读写锁、torch_memory_saver、IPC 还原）见 [10_engine_internals_sglang.md](10_engine_internals_sglang.md)；HF 转换的通用实现见 [11_megatron_and_bridge_internals.md](11_megatron_and_bridge_internals.md) 第二部分。

- 权重同步 = 分片 gather（TP/EP）→ HF 转换 → 分桶 → NCCL/IPC/磁盘传输 → 引擎落盘或广播加载；pause/flush/continue 保证一致性，全局锁防 NCCL 死锁；
- 异构引擎（PD 分离）、新引擎热加入（容错）、版本对账都被一等支持；
- 显存错峰三档：rollout release（KV/权重）→ train sleep（torch_memory_saver）→ release-train（杀进程）；
- FP8 传输降带宽，配合 TIS 守住 true on-policy 语义。
- 下一篇（05）进入训练侧内部：PPO/GRPO/GSPO 的 loss 是怎么算的。
