# 01 顶层架构与 Ray 编排：一次 RL 训练是如何被组织起来的

> 对应综述（`00_rl_infra_survey.md`）§2.1「编排与资源管理」。
> 本篇面向零基础读者：先讲清楚"一个 RL 训练任务里有哪些角色"，再逐行走进代码。

---

## 1. 背景知识：这个系统里有哪些"人"

一个 slime 任务启动后，集群里存在三类长期存活的 Ray actor（可以把 Ray actor 理解为"一个有状态、可被远程调用的进程"）：

| 角色 | 代码位置 | 占用资源 | 职责 |
|---|---|---|---|
| `RolloutManager` | `slime/ray/rollout.py:427` | 1 CPU，**0 GPU** | rollout 侧"总控"：取数据、调用生成函数、把生成结果整理成训练 batch、管理推理引擎生命周期 |
| `SGLangEngine` | `slime/backends/sglang_utils/sglang_engine.py` | 共享 GPU | 每个 actor 包装一个 SGLang HTTP server 子进程，暴露热更新权重、释放显存等 RL 专用端点 |
| `MegatronTrainRayActor` | `slime/backends/megatron_utils/actor.py:51` | 每 rank 1 GPU（colocate 时共享） | 一个 Megatron 训练进程（一个 rank） |

外加一个"短命"的角色：**driver 进程**（你执行的 `python train.py ...` 那个进程），它负责搭好以上三者，然后跑主循环。

调用关系（谁调用谁）：

```
driver (train.py)
   │  ray remote call
   ├──────────────► RolloutManager.generate(rollout_id)      # 要一批训练数据
   │                      │  HTTP
   │                      └────► sglang_router ──► SGLang server (generate)
   │
   ├──────────────► RayTrainGroup.async_train(rollout_id, data_ref)
   │                      └────► 每个 MegatronTrainRayActor.train(...)
   │
   └──────────────► RayTrainGroup.update_weights()
                          └────► 训练 rank 经 NCCL/IPC/磁盘 把权重推给 SGLangEngine
```

注意：**driver 不做任何计算**，它只发号施令并 `ray.get` 等待结果。这与 verl 的"单控制器"神似，但 slime 的控制器逻辑极简（不到 100 行），重逻辑都在各 actor 内部。

---

## 2. 入口 `train.py` 全链路解读

整个同步训练入口只有约 100 行，是理解全系统的最佳起点：

```9:27:train.py
def train(args):
    configure_logger()
    release_train = args.release_train

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()
```

启动阶段做 5 件事：

1. **`create_placement_groups(args)`**（train.py:14）：向 Ray 申请 GPU 资源池（详见 §3）。
2. **`create_rollout_manager(...)`**（train.py:19）：创建 `RolloutManager` actor。它内部会启动全部 SGLang server 和 router。**先建它的原因**：如果用户没指定 `--num-rollout`，需要从数据源算出"一个 epoch 有多少步"再乘 `num_epoch`（`slime/ray/placement_group.py:240-244`）。
3. **`create_training_models(...)`**（train.py:21）：创建 actor（以及 PPO 时的 critic）训练 actor 组。
4. **`actor_model.update_weights()`**（train.py:27）：训练侧加载完 checkpoint 后，先做一次全量权重同步，保证推理引擎与训练侧一致再开始产数据。
5. 可选的 `check_weights(action="compare")`（train.py:29-30）：**正确性校验**——把推理引擎里的权重和训练侧逐一比较，用于 CI 和调试（这是"RL bug 不报错只降智"哲学的体现，见 08 篇）。

### 2.1 主循环：一步 RL 的 9 个动作

```49:91:train.py
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        ...
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())
        ...
        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
        ...
        actor_model.save_model(rollout_id, force_sync=force_sync)
        ...
        actor_model.update_weights()
        ...
```

每个 `rollout_id`（可以通俗地理解为"第几步 RL"）按序执行：

| 顺序 | 动作 | 代码 | 说明 |
|---|---|---|---|
| 1 | 生成数据 | `rollout_manager.generate.remote(rollout_id)` | 异步提交一整批 prompt，收回 `rollout_batch_size` 组样本 |
| 2 | （可选）rollout 让显存 | `rollout_manager.offload.remote()` | colocate 模式下释放推理显存给训练用 |
| 3 | 训练一步 | `actor_model.async_train(...)` | 数据经 Ray object store 传给所有训练 rank，执行 PPO/GRPO 更新 |
| 4 | （周期性）存 checkpoint | `save_model(...)` | 含 `should_run_periodic_action` 判断 |
| 5 | 训练侧让显存 | `offload_train(...)` 闭包 | 非 offload 模式就 `clear_memory()` |
| 6 | 权重同步 | `actor_model.update_weights()` | 新权重推回推理引擎 |
| 7 | （可选）恢复 KV | `onload_kv.remote()` | 推理引擎重新占位 KV cache 显存 |
| 8 | （周期性）评测 | `rollout_manager.eval.remote(...)` | eval 数据集上生成+打分 |

**举例**：假设 `rollout_batch_size=128`、`n_samples_per_prompt=8`，那么第 1 步会采集 128 个 prompt × 8 = 1024 条回答（GRPO 的 128 个"group"）；第 3 步训练把这 1024 条按 DP 切分喂给 Megatron 做若干 micro-batch 更新；第 6 步把更新后的几百 GB 权重推给所有 SGLang server。

**PPO（有 critic）时的细节**（train.py:61-67）：先让 critic 训 value model，拿到 `value_refs`（critic 对这批样本的估值），再以 `external_data=value_refs` 传给 actor 训练——actor 算 GAE 需要 critic 的 values。前 `num_critic_only_steps` 步只训 critic（预热 value model）。

---

## 3. GPU 资源划分：placement group

### 3.1 什么是 placement group（PG）

Ray 的 PG 把一组资源"bundle"打包原子分配。slime 为**全部** GPU 创建**一个** PG（策略 `PACK`，尽量紧凑放置），每个 bundle 是 `{"GPU": 1, "CPU": 1}`：

```47:48:slime/ray/placement_group.py
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
```

**为什么训练侧和 rollout 侧共用同一个 PG，而不是各建一个？** 因为 colocate 模式下同一批卡要先跑推理、再跑训练，必须是同一组物理资源。用一个 PG 再把 bundle 列表切两段，就同时支持共置与分离（见 §3.3）。

### 3.2 逻辑序号 → 物理 GPU 的重排序

PG 的 bundle 序号与"哪台机的哪张卡"的对应关系是 Ray 调度决定的，不稳定。slime 用一个技巧探测真实拓扑：

```70:88:slime/ray/placement_group.py
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
```

即：往每个 bundle 里临时塞一个占 1 GPU 的 `InfoActor`（placement_group.py:15-18），它报告自己所在节点 IP 和 `ray.get_gpu_ids()`，随后立刻 `ray.kill`。然后按 `(IP, gpu_id)` 排序得到 `reordered_bundle_indices`——**这样训练 rank 0 永远落在 IP 最小的节点的 GPU 0 上**，多机 NCCL 拓扑变得确定。之后所有 actor 都按 `placement_group_bundle_index=reordered_bundle_indices[rank]` 钉在指定 bundle 上（actor_group.py:121-124）。

### 3.3 四种资源布局

`_get_placement_group_layout`（placement_group.py:100-117）返回 `(总GPU数, rollout偏移)`：

```100:117:slime/ray/placement_group.py
def _get_placement_group_layout(args) -> tuple[int, int]:
    actor_num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    if args.debug_train_only:
        return actor_num_gpus, 0
    if args.rollout_external:
        ...
    if args.debug_rollout_only:
        return args.rollout_num_gpus, 0
    if args.colocate:
        return max(actor_num_gpus, args.rollout_num_gpus), 0
    return actor_num_gpus + args.rollout_num_gpus, actor_num_gpus
```

| 模式 | 布局 | 含义 |
|---|---|---|
| 默认（分离） | `actor + rollout`，offset=actor 数 | PG 前段给训练、后段给推理，各占各的卡 |
| `--colocate` | `max(actor, rollout)`，offset=0 | 训练与推理**同一段 bundle**（同一批卡），靠显存错峰复用 |
| `--rollout-external` | 只要 actor 的卡 | 推理引擎在任务外部管理（External Rollout Engines），只留训练资源 |
| 调试 | `debug_train_only` / `debug_rollout_only` | 只起一半系统，分离调试（见 08 篇） |

**举例**：`actor_num_nodes=2`、`actor_num_gpus_per_node=8`、`rollout_num_gpus=8`、非 colocate → 创建 24 个 bundle；`"actor"` 拿前 16 个，`"rollout"` 拿后 8 个（placement_group.py:127-132）。critic 与 actor 共用同一段（`result["critic"] = result["actor"]`，placement_group.py:135）——因为 critic 与 actor 从不同时占满显存。

---

## 4. 训练 actor 组：`RayTrainGroup`

### 4.1 为什么 `num_gpus_per_actor=0.4`？

```150:160:slime/ray/placement_group.py
    return RayTrainGroup(
        args=args,
        ...
        num_gpus_per_actor=0.4,
        ...
    )
```

这是 Ray 的**分数 GPU**机制：声明每个训练 actor 只用 0.4 张卡，于是 colocate 时同一张物理卡上可以再放一个 SGLangEngine（也声明小于 1 的份额），Ray 才不会认为"这张卡已被占满"。它不改变 CUDA 可见性——actor 实际看到的是整张卡，只是调度层面的"拼房许可"。

### 4.2 actor 创建与初始化

`_allocate_gpus_for_actor`（actor_group.py:57-128）做三件值得注意的事：

1. **环境变量注入**（actor_group.py:64-97）：
   - `NCCL_CUMEM_ENABLE=0`：注释说明"因为 sglang 总是把它设为 0，我们也得设 0 防止 nccl error"——训推共进程组时 NCCL 显存分配方式必须一致；
   - `--offload-train` 时配置 `LD_PRELOAD=torch_memory_saver_hook_mode_preload*.so` 与 `TMS_INIT_ENABLE=1`：把 **torch_memory_saver** 的动态库预加载进训练进程，它 hook 了 CUDA 显存分配，可以把训练显存整体"挂起"到 CPU（详见 04 篇）；
   - `ENABLE_ROUTING_REPLAY=1`：MoE 路由重放（用 rollout 时记录的专家路由，保证训推一致，见 05/06 篇）。
2. **动态加载 actor 类**：默认 `MegatronTrainRayActor`，也可传 `actor_cls` 自定义（actor_group.py:99-104）。
3. **逐 rank 创建并确定 master**：rank 0 先建，取回 `master_addr/master_port` 再传给其他 rank（actor_group.py:117-128）——这是手写分布式初始化的常见模式，Megatron 的 `torch.distributed` 初始化需要一个 rendezvous 地址。

`create()`（actor_group.py:187-207）随后对每个 actor 调 `actor.init.remote(args, role, with_ref=..., with_opd_teacher=...)`，返回各 rank 的 `start_rollout_id`（从 checkpoint 恢复时决定从哪一步继续），并要求全组一致（placement_group.py:216-219）。

### 4.3 组级操作都是"广播 + 等待"

`RayTrainGroup` 的方法模式高度统一，例如：

```161:164:slime/ray/actor_group.py
    def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        if not self._full_disk_weight_update_enabled():
            return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])
```

即"对所有 actor 发同一个远程调用，`ray.get` 全部等齐"。`async_train` 是唯一例外：它**不等待**，直接返回 ref 列表（actor_group.py:130-148），这样 critic 可以先异步开训、driver 再把 value 结果喂给 actor（对应 train.py:62-67 的流水）。

---

## 5. 同步 vs 异步：`train.py` 与 `train_async.py`

`train_async.py` 实现的是 **one-step-async**（比同步版重叠"下一步 rollout"与"当前步训练"）：

```32:40:train_async.py
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
```

关键技巧：`generate.remote()` 立即返回 future，不等结果。于是"第 N+1 步数据生成"与"第 N 步训练"并行。两个约束：

- **不能用 colocate**（train_async.py:11 直接 assert）——同一张卡不能既训练又推理；
- **权重更新前必须等生成收尾**：

```66:70:train_async.py
        if release_train or (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()
```

注意这里引入了**一步 off-policy**：第 N+1 步的数据是用第 N 步训练前的旧权重生成的。`update_weights_interval > 1` 时 off-policy 程度更大。这是用"轻微的策略滞后"换"rollout 与训练时间重叠"——综述 §2.5 讨论的就是这类取舍。更激进的全异步见 03 篇（fully async）。

---

## 6. 参数体系速览

slime 的参数分三类（README「参数说明」）：

1. **Megatron 参数**：原样透传（如 `--tensor-model-parallel-size`），解析在 `slime/backends/megatron_utils/arguments.py`；
2. **SGLang 参数**：加 `--sglang-` 前缀（如 `--sglang-mem-fraction-static`），在启动 server 时剥掉前缀传给 SGLang；
3. **slime 自身参数**：`slime/utils/arguments.py`，本篇涉及的编排类开关：

| 参数 | 作用 |
|---|---|
| `--colocate` | 训推共置（§3.3） |
| `--actor-num-nodes/--actor-num-gpus-per-node` | 训练资源规模 |
| `--rollout-num-gpus` 等 | 推理资源规模 |
| `--offload-rollout/--offload-train` | 显存错峰（04 篇） |
| `--release-train` | 更激进的模式：每步训练后**销毁**训练 actor、下次训练前重建（actor_group.py:180-185 的 `release()` 会 `ray.kill` 全部训练进程），把 GPU 完全还给 rollout；配合 disk 权重更新使用 |
| `--update-weights-interval` | 异步模式下每几步同步一次权重 |
| `--rollout-external` | 推理引擎外部托管 |
| `--num-rollout` / `--num-epoch` | 总步数（缺省由数据换算） |

---

## 7. 小结与阅读路线

- driver（`train.py`）搭台：PG 分资源 → RolloutManager 起推理 → RayTrainGroup 起训练 → 首轮权重同步 → 主循环 9 动作。
- 三类 actor 通过 Ray handle 互相直连，无中心控制器；数据经 object store 流动，权重经 NCCL/IPC/磁盘流动。
- 下一篇（02）钻进 `RolloutManager` 内部，看 server 模式的 rollout 是如何生成数据的。
