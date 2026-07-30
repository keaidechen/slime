# 10 TransferQueue：RL 系统的独立数据平面

> 对应综述（`00_rl_infra_survey.md`）§3.10 与趋势 #2。
> 本篇基于你补充到仓库根目录的第三方源码 `TransferQueue/`（Ascend 开源，论文 *AsyncFlow: An Asynchronous Streaming RL Framework*，arXiv 2507.01663）。它不是 slime 的一部分，而是"数据流中间件"这一新兴子领域的代表实现——值得单独精读。

---

## 1. 它解决什么问题

### 1.1 单控制器的数据瓶颈

verl 这类"单控制器 + worker 组"架构里，**所有数据都要经过 driver 进程**：rollout 产出 → driver 收集 → 转发给 ref/reward 算 logprob/score → driver 汇总 → 算 advantage → 分发给训练 worker。每一步数据在 driver 里序列化/反序列化/复制多遍，driver 成为吞吐天花板。AsyncFlow 论文与 verl 官方 PR（#5401，2025-12）的结论是：把数据面从控制面拆出去后，**端到端吞吐提升 49.1%**（论文数据：Qwen2.5-7B 128 卡同步 GRPO，r/s 1328.9 → 1980.7）。

### 1.2 数据依赖天然是"细粒度、异步、多对多"的

RL 一步里有多种角色（Actor / Rollout / Ref / RM / Advantage），它们对数据的需求不同：

- Ref 只需要 `prompt+response tokens`；RM 只需要 `response`；训练需要全部字段；
- rollout 样本**乱序完成**，而训练要攒批——"哪些样本已齐、被谁消费过"需要全局账本；
- 数据重分配（resharding）在 DP/PP 维度反复发生。

这些都不是"一次 RPC 传一个 batch"能优雅表达的。TransferQueue 的做法：**把数据放进共享存储，系统里流动的只有轻量元数据（BatchMeta）**，消费方按元数据直接去存储拉取。

---

## 2. 总体架构：控制面 / 数据面 / 客户端三层

```
                 ┌──────────────────────────────┐
   控制面(元数据) │ TransferQueueController (Ray actor) │
                 │  partition 账本:               │
                 │   生产状态矩阵 [样本 × 字段]     │
                 │   消费状态矩阵 [样本 × 任务]     │
                 └───────▲───────────────▲───────┘
              ZMQ 注册/查询│               │ZMQ 查询
   ┌──────────┴───┐   ┌───┴──────────┐   ┌┴─────────────┐
   │ Rollout 客户端 │   │ 训练/Ref 客户端 │   │ Sampler 等    │
   │  put(data)   │   │  get_data(meta)│   │              │
   └──────┬───────┘   └──────┬───────┘   └──────────────┘
          │ 直接读写          │ 直接拉取（不过 controller）
   ┌──────▼─────────────────▼────────────────────────────┐
   数据面  可插拔存储后端: SimpleStorageUnit / Mooncake / Yuanrong / RayRDT
   └─────────────────────────────────────────────────────┘
```

- **控制面**（`transfer_queue/controller.py`）：单个 Ray actor，只存元数据，不碰张量本体；与各组件经 ZMQ 通信（`utils/zmq_utils.py` 的 `ZMQRequestType`）；
- **数据面**（`transfer_queue/storage/`）：可插拔后端，张量按 (partition, sample, field) 存储；
- **客户端**（`transfer_queue/client.py`）：`put` 时先向 controller 注册"占位元数据"、写存储、再回执"生产完成"；`get_data` 时拿 BatchMeta 直接读存储组装。

---

## 3. 核心概念词汇表

| 概念 | 含义 |
|---|---|
| `partition_id` | 数据分区（如一次 rollout、一个 replay buffer），样本在其中按 `global_index` 编号 |
| `global_index` | 分区内样本的全局序号，由 controller 的 `PartitionIndexManager` 统一分配（controller.py:65-190） |
| `field` | 样本的一个字段（如 `input_ids`、`old_log_probs`、`reward`），**字段级**是状态跟踪的最小粒度 |
| `production_status` | `[样本数 × 字段数]` 的 0/1 矩阵：某样本的某字段是否已写入 |
| `consumption_status` | `[样本数 × 任务数]` 矩阵：某样本是否已被某任务（`task_name`）消费 |
| `task_name` | 消费方身份（如 `"actor_train"`），同一批数据可被多个任务独立消费、独立记账 |
| `BatchMeta` | 在系统里流动的元数据对象（详见 §4） |

**举例**：一轮 GRPO，rollout 把 1024 条样本的 `input_ids/responses/old_log_probs` 写入 partition `"rollout_5"`；RM 任务轮询 `production_status`，发现样本 0-100 的 `responses` 已就绪就先拉去打分、回写 `reward` 字段；训练任务只取"`input_ids+responses+reward` 三个字段全就绪"的样本。每个任务有自己的消费账本，互不阻塞。

---

## 4. 核心数据结构：`BatchMeta`

`transfer_queue/metadata.py:194-875`。它是**列式（columnar）**的：不是"每样本一个 dict"，而是"每列一个数组"——`global_indexes: list[int]`、`partition_ids: list[str]`、`production_status: np.ndarray[int8]`、`field_schema: dict[str, {...}]`、`custom_meta: list[dict]`。列式布局让切片/拼接是廉价的数组操作。

要点解读：

- **field_schema 而非逐样本元数据**（`extract_field_schema`，metadata.py:132-191）：每个字段记 `{dtype, shape, is_nested, is_non_tensor, per_sample_shapes}`。变长字段（不同样本 token 数不同）用 PyTorch nested tensor 表达，逐样本形状收进 `per_sample_shapes`；
- **丰富的代数操作**：`chunk`（均分）、`chunk_by_partition`、`concat`（拼接，校验字段/dtype 一致，metadata.py:696-819）、`union`（按样本合并字段，生产状态取**按位与**——两边都就绪才算就绪，metadata.py:664-667）、`select_samples/select_fields/reorder`。这套"元数据代数"让上层可以随便拆批/拼批而不用碰数据本体；
- **`_SampleView` 惰性行视图**（metadata.py:50-129）：需要按样本访问时给只读视图，避免物化整行 dict；
- **Ray 序列化细节**（metadata.py:292-310）：`__setstate__` 里专门处理 Ray Arrow 零拷贝反序列化产生的**只读 numpy 数组**（`.copy()` 成可写）——与 Ray 深度集成时会踩的典型坑；
- **`KVBatchMeta`**（metadata.py:879-1100）：KV 接口的元数据，样本以字符串 `keys` 寻址、带 `tags`，服务 UniRL 式的 KV 语义。

---

## 5. Controller：全局账本

`TransferQueueController`（controller.py:939-1600+）内部三个关键组件：

### 5.1 `PartitionIndexManager`（controller.py:65-190）

按 partition 分配/回收 `global_index`。支持"预分配 + 激活"（`register_pre_allocated_indexes` / `activate_pre_allocated_indexes`，controller.py:389-443）——异步流水线里生产方先占位、慢慢填，索引不乱。

### 5.2 `DataPartitionStatus`（controller.py:320-936）

一个 partition 的全部账本：

- **生产状态矩阵**：`production_status` 是 `[样本容量 × 字段数]` 的张量，写入即置 1（controller.py:541-543 的向量化赋值）；样本/字段容量按需倍增扩容（`ensure_samples_capacity/ensure_fields_capacity`）；
- **字段元数据** `FieldMeta`（controller.py:192-318）：每个字段记录"哪些 global_index 已拥有它"（集合）+ schema。`scan_data_status`（controller.py:709）据此快速算出"哪些样本的指定字段集全部就绪"；
- **消费状态**：按 `task_name` 分开记账（`mark_consumed` / `get_consumption_status` / `reset_consumption`，controller.py:587-666）；
- **快照** `to_snapshot`（controller.py:869）：配合 client 的 `save/load_controller_checkpoint`（client.py:1069-1146）做容错。

### 5.3 `generate_batch_meta`：三种模式（controller.py:1374-1458）

```1374:1429:TransferQueue/transfer_queue/controller.py
    def generate_batch_meta(
        self,
        partition_id: str,
        batch_global_indexes: list[int],
        data_fields: list[str],
        mode: str = "fetch",
    ) -> BatchMeta:
```

- `fetch`：消费方拉数据——直接信任账本就绪状态，`production_status` 全置 1；
- `insert`：生产方占位——为未注册字段补空 schema，状态全 0，等 `update_production_status` 回执；
- `force_fetch`：逐样本按真实矩阵状态填——用于"尽力而为"的拉取（部分字段没好就标 0）。

---

## 6. Client：读写链路

`AsyncTransferQueueClient`（client.py:49-1190）+ 同步包装 `TransferQueueClient`（client.py:1192-1744，后台线程跑事件循环，把 async 方法绑定成同步）。

**put（生产）三步**（`async_put`，client.py:259）：

1. 向 controller 申请/注册元数据（insert 模式的 `generate_batch_meta`）；
2. 把 TensorDict 按字段写入存储后端（可指定字段写到不同存储单元）；
3. 回执 `update_production_status` 置 1——**此后消费方才看得见这批数据**（"先占位后回执"避免了"读到写了一半的数据"）。

**get_data（消费）**（client.py:375-416）：拿 BatchMeta → 按字段路由到对应存储单元 → 并发拉取 → 组装回 TensorDict。数据**不经过 controller**，这是性能的关键。

其余接口：生产/消费状态查询与检查（`check_consumption_status` 等）、按样本/分区清理、`kv_retrieve_meta/keys/list`（KV 语义）、双侧 checkpoint。

---

## 7. 消费策略抽象：Sampler 与 StreamingDataLoader

controller 只管"哪些就绪、哪些被消费过"，**"这一批取哪几个"是可插拔的策略**（`sampler/base.py:20-41` 的设计说明写得很清楚：生产状态归 controller、消费策略归 sampler）：

```20:41:TransferQueue/transfer_queue/sampler/base.py
class BaseSampler(ABC):
    """...
    Available Samplers:
    - **SequentialSampler**: Default sampler, selects samples sequentially without replacement
    - **GRPOGroupNSampler**: A sampler that performs sampling on continuous N samples only when all of them are ready.
    - **RankAwareSampler**: Rank-aware sampling for distributed training where each rank retrieves data independently.
                            This sampler will guarantee ranks of the same DP group consume identical samples.
    ...
    """
```

- **`SequentialSampler`**：顺序取，无放回；
- **`GRPOGroupNSampler`**：GRPO 专用——同一 prompt 的 N 个采样**整组就绪才取**（对应 slime 里 group 级收集 + dynamic filter 的角色）；
- **`RankAwareSampler`**：分布式训练每个 rank 独立拉取，但保证同 DP 组各 rank 拿到**相同样本**；
- **`SeqLenBalancedSampler`**：按序列长度均衡组批（token 负载均衡）。

上层封装：`StreamingDataset`（`dataloader/streaming_dataset.py:37`）是 torch `IterableDataset`，`__iter__` 里循环"查就绪 → sampler 选样 → 拉数据 → 标记消费"；`StreamingDataLoader`（streaming_dataloader.py:37）再包一层标准 DataLoader 接口（`reset/step/get_buffer`）。训练侧因此可以像写普通 PyTorch 训练一样消费异步产生的 RL 数据。

### 7.1 深入拆解：`GRPOGroupNSampler` 怎么判断"整组就绪"（`sampler/grpo_group_n_sampler.py`）

它的假设很朴素但很关键：**同一个 prompt 的 N 个采样在写入时必须占用连续的 `global_index`**（例如 prompt 0 的 4 个样本占 `[0,1,2,3]`，prompt 1 占 `[4,5,6,7]`）——这样"判断一组是否完整"就退化成"判断一段连续 N 个整数是否都在 `ready_indexes` 里"，不需要额外维护"哪个样本属于哪个 prompt"的映射表。

```python
sorted_ready_indexes = sorted(ready_indexes)
i = 0
while i <= len(sorted_ready_indexes) - n and found_groups < required_groups:
    potential_group = sorted_ready_indexes[i : i + n]
    is_consecutive = all(potential_group[j+1] - potential_group[j] == 1 for j in range(n-1))
    if is_consecutive:
        complete_group_indices.extend(potential_group)
        found_groups += 1
        i += n              # 命中一组，跳过整组继续扫
    else:
        i += 1              # 没命中，只前进一格（因为落单的样本可能是下一组的起点）
```

**一个数字例子**：`n_samples_per_prompt=3`，某一时刻 `ready_indexes = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]`（表示 prompt 0 只完成了 2/3 个采样、prompt 1（3,4,5）和 prompt 2（6,7,？）..实际上按 3 个一组来看：`[0,1]` 不成组，`[3,4,5]` 连续成组，`[6,7]` 不连续于后续的 9 所以不成组，`[9,10,11]` 连续成组）。扫描过程：`i=0` 看 `[0,1,3]`，`1→3` 差 2 不连续，`i+=1`；`i=1` 看 `[1,3,4]`，不连续，`i+=1`；`i=2` 看 `[3,4,5]`，连续！收进结果，`found_groups=1`，`i+=3=5`；`i=5` 看 `[6,7,9]`，`7→9` 差 2 不连续，`i+=1`；`i=6` 看 `[7,9,10]`，不连续，`i+=1`；`i=7` 看 `[9,10,11]`，连续！`found_groups=2`，命中 `required_groups`（若 `batch_size=6`）。最终 `sampled_indexes=[3,4,5,9,10,11]`——**恰好跳过了不完整的 prompt 0 和"看似有希望但实际不连续"的 6/7**，这与代码自带的 docstring 示例完全一致。

**`_states` 缓存的意义**（141-191 行）：以 `(partition_id, task_name, dp_rank, batch_index)` 为 key 缓存采样结果——**同一个 `(dp_rank, batch_index)` 组合永远返回相同的样本集合**，即使底层 `ready_indexes` 后续发生了变化（比如更多样本就绪了）。这解决了分布式训练里的一个隐蔽问题：如果不缓存，某个 rank 因为网络延迟晚一点调用 `sample()`，看到的 `ready_indexes` 集合与其它 rank 不同，可能采出不同的样本组合，导致同一 DP 组的不同 rank 训练不同的数据（破坏 DP 的"同步梯度平均"假设）——缓存把"这一批具体是哪些样本"在第一次决定后就冻结下来，后续任何 rank/重试再问同一个 `batch_index` 都拿到完全一致的答案。

### 7.2 一个完整数字例子：8 个 prompt × 4 采样的 GRPO 端到端数据流

设 `n_samples_per_prompt=4`，一轮 rollout 产出 8 个 prompt（`global_index` 0-31，prompt k 占 `[4k, 4k+3]`）：

1. **写入**：rollout worker 陆续调用 `client.async_put`，每完成一条样本的 `input_ids/response/reward` 就写一次——由于生成是并发的，完成顺序可能是 `prompt3-sample1, prompt0-sample2, prompt3-sample0, ...` 完全乱序，但每次 `async_put` 只影响自己那一行的 `production_status`，不需要等其它样本；
2. **controller 账本**：某一时刻 `production_status` 矩阵里，index `[12,13,14,15]`（prompt 3）四行的 `reward` 列全为 1，而 index `[0,1,2,3]`（prompt 0）还差一个——`scan_data_status` 能立刻算出"当前哪些样本的 `{input_ids, response, reward}` 三字段全齐"，返回 `ready_indexes` 大约是 `[4,5,...,31]`（减去 prompt 0 缺的那个）；
3. **`GRPOGroupNSampler.sample(ready_indexes, batch_size=16, ...)`**：`required_groups=4`，扫描连续段，跳过不完整的 prompt 0，从 prompt 1 开始收集直到凑够 4 组（16 条），返回 `sampled_indexes`（例如 prompt 1/2/3/4 各 4 条）；
4. **`get_data`**：训练客户端拿这 16 个 index 对应的 `BatchMeta`，直连存储后端并发拉取张量，本地组装成 `TensorDict`（不经过 controller，避免这一步成为瓶颈）；
5. **训练侧消费**：GRPO advantage 计算天然要求"同一 prompt 的样本在同一批里"（组内做归一化），这正是 `GRPOGroupNSampler` 保证"要么整组都在，要么整组都不在"的原因——如果允许部分组样本混进批次，advantage 的组统计量就会算错；
6. **prompt 0 怎么办**：它会留在 `ready_indexes` 之外，等第 4 个样本的 reward 写完之后，下一次 `sample()` 调用会把它纳入下一批（或本批次的"补齐"逻辑，取决于调用方的重试策略）——这与 slime 里 partial rollout"半成品留在 buffer 里等下一轮"是同一个思想的不同实现（03 篇 §3）。

---

## 8. 可插拔存储后端

`storage/` 下：

| 组件 | 说明 |
|---|---|
| `simple_storage.py` | 默认后端：进程内 dict 按 (partition, sample, field) 存储，单测/小规模用 |
| `managers/` | 客户端侧存储管理器抽象（按字段路由到不同存储单元） |
| `clients/` | 各后端客户端：**Yuanrong**（华为元戎分布式数据系统）、**Mooncake**（月之暗面开源的 KVCache 中心化存储，做参数服务器式张量分发）、**RayRDT**（Ray 生态 RDMA 张量传输） |
| `bootstrap/` | 各后端的启动脚本（storage unit 以独立进程/Ray actor 拉起） |

后端可以混用——例如热数据（当前 step 的 rollout）放 Mooncake、冷数据（replay buffer）放 Yuanrong。`BatchMeta._custom_backend_meta` 字段就是留给后端记录私有定位信息的（如 Mooncake 的对象句柄）。

---

## 9. 生态集成现状

| 框架 | 集成方式 |
|---|---|
| **verl** | 官方 PR #5401（2025-12）：`DataLoader`→TransferQueue 后端，e2e 吞吐 +49.1% |
| **ROLL** | 用作 `RemoteBatch` 的分布式存储，替代 Ray object store |
| **UniRL** | 经 `interface.py` 的 KV 接口（`KVStorage`，按 key/tag 存取）做异步数据协调 |
| **Relax** | 宣称通过 TransferQueue 把 Actor/Rollout/Ref/Advantage 解耦到独立集群 |

README 中还给出了 roadmap 信号：RAY 原生 RDT 支持（已完成）、RPC 后端泛化（已完成）、sampling-as-scheduling、cross-engine KV 同步（slime/vLLM）、gRPC/UCX 传输、分层 replay buffer、可插拔一致性协议（lease/fencing）、支持 slime。

---

## 10. 与 slime 数据流的对照

slime 没有采用 TransferQueue，两者代表数据流设计的两个点。**提醒**：本篇属于"对比参考"性质——TransferQueue 不是 slime 依赖的组件，读者可以按需查阅，不属于理解 slime 自身架构的必读路径（真正必读的是 01-09 与 11-13）。

| 维度 | slime | TransferQueue 系（verl/ROLL…） |
|---|---|---|
| 数据通路 | RolloutManager 收集 → Ray object store → 训练 rank 拉取（01 篇 §5） | 共享存储 + 元数据账本，消费方直连数据面 |
| 就绪/消费跟踪 | 进程内 buffer + `Sample.status` 状态机（03 篇） | controller 的样本×字段 / 样本×任务矩阵 |
| 乱序/异步 | over-sampling + abort + fully async worker | sampler 按就绪集选批，天然乱序 |
| 组语义（GRPO） | `generate_and_rm_group` 整组收集 | `GRPOGroupNSampler` 整组就绪才取 |
| 张量直传 | `rollout_data_transport="nixl"`（`slime/ray/placement_group.py:235`）、`enable_tensor_transport`（`slime/ray/actor_group.py:110`）——Ray 的 NIXL GPU 张量直传 | Mooncake/RayRDT 等后端 |
| 适用哲学 | 数据流是训练循环的内部实现（单一生命周期） | 数据流是独立基础设施（多生产/消费方、跨集群、长生命周期 replay） |

**什么时候需要 TransferQueue 这样的独立数据平面**：当数据的生产/消费方超过"一个训练循环"的范畴——多集群拆分（Relax）、外部系统持续写入（线上回流）、长生命周期 replay buffer、多个训练任务共享同一数据池。slime 的 `slime_plugins/rollout_buffer`（03 篇 §4.3）是朝这个方向迈出的轻量一步，而 TransferQueue 是这条路线的完整形态。注意其 README 的 roadmap 里已列出"support slime"。

---

## 11. 小结

- TransferQueue = **控制面（Ray actor 元数据账本）+ 数据面（可插拔张量存储）+ 客户端（先注册后回执的 put / 按元数据直拉的 get）**；
- 核心抽象：`BatchMeta` 列式元数据 + 样本×字段生产矩阵 + 样本×任务消费矩阵 + 可插拔 sampler；
- 价值：把单控制器的数据复制/转发瓶颈从关键路径上移除（verl +49.1%），并让异步、乱序、多方共享的 RL 数据流有了统一账本；
- 学习建议：先读 `metadata.py`（BatchMeta 代数）→ `controller.py`（账本）→ `client.py`（读写协议）→ `sampler/` + `dataloader/`（消费侧）→ `storage/simple_storage.py`（最小后端），再对照 AsyncFlow 论文理解异步流水全景。
