# 13 训练侧内部实现：slime 内建 HF↔Megatron 权重转换

> 衔接 [04_weight_sync_and_memory.md](04_weight_sync_and_memory.md)、[06_megatron_backend_and_mbridge.md](06_megatron_backend_and_mbridge.md) 与 [12_megatron_lm_internals.md](12_megatron_lm_internals.md)。
>
> **历史说明**：文件名为兼容旧链接而保留。slime v0.3.1 已删除 `Megatron-Bridge/`、`AutoBridge`、`--megatron-to-hf-mode bridge` 与 bridge iterator。本篇只解释当前仓库内的 `hf_to_megatron/`、`megatron_to_hf/` 和 `HfWeightIteratorDirect`。

---

## 1. 先回答三个最容易混淆的问题

### 1.1 模型结构和 checkpoint 转换是一回事吗？

不是。`model_provider.py` 先用 Megatron 参数、`--spec` 或 `--custom-model-provider-path` 构造出每个 rank 应持有的模型结构；随后 checkpoint loader 才把 HF 或 Megatron 权重填进去。

因此支持一个新模型至少有两个独立任务：

1. **结构支持**：Megatron 能构造正确的 layer、attention、MoE/MTP/MLA 结构；
2. **权重支持**：slime 知道 HF 名称/布局与该 Megatron 结构之间如何互转。

只增加 `_LOADERS` 不会让未知结构自动可训；只增加 custom provider 也不会让 HF safetensors 自动对齐。

### 1.2 为什么既有 `hf_to_megatron/`，又有 `megatron_to_hf/`？

两条路径的使用频率和数据所有权不同：

| 方向 | 主要时机 | 输入形态 | 输出形态 |
|---|---|---|---|
| HF → Megatron | 启动时从 HF checkpoint 初始化 | CPU safetensors，HF 名称，完整逻辑 tensor | 当前 rank 的 Megatron parameter/buffer shard |
| Megatron → HF | 每轮权重同步、磁盘发布、HF 导出 | PP/TP/EP 分散的 Megatron 参数或 actor CPU backup | 分桶的 `(hf_name, tensor)`，随后走 IPC/NCCL/磁盘 |

前者“目标 parameter 已存在，读一个 HF tensor 后切给本 rank”；后者“来源散在多个 rank，必须先找齐并 gather，再拆成 HF tensor”。它们共享布局知识，但不是同一个控制流的正反播放。

### 1.3 为什么不继续依赖通用 Bridge？

从当前代码能得到的直接结论是：slime 选择了**显式模型族映射 + 单一 direct iterator**。收益是依赖面和热路径更小，失败位置清楚，也容易把分桶、量化、SGLang 命名兼容直接放进 RL 高频同步链路；代价是每个新模型族都必须在双向转换目录中显式适配并测试，不能承诺任意 HF 架构自动支持。

---

## 2. HF → Megatron：启动加载链路

入口在 `slime/backends/megatron_utils/checkpoint.py::load_checkpoint`：

```text
load path
  ├─ 有 latest_checkpointed_iteration.txt 或目录名是 iter_XXXXXXX
  │    └─ Megatron 原生 load_checkpoint（含可用的训练状态）
  └─ 其他非空目录
       └─ _load_checkpoint_hf
            └─ hf_to_megatron.load_hf_weights
```

HF 目录只有模型权重，没有 Megatron optimizer/RNG/iteration 语义。加载完成后 optimizer 若存在会 `reload_model_params()`，函数返回 iteration 0；参数校验也会设置 `no_load_optim=True`、`no_load_rng=True`、`finetune=True`。这就是为什么“能从 HF 起训”不等于“能从 HF 无损断点续训”。

### 2.1 注册表只按 `config.model_type` 分发

`hf_to_megatron/__init__.py` 的 `_LOADERS` 是显式字典：

```python
_LOADERS = {
    "deepseek_v3": deepseek_hf_tensor,
    "glm4": glm4_hf_tensor,
    "qwen3": qwen_hf_tensor,
    "qwen3_moe": qwen_moe_hf_tensor,
    "qwen3_5": qwen3_5_hf_tensor,
    # ...
}
```

`supports_hf_weight_loading` 和 `load_hf_weights` 都使用 `AutoConfig.model_type` 查表。没有命中就抛异常，不会按名字相似度选择一个“可能能用”的 loader。这个 fail-fast 很重要：错误权重映射往往形状仍能对上，却会把语义不同的块静默装错。

### 2.2 `SafetensorReader` 的 `maxsize=1` 到底缓存什么？

reader 先读 `model.safetensors.index.json` 建立 `tensor name -> filename` 映射；无 index 时扫描所有 safetensors 的 key。这里很容易把两层缓存混淆：

- `self._files[filename]` 保存每个已经访问过的 `safe_open` 对象，同一 shard 后续读取会复用它；
- `get_tensor` 上的 `lru_cache(maxsize=1)` 缓存的是最近一次以 `name` 为 key 的**返回 tensor**，不是最近一个文件 handle；
- 因此切换 shard 不会从 `self._files` 移除旧 handle，当前实现并没有把打开的 shard 数量限制为 1；
- `safe_open` 使用 mmap/lazy access，避免 eager 地把所有 shard 内容装进 RAM。内存友好来自按 tensor 延迟读取，而不是“一次只开一个文件”。

若 1-byte FP8 tensor 同时存在 `<name>_scale_inv`，reader 按 scale 的二维 shape 推出 128×128 block，先 pad、反量化成 BF16，再裁回原形状。

### 2.3 从完整 HF tensor 到本 rank shard

`load_model_hf_weights` 遍历当前 Megatron model 的 `named_params_and_buffers`。对每个目标参数：

1. 模型族函数按 Megatron 参数名取出/合并 HF tensor；
2. `_pad_vocab` 把 embedding/output layer 补到 `padded_vocab_size`；
3. `shard_mcore_tensor` 读取目标 parameter 上的 `tensor_model_parallel`、`parallel_mode`、`partition_dim`、`partition_stride`；
4. 普通参数按 TP rank 切，expert 参数按 expert-TP rank 切；
5. 形状必须与本 rank parameter 完全相等，随后转 dtype/device 并 `copy_`。

这里**不需要 PP scatter**：函数只遍历本 PP stage 实际构造出来的参数，因此每个 stage 自然只请求自己那部分层。

---

## 3. 一个可验证的 Qwen GQA/QKV 例子

设 `num_attention_heads=32`、`num_key_value_heads=8`、`head_dim=128`、`hidden_size=4096`。HF 有：

```text
q_proj.weight: [4096, 4096]  # 32 * 128
k_proj.weight: [1024, 4096]  #  8 * 128
v_proj.weight: [1024, 4096]
```

Megatron 的 `linear_qkv.weight` 不是简单的 `[all Q][all K][all V]`。`merge_qkv` 先按 8 个 query group reshape：

```text
每组 = [4 个 Q head, 1 个 K head, 1 个 V head]
     = [q0 q1 q2 q3 k0 v0]
下一组 [q4 q5 q6 q7 k1 v1]
...
```

具体代码等价于：

```python
q = q.reshape(num_groups, heads_per_group * head_dim, hidden)
k = k.reshape(num_groups, head_dim, hidden)
v = v.reshape(num_groups, head_dim, hidden)
linear_qkv = torch.cat((q, k, v), dim=1).reshape(-1, hidden)
```

若目标 `linear_qkv` 是 TP column-parallel 参数，`shard_mcore_tensor` 再沿 `partition_dim` 把交织后的 tensor 切给对应 TP rank。顺序不能反：如果先把 HF 的 Q/K/V 各自朴素切 TP，再在不了解 GQA group 边界的情况下拼接，某些 rank 会拿到不配套的 KV group。

反方向的 `megatron_to_hf/qwen2.py::convert_qwen2_to_hf` 做严格逆操作：把完整 `linear_qkv` reshape 成 `[num_query_groups, ?, head_dim, hidden]`，按 `[heads_per_group, 1, 1]` split，再分别 flatten 成 q/k/v。gate/up 也是同样思路：HF→Megatron 用 `cat` 合成 `linear_fc1`，Megatron→HF 用 `chunk(2)` 拆回两个名字。

---

## 4. Megatron → HF：两条分布式热路径

两条路径共享 `megatron_to_hf.convert_to_hf` 和最终 HF 命名契约，但 gather 控制流并不相同：

- `UpdateWeightFromTensor`、HF writer 与完整 disk sync 使用 `update_weight/hf_weight_iterator_direct.py::HfWeightIteratorDirect`；
- 专用 `UpdateWeightFromDistributed` 自己流式 TP/EP gather；`UpdateWeightFromDiskDelta` 继承该实现。

### 4.1 先建立全局 `ParamInfo`

每个 rank 从本地参数收集：名称、dtype、shape、字节数、`src_rank` 和 TP 属性。随后：

- PP 组 `all_gather_object` 交换各 stage 的参数表；
- EP 组补齐各 expert 所在 rank；
- virtual PP/MTP 可能让同名参数出现多次，选择最小 `src_rank` 作为确定来源；
- 最后在 Gloo 全局组交换并断言每个 rank 看到的排序后名称/shape/dtype 一致。

这个元数据对账回答了“为什么非源 rank 也能按同样顺序参加 collective”：所有 rank 先冻结出同一份参数计划，后续不会各自按本地 `named_parameters()` 顺序猜。

### 4.2 分桶估算为何要乘 TP size？

本地 `ParamInfo.size` 只是一个 TP shard 的字节数，但传给 SGLang 的通常是 all-gather 后的完整 tensor。`pack_param_info_buckets` 对普通参数乘 TP size，对 expert 参数乘 expert-TP size，再与 `--update-weight-buffer-size` 比较。否则 bucket 会系统性低估峰值显存和传输大小。

单个参数本身超过上限时仍会独占一个超大 bucket；这个参数是软分桶边界，不会切开一个 parameter。

### 4.3 `_get_megatron_full_params` 做了哪些通信？

对一个 bucket：

1. `src_rank` 从 actor backup 取 tensor，其他 rank 分配目标 shape 的空 tensor；
2. PP broadcast 让需要参与后续步骤的 stage 看见参数；
3. expert 参数在 EP 组 broadcast；
4. 恢复 parameter 的 TP 属性；
5. `all_gather_params_async` 批量重建完整 TP/ETP tensor。

之后 `_convert_to_hf_named_tensors` 对每个完整 tensor 调 `convert_to_hf`。`convert_to_hf` 先去 `module.` wrapper、移除 vocab padding，再按 `model_name` 分派到模型族转换器，最后按 rollout 的 quantization config 做 FP8/int4/fp4 处理。

### 4.4 direct 与 distributed 路径如何分工

```text
HfWeightIteratorDirect.get_hf_weight_chunks
  ├─ UpdateWeightFromTensor -> colocated CUDA IPC / 混合拓扑中的远端 NCCL
  └─ HF checkpoint writer   -> safetensors + index
       ├─ UpdateWeightFromDisk -> engine reload
       └─ save_hf_model_to_path -> HF export

UpdateWeightFromDistributed
  ├─ _iter_non_expert_chunks -> TP gather -> convert_to_hf -> NCCL
  └─ _iter_expert_chunks     -> TP + EP gather -> convert_to_hf -> NCCL
       └─ UpdateWeightFromDiskDelta 复用这些 chunk 做差量编码
```

direct 路径先形成全局 `ParamInfo` 计划，适合要统一处理 colocate、混合 engine 拓扑和 checkpoint writer 的场景。专用 distributed 路径则按当前参数流逐个/逐批 gather，只有 PP source rank 生成 HF chunk 并向 engine 广播；disk-delta 覆盖消费动作但保留这套 gather 顺序。

所以“复用”的边界是 `convert_to_hf` 和 chunk 的 HF 命名格式，而不是所有 updater 都通过同一个 iterator。`HfWeightIteratorBase.create` 当前只返回 direct 实现，也没有旧的 bridge wrapper。

---

## 5. HF 导出并不是 `torch.save(state_dict)`

`hf_checkpoint_saver.save_hf_model_direct_to_path` 还需要处理：

1. 输出目录不能与 `--hf-checkpoint` 相同，避免清理旧权重时把模板目录毁掉；
2. 先复制 config、tokenizer 等非权重资产；
3. 多节点 writer 按 chunk 取模分工，每个 writer 仍观察所有 chunk，以维持某些跨 chunk 的有状态配对；
4. 每个 shard 检查 HF tensor 名不能重复；
5. 所有 rank `all_gather_object` 汇总 `weight_map/total_size/shard_files`；
6. 确定性重命名为 `model-00001-of-XXXXX.safetensors`，rank 0 写 `model.safetensors.index.json`。

因此磁盘全量同步和“导出一个可被 Transformers/SGLang 读取的 HF checkpoint”复用同一 writer 是合理的：两者都需要完整的 HF 目录语义，而不只是若干 tensor 文件。

---

## 6. 如何新增一个模型族

最小检查清单：

1. 在 `model_provider.py` 的原生参数/`--spec` 路径或自定义 provider 中保证 Megatron 结构正确；
2. 在 `hf_to_megatron/` 实现 `get_hf_tensor(name, reader, config)`，并按 `config.model_type` 注册；
3. 在 `megatron_to_hf/` 实现逆向命名和布局拆分，并加入 `_convert_to_hf_core` 分派；
4. 覆盖 embedding/lm head tying、QKV bias、QK norm、gate-up、MoE experts/shared expert/router、MTP/MLA、vocab padding 等该模型实际存在的参数；
5. 做 HF→Megatron→HF round-trip，检查名称集合、shape、dtype 和 tensor 值；
6. 再做一次真实 weight sync，确认 SGLang 接收名称与导出名称一致。

“shape 都对”不是充分条件。QKV/GQA、MoE expert 编号和 gate/up 次序即使装错也可能保持相同 shape，必须做数值 round-trip 或小模型 forward 对账。

---

## 7. 小结

- slime 当前没有 Megatron-Bridge 运行路径；结构由 Megatron provider 构造，权重由 slime 内建转换器显式适配；
- HF→Megatron 是启动加载：按目标 parameter 属性切本 rank shard；
- Megatron→HF 是高频分布式路径：先统一 `ParamInfo`、跨 PP/EP/TP 收齐，再按模型族拆分和分桶；
- `HfWeightIteratorDirect` 服务 colocate/hybrid tensor sync、完整磁盘同步和 HF 导出；专用 NCCL 与 disk-delta updater 使用自己的流式 gather；
- 新模型必须分别验证结构、双向映射和 SGLang 命名契约，不能把通用文件格式支持误当成模型语义支持。
