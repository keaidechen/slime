# AdamW 与 Muon：参数分组、矩阵更新与状态 owner

> 基线：工作区 `Megatron-LM` commit `21fe0fe1597932421f1bb0efd93f469376a7a255`，于 2026-10-12 核对，所读子模块无工作区修改。Megatron 的接线由这个 commit 固定；外部 `emerging-optimizers` 的实际版本需另行记录。本篇没有运行 GPU 优化器。


## 先把一次更新的输入固定

需要知道参数的逻辑 shape、TP partition、expert 属性、优化器分组，以及梯度是否已经按正确目标归约。工厂负责建立这些对象间的连接，optimizer step 才消费 ready gradients。构造成功不能证明某个硬件和并行配置下的更新已经正确。

```text
最终 OptimizerConfig + 模型参数属性 + process groups
  → 参数组与 optimizer override
  → Adam/SGD 或 emerging optimizer 实例
  → FP32/BF16 wrapper 与分布式 owner
  → ready gradients → 更新 master/state → 同步计算参数
  → 保存相容参数、组身份、状态与计数
```

## 名为 adam 的配置可能执行 AdamW

[`optimizer/__init__.py`](<../../../../../../Megatron-LM/megatron/core/optimizer/__init__.py>)中的 `_get_megatron_optimizer_based_on_param_groups` 在 `config.optimizer == 'adam'` 时还检查 `decoupled_weight_decay`：

| 实际后端 | 决定规则 |
|---|---|
| PyTorch fallback | True 选择 `torch.optim.AdamW`，False 选择 `torch.optim.Adam` |
| TE/Apex fused Adam | 传入 `adam_w_mode=config.decoupled_weight_decay` |

同文件顶部先选择可用后端；只看类名 `Adam` 或命令行字符串不足以判断衰减是否进入 m/v。[`OptimizerConfig`](<../../../../../../Megatron-LM/megatron/core/optimizer/optimizer_config.py>)中该开关默认 True，最终配置覆盖仍优先。

混合精度 wrapper 负责 master 参数、梯度转换和更新返回值等边界；fused kernel 改变实现方式，算法对照仍应记录 beta、epsilon、偏差校正、decay、dtype 与缺梯度行为。

### 先手算 AdamW，才看 fused 后端

简单梯度下降仅用当前梯度 g；Adam 为每个参数保留一阶矩 m（梯度的指数移动平均）和二阶矩 v（梯度平方的指数移动平均），让更新尺度随历史变化。对某一个标量参数，初始 `m₀=v₀=0`，第 t 次成功更新为：

```text
m_t = β₁ m_(t-1) + (1-β₁) g_t
v_t = β₂ v_(t-1) + (1-β₂) g_t²
m̂_t = m_t / (1-β₁^t)
v̂_t = v_t / (1-β₂^t)
w_t = (1-ηλ) w_(t-1) - η m̂_t / (sqrt(v̂_t)+ε)
```

β₁/β₂ 是 0 到 1 之间的无单位衰减系数；η 是学习率，λ 是解耦权重衰减系数，ε 防止分母过小，t 是该状态的更新计数。m 与梯度同尺度，v 与梯度平方同尺度；所有数组与被更新参数同形状。偏差校正来自零初始化：若梯度恒为 g，则 `m_t=(1-β₁)(1+β₁+...+β₁^(t-1))g=(1-β₁^t)g`，除以该因子才恢复 g 的尺度，v 同理。

令 `w₀=1,g₁=2,β₁=0.9,β₂=0.99,η=0.1,λ=0.01`，暂忽略 ε：`m₁=0.2,v₁=0.04`，校正后 `m̂₁=2,v̂₁=4`，方向为 `2/sqrt4=1`；衰减项让权重先成为 0.999，减去 0.1 后 `w₁=0.899`。这是公式算例，实际 backend 的 epsilon、精度与运算顺序要按配置对照。

AdamW 的 λw 不进入 m/v；若把 L2 正则梯度 `λw` 先加到 g 再使用 Adam，m/v 都被改变，便不是同一更新。fused 表示多个操作由融合实现完成，并不自动决定采用哪种衰减语义。

## Muon 先路由参数，再构造混合优化器

入口仍为 `get_megatron_optimizer`，emerging 路径进入 `_get_megatron_emerging_optimizer`。注册表在 [`emerging_optimizers.py`](<../../../../../../Megatron-LM/megatron/core/optimizer/emerging_optimizers.py>)：

1. `_is_nonlinear_or_embedding` 把非二维参数或带 `is_embedding_or_output_parameter` 的参数识别为默认辅助优化器组。
2. 工厂复制注册表 override，再以 `config.muon_scalar_optimizer` 设置辅助优化器；这个配置默认 `adam`，并非永远硬编码 Adam。
3. `_get_param_groups` 形成含 optimizer 身份和 expert 属性的组；工厂按 `(optimizer_name, is_expert_parallel)` 分桶。LayerWise 路径对 expert 有自己的处理。
4. 主矩阵组通过 `_create_emerging_optimizer` 构造 `TensorParallelMuon`；辅助组沿标准工厂路径构造，再包装和组合。

“scalar optimizer”是配置名，实际可能管理二维 embedding；不能把名字当 shape 判断。参数组表应列出参数名、逻辑 shape、optimizer、decay、LR 与 owner。混合 checkpoint 也需要保留两类状态。

## 默认值要看最终配置，不能只读类签名

`_muon_config_to_kwargs` 通过 `_kwargs_from_config` 优先读取 `muon_` 前缀配置，再读取无前缀项，传给 `TensorParallelMuon`。本快照有几个容易看错的默认值：

| 参数 | `TensorParallelMuon` 类签名 | `OptimizerConfig` |
|---|---|---|
| `nesterov` | True | `muon_nesterov=False` |
| `split_qkv` | False | `muon_split_qkv=True` |
| `tp_mode` | duplicated | `muon_tp_mode=blockwise` |

Megatron-LM 的 argparse 和调用方还可再次改变配置。实际训练应保存最终解析值，而不是把任一层默认值复制成“Muon 默认算法”。尤其 blockwise 会改变正交化的逻辑矩阵范围。

### Muon 为什么关心“完整矩阵”

Muon 的核心是对二维动量矩阵做矩阵级变换。设动量为 `G:[m,n]`，若秩为 q，紧凑奇异值分解为 `G=UΣVᵀ`，其中 `U:[m,q]`、`Σ:[q,q]`、`V:[n,q]`。U/V 的列为对应方向，Σ 的正对角元素是各非零方向的伸缩量；SVD 是 Singular Value Decomposition（奇异值分解）的缩写。理想化的极因子 `UVᵀ` 去掉非零奇异方向的幅值差别，使更新不被最大的奇异值方向完全主导；实际 Muon 使用有限步近似和额外尺度规则，并非精确 SVD。

看一个可手算的对角例子：`G=diag(3,1)` 的理想极因子是 `diag(1,1)`，而逐元素 Adam 通过各元素的二阶历史决定尺度，依据不同。为说明近似方法，采用教学用 Newton–Schulz（NS）迭代：先归一化 `X₀=G/||G||_F`，再做 `X_(k+1)=1.5X_k-0.5(X_k X_kᵀ)X_k`。`||G||_F=sqrt(Σ_ij G_ij²)` 为 Frobenius 范数；零矩阵需单独处理。若非零奇异值处于合适收敛区间，标量递推 `s'=1.5s-0.5s³` 会将它推向 1。

在本例 `||G||_F=sqrt10`，X₀ 的对角值约为 0.9487、0.3162；第一轮得到约 0.9961、0.4585，大值接近 1，小值逐步增大。实际 `newton_schulz_tp` 可以使用不同多项式、系数、迭代次数和精度；上述迭代只解释原理，不能据此复现当前外部包的逐位更新。

为什么不能把矩阵任意分给 DP owner？`XXᵀ` 中的元素是行与行的点积。若 `G=[[1,1],[0,1]]`，其行 Gram 矩阵 `GGᵀ=[[2,1],[1,1]]` 有非零交叉项。分别处理两行时会丢掉这个交叉项，产生不同的矩阵变换。Adam 的逐元素 m/v 更新没有这种矩阵内耦合，因此可沿平坦 buffer 任意切片；Muon 的 owner 必须匹配它所采用的矩阵范围。

## TP 和 QKV 在哪里进入正交化

`TensorParallelMuon.orthogonalize(p, grad)` 中的 grad 是传入该阶段的动量更新矩阵，不能仅凭变量名把它当作原始反向梯度。

- 根据 `p.expert_tp` 选择 `pg_collection.expt_tp` 或 `pg_collection.tp`。
- Blockwise 模式将 `partition_dim` 设为 None；其他模式读取参数的 `partition_dim`，并把 -1 视为未切分。
- 开启 QKV split 时，优先读取参数自己的 `qkv_split_shapes`，否则使用模型配置推导值；按 query group reshape、split，分别正交化，再按原布局拼回。

内部 `scaled_orthogonalize_fn` 调用外部 `newton_schulz_tp`，传入 steps、coefficient type、TP group、partition dim 与模式。它根据 partition dim 和 group size 恢复用于缩放的 shape，再调用 `get_muon_scale_factor`。

Blockwise 分支在这里传给外部函数的 mode 名称是 duplicated，但同时传入 `partition_dim=None`；这不能读成“自动 gather 全矩阵”。完整意义必须结合两个参数。底层 collective 和系数实现来自外部包，Megatron 接口本身不足以证明具体通信量。

该包的[官方源码入口](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/main/emerging_optimizers/orthogonalized_optimizers/muon_utils.py)用于继续追踪；它是滚动分支，不属于本篇固定快照。实际运行应记录包版本/commit、NS 系数、归一化与 matmul 精度。`adaptive_muon` 还加入二阶状态，其账本不能套用标准 Muon 的单动量假设。

## DP owner 为什么按完整参数安排

[`LayerWiseDistributedOptimizer`](<../../../../../../Megatron-LM/megatron/core/optimizer/layer_wise_optimizer.py>)的布局路径用 `_compute_per_buffer_param_layout` 安排参数，再由 `_shard_params_from_layout` 读取 owner。这里检查单个受管参数不能跨越 owner shard 边界；缺少 layout 时还有按参数大小分配的旧 ping-pong 路径。

这里的“完整参数”是当前 TP rank 上的参数对象。若模型已按 TP 切分，它仍只是全局矩阵的一片，所以 DP owner 保持完整不等于 NS 已获得全局矩阵。

在 DDP 使用 distributed optimizer buffer 且开启 LayerWise 的路径中：

| 参数组 | owner 与 buffer |
|---|---|
| Muon 受管矩阵 | LayerWise 的 shard-aligned layout |
| 辅助 Adam 等参数 | 过滤后的非 LayerWise buffer，交给单独的 `DistributedOptimizer` |

旧 ping-pong 路径未必采用这种分离方式。应追踪 `use_separate_distributed_optimizer` 条件，不能把一条路径的 owner 规则概括成全部 Muon 部署。

`step_with_ready_grads` 先执行更新，再决定参数同步：buffer 路径使用受管 bucket 的同步，旧路径使用自身 gather/broadcast；若开启 `overlap_param_gather`，同步可延后到前向 pre-hook。这个开关与 `overlap_param_gather_with_optimizer_step` 不同，后者在该 emerging + LayerWise 工厂组合中被显式拒绝。

例如当前 TP rank 上有两个矩阵参数 A（6 个元素）、B（2 个元素），DP=2。按元素均分会让 owner0 持有 A 的前 4 个、owner1 持有 A 的后 2 个和 B；适合逐元素 Adam，但不方便对完整 A 做矩阵更新。按完整参数分配可让 owner0 管 A、owner1 管 B，代价是状态和计算不均衡，或需要 padding 调整 bucket 边界。

LayerWise owner 的更新完成后还要使其他数据副本获得新参数；若 gather 延迟到 forward pre-hook，只有 hook 等到通信完成后消费者才可读。TP blockwise 模式又是另一个选择：它有意把 TP-local 矩阵作为更新范围。DP owner 保持 A 完整，并不取消 TP blockwise 带来的算法差异。

## 配置与恢复有哪些明确边界

| 快照中的检查位置 | 已核对的限制 | 不能扩大的结论 |
|---|---|---|
| emerging 工厂 | 缺少外部包报错；拒绝 FP16 | BF16/FP32 构造路径存在不代表所有组合实测通过 |
| LayerWise 与辅助 DistOpt 分离路径 | 要求 `num_distributed_optimizer_instances == 1` | 该限制有具体分支条件 |
| Megatron-LM `validate_args` | emerging 路径拒绝 Torch-FSDP2、Megatron-FSDP；限定 checkpoint 格式 | Core API、其他调用方和未来版本需单独核对 |
| LayerWise `sharded_state_dict` | 注释限定 fixed DP usage，并处理空组/副本元数据 | 不能仅凭 sharded 格式宣称任意 DP 重分片恢复 |

参数校验入口见 [`training/arguments.py`](<../../../../../../Megatron-LM/megatron/training/arguments.py>)。从旧 `dist_muon` 名称或 `--use-distributed-optimizer` 到最终 LayerWise 配置还经过转换，所以配方记录应包括解析后的配置和 DDP layout。

恢复至少核对矩阵/辅助参数组身份、各自动量或 m/v、主权重、计数与 scheduler、QKV 分组、TP 模式、缩放和 NS 配方。改变矩阵切分后，即使权重可加载，下一次更新也可能已变；参考[分片反例](<../../../training/Muon与矩阵优化.md#分片反例完整矩阵与各行独立更新不同>)。

## 怎样取得可审查的运行证据

先用固定梯度核对一次更新：保存更新前参数与状态、参数组表、最终配置、更新方向和更新后参数；分别比较矩阵组和辅助组。之后在相同配置下比较连续运行与保存/恢复后的下一步，再增加 TP 和 DP。Blockwise 的正确参考应是同样 blockwise 配方，不能要求它等于完整矩阵极因子。

性能证据分开记录 NS、梯度规约、gather/sync、optimizer step、峰值工作区与完整 step；按相同质量门槛比较总时间。CPU [教学参考](<../../../../examples/optimizer_walkthroughs.py>)只能支持算术和矩阵反例，不能替代本节运行证据。

[Megatron 目录](<../README.md>) · [训练专题](<../../../training/README.md>) · [分布式训练性能](<../../../performance/04_megatron.md>)
