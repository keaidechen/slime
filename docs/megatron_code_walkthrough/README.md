# Megatron-LM / Megatron Core 系统学习与代码走读

这套文档面向第一次接触 Megatron 的 infra 工程师。目标不是记住启动参数，而是能回答四类问题：

1. 一个训练 step 从哪里进入，数据、前向、反向、通信和 optimizer step 如何串起来？
2. TP、PP、DP、CP、EP 分别切什么，创建哪些进程组，通信量和显存代价是什么？
3. 遇到 OOM、吞吐下降、collective hang、loss 漂移、断点不兼容时，如何定位到正确层次？
4. 如何安全地改模型、调度器、并行布局、checkpoint 或融合算子，并设计验证实验？

## 阅读基线

- 本仓库 Megatron-LM 快照：`21fe0fe1597932421f1bb0efd93f469376a7a255`
- 文中行号只服务于该快照；上游变化后优先搜索类名或函数名。
- “Megatron-LM”指端到端参考训练程序；“Megatron Core / MCore”指 `megatron/core/` 下可组合的训练组件。
- slime 集成视角另见 `docs/code_walkthrough/12_megatron_lm_internals.md`；本系列聚焦上游本身。

## 推荐顺序

| 顺序 | 文档 | 学完应能做到 |
|---|---|---|
| 0 | [00_learning_map.md](00_learning_map.md) | 建立能力地图，知道哪些是主线、哪些是进阶 |
| 1 | [01_training_mainline.md](01_training_mainline.md) | 跟踪 `pretrain_gpt.py` 到一次参数更新 |
| 2 | [02_model_and_transformer.md](02_model_and_transformer.md) | 理解 GPT、Transformer spec 和 Transformer Engine 的边界 |
| 3 | [03_parallel_state_and_tensor_parallel.md](03_parallel_state_and_tensor_parallel.md) | 手算 rank 分组，读懂 TP/SP 的通信 |
| 4 | [04_pipeline_parallel.md](04_pipeline_parallel.md) | 读懂 1F1B、交错流水线、bubble 和 P2P |
| 5 | [05_data_parallel_optimizer_checkpoint.md](05_data_parallel_optimizer_checkpoint.md) | 理解 DDP bucket、分布式 optimizer 和分布式 checkpoint |
| 6 | [06_context_moe_precision.md](06_context_moe_precision.md) | 掌握 CP、MoE/EP、FP8 及组合约束 |
| 7 | [07_performance_debugging_practice.md](07_performance_debugging_practice.md) | 建立容量估算、profiling、故障定位与改码验证闭环 |

## 合格标准

完成后至少应能独立完成：

- 给定 `world_size/model/batch/seq_len`，提出一个可解释的 TP×PP×CP×EP×DP 布局；
- 从日志中的 rank、通信组、microbatch 和 pipeline stage 还原执行位置；
- 区分参数显存、optimizer state、gradient、activation、临时通信 buffer；
- 用 Nsight Systems/PyTorch profiler 判断是算子、通信、pipeline bubble、输入还是 checkpoint 瓶颈；
- 解释为何 TP 常配 SP、为何 EP+TP 要求 SP、为何 PP 的 global batch 必须产生足够 microbatch；
- 修改一个 `ModuleSpec`、pipeline layout 或 optimizer 配置，并用小规模 correctness test 与多卡性能实验验证。

## 一手资料

- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Megatron Core 官方 User Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html)
- [并行策略指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Pipeline Parallel API](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/pipeline_parallel.html)
- [Megatron-LM 论文](https://arxiv.org/abs/1909.08053)
- [高效大规模流水线训练论文](https://arxiv.org/abs/2104.04473)

## 如何把这套文档当作源码教材

每一章都建议分三遍读：

1. **第一遍只走控制流**：从章节列出的入口函数开始，用 IDE “Go to Definition” 跟到返回，不钻进 CUDA kernel。
2. **第二遍记分布式语义**：为每个 tensor 写下 global shape、local shape、所属 rank group、产生它的 stream。
3. **第三遍做实验**：使用章节末尾的最小实验，把日志或 profiler 结果与源码预测逐项对齐。

推荐维护一张个人表格：

| 符号 | 输入布局 | 输出布局 | collective | 生命周期/所有者 |
|---|---|---|---|---|
| `ColumnParallelLinear` | replicated X | output-sharded Y | backward reduce | TP group |
| `RowParallelLinear` | input-sharded X | replicated/sequence-sharded Y | forward AR/RS | TP group |
| pipeline activation | stage-local | next stage input | P2P | microbatch |
| grad buffer bucket | local grads | DP-reduced shard/full | RS/AR | iteration |

### 源码引用约定

文中的代码片段会删除日志、类型注解或少数旁支，突出控制流。它们是用于讲解的“等价摘录”，完整实现请回到固定快照查看。每一章末尾的“延伸阅读”都与本章主题直接对应，而不是泛泛的资料堆积。

### 建议工具

```bash
# 找符号，而不是依赖易漂移的行号
rg -n "def train_step|class DistributedOptimizer" Megatron-LM/megatron

# 看一个函数最近为何变化
git -C Megatron-LM log -L :train_step:megatron/training/training.py

# 只抓稳定的几个训练 step
nsys profile -t cuda,nvtx,osrt,cublas,nccl \
  -o megatron_trace torchrun ...
```

如果本机 git 因 submodule ownership 拒绝执行，可直接用编辑器阅读；不要为了读代码修改仓库全局安全配置。
