# 官方文档覆盖矩阵

相关基础：[通信与 tensor 基础](<../../../communication/两卡通信与torchrun.md>)。


本表以仓库内 `Megatron-LM/docs/` 为版本基线；官方 `latest` 链接用于继续查阅，不承诺它与本地快照逐项一致。状态含义：**详解**＝中文章节解释机制与源码；**归纳**＝提炼使用边界；**索引**＝不复写自动 API/维护流程，只给入口。

“详解”表示已有教学解释，不表示本文已经测试所有硬件和配置。先确认自己在找哪一类答案：想知道为什么用 RS，看梯度所有权；想知道某开关在此版本能否组合，看参数校验及对应分支；想知道目标集群能快多少，需要实际运行。后两种答案不能只凭第一种机制解释得出。

一个具体查询流程是“SP 为什么漏了 norm 梯度”：先读 SP 执行篇的两 token 手算，知道 norm 的 γ/β 在不同 token 上累积不同贡献；再查 `_allreduce_non_tensor_model_parallel_grads` 是否识别参数标记；最后在实际 backend 的 finalize 前后观察同一参数的梯度。这样 API 名称才能和它要维护的不变量对应起来。

| 官方主题 | 本系列章节 | 状态 |
|---|---|---|
| Overview / Installation / Quickstart | `01_foundations/00_ecosystem_install_and_data.md` | 归纳 |
| Data preparation / Data loading | `01_foundations/00_ecosystem_install_and_data.md`、训练主线 | 归纳+源码入口 |
| Multi-Storage Client integration | 生态、安装与数据 | 归纳 |
| Training examples / arguments | 训练主线、配置评审 | 归纳 |
| Parallelism guide | `02_parallelism/` 全部 | 详解 |
| Transformer / GPT/BERT/T5 API | 模型与 ModuleSpec、模型谱系 | GPT/Transformer 详解，BERT/T5 索引 |
| Tensor parallel API | TP/SP 与进程组 | 详解 |
| Sequence Parallel 执行 | `02_parallelism/03_sequence_parallel_execution.md` | AG/RS autograd、norm 梯度与 local/TE 边界 |
| Pipeline parallel API / custom layout | PP 章节 | 详解 |
| Distributed package / optimizer | DDP/Optimizer/Checkpoint | 详解 |
| Distributed checkpointing / async save | DDP/Optimizer/Checkpoint、FSDP | 详解 |
| Context Parallel | `02_parallelism/04_context_parallel_execution.md`、CP/MoE/低精度深读 | batch/adapter/梯度链路详解；外部 TE kernel 为依赖边界 |
| EP/ETP 与 parallel folding | `02_parallelism/05_ep_etp_execution.md`、ETP 原理篇 | 两套网格、普通 AlltoAll dispatcher 和 expert TP 详解；其他后端单独验收 |
| Megatron MoE / router replay / paged stash | MoE/MLA/MTP、深读 | MoE/replay 详解，paged stash 索引 |
| MTP / MLA | MoE/MLA/MTP | 归纳+源码入口 |
| Megatron-FSDP | `03_state_and_memory/02_megatron_fsdp.md` | 详解 |
| CPU/fine-grained activation offload | 显存、确定性与 CUDA Graph；深读 | 详解 |
| CUDA Graph | 显存、确定性与 CUDA Graph | 归纳 |
| Deterministic training | 显存、确定性与 CUDA Graph；源码问题详解 | 归纳+验收方法 |
| Tokenizers | 模型谱系与 Tokenizer | 归纳 |
| Hybrid model migration | Hybrid/多模态/RL | 归纳 |
| Supported LLM / multimodal models | 模型谱系；Hybrid/多模态/RL | 归纳 |
| Energon / RL | Hybrid/多模态/RL | 边界说明 |
| Microbatch calculator / scheduler | 训练主线 | 详解 |
| 自动生成 apidocs | 对应各章源码入口 + 官方 API | 索引 |
| 跨章节源码问题 | `06_reference/03_source_questions.md` | 详解 |
| 并行等价性 | `05_design_and_failures/03_parallelism_validation.md` | 数学参考、运行时协议与性能证据；没有 GPU 实测结论 |
| Release notes / roadmap | [官方 release notes](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/releasenotes.html) | 索引，动态内容 |
| Contribute / submit / on-call / docs build | `Megatron-LM/docs/developer/` | 索引，维护流程 |
| API backward compatibility | `Megatron-LM/docs/api-backwards-compatibility-check.md` | 索引，贡献流程 |

## 为什么不逐页复制 API Reference

API 页由源码签名与 docstring 自动生成，中文副本会迅速漂移。遇到具体符号时，先从本系列理解它所属的数据流，再查当前源码和官方 API；升级时用 `rg` 搜符号，不依赖旧行号。

### 怎样处理版本变化与外部依赖

记录四个字段：源码 commit、符号名、分支条件、外部库版本。例如 CP 的 Megatron adapter 接收 `cp_comm_type`，只证明该选项被交给 TE；后续使用哪个通信 kernel、支持哪些 mask，要继续检查对应 TE 版本。不能把 Core 源码中的一个参数当成外部实现已经验证过的证据。

升级时可在本地按符号查找，而不盲跟旧行号：

```bash
rg -n 'def forward_step_calc_loss|calculate_per_token_loss' Megatron-LM/megatron/core/pipeline_parallel/schedules.py
rg -n '_allreduce_non_tensor_model_parallel_grads' Megatron-LM/megatron/core/distributed/finalize_model_grads.py
```

先看签名和调用方，再看分支与数据流。一个同名函数可能改了默认值、输入布局或同步责任；旧示例仍能运行，不代表数学契约没变。本文链接为检索入口，实际升级结论应注明核对的版本和范围。

## 一手入口

- [Megatron Core User Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/)
- [Advanced Features](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/index.html)
- [API Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- 本仓库上游文档：`Megatron-LM/docs/`
