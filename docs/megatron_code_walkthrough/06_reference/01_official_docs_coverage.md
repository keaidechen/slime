# 官方文档覆盖矩阵

本表以仓库内 `Megatron-LM/docs/` 为版本基线，并用官方 latest 目录核对。状态含义：**详解**＝中文章节解释机制与源码；**归纳**＝提炼使用边界；**索引**＝不复写自动 API/维护流程，只给入口。

| 官方主题 | 本系列章节 | 状态 |
|---|---|---|
| Overview / Installation / Quickstart | `01_foundations/00_ecosystem_install_and_data.md` | 归纳 |
| Data preparation / Data loading | `01_foundations/00_ecosystem_install_and_data.md`、训练主线 | 归纳+源码入口 |
| Multi-Storage Client integration | 生态、安装与数据 | 归纳 |
| Training examples / arguments | 训练主线、配置评审 | 归纳 |
| Parallelism guide | `02_parallelism/` 全部 | 详解 |
| Transformer / GPT/BERT/T5 API | 模型与 ModuleSpec、模型谱系 | GPT/Transformer 详解，BERT/T5 索引 |
| Tensor parallel API | TP/SP 与进程组 | 详解 |
| Pipeline parallel API / custom layout | PP 章节 | 详解 |
| Distributed package / optimizer | DDP/Optimizer/Checkpoint | 详解 |
| Distributed checkpointing / async save | DDP/Optimizer/Checkpoint、FSDP | 详解 |
| Context Parallel | CP/MoE/低精度深读 | 详解 |
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
| Release notes / roadmap | [官方 release notes](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/releasenotes.html) | 索引，动态内容 |
| Contribute / submit / on-call / docs build | `Megatron-LM/docs/developer/` | 索引，维护流程 |
| API backward compatibility | `Megatron-LM/docs/api-backwards-compatibility-check.md` | 索引，贡献流程 |

## 为什么不逐页复制 API Reference

API 页由源码签名与 docstring 自动生成，中文副本会迅速漂移。遇到具体符号时，先从本系列理解它所属的数据流，再查当前源码和官方 API；升级时用 `rg` 搜符号，不依赖旧行号。

## 一手入口

- [Megatron Core User Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/)
- [Advanced Features](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/index.html)
- [API Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- 本仓库上游文档：`Megatron-LM/docs/`
