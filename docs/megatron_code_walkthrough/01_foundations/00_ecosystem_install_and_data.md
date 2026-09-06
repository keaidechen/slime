# 生态、安装、首次训练与数据管线

<!-- learning-position -->
> **学习定位**：A3 · 必修。
> **前置**：[通信与 tensor 基础](<../../../learn_docs/00_Foundations/06_两卡通信与torchrun.md>)。
> **首读/二读**：环境、数据、第一次 step；明确预训练示例与 RL backend 的关系。
> **进度与实验**：[学习清单](<../../../learn_docs/学习清单.md>) · [总入口](<../../../learn_docs/README.md>)。
<!-- /learning-position -->

<a id="beginner-example"></a>

## 入门例子：第一次训练只走最小闭环

先选择仓库已有的小模型与最小可行配置，记录源码版本、模型/tokenizer、数据格式和并行度。首次运行只确认能完成初始化、一个 forward/backward、optimizer update 与保存，不同时打开所有优化选项。

| 阶段 | 要看到的证据 | 常见前置错误 |
|---|---|---|
| 数据准备 | 样本数、tokenizer 与索引匹配 | 路径存在但格式或 vocab 不符 |
| 分布式初始化 | 所有 rank 都打印 group/设备身份 | 只检查 rank 0 |
| 首个 update | loss、grad norm、有效 token、step 递增 | 把仅 forward 当作完成训练 |
| 保存与接续 | 恢复后 step、LR、数据位置一致 | 只看模型能加载 |

正式源码调用在下一篇训练主线中跟踪。实验启动参数以本快照 parser 与已有配置为准，不把教程的示意参数当作跨版本通用命令。

## 机制与实现

## 1. 先分清三个项目

`Megatron Core` 提供 Transformer 组件、并行状态、pipeline schedule、分布式 optimizer/checkpoint 等库能力；`Megatron-LM` 提供参数解析、训练循环、数据集和模型入口；`Megatron Bridge` 负责外部模型格式转换。排错时先判断问题属于库层、应用层还是转换层。

## 2. 安装选择

官方给出 PyPI、源码和 NGC 容器三条路径：

- 只使用稳定 API：优先匹配发布版的 `megatron-core`。
- 修改源码或跟踪当前仓库：使用 editable source install。
- 需要 CUDA、NCCL、Transformer Engine 的已验证组合：使用与当前 Megatron 版本匹配的 NGC PyTorch 容器。

不要只记录 Python 包版本。可复现实验至少固定镜像 digest、GPU/驱动、CUDA/NCCL、PyTorch、Transformer Engine、Megatron commit 与启动命令。FP8 还受 GPU 架构约束。

本仓库的最小 smoke test 应先用 mock data 和小模型跑通，再接真实数据。入口通常是 `Megatron-LM/pretrain_gpt.py`，其三个注入点会在[训练主线](02_training_mainline.md)展开。

## 3. 数据准备不是运行时分词

典型预训练流程为：

```text
JSONL 文本
  -> tools/preprocess_data.py + tokenizer
  -> indexed dataset: .bin（token） + .idx（偏移与长度）
  -> GPTDataset 构造 sample/document/shuffle index
  -> DataLoader
  -> microbatch
```

关键不变量：Tokenizer、词表大小、EOD/PAD ID 和 preprocessing 配置必须与训练一致。`.bin` 保存紧凑 token 数据，`.idx` 提供随机访问；训练不会为每条样本重复做昂贵分词。

数据主入口：

- `Megatron-LM/tools/preprocess_data.py`
- `Megatron-LM/megatron/core/datasets/indexed_dataset.py`
- `Megatron-LM/megatron/core/datasets/gpt_dataset.py`
- `Megatron-LM/megatron/core/datasets/blended_megatron_dataset_builder.py`

## 4. 大规模数据加载

到数百节点后，瓶颈往往是元数据风暴和 cache 构建，而非顺序读带宽。官方建议可归纳为：合并过碎的数据文件；离线预建 dataset cache；必要时预建每个数据集的 metadata；让计算节点复用只读 cache；对象存储场景使用专门的存储客户端与本地 cache。

Multi-Storage Client（MSC）用 `msc://<profile>/<path>` 统一访问 S3/GCS 等对象存储和普通文件系统。它必须显式 `--enable-msc`；对象存储数据还要配置 object-storage cache 与 data cache，并关闭 bin mmap。checkpoint 对象存储当前只支持官方说明的 `torch_dist` 路径。缓存应放在本地 NVMe，并把凭据留在 MSC profile/环境中，不能写进训练命令或文档。

排查顺序：先用 mock data 得到计算上限，再用单数据集排除 blending，随后测 cache cold/warm 两种路径，最后比较各 rank 的 data-wait 分布。仅看 rank 0 会漏掉共享文件系统的尾延迟。

## 5. 首次训练的验收

首次运行不以“loss 打印出来”为唯一成功条件。至少确认：

1. world size 与 TP/PP/CP/EP/DP 分解一致；
2. global batch 满足 `MBS × DP × num_microbatches`；
3. tokenizer special token 与数据索引一致；
4. 连续若干 step loss 有限且 optimizer 确实更新；
5. 保存后能恢复并继续训练；
6. 日志记录完整环境和配置。

官方原页：[安装](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)、[首次训练](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)、[数据准备](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/data-preparation.html)、[规模化数据加载](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/data-loading.html)。
