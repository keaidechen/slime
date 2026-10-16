# Megatron 的程序边界、运行环境与数据管线

相关基础：[通信与 tensor 基础](<../../../communication/两卡通信与torchrun.md>)。


## 从数据到保存的最小训练闭环

训练程序的最小闭环包含初始化、forward/backward、optimizer update 与保存。源码版本、模型/tokenizer、数据格式和并行度共同决定这些阶段的含义：打印 loss 只能说明到达了某个计算位置，不能说明状态更新与恢复已经闭合。

| 阶段 | 要看到的证据 | 常见前置错误 |
|---|---|---|
| 数据准备 | 样本数、tokenizer 与索引匹配 | 路径存在但格式或 vocab 不符 |
| 分布式初始化 | 所有 rank 都打印 group/设备身份 | 只检查 rank 0 |
| 首个 update | loss、grad norm、有效 token、step 递增 | 把仅 forward 当作完成训练 |
| 保存与接续 | 恢复后 step、LR、数据位置一致 | 只看模型能加载 |

正式源码调用在训练循环实现资料中跟踪。启动参数的含义以本快照 parser 与已有配置为准，不把教程的示意参数当作跨版本通用命令。

## 1. 先分清三个项目

`Megatron Core` 提供 Transformer 组件、并行状态、pipeline schedule、分布式 optimizer/checkpoint 等库能力；`Megatron-LM` 提供参数解析、训练循环、数据集和模型入口；`Megatron Bridge` 负责外部模型格式转换。排错时先判断问题属于库层、应用层还是转换层。

这里的“库”可以类比程序中被调用的一组函数，“训练应用”则负责决定先调用谁、循环多少次、失败时保存什么。Bridge 还提供训练适配等能力；本系列主要使用它的格式转换边界。例如读不懂外部 checkpoint 的 QKV 排列，应先查转换；一批 token 的 mask 错了，应先查数据和任务入口；矩阵切片怎样汇总，则进入 Core。一次运行可能同时经过三者，项目名不能代替调用链。

## 2. 软件组合决定可用实现

官方给出 PyPI、源码和 NGC 容器三条路径：

- 只使用稳定 API：优先匹配发布版的 `megatron-core`。
- 修改源码或跟踪当前仓库：使用 editable source install。
- 需要 CUDA、NCCL、Transformer Engine 的已验证组合：使用与当前 Megatron 版本匹配的 NGC PyTorch 容器。

不要只记录 Python 包版本。可复现实验至少固定镜像 digest、GPU/驱动、CUDA/NCCL、PyTorch、Transformer Engine、Megatron commit 与启动命令。FP8 还受 GPU 架构约束。

CUDA 是 GPU 计算的软件平台；NCCL 是 GPU 间通信库；Transformer Engine（TE）提供面向 Transformer 的优化算子与低精度实现。Python 能导入包，仅证明 Python 层可见，不能证明编译扩展与驱动相容。可以把排查分为“导入 → 分配 GPU 张量 → 单卡算子 → 两卡通信 → 完整更新”五级：每一级新增一个依赖，失败时才有明确的缩小范围。

mock data 隔离数据读取，真实数据增加 tokenizer、索引与存储路径的约束。入口通常是 `Megatron-LM/pretrain_gpt.py`，其三个注入点会在[训练主线](<02_training_mainline.md>)展开。

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

### 用五个整数走完一条样本

Tokenizer（分词器）将文本映射为词表中的整数 ID；token 可以是字、子词或其他片段，不必对应一个完整词。设一段预处理结果为 `[5,8,3,9,2]`，其中 `2` 是文档结束符 EOD（End of Document）。做长度 `S=4` 的下一 token 预测，需要五个位置：

```text
原始 token 窗口   [5,8,3,9,2]
输入 tokens      [5,8,3,9]   shape [B=1,S=4]
监督 labels      [8,3,9,2]   shape [1,4]
loss_mask        [1,1,1,1]   shape [1,4]
```

位置 0 的任务是根据 token 5 预测 token 8，最后位置根据已有前缀预测 EOD。具体数据选项可以改变 EOD 的 loss mask、位置重置与跨文档行为，应在 `GPTDataset` 核对；此例假设四个目标都参与训练。PAD（Padding）是补齐长度的占位符，通常不应被当成真实预测目标。attention mask 控制“能看谁”，loss mask 控制“哪几个误差计入目标”，二者不能互换：即使不计算某位置的 loss，它仍可能作为后续位置的上下文。

`.idx` 中的文档长度和偏移让读取器找到上述五个整数；sample index 决定本轮取哪个窗口，shuffle index 决定样本访问次序。CPU 的数据读取/组批程序准备整数张量，随后搬到 GPU；embedding 才把每个 ID 换成一个长度 H 的浮点向量。GPU 不直接把 `.bin` 当作权重或 hidden states。

数据主入口：

- `Megatron-LM/tools/preprocess_data.py`
- `Megatron-LM/megatron/core/datasets/indexed_dataset.py`
- `Megatron-LM/megatron/core/datasets/gpt_dataset.py`
- `Megatron-LM/megatron/core/datasets/blended_megatron_dataset_builder.py`

## 4. 大规模数据加载

到数百节点后，瓶颈往往是元数据风暴和 cache 构建，而非顺序读带宽。官方建议可归纳为：合并过碎的数据文件；离线预建 dataset cache；必要时预建每个数据集的 metadata；让计算节点复用只读 cache；对象存储场景使用专门的存储客户端与本地 cache。

Multi-Storage Client（MSC）用 `msc://<profile>/<path>` 统一访问 S3/GCS 等对象存储和普通文件系统。它必须显式 `--enable-msc`；对象存储数据还要配置 object-storage cache 与 data cache，并关闭 bin mmap。checkpoint 对象存储当前只支持官方说明的 `torch_dist` 路径。缓存应放在本地 NVMe，并把凭据留在 MSC profile/环境中，不能写进训练命令或文档。

排查顺序：先用 mock data 得到计算上限，再用单数据集排除 blending，随后测 cache cold/warm 两种路径，最后比较各 rank 的 data-wait 分布。仅看 rank 0 会漏掉共享文件系统的尾延迟。


cold cache 指首次访问必须构建索引或从远端读取；warm cache 指所需数据已在缓存中。例如每批计算 10 ms，读取 6 ms：完全串行需要 16 ms；有独立 CPU worker 和预取队列时，稳态下可能接近 `max(10,6)=10 ms`。若某批读取要 18 ms，GPU 仍可能等到它完成。预取减少暴露等待，却增加主机内存、在途批次和恢复数据游标的复杂度。已排队的批次不等于已被训练消费的批次，checkpoint 应按训练认可的消费边界恢复。

异步 H2D（Host to Device，主机到设备）复制期间，源缓冲区不能被数据 worker 改写；GPU 消费者必须等待复制完成。worker、预取队列和 GPU stream 是不同执行主体，“Python 已返回”不足以证明数据已经可以释放。

## 5. 一次训练闭环的六个不变量

一次可解释的参数更新，需要以下条件同时成立：

MBS（Micro Batch Size）是每个数据并行参与者一次前反向处理的样本数；GBS（Global Batch Size）是一次逻辑更新的全局样本数。固定均匀批次时 GBS=MBS×DP 大小×累积次数。

1. world size 与 TP/PP/CP/EP/DP 分解一致；
2. global batch 满足 `MBS × DP × num_microbatches`；
3. tokenizer special token 与数据索引一致；
4. 连续若干 step loss 有限且 optimizer 确实更新；
5. 保存后能恢复并继续训练；
6. 日志记录完整环境和配置。

官方原页：[安装](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)、[首次训练](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)、[数据准备](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/data-preparation.html)、[规模化数据加载](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/data-loading.html)。
