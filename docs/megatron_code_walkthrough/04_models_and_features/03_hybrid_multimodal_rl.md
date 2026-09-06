# Hybrid、Mamba、多模态、Energon 与 RL 生态

<!-- learning-position -->
> **学习定位**：A8 · 参考。
> **前置**：[GPU、tensor 与通信基础](<../../../learn_docs/00_Foundations/README.md>)。
> **首读/二读**：RL 相关入口可提前；Hybrid/Mamba/多模态/Energon 后续按需。
> **进度与实验**：[学习清单](<../../../learn_docs/学习清单.md>) · [总入口](<../../../learn_docs/README.md>)。
<!-- /learning-position -->

<a id="beginner-example"></a>

## 入门例子：进入 Hybrid 或多模态前先找主循环差异

标准自回归文本模型有逐 token 的 KV；状态空间或混合层可能维护不同的递归状态。多模态还增加解码、预处理、编码器和跨模态输入，其 CPU/GPU 长尾不能直接归因到语言模型主干。

先拿一条最小输入画出“读取→预处理→编码→语言模型→loss”，标出哪些步骤可批处理、哪些 shape 动态、哪些状态需要保存。Energon 等数据系统应放在这条实际数据流中理解。

当前 RL 主线可先读本篇的 RL 接口部分。Hybrid/多模态作为专项保留；验收是能对比新增状态与数据路径，而非记住所有模型名称。

## 机制与实现

## 1. HybridModel 的核心

HybridModel 用 layer pattern 描述 Transformer、Mamba/SSM 等不同 layer 的组合，而不再假定 decoder 是同构层序列。迁移 GPTModel 时先选择架构保持的 pattern，再转换 checkpoint；pattern 展开后的 layer index 会影响 PP layout、checkpoint key 与自定义 provider。

安全迁移顺序：单层单卡参数映射 → 完整模型固定输入 → checkpoint 双向转换 → PP 布局 → 目标规模训练。不要把架构迁移与并行重排放在同一个首次实验里。

## 2. 多模态

MIMO 是面向任意输入/输出模态的实验框架；仓库还包含 LLaVA/NVLM、vision encoder 等例子。多模态训练新增的系统问题包括不同模态变长 batch、encoder/decoder 负载不平衡、projector placement、冻结参数、组合 checkpoint 和数据 collate。

模型入口在 `megatron/core/models/mimo/`、`multimodal/`、`vision/`，示例在 `Megatron-LM/examples/` 对应目录。实验 API 要预期变化，优先在应用层隔离适配。

## 3. Energon

Megatron Energon 面向大规模、多模态 dataset 与 blending。它处理 shard、worker、样本恢复和多源混合等数据层问题，但不改变训练 step 的并行语义。评估时分别测 dataset decode/collate、host→device、模型计算，避免把图像解码瓶颈误判为 GPU bubble。

## 4. Megatron RL

Megatron RL 把 rollout、experience、训练 engine 等组件接入 MCore 能力。RL 场景的主要差异是数据在线生成、长度和 batch 动态、policy/reference/reward 多模型状态，以及训练/推理权重同步。该模块仍是快速演进能力；本仓库 slime 的集成细节应继续读 `docs/code_walkthrough/`，不能只靠上游概览推断。

## 5. 共同设计原则

无论混合层、多模态还是 RL，都把“模型结构”“数据契约”“并行 placement”“状态保存”分开定义；为每层建立独立 correctness oracle，再组合。这样才能判断错误来自数学实现、batch 组织、进程组还是生命周期。
