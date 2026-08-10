# Hybrid、Mamba、多模态、Energon 与 RL 生态

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

