# Hybrid、Mamba、多模态、Energon 与 RL 生态

相关基础：[GPU、tensor 与通信基础](<../../../foundations/README.md>)。


## 入门例子：进入 Hybrid 或多模态前先找主循环差异

标准自回归文本模型有逐 token 的 KV；状态空间或混合层可能维护不同的递归状态。多模态还增加解码、预处理、编码器和跨模态输入，其 CPU/GPU 长尾不能直接归因到语言模型主干。

先拿一条最小输入画出“读取→预处理→编码→语言模型→loss”，标出哪些步骤可批处理、哪些 shape 动态、哪些状态需要保存。Energon 等数据系统应放在这条实际数据流中理解。

RL 接口、Hybrid 与多模态分别增加不同的状态和数据路径：前者连接样本与训练目标，后两者改变模型内部组成。比较这些机制比枚举模型名称更能解释实现约束。

## 1. HybridModel 的核心

HybridModel 用 layer pattern 描述 Transformer、Mamba/SSM 等不同 layer 的组合，而不再假定 decoder 是同构层序列。迁移 GPTModel 时先选择架构保持的 pattern，再转换 checkpoint；pattern 展开后的 layer index 会影响 PP layout、checkpoint key 与自定义 provider。

安全迁移顺序：单层单卡参数映射 → 完整模型固定输入 → checkpoint 双向转换 → PP 布局 → 目标规模训练。不要把架构迁移与并行重排放在同一个首次实验里。

SSM（State Space Model，状态空间模型）的直观起点是递推状态：`s_t=a s_(t-1)+b x_t,y_t=c s_t`。x_t 是本位置输入，s_t 汇总过去信息，a/b/c 控制遗忘、写入和读出。取标量 `a=1/2,b=c=1,s₀=0`，输入 `[2,0,1]`，状态依次为 2、1、1.5，输出也为 `[2,1,1.5]`。推理只须保留最后状态继续递推，不必为这个简化模型保存所有历史 K/V。

Mamba 的选择性状态空间模型使部分变换依赖输入，并有向量状态、卷积和门控，上述标量式只解释“状态如何传递”。训练的反向仍要考虑沿时间的依赖，不能据推理只需最后状态，就断言训练也只存一个标量。高效扫描实现、重计算和 checkpoint 管理仍是系统问题。

若四层 pattern 为 `[attention,SSM,attention,SSM]`，第 2 层的参数与状态类型依赖这个位置；换成 `[attention,attention,SSM,SSM]` 不是简单重命名层号，前向函数已经变化。PP 可以在保持逻辑顺序的前提下改变设备归属，架构转换则必须另行定义。

## 2. 多模态

MIMO 是面向任意输入/输出模态的实验框架；仓库还包含 LLaVA/NVLM、vision encoder 等例子。多模态训练新增的系统问题包括不同模态变长 batch、encoder/decoder 负载不平衡、projector placement、冻结参数、组合 checkpoint 和数据 collate。

模型入口在 `megatron/core/models/mimo/`、`multimodal/`、`vision/`，示例在 `Megatron-LM/examples/` 对应目录。实验 API 要预期变化，优先在应用层隔离适配。

用一张缩小的图片和两个文本 token 看 shape：RGB 图片 `[3,4,4]` 按 `2×2` patch 切成 4 块，每块展平为 12 个像素值；视觉编码器将它们映射为 `[4,H_v=3]`，projector 权重 `[3,H=4]` 将视觉宽度接到语言宽度，得到 `[4,4]`。若接口直接拼入两个文本 embedding `[2,4]`，语言模型输入就是 `[6,4]`，再按 batch/layout 增加相应轴。

这是教学拼接方案，真实模型可能还有 CLS token、图像占位符、不同分辨率及交叉注意力。位置编码、attention mask 和 loss mask 必须配合实际方案。若只监督文本，视觉位置的 loss mask 可为 0，但它们仍通过文本预测接收梯度。冻结视觉编码器表示不更新其参数；projector 仍可训练，不能把“无视觉 loss”误读为“整个视觉路径无需反向”。

生命周期是 CPU 解码/预处理 → GPU 视觉编码 → projector → 语言模型 → loss。冻结模块且输入也不求梯度时可减少其 autograd 保存，但传给可训练 projector 的视觉特征仍要保留到相应反向消费结束。不同图片分辨率还会同时改变 CPU 解码时间、视觉 token 数和语言模型负载。

## 3. Energon

Megatron Energon 面向大规模、多模态 dataset 与 blending。它处理 shard、worker、样本恢复和多源混合等数据层问题，但不改变训练 step 的并行语义。评估时分别测 dataset decode/collate、host→device、模型计算，避免把图像解码瓶颈误判为 GPU bubble。

例如两个 worker 各预取两条样本，训练主循环此刻只消费前三条。恢复时应重现“接下来尚未消费的逻辑样本”，不能只恢复文件读指针到 worker 已预取的位置，否则可能跳过第四条甚至更多样本。样本 ID、worker 的随机变换状态、混合数据源选择与训练消费边界共同决定连续性。

若图像解码 8 ms、H2D 2 ms、模型计算 10 ms，串行一次为 20 ms；在资源与依赖允许的稳定流水中可接近最长阶段 10 ms，但首批填充、队列不足和慢样本仍会暴露。测量时把 decode、collate（组批/补齐）和 GPU 计算分开，才能知道改数据 worker 数还是改并行度。

## 4. Megatron RL

Megatron RL 把 rollout、experience、训练 engine 等组件接入 MCore 能力。RL 场景的主要差异是数据在线生成、长度和 batch 动态、policy/reference/reward 多模型状态，以及训练/推理权重同步。该模块仍是快速演进能力；本仓库 slime 的集成细节应继续读 [slime 源码系列](<../../slime/README.md>)，不能只靠上游概览推断。

RL（Reinforcement Learning，强化学习）与离线 next-token 训练的关键区别，是样本由某版 policy（策略模型）生成，reward（奖励）再评价结果。一个最小数据流是：策略 θ_old 生成 response → 记录 token、mask、旧策略 log-prob 和版本 → 计算奖励及 advantage（相对基线的好坏程度）→ 训练策略 θ → 发布新权重供下一轮生成。reference model 常用于约束策略偏离，reward model/规则用于评分，两者职责不同。

例如某位置生成动作 a 的旧概率为 0.2，新概率为 0.3，则概率比 `r=π_θ(a|prefix)/π_old(a|prefix)=1.5`，也可由 `exp(logp_new-logp_old)` 计算。若某算法用 `r×advantage` 作为更新项的一部分，旧 log-prob 若错误对应另一版策略，这个比值就不再表示原样本的变化。PPO 等算法还会做裁剪和其他目标处理，不能把此式当作 slime 所有训练模式的完整 loss。

因此训练/推理权重同步需要版本边界：生成 worker 应知道自己使用哪一版，trainer 应知道 batch 来自哪一版，同步的临时 buffer 必须保持有效直到接收方完成。变长 response 还需要正确的有效 token 分母；它与预训练中的 packed/变长归一化问题相通。

## 5. 共同设计原则

无论混合层、多模态还是 RL，都把“模型结构”“数据契约”“并行 placement”“状态保存”分开定义；为每层建立独立 correctness oracle，再组合。这样才能判断错误来自数学实现、batch 组织、进程组还是生命周期。
