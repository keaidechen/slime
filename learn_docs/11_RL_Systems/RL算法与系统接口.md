# RL 算法需要系统保存什么

> 类型：算法—系统接口。前置：[训练 step](../09_Training_Systems/训练步与优化器.md)、[生成与采样](../10_Inference_Serving/采样与推测解码.md)。这里只讨论与 Infra 直接相关的数据语义，不代替完整 RL 教材。

## 一条轨迹有哪些身份

在语言模型 RL 中，prompt 和已生成前缀构成决策上下文，生成 token 是动作，环境或奖励过程提供反馈。一条训练样本至少应能关联 prompt、response、终止原因、mask、行为策略版本和相应概率信息。

Actor 产生并更新策略；reward 给出评价；reference 常用于约束偏离；critic/value 是否存在取决于算法。角色名称相同也不保证每个框架采用相同执行顺序和数据格式。

## 重要性比率为什么要求行为信息

一个常见 token 级比率为 $r_t=\exp(\log\pi_\theta(a_t|s_t)-\log\mu(a_t|s_t))$。分母对应实际产生数据的行为分布。若分母混用了不同权重、模板或采样处理，比率就不再表示原来想要的关系。

PPO 风格的裁剪目标还需要优势估计和有效位置。GRPO 等组相对方法依赖样本分组与奖励统计；把组拆散后任意重组，可能改变归一化语义。系统调度必须保留算法要求的关联。

## Token 对齐是一种数据契约

logits 的位置、预测 token、response mask、padding 和 packed offsets 必须一致。一个 off-by-one 可以让程序正常运行却优化错误目标。截断、工具调用、失败重试和 partial rollout 需要定义哪些 token 属于可训练动作。

## 吞吐之外的有效工作

完成的 token 不一定都进入训练，进入训练的样本也不一定具有所需有效性。应记录生成、通过过滤、训练消费和因版本或失败丢弃的数量，才能判断优化是否增加有效数据供给。

## 演进中的交换

同步批次易于界定策略版本，但会等待长尾；流水化减少空闲，增加版本差；异步系统提高资源解耦程度，却需要明确样本陈旧度、行为信息和校正策略。系统层能记录版本，无法仅靠版本号证明算法质量不受影响。

实现：[slime RL 算法](../../docs/code_walkthrough/05_rl_algorithms.md)、[样本与异步 buffer](../../docs/code_walkthrough/03_data_buffer_partial_rollout_async.md)。

参考：[PPO](https://arxiv.org/abs/1707.06347)、[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)。

## 完整推演与演进资料

- [从轨迹到目标：Token对齐、GRPO组统计与PPO裁剪](推演_从轨迹到PPO与GRPO目标.md)：3条回答；对齐、组统计、裁剪梯度、归约与GAE。


---

[所属专题](README.md) · [百科总览](../README.md)
