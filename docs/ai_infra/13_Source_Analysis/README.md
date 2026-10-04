# 从一次执行理解四层 AI Infra

同一个 token 在不同层次具有不同身份：它可以是 RL 轨迹的一部分、训练 batch 中的一个有效位置、推理请求的历史状态，也可以是 Attention kernel 的一行输入。源码系列沿这些身份变化展开，解释每层如何分配工作、持有状态、等待依赖和释放资源。

| 系列 | 贯穿场景 | 重点 |
|---|---|---|
| [slime：一轮 RL](slime/README.md) | 生成轨迹 → 奖励 → 目标 → 更新 → 换权 | 角色编排、样本语义、异步与版本一致性 |
| [Megatron：一次参数更新](megatron/README.md) | batch → 前反向 → 梯度同步 → 优化器 | 并行布局、状态分片、调度和恢复 |
| [SGLang：一次生成请求](sglang/README.md) | 接收 → 准入 → prefill/decode → 结束 | 调度、KV 所有权、执行与控制面 |
| [FlashAttention：一次分块计算](flash_attention/README.md) | Q/K/V → tile → 在线归一化 → O | 地址、线程分工、片上状态与流水 |

源码摘录对应各系列声明的快照。教学伪代码只表达关键语义；数学等价不自动意味着所有精度与 backend 都逐位一致。每篇先沿一条确定路径建立解释，再讨论可选分支。

## 跨系列连接

slime 的权重发布由训练端的 gather/转换和 SGLang 的加载、缓存失效共同完成；两侧必须放在同一个版本边界下理解。Megatron 和 SGLang 都会调用 Attention，但训练保存状态、Prefill 和 Decode 的形状及性能目标不同。FlashAttention 的局部速度要通过上层调度和数据供给才能变成端到端收益。

[原理与设计](../README.md)提供概念主文，[性能分析](../12_Performance_Reliability/README.md)解释这些执行路径如何体现在指标和时间线上。
