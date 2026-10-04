# RL 数据与数值正确性：系统优化前先确定等价关系

> 前置：已有 PPO/GRPO 概念，学过 tensor、DP、prefill/decode。目标：把算法目标写成分布式执行中的不变量。


## 一个样本至少有四种身份

样本不仅是 prompt 和 response。需要分清逻辑 sample id、token 序列、生成时的 policy version，以及消费它的 training update。多轮任务还可能包含多个生成片段、环境动作和不同版本。

先画这条链：

```text
prompt/template/tokenizer
  → 行为策略生成 response + 行为 logprob
  → reward/filter
  → packing/mask/advantage
  → 当前策略前向与 loss
  → optimizer update
  → 导出与换权
```

任何优化若改变链中一步，都要列出哪些数学量保持不变，哪些只是允许在容差内变化。


## 平均 loss 的平均通常不是全局 token mean

rank 0 有 2 个有效 token，loss 为 `[1,3]`，local mean 是 2；rank 1 有 1 个有效 token，loss 为 `[9]`，local mean 是 9。

- 平均两个 local mean：`(2+9)/2 = 5.5`。
- 全局 token mean：`(1+3+9)/3 = 13/3`。

两个目标不同。PPO/GRPO 的具体实现也可能有 sequence mean、group mean 等目标，所以第一步是确认目标定义，不能见到不等长就擅自改为 token mean。

若目标确实是全局 token mean，设各 rank 的可微 loss sum 为 `S_r`，全局有效 token 数为 `N`，DDP 对梯度做平均且参与者数为 `D`，则每 rank 用 `D * S_r / N` backward，平均后的梯度才对应 `sum_r S_r / N`。

这里假设普通 DDP 梯度平均，没有额外的 microbatch、CP 或框架缩放；实际 Megatron/slime 必须把那些因子一起推导。用于日志的 detached sum/count 规约，也不能自动替代训练梯度路径。

运行纯 CPU 算例：[loss_normalization.py](../examples/loss_normalization.py)。


## Mask 是训练目标的一部分

至少区分 padding mask、attention mask、response loss mask，以及 partial rollout 中不参与本次 loss 的旧片段。它们不一定是同一个 tensor。

例如 `[prompt,prompt,response,response,pad]` 中，loss mask 可以是 `[0,0,1,1,0]`，但 attention 还需要表达可见上下文。不能为了少算 prompt 的 loss 就让 response 看不到 prompt。

检查项：shift 后 token 与 logprob 是否对齐；EOS 是否计入；截断与正常结束如何区分；全零 mask 分母怎么处理；packing 后样本边界是否泄漏 attention。


## 行为 logprob 与当前 logprob

重要性比值需要明确 numerator 和 denominator：一个来自正在优化的策略，另一个来自产生动作的行为策略或算法约定的旧策略。重新用当前权重算一遍，并不自动等价于生成时的行为 logprob。

温度、top-p/top-k、tokenizer/template、MoE 路由、精度和参数版本都可能影响对齐。返回更多 metadata 有助于追溯，但不保证恢复“完整随机性”或严格 on-policy；应按实际采样分布和算法定义验证。


## 权重同步的正确性要拆成三层

| 层次 | 要验证什么 | 例子 |
|---|---|---|
| 表示 | 名字、shape、dtype、shard 是否正确 | QKV 拼接顺序、TP 分片往返 |
| 发布 | 所有需要的 rank 是否完成，何时恢复请求 | 更新失败时不能仅更新 version 标签 |
| 语义 | 请求、KV、采样配置与权重是否匹配 | 旧 prefix KV 是否需要失效 |

暂停生成、abort 和换权出现在同一接口中，不代表跨进程事务或失败自动回滚。当前源码保证与生产层希望达到的目标必须分开写。


## 局部数值一致与长期质量

小 batch 对齐用于发现 shape、mask、缩放与版本错误；若更改了异步程度或采样策略，还需要更长运行检查 reward/评估质量。短程 loss 正常不能证明长期训练等价，吞吐增加也不等于 time-to-quality 改善。


## 目标与状态如何同时闭合

两个 rank 的 token 数不同，先求局部平均再平均 rank 就会改变 token 的权重。对一条包含 prompt、回答、EOS 和 padding 的轨迹，token、logprob 和 mask 的对齐决定分子和分母实际在统计什么。

权重发布成功时，请求与缓存需要进入相容版本；中途失败则可能留下部分已更新的资源。数值容差只能解释舍入误差，不能替代 token 身份和版本不变量。性能判断还应分别说明有效产出和长期质量。

源码连接：[算法实现](../13_Source_Analysis/slime/05_rl_algorithms.md)、[权重转换](../13_Source_Analysis/slime/13_megatron_bridge_internals.md)、[Megatron 边界](../13_Source_Analysis/megatron/06_reference/03_source_questions.md)、[SGLang 换权](../13_Source_Analysis/sglang/04_interfaces_and_models/03_control_plane_and_post_training.md)。

## 完整推演

[轨迹到loss完整推演](<RL算法与系统接口.md#从轨迹到目标token对齐grpo组统计与ppo裁剪>)：具体位置、概率、优势、mask和梯度。


---


<!-- series-nav -->

[上一篇：RL 系统全景：角色、状态与执行阶段](<RL系统全景.md>) · [专题目录](<README.md>) · [下一篇：Ray、队列与背压：从远程调用走到 RL 调度](<Ray与队列调度.md>)

<!-- /series-nav -->
