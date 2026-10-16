# 05 RL 算法实现：从 logprob 到 loss 的完整链路

相关基础：[Ray、队列与前置系统](<../../../06_Post_Training/27_权重发布异步与资源调度.md#s4>)。


> 对应综述（`00_rl_infra_survey.md`）§2.6「RL 算法」。
> 涉及文件：`slime/backends/megatron_utils/loss.py`（流程编排）、`slime/utils/ppo_utils.py`（算法原语，移植自 OpenRLHF）、`slime/backends/megatron_utils/cp_utils.py`（Context Parallel 支持）。
> 本篇按"训练一步时数据如何流动"的顺序讲，每个公式都配数值例子。

---

### 先建立学习问题与符号

普通分类训练有“正确类别”，而一条工具轨迹通常只有最终成功/失败。我们要让获得较好反馈的动作在相似上下文中更可能发生，同时避免一次更新把分布推得过远。这就需要把 **奖励 → 相对优势 → 概率变化 → 可求导损失** 连起来。

本篇用 $x$ 表示 prompt，$y_t$ 表示第 t 个 response token，$s_t=(x,y_{<t})$ 表示生成它之前的上下文（含已出现的工具观察）。策略 $\pi_\theta(y_t\mid s_t)$ 是参数为 $\theta$ 的模型选该 token 的概率。概率无量纲，logprob 使用自然对数，信息量单位称 nat；reward 的尺度由任务定义，若用成功=1/失败=0，则可视为无量纲分数。未经标准化的 value/advantage 与 reward 同尺度，组内标准化后的 advantage 无量纲。

| 符号 | 含义 | 在代码里常见的位置 |
|---|---|---|
| $\mu$ | 实际采样动作的行为分布，含实际采样处理和权重版本 | `rollout_log_probs`，但须核对端点返回语义 |
| $\pi_{old}$ | 本次 PPO 更新选用的基准策略 | 缓存 `batch["log_probs"]`，或按开关使用 rollout 概率 |
| $\pi_\theta$ | 当前可求导策略 | 本次训练 forward 算出的 `log_probs` |
| $\pi_{ref}$ | KL 正则的参考策略 | `ref_log_probs`；通常在一个训练阶段保持冻结 |
| $m_t\in\{0,1\}$ | 该 response 位置是否计入直接损失 | `loss_masks` |
| $A_t$ | 相比基线，这个动作应被鼓励还是抑制 | `advantages` |

不要因为参数值一样就把 $\mu$、$\pi_{old}$ 和 $\pi_\theta$ 的计算路径合并。推理的精度、温度、候选裁剪以及 tokenizer/context 都会影响概率；另一方面，“old”是本次优化的参照，并不一定是最初的 reference 模型。

## 1. 总览：loss 的计算位置

Megatron 的 train step 里，模型 forward 得到 logits `[1, T, V]`（T = 拼接后的总 token 数，V = 词表），随后调用：

```
train_one_step
  └─ loss_function(args, batch, num_microbatches, step_global_batch_size, logits)   loss.py:1220
       ├─ get_sum_of_sample_mean(...)                        # 归一化器
       ├─ policy_loss_function  (loss_type="policy_loss")    loss.py:881   ← actor
       ├─ value_loss_function   (loss_type="value_loss")     loss.py:1113  ← critic
       └─ sft_loss_function     (loss_type="sft_loss")       loss.py:1170
```

在进入具体 loss 前，先看两个"地基"：logprob 怎么从 logits 里抠出来（§2），advantage 怎么从 reward 造出来（§3）。

---

## 2. 地基一：per-token logprob 的计算

### 2.1 为什么不在 rollout 时直接用推理 logprob？

`--use-rollout-logprobs` 可以让记录的 rollout 概率充当旧策略分母，但不能替代可求导的当前概率。当前策略仍需 forward；Python 数值或 detached tensor 中保存的历史 logprob 没有连接到当前模型参数的计算图，无法提供这次梯度。难点：batch 里多条样本拼接成一维序列（packing），还要兼容 TP（词表被切分）与 CP（序列被切分）。

先在单卡算一次。词表只有四个 token，模型对某位置输出 logits（未归一化实数评分）

$$
z=[0,\ln2,\ln3,\ln4].
$$

softmax 将它变成概率：$p_j=e^{z_j}/\sum_k e^{z_k}$。指数分别为 `[1,2,3,4]`，分母为 10，因此 $p=[0.1,0.2,0.3,0.4]$。若实际生成第 3 个 token（从 0 编号为 2），它的 logprob 是 $\ln0.3=\ln3-\ln10\approx-1.2040$。logits `[4]`、概率 `[4]`，最终只取出的 logprob 是一个标量。

为什么用 logprob？一条长回答的概率是各步条件概率的连乘，小数连乘易下溢；取对数后变成相加，并且对 logits 的导数简单：

$$
\frac{\partial\log p_a}{\partial z_j}
=\frac{\partial(z_a-\log\sum_k e^{z_k})}{\partial z_j}
=\mathbf1[j=a]-\frac{e^{z_j}}{\sum_k e^{z_k}}
=\mathbf1[j=a]-p_j.
$$

例如最小化 $-A\log p_a$，且 $A>0$，对已选 token 的梯度是 $-A(1-p_a)<0$，梯度下降会增加它的 logit；对其它 token 的梯度是 $Ap_j>0$，会降低它们的 logit。这就是“好回答提高概率”的具体计算来源。

temperature（温度）$\tau>0$ 用 $z/\tau$ 替代 z；$\tau=0$ 通常由采样器单独处理为贪心选择，不能在此公式里直接除以零。若 top-p 回放留下集合 `{2,3}`，同一组 logits 在该支持集合内的概率变成 `[3/7,4/7]`，已选 token 2 的 logprob 从 $\log(3/10)$ 变成 $\log(3/7)$。把前者与后者当同一分布计算比例会制造人为偏差。

### 2.2 `get_log_probs_and_entropy`（loss.py:470-561）

关键设计：**在整个 `[T, V]` 上一次性算完，再按样本切回**，让 backward 只遍历一次大 tensor：

1. **温度对齐**（loss.py:497-500）：`logits /= rollout_temperature`。采样时用了温度，训练侧算 logprob 必须除以同一个温度，否则训推不可比；
2. **shifted tokens**（`_build_shifted_tokens`，loss.py:230-280）：语言模型是"用前 t-1 个 token 的 logits 预测第 t 个"，所以目标 token 要左移一位；
3. **top-p 掩码**（`_build_topp_keep_mask`，loss.py:306-386）：rollout 用 top-p 采样时，训练侧只在相同的候选核内算 logprob（02 篇的回放数据在此消费），否则训推分布不同；
4. **`calculate_log_probs_and_entropy`**（ppo_utils.py:746）：TP 感知的 logsumexp——词表按 TP 切分，每个 rank 只有 `V/tp` 的 logits，需要跨 TP 组归约（`_VocabParallelLogProbEntropy` 自定义 autograd Function，ppo_utils.py:187）；`log_probs_chunk_size` 分块防 OOM；
5. **按样本切回**（`_extract_per_sample`，loss.py:389-467）。

再核对一次 shifted target。`tokens=[10,11,20,21]`，前两项是 prompt、后两项是 response。位置 1 的 logits 预测 20，位置 2 的 logits 预测 21，所以 response 用 logits 切片 `[1:3]` 配 tokens `[2:4]`。最后位置 3 的 logits 预测的是尚未记录的下一个 token，不能拿来配 21。

设两条完整序列长度为 4 和 3，packing 后总长 $T=7$，单卡逻辑 logits 是 `[1,7,V]`，各条 response 概率仍应分别切回 `[R_0]` 和 `[R_1]`。TP=q 时本地词表维通常是 `V/q`（按实际 padding 后大小），CP=c 时本地 token 维也会分片；文章中的全局 `[1,T,V]` 不是每个 rank 都常驻的真实大小。

### 2.3 Context Parallel 的三种布局

长序列训练时 CP 把序列切到多卡，slime 支持三种（loss.py:96-148 的 `get_responses` 里体现）：

- **cp=1**：不切，直接切片 `[start-1:end-1]`；
- **zigzag CP**（默认 ring-attention 布局）：每条序列被切成"首尾配对"的两段分在两个 rank 上，取 response 段要用 `get_logits_and_tokens_offset_with_cp` 算两套 offset 再拼接（loss.py:125-144）；
- **allgather CP**（`--allgather-cp`）：先 all-gather 成全局序列再连续切，取回时用一次**可微分 all-reduce** 重建全量 response 再切回 zigzag（`_allgather_cp_redistribute`，loss.py:151-227）。

这段代码注释里有句大实话：`# TODO: this is super ugly... do better abstraction.`（loss.py:126）——CP 与 packing/loss 的笛卡尔积是 RL infra 里最易错的区域之一，读代码时抓住"全局位置 ↔ 本地位置"的换算即可。

### 2.4 深入拆解：`_VocabParallelLogProbEntropy`（`slime/utils/ppo_utils.py:187-316`）——TP 下如何"跨卡算 softmax 却只传一个标量"

这是全代码库里最需要"手推公式"才能看懂的一个自定义 autograd Function。背景：TP 把词表 `V` 切成 `tp_size` 份，每个 rank 只持有 `logits[:, rank*V/tp : (rank+1)*V/tp]`，而 `log_softmax(logits)[target] = logits[target] - logsumexp(logits)` 里的 `logsumexp` 需要**全词表**才能算对——直接 all-gather 整个 `[T, V]` logits 到每个 rank 会爆显存（`V` 常常是几万到十几万），所以要设计一个"只传必要的几个标量、大头计算量分布式做"的算法。

**forward 的三段式**（对应 Megatron 原生 `vocab_parallel_cross_entropy` 的思路，但这里额外支持 top-p 掩码和熵）：

1. **判断 target 是否在本 rank 词表范围内**（`target_mask`，205-207 行）：`target_mask=True` 表示这个 token 的正确答案不在我这个 rank 的词表分片里，此时先把 `masked_target_1d` 强行置 0（避免后面 gather 越界），最后再用 `masked_fill_(target_mask, 0.0)` 把这些位置的贡献清零；
2. **`vocab_parallel_softmax` 闸门函数**（210-229 行）：先 `all_reduce(MAX)` 拿全局最大值做数值稳定（跨 rank 的 max 才是真正的全局 max），减完之后 `exp_()`（就地操作省显存）算局部 `sum_exp_logits`，再 `all_reduce(SUM)` 拿到全局分母——**这里只通信了两个标量（每个 token 一个 max 值 + 一个 sum 值），而不是整个 logits**，这是"分布式 logsumexp"的标准写法：`logsumexp(x) = max(x) + log(sum(exp(x-max(x))))`，max 和 sum 都可以先局部算再跨 rank 归约；
3. **`predicted_logits` 的全局归约**（273-274 行）：每个 token 的正确答案只落在某一个 rank 上，其余 rank 该值已被清零，`all_reduce(SUM)` 后自然只有那个 rank 的贡献生效——**这是一种"用 all_reduce 实现跨 rank 的 gather"的技巧**：既然只有一个 rank 非零，sum 就等价于 gather。

**熵的计算复用同一套 softmax**：`entropy = logits_max + log(sum_exp) - sum(softmax * logits)`（249 行），这是熵 `H = -Σp·log(p)` 展开后代入 `log(p) = logit - logsumexp` 的等价式，好处是不需要再单独算一次 `log(softmax)`。

**backward 为什么要 `ctx.mark_non_differentiable(entropy)`**（278 行）：当 `entropy_coef=0`（不参与 loss）时，熵只是用来打日志的指标，`with_entropy_grad=False`，此时**故意不保存计算熵梯度所需的 `[T,V]` 大张量**（`saved_entropy_softmax` 等在 283-287 行被替换成 `new_empty((0,))` 空张量）——这是显式的"反向传播成本裁剪"：一个 `[T,V]` 的 float32 张量在 `T=4096, V=150000` 时就是 2.4GB，如果熵不参与梯度就不该白白占着显存等 backward。

**一个数字直觉**：4 卡 TP，词表 15 万，某 rank 持有 `[T, 37500]` 的 logits 分片；对每个 token，通信量是 2 次 all_reduce（各 1 个标量/token）+ 1 次 all_reduce（predicted_logits，1 个标量/token）= 3 个标量/token 的跨卡通信，而不是把 37500 维的分片传来传去——这减少了对词表维度的通信依赖，但实际是否成为瓶颈仍取决于序列长度、归约次数和网络延迟。

把上一小节的四词词表分给两卡：rank 0 持有 `[0,ln2]`，rank 1 持有 `[ln3,ln4]`，本地形状都是 `[1,2]`。

1. 两卡局部最大值是 `ln2` 和 `ln4`，MAX 归约得到 $M=\ln4$。
2. 减 M 后指数分别为 `[1/4,2/4]`、`[3/4,1]`，局部和是 0.75 和 1.75，SUM 归约得到 $S=2.5$。
3. 全局 logsumexp 为 $M+\ln S=\ln4+\ln2.5=\ln10$。
4. target=2 只在 rank 1 上，rank 0 对 target 贡献 0，SUM 取出 $z_2=\ln3$，得到同样的 $-1.2040$。

实际通信的是长度为 T 的统计向量，不是整批只传一个数字；熵和 top-p 重归一化还可能增加归约，不能把三次归约视为所有配置的固定总数。优势是统计向量规模随 T 增长，而全 logits 大小随 $T\times V$ 增长。小 batch 或网络延迟较高时，归约仍可能成为瓶颈。

熵公式也可逐步展开：

$$
H=-\sum_jp_j\log p_j
=-\sum_jp_j(z_j-\log Z)
=\log Z\sum_jp_j-\sum_jp_jz_j
=\log Z-\sum_jp_jz_j,
$$

其中 $Z=\sum_j e^{z_j}$，利用了 $\sum_jp_j=1$。四词例子的熵约为 1.2799 nat。若熵只用来记录指标，反向无需保存它专用的中间值；若损失含熵正则，则这些中间值必须活到 backward，不能提前释放。

### 2.5 zigzag CP 的一个切分例子

假设一条序列长度 8（token 0-7），`CP=2`。zigzag（Ring Attention 的标准切法）不是简单地"前 4 个给 rank0，后 4 个给 rank1"（这样 rank0 全是"简单"的早期 token、rank1 全是"难"的后期 token，causal attention 下两个 rank 的计算量严重不均衡），而是"掐头去尾配对"：

```
全局位置:   0  1  2  3  4  5  6  7
zigzag 分配: 0  0  1  1  1  1  0  0    (分成 2×CP=4 个等长块，rank r 取块 r 和 3-r)
```

rank 0 拿 `[0,1,6,7]`，rank 1 拿 `[2,3,4,5]`。在朴素 causal attention 的成对计算计数下，rank 0 为 `1+2+7+8=18`，rank 1 为 `3+4+5+6=18`，比前后连续平分的 `10` 与 `26` 更均衡。具体 offset 还须计入 padding 和预测位置平移，见 `get_logits_and_tokens_offset_with_cp`。取 response 部分时必须知道这种"跳跃式"映射关系，才能把两个 rank 各自算出的 logprob 正确拼回原始顺序——这也是为什么 `allgather_cp` 模式提供了一个"先拼成正常顺序、算完再拆回 zigzag"的旁路（用一次通信换取代码简单性，用于调试或不关心额外通信开销的场景）。

---

## 3. 地基二：advantage 的构造

`compute_advantages_and_returns`（loss.py:661-828）是算法选择的总开关。只在 pipeline 最后一段执行（697 行），因为只有那里有 logits。

### 3.1 先算 KL

```700:712:slime/backends/megatron_utils/loss.py
    if args.kl_coef == 0 or not log_probs:
        xs = log_probs or rollout_log_probs or values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]
```

`compute_approx_kl`（ppo_utils.py:12-51）实现 Schulman 博客的三种估计器：

| 类型 | 公式 | 性质 |
|---|---|---|
| k1 | $d = \log\pi_\theta - \log\pi_{ref}$ | 对应分布采样时无偏，单项可为负 |
| k2 | $d^2/2$ | 非负，通常有偏 |
| k3 | $\exp(-d)-1+d$ | 非负；无偏性依赖采样分布、共同支持和未截断 |

`importance_ratio` 参数（loss.py:1057-1058 的 `--use-unbiased-kl`）对应 DeepSeek-V3.2 的无偏 KL 修正：KL 乘以 IS ratio。

KL（Kullback–Leibler divergence，分布间的相对熵）不是单个 token 的 log 比，而是其期望。固定上下文，令 $p=\pi_\theta$、$q=\pi_{ref}$：

$$
D_{KL}(p\|q)=\sum_a p(a)\log\frac{p(a)}{q(a)}.
$$

若 $a\sim p$，k1 的期望就是该式。k3 多加了一项 $q(a)/p(a)-1$；当支持集合相容时，它的期望为 $\sum_a q(a)-\sum_a p(a)=0$，所以不改变期望。令 $u=q(a)/p(a)>0$，则 k3 为 $u-1-\ln u\ge0$。这个不等式保证每个样本非负，并不保证所有分布下方差都更小。

取 $p=[0.75,0.25]$、$q=[0.5,0.5]$：两动作 k1 分别为 `ln1.5≈0.4055`、`ln0.5≈-0.6931`，期望 $0.75\times0.4055+0.25\times(-0.6931)\approx0.1308$。k3 分别为 `0.0721` 和 `0.3069`，按同一 p 加权也得 0.1308。若动作实际从 q 采样，直接平均 k1 就变成约 -0.1438，已经不是 $D_{KL}(p\|q)$。这解释了为什么代码还提供重要性权重。

当前 `low_var_kl` 在 k3 之后还有 `[-10,10]` 数值截断；截断触发时不能继续声称严格无偏。`kl_coef` 用于构造奖励/优势的路径，`kl_loss_coef` 用于显式 loss 正则；两处系数与 `use_kl_loss` 的开关不是同一个变量，应沿实际分支核对。

### 3.2 五种 advantage 估计器

**GRPO（组内相对）**——当前 RLVR 主流。同一 prompt 的 N 个采样 reward 直接广播到每个 token（ppo_utils.py:361-368）：

```361:368:slime/utils/ppo_utils.py
def get_grpo_returns(rewards, kl):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

**数值例子**：prompt "1+1=?" 采 8 条，reward = [1,1,1,0,1,0,1,1]。GRPO 的"组内归一化"发生在 reward 计算侧（rollout 端对组内 reward 减均值除标准差，见 `--advantage-estimator` 相关 RM 后处理），这里拿到的 rewards 已是相对值；`kl_coef=0` 时 advantage 就是常数广播——好回答的每个 token 都被同等鼓励。**这就是 GRPO 省掉 critic 的本质：用组内统计量替代 value 网络。**

GAE（Generalized Advantage Estimation，广义优势估计）把多步时序差分残差按 (γλ) 的幂加权合并，用来估计动作相对价值基线的优势。终止与时间截断会决定是否继续递推和 bootstrap。

**PPO（GAE）**：critic 给每个 token 估值，`vanilla_gae/chunked_gae`（ppo_utils.py:579/603）按 $ \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) $、$ A_t = \sum_l (\gamma\lambda)^l \delta_{t+l} $ 递推；reward 被加在最后一个 token 的 KL 惩罚项上（loss.py:727-738）。

**REINFORCE++**：token 级 reward = `-kl_coef * kl`，序列 reward 加在最后一个有效 token（`last_idx`，ppo_utils.py:418-419），再按 $ G_t = r_t + \gamma G_{t+1} $ 倒推（421-426）；`reinforce_plus_plus_baseline` 则是 `(reward - 组基线)` 直接广播（441-468）。

**GSPO / CISPO**：advantage 与 GRPO 相同（loss.py:720-724 归为同一分支），差异在 loss 的比率定义（§4.2）。

#### 手算 GRPO：基线为什么能提供方向

一题两条回答奖励 `[1,0]`，均值 $\bar r=(1+0)/2=0.5$。代码默认样本标准差采用分母 $G-1$，其中 $G=2$：

$$
s^2=\frac{(1-0.5)^2+(0-0.5)^2}{2-1}=0.5,
\qquad s=\sqrt{0.5}\approx0.7071.
$$

优势 $A_i=(r_i-\bar r)/(s+\epsilon)$，$\epsilon=10^{-6}$ 防止零分母，故约为 `[0.7071,-0.7071]`。若两条 response 长度分别是 3、2，广播后优势形状为 `[3]`、`[2]`，各自填同一个标量；其中工具位置仍由 mask 排除。对 `[1,1]`，减均值后全零；对单样本组 `G=1`，带 Bessel 校正的 std 无定义，不能把它当有效的相对比较组。

这种信号把整条结果分摊给动作，并不证明答对轨迹的每一步都正确；它是低成本的信用分配估计，省掉 critic，却依赖同题多次采样和有区分度的反馈。

#### 手算 GAE：为何要反向递推

PPO 中 value $V(s_t)$ 估计从当前上下文开始的折扣回报。设两步奖励 $r_0=0,r_1=1$，价值 $V(s_0)=0.2,V(s_1)=0.4$，真正终止后 $V(s_2)=0$。折扣因子 $\gamma=1$、混合系数 $\lambda=0.5$ 都无量纲。

先算时序差分残差：

$$
\delta_0=0+1\times0.4-0.2=0.2,\qquad
\delta_1=1+1\times0-0.4=0.6.
$$

从终点开始，$A_2=0$，再用 $A_t=\delta_t+\gamma\lambda A_{t+1}$：

$$
A_1=0.6,\qquad A_0=0.2+0.5\times0.6=0.5.
$$

value 的回归目标为 $\widehat R_t=A_t+V(s_t)$，得到 `[0.7,1.0]`。一般式由不断代入下一步得到 $A_t=\delta_t+(\gamma\lambda)\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots$，所以 $\lambda=0$ 只看一步残差，增大 $\lambda$ 会引入更远的反馈，也更依赖后续轨迹。

这个例子假设真实终止。若只是因长度上限截断，环境并未结束，合理的 bootstrap（用尾部价值接上未观测未来）与直接设零不同。阅读当前 GAE 实现时要检查尾部 value 和 mask 的具体处理，不能从 `TRUNCATED` 字段存在就推断自动采用正确的时间截断语义。

### 3.3 全局白化

`--normalize-advantages` 时（loss.py:776-825），把整个 DP 组的 advantage 拼起来做 masked 白化（减均值除标准差）——`distributed_masked_whiten` 跨 DP 组聚合统计量，mask 保证只统计有效 token。注意 CP 下 mask 也要按相同 zigzag 切分（782-810 行）。

### 3.4 OPD：on-policy distillation 的 advantage 修正

`apply_opd_kl_to_advantages`（loss.py:620-658）：student 生成轨迹、teacher 打分（teacher_log_probs 由 rollout 侧用 `max_new_tokens=0` 前向取得），advantage 减去 `opd_kl_coef × (student_logp - teacher_logp)` 作为逐 token 蒸馏信号。student/teacher log 比可正可负，只有在明确采样分布后，其期望才对应相应 KL；这个修正可与前面选用的 advantage 估计组合。

---

## 4. Policy loss：PPO / GSPO / CISPO

`policy_loss_function`（loss.py:881-1110）：

### 4.1 PPO clip（基准）

```125:148:slime/utils/ppo_utils.py
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
```

`ppo_kl = old_log_probs - log_probs`（loss.py:976），故 `ratio = exp(log_probs - old_log_probs)` = 新旧策略比值。`eps_clip_high` 与 `eps_clip` 分离即 DAPO 的 **clip-higher**（上限更大，鼓励探索）；`eps_clip_c` 是 dual-clip PPO。

从最大化收益到最小化 loss，中间只差一个负号。令 $\rho=\pi_\theta(a\mid s)/\pi_{old}(a\mid s)=e^{\ell_\theta-\ell_{old}}$，这里 $\ell$ 为 logprob，$\rho$ 无量纲。PPO 使用

$$
L=-\min(\rho A,\operatorname{clip}(\rho,1-\epsilon_l,1+\epsilon_h)A)
=\max(-\rho A,-\operatorname{clip}(\rho)A).
$$

`maximum` 是负号把 `minimum` 反过来的结果。取对称区间 `[0.8,1.2]`、$A=\pm1$：

| A | 比率 $\rho$ | 原项 $-\rho A$ | 裁剪项 | 最终 loss | 该位置的 PPO 比率梯度 |
|---|---:|---:|---:|---:|---|
| +1 | 1.5 | -1.5 | -1.2 | -1.2 | 平台区，继续提高概率不再增加收益 |
| +1 | 0.5 | -0.5 | -0.8 | -0.5 | 保留，纠正好动作概率下降 |
| -1 | 0.5 | +0.5 | +0.8 | +0.8 | 平台区，继续降低概率不再增加收益 |
| -1 | 1.5 | +1.5 | +1.2 | +1.5 | 保留，纠正坏动作概率上升 |

所以“ratio 超出区间就没有梯度”是错误的；要看 A 的符号和实际被选中的分支。代码的 `clipfrac` 是裁剪分支严格胜出的比例，也不等同于所有越界比率的比例。共享参数还会受其它 token 的梯度影响，PPO clip 不是参数距离或 KL 的硬约束。

若优化前 old 与 new 数值相同，$\rho=1$，仍可有梯度：分母停止求导，分子对参数求导，$\nabla_\theta e^{\ell_\theta-\operatorname{stopgrad}(\ell_\theta)}=\nabla_\theta\ell_\theta$。若把分母也连入同一计算图，两项才会互相抵消。

### 4.2 GSPO：序列级重要性比率

MoE（mixture of experts，混合专家）每个 token 可选择不同专家，token 级概率比可能出现较大波动；这不是所有 MoE 配置必然不稳定的结论。GSPO（Group Sequence Policy Optimization，组序列策略优化）先平均一条序列上有效 token 的新旧 logprob 差，再取指数得到序列比率。CP 分片时需先汇集相应序列信息，再把序列级结果广播回 token 位置（`compute_gspo_kl` 与 `policy_loss_function`）。平均 log 比与概率比不能直接互换。

设有效 token 数 $N=\sum_t m_t$，当前实现的序列比率为

$$
\rho_{seq}=\exp\left(\frac{\sum_t m_t(\ell_{\theta,t}-\ell_{old,t})}{N}\right)
=\left(\prod_{t:m_t=1}\rho_t\right)^{1/N}.
$$

它是 token 比率的几何平均。两 token 的比率为 `[2,0.5]` 时，序列比率为 $\sqrt{2\times0.5}=1$，算术平均却是 1.25；二者不能混用。完整回答概率之比是乘积 1，本例恰好一样；若两个比率都是 2，完整乘积为 4，GSPO 比率为 2。长度归一化改变了目标，不能称它为未经修改的整序列重要性权重。它减少同一序列内裁剪决策的差异，也会让不同 token 的正负偏移互相抵消。

### 4.3 CISPO：clip 权重而非比值

```167:171:slime/utils/ppo_utils.py
    ratio = (-ppo_kl).exp()
    ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
    pg_losses = -ratio_truncated.detach() * advantages * log_probs
```

MiniMax-M1 的做法：比率 clamp 后 **stop-gradient** 当权重，梯度从 `log_probs` 流过。CISPO 的该项梯度是 `-clipped_ratio × advantage × ∇logprob`；即使比率被裁剪仍可有梯度，而 PPO 只有进入所选裁剪平台区的对应项梯度为 0。

### 4.4 OPSM：序列级拒采

`compute_opsm_mask`（ppo_utils.py:54-92）：**负 advantage 且序列 KL 超过 `opsm_delta` 的序列整段 mask 掉**——策略已明显偏离旧策略的"坏样本"不再贡献梯度，是 off-policy 防护的另一种形态。

### 4.5 完整数字例子：一组 GRPO 样本从 reward 走到 policy loss

设一个 prompt 采样 8 条（`n_samples_per_prompt=8`），规则奖励 `reward = [1,1,0,0,1,0,1,1]`（1=答对，0=答错），`kl_coef=0`（不做 KL 惩罚，纯 GRPO）：

1. **组内归一化**（发生在 `RolloutManager._post_process_rewards`，而非 `loss.py`）：`mean=0.625`。PyTorch `std()` 默认做 Bessel 校正，样本标准差约为 `0.518`，所以归一化后 `reward_norm ≈ [0.7246, 0.7246, -1.2076, -1.2076, 0.7246, -1.2076, 0.7246, 0.7246]`（实际分母还有 `+1e-6`）——这一步是“GRPO 省 critic”的关键；
2. **广播到 token**（`get_grpo_returns`）：每条样本内所有 response token 的 advantage 都等于该样本的 `reward_norm`（例如样本 0 的 5 个 response token全为约 `0.725`）；
3. **`ppo_kl = old_log_probs - log_probs`**：假设更新前冻结的训练评分为 `old_log_probs=-0.7`，当前可求导评分为 `log_probs=-0.5`，则 `ppo_kl=-0.2`。只有启用 `use_rollout_logprobs` 等相应路径时，分母才直接来自生成记录；默认缓存评分、实际行为评分与当前评分要按篇首的符号表分别追踪；
4. **`ratio = exp(-ppo_kl) = exp(0.2) ≈ 1.221`**：新策略比旧策略更倾向于生成这个 token（比值 >1）；
5. **PPO-clip**（设 `eps_clip=0.2, eps_clip_high=0.28`）：`ratio` 落在 `[0.8, 1.28]` 内，`pg_loss = -1.221 × 0.725 ≈ -0.885`，梯度推动模型提高该 token 概率；
6. **样本 2**（答错，advantage≈-1.2076）：若其某 token 的 `ratio≈1.22` 同样未被裁剪，`pg_loss ≈ 1.473`，梯度降低该 token 概率；
7. **归一化求和**（§6）：8 条样本的所有 token loss 按"每条样本内部先按 token 数取平均，再对样本取平均"（或选择 `--calculate-per-token-loss` 的有效 token 归约；拆分轨迹还需按 rollout 归约）汇总成一个标量，反传给 Megatron 做梯度累积。

**直觉**：GRPO 把"这条样本比组内平均水平好还是差"变成一个恒定的乘数，PPO-clip 再把"这一步参数更新是不是走得太快"这件事通过 ratio 裁剪限制住——两者一个负责"往哪个方向走"，一个负责"这一步别走太远"。

---

## 5. TIS：训推数值不一致的修正

推理精度、kernel、采样处理和权重陈旧度都可能使 rollout 评分与训练评分不同。下列代码对记录的评分比率做截断加权；它是否对应严格的行为重要性比率，要核对两个字段的概率语义及支持集合，不能只凭“同参数”或字段名判断。通用定义与适用条件见[四份概率与行为校正](<../../../06_Post_Training/27_权重发布异步与资源调度.md#s2>)。

TIS（Truncated Importance Sampling，截断重要性采样）对目标策略与行为策略的概率比进行截断或限制，以控制大权重。它可降低权重波动，但截断可能引入偏差，不能自动修复动作或上下文错位。

本篇代码块保留走读时的行号。2026-10-12 核对当前工作区：`policy_loss_function` 为 TIS 优先取缓存 `batch["log_probs"]`，缺失时取当前评分的 detached 值；PPO 分母还受 `use_rollout_logprobs` 控制。因此当前 detached 评分不应被一律称作冻结 old。请按符号追踪调用链，区分快照说明与当前路径。

```840:852:slime/backends/megatron_utils/loss.py
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    ...
    pg_loss = pg_loss * tis_weights
```

`icepop_function`（loss.py:855-878）是变体：超出 `[tis_clip_low, tis_clip]` 的 token 权重直接置 0（拒绝而非截断）。配套观测指标：`tis / tis_clipfrac / tis_abs / ois`（loss.py:1093-1100），以及长期健康指标 `train_rollout_logprob_abs_diff`（loss.py:1073-1077）——这个值爬升需要结合采样处理、版本差和跨引擎同权重评分对照定位。`--custom-tis-function-path` 可替换自定义修正策略（loss.py:1011-1014）。

---

### 5.1 从换测度公式理解 TIS 的收益与边界

固定上下文，对任意动作函数 $f(a)$，若目标策略 p 的非零概率动作都在行为策略 $\mu$ 的支持集合内，则

$$
\mathbb E_{a\sim p}[f(a)]
=\sum_a p(a)f(a)
=\sum_a\mu(a)\frac{p(a)}{\mu(a)}f(a)
=\mathbb E_{a\sim\mu}[w(a)f(a)].
$$

这就是重要性采样，不是凭经验添加的放大系数。取 $\mu=[0.5,0.5]$、$p=[0.8,0.2]$、$f=[1,0]$：真实期望为 0.8；权重 `[1.6,0.4]` 给出 $0.5\times1.6\times1+0.5\times0.4\times0=0.8$。若将权重上限截为 1.2，估计期望变为 0.6，波动受到限制，但引入了偏差。

此式只对固定上下文的动作分布做变换。token 权重通常不等于对整条历史状态分布完成校正；若 $\mu(a)=0,p(a)>0$，行为样本永远看不到该动作，权重也无法凭空补出它。旧 token 的 mask、top-p 回放、TIS 因此解决不同问题，不能互相替代。

在实现中，PPO 比率可能是 `new/old`，TIS 是 `train_score/rollout_score`。只有 `train_score` 恰好是冻结 old，且不做裁剪、额外 mask、目标变换等操作时，两个原始比率才可形式上相乘为 `new/rollout`；不能直接把代码里的两次裁剪折叠成一次。

## 6. 归一化与缩放（容易踩坑的部分）

`loss_function`（loss.py:1220-1320）最后做两件"会计工作"：

1. **归一化器**：`get_sum_of_sample_mean` 按"先每样本对有效 token 求均值、再对样本求和"规约；`--calculate-per-token-loss` 时改为全局 token 数归一（DAPO 的 token-level loss，长短不一的样本不再被平均权重拉平）；
2. **Megatron 梯度累积对齐**（loss.py:1289-1298）：per-rollout-mean 模式下要乘 `num_microbatches / step_global_batch_size * dp_size` 抵消 Megatron 内部的平均；per-token 模式乘 `cp_size` 抵消 CP 维度的平均。注释（1289 行）明言"divide by cp_size to cancel the multiply in Megatron"。

先手算为什么“求平均”也是算法定义。两条样本的有效 token loss 分别为 `[2]` 和 `[0,0,0]`：每样本先平均再平均，得到 $(2+0)/2=1$；按全体 token 平均，得到 $2/4=0.5$。前者让每条轨迹等权，后者让每个有效 token 等权。

若同一次 agent 执行拆成两个 sibling 样本，当前 per-rollout 归约还会用共同 `rollout_id` 和 `rollout_mask_sums`，先在一次执行内归约，再按执行数缩放。不能因为输出从一条拆成两条就使这次探索权重翻倍。默认每条样本对应独立 rollout 时才退化成通常的 per-sample mean。

DP=2 时，两个副本若分别有 1 和 3 个有效 token，先各自平均再把两个均值平均会得 1，而全 token 目标应为 0.5。正确实现必须合并分子、分母并匹配框架的梯度平均，不能只对已经归一化的 loss 再平均。micro-batch 梯度累积、DP 平均、CP 分片都参与这份账本；loss 缩放不正确会像学习率被悄悄改了一样。

另有三个"防呆"细节：

- `log_probs.numel() == 0` 时 `loss += 0 * logits.sum()`（loss.py:1069-1071）：某 rank 全是 padding 时强制 autograd 走完整图，否则 CP 通信的 backward 不触发、**其他 rank 死锁**（1281-1287 行的注释详细解释了 allgather-CP 场景）；
- `--recompute-loss-function`：loss 计算也做 activation checkpointing 省显存（1276-1277）；
- metric 以 `[count, metric1, ...]` 的 tensor 形式返回（1312-1318），由上层跨 DP all-reduce 后以 count 为分母出报表。

---

## 7. Critic 与 SFT

- `value_loss_function`（loss.py:1113-1167）：PPO 式 value clip——`max((clip(V)-R)², (V-R)²)`；
- `sft_loss_function`（loss.py:1170-1217）：response 段的负 log 似然，与 RL 共用同一套 packing/CP 管线——slime 的 SFT 只是"没有 RL 信号的特例"，这正是"SFT 与 RL 同构"框架观的代码体现（配合 `slime/rollout/sft_rollout.py` 的伪 rollout）。

---

## 8. 小结：一张调用链图

```
logits [1,T,V]
  → get_log_probs_and_entropy（温度对齐 / top-p 掩码 / TP 归约 / CP 布局）
  → compute_advantages_and_returns（KL(k1/k2/k3) → GRPO|PPO-GAE|R++|R++-baseline
     → OPD 修正 → DP 白化）
  → policy_loss_function（PPO-clip | GSPO 序列级 | CISPO 权重裁剪 | OPSM 掩码
     → TIS 修正 → KL loss 项 → entropy 项）
  → loss_function（归一化 + Megatron 缩放 + 指标打包）
```

`--advantage-estimator` 选择的目标会改变 clipping 与归一化语义；`pg_clipfrac`、`ppo_kl` 和 `tis` 分别观察不同关系。比较 GRPO、GSPO、CISPO 时，指标必须放回各自目标定义，不能只凭数值大小判断优劣。
