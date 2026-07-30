# 05 RL 算法实现：从 logprob 到 loss 的完整链路

> 对应综述（`00_rl_infra_survey.md`）§2.6「RL 算法」。
> 涉及文件：`slime/backends/megatron_utils/loss.py`（流程编排）、`slime/utils/ppo_utils.py`（算法原语，移植自 OpenRLHF）、`slime/backends/megatron_utils/cp_utils.py`（Context Parallel 支持）。
> 本篇按"训练一步时数据如何流动"的顺序讲，每个公式都配数值例子。

---

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

可以（`--use-rollout-logprobs`），但默认训练侧要用**当前参数**重新 forward 一遍算 logprob——PPO 的梯度就来自这个新 logprob。难点：batch 里多条样本拼接成一维序列（packing），还要兼容 TP（词表被切分）与 CP（序列被切分）。

### 2.2 `get_log_probs_and_entropy`（loss.py:470-561）

关键设计：**在整个 `[T, V]` 上一次性算完，再按样本切回**，让 backward 只遍历一次大 tensor：

1. **温度对齐**（loss.py:497-500）：`logits /= rollout_temperature`。采样时用了温度，训练侧算 logprob 必须除以同一个温度，否则训推不可比；
2. **shifted tokens**（`_build_shifted_tokens`，loss.py:230-280）：语言模型是"用前 t-1 个 token 的 logits 预测第 t 个"，所以目标 token 要左移一位；
3. **top-p 掩码**（`_build_topp_keep_mask`，loss.py:306-386）：rollout 用 top-p 采样时，训练侧只在相同的候选核内算 logprob（02 篇的回放数据在此消费），否则训推分布不同；
4. **`calculate_log_probs_and_entropy`**（ppo_utils.py:746）：TP 感知的 logsumexp——词表按 TP 切分，每个 rank 只有 `V/tp` 的 logits，需要跨 TP 组归约（`_VocabParallelLogProbEntropy` 自定义 autograd Function，ppo_utils.py:187）；`log_probs_chunk_size` 分块防 OOM；
5. **按样本切回**（`_extract_per_sample`，loss.py:389-467）。

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

**一个数字直觉**：4 卡 TP，词表 15 万，某 rank 持有 `[T, 37500]` 的 logits 分片；对每个 token，通信量是 2 次 all_reduce（各 1 个标量/token）+ 1 次 all_reduce（predicted_logits，1 个标量/token）= 3 个标量/token 的跨卡通信，而不是把 37500 维的分片传来传去——这正是为什么 TP 下算 logprob 不会成为通信瓶颈。

### 2.5 zigzag CP 的一个切分例子

假设一条序列长度 8（token 0-7），`CP=2`。zigzag（Ring Attention 的标准切法）不是简单地"前 4 个给 rank0，后 4 个给 rank1"（这样 rank0 全是"简单"的早期 token、rank1 全是"难"的后期 token，causal attention 下两个 rank 的计算量严重不均衡），而是"掐头去尾配对"：

```
全局位置:   0  1  2  3  4  5  6  7
zigzag 分配: 0  1  1  0  0  1  1  0    (示意：rank0 拿 {0,3,4,7} 之类的对称配对，具体实现是"前一半"与"后一半"分别对半镶嵌)
```

（slime 实际实现细节见 `get_logits_and_tokens_offset_with_cp`，核心思想是每个 rank 拿到"一段从序列头部、一段从序列尾部对称位置"的 token，使得每个 rank 的 causal attention 计算量基本相等）。取 response 部分时必须知道这种"跳跃式"映射关系，才能把两个 rank 各自算出的 logprob 正确拼回原始顺序——这也是为什么 `allgather_cp` 模式提供了一个"先拼成正常顺序、算完再拆回 zigzag"的旁路（用一次通信换取代码简单性，用于调试或不关心额外通信开销的场景）。

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
| k1 | \(r = \log\pi_\theta - \log\pi_{ref}\) | 无偏但可为负、方差大 |
| k2 | \(r^2/2\) | 恒正但有偏 |
| k3 | \(\exp(-r)-1+r\) | 恒正、无偏、低方差（推荐） |

`importance_ratio` 参数（loss.py:1057-1058 的 `--use-unbiased-kl`）对应 DeepSeek-V3.2 的无偏 KL 修正：KL 乘以 IS ratio。

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

**PPO（GAE）**：critic 给每个 token 估值，`vanilla_gae/chunked_gae`（ppo_utils.py:579/603）按 \( \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \)、\( A_t = \sum_l (\gamma\lambda)^l \delta_{t+l} \) 递推；reward 被加在最后一个 token 的 KL 惩罚项上（loss.py:727-738）。

**REINFORCE++**：token 级 reward = `-kl_coef * kl`，序列 reward 加在最后一个有效 token（`last_idx`，ppo_utils.py:418-419），再按 \( G_t = r_t + \gamma G_{t+1} \) 倒推（421-426）；`reinforce_plus_plus_baseline` 则是 `(reward - 组基线)` 直接广播（441-468）。

**GSPO / CISPO**：advantage 与 GRPO 相同（loss.py:720-724 归为同一分支），差异在 loss 的比率定义（§4.2）。

### 3.3 全局白化

`--normalize-advantages` 时（loss.py:776-825），把整个 DP 组的 advantage 拼起来做 masked 白化（减均值除标准差）——`distributed_masked_whiten` 跨 DP 组聚合统计量，mask 保证只统计有效 token。注意 CP 下 mask 也要按相同 zigzag 切分（782-810 行）。

### 3.4 OPD：on-policy distillation 的 advantage 修正

`apply_opd_kl_to_advantages`（loss.py:620-658）：student 生成轨迹、teacher 打分（teacher_log_probs 由 rollout 侧用 `max_new_tokens=0` 前向取得），advantage 减去 `opd_kl_coef × (student_logp - teacher_logp)` 的 reverse KL——**学生偏离老师越远惩罚越重**，蒸馏信号与任意 advantage 估计器正交叠加。

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

### 4.2 GSPO：序列级重要性比率

token 级比率对 MoE 不稳定（每个 token 激活的专家不同，比率方差大）。GSPO 改用**整条序列的平均 KL 作为每个 token 的比率**（ppo_utils.py:95-121）：先 all-gather 拼回完整序列（loss.py:939-951），算序列级 KL 再 `expand_as` 广播回每个 token。

### 4.3 CISPO：clip 权重而非比值

```167:171:slime/utils/ppo_utils.py
    ratio = (-ppo_kl).exp()
    ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
    pg_losses = -ratio_truncated.detach() * advantages * log_probs
```

MiniMax-M1 的做法：比率 clamp 后 **stop-gradient** 当权重，梯度从 `log_probs` 流过——被 clip 的 token 仍贡献梯度（PPO 里它们的梯度为 0）。

### 4.4 OPSM：序列级拒采

`compute_opsm_mask`（ppo_utils.py:54-92）：**负 advantage 且序列 KL 超过 `opsm_delta` 的序列整段 mask 掉**——策略已明显偏离旧策略的"坏样本"不再贡献梯度，是 off-policy 防护的另一种形态。

### 4.5 完整数字例子：一组 GRPO 样本从 reward 走到 policy loss

设一个 prompt 采样 8 条（`n_samples_per_prompt=8`），规则奖励 `reward = [1,1,0,0,1,0,1,1]`（1=答对，0=答错），`kl_coef=0`（不做 KL 惩罚，纯 GRPO）：

1. **组内归一化**（发生在 rollout 侧的 reward 后处理，而非 `loss.py`）：`mean=0.625, std≈0.484`，归一化后 `reward_norm ≈ [0.775, 0.775, -1.29, -1.29, 0.775, -1.29, 0.775, 0.775]`——这一步是"GRPO 省 critic"的关键：不需要学一个 value 网络，直接靠组内统计量得到一个零均值的相对优势信号；
2. **广播到 token**（`get_grpo_returns`）：每条样本内所有 response token 的 advantage 都等于该样本的 `reward_norm`（例如样本 0 的 5 个 response token 全部 advantage=0.775）；
3. **`ppo_kl = old_log_probs - log_probs`**：假设样本 0 某个 token 训练时算出 `log_probs=-0.5`，生成时记录的 `old_log_probs=-0.7`（同一份权重理论上应该相等，但因为 packing/精度/若干步内已更新过参数等原因会有细微差异），`ppo_kl = -0.7-(-0.5) = -0.2`；
4. **`ratio = exp(-ppo_kl) = exp(0.2) ≈ 1.221`**：新策略比旧策略更倾向于生成这个 token（比值 >1）；
5. **PPO-clip**（设 `eps_clip=0.2, eps_clip_high=0.28`，DAPO 式 clip-higher）：`ratio` 落在 `[0.8, 1.28]` 内未被裁剪，`pg_loss = -min(ratio·A, clip(ratio)·A) = -1.221 × 0.775 ≈ -0.946`（因为 advantage 为正，未裁剪分支占优，loss 取更小的负值即更大的"损失下降量"，梯度会推动模型进一步提高这个 token 的概率）；
6. **样本 2**（答错，advantage=-1.29）：若其某 token 的 `ratio≈1.22` 同样未被裁剪，`pg_loss = -1.22 × (-1.29) ≈ 1.574`（正的 loss，梯度会*降低*这个 token 的概率——错误答案里出现的 token 被抑制）；
7. **归一化求和**（§6）：8 条样本的所有 token loss 按"每条样本内部先按 token 数取平均，再对样本取平均"（或反之，取决于 `--loss-agg-mode`）汇总成一个标量，反传给 Megatron 做梯度累积。

**直觉**：GRPO 把"这条样本比组内平均水平好还是差"变成一个恒定的乘数，PPO-clip 再把"这一步参数更新是不是走得太快"这件事通过 ratio 裁剪限制住——两者一个负责"往哪个方向走"，一个负责"这一步别走太远"。

---

## 5. TIS：训推数值不一致的修正

问题：FP8 rollout、推理引擎 kernel 差异 → `rollout_log_probs`（行为策略）≠ 训练侧同参数下的 logprob（目标策略），严格说已 off-policy。TIS（truncated importance sampling）用比值修正：

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

`icepop_function`（loss.py:855-878）是变体：超出 `[tis_clip_low, tis_clip]` 的 token 权重直接置 0（拒绝而非截断）。配套观测指标：`tis / tis_clipfrac / tis_abs / ois`（loss.py:1093-1100），以及长期健康指标 `train_rollout_logprob_abs_diff`（loss.py:1073-1077）——这个值缓慢爬升通常意味着训推某一侧出了数值问题。`--custom-tis-function-path` 可替换自定义修正策略（loss.py:1011-1014）。

---

## 6. 归一化与缩放（容易踩坑的部分）

`loss_function`（loss.py:1220-1320）最后做两件"会计工作"：

1. **归一化器**：`get_sum_of_sample_mean` 按"先每样本对有效 token 求均值、再对样本求和"规约；`--calculate-per-token-loss` 时改为全局 token 数归一（DAPO 的 token-level loss，长短不一的样本不再被平均权重拉平）；
2. **Megatron 梯度累积对齐**（loss.py:1289-1298）：per-rollout-mean 模式下要乘 `num_microbatches / step_global_batch_size * dp_size` 抵消 Megatron 内部的平均；per-token 模式乘 `cp_size` 抵消 CP 维度的平均。注释（1289 行）明言"divide by cp_size to cancel the multiply in Megatron"。

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

读完本篇建议动手：把 `--advantage-estimator` 在 grpo/gspo/cispo 间切换，观察 `pg_clipfrac`、`ppo_kl`、`tis` 指标变化，是理解这些算法差异最快的方式。
