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
