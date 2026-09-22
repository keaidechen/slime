# 贯穿案例一：一组Rollout如何变成一次参数更新

> 证据：当前工作区源码已核对，基础数值参考可在CPU运行；本轮没有运行模型训练。源码版本、文件哈希和函数位置见[证据快照](../reference/贯穿案例源码证据.json)。本文采用Megatron后端的普通文本GRPO路径，其他后端、GSPO/CISPO、OPD、TIS、工具轨迹及动态过滤需要另加分支。

## 要证明的事情

同一条回答的动作、概率、优势和mask在数据转换及并行切分后仍然对应；整次更新使用预期归一化；最终参数变化来自这组样本。三个结论分别需要数据、loss/梯度、更新状态证据，单看训练日志里的一个loss不够。

先固定小模型、tokenizer/template、seed、无dropout、一个prompt的三条回答，关闭额外KL、熵损失、dual clip、推测解码与动态过滤。最初用TP=PP=CP=DP=1、单次优化，再保持输入不变改变一个并行维度。模型和权重沿用已经能成功运行的配置，不在本例猜测可用GPU或模型路径。

三条回答可以来自真实采样；若要得到前篇精确数字[0,1,2]和五个token loss，则使用显式教学fixture。真实模型的概率不必等于fixture中的0.2，不能把教学数值标作该模型实测。

## 源码链路与每一站的契约

| 环节 | 当前源码入口 | 需要跟踪的输入输出 |
|---|---|---|
| 生成与评分 | [sglang_rollout.py](../../slime/rollout/sglang_rollout.py)，`generate_and_rm_group` | group、各Sample的tokens/response_length/reward/rollout_log_probs |
| reward后处理 | [rollout.py](../../slime/ray/rollout.py)，`_post_process_rewards` | raw reward与中心化、标准差归一化后的reward |
| 样本转训练列 | 同文件`_convert_samples_to_train_data` | tokens、response_lengths、loss_masks、rollout_ids、rollout_mask_sums |
| 数据布局 | [data.py](../../slime/backends/megatron_utils/data.py)，`get_batch` | packed输入、样本长度、对应microbatch和mask |
| 计算old/ref与优势 | [actor.py](../../slime/backends/megatron_utils/actor.py)，`train_actor` | 实际走过的模型切换、概率重算/复用分支 |
| 可求导概率与目标 | [loss.py](../../slime/backends/megatron_utils/loss.py)，`get_log_probs_and_entropy`、`policy_loss_function` | 当前logprob、old来源、优势、逐token损失 |
| 归约与缩放 | [cp_utils.py](../../slime/backends/megatron_utils/cp_utils.py)，`get_sum_of_sample_mean`；`loss.py::loss_function` | 每个片段的分母、CP贡献、microbatch/DP补偿 |
| 调度与更新 | [model.py](../../slime/backends/megatron_utils/model.py)，`train_one_step` | backward结束、有效步判断、optimizer.step和scheduler推进 |
| 导出证据 | [train_data_utils.py](../../slime/observability/train_data_utils.py)，`save_debug_train_data` | CP还原、DP汇总、样本与布局分开保存 |

这里的“调用链”含嵌套回调，不能把表格读作八个独立进程。`train_one_step`启动前后向调度，loss函数在该调度中被调用。

## 用一条回答核对预测位置

prompt=[p0,p1]，response=[a0,EOS]，完整tokens=[p0,p1,a0,EOS]。预测a0需要位置1的logits，预测EOS需要位置2；response logprob应长2。训练端还需按当前布局还原这些位置，不能直接把本rank的flatten后索引当原序。

逐站记录以下账本；这些是**观察字段建议**，并非每个都已存在于日志：

| 身份/量 | 示例或定义 | 必须保持的条件 |
|---|---|---|
| 顶层rollout编号 | 本轮dump的`rollout_id` | 对齐同一次数据生成与训练 |
| 回答身份 | Sample.index、group_index | group内多条回答不能互相覆盖 |
| dump位置 | train样本的`rollout_position` | 对应本次rollout dump的samples下标 |
| 归约身份 | Sample.rollout_id → `rollout_ids` | 拆成多个训练片段时关联同一逻辑rollout |
| 总长/回答长 | 4 / 2 | prompt_length=total−response |
| 有效mask和 | 2 | padding、工具返回、过滤后的语义明确 |
| 归约分母 | `rollout_mask_sums[i]` | 同一逻辑rollout的片段共享整组分母 |

尤其不能把顶层rollout迭代编号、奖励group身份和样本的归约身份当成同一个ID。它们解决不同问题。

## 同样的公式，配置可能选择不同概率

`train_actor`在满足`can_reuse_log_probs_in_loss`条件时，可以省掉actor单独重算。`policy_loss_function`也可能在没有old结果时采用当前forward的detach快照。开启`use_rollout_logprobs`又会改变分母来源。

因此必须记录“old来自生成、训练重算，还是本次forward快照”。在首次更新且old取当前detach时，ratio应在数值意义上接近1；不能要求它复现教学表里2或0.5的比率。reference概率则服务于另一种约束，不能替换old。

当前dump中的`log_probs`有时来自单独重算，有时来自训练forward捕获；dump在训练调用之后写出，不表示其中每个概率都是更新后权重再次计算的结果。

## 把局部loss接到全局更新

[手工目标](../11_RL_Systems/推演_从轨迹到PPO与GRPO目标.md)的逐token损失为[2,0.8,0,−1.2,−0.5]，有效长度[2,1,2]。若三条回答是三个独立归约单位，序列均值再平均约0.183333；按五个token平均则为0.22。

当前转换器会预先累计同一`rollout_id`的mask总数。若一条长度4的逻辑回答被拆成两段、每段2个有效token，两个片段的分母都应为4；不能各除以2再把片段当两条独立回答。

`loss_function`在非per-token路径还有`num_microbatches / step_global_batch_size × DP-with-CP-size`补偿，后续Megatron调度和梯度归约继续作用。看见局部loss比参考值大，不能立即删掉该因子；应记录各层的乘除关系，直到最终参数梯度。

`calculate_per_token_loss`路径返回token normalizer，并由对应训练逻辑完成缩放。全mask样本、多个逻辑rollout和动态batch需要按实际代码处理，不能只把所有response_lengths相加充当分母。

## 利用现有dump取得真实数据

在已验证的完整训练启动配置上追加以下参数；这段是参数片段，不是完整启动命令。目录是运行机器上的新实验目录：

```text
--save-debug-rollout-data /tmp/slime-case1/rollout_{rollout_id}.pt
--save-debug-train-data /tmp/slime-case1/train_{rollout_id}.pt
```

需要固定训练输入时，使用`--load-debug-rollout-data`重放保存的数据。它会启用训练侧debug路径，不能再把此次重放耗时当真实生成耗时。完整用法见[Debug指南](../../docs/zh/developer_guide/debug.md)。

下面的片段在具有PyTorch的环境中，核对自己生成的两个dump；本轮没有真实dump，未执行此段。基线限定为没有过滤或新增训练片段的普通路径：

```python
import torch

rollout = torch.load('/tmp/slime-case1/rollout_0.pt', map_location='cpu', weights_only=False)
train = torch.load('/tmp/slime-case1/train_0.pt', map_location='cpu', weights_only=True)
assert train['format_version'] == 2
assert train['rollout_id'] == rollout['rollout_id']
seen = set()
for sample in train['samples']:
    pos = sample['rollout_position']
    assert pos not in seen
    seen.add(pos)
    source = rollout['samples'][pos]
    assert sample['response_lengths'] == source['response_length']
    assert torch.equal(torch.as_tensor(sample['tokens']), torch.as_tensor(source['tokens']))
    actual = torch.as_tensor(sample['rollout_log_probs'])
    expected = torch.as_tensor(source['rollout_log_probs'], dtype=actual.dtype)
    assert actual.numel() == sample['response_lengths']
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
assert seen == set(range(len(rollout['samples'])))
print('PASS carried rollout probabilities and sample alignment; no optimizer claim')
```

如果dump缺失行为概率，应直接报告缺失，不能跳过所有样本仍宣布通过。`1e-3`沿用已有[8卡dump核对案例](../../tests/test_qwen2.5_0.5B_debug_train_dump_e2e.py)的容差，正式实验应记录误差分布并按精度要求选择容差；不意味着任意概率错误都能容忍此值。

## 参数更新需要增加哪一份证据

现有dump不自动保存每个microbatch的可求导loss、完整梯度和Adam状态。要证明一步更新，需在隔离的小模型实验中，于`optimizer.step()`前后额外采集选定参数的全局身份、主权重、归约后梯度、m/v、step以及真实超参数；分片状态按owner还原。GPU梯度采集会影响性能，因此与正式测速分开。

先以完整batch参考更新对比，再使用同一数据做microbatch或DP切分；比较token目标、梯度和新权重三层。不要仅比较grad_norm，它无法发现参数梯度被错配但范数相同的问题。纯数学预检可运行[training_rl_walkthroughs.py](../labs/training_rl_walkthroughs.py)，其通过不等于真实Adam融合实现已验证。

## 可控反例与完成标准

在离线副本中给一个保存的行为logprob加0.1，应使上述对齐检查失败；把某条回答的token顺序交换，应先在tokens检查失败。随后用教学fixture把“同rollout两片共享分母4”改成各自分母2，应观察到目标变化。

需要GPU的下一阶段反例是固定数据、仅改变packing或并行布局。允许舍入差异，但有效样本、mask总数与逻辑目标不得改变。最终证据包要含运行配置、两个dump、比对结果，以及单步参数对照；本轮完成源码映射和CPU参考，运行证据仍待真实实验产生。

---

[四个贯穿案例](贯穿案例索引.md) · [RL原理](../11_RL_Systems/README.md)
