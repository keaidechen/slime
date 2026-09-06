# 12 训练侧内部实现：Megatron-LM 篇

> 本页为兼容入口，正文已整合到下面的主章节；学习时直接进入主章节即可。

重复的引擎解释已由专门系列统一维护。纠正旧稿：普通参数与 expert 参数使用不同 rank 网格，不能无条件把 EP 再乘入 world size；DistributedOptimizer 先用规约后的梯度更新本地 shard，再聚合更新后的参数（可能延迟/重叠）。StatelessAdam 改变跨步矩估计，不只是纯性能技巧。

- [一次 step 与模型构建](<../megatron_code_walkthrough/01_foundations/02_training_mainline.md>)
- [rank 网格与 TP/SP](<../megatron_code_walkthrough/02_parallelism/01_parallel_state_and_tensor_parallel.md>)
- [DistributedOptimizer、更新与参数聚合](<../megatron_code_walkthrough/03_state_and_memory/01_data_parallel_optimizer_checkpoint.md>)
- [microbatch 与 1F1B](<../megatron_code_walkthrough/02_parallelism/02_pipeline_parallel.md>)
- [slime 接入与 StatelessAdam 的算法变化](<06_megatron_backend_and_mbridge.md>)
- [EP/DP 与 batch 归一化辨析](<../megatron_code_walkthrough/06_reference/03_source_questions.md>)


<details>
<summary>旧章节定位（供历史链接使用）</summary>

<a id="12-训练侧内部实现megatron-lm-篇"></a>

12 训练侧内部实现：Megatron-LM 篇 → [进入主章节](<../megatron_code_walkthrough/01_foundations/02_training_mainline.md>)

<a id="0-slime-的调用入口回顾"></a>

0. slime 的调用入口回顾 → [进入主章节](<../megatron_code_walkthrough/01_foundations/02_training_mainline.md>)

<a id="1-get_model模型是怎么被切出来的"></a>

1. `get_model`：模型是怎么被"切"出来的 → [进入主章节](<../megatron_code_walkthrough/01_foundations/02_training_mainline.md>)

<a id="2-parallel_stateinitialize_model_parallel通信组是怎么排的"></a>

2. `parallel_state.initialize_model_parallel`：通信组是怎么排的 → [进入主章节](<../megatron_code_walkthrough/01_foundations/02_training_mainline.md>)

<a id="21-深入拆解rankgenerator-与-generate_masked_orthogonal_rank_groups-的精确公式"></a>

2.1 深入拆解：`RankGenerator` 与 `generate_masked_orthogonal_rank_groups` 的精确公式 → [进入主章节](<../megatron_code_walkthrough/01_foundations/02_training_mainline.md>)

<a id="3-get_megatron_optimizer-与-distributedoptimizer"></a>

3. `get_megatron_optimizer` 与 DistributedOptimizer → [进入主章节](<../megatron_code_walkthrough/01_foundations/02_training_mainline.md>)

<a id="4-配置与-checkpoint"></a>

4. 配置与 checkpoint → [进入主章节](<../megatron_code_walkthrough/01_foundations/02_training_mainline.md>)

<a id="5-moe-与流水线调度训练一步的执行面"></a>

5. MoE 与流水线调度（训练一步的执行面） → [进入主章节](<../megatron_code_walkthrough/01_foundations/02_training_mainline.md>)

<a id="51-深入拆解1f1b-调度到底在调度什么"></a>

5.1 深入拆解：1F1B 调度到底在"调度"什么 → [进入主章节](<../megatron_code_walkthrough/01_foundations/02_training_mainline.md>)

<a id="6-小结"></a>

6. 小结 → [进入主章节](<../megatron_code_walkthrough/01_foundations/02_training_mainline.md>)

</details>
