# 并行策略与 Bubble 分析

> 本页为兼容入口，正文已整合到下面的主章节；学习时直接进入主章节即可。

原理与操作已放在同一篇中。原有重复的定义、基础采集步骤与资料列表由主章节统一维护。

- [并行调度在时间线上的证据](<../../docs/performance_analysis_guide/04_megatron.md#concept-10>)


<details>
<summary>旧章节定位（供历史链接使用）</summary>

<a id="并行策略与-bubble-分析"></a>

并行策略与 Bubble 分析 → [进入主章节](<../../docs/performance_analysis_guide/04_megatron.md#concept-10>)

<a id="1-并行维度的时间线指纹"></a>

1. 并行维度的时间线指纹 → [进入主章节](<../../docs/performance_analysis_guide/04_megatron.md#1-并行维度的时间线指纹>)

<a id="2-pipeline-bubble"></a>

2. Pipeline bubble → [进入主章节](<../../docs/performance_analysis_guide/04_megatron.md#2-pipeline-bubble>)

<a id="怎么从-trace-量"></a>

怎么从 trace 量 → [进入主章节](<../../docs/performance_analysis_guide/04_megatron.md#怎么从-trace-量>)

<a id="3-ddpfsdp最后一个-bucket-最重要"></a>

3. DDP/FSDP：最后一个 bucket 最重要 → [进入主章节](<../../docs/performance_analysis_guide/04_megatron.md#3-ddpfsdp最后一个-bucket-最重要>)

<a id="4-tp算子变小与通信变频繁"></a>

4. TP：算子变小与通信变频繁 → [进入主章节](<../../docs/performance_analysis_guide/04_megatron.md#4-tp算子变小与通信变频繁>)

<a id="5-cp平均-token-数不够"></a>

5. CP：平均 token 数不够 → [进入主章节](<../../docs/performance_analysis_guide/04_megatron.md#5-cp平均-token-数不够>)

<a id="6-ep通信与-expert-straggler-耦合"></a>

6. EP：通信与 expert straggler 耦合 → [进入主章节](<../../docs/performance_analysis_guide/04_megatron.md#6-ep通信与-expert-straggler-耦合>)

<a id="7-组合并行的归因方法"></a>

7. 组合并行的归因方法 → [进入主章节](<../../docs/performance_analysis_guide/04_megatron.md#7-组合并行的归因方法>)

<a id="8-优化顺序"></a>

8. 优化顺序 → [进入主章节](<../../docs/performance_analysis_guide/04_megatron.md#8-优化顺序>)

<a id="资料"></a>

资料 → [进入主章节](<../../docs/performance_analysis_guide/04_megatron.md#concept-10>)

</details>
