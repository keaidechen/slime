# 显存分析与 Out of Memory（OOM，内存不足）

> 本页为兼容入口，正文已整合到下面的主章节；学习时直接进入主章节即可。

原理与操作已放在同一篇中。原有重复的定义、基础采集步骤与资料列表由主章节统一维护。

- [显存峰值与资源生命周期](<../../docs/performance_analysis_guide/02_pytorch.md#concept-08>)


<details>
<summary>旧章节定位（供历史链接使用）</summary>

<a id="显存分析与-out-of-memoryoom内存不足"></a>

显存分析与 Out of Memory（OOM，内存不足） → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#concept-08>)

<a id="1-三个数字"></a>

1. 三个数字 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#concept-08>)

<a id="2-一步训练为何出现多个峰值"></a>

2. 一步训练为何出现多个峰值 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#2-一步训练为何出现多个峰值>)

<a id="3-建立显存账本"></a>

3. 建立显存账本 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#3-建立显存账本>)

<a id="4-snapshot-工作流"></a>

4. Snapshot 工作流 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#concept-08>)

<a id="5-碎片与真正容量不足"></a>

5. 碎片与真正容量不足 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#5-碎片与真正容量不足>)

<a id="6-泄漏常见来源"></a>

6. 泄漏常见来源 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#6-泄漏常见来源>)

<a id="7-非-pytorch-显存"></a>

7. 非 PyTorch 显存 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#7-非-pytorch-显存>)

<a id="8-推理中的-capacity-与性能耦合"></a>

8. 推理中的 capacity 与性能耦合 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#8-推理中的-capacity-与性能耦合>)

<a id="9-修复优先级"></a>

9. 修复优先级 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#9-修复优先级>)

<a id="资料"></a>

资料 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#concept-08>)

</details>
