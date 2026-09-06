# Nsight Compute：Kernel 级分析

> 本页为兼容入口，正文已整合到下面的主章节；学习时直接进入主章节即可。

原理与操作已放在同一篇中。原有重复的定义、基础采集步骤与资料列表由主章节统一维护。

- [从热点读懂 Kernel 报告](<../../docs/performance_analysis_guide/03_cuda_kernels.md#concept-04>)


<details>
<summary>旧章节定位（供历史链接使用）</summary>

<a id="nsight-computekernel-级分析"></a>

Nsight Compute：Kernel 级分析 → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#concept-04>)

<a id="1-先缩小目标"></a>

1. 先缩小目标 → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#concept-04>)

<a id="2-四步读报告"></a>

2. 四步读报告 → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#2-四步读报告>)

<a id="步骤-a确认工作量"></a>

步骤 A：确认工作量 → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#步骤-a确认工作量>)

<a id="步骤-b看-speed-of-light"></a>

步骤 B：看 Speed of Light → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#步骤-b看-speed-of-light>)

<a id="步骤-c看-occupancy-与-wave"></a>

步骤 C：看 Occupancy 与 Wave → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#步骤-c看-occupancy-与-wave>)

<a id="步骤-d看-scheduler-stall-与-memory-access"></a>

步骤 D：看 scheduler stall 与 memory access → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#步骤-d看-scheduler-stall-与-memory-access>)

<a id="3-roofline"></a>

3. Roofline → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#concept-04>)

<a id="4-一个简化例子"></a>

4. 一个简化例子 → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#4-一个简化例子>)

<a id="5-baseline-comparison"></a>

5. Baseline comparison → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#5-baseline-comparison>)

<a id="6-常见陷阱"></a>

6. 常见陷阱 → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#6-常见陷阱>)

<a id="7-2026-工具边界"></a>

7. 2026 工具边界 → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#concept-04>)

<a id="资料"></a>

资料 → [进入主章节](<../../docs/performance_analysis_guide/03_cuda_kernels.md#concept-04>)

</details>
