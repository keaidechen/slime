# CPU、数据加载与 Host-to-Device 路径

> 本页为兼容入口，正文已整合到下面的主章节；学习时直接进入主章节即可。

原理与操作已放在同一篇中。原有重复的定义、基础采集步骤与资料列表由主章节统一维护。

- [CPU 数据供应与 GPU 等待](<../../docs/performance_analysis_guide/02_pytorch.md#concept-07>)


<details>
<summary>旧章节定位（供历史链接使用）</summary>

<a id="cpu数据加载与-host-to-device-路径"></a>

CPU、数据加载与 Host-to-Device 路径 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#concept-07>)

<a id="1-供应链视图"></a>

1. 供应链视图 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#1-供应链视图>)

<a id="2-dataloader-参数不是越大越好"></a>

2. DataLoader 参数不是越大越好 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#2-dataloader-参数不是越大越好>)

<a id="3-pin_memory-与-non_blocking"></a>

3. `pin_memory` 与 `non_blocking` → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#3-pin_memory-与-non_blocking>)

<a id="4-用控制实验定位"></a>

4. 用控制实验定位 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#4-用控制实验定位>)

<a id="5-时间线上的典型证据"></a>

5. 时间线上的典型证据 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#5-时间线上的典型证据>)

<a id="6-python-与隐式同步"></a>

6. Python 与隐式同步 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#6-python-与隐式同步>)

<a id="7-编译与小-kernel"></a>

7. 编译与小 kernel → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#7-编译与小-kernel>)

<a id="8-分布式放大效应"></a>

8. 分布式放大效应 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#8-分布式放大效应>)

<a id="资料"></a>

资料 → [进入主章节](<../../docs/performance_analysis_guide/02_pytorch.md#concept-07>)

</details>
