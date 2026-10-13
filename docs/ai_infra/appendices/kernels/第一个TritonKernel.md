# 第一个 Triton Kernel：把线程工作与数据地址连起来

> 前置：GPU 执行模型、tensor 连续布局与正确计时。本篇用向量加说明地址、尾块与数据搬运。


## 先缩小要实现的数学问题

只做一维连续 FP32 向量 `z=x+y`。没有 autograd、跨卡、广播和非连续 stride 支持。输入范围明确后，才能判断代码正确性。

完整脚本：[vector_add.py](<../../examples/vector_add.py>)。运行依赖 Linux、CUDA-enabled PyTorch 和 Triton；这些依赖限定示例实现的适用范围。

脚本没有命令行参数；GPU 名称与 PyTorch 版本标识执行环境，结果字典的字段定义测量口径：`n` 是向量元素个数，`torch_ms_median_min_max` 和 `triton_ms_median_min_max` 依次是中位数、最小值和最大值，单位都是毫秒。

正确性检查成功时不会额外打印“通过”；整个程序正常退出就表示所有 `assert_close` 检查已通过。如果环境没有可用 CUDA，脚本会以 `This lesson requires CUDA and Triton` 错误退出；这是环境前置条件不满足，不是 Kernel 数值检查失败。


## Program、tile 与尾块

Triton 的 program instance 处理一块元素。若 `BLOCK=256`，第 `p` 个实例处理 `p*256 + [0,...,255]`。这不是要求你把每个元素固定绑定到某个物理 CUDA Core；实际线程和指令映射由编译与硬件执行决定。

当长度为 1000 时，需要 4 个 program；最后一个只处理 232 个有效元素，余下位置必须屏蔽。`mask=offset<n` 既用于 load 也用于 store，不能依赖越界位置“刚好没有被访问”。


## 为什么这个算子适合学数据搬运

每元素读两个 FP32、写一个 FP32，理想流量约 12 字节，做一次加法。算术强度约为 `1/12 FLOP/byte`，很低。大输入时通常优先考虑内存流量，小输入则可能主要受 launch 开销影响；最终仍需实测。

有效带宽可按 `12*n/time` 计算，但这是逻辑流量口径，不是硬件计数器直接测得的 HBM 流量。缓存、写入行为与小负载都会让它不同于 DRAM 实际读写。


## 先正确，再计时

脚本先对多个长度与随机种子检查 PyTorch reference。计时前先执行以排除首次编译。PyTorch 基线使用预分配输出，Triton 也复用输出，避免一边计分配、一边不计。

计时重复若干次，输出中位数与范围。不要把这份教学微基准的结果直接作为框架加速结论：它没有评估融合机会、生产 shape、其他 stream 干扰或端到端关键路径。


## 从这个例子进入 Kernel 专题

| 观察 | 下一步问题 | 阅读 |
|---|---|---|
| 小输入时间几乎不变 | 固定 launch 成本占多大比例 | [CUDA 执行](<../hardware/CUDA执行模型与调度.md>) |
| 大输入随 n 增长 | 逻辑带宽与真实带宽怎样比较 | [内存与搬运](<GPU内存编程与数据搬运.md>) |
| 改 BLOCK 后速度变化 | 资源、并行粒度与尾块浪费 | [Kernel 成本分析](<Kernel性能与错误成因.md>) |
| 两个 elementwise 串联 | 能否减少一次中间读写 | [PyTorch 编译栈](<../../03_Execution/10_编译器与运行时.md#s3>) |


## 尾块和规模怎样改变这个例子

当 `n=1000、BLOCK=256`，四个 program 分别覆盖 0–255、256–511、512–767、768–1023。最后只有 768–999 有效，1000–1023 必须被 mask 排除。这里的地址公式还假设连续布局；非连续 tensor 需要显式处理 stride。

小输入可能主要受 launch 固定开销限制，输入增大后数据搬运和可用并行度才逐渐主导。因而一个 shape 的微基准不能代表整个模型，也不能只看单次计时。正确性、重复统计与实际 workload 共同限定性能判断。

参考：[Triton Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)。
