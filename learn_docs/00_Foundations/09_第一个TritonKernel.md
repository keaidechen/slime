# 第一个 Triton Kernel：把线程工作与数据地址连起来

> 前置：GPU 执行模型、tensor 连续布局与正确计时。预计 3–4 小时。目标是理解并验证一个小算子，不以超越成熟库为完成条件。

## 1. 先缩小要实现的数学问题

只做一维连续 FP32 向量 `z=x+y`。没有 autograd、跨卡、广播和非连续 stride 支持。输入范围明确后，才能判断代码正确性。

完整脚本：[vector_add.py](../labs/vector_add.py)。需要 Linux、CUDA-enabled PyTorch 和可用 Triton；先在已配置的实验环境运行，不在运行中的训练环境临时升级依赖。

```bash
python learn_docs/labs/vector_add.py
```

这个脚本没有命令行参数。第一行输出会显示 GPU 名称和 PyTorch 版本；之后每行是一个字典：`n` 是向量元素个数，`torch_ms_median_min_max` 和 `triton_ms_median_min_max` 依次是中位数、最小值和最大值，单位都是毫秒。

正确性检查成功时不会额外打印“通过”；整个程序正常退出就表示所有 `assert_close` 检查已通过。如果环境没有可用 CUDA，脚本会以 `This lesson requires CUDA and Triton` 错误退出；这是环境前置条件不满足，不是 Kernel 数值检查失败。

## 2. Program、tile 与尾块

Triton 的 program instance 处理一块元素。若 `BLOCK=256`，第 `p` 个实例处理 `p*256 + [0,...,255]`。这不是要求你把每个元素固定绑定到某个物理 CUDA Core；实际线程和指令映射由编译与硬件执行决定。

当长度为 1000 时，需要 4 个 program；最后一个只处理 232 个有效元素，余下位置必须屏蔽。`mask=offset<n` 既用于 load 也用于 store，不能依赖越界位置“刚好没有被访问”。

## 3. 为什么这个算子适合学数据搬运

每元素读两个 FP32、写一个 FP32，理想流量约 12 字节，做一次加法。算术强度约为 `1/12 FLOP/byte`，很低。大输入时通常优先考虑内存流量，小输入则可能主要受 launch 开销影响；最终仍需实测。

有效带宽可按 `12*n/time` 计算，但这是逻辑流量口径，不是硬件计数器直接测得的 HBM 流量。缓存、写入行为与小负载都会让它不同于 DRAM 实际读写。

## 4. 先正确，再计时

脚本先对多个长度与随机种子检查 PyTorch reference。计时前先执行以排除首次编译。PyTorch 基线使用预分配输出，Triton 也复用输出，避免一边计分配、一边不计。

计时重复若干次，输出中位数与范围。不要把这份教学微基准的结果直接作为框架加速结论：它没有评估融合机会、生产 shape、其他 stream 干扰或端到端关键路径。

## 5. 从这个例子进入 Kernel 专题

| 观察 | 下一步问题 | 阅读 |
|---|---|---|
| 小输入时间几乎不变 | 固定 launch 成本占多大比例 | [CUDA 执行](../01_Kernel_GPU_Programming_Compiler/01_CUDA执行模型与调度.md) |
| 大输入随 n 增长 | 逻辑带宽与真实带宽怎样比较 | [内存与搬运](../01_Kernel_GPU_Programming_Compiler/02_GPU内存编程与数据搬运.md) |
| 改 BLOCK 后速度变化 | 资源、并行粒度与尾块浪费 | [Kernel 实验](../01_Kernel_GPU_Programming_Compiler/09_实验与排查手册.md) |
| 两个 elementwise 串联 | 能否减少一次中间读写 | [PyTorch 编译栈](../01_Kernel_GPU_Programming_Compiler/06_PyTorch编译栈.md) |

## 6. 验收

- 画出 n=1000、BLOCK=256 时四个 program 的索引范围。
- 解释 mask 的必要性，以及连续布局假设。
- 比较至少三个数量级的 shape，报告正确性和重复计时。
- 不论快慢，给出一个受证据支持的解释，并说明微基准限制。

Triton 编程接口参考：[官方 Vector Addition 教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)。更复杂的 attention 和 GEMM 留到理解本例后学习。
