# 4. 从求和归约到逐行 Softmax

[上一章](03_debug_profile_and_slime.md) · [目录](README.md) · [下一章](05_materialized_attention.md)

这一章只增加一个问题：**一行有多个值，如何让线程先协作得到共同的统计量，再分别写回结果？** 完整实现为 [attention_lab.cu](examples/attention_lab.cu) 中的 `warp_reduce`、`block_reduce` 和 `row_softmax`。

## 4.1 先在纸上计算一行

Softmax 把分数变成非负、和为 1 的权重。对有限输入 `x=[0, ln(2), ln(4)]`：

```text
m = max(x) = ln(4)
e = exp(x-m) = [1/4, 1/2, 1]
l = sum(e) = 7/4
p = e/l = [1/7, 2/7, 4/7]
```

减最大值不改变结果，因为分子、分母都乘上同一个 `exp(-m)`。它让指数输入不大于零，避免直接计算 `exp(1000)` 的溢出。算法因此分成三轮：求最大值 → 求指数之和 → 写归一化结果。每一轮必须使用整行共同的 m/l，不能各线程各自归一化。

## 4.2 一条线程可以处理多个元素

先用 32 线程的 block，即一个完整 warp，负责一行。线程 t 处理 `t, t+32, t+64, ...`。`cols=70` 时，线程 0 处理 0/32/64，线程 6 处理 6/38，最后一段只有 6 个有效元素。

源码中的第一轮是：

```cpp
float local_max = -CUDART_INF_F;
for (int j = t; j < cols; j += B)
    local_max = fmaxf(local_max, x[row * cols + j]);
```

`B` 是本次编译实例的线程数。没有元素的线程保留负无穷，它对取最大值没有影响。求和时对应的中性值是 0。**无有效数据的线程仍参与协作，只是贡献中性值。**

## 4.3 shuffle：在线程寄存器之间传值

`__shfl_down_sync(mask, x, delta)` 让当前 lane 取得编号较大 lane 的 x；这里所有 32 个 lane 都执行，使用完整 mask `0xffffffffu`。例如 delta=16 时，lane 0 取得 lane 16 的值。它不需要先把 x 放进 shared。

```cpp
for (int delta = 16; delta > 0; delta >>= 1) {
    float other = __shfl_down_sync(0xffffffffu, x, delta);
    x = IsMax ? fmaxf(x, other) : x + other;
}
```

这是编译期参数 `IsMax` 控制的同一个归约结构。每轮距离减半，最后 **lane 0** 的值覆盖 32 个输入；其他 lane 的值不能当成完整结果。以 8 lane 的缩小示意来看，lane 0 先接上 4，再接上已包含 2/6 的 2，最后接上已包含 1/3/5/7 的 1。完整示例使用真正的 32 lane。

本实现的前提是完整 warp、所有 lane 在这些调用处共同参与；不能把 mask 原样复制进任意提前退出或分歧分支。shuffle 传的是参与线程的值，不替你完成任意 shared/global memory 的同步。API 约束见 [CUDA C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)。

## 4.4 四个 warp 怎样合成一个 block 的结果

把 B 增大到 128：

1. 每条线程先归约自己的元素，然后四个 warp 各自得到一个结果。
2. 每个 warp 的 lane 0 写 `scratch[warp]`，得到四个 partials。
3. 全 block barrier 后，第一个 warp 的 lane 0..3 分别读取它们，其他 lane 填中性值。
4. 第一个 warp 再归约，lane 0 把最终值写回 `scratch[0]`。
5. 全 block barrier 后，所有线程读取 `scratch[0]`；再做一次 barrier，保护 scratch 被下一次归约复用。

第三道 barrier 看似多余，实际上避免快线程开始下一次归约、覆写 scratch 时，慢线程还未读走旧结果。这里选择容易检查的实现，不追求最少指令。

硬件层次也正好贯穿这条路径：线程局部统计量通常在 SM 寄存器文件中；warp 内的 shuffle 交换参与 lane 的值；跨 warp 的四个 partial 通过该 block 的 16 字节 shared scratch 交接；最后用 block barrier 让所有 warp 看到结果。四个 warp 属于同一 block，因而在同一 SM 上协作；它们不是跨四个 SM 求和。32 线程版本的 scratch 则只有一个 float。

`block_reduce<B, true>` 求最大值，`block_reduce<B, false>` 求和。尖括号是 **C++ 模板参数**，例如 B=128 与 B=32 会生成不同的编译实例；括号里的 x/scratch 才是运行时参数。源码的 `static_assert` 要求 B 是 32 的倍数，launch 还必须真的启动 B 个线程。

## 4.5 完整 Softmax 的数据生命周期

| 阶段 | 线程私有值 | 整行共同值 | 是否可写最终 P |
|---|---|---|---|
| 扫描 x | local_max | 还没有 m | 否 |
| 最大值归约后 | 当前线程的元素 | m | 否 |
| 扫描 exp(x-m) | local_sum | m | 否 |
| 求和归约后 | 当前线程的元素 | m、l | 是 |

完整 kernel 重读输入来计算指数，容易理解但存在重复加载和求指数的成本。后续可以每线程保留多个寄存器值来复用；寄存器增长也会影响资源占用，不能默认这样一定更快。

mask 用 `-inf` 表示不参与的 score，其指数贡献为零。**全行都被 mask 时**，不能执行 `-inf-(-inf)`：本例在得到最大值后统一分支，把输出整行写零。这是教学接口约定，不是对“空集合上的 Softmax”套公式所得的概率分布。输入允许有限值及用于 mask 的负无穷；NaN/正无穷不在本示例支持范围内。

## 4.6 边界条件怎样暴露实现错误

从项目根目录编译与执行；环境要求沿用第 0 章：

```bash
nvcc -O2 -lineinfo -std=c++17 docs/ai_infra/cuda/examples/attention_lab.cu -o /tmp/slime_attention_lab
/tmp/slime_attention_lab
compute-sanitizer --tool memcheck /tmp/slime_attention_lab
compute-sanitizer --tool racecheck /tmp/slime_attention_lab
compute-sanitizer --tool synccheck /tmp/slime_attention_lab
```

程序也会执行后两章的 Attention。先只阅读 Softmax 部分即可。它比较 32/128 线程两种版本与 FP64 CPU 参考，覆盖 1、31、32、33、127、128、129、513 等列数，以及大正偏移、部分 mask、全 mask。

这几种输入揭示了不同的不变量：

| 输入或实现变化 | 数学与执行含义 |
|---|---|
| 输入 `[0,ln(2),ln(4)]` | 得到 `[1/7,2/7,4/7]` |
| 所有有限输入加 1000 | 权重应基本不变；直接 exp 原始分数会出问题 |
| 用 128 线程处理 7 个元素 | 121 个线程仍参与归约，不能提前 return |
| 只用线程私有 local_max 计算指数 | 各线程的指数基准不同，结果不再是整行 Softmax |

统计量沿“线程私有值 → warp → block → 每个线程”传播，各线程最终必须使用同一个行最大值与分母。GPU 示例尚需在目标机器编译和实测；本机的 CPU 检查不能证明这些 barrier 正确。
