# 6. 将普通 Attention 改成分块在线计算

[上一章](<05_materialized_attention.md>) · [目录](<README.md>) · [下一章](<07_mma_and_cute.md>)

这一章先推导状态更新，再对应 [attention_lab.cu](<examples/attention_lab.cu>) 的 `online_attention`。它是**用于理解 FlashAttention 核心算法的简化 CUDA 实现**：每个 block 处理一个 query，FP32 标量运算，不使用 Tensor Core，也不实现 FA2 的线程分工和性能优化。

## 6.1 为什么不能分别对每块做 Softmax

两块分数分别为 `[0,ln(2)]` 和 `[ln(4),ln(8)]`，分别 Softmax 都得到 `[1/3,2/3]`。拼起来的权重和为 2；即使再平均，也错误地赋予两块相同的总权重。真正的整行权重是 `[1,2,4,8]/15`，第二块权重总和应为 12/15。

所以每块不能只留下归一化后的局部输出；还要保存它在整个归一化过程中所占的“量”。

## 6.2 用三个状态完整描述已经处理的 key

对已处理的 key 集合，保存：

```text
m    = 当前最大的 score
l    = sum(exp(score-m))
a[c] = sum(exp(score-m) * V[key,c])
```

这里 a 是未归一化的输出向量；处理完所有 key 才输出 `a/l`。当新块使最大值从 m 升为 m_new，旧的每个指数必须换基准：

```text
exp(score-m_new) = exp(score-m) * exp(m-m_new)
```

定义 `alpha=exp(m-m_new)`，则：

```text
m_new = max(m, 当前块最大值)
p[j]  = exp(当前块score[j] - m_new)
l_new = alpha*l + sum(p)
a_new = alpha*a + sum_j(p[j]*V[j])
```

旧分母和旧分子必须一起缩放。p 是相对于共同基准的指数权重，尚未归一化；不能因为变量名像 probability 就认为它的和为 1。

## 6.3 两轮完整算例，以及一个故意算错的版本

继续取 scores=`[0,ln(2),ln(4),ln(8)]`，V=`[1,2,3,4]`，每块两个 key：

| 时刻 | m | alpha | l | a |
|---|---|---|---|---|
| 初始 | -inf | — | 0 | 0 |
| 第一块后 | ln(2) | 0 | 1.5 | 2.5 |
| 第二块后 | ln(8) | 1/4 | `1.5/4+1.5=1.875` | `2.5/4+5.5=6.125` |

最终 O=`6.125/1.875=49/15≈3.266667`。如果第二轮忘了缩放旧状态，就得到 `l=3, a=8, O≈2.666667`。这不是小的浮点误差，而是算法改变了。

[attention_cpu.py](<examples/attention_cpu.py>) 会打印这两轮 m/alpha/l/a，也会运行这个故意遗漏缩放的反例。参数 `rescale=False` **只用于演示错误**，不应作为正常算法选项。

## 6.4 把数学量分配给 CUDA 线程

本例 `THREADS=128`，每个 block 拥有一个 query 行：

| 量 | 谁持有 | 生命周期 |
|---|---|---|
| 当前 tile 的 score | 线程 t 的寄存器，对应 key=base+t | 当前一轮 |
| m、l、alpha | shared 标量，线程 0 更新 | m/l 跨轮保留 |
| 当前 tile 的 p | shared `weights[128]` | 当前一轮，PV 读完后可覆盖 |
| a 的第 c 个通道 | 线程 c 的局部 `numerator` | 跨轮保留，最后除以 l |

前半轮 t 代表 key 列，后半轮 t 代表输出通道。**线程的任务由代码阶段决定，不是固定绑定一个数学维度。** 这也是理解真实 FA2 搬运和计算分工不同的第一步。

本实现限制 `1 <= dv <= 128`，从而每个有效输出通道恰好对应一个线程；d 由点积循环处理，Nk 可以超过一个 tile。更大的 dv 需要每线程持有多个输出，不能只改输入尺寸。

对应 [硬件与存储图](<01_execution_and_memory.md#memory-paths>)，四个 warp 在同一 SM 上共同维护这一行：score/numerator 是线程私有状态，weights 和 m/l/alpha 用 block 的 shared 交接，Q/K/V/O 则是 global 数据。源码中显式 shared 数组与标量合计 `128*4 + 4*4 + 3*4 = 540` 字节，实际编译分配仍可能有对齐等差异。这里省去整张 S/P 的 global 存储，却没有消除片上状态的成本。

一个 block 只有一个 query，因此 grid 只有 Nq blocks。Nq 很小时，即使 Nk 很长、每个 block 工作很多，也不一定能让很多 SM 同时忙碌。这说明“算法正确且省显存”不等于“硬件并行度足够”，也为第 9 章比较 FA2 的工作划分做准备。

## 6.5 沿 kernel 跑一轮

1. 线程 t 计算 `Q[row]` 与 `K[base+t]` 的点积，乘 `1/sqrt(d)`。无效 key 提供 `-inf`。
2. `block_reduce<THREADS,true>` 得到 tile 最大值。整块全无效则所有线程一起跳过这一轮。
3. 线程 0 更新 m/alpha，全 block 同步后大家使用同一个基准。
4. 每线程计算一个指数权重，写 `weights[t]`；再次归约得到 tile_sum。
5. 线程 0 更新 l；线程 c 将自己的 numerator 乘 alpha，再累加本块 `weights[j]*V[base+j,c]`。
6. 全 block 同步，保证 weights 的读者结束，才允许下一轮覆盖。

`block_reduce` 内部的 barrier 也使 weights 的写入在第 5 步可见；若以后重写 helper，不能只保留数值归约而忽略这一同步依赖。检查一个 barrier 时，要列出它前后的内存读写，而不是只看它离哪行代码最近。

初始 m=-inf。第一块非空时新最大值有限，alpha=`exp(-inf)=0`，正确丢弃初始零状态。全 mask 块被跳过，因此不会出现 `-inf-(-inf)`；全 mask 行最终 l=0，按接口约定输出零。这里假设有效 score 有限。

## 6.6 改掉了哪些内存需求

在线 kernel 不接收 S/P 指针，不为整张分数或概率矩阵分配设备存储。它读取 Q/K/V，保存当前 tile 权重和跨轮状态，最后只写 O。

示例驱动 `check_attention` 为了在同一次运行中比较两种实现，**仍然为普通版本分配了 S/P**。观察整个进程显存时不会看到这两块消失；要比较峰值显存，需要分别运行两条路径并去掉不使用的分配。

我们也没有减少完整 dense attention 所需的有效 Q/K 配对，更没有证明这个朴素在线 kernel 比优化库快。它的目的在于隔离算法：先确保分块和归一化正确，再优化指令与数据搬运。

## 6.7 分块、遍历顺序与缩放的不变量

| 情形 | 结果及原因 |
|---|---|
| tile 大小为 1、2、7、32、128 | 与一次性归一化的参考在容差内一致 |
| 反向遍历 tile | 实数算法相同，浮点结果允许微小差异 |
| 输入包含全 mask 行、尾块和不同 dv | 空行输出零，边界不越界，输出 shape 不变 |
| 只缩放旧 l，不缩放旧 a | 分子分母使用不同指数基准，输出失去正确归一化 |
| 逐轮的 m/l/a 状态 | 能定位第一次偏离参考状态的轮次 |

CPU 脚本已包含分块尺寸/顺序组合验证。GPU 示例包含 Nk=129/257 的尾块、dv=17/128、等长 causal 和全 mask 行；还单列了最后一块最大值突然增大的例子：前 128 个 score 为 0、V 为 1，最后一个 score 为 ln(128)、V 为 3，正确输出为 2，以检查 GPU 代码是否同时缩放旧分子与分母。编译和 Sanitizer 命令见第 4 章。

在线状态更新始终围绕最大值、分母和加权分子三个量展开。接下来的 Tensor Core、CuTe 和流水改变执行方式；阅读时始终保留这三个状态作为算法对照。更多数学及 LSE 推导见 [在线 Softmax 导读](<../appendices/source/flash_attention/01_algorithm_and_online_softmax.md>)，它也说明真实代码的 `exp2`、scale 与延迟分母归约。
