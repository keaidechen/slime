# 8. 从同步加载到异步双缓冲

[上一章](07_mma_and_cute.md) · [目录](README.md) · [下一章](09_read_fa2.md)

本章把 Attention 的数学暂时移开，用 [async_tiles.cu](examples/async_tiles.cu) 单独学习“谁可以在什么时候写入、读取、复用一块 shared memory”。这项实验与上一章的 WMMA/CuTe 独立，可逐个编译。

## 8.1 单缓冲为什么会等待

第 2 章矩阵乘每轮是：加载 → barrier → 计算 → barrier → 下一轮加载。第一道 barrier 保证计算读到新数据，第二道保证加载不会覆盖还在使用的旧数据。

如果加载耗时 L，计算耗时 C，单缓冲的稳态每轮大致需要 L+C。双缓冲的目标是在计算当前块时准备下一块，使两者重叠；理想稳态才可能靠近 `max(L,C)`，实际还受资源、依赖和提交开销影响。

多分配一个数组并不自动得到重叠。需要可以异步发起的搬运、独立的可用缓冲区，以及准确的就绪条件。

这里的加载是 **GPU global→当前 SM 的 shared**，不是 CPU 经 PCIe 把数据送到 GPU。第 1 章的“等待时调度另一个 warp”与本章的“发起下一块搬运后继续当前计算”是两种可以结合的隐藏延迟方式：前者需要就绪 warp，后者需要独立工作和安全的缓冲区。

## 8.2 先学会写一张时序表

两个槽位叫 A/B，处理四个 tile：

| 阶段 | A 槽位 | B 槽位 | 当前计算 |
|---|---|---|---|
| prologue | 加载 tile 0 | 空 | 无 |
| 第 0 轮 | tile 0 就绪、被读 | 发起 tile 1 加载 | tile 0 |
| 第 1 轮 | 发起 tile 2 加载 | tile 1 就绪、被读 | tile 1 |
| 第 2 轮 | tile 2 就绪、被读 | 发起 tile 3 加载 | tile 2 |
| 第 3 轮 | 可复用 | tile 3 就绪、被读 | tile 3 |

开始读某槽位前，必须等该槽位的搬运完成；开始覆写某槽位前，必须等其所有读者结束。这是两个不同条件：**数据已经到达，不代表旧的消费者已经离开。**

## 8.3 完整小实验做什么

每个 block 处理四块输入，每块 B=128 个元素。输出是 tile 内相邻两个输入之和：

```text
y[tile*B+t] = tile_value[t] + tile_value[(t+1)%B]
```

邻居按 tile 内循环，最后一个不足 B 的 tile 先补零。`n=129` 时，第二块只有一个有效元素，该元素的右邻居是补的零；不是重新绕回这一个有效元素。

一个 block 的四个 warp 共同使用 `slots[2][128]`，显式 shared 预算为 1 KiB；单槽位则是 512 字节。两个槽位都属于同一个 block、由同一个 SM 的资源承载，不是两个 SM 各分担一个槽位。更大的实际算子使用双缓冲时，这一存储翻倍可能减少驻留 block，必须把流水收益与 [资源预算](01_execution_and_memory.md#residency)一起看。

这个简单运算故意读取其他线程加载的值，使跨线程的可见性真正成为必要条件。它不适合证明性能收益，计算量太小，只用来练习同步。

小实验采用 `<cuda_pipeline.h>` 的 primitives：

| 操作 | 本例如何理解 |
|---|---|
| `__pipeline_memcpy_async(dst,src,4)` | 发起当前线程的 global→shared 拷贝 |
| `__pipeline_commit()` | 提交当前一批拷贝；所有线程在收敛点调用 |
| `__pipeline_wait_prior(0)` | 等当前线程此前提交的所有批次完成 |
| `__syncthreads()` | 让整个 block 的生产/消费阶段协调好 |

本例每轮消费前只有需要等待的当前批次，所以使用 wait_prior(0)。更深流水可以保留较新的 pending 批次，但必须根据实际提交序列证明安全，不能把 0 随意改成 1。官方语义与示例见 [CUDA Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)。

## 8.4 沿代码检查一次槽位交换

完整文件首先将 tile 0 发入 `slots[0]`。循环内按以下顺序执行：

```cpp
__pipeline_wait_prior(0);
__syncthreads();
// 若还有下一块，发起它到另一个槽位的拷贝并 commit。
// 从当前槽位读取自己和邻居的数据，计算输出。
__syncthreads();
```

第一道 barrier 之前，各线程先等待自己的拷贝；barrier 之后，当前块所有元素均可被其他线程读取。下一块进入另一槽位，与当前计算存在重叠机会。末尾 barrier 保护当前消费者；下一轮才可以把已经读完的槽位用于更后面的输入。

边界线程也执行 commit、wait 和 barrier。仅拷贝动作按有效性分支：无效位置写零，不构造越界的 global 读。每个 block 执行相同数量的 tile 轮次，最后全无效的 tile 也不会写出越界结果。

## 8.5 编译和验证

下面命令面向 SM80 示例目标；目标 GPU 必须兼容，其他架构请相应修改 `-arch`。本实验要求 compute capability ≥8.0，且 Toolkit 提供对应 API：

```bash
nvcc -O2 -lineinfo -std=c++17 -arch=sm_80 docs/cuda_tutorial/examples/async_tiles.cu -o /tmp/slime_async_tiles
/tmp/slime_async_tiles
compute-sanitizer --tool memcheck /tmp/slime_async_tiles
compute-sanitizer --tool racecheck /tmp/slime_async_tiles
compute-sanitizer --tool synccheck /tmp/slime_async_tiles
```

尺寸覆盖 1、127、128、129、511、512、513、1031，分别验证 tile 边界、block 边界及多个 block。程序逐元素与 CPU 对照，不报告虚构的加速比。若要测性能，先建立同步版本与更有代表性的工作负载，再使用第 3 章的方法。

## 8.6 再看 FA2 的等待与预取

FA2 当前配置并不是把上面的两个相同槽位机械替换为 Q/K/V。其基本循环可以理解为：

```text
等当前 K → 发起当前 V → 算 QK
等当前 V，并确认旧 K 可覆写 → 发起下一块 K → 在线 Softmax、算 PV
```

这里利用的是 QK 不需要 V、PV 不需要下一块 K 这两个独立窗口。`cp_async_fence` 提交相关拷贝组，`cp_async_wait<0>` 等待，block barrier 协调不同线程的使用与复用。不能把名字中的 fence 误读成“所有数据已经到达”。

具体等待条件与 Q 是否在寄存器、shared 是否复用、首尾块等配置有关。固定配置的两轮执行见下一章，再到已有 [流水精读](../flash_attention_code_walkthrough/03_layout_and_pipeline.md) 核对寄存器预取等细节。

## 8.7 练习与通过标准

| 思考或修改 | 应得到的解释 |
|---|---|
| 删除 consumer 之后的 barrier | 某线程可能覆写另一个线程还在读的槽位 |
| 只留下 block barrier，删除异步 wait | 到达 barrier 不等于异步搬运已经完成 |
| 所有拷贝发出后立即全部等待，再计算 | 可能保持正确，但失去目标中的搬运/计算重叠 |
| 三个槽位是否必然更快 | 不保证；shared 占用增加可能减少驻留并行度 |

前两项应先在纸上画出可能的竞争；若在实验副本中实际删同步，仅偶然运行成功不能作为正确证据。通过标准是能为每块缓冲区写出“谁写、谁读、何时就绪、何时可复用”四个答案。
