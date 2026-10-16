# 显存技术、确定性与 CUDA Graph

相关基础：[通信与 tensor 基础](<../../../communication/两卡通信与torchrun.md>)。


## 入门例子：三种优化，三种不同的交换

Activation recompute 用更多计算换少存中间结果；offload 用数据搬运和同步换 GPU 驻留；CUDA Graph 用固定可重放的执行结构减少 host 提交开销。它们可能组合，但不能作为同一种“省显存开关”。

例如 forward 后把 activation 放到 CPU，backward 前必须及时取回；若预取晚到，GPU 会等待。CUDA Graph 则需要满足实际实现对地址、shape 和控制流的约束；新 shape 的编译/capture 与稳态重放要分开统计。

pinned memory（页锁定主机内存）在注册期间保持物理页稳定，可供某些异步 DMA 路径使用。它消耗主机资源，是否真正与计算重叠还取决于设备、stream 依赖和缓冲区生命周期。

RNG（Random Number Generator，随机数生成器）具有随抽样推进的内部状态。仅保存初始 seed 不能恢复已推进的位置，重计算和恢复训练通常还需保留状态及对应的消费顺序。

| 改动 | 正确性检查 | 性能证据 |
|---|---|---|
| recompute | RNG/dropout 与梯度是否符合预期 | 峰值减少量、额外 backward 计算 |
| offload | consumer 读取前搬运完成 | H2D/D2H、CPU pinned memory、等待 |
| graph | 输入更新与输出消费符合重放契约 | host launch 空洞、capture 次数、显存池 |

验收：先关闭优化建立对照，再一次打开一项。记录峰值和总 step，不能只报告其中改善的那个指标。

## 1. 三类技术解决三类问题

- recompute：少保存 activation，backward 重算；用计算换显存。
- activation offload：把 activation 搬到 host，backward 前搬回；用 PCIe/NVLink-C2C 带宽换显存。
- CUDA Graph：捕获稳定执行图，减少 CPU launch gap；通常增加静态 buffer 和 shape 约束。

它们能组合，但不是三个互不相关的开关。Graph capture 要求地址、控制流和 shape 足够稳定；offload 的异步 enqueue 和动态 MoE shape 可能改变捕获边界；recompute 会改变 forward/backward 的执行次数与 RNG 处理。

### 一条小计算图如何重算

设 `a=f(x), b=g(a), y=h(b)`，每个中间激活占 A bytes。普通反向可能要保留 a、b 直到各自反向节点用完。若把 f/g 所在区域设为 checkpoint，只保留边界输入 x，反向到达时再计算 f、g 来恢复 a、b，然后立即反向。少存的具体字节数由 saved tensors 决定，代价至少是被重放区域的额外前向。

例如原来前向 3 ms、反向 6 ms，若重算区域需要 2 ms，忽略其他成本时从 9 ms 变成 11 ms，增加 `2/9≈22.2%`。它仍可能提高整体吞吐：省出的显存若允许更大的 microbatch，可能让 GEMM 更高效。但固定 GBS/DP 时，增大 MBS 会减少累积批次数 M，反而可能增加 PP 气泡；收益需要端到端测量，不能只报告 activation 下降。

重计算必须重建同一个函数。设 dropout 概率 1/2，第一次对 `[1,2]` 的掩码是 `[1,0]`，输出为 `[2,0]`；重算若抽到 `[0,1]`，输出会变成 `[0,4]`，反向就不是原来那个随机函数的导数。因此要重放对应 RNG 状态，并避免把计数器、自定义缓存写入等副作用执行两次。

## 2. Fine-grained offload

当前实现可按 attention/MLP/MoE 子模块选择 offload，而非整层全搬。预算必须同时满足：可隐藏窗口、D2H/H2D 带宽、pinned pool、NUMA 距离和 inflight 数。若搬运暴露在 critical path，显存下降可能换来更差吞吐。

源码从 `megatron/core/pipeline_parallel/fine_grained_activation_offload.py` 和 pipeline schedule 中的 handler 生命周期开始读。

### offload 的最小依赖图

```text
GPU 计算产生 activation
  → D2H 复制到 pinned host buffer
  → 复制已完成且 GPU 不再需要原值：释放 GPU payload
  → 反向前预取 H2D 到 GPU buffer
  → H2D 完成事件：反向消费者读取
  → 最后消费者结束：GPU/host buffer 才可复用
```

更精确地说，若 host 副本不再被其他任务使用，H2D 读取完成后就可复用；GPU 副本则要等反向等设备消费者结束。上图给出保守的整组回收边界，实际内存池可以分别跟踪两者。

D2H 是 Device to Host，方向与 H2D 相反。copy stream 负责搬运，compute stream 负责算子，两者依靠事件传递“数据已准备好”。主机上的对象引用仍在，不代表 GPU 版本仍在；反过来，主机复制发出后立即覆写源 GPU tensor 也可能得到错误快照。

设搬运量 A=64 MiB、实测单向带宽 β=16 GiB/s，则一次复制至少 `A/β=(64/1024)/16 s≈3.91 ms`，来回约 7.81 ms，另加启动和排队。前向后有空档不保证能藏住返程：H2D 必须在对应反向需要它之前完成。多个 microbatch 同时 offload 会争用同一链路与 pinned pool；NUMA（非一致内存访问）描述不同 CPU/内存位置的访问代价，远端内存还可能增加搬运成本。

Optimizer CPU offload 搬运的是 optimizer 更新所需状态与参数，而不是 activation。GPU→CPU gradient copy、CPU step、CPU→GPU parameter copy都可能暴露；官方建议在适用配置下用 overlap 开关并行这些阶段。评审时分别核算 host memory、NUMA、CPU 算力和互联带宽，不能只看 GPU 显存下降。

## 3. CUDA Graph 三种范围

官方当前区分 local implementation、Transformer Engine implementation 与 full-iteration capture。local/TE 常按 layer 或 attn/mlp 模块捕获；full iteration 可进一步消除 launch gap，但要求固定 microbatch/shape、稳定调度，并对动态 MoE、offload 和某些重计算组合有额外限制。

验收不能只看 graph 成功 capture：比较 eager/graph 的前若干步 loss 与梯度，确认 replay 没有复用过期输入或 RNG 状态，再测 CPU gap 和端到端 step time。

把 capture/replay 理解成“记录一组设备任务及依赖，然后重复提交”。假设 100 个小 kernel 每个算 5 μs，CPU 每次提交的串行开销是 3 μs；在这些开销无法被隐藏的玩具模型中，eager 总计 800 μs。Graph 若把提交开销降到一次 10 μs，则约 510 μs，kernel 自身仍算 500 μs。实际 CPU/GPU 可并行，不能直接把此估算当成加速保证；大 GEMM 已遮住提交开销时，Graph 可能收益很小。

典型静态输入路径将新 batch 的值复制进预先分配的输入 buffer，再 replay；换一个 Python 变量指向新 tensor，不会自动改掉图捕获的设备地址。旧输出若位于重放会覆盖的 storage，下游须先消费或复制后再 replay。capture 的一次性耗时、图私有内存池和不同 shape 的多份图都应单独记账。

## 4. 确定性是全栈属性

bitwise 相同（逐位一致）要求同一浮点格式下每个结果的二进制位完全相同；满足数值容差只要求误差足够小，两种验收标准不同。

固定 seed 不等于 bitwise deterministic。还要固定数据顺序、并行布局、microbatch、kernel/通信算法、环境变量、gradient accumulation/reduction 次序与精度。PP 布局改变 LayerNorm 或 loss 所在 stage 时，归约顺序也可能改变。

验证方法：保存两次独立运行的逐 step loss、grad norm，必要时保存少量参数/梯度 hash；先在同硬件同拓扑比对，再讨论跨拓扑的数值容差。checkpoint 恢复测试应比较连续跑 N 步与 K 步保存、恢复后跑 N-K 步。

浮点加法本身也不满足严格结合律。以 FP32 为例，`(10⁸ + (-10⁸)) + 1` 得 1，而 `10⁸ + ((-10⁸)+1)` 中后一个括号的 1 可能被舍入掉，最终得 0。换归约树、TP/DP 分组或 bucket 顺序，都会改变相加顺序，即使样本、seed 和数学目标完全相同。

因此要先选择验收标准：相同软件硬件执行路径可追求逐位一致；改变并行布局时通常比较 `|a-b|≤atol+rtol·|b|`，其中 atol 是绝对容差、rtol 是相对容差。容差应匹配 dtype、数值尺度和累积误差，不能因结果不一致再随意放宽，也不能用一个参数 hash 不同就断言数学实现错误。

## 5. 组合矩阵

每引入一项优化，都至少跑：单卡 reference、目标并行小模型 correctness、save/load continuity、目标规模短性能测试。对 `{FP8, MoE, PP/VPP, recompute, offload, graph}` 建立显式组合矩阵；未测试的组合应视为未知，而不是默认兼容。
