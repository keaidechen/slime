# Nsight Compute：Kernel 级分析

Nsight Compute（命令行 `ncu`）针对单个 CUDA kernel 收集硬件计数器、指令、memory workload、scheduler 状态与 source correlation。它回答“这个 kernel 为什么没有接近硬件上限”。

## 1. 先缩小目标

```bash
ncu --set basic \
    --kernel-name regex:my_kernel \
    --launch-skip 20 \
    --launch-count 3 \
    -o kernel_report \
    python workload.py
```

从 `basic` 或少量 section 开始；`--set full` 可能需要多次 replay，开销和报告体积很大。动态服务应先构造可复现离线输入，不要直接对全流量长时间 profile。

## 2. 四步读报告

### 步骤 A：确认工作量

先核对 grid/block、register/thread、shared memory/block、输入 shape、dtype、调用次数。形状不同的同名 kernel 不应混为一个平均值。

### 步骤 B：看 Speed of Light

Speed of Light（SOL，理论极限占比）比较 SM compute 与 memory subsystem 的吞吐占比：

- compute 高、memory 低：更可能 compute-bound；
- memory 高、compute 低：更可能 memory-bound；
- 两者都低：可能 latency、依赖、低并行度、小 grid 或 launch-bound。

“两者都低”不能直接得出 kernel 很差；一个只运行几微秒的小工作量没有足够时间进入稳态。

### 步骤 C：看 Occupancy 与 Wave

Occupancy（占用率）是 active warps 相对硬件上限的比例。限制可能来自 register、shared memory、threads/block 或 blocks/SM。

高 occupancy 不保证快；低 occupancy 也不一定坏。高寄存器 tiling 可能降低 occupancy，却提高数据复用和 Tensor Core 吞吐。真正问题是是否有足够 eligible warps 隐藏 latency。

### 步骤 D：看 scheduler stall 与 memory access

| 现象 | 含义候选 | 优化方向 |
|---|---|---|
| long scoreboard | 等待 global/local memory dependency | 合并访问、prefetch、提高复用 |
| short scoreboard | 等待 shared memory 或特殊单元 | 调整流水、bank/layout |
| barrier | block 内同步等待 | 减少 barrier、均衡 warp 工作 |
| not selected | 有其他 eligible warp 被选中 | 未必是问题 |
| no instruction / dispatch stall | 前端/依赖/供给不足 | 看指令混合与 dependency chain |

stall reason 是症状，不是修复建议。例如 long scoreboard 可能来自真实 HBM latency，也可能是 local memory spill；必须结合 memory table 与 source line。

## 3. Roofline

Arithmetic Intensity（AI，算术强度）定义为：

$$AI = \frac{\text{FLOPs}}{\text{bytes moved from target memory level}}$$

可达性能上界：

$$P \le \min(P_{peak},\ AI \times BW)$$

其中 $BW$ 是带宽。ridge point（屋脊转折点）为 $P_{peak}/BW$。点在左侧通常受带宽限制，右侧通常受计算限制。

关键细节：AI 取决于所选 memory level。相同 kernel 相对 HBM 可能高 AI，相对 L1/L2 却有不同结论；FLOPs 口径也应匹配 dtype 与指令类型。

## 4. 一个简化例子

对向量逐元素 `y = a*x + y`，每元素约 2 FLOPs，却至少读 `a/x/y`、写 `y`，AI 很低，通常 memory-bound。对大型矩阵乘，tile 被反复复用，AI 随 tile/矩阵尺寸提高，通常更可能 compute-bound。于是：

- 前者优先 fuse 邻接算子、减少中间 tensor 和 bytes；
- 后者优先 Tensor Core dtype/layout、tile、pipeline 与 occupancy。

## 5. Baseline comparison

在 GUI 或 CLI 中保存 baseline，比较的不是“某个百分比更高”而是：

1. work 是否相同；
2. kernel duration 是否下降；
3. bytes、instructions、waves、occupancy 如何变化；
4. 改动是否把瓶颈从一个 subsystem 推到另一个。

如果新版本少做了工作或精度改变，counter 更漂亮没有意义。

## 6. 常见陷阱

- 只盯 theoretical occupancy，盲目减少寄存器导致 spill；
- 将 DRAM bandwidth 低解释为“不 memory-bound”，忽略 latency-bound random access；
- profile 首次 JIT/autotune kernel；
- 把 communication kernel 当普通 compute kernel 优化；
- 对 CUDA Graph replay 中的同名 kernel 不核对 graph node 与 shape；
- 采集全 section 后用被严重扰动的 duration 做端到端结论。

## 7. 2026 工具边界

Nsight Compute 2026.2 支持 CUDA 13.3，并提供更细的 SASS instruction size 等统计。指标名称和可用性依 GPU architecture 变化，不能把 H100 上的阈值机械套到 B200、Rubin 或非 NVIDIA accelerator。

## 资料

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Kernel 专题：GEMM 与高性能模式](../01_Kernel_GPU_Programming_Compiler/03_GEMM与高性能Kernel模式.md)
