# 显存技术、确定性与 CUDA Graph

<!-- learning-position -->
> **学习定位**：A3/A6 · 必修。
> **前置**：[通信与 tensor 基础](<../../../learn_docs/00_Foundations/06_两卡通信与torchrun.md>)。
> **首读/二读**：offload、确定性、CUDA Graph 的具体生命周期与限制。
> **进度与实验**：[学习清单](<../../../learn_docs/学习清单.md>) · [总入口](<../../../learn_docs/README.md>)。
<!-- /learning-position -->

<a id="beginner-example"></a>

## 入门例子：三种优化，三种不同的交换

Activation recompute 用更多计算换少存中间结果；offload 用数据搬运和同步换 GPU 驻留；CUDA Graph 用固定可重放的执行结构减少 host 提交开销。它们可能组合，但不能作为同一种“省显存开关”。

例如 forward 后把 activation 放到 CPU，backward 前必须及时取回；若预取晚到，GPU 会等待。CUDA Graph 则需要满足实际实现对地址、shape 和控制流的约束；新 shape 的编译/capture 与稳态重放要分开统计。

| 改动 | 正确性检查 | 性能证据 |
|---|---|---|
| recompute | RNG/dropout 与梯度是否符合预期 | 峰值减少量、额外 backward 计算 |
| offload | consumer 读取前搬运完成 | H2D/D2H、CPU pinned memory、等待 |
| graph | 输入更新与输出消费符合重放契约 | host launch 空洞、capture 次数、显存池 |

验收：先关闭优化建立对照，再一次打开一项。记录峰值和总 step，不能只报告其中改善的那个指标。

## 机制与实现

## 1. 三类技术解决三类问题

- recompute：少保存 activation，backward 重算；用计算换显存。
- activation offload：把 activation 搬到 host，backward 前搬回；用 PCIe/NVLink-C2C 带宽换显存。
- CUDA Graph：捕获稳定执行图，减少 CPU launch gap；通常增加静态 buffer 和 shape 约束。

它们能组合，但不是三个互不相关的开关。Graph capture 要求地址、控制流和 shape 足够稳定；offload 的异步 enqueue 和动态 MoE shape 可能改变捕获边界；recompute 会改变 forward/backward 的执行次数与 RNG 处理。

## 2. Fine-grained offload

当前实现可按 attention/MLP/MoE 子模块选择 offload，而非整层全搬。预算必须同时满足：可隐藏窗口、D2H/H2D 带宽、pinned pool、NUMA 距离和 inflight 数。若搬运暴露在 critical path，显存下降可能换来更差吞吐。

源码从 `megatron/core/pipeline_parallel/fine_grained_activation_offload.py` 和 pipeline schedule 中的 handler 生命周期开始读。

Optimizer CPU offload 搬运的是 optimizer 更新所需状态与参数，而不是 activation。GPU→CPU gradient copy、CPU step、CPU→GPU parameter copy都可能暴露；官方建议在适用配置下用 overlap 开关并行这些阶段。评审时分别核算 host memory、NUMA、CPU 算力和互联带宽，不能只看 GPU 显存下降。

## 3. CUDA Graph 三种范围

官方当前区分 local implementation、Transformer Engine implementation 与 full-iteration capture。local/TE 常按 layer 或 attn/mlp 模块捕获；full iteration 可进一步消除 launch gap，但要求固定 microbatch/shape、稳定调度，并对动态 MoE、offload 和某些重计算组合有额外限制。

验收不能只看 graph 成功 capture：比较 eager/graph 的前若干步 loss 与梯度，确认 replay 没有复用过期输入或 RNG 状态，再测 CPU gap 和端到端 step time。

## 4. 确定性是全栈属性

固定 seed 不等于 bitwise deterministic。还要固定数据顺序、并行布局、microbatch、kernel/通信算法、环境变量、gradient accumulation/reduction 次序与精度。PP 布局改变 LayerNorm 或 loss 所在 stage 时，归约顺序也可能改变。

验证方法：保存两次独立运行的逐 step loss、grad norm，必要时保存少量参数/梯度 hash；先在同硬件同拓扑比对，再讨论跨拓扑的数值容差。checkpoint 恢复测试应比较连续跑 N 步与 K 步保存、恢复后跑 N-K 步。

## 5. 组合矩阵

每引入一项优化，都至少跑：单卡 reference、目标并行小模型 correctness、save/load continuity、目标规模短性能测试。对 `{FP8, MoE, PP/VPP, recompute, offload, graph}` 建立显式组合矩阵；未测试的组合应视为未知，而不是默认兼容。
