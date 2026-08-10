# 显存技术、确定性与 CUDA Graph

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
