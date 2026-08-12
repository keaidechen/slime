# 术语表

| 缩写/术语 | 中文解释 | 不要混淆 |
|---|---|---|
| MCore | Megatron Core 可组合训练库 | 不等于端到端 Megatron-LM 应用 |
| DP | 数据并行，切 batch | Distributed Optimizer/FSDP 是 DP 状态管理方式 |
| TP | 张量并行，切单层 tensor | 高频通信，通常局限高速互联域 |
| SP | 序列并行，降低 TP 区域非张量并行 activation | 不等于长上下文 CP |
| PP/VPP | 流水线/虚拟流水线，切 layer/chunk | VPP 不增加物理 rank |
| CP | 上下文并行，切 attention 的 sequence/context | 通信发生在 attention 语义内 |
| EP | 专家并行，切 MoE experts | 通常伴随 token dispatch/all-to-all |
| MBS/GBS | micro/global batch size | `GBS=MBS×DP×microbatches`（常规情形） |
| ModuleSpec | 模块结构与实现选择的描述 | 不是序列化 checkpoint schema |
| Distributed Optimizer | 在 DP 组分片 optimizer/主参数等状态 | 不等同完整 FSDP 生命周期 |
| FSDP | 可配置分片 optimizer、gradient 与 training parameter | 参数也分片时，unit 粒度决定 materialize 峰值和通信 |
| DCP | Distributed Checkpointing | 重点是 global state 到 shard 的映射 |
| TE | NVIDIA Transformer Engine | 提供 kernel/FP8 等实现，不负责完整训练循环 |
| recompute | backward 前重算 activation | 与把 activation 搬到 CPU 的 offload 不同 |
| CUDA Graph | 捕获并 replay 稳定 CUDA 工作图 | 不自动优化 GPU kernel 本身 |
| MLA | Multi-Latent Attention | 与 MoE/MTP 是不同架构维度 |
| MTP | Multi-Token Prediction | 训练目标/额外模块，不是推理 speculative decoding 本身 |
| critical path | 决定 step 完成时间的暴露路径 | profiler 中累计耗时最大者未必在关键路径 |
