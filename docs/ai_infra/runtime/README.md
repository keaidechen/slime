# 推理 Runtime 架构对比

调度预算、KV 所有权和 worker 边界决定一个请求如何进入、推进和离开运行时。比较框架时沿同一组问题追踪状态，比只比较项目名称或吞吐数字更有解释力。

- [vLLM V1](vLLM-V1.md)：token budget、KV manager、Engine Core 与 worker。
- [TensorRT-LLM](TensorRT-LLM.md)：执行系统、backend 与服务组件的分工。
- [SGLang](../13_Source_Analysis/sglang/README.md)：本仓库固定版本的请求与控制面实现。

[技术演化](../papers/11_LLM推理Runtime技术演化总览.md)解释设计问题的变化，[推理性能](../12_Performance_Reliability/05_inference.md)限定负载和指标口径。文中的历史结构不等于每个框架当前版本的完整支持矩阵。

[总入口](../README.md)

## 以功能为入口对照引擎

下面是查证路径，不是未经验证的支持勾选表。核对时固定 engine 版本/commit、模型、硬件、精度和功能组合，分别记录支持、不支持、未测试。

| 功能边界 | 通用原理 | 实现或官方核对入口 |
|---|---|---|
| 调度、KV 与阶段交接 | [KV](../10_Inference_Serving/推理内存系统.md)、[PD](../10_Inference_Serving/案例_Prefill-Decode分离.md) | [SGLang 快照](../13_Source_Analysis/sglang/README.md)、[vLLM V1 架构笔记](vLLM-V1.md) |
| 执行与功能组合 | [Graph/overlap/并行组合](../10_Inference_Serving/执行优化与功能组合.md) | [TensorRT-LLM 组合矩阵](https://nvidia.github.io/TensorRT-LLM/latest/features/feature-combination-matrix.html) |
| 量化 | [权重、激活与 KV](../10_Inference_Serving/量化推理与缓存精度.md) | [vLLM](https://docs.vllm.ai/en/stable/features/quantization/)、[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html) |
| Grammar、adapter、多模态 | [高级服务功能](../10_Inference_Serving/结构化生成与多模型服务.md) | [vLLM 结构化输出](https://docs.vllm.ai/en/stable/features/structured_outputs/)、[LoRA](https://docs.vllm.ai/en/stable/features/lora/)、[SGLang 快照](../13_Source_Analysis/sglang/README.md) |
| 路由、弹性与跨引擎编排 | [PD 交接](../10_Inference_Serving/案例_Prefill-Decode分离.md) | [Dynamo 官方文档](https://docs.nvidia.com/dynamo/latest/)；编排层与模型执行层分别核对 |

2026-10-12 核对官方 [TensorRT backend removal](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html)：当前 TensorRT-LLM 已移除 TensorRT engine 后端，PyTorch 为唯一后端。旧 `trtllm-build` 等流程仅用于理解历史版本，迁移边界见[TensorRT-LLM 第 13 节](TensorRT-LLM.md#13-从-tensorrt-engine-到-pytorch-执行后端)。本次核对不等于已用 GPU 验证所有功能组合。
