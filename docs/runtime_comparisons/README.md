# 推理 Runtime 架构对比

先完成 SGLang 普通请求主线，再用相同问题比较其他执行系统。比较请求状态、KV owner、调度预算、worker 边界与观测点，不凭项目名字判断速度。

- [vLLM V1](vLLM-V1.md)：token budget、KV manager、Engine Core 与 worker。
- [TensorRT-LLM](TensorRT-LLM.md)：执行系统、backend 与服务组件边界。
- [SGLang 概念课](../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md)：本课程主线。
- [技术演化总览](../paper_read/11_LLM推理Runtime技术演化总览.md)：论文与 Runtime 的不同层次。

原材料的版本与性能条件需要随实验核对。统一使用[推理性能教程](../performance_analysis_guide/05_inference.md)设计公平负载，回到[总学习清单](../../learn_docs/学习清单.md)安排时间。
