# 推理 Runtime 架构对比

调度预算、KV 所有权和 worker 边界决定一个请求如何进入、推进和离开运行时。比较框架时沿同一组问题追踪状态，比只比较项目名称或吞吐数字更有解释力。

- [vLLM V1](vLLM-V1.md)：token budget、KV manager、Engine Core 与 worker。
- [TensorRT-LLM](TensorRT-LLM.md)：执行系统、backend 与服务组件的分工。
- [SGLang](../13_Source_Analysis/sglang/README.md)：本仓库固定版本的请求与控制面实现。

[技术演化](../papers/11_LLM推理Runtime技术演化总览.md)解释设计问题的变化，[推理性能](../12_Performance_Reliability/05_inference.md)限定负载和指标口径。文中的历史结构不等于每个框架当前版本的完整支持矩阵。

[总入口](../README.md)
