# 推理与服务系统

从自回归生成和 KV 生命周期进入批处理、排队、采样、并行和阶段分离。

## 主题地图

| 条目 | 定位 |
|---|---|
| [Transformer执行与KV基础](Transformer执行与KV基础.md) | Transformer 执行与 KV 基础 |
| [推理内存系统](推理内存系统.md) | 推理内存系统：KV Cache、Paging 与复用 |
| [批处理排队与调度演进](批处理排队与调度演进.md) | 批处理、排队与推理调度的演进 |
| [服务准入路由与弹性控制](服务准入路由与弹性控制.md) | 多租户合同、工作与状态准入、缓存路由、反馈延迟与排空 |
| [采样与推测解码](采样与推测解码.md) | 采样、自回归依赖与推测解码 |
| [推理并行方案设计](推理并行方案设计.md) | 固定模型与混合到达，比较副本、TP、前缀、量化与 PD |
| [案例_Prefill-Decode分离](案例_Prefill-Decode分离.md) | 案例：Prefill–Decode 分离与 KV Cache 传输 |
| [Attention 与 FFN 分离的边界](Attention与FFN分离的边界.md) | 逐层激活往返、阶段配比与收益阈值 |
| [端边云部署与交互预算](端边云部署与交互预算.md) | 首播放时间线、网络阈值、端侧约束与回退 |
| [量化推理与缓存精度](量化推理与缓存精度.md) | 权重、激活与 KV 精度的容量、执行和质量边界 |
| [执行优化与功能组合](执行优化与功能组合.md) | CUDA Graph、编译、CPU/GPU overlap、长上下文与组合约束 |
| [结构化生成与多模型服务](结构化生成与多模型服务.md) | Grammar、Multi-LoRA、多模态、adapter 与缓存身份 |

## 完整推演与历史主线

| 条目 | 可以逐项核对的内容 |
|---|---|
| [手工推演 KV：分页、前缀共享、分叉与取消](<推理内存系统.md#数值展开-kv分页前缀共享分叉与取消>) | 6个物理页；共享、分叉、COW、取消与回收 |
| [手工推演运行时：请求队列、Token Budget、CUDA Graph 与结果返回](<批处理排队与调度演进.md#数值展开运行时请求队列token-budgetcuda-graph-与结果返回>) | 3个请求、token预算6；批次、Graph bucket与设备完成 |
| [服务容量与Prefill–Decode交接：算清页、时间和完成边界](<案例_Prefill-Decode分离.md#服务容量与prefilldecode交接算清页时间和完成边界>) | 24个KV页；准入、传输、交接和服务指标 |
| [推理运行时演进史：从执行一个batch到管理不断变化的状态](推理运行时演进史.md) | 连续批次、分页、前缀复用、分块、PD分离与执行图 |

全局入口：[手工推演索引](../手工推演索引.md) · [技术演进索引](../技术演进索引.md)。

## 读完本专题之后

减少重复计算与分配浪费后，调度、KV 传输和尾延迟成为关键；吞吐必须在延迟与质量条件下比较。



## 查阅与关联

[系列总览](../README.md) · [术语索引](../术语索引.md) · [问题索引](../问题索引.md)

## 贯穿到当前实现

[请求、执行与KV生命周期](../12_Performance_Reliability/贯穿案例_生成请求与KV生命周期.md)。

## 原理、源码与性能的对应入口

| 系统问题 | 通用正文 | 实现走读 | 性能或案例 |
|---|---|---|---|
| 请求、调度与 KV 生命周期 | [KV](推理内存系统.md)、[调度](批处理排队与调度演进.md) | [SGLang 固定版本](../13_Source_Analysis/sglang/README.md) | [生成与 KV 案例](../12_Performance_Reliability/贯穿案例_生成请求与KV生命周期.md) |
| 多副本服务的资源承诺 | [服务控制](服务准入路由与弹性控制.md) | [SGLang 部署与路由](../13_Source_Analysis/sglang/03_scaling_and_deployment/02_deployment_topology_and_routing.md) | [排队与容量](排队长尾与容量规划.md) |
| 执行、量化与功能组合 | [执行优化](执行优化与功能组合.md)、[量化](量化推理与缓存精度.md) | [跨引擎 runtime](../runtime/README.md) | [推理性能](../12_Performance_Reliability/05_inference.md) |
| Grammar、adapter 与媒体状态 | [高级服务功能](结构化生成与多模型服务.md) | [SGLang 固定版本](../13_Source_Analysis/sglang/README.md) | [功能组合算例](../12_Performance_Reliability/贯穿案例_生成请求与KV生命周期.md#功能组合算例前缀复用与-adapter-身份) |
| 长上下文与阶段交接 | [并行](推理并行方案设计.md)、[PD](案例_Prefill-Decode分离.md) | [runtime 对比](../runtime/README.md) | [PD 容量推演](案例_Prefill-Decode分离.md) |


## 内容深化与验证

[排队、长尾与容量规划](排队长尾与容量规划.md)：服务时间分布、调度时间线、开闭环压测和扩缩容。

## Omni 流式与多阶段服务

[流式生成与会话状态](Omni流式生成与会话状态.md)解释首音频、播放缺口、打断和历史；[多阶段运行时](Omni多阶段运行时.md)解释组批、交接、背压、取消和资源回收。前置表示与后续训练/评估见 [Omni 扩展路线](../guides/Omni扩展阅读路线.md)，真实实现见 [vLLM-Omni 固定版本](../runtime/vLLM-Omni.md)。
