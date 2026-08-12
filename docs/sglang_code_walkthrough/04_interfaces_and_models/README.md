# 第四部分：接口、模型与控制面

这一部分专门解释 runtime 两侧的边界：左侧是外部协议怎样变成内部请求，右侧是模型/权重怎样成为可执行状态。三章各自回答一组不同的问题，避免把 HTTP、模型加载和在线换权混成一条模糊的“serving 流程”。

## 章节职责

| 章节 | 核心问题 | 主要状态所有者 |
|---|---|---|
| [4.1 API、请求规范化与采样](01_api_request_and_sampling.md) | 用户参数最后变成什么？采样与 stop 在哪里生效？ | OpenAI serving、`GenerateReqInput`、`SamplingParams`、`SamplingBatchInfo` |
| [4.2 模型支持、多模态与 Fallback](02_model_support_and_multimodal.md) | 模型类如何选择？媒体如何变成 LLM 输入？不同 task 在哪里分流？ | `ModelRegistry`、model loader、multimodal processor、task serving |
| [4.3 控制面、在线权重更新与后训练](03_control_plane_and_post_training.md) | 换权如何隔离请求？失败是否可回滚？LoRA 如何安全上下线？ | `TokenizerManager`、weight updater、scheduler、`LoRARegistry` |

## 推荐阅读顺序

先完成 [1.2 进程拓扑与请求链路](../01_foundations/02_process_topology_and_request_path.md)，再读 4.1。模型接入或 VLM 问题读 4.2；RL、在线换权、动态 LoRA 和运维控制面读 4.3。

下面三个判断贯穿本部分：

1. 外部协议对象不是 scheduler 直接消费的对象，中间至少经历协议转换、输入规范化和 tokenization；
2. “模型可由 Transformers 加载”不等于“模型已走 SGLang 原生高性能路径”；
3. “控制面操作被串行化”不等于“跨 rank 更新具备事务回滚”。

每章末尾都有源码定位清单，路径相对仓库的 `sglang/python/sglang/`。
