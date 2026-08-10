# 7.1 官方文档完整覆盖索引

## 1. 基线与口径

- SGLang commit：`f5155d960286db25952217f343ee0d3c358f7f77`
- 官方文档源码：`sglang/docs_new`
- 在线站点：<https://docs.sglang.io/>
- 导航来源：`sglang/docs_new/docs.json`

该快照共有 288 个 `.md`、`.mdx` 或 `.ipynb` 文件，约 5.3 MB：`docs/` 正文 174 篇、`cookbook/` 109 篇，其余为首页、说明、贡献指南和 demo。

本系列不是逐句镜像，而是按源码和工程问题重组。概念、调用链和不变量用中文解释；快速变化的 flags、默认值、支持矩阵和模型 recipe 保留官方路径，执行前核对同 commit 的原文与代码。

## 2. 官方分区到中文章节

| 官方分区 | 中文章节 |
|---|---|
| Get started | 1.2、3.2、3.3、4.1 |
| Basic usage | 1.2、4.1、4.2 |
| Supported models | 4.2、5.2 |
| Advanced features | 2.2–2.4、3.1–3.3、4.2–4.3、5.1–5.3 |
| References | 3.2、4.1、4.3、5.1、5.3 |
| Developer guide | 5.1–5.3 |
| Hardware platforms | 3.3 |
| SGLang Diffusion | 6.1–6.3 |

## 3. Advanced Features 逐主题映射

| 官方主题 | 中文章节 | 重点 |
|---|---|---|
| Attention backend | 2.3、5.2 | metadata 契约、backend 验证 |
| CUDA Graph / piecewise / breakable | 2.3 | shape、buffer、fallback |
| DP/DPA/router、EP | 3.1、3.2 | 一致调度、all-to-all、路由 |
| Pipeline parallelism | 3.1 | stage 与 bubble |
| PD/EPD、llm-d | 3.1、3.2 | KV ownership、传输和清理 |
| HiCache、HiSparse | 2.2 | 分层 KV、锁和驱逐 |
| Quantization / quantized KV | 2.2、2.3、3.3 | 精度、kernel 与平台 |
| Speculative decoding | 2.4 | draft/verify/accept、KV 回滚 |
| Structured output / reasoning | 2.4、4.1 | grammar、parser 与 sampling |
| LoRA、model loading、object storage | 4.3、5.3 | 版本、加载和失效状态 |
| VLM、multimodal DP/graph | 4.2 | encoder 与 LLM 分阶段观测 |
| Model gateway | 3.2 | 亲和路由、重试和容量 |
| Observability | 5.1 | metrics、trace、profiling |
| Deterministic inference | 2.4 | request-local RNG 与调度 |
| SGLang for RL / checkpoint engine | 4.3 | policy 版本与在线更新 |
| Forward hooks、R-Fork | 2.3、5.2 | 执行插桩和扩展 |

## 4. Cookbook 阅读方式

Cookbook 的 109 篇材料按模型组织，包括自回归模型、Diffusion、Omni/VLA、SpecBundle 和 benchmark。模型名及启动命令不翻译，用同一张检查表阅读：

| 检查项 | 记录内容 |
|---|---|
| Artifact | 模型 id/revision、镜像、SGLang commit |
| Hardware | 型号、节点、互联和显存 |
| Precision | weight/KV/activation dtype、量化格式 |
| Parallelism | TP/DP/EP/PP/SP/PD 和拓扑 |
| Workload | token 长度或分辨率/帧数/steps 分布 |
| Correctness | 数据集、参考实现、容差和 seed |
| Performance | 延迟、吞吐、阶段耗时和显存 |
| Limitations | backend、平台、LoRA、graph、cache 兼容性 |

LLM recipe 的原理回到第 2–5 部分；Diffusion recipe 回到第 6 部分。

## 5. 源码总导航

| 中文部分 | 主要源码/文档 |
|---|---|
| 1 基础与请求 | `srt/entrypoints/`、`srt/managers/` |
| 2 Runtime 核心 | scheduler、`srt/mem_cache/`、`srt/model_executor/`、sampling/speculative |
| 3 扩展与部署 | `srt/disaggregation/`、distributed、multi-node、hardware platforms |
| 4 接口与模型 | basic usage、supported models、post-training |
| 5 生产工程 | benchmark、observability、developer guide |
| 6 Diffusion | `python/sglang/multimodal_gen/`、`docs/sglang-diffusion/` |

以上 `srt/` 均相对于 `sglang/python/sglang/`，文档路径均相对于 `sglang/docs_new/docs/`。

## 6. 防止文档漂移

更新子模块后至少复核：新增导航主题、参数默认值、支持矩阵、弃用功能、符号重命名、性能结论的硬件/workload，以及 Cookbook 新模型是否引入新的 runtime 机制。

```powershell
rg --files sglang/docs_new | rg '\.(md|mdx|ipynb)$'
rg -n "ServerArgs|add_argument|SGLANG_" sglang/python/sglang
rg -n "class Scheduler|class ModelRunner|class ForwardBatch|class ReqToTokenPool" `
  sglang/python/sglang/srt
```

