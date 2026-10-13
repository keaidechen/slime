# vLLM-Omni：固定版本的多阶段请求走读

本篇将 [Omni 多阶段运行时](<../serving/Omni多阶段运行时.md>)对应到真实代码。阅读基线为公开仓库 **v0.14.0，commit `ed89c8b0436999e9210f11363f4eb512330a9dfa`**，2026-10-12 下载并静态核对。它不是当前最新版本说明，也没有在本机执行模型或 GPU 测试。

固定源码链接指向该 commit。[源码清单](<../../reference/omni_source_manifest.json>)记录所读 10 个文件的 SHA-256 与符号位置，[教学核对报告](<../../reference/omni_checks_20261012.json>)记录数值/状态模拟和文件身份核对结果。模型权重、vLLM 依赖和设备环境未组成可运行部署快照，不能用这些文件 hash 证明真实服务性能。

## 先读阶段配置，而不是猜统一接口

[Qwen3-Omni 阶段配置](https://github.com/vllm-project/vllm-omni/blob/ed89c8b0436999e9210f11363f4eb512330a9dfa/vllm_omni/model_executor/stage_configs/qwen3_omni_moe.yaml)划分三个阶段：

| stage | 模型角色 | 配置中的执行选择 | 对外输出 |
|---|---|---|---|
| 0 | Thinker | `worker_type: ar`、`OmniARScheduler`，设备 0 | text |
| 1 | Talker | `worker_type: ar`、`OmniARScheduler`，设备 1 | 作为后续输入 |
| 2 | Code2Wav | `worker_type: generation`、`OmniGenerationScheduler`，也在设备 1 | audio |

这里 Code2Wav 的 `stage_type` 仍是 `llm`；它的执行区别由 worker 和 scheduler 等配置共同决定。因而 `stage_type: llm` 不足以证明该阶段在做普通文本逐 token 解码。两个阶段可以共享一张设备，也不能由三个 stage 推出需要三张 GPU。

另有独立的 [async chunk 配置](https://github.com/vllm-project/vllm-omni/blob/ed89c8b0436999e9210f11363f4eb512330a9dfa/vllm_omni/model_executor/stage_configs/qwen3_omni_moe_async_chunk.yaml)，不仅打开 `async_chunk`，还指定两个 `custom_process_next_stage_input_func`。静态配置含有上游验证说明，但本篇没有复现其硬件结论。

## 构造阶段与提交请求

从 [OmniBase._initialize_stages](https://github.com/vllm-project/vllm-omni/blob/ed89c8b0436999e9210f11363f4eb512330a9dfa/vllm_omni/entrypoints/omni.py#L190)开始：加载模型或 YAML 的配置，构造 connector 与 `OmniStage`，建立各阶段队列，启动 worker 并等待 ready。默认多进程路径的队列设置为 `maxsize=0`，因此不能从这个入口推导出教学篇的“四块有界队列”。

[AsyncOmni.generate](https://github.com/vllm-project/vllm-omni/blob/ed89c8b0436999e9210f11363f4eb512330a9dfa/vllm_omni/entrypoints/async_omni.py#L239)为请求建立 `ClientRequestState`，检查 sampling 参数列表与阶段数相符，把 prompt 提交给 stage 0。之后按 `async_chunk` 分为两条执行路径。入口是 async generator，只表示它能异步产出对象，尚不能确定产出粒度和用户能否播放。

## 顺序路径：完成后构造下一阶段输入

在 `_process_sequential_results` 中，当前 stage 的结果可以陆续交给调用者；只有当前 stage 完成后，才通过下一 stage 的 `process_engine_inputs` 构造后续输入。这里“顺序”指阶段交接条件，不表示当前阶段必须只返回一次。

```text
stage 0 完成
 → 保存 stage 输出
 → 下一 stage 的 process_engine_inputs
 → 自定义 thinker2talker
 → connector.put 数据 + 队列通知
 → worker 接收 connector 数据
 → stage_engine.generate
```

[OmniStage.process_engine_inputs](https://github.com/vllm-project/vllm-omni/blob/ed89c8b0436999e9210f11363f4eb512330a9dfa/vllm_omni/entrypoints/omni_stage.py#L436)按 `engine_input_source` 找上游，或调用自定义转换。[thinker2talker](https://github.com/vllm-project/vllm-omni/blob/ed89c8b0436999e9210f11363f4eb512330a9dfa/vllm_omni/model_executor/stage_input_processors/qwen3_omni.py#L120)实际打包 embeddings、hidden states、token IDs 和 TTS 特殊嵌入等字段，并生成长度匹配的占位 prompt IDs。这是一份模型专用契约，不能解释为“把上游生成文字直接送入普通 TTS”。

这些字段的命名和传递不等于已证明模型在所有位置消费所有 hidden states；若研究架构语义，还应沿模型 forward 追踪实际选择。通用原理篇只说“文本与多模态条件”，避免把运行时载荷名当作统一模型定义。

## 数据传输与通知分开

[try_send_via_connector](https://github.com/vllm-project/vllm-omni/blob/ed89c8b0436999e9210f11363f4eb512330a9dfa/vllm_omni/distributed/omni_connectors/adapter.py#L24)先把输入和元数据交给 `connector.put`，成功后向下一阶段提交轻量通知；接收侧 `try_recv_via_connector` 根据边取回载荷。`OmniStage.submit` 还会把全局 request ID 放入附加信息，连接跨阶段身份。

一个容易误读的分支是 connector 失败：adapter 内的日志仍提到 fallback，但 `_process_sequential_results` 的调用者在发送失败时抛 `RuntimeError`。分析实际行为必须继续读调用者，不能仅根据注释或日志声称有队列回退。此处不会把“发送失败”默默当作“下游已收到”。

传输名也不证明 GPU 零拷贝。顺序转换存在 dtype/device 转换，async chunk 的部分字段显式 `.cpu()`；具体 connector、序列化和设备同步需要另查。教学篇的字节数是形状账本，不是此实现的实测通信量。

## async chunk 路径：下游早启动，数据逐块到达

`_process_async_results` 为各阶段消费独立结果队列，在收到 stage 0 的首个结果后，为后续阶段提交初始化任务；不是等 Thinker 整个回答结束再启动 Talker。配置中的 `thinker2talker_async_chunk` 和 `talker2code2wav_async_chunk`负责构造分块载荷，其中包含 `finished` 信息。

connector adapter 的 `get_chunk` 使用请求身份、上游 stage 和 chunk 计数查找载荷，区分新请求和已有请求。读代码时要同时追踪控制队列与块数据通道：下游“任务已经启动”，不代表这一轮要消费的所有条件已经到齐。

这条路径与客户端播放协议仍隔着服务封包和客户端缓冲。它不自动提供本系列教学协议里的播放 ACK、epoch、重复包处理或全双工决策；这些需要单独核验和端到端测试。

## 输出与取消各自承诺什么

`_process_single_result` 检查 stage 错误，并根据 `final_output` 与 `final_output_type` 包装 `OmniRequestOutput`。Thinker 的 text 和 Code2Wav 的 audio 都可以成为对外结果。stage 完成、全请求完成与音频实际播放结束仍是三个不同事件。

[AsyncOmni.abort](https://github.com/vllm-project/vllm-omni/blob/ed89c8b0436999e9210f11363f4eb512330a9dfa/vllm_omni/entrypoints/async_omni.py#L641)把 ABORT 任务提交给每个 stage；异步 worker 收到后用 `asyncio.create_task(stage_engine.abort(...))` 发起引擎取消。这里没有等待客户端播放停止的确认，广播返回也不等于所有设备引用已经释放。

入口捕获 `CancelledError` 和 `GeneratorExit` 时会调用 abort。通用异常、connector 失败、worker 死亡等其他退出路径还要分别追踪；不能把一个取消分支的存在解释为所有失败路径均有相同清理保证。

## 扩散分支：异步不等于逐步可见

[AsyncOmniDiffusion.generate](https://github.com/vllm-project/vllm-omni/blob/ed89c8b0436999e9210f11363f4eb512330a9dfa/vllm_omni/entrypoints/async_omni_diffusion.py#L111)把 `DiffusionEngine.step` 放入单 worker 的线程池，等待结果后返回。`generate_stream` 在这一版本只是等待 `generate` 完成再 yield 一次；不能把接口名字解释为逐去噪步输出或连续预览。

更关键的是，[DiffusionEngine.abort](https://github.com/vllm-project/vllm-omni/blob/ed89c8b0436999e9210f11363f4eb512330a9dfa/vllm_omni/diffusion/diffusion_engine.py#L374)只有未实现提示和 `pass`。虽然上层有 abort 方法，这一固定版本的该扩散引擎路径并未实现实际请求取消。教学案例中“停止未来迭代、等待在途步完成”的协议是设计要求，不能当作这里已具备的能力。

## 复查方法与阅读收获

在已有固定 checkout 上运行：

```bash
python3 docs/ai_infra/examples/omni_walkthroughs.py --source-root /path/to/vllm-omni
```

它检查清单中 10 个文件的内容 hash，不导入或执行上游代码，也不检查整个仓库所有依赖。读其他版本时，先重新核对入口分支、配置和下游行为，再生成新的独立证据；不要静默覆盖本快照来掩盖语义变化。

本次走读可建立三个对应关系：阶段配置决定角色与局部执行，connector 连接数据和请求身份，取消要一直追到实际执行层。下一步做真实实验时，按[评估篇](<../performance/Omni性能与质量评估.md>)采集请求、块和播放时间线，才能验证服务体验。

[多模态资料索引](<../models/README.md>) · [Runtime 对比](<README.md>)
