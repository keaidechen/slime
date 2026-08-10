# 4.1 API、请求规范化与采样

## 1. 接口分层

| 接口 | 场景 | 边界 |
|---|---|---|
| OpenAI 兼容 | 复用 SDK、网关和应用 | 协议兼容语义 |
| Anthropic/Ollama 兼容 | 对应生态客户端 | 只保证文档声明的子集 |
| SGLang Native | 特有参数和管理能力 | 客户端与 SGLang 耦合 |
| Offline Engine | 批处理、评测、同进程集成 | 调用方管理生命周期 |

协议不同不一定改变推理内核。先比较规范化后的内部请求，再定位 tokenizer、scheduler 或 model executor。

## 2. 请求规范化流水线

```text
HTTP/SDK payload
 -> 协议校验与默认值
 -> chat template / multimodal prompt
 -> tokenize
 -> SamplingParams / task params
 -> scheduler request
 -> task-specific result
 -> streaming / protocol response
```

chat template 会改变 token，不是展示格式。Sampling 默认值可能来自协议、模型配置或显式参数，调试时记录最终值。

## 3. Sampling 按作用点理解

| 阶段 | 参数类别 | 风险 |
|---|---|---|
| logits 变换 | temperature、penalty、logit bias | 顺序改变分布 |
| 候选截断 | top-k、top-p、min-p | 精度和顺序影响候选集 |
| 随机采样 | seed、RNG | continuous batching 下保持请求级语义 |
| 停止判定 | stop、max tokens | detokenize 与流式边界 |
| 附加输出 | logprobs | 增加计算、内存和传输 |
| 约束 | regex、JSON schema、grammar | 每 token mask 的 CPU 成本 |

完整字段与默认值只从当前 `sampling_params.mdx` 和代码读取。

## 4. Streaming 和错误语义

Streaming 只应改变结果送达时间，不应改变 token 序列。测试 disconnect、abort、timeout、stop 跨 chunk、空输出和 max length，并比较 token id、finish reason、usage 与错误码；只比较最终文本会漏掉边界错误。

## 5. API 测试矩阵

- Native 与实际使用的兼容协议；
- 流式/非流式、单条/批量；
- 文本、长 prompt、多模态和非法 schema；
- greedy、固定 seed、logprobs 和 grammar；
- abort、disconnect、timeout 和 shutdown。

## 6. 官方入口

- `sglang/docs_new/docs/basic_usage/`
- `sglang/docs_new/docs/get-started/quickstart.mdx`
- `sglang/docs_new/docs/references/custom_chat_template.mdx`

