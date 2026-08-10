# 4.2 模型支持、多模态与 Fallback

## 1. 模型支持是能力矩阵

| 任务 | 输出 | Runtime 重点 |
|---|---|---|
| 生成式 LLM | token 序列 | KV、采样、停止条件 |
| 多模态语言模型 | token 序列 | encoder、占位 token、跨模态 shape |
| embedding | 向量 | pooling、归一化、batch shape |
| rerank | relevance score | pair 编码和排序语义 |
| reward/classification | score/class | 输出头与标签映射 |
| diffusion language model | 迭代生成结果 | 非自回归的迭代状态 |

模型能被 Transformers 加载，不代表 SGLang 原生支持。分别验证配置识别、权重映射、算子路径、并行切分、量化、长上下文、多模态预处理、准确率与 serving 并发。

## 2. Transformers Fallback

Fallback 扩大可运行范围，但不自动继承原生路径的全部优化。评审四点：

1. forward 能否接入 paged KV 和 SGLang attention backend；
2. TP 权重是否正确切分；
3. CUDA Graph、量化、LoRA、speculative 等是否兼容；
4. prefill 与 decode 是否和参考实现数值一致。

## 3. 多模态的两段计算

```text
media bytes/URL
 -> download/decode/preprocess
 -> vision/audio encoder
 -> embeddings / feature tokens
 -> LLM prefill
 -> autoregressive decode
```

encoder 和 LLM 的 shape、算力和缓存行为不同。监控至少拆分下载、预处理、encoder、prefill 和 decode，不能只看总 TTFT。

多模态 DP 与 encoder CUDA Graph 应独立评估：encoder batch 兼容性和 LLM continuous batching 不是同一问题。

## 4. 新模型接入顺序

1. 找结构最接近的现有实现；
2. 最小化 config、model class、weight loader 和 registry 改动；
3. 对齐参考实现的 logits/embedding/score；
4. 验证 prefill、decode、batch 和长上下文；
5. 再打开 TP、量化、graph、LoRA 等高级功能；
6. 最后跑 serving accuracy 与真实 workload benchmark。

## 5. 官方入口

- `sglang/docs_new/docs/supported-models/`
- `sglang/docs_new/docs/advanced_features/vlm_query.mdx`
- `sglang/docs_new/docs/advanced_features/dp_for_multi_modal_encoder.mdx`
- `sglang/docs_new/docs/advanced_features/cuda_graph_for_multi_modal_encoder.mdx`

