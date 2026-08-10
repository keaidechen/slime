# 6.1 Pipeline 架构、服务接口与动态批处理

## 1. 与 LLM Runtime 的根本差异

LLM decode 每步产生 token 并增长 KV；Diffusion 在 timestep 上反复更新 latent，再解码为图像或视频。

```text
prompt / media
 -> tokenizer + text encoder / prompt enhancement
 -> latent preparation
 -> denoising loop: timestep -> DiT -> scheduler update
 -> VAE decode
 -> post-processing
 -> storage / API response
```

性能必须按阶段拆分，只优化 denoising 不保证端到端收益。

## 2. 服务入口

官方文档提供一次性 `generate`、常驻 `serve`、配置文件/Python 组合和 OpenAI 兼容图像/视频 API。长视频通常是异步任务，要把生成、后处理、上传、可下载和失败清理建成明确状态。

每次请求记录模型与组件 revision、分辨率、帧数、steps、guidance、seed、dtype/量化、并行度和后处理。

## 3. 组件边界

Pipeline 可包含 tokenizer、text encoder、DiT/Transformer、VAE、image encoder、prompt enhancer、scheduler 和后处理器。组件路径覆盖用于替换实现/权重，attention backend 覆盖用于选择组件内部 kernel，两者不要混淆。

## 4. 动态批处理

Diffusion 请求的资源权重由分辨率、帧数、steps 和组件共同决定，不能只按请求数 batch。兼容性至少考虑 latent shape、timestep/scheduler、guidance 分支、LoRA、输出格式和取消状态。

大 batch 能复用权重读取和提高 GEMM 效率，也会让大请求拖慢小请求。容量测试必须模拟真实 image/video 分布。

## 5. 输出与后处理

插帧、超分、视频编码和对象存储可能成为端到端瓶颈。分别统计 denoising、VAE、后处理、编码与上传耗时，并给各阶段设置 timeout 和重试边界。

## 6. 官方入口

- `sglang/docs_new/docs/sglang-diffusion/index.mdx`
- `sglang/docs_new/docs/sglang-diffusion/installation.mdx`
- `sglang/docs_new/docs/sglang-diffusion/api/`
- `sglang/docs_new/docs/sglang-diffusion/dynamic_batching.mdx`

