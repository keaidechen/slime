# 6.2 并行、Backend、缓存与量化

<!-- learning-position -->
> **学习定位**：A8 · 参考。
> **前置**：[GPU、tensor 与通信基础](<../../../learn_docs/00_Foundations/README.md>)。
> **首读/二读**：Diffusion 并行/cache/量化按相关任务查阅。
> **进度与实验**：[学习清单](<../../../learn_docs/学习清单.md>) · [总入口](<../../../learn_docs/README.md>)。
<!-- /learning-position -->

<a id="beginner-example"></a>

## 入门例子：把缓存与量化的收益写成可比较条件

减少采样步数、复用中间结果和降低精度都会影响执行成本，但也可能改变输出质量。比较时固定 prompt、种子、尺寸、步数与评价方式，并明确哪些量是实验变量。

并行可以切组件、batch 或内部计算，通信成本与显存收益不同；先画 latent/条件张量在哪里，再讨论多卡。缓存则要说明何时可复用、何时因条件变化必须失效。

验收：给出速度、显存与质量三项结果及适用条件。LLM 的 token/s 与扩散的请求/图像或采样步耗时不能直接比较。

## 机制与实现

## 1. 并行和解耦

Sequence Parallel 切分 token/patch 序列，适合高分辨率图像和长视频；CFG parallel 可分离条件/无条件分支；TP 切分权重；组件解耦把 text encoder、DiT 和 VAE 放入不同资源池。

选择并行度时同时测阶段计算、collective、pipeline 空洞和跨阶段传输。Diffusion 解耦传输 tensor/latent，LLM PD 传输 KV，容量和失败模型不同。

## 2. Attention Backend

选择受硬件、dtype、attention 变体、head/sequence shape、并行方式和 mask 影响。对每个主要 shape 记录实际 kernel、warmup 后耗时、workspace、峰值显存、参考误差和平台限制。

Sliding Tile Attention 等稀疏方法可能改变输出质量，不能与严格等价优化混在同一组结论中。

## 3. 缓存加速

Cache-DiT、TeaCache 利用相邻 timestep 特征变化有限，近似跳过或复用计算；它们不是 LLM prefix cache。

| LLM prefix cache | Diffusion feature cache |
|---|---|
| 相同 token 前缀复用 KV | 相邻 step 近似复用特征 |
| 通常不改变语义 | 可能引入质量误差 |
| key 是离散 token | 判定依赖特征差异、step 和模型 |

报告阈值、命中/跳步率、固定 seed 质量和加速比。

## 4. Progressive Resolution

早期 timestep 使用较低空间/时间分辨率，后期恢复完整分辨率。收益依赖模型、输出尺寸、切换策略和 VAE 占比。这是近似优化，必须与 full-resolution 基线做同 seed 对比。

## 5. 量化

先明确量化组件、weight/activation/accumulation dtype、敏感层跳过规则、硬件 kernel，以及与并行、LoRA、cache 和 graph 的兼容性。图像/视频质量评估不能只看数值误差。

官方量化章节涵盖 FP8、FP4/MXFP4、Nunchaku/SVDQuant、ModelOpt 和 ModelSlim；具体支持以当前兼容矩阵为准。

## 6. 优化分类

| 类别 | 示例 | 验收 |
|---|---|---|
| 尽量等价 | backend、并行、通信优化 | 数值/视觉对齐 |
| 精度变化 | 量化 | 质量门槛与性能 |
| 近似计算 | feature cache、progressive resolution | 同 seed 质量与收益曲线 |

## 7. 官方入口

- `attention_backends.mdx`
- `ring_sp_performance.mdx`、`disaggregation.mdx`
- `caching-acceleration.mdx`、`cache_dit.mdx`、`teacache.mdx`
- `progressive_resolution.mdx`、`quantization.mdx`

以上路径均位于 `sglang/docs_new/docs/sglang-diffusion/`。
