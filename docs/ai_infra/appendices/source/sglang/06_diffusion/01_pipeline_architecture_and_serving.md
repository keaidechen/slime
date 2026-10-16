# 6.1 Pipeline 架构、服务接口与动态批处理

相关基础：[GPU、tensor 与通信基础](<../../../foundations/README.md>)。


## 入门例子：用一次图像请求对比文本生成

latent（潜在表示）是编码器或模型构造的中间表示，通常不直接等于像素、波形或 token ID。其形状、压缩率和解码成本需要与原始数据表示分开核算。

VAE（Variational Autoencoder，变分自编码器）学习编码与解码潜在表示；扩散系统中常在潜在空间生成，再经解码器得到媒体，潜在网格大小与原图分辨率不同。

文本 decode 往往每次推进一个 token；扩散采样则在若干去噪步中反复更新 latent，并可能前后连接文本编码器、VAE 或其他组件。因此 LLM 的 KV/page 调度直觉不能直接当作扩散主循环。

先列输入分辨率、batch、采样步数、编码器和输出解码，分别记录每阶段 shape 和驻留状态。动态 batching 还要判断不同请求的步数、尺寸与条件是否可兼容。

这是一条独立的扩散模型专题：pipeline 把文本编码、去噪与解码组合起来，关键路径由这些阶段的依赖和资源共同决定。它与 RL 文本运行时的差异在下文展开。

## 机制与实现

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

本组源码中的 `runtime/` 相对于 `sglang/python/sglang/multimodal_gen/`。DiT 是 Diffusion Transformer（扩散 Transformer），使用 Attention 等层处理连续特征；它属于反复调用的网络，scheduler 则根据其预测更新状态。

### 为什么需要反复更新 latent

图像不是逐个词表 token 顺序补出的答案。以潜在空间中的流匹配模型为例，训练让网络学会在给定噪声/时间位置和文本条件下预测“往哪里更新”；推理从噪声初始化的 latent 出发，沿时间网格多次调用网络，再由解码器转换为像素。这里把多种图像生成管线统称为 Diffusion，但它们可能预测噪声、干净样本或速度，不能交换采样公式。

先固定一个玩具配置：输出图像 `[B,3,H,W]=[1,3,8,8]`，VAE 的空间压缩倍率 f=2、latent 通道数 C=2，于是 latent 形状为 `[1,2,4,4]`。B 是 batch 大小，H/W 是输出高/宽，3 表示 RGB 通道数。Latent 有 32 个浮点数，而图像有 192 个；C 与 f 是本例假设，不是所有模型的默认值。文本经编码器得到条件 `c:[1,L_text,d_text]`，例如 `[1,3,4]`；`L_text` 是文本条件序列长度，`d_text` 是其特征维度。c 在这些采样步中可以保持不变，latent 则每步都变化。

若 DiT 把 latent 的每个 2×2 空间块打成一个 patch，patch 数 `N=(4/2)×(4/2)=4`，每个 patch 含 `C×2×2=8` 个数，模型输入可表示为 `[1,4,8]`，再投影到内部隐藏维度。Patch token 是连续向量，不是语言模型词表中的整数 ID。帧维存在时，还需说明视频的时间压缩与 patch 规则。

### 两步 Euler 更新：把采样器算一遍

仅对预测速度的简单流匹配 Euler 路径，设时间/噪声参数为 σ，网络输出 `v_θ(x,σ,c)`，θ 是固定模型权重。局部把变化率视为常量，即 `dx/dσ≈v`，在相邻网格间积分得到：

```text
x_next - x ≈ (σ_next - σ) × v_θ(x,σ,c)
x_next     = x + Δσ × v
```

σ 是无单位的采样坐标，不是墙钟秒；v 与 x 形状相同，表示每单位 σ 的 latent 变化量。取网格 `[1,0.5,0]`，只查看 32 个元素中的两个，初始 `x=[2,-1]`：第一步网络假设输出 `[1,-2]`，`Δσ=-0.5`，所以新值 `[1.5,0]`；第二步假设输出 `[0.4,0.2]`，得到 `[1.3,-0.1]`。这些速度是假设值，用于演示更新，不是实际生成结果。

CPU 组织步数与参数，GPU 执行网络和张量更新。下一步网络依赖本步更新后的 x，不能把所有步当作互不依赖的 batch。原地复用 latent buffer 前需确保所有消费者读完；多步求解器还可能保存若干次旧预测，不能统一丢弃历史。最后一步完成也不等于可交付图像，还要经过 VAE、后处理和输出传输。

本地 `runtime/models/schedulers/scheduling_flow_match_euler_discrete.py` 的普通确定性更新分支包含 `sample + dt * model_output`；随机采样、其他 prediction type 或特殊调度分支使用不同规则。这个例子用于认识依赖，不能代替具体模型 scheduler 配置。

## 2. 服务入口

官方文档提供一次性 `generate`、常驻 `serve`、配置文件/Python 组合和 OpenAI 兼容图像/视频 API。长视频通常是异步任务，要把生成、后处理、上传、可下载和失败清理建成明确状态。

每次请求记录模型与组件 revision、分辨率、帧数、steps、guidance、seed、dtype/量化、并行度和后处理。

## 3. 组件边界

Pipeline 可包含 tokenizer、text encoder、DiT/Transformer、VAE、image encoder、prompt enhancer、scheduler 和后处理器。组件路径覆盖用于替换实现/权重，attention backend 覆盖用于选择组件内部 kernel，两者不要混淆。

例如 `runtime/pipelines/flux.py:FluxPipeline` 声明 tokenizer、text encoder、transformer、VAE、scheduler 等组件，并在 `create_pipeline_stages()` 中组装阶段。沿这条链阅读时，将每次输出当成下一阶段的输入契约：文本编码得到条件 embedding，latent 准备产生初始状态，denoising 输出最终 latent，VAE 输出媒体张量。

| 状态 | 跨越哪些阶段 | 何时可释放或复用 |
|---|---|---|
| 文本条件 embedding | 编码后到去噪结束 | 最后一个使用它的网络操作完成后；跨请求缓存另查身份 |
| 当前 latent | 整个去噪循环直到 VAE | VAE 不再读取后 |
| 网络临时激活 | 通常一个网络 forward | 对应 stream 消费完后，推理不保留训练反向图 |
| 求解器历史/特征缓存 | 多个 step | 算法不再依赖或请求结束后 |
| 输出帧、编码 buffer | 后处理、视频编码、上传 | 下游复制/读取完成后 |

Offload（卸载）把暂时不用的组件权重移到 CPU 等位置，可以减少同时驻留的设备内存，但再次使用要搬回。若每步都搬运 DiT 权重，省显存的同时可能让传输成为主要耗时；只在阶段边界卸载 text encoder/VAE 的代价结构不同。

## 4. 动态批处理

Diffusion 请求的资源权重由分辨率、帧数、steps 和组件共同决定，不能只按请求数 batch。兼容性至少考虑 latent shape、timestep/scheduler、guidance 分支、LoRA、输出格式和取消状态。

大 batch 能复用权重读取和提高 GEMM 效率，也会让大请求拖慢小请求。容量测试必须模拟真实 image/video 分布。

本快照官方 `dynamic_batching.mdx` 明确描述的是：将**兼容的排队请求合并成一次 native pipeline batch**，与 LLM continuous batching 分开；它不承诺在任意去噪步插入新请求。默认 `--batching-max-size 1` 不合批，启用后仍受模型与配置规则限制。

例如 A/B 都是上述 `[1,2,4,4]` latent、相同步数和采样配置，可以合成 `[2,2,4,4]`，文本条件也按 batch 对齐；C 的 latent 若为 `[1,2,8,8]`，不能直接沿 batch 维拼入。即使 A/B shape 相同，也要检查 LoRA、guidance、输出处理及实现支持。每条请求的 seed/生成器状态和结果索引需独立，拆分输出时恢复原请求身份。

若等待合批最多 5 ms，单条执行 100 ms、两条合批 150 ms，假设连续满批，吞吐从约 10 张/s 提至 13.3 张/s，但这一批中先到请求可能等待约 155 ms 才拿到结果。低负载时凑不齐 batch，可能只增加等待。官方文档还明确本路径没有自动 OOM 重试或自动退回 singleton：合批失败会让相关请求失败，所以配置上限必须经过实际 shape 的容量验证。

## 5. 输出与后处理

插帧、超分、视频编码和对象存储可能成为端到端瓶颈。分别统计 denoising、VAE、后处理、编码与上传耗时，并给各阶段设置 timeout 和重试边界。

例如一个异步视频任务应区分 `queued→running→encoding→uploading→available`，失败或取消是另一路径。GPU 已结束时文件可能尚未生成，编码完成时对象存储地址也可能尚不可下载。任务状态由服务控制层推进，设备事件只证明设备工作完成。把“生成完成”与“交付可用”分开计时，才不会用较短的 GPU 时间冒充用户等待时间。

## 6. 官方入口

- `sglang/docs_new/docs/sglang-diffusion/index.mdx`
- `sglang/docs_new/docs/sglang-diffusion/installation.mdx`
- `sglang/docs_new/docs/sglang-diffusion/api/`
- `sglang/docs_new/docs/sglang-diffusion/dynamic_batching.mdx`
