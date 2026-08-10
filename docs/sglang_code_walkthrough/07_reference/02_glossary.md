# 7.2 中英文术语表

## Runtime 与性能

| 英文 | 译法 | 说明 |
|---|---|---|
| serving / runtime | 推理服务 / 运行时 | runtime 指执行与调度内核 |
| prefill / extend | 预填充 / 扩展 | 批量处理已有输入 token |
| decode | 解码 | 自回归地产生后续 token |
| continuous batching | 连续批处理 | 请求在迭代边界动态加入/退出 |
| TTFT | 首 token 延迟 | 请求到第一个输出 token |
| ITL | token 间延迟 | 相邻输出 token 间隔 |
| TPOT | 每输出 token 时间 | 首 token 后的平均处理时间 |
| throughput | 吞吐 | 单位时间请求或 token 数 |
| goodput | 有效吞吐 | 满足 SLO 的吞吐 |
| warmup | 预热 | 排除加载、编译、capture 等冷启动 |

## 内存与调度

| 英文 | 译法 |
|---|---|
| KV cache | KV 缓存 |
| paged KV cache | 分页 KV 缓存 |
| prefix cache | 前缀缓存 |
| cache eviction | 缓存驱逐 |
| cache hit | 缓存命中 |
| admission control | 准入控制 |
| backpressure | 背压 |
| retract / preemption | 回撤 / 抢占 |
| chunked prefill | 分块预填充 |
| overlap scheduling | 重叠调度 |

## 并行与部署

| 英文 | 译法 |
|---|---|
| tensor parallelism (TP) | 张量并行 |
| data parallelism (DP) | 数据并行 |
| expert parallelism (EP) | 专家并行 |
| pipeline parallelism (PP) | 流水线并行 |
| sequence parallelism (SP) | 序列并行 |
| disaggregation | 解耦部署 |
| PD disaggregation | 预填充—解码解耦 |
| collective | 集合通信 |
| replica | 副本 |
| rank / world size | 进程编号 / 总进程数 |

## 生成与模型

| 英文 | 译法 |
|---|---|
| speculative decoding | 推测解码 |
| draft / verify | 草稿 / 验证 |
| accept / reject | 接受 / 拒绝 |
| structured output | 结构化输出 |
| grammar constraint | 语法约束 |
| reasoning parser | 推理内容解析器 |
| fallback | 回退路径 |
| weight loader | 权重加载器 |
| weight tying | 权重共享 |

## Diffusion

| 英文 | 译法 |
|---|---|
| diffusion | 扩散生成 |
| denoising | 去噪 |
| latent | 潜变量 |
| timestep | 时间步 |
| scheduler | 采样调度器（注意与请求 Scheduler 区分） |
| DiT | Diffusion Transformer |
| VAE | 变分自编码器 |
| guidance / CFG | 引导 / 无分类器引导 |
| progressive resolution | 渐进分辨率 |
| feature cache | 特征缓存 |

API、类名、参数、环境变量、模型名和 kernel 名保持英文，首次出现时用中文解释。

