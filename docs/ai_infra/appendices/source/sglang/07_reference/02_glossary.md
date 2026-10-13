# 7.2 中英文术语表

相关基础：[Transformer 与 KV](<../../../../05_Inference/19_生成执行与采样.md#s1>)。


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
| SLO | 服务等级目标 | 可检验的质量、延迟或成功率门槛；需说明阈值、统计窗口与请求分母 |
| warmup | 预热 | 排除加载、编译、capture 等冷启动 |

## 内存与调度

| 英文 | 译法 | 在本运行时中的含义 |
|---|---|---|
| KV cache | KV 缓存 | 保存已计算位置的 Key/Value，供后续 Attention 读取，减少重复计算 |
| paged KV cache | 分页 KV 缓存 | 将逻辑 token 块映射到物理 KV 块，避免每条请求都要求一段连续显存 |
| prefix cache | 前缀缓存 | 复用相同有效前缀已计算的 KV；token、模型、位置和其他影响计算的身份必须相容 |
| cache eviction | 缓存驱逐 | 按策略移除可回收的缓存保留状态；仍被在途工作读取的存储不能提前复用 |
| cache hit | 缓存命中 | 查到语义相容且可用的缓存结果；命中请求比例与省下的 token 工作量不是同一指标 |
| admission control | 准入控制 | 在开始处理前检查 KV、token 或配额预算，决定接收、排队或拒绝请求 |
| backpressure | 背压 | 下游容量不足时限制上游继续生产或提交，避免无界积压 |
| retract / preemption | 回撤 / 抢占 | 暂停请求推进并回收部分执行资源，保存或重建恢复所需的 token、KV 与采样状态 |
| chunked prefill | 分块预填充 | 把长输入分多轮处理，限制单轮预算，以减轻对正在 Decode 的请求的阻塞 |
| overlap scheduling | 重叠调度 | 在当前设备工作期间准备后续批次；须保持依赖正确，重叠区间不直接等于端到端收益 |

## 并行与部署

| 英文 | 译法 | 在本运行时中的含义 |
|---|---|---|
| tensor parallelism (TP) | 张量并行 | 将同一层的张量计算分给多个参与者，并按分片或部分和语义合并结果 |
| data parallelism (DP) | 数据并行 | 推理时把请求分给多个模型副本；DP Attention 与 EP 组合时还需按模型布局交换数据 |
| expert parallelism (EP) | 专家并行 | 将不同专家放到不同设备，把路由后的 token 发给专家，再把结果送回对应位置 |
| pipeline parallelism (PP) | 流水线并行 | 把模型层分给不同阶段，阶段间传递当前批次的激活；推理不执行训练反向 |
| sequence parallelism (SP) | 序列并行 | 沿序列维拆分部分计算或激活；各实现覆盖范围不同，须核对是否包含跨片 Attention |
| disaggregation | 解耦部署 | 将不同执行阶段或组件部署到不同资源池，用交接协议连接状态与数据 |
| PD disaggregation | 预填充—解码解耦 | 分开调度 Prefill 与 Decode，前者产生的 KV 必须正确安装到后者后才能继续生成 |
| collective | 集合通信 | 指定进程组共同参与求和、汇集或交换等操作，操作顺序、形状与类型需相容 |
| replica | 副本 | 可独立承担模型请求的一份执行实例；一个副本也可以由多张 TP/PP GPU 组成 |
| rank / world size | 进程编号 / 总进程数 | rank 标识组内参与者，world size 计该组成员；进程编号不必等于全局 GPU 编号 |

## 生成与模型

| 英文 | 译法 | 在本运行时中的含义 |
|---|---|---|
| speculative decoding | 推测解码 | 先产生多个候选 token，再由目标模型验证，以减少逐 token 执行的串行轮数 |
| draft / verify | 草稿 / 验证 | 草稿提出候选，验证决定按目标生成规则保留多少；不能仅把草稿直接当最终输出 |
| accept / reject | 接受 / 拒绝 | 按算法接受候选，或在拒绝后校正采样并回收未采用分支的 KV；贪心与随机采样规则不同 |
| structured output | 结构化输出 | 输出遵守指定格式或 schema，例如 JSON；格式合法不自动证明内容正确 |
| grammar constraint | 语法约束 | 根据当前语法状态限制允许的下一 token，使生成沿合法语言规则推进 |
| reasoning parser | 推理内容解析器 | 按模型输出约定划分推理与回答片段；解析器不负责训练模型的推理能力 |
| fallback | 回退路径 | 原生模型类或优化实现不可用时选择替代实现，需重新验证支持范围、数值与性能 |
| weight loader | 权重加载器 | 将文件中的参数名、形状、分片和格式映射到运行模型，名称匹配之外还需保持张量语义 |
| weight tying | 权重共享 | 多个逻辑模块使用同一参数，例如 embedding 与输出头；加载、更新和显存计数需保持共享关系 |

## Diffusion

| 英文 | 译法 | 在本运行时中的含义 |
|---|---|---|
| diffusion | 扩散生成 | 从噪声等初始状态沿采样时间网格反复更新表示，最后生成图像、视频或其他媒体 |
| denoising | 去噪 | 网络在给定时间步和条件下预测更新所需的量；它可能是噪声或速度，不能一律解释为直接输出最终图像 |
| latent | 潜变量 | 模型内部的中间表示，通常需经解码器转换为媒体，尺寸不必等于原始像素网格 |
| timestep | 时间步 | 采样路径上的噪声或时间参数；网络调用次数与墙钟耗时应分别记录 |
| scheduler | 采样调度器 | 决定时间网格以及如何根据网络预测更新状态；请求 Scheduler 则决定哪些请求何时执行 |
| DiT | Diffusion Transformer | 使用 Transformer 组织扩散网络计算，一次完整生成通常要反复调用网络 |
| VAE | 变分自编码器 | 学习潜在表示的编码与解码；潜在空间生成后，媒体解码仍有工作量与延迟 |
| guidance / CFG | 引导 / 无分类器引导 | 组合有条件与无条件等预测来调节条件影响，可能增加网络执行量并改变输出分布 |
| progressive resolution | 渐进分辨率 | 不同阶段采用不同表示尺寸，减少部分早期计算；尺寸切换与质量影响需由具体算法验证 |
| feature cache | 特征缓存 | 复用中间特征；跨去噪步的输入可能已变化，因此须区分精确复用与允许误差的近似复用 |

API、类名、参数、环境变量、模型名和 kernel 名保持英文，首次出现时用中文解释。
