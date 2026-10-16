# 6.3 Profiling、新模型接入与正确性

相关基础：[GPU、tensor 与通信基础](<../../../foundations/README.md>)。


## 入门例子：先定位阶段，再检查扩展后的结果

对一次 pipeline 分别计编码、去噪迭代、解码和输出 I/O。长阶段内再进入算子/Kernel；第一次编译与稳态迭代分开统计，避免把冷启动误当持续吞吐。

接入新模型后，先使用固定小输入验证 shape、dtype、状态生命周期和输出，再增加分辨率、并发与多卡。只比较一张随机生成图的肉眼观感不足以判断数值或质量正确性。

环境、输入配置与阶段 trace 说明成本发生在哪里，参考输出和评价则说明性能变化是否保留质量。通用 profiler 的观测边界见性能专题。

## 机制与实现

## 1. Profiling 路线

VAE（Variational Autoencoder，变分自编码器）学习编码与解码潜在表示；扩散系统中常在潜在空间生成，再经解码器得到媒体，潜在网格大小与原图分辨率不同。

先用阶段计时定位 text encoder、denoising、scheduler、VAE、后处理、通信或存储，再用 PyTorch Profiler 看 operator/CPU-GPU 关系，最后用 Nsight Systems 看 kernel、collective 和空洞。

Trace 只覆盖少量稳定 denoising steps，排除首次编译、加载和 warmup；另做端到端测试保留这些真实冷启动成本。

### 从阶段账本推导下一步优化

假设一次生成中，文本编码 20 ms、10 次去噪每步 12 ms、VAE 30 ms、后处理/上传 30 ms，且这些阶段串行，则总时间 `T=20+10×12+30+30=200 ms`。把每步去噪降到 6 ms，得到 140 ms，端到端加速 `200/140≈1.43` 倍，而不是两倍。

一般写成 `T_total=T_encode+sum_{i=1}^S T_step,i+T_decode+T_output`，S 是采样步数，每项单位 ms；有 overlap 时应改算依赖图的关键路径。不能把某一阶段的最快 step 乘 S 当平均总时间，因为首步、分辨率切换、缓存 miss 与末步可能不同。

GPU 时间用设备 event 或 profiler 读取，CPU 提交耗时只说明 launch 有多快。采样器的 timestep 是噪声/时间坐标，和 trace 的毫秒轴不是同一单位。记录第 i 步时最好同时保存 step index、σ/timestep、latent shape、是否缓存及相应设备区间，才能解释一条异常长的迭代。

## 2. 新模型最小接入

1. 选择结构最接近的参考 pipeline；
2. 定义 sampling params；
3. 定义 pipeline config 和组件；
4. 串联 pre-denoising、denoising 与 post-processing；
5. 注册模型；
6. 对齐参考实现的 shape、scheduler、seed 和输出；
7. 再加入服务、并行、量化、缓存和平台测试。

实际接入时，从 `python/sglang/multimodal_gen/runtime/pipelines/` 找相近组件组合，再沿 `pipelines_core/stages/denoising.py` 和模型专用 stage 追踪更新。注册成功只表明入口可以找到类；还需核对模型输出是噪声 ε、干净样本 x0 还是速度 v，以及 scheduler 期待哪一种。形状都与 latent 相同，所以传错预测类型可能完全不报 shape 错，却产生错误媒体。

采用 6.1 的 `[1,2,4,4]` 玩具 latent，先对齐初始噪声和条件张量，再对齐第一步网络预测、第一步更新、最后 latent 和 VAE 输出。若第一步网络已错，查权重/输入/后端；网络正确而更新错，查 timestep、prediction type、guidance 与 scheduler；latent 正确但像素错，再查 VAE scale、通道顺序和后处理。这样从最早差异找原因，比只查看最终图片更明确。

## 3. 正确性分层

| 层 | 验证 |
|---|---|
| 组件 | tensor shape、dtype、参考输出 |
| Pipeline | 固定 seed/scheduler/steps/分辨率的输出 |
| 优化 | 等价优化测误差；近似优化测质量 |
| 服务 | 并发、batch、取消、LoRA、异步任务 |
| 平台 | 每种硬件独立建立质量、性能和显存基线 |

### 误差怎样沿去噪循环传播

对 Euler 示例，参考更新 `x_next=x+Δσ v(x)`，新实现为 `x_hat_next=x_hat+Δσ v_hat(x_hat)`。令本步状态误差 `e=x_hat-x`，相减得到：

```text
e_next = e + Δσ × [v_hat(x_hat)-v(x)]
```

所以最终差异既包含旧误差，也包含“网络实现误差”和“输入已经偏移导致的预测差异”；一次 kernel 的小误差不必等于最终图像中的同等小误差。向量绝对误差可用平均绝对误差 `MAE=sum_i |x_hat_i-x_i|/n`，或最大绝对误差，n 是元素数；比较时必须先对齐 shape、dtype 和取值尺度。

例如参考两元素 `[1.3,-0.1]`，新实现 `[1.31,-0.08]`，MAE=`(0.01+0.02)/2=0.015`，最大误差为 0.02。这能描述数值偏移，不能单独证明图像细节、提示词符合度或视频时序质量合格。等价 backend 改动重点看数值；cache、降步数或量化改动还需一组固定 prompt/seed 的质量评价，不能只挑一张“看着不错”的图。

同 seed 也不必产生相同初始张量：生成器类型、设备、张量形状和随机数消费顺序可能改变。严格定位时可保存并复用初始 latent，而非只保存整数 seed。CFG 的条件/无条件分支也应分别对齐，避免相减放大误差后误判来源。

## 4. CI 性能基线

固定模型 revision、输入、seed、硬件、warmup 和采样参数。分别设置 correctness 与 performance gate；性能波动不能通过放宽正确性阈值解决。保存阶段耗时和峰值显存，避免端到端单值掩盖瓶颈迁移。

CI 是 Continuous Integration（持续集成），gate 是决定是否通过的验收门槛。可把基线拆成三项：同输入的数值/质量是否合格，阶段耗时与端到端耗时是否退化，峰值资源是否超出约束。比如去噪从 120 降至 80 ms，而 VAE 因 offload 从 30 升至 80 ms，端到端反而多 10 ms；只保存去噪平均时间会漏报。

服务回归还需在合批中取消一条请求，确认其他输出的索引/seed 不变、共享 forward 的消费者结束后才回收 buffer、后处理任务不会继续上传已取消结果。多卡取消不能让某个 rank 擅自跳过下一次 collective。重复运行取消与正常完成，比较可复用内存和在途任务计数，才能发现生命周期泄漏。

## 5. 支持矩阵的用法

兼容矩阵是“模型 × 平台 × 优化”的声明，不是组合正确性的证明。部署实际组合前仍运行最小生成、固定 seed 对齐、并发和显存测试，并记录 fallback 与特殊组件依赖。

## 6. 官方入口

- `sglang/docs_new/docs/sglang-diffusion/profiling.mdx`
- `sglang/docs_new/docs/sglang-diffusion/ci_perf.mdx`
- `sglang/docs_new/docs/sglang-diffusion/support_new_models.mdx`
- `sglang/docs_new/docs/sglang-diffusion/compatibility_matrix.mdx`
- `sglang/docs_new/docs/sglang-diffusion/contributing.mdx`
