# 6.3 Profiling、新模型接入与正确性

## 1. Profiling 路线

先用阶段计时定位 text encoder、denoising、scheduler、VAE、后处理、通信或存储，再用 PyTorch Profiler 看 operator/CPU-GPU 关系，最后用 Nsight Systems 看 kernel、collective 和空洞。

Trace 只覆盖少量稳定 denoising steps，排除首次编译、加载和 warmup；另做端到端测试保留这些真实冷启动成本。

## 2. 新模型最小接入

1. 选择结构最接近的参考 pipeline；
2. 定义 sampling params；
3. 定义 pipeline config 和组件；
4. 串联 pre-denoising、denoising 与 post-processing；
5. 注册模型；
6. 对齐参考实现的 shape、scheduler、seed 和输出；
7. 再加入服务、并行、量化、缓存和平台测试。

## 3. 正确性分层

| 层 | 验证 |
|---|---|
| 组件 | tensor shape、dtype、参考输出 |
| Pipeline | 固定 seed/scheduler/steps/分辨率的输出 |
| 优化 | 等价优化测误差；近似优化测质量 |
| 服务 | 并发、batch、取消、LoRA、异步任务 |
| 平台 | 每种硬件独立建立质量、性能和显存基线 |

## 4. CI 性能基线

固定模型 revision、输入、seed、硬件、warmup 和采样参数。分别设置 correctness 与 performance gate；性能波动不能通过放宽正确性阈值解决。保存阶段耗时和峰值显存，避免端到端单值掩盖瓶颈迁移。

## 5. 支持矩阵的用法

兼容矩阵是“模型 × 平台 × 优化”的声明，不是组合正确性的证明。部署实际组合前仍运行最小生成、固定 seed 对齐、并发和显存测试，并记录 fallback 与特殊组件依赖。

## 6. 官方入口

- `sglang/docs_new/docs/sglang-diffusion/profiling.mdx`
- `sglang/docs_new/docs/sglang-diffusion/ci_perf.mdx`
- `sglang/docs_new/docs/sglang-diffusion/support_new_models.mdx`
- `sglang/docs_new/docs/sglang-diffusion/compatibility_matrix.mdx`
- `sglang/docs_new/docs/sglang-diffusion/contributing.mdx`

