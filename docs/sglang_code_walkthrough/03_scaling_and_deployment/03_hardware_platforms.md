# 3.3 硬件平台与安装选择

## 1. 安装方式是可复现性选择

| 方式 | 适合 | 必须固定 |
|---|---|---|
| pip/uv release | 使用稳定版 | package、torch、driver |
| nightly | 验证最新修复 | wheel 日期和完整依赖 |
| source | runtime/kernel 开发 | commit、子模块、编译器 |
| Docker/Compose | 单机可复现部署 | image digest、driver、设备映射 |
| Kubernetes/云 | 弹性和多节点 | 镜像、网络、调度与健康检查 |

统一记录 SGLang commit/package version、镜像 digest、模型 revision、驱动、CUDA/ROCm/CANN、torch 和 kernel 包版本。

## 2. 平台支持不是布尔值

官方文档覆盖 NVIDIA、AMD、Ascend、Intel XPU、TPU、Moore Threads、CPU、Apple Metal、Jetson 和插件系统。每个平台逐层核对：

1. 安装包和编译工具链；
2. dtype 与量化格式；
3. attention、MoE、norm 等 kernel backend；
4. graph capture 或编译能力；
5. collective 与多节点网络；
6. 模型、任务和高级功能支持矩阵；
7. profiling 工具和指标语义。

“可以启动”不等于与 CUDA 路径功能、性能等价。用相同输入和显式容差做数值回归，并单独记录 fallback。

## 3. Backend 选择顺序

```text
模型结构与算子
 -> dtype/量化
 -> 硬件与驱动能力
 -> 可用 kernel backend
 -> graph/compile 兼容性
 -> 代表性 shape benchmark
 -> 准确率回归
```

不要仅根据 GPU/NPU 型号硬编码 backend。head dimension、GQA/MLA、mask、sequence 长度、并行切分和量化都会改变最佳选择。

## 4. 资源受限平台

CPU、Jetson 和 Apple Silicon 等平台通常更受内存容量、带宽、统一内存和可用 kernel 限制。优先缩小模型、上下文、batch 和静态预留，再评估量化。不能直接复用数据中心 GPU 的 TP、CUDA Graph 或显存比例经验值。

## 5. 平台迁移验收

| 层 | 验收 |
|---|---|
| 安装 | 冷启动可复现、依赖无隐式漂移 |
| 模型 | 权重加载、tokenizer、参考输出 |
| Runtime | prefill/decode、KV、abort、长上下文 |
| 高级功能 | 量化、LoRA、graph、speculative、PD 等逐项声明 |
| 分布式 | collective、故障传播、多节点启动 |
| 性能 | 真实 workload 的延迟、吞吐、显存与功耗 |

## 6. 权威入口

- `sglang/docs_new/docs/get-started/install.mdx`
- `sglang/docs_new/docs/hardware-platforms/overview.mdx`
- `sglang/docs_new/docs/hardware-platforms/`
- `sglang/docs_new/docs/advanced_features/attention_backend.mdx`

