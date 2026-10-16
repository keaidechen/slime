# 3.3 硬件平台与安装选择

相关基础：[必要基础](<../../../foundations/README.md>)。


## 入门例子：记录实际选中的 backend

安装包成功不等于目标模型的所有 kernel 可用。先记录 GPU、驱动、框架、attention backend、dtype 和模型特性，再查看启动日志确认实际选择；fallback 能运行也不代表性能相同。

对 H20 不直接套用 H100 教程中的特定性能数字。实际可执行特性由硬件、编译目标、库和框架路径共同决定。遇到不支持先缩小到标准文本、基础 dtype 和普通 attention，再逐项加回。

启动日志中的模型类与 backend 说明实际选择的实现，固定输入连接数值与性能口径。其他平台的能力说明只提供比较参考，不能代替当前硬件的执行证据。

## 机制与实现

## 1. 安装方式是可复现性选择

| 方式 | 适合 | 必须固定 |
|---|---|---|
| pip/uv release | 使用稳定版 | package、torch、driver |
| nightly | 验证最新修复 | wheel 日期和完整依赖 |
| source | runtime/kernel 开发 | commit、子模块、编译器 |
| Docker/Compose | 单机可复现部署 | image digest、driver、设备映射 |
| Kubernetes/云 | 弹性和多节点 | 镜像、网络、调度与健康检查 |

统一记录 SGLang commit/package version、镜像 digest、模型 revision、驱动、CUDA/ROCm/CANN、torch 和 kernel 包版本。

安装链可以按普通编译程序理解：驱动让操作系统控制设备，CUDA/ROCm/CANN 等运行库提供设备执行接口，PyTorch 组织张量算子，SGLang 再调用专用 kernel 包。Wheel 是预编译 Python 分发包，JIT（Just-In-Time，即时编译）则在首次遇到代码或形状时生成执行产物。Python 能 `import` 只证明模块可加载，不证明某个 GPU 架构、dtype 和 shape 的专用 kernel 已可执行。

例如标准 dtype 的矩阵乘法可用，但某种低精度 attention kernel 未编译，服务可能在首次 decode 才失败或 fallback。验证链应从“导入→加载短模型→prefill→至少一次 decode→所需高级路径”逐步推进，每一步都记录实际实现。

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

### 为什么算力更高未必 decode 更快

设一次操作需 F 次浮点运算，搬运 D 字节，硬件在该精度下的有效计算速率是 P FLOP/s、有效内存带宽是 B byte/s。只看这两项，耗时下界为 `max(F/P,D/B)`；算术强度 `I=F/D` 的单位是 FLOP/byte，用于判断当前更受哪一项限制。实际还需加无法重叠的 launch、同步和通信。

教学例取 F=100 GFLOP、D=10 GB，设备 A 的 P=100 TFLOP/s、B=1 TB/s，分别需 1 ms 和 10 ms，下界是 10 ms。设备 B 的计算速率减半但带宽翻倍，则分别是 2 ms 和 5 ms，下界反而降至 5 ms。此处 G/T 使用十进制 `10^9/10^12`，没有混用 GiB/TiB。缓存命中、融合和 batch 会改变 D，因此不能只凭芯片峰值数字预测服务延迟。

## 4. 资源受限平台

CPU、Jetson 和 Apple Silicon 等平台通常更受内存容量、带宽、统一内存和可用 kernel 限制。优先缩小模型、上下文、batch 和静态预留，再评估量化。不能直接复用数据中心 GPU 的 TP、CUDA Graph 或显存比例经验值。

统一内存表示 CPU/加速器共享某种物理内存或地址管理体系，不代表访问没有带宽、同步和容量成本。权重从 16 位降为 8 位理论上能减半原始权重字节，但 scale、临时反量化和 activation 仍占空间，KV 也不会随权重精度自动减半。只有目标硬件有对应算子、模型误差满足要求且当前瓶颈确实受益，量化才可能同时改善容量与时延。


## 5. 平台迁移的相容条件

| 层 | 相容性证据 |
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
