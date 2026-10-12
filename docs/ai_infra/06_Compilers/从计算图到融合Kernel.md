# 从计算图到融合 Kernel：一个表达式的执行与字节账本

前置：[编译器分层](编译器分层与执行图.md)、[GEMM 与分块](../05_Kernels/GEMM与高性能Kernel模式.md)。本篇固定表达式，区分数学图、调度、设备代码和实测；不以示意 kernel 代替某个版本的编译输出。

## 三个算子怎样变成一次循环

取长度 N 的 FP32 连续数组 X、A、B，三者无别名，计算 Y=max(X×A+B,0)。三个输入均为数组，不能按标量广播的带宽计算。

未融合的示意图为 multiply→add→relu，前两步各写一个长度 N 的临时张量。若每步都从 HBM 读写、没有缓存复用：

| 阶段 | 读取 | 写入 | 元素访问数 |
|---|---|---|---:|
| T₁=X×A | X、A | T₁ | 3N |
| T₂=T₁+B | T₁、B | T₂ | 3N |
| Y=max(T₂,0) | T₂ | Y | 2N |
| 融合 | X、A、B | Y | 4N |

FP32 字节流量从理想 32N 降到 16N，kernel 提交次数从示意的三次降到一次。缓存命中可能让实际 HBM 差距更小；小 N 可能主要节省 launch，大 N 可能主要节省搬运。这些都是待测假设，不是保证加速两倍。

融合循环的伪代码如下；临时值留在线程局部寄存器中，而不是写成完整数组：

```text
i = program_id * BLOCK + lane
if i < N:
    x, a, b = load(X[i]), load(A[i]), load(B[i])
    y = max(x * a + b, 0)
    store(Y[i], y)
```

地址与尾块 mask 由布局决定。寄存器是否 spill、乘加是否合成 FMA，要查看生成代码与资源报告。FMA 可能改变舍入次数，见[数值反例](../01_Math_Numerics/浮点数与低精度.md)。

## 编译链条各自做了哪件事

Python 捕图把执行中的张量操作表达为图，并为相关 shape、dtype、stride 或 Python 状态建立适用条件。图级优化可消去多余节点、决定融合区域；降低到循环／tile 后安排索引、向量化与内存访问；后端再完成设备指令选择、寄存器分配和机器代码生成。PyTorch 各层的实际分工见[编译栈](PyTorch编译栈.md)与 [torch.compiler 文档](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial)。

对于本例，只需要每个位置自己的三个输入。换成矩阵乘、归约或 Attention，就增加跨元素协作，tile、同步和共享内存成为新的调度选择。公式能写成一行不代表适合一个 kernel。

## 反向图还需要保存什么

若 U=X×A+B，输出梯度为 G，并约定 ReLU 在零点导数取零，则 H=G·1[U>0]，dX=H×A、dA=H×X、dB=H。X/A 是原输入，U 的符号可以保存 mask 或重算；取舍包括保存字节数、重算 FLOPs 和输入生命周期。

前向中间张量被消去，不代表整个训练图不需要状态。图编译也不能随意删除为反向所需的信息。若 B 实际是标量或广播向量，dB 还需要归约；本篇按逐元素 B 定义，不能把这个梯度直接用于广播版本。

## 哪些变化会阻止同一执行方案复用

| 改动 | 新问题 |
|---|---|
| 输入变成非连续 view | 地址公式与合并访问改变；可能需要另一种调度 |
| Python 分支依赖 tensor 的值 | 需要支持的图内控制流，或产生图边界 |
| T₁ 被其他消费者使用 | 消掉 T₁ 可能增加重复计算，需比较两种代价 |
| 输入与输出存在别名或原地写入 | 必须维护读写顺序和可观察副作用 |
| 融合区域不断扩大 | 寄存器、代码体积和并行度可能恶化 |
| N 改变 | 动态形状支持、guard 和特化决定是否重编译 |

Graph break、重新编译和 kernel 不融合是三件事。CUDA Graph 则重放设备工作提交，不能自动完成上述代数和存储变换。

## 在真实编译器上如何取证

下面是需要 PyTorch 的独立实验，本次 macOS 标准库环境未执行：

```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
def fn(x, a, b):
    return torch.relu(x * a + b)

inputs = [torch.randn(1 << 20, device=device) for _ in range(3)]
compiled = torch.compile(fn, fullgraph=True)
torch.testing.assert_close(compiled(*inputs), fn(*inputs), atol=1e-5, rtol=1e-5)
```

先保存软件版本、输入布局和生成代码，再确认临时张量、kernel 数及资源占用。将首次编译时间单独计时；稳态预热后分别测端到端和设备执行。用 profiler 验证实际 kernel，使用多组 N 和 stride 检查原假设是否成立。正确性对照需要同时覆盖反向、尾块与别名边界；上面的片段只检查当前前向输入。

不要把上面选用的容差当作所有表达式的通用阈值。接近 ReLU 零点时很小的数值差异也可能改变分支，因此前向误差和梯度误差应分别报告。

[编译器目录](README.md) · [Kernel 性能](../05_Kernels/Kernel性能与错误成因.md) · [PyTorch 性能分析](../12_Performance_Reliability/02_pytorch.md)
