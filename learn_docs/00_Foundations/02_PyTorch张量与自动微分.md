# PyTorch 的系统视角：张量、存储、复制与自动微分

> 前置：[运行环境](01_Linux与运行环境.md)。目标：看见代码背后的存储与生命周期，而不只是数学表达式。预计 3–4 小时；前半课只需 CPU。

## 1. Tensor 不只是一个数值矩阵

一个 tensor 同时有数值和描述这些数值如何存放的元数据：shape、dtype、device、stride、storage offset。多个 tensor 可以指向同一块 storage，因此创建一个新 Python 变量不一定分配新的数据。

```python
import torch

x = torch.arange(12).reshape(3, 4)
y = x.t()
print(x.shape, x.stride())  # [3, 4], (4, 1)
print(y.shape, y.stride())  # [4, 3], (1, 4)
print(x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr())
y[0, 1] = 99
print(x[1, 0].item())      # 99：两个视图共享数据
```

对于二维 tensor，位置 `(i,j)` 对应的元素偏移约为 `storage_offset + i*stride[0] + j*stride[1]`。这里偏移单位是元素；换成字节还要乘 `element_size()`。

转置本身通常只改变元数据，但消费转置结果的算子可能需要另一种布局。这就是为什么“前一行很便宜，后一行突然慢”仍可能由前一行的布局决定。

## 2. view、reshape、contiguous 与 clone

| 操作 | 需要知道的语义 | 不能假定什么 |
|---|---|---|
| `view` | 在兼容 stride 时共享 storage，否则报错 | 任意转置后都能 view 成任意 shape |
| `reshape` | 能返回视图时返回视图，否则可能复制 | 一定零复制 |
| `contiguous` | 已满足所需连续布局时可返回原对象，否则复制 | 每次都会新分配 |
| `clone` | 建立数据副本 | 自动切断 autograd |
| `detach` | 切断这条 autograd 跟踪，仍可能共享 storage | 已复制数据或已释放显存 |

试着打印 `x.t().reshape(-1)` 与 `x.t().contiguous()` 的 storage 地址。性能报告应记录 shape 和 stride，否则两个“相同矩阵尺寸”的算子可能有不同的数据搬运成本。

## 3. 从一次训练理解生命周期

```text
参数 W、优化器状态：跨 step 保留
输入 batch：进入当前 step
forward：产生输出，保存 backward 需要的中间状态
backward：沿计算图产生梯度，按需释放 saved tensors
optimizer.step：用梯度更新参数与优化器状态
清理梯度/失去引用：为下一轮释放或复用存储
```

```python
import torch

w = torch.tensor([2.0], requires_grad=True)
x = torch.tensor([3.0])
loss = ((w * x) - 1).square().sum()
loss.backward()
print(w.grad)  # 2 * (2*3-1) * 3 = 30
```

再次构建同样的 loss 并 backward，梯度默认累加到已有 `w.grad`。所以 microbatch 梯度累积既涉及执行次数，也涉及 loss 缩放；`optimizer.step()` 不会自动替你清空梯度。

`retain_graph=True` 会保留本可释放的图资源，不能作为所有“第二次 backward 报错”的通用修复。保存 `loss` 或模型输出到长列表，也需要想清楚是否保留了图。仅用于记录时可保存脱离图的值，但 GPU 上 `.item()` 取标量可能引入同步，因此日志频率也会影响性能。

## 4. 三种显存数字

- **allocated**：PyTorch allocator 当前为活跃 tensor 等分配的内存。
- **reserved**：allocator 管理的内存池，包括可复用的空闲块。
- **设备整体 used**：还可能包含 CUDA context、通信库、其他进程和非 PyTorch 分配。

删除一个 tensor 的最后引用之后，allocated 可能下降而 reserved 不降。缓存分配器保留空闲块是为了下一次复用。`empty_cache()` 不能释放仍被活跃对象引用的内存，也不应无依据地放进每一步训练。语义见 [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)。

## 5. 一个明确假设的显存心算

假设有 10 亿个参数，训练使用 BF16 参数（2 B）、FP32 梯度（4 B）、FP32 主权重（4 B）、Adam 的两个 FP32 矩（8 B），这些常驻状态合计约 18 GB，约 16.76 GiB。

这里没有算 activation、临时 buffer、通信与碎片；也不是所有框架都采用这套 dtype 或同时保留这些副本。先列“对象×数量×每元素字节”，再谈 FSDP 或 offload 节省了哪一项。详见[显存核算](../02_Distributed_Communication_Memory/02_GPU内存组成与显存核算.md)。

## 6. 练习与验收

1. 完成上面的共享 storage 实验，解释为什么修改 y 会改变 x。
2. 不运行代码，手算一维例子的梯度，再用输出校验。
3. 在 GPU 上观察分配一个大 tensor、创建 view、clone、删除引用时 allocated/reserved 的变化。
4. 找到实际模型的一层，记录参数、activation 和梯度的 shape/dtype。

能回答“哪里复制了数据”“谁还持有引用”“backward 为什么需要中间状态”，再进入并行切分。下一课：[进程、线程与四种异步](03_进程线程与四种异步.md)。

参考：[Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)、[Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)。
