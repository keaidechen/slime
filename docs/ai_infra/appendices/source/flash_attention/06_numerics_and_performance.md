# 数学等价、数值误差与执行成本

在线 Attention 的等价性来自归一化状态的可合并性，但真实 kernel 还受浮点舍入、线程布局与同步约束。本文从同一组数值出发，区分数学参考、GPU 实现和性能测量分别能说明什么。


## 1. 在线归一化的参考状态

[`online_attention.py`](<examples/online_attention.py>) 只依赖 Python 标准库。输入是单个 head 的二维列表，使用 Python 浮点数计算。它会输出：

```text
row=0, keys=[0, 1], m=0.693147, l=1.500000, a=[2.5]
row=0, keys=[2, 3], m=2.079442, l=1.875000, a=[6.125]
Output = 3.266666667; expected 49/15 = 3.266666667
```

参考程序中的函数与数学对象一一对应：

| 函数 | 对应知识点 |
|---|---|
| `dense_attention` | 一次性算完整 score 行、softmax、加权输出 |
| `online_attention` | 逐块维护 m/l/a，无需保存完整 score 行 |
| `visible` | 非方阵 causal mask 的右下对齐 |
| `combine_partials` | 通过 LSE 合并多个局部输出 |
| `reference_backward` | 第 04 节的梯度公式 |
| `self_check` | 与独立参考、有限差分对照 |

检查覆盖 64 组分块前向组合，包括 Q/K 长度不同、causal 与非 causal、正序与逆序、tile 大于或不能整除 K 长度。还检查 split 合并、大 score 稳定性和反向有限差分；`N_q>N_k` 的 causal case 包含完全遮蔽的行。

原稿记录这些检查在其文档编写环境中已运行通过，梯度有限差分最大绝对误差约 `2.13e-10`。这是 Python 数学实现的结果，不是上游 GPU kernel 的实测结果。脚本没有实现 GPU 线程、FP16 舍入、dropout、GQA 或 paged KV。

程序的前向与反向 `PASS` 分别对应公式参考中的断言，`AssertionError` 表示某项比较不成立。**有限差分**是从输入两侧各挪一点，估算变化率：`[loss(x+epsilon)-loss(x-epsilon)]/(2*epsilon)`。脚本把这个独立估计与反向公式的结果比较，因此它能发现公式错误，而不是仅仅检查某个函数有没有返回数组。前向等价性不自动保证反向实现正确，两类断言分别承担证据责任。

## 2. 分块、扫描顺序和坐标为何不能混为一谈

对本系列的四个 key 算例，tile size 为 1、2、3、8 时分别处理 4、2、2、1 轮；tile=3 的最后一块只有一个 key。实数运算下输出均为 `49/15≈3.266666667`，浮点计算可能存在舍入差异。分块改变中间状态与执行颗粒度，没有改变归一化的数学对象。

逆序扫描也成立。tile=2 时首先处理后一块，得到 `m≈2.079442、l=1.5、a=[5.5]`，再合入前一块。若新最大值改变时只保留旧 `a` 而不乘 correction，旧贡献仍在旧指数基准下，输出比例就会错误。这个反例说明 rescale 是算法本身，而非性能技巧。

线程坐标属于另一层映射。按布局篇的配置，线程 38 的 `warp=1、lane=6`，首次搬 Q 的局部行是 4、列是 48–55；当 `m_block=n_block=1`，其 score 行为 `145、153、209、217`，每行 key 列为 `132、133、140、141、…、252、253`。搬运坐标与持有 score 的坐标由不同 partition 定义，不能把它们当作同一组地址。

## 3. 同名包为何可能对应不同实现

Python 包、CUDA 扩展和本地源码可能来自不同位置。`flash_attn.__file__` 与 `flash_attn_2_cuda.__file__` 分别定位包装层和扩展；PyTorch、CUDA 版本与 GPU 决定构建兼容性。只有这些身份相容，执行结果才能对应到本文的源码快照。


以下属性分别描述 PyTorch 构建、CUDA 版本、实际设备、Python 包与二进制扩展的身份。它们相互补充：仅知道包名无法说明加载了哪个扩展。

```python
import torch
import flash_attn
import flash_attn_2_cuda

print("torch:", torch.__version__, "torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
print("Python package:", flash_attn.__file__)
print("CUDA extension:", flash_attn_2_cuda.__file__)
```

固定子模块的 [README](<../../../../../flash-attention/README.md>) 与 [setup.py](<../../../../../flash-attention/setup.py>)定义依赖。`csrc/cutlass` 是相关子模块；`FLASH_ATTENTION_FORCE_BUILD=TRUE` 强制本快照源码构建，`MAX_JOBS` 限制构建并发，`--no-build-isolation` 并不会自动补齐 PyTorch/CUDA 依赖。

原文记录的编写环境没有 `nvcc`，GPU 代码没有据此产生实测结果。本文保留数值参考与实现身份的区分，不把安装成功当作 kernel 正确性证明。

## 4. 用显式参考计算检查前向和反向

下面的代码展示两条数学路径怎样比较。参考路径显式计算 score、softmax、PV，并用 FP64 减少参考舍入误差。这里使用小尺寸、相同 Q/K 长度和无 mask 的基础场景；不要把它直接当作非方阵 causal 测试。

```python
import torch
from flash_attn import flash_attn_func

torch.manual_seed(0)
q, k, v = [
    torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.float16,
                requires_grad=True)
    for _ in range(3)
]
out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

qr, kr, vr = [x.detach().double().requires_grad_(True) for x in (q, k, v)]
# 先把 head 轴换到序列轴前面，得到 [B,H,N,D]，再做逐 head 矩阵乘。
qh, kh, vh = [x.transpose(1, 2) for x in (qr, kr, vr)]
scores = (qh @ kh.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
ref = (scores.softmax(dim=-1) @ vh).transpose(1, 2)

g = torch.randn_like(out)
grads = torch.autograd.grad(out, (q, k, v), g)
ref_grads = torch.autograd.grad(ref, (qr, kr, vr), g.double())

for name, actual, expected in zip(
    ("O", "dQ", "dK", "dV"), (out, *grads), (ref, *ref_grads)
):
    error = (actual.double() - expected).abs()
    print(name, "max abs error:", error.max().item(),
          "mean abs error:", error.mean().item())
    torch.testing.assert_close(actual.double(), expected, rtol=2e-2, atol=2e-2)
```

这里的容限只服务于这个初步 FP16 随机样例。正式验证应覆盖自己的 shape、数值范围和功能组合，并参考上游测试如何比较 FlashAttention、低精度参考与高精度参考。相对误差在参考值接近 0 时容易放大，因此同时查看绝对误差。

这段示例里，`detach()` 让参考输入不再依赖原输入的求导图，`double()` 转成 FP64，`requires_grad_(True)` 再为参考路径开启求导。`g` 是人为指定的 dO；两次 `autograd.grad` 都求同一个 `loss=sum(O*g)` 的输入梯度，才能公平比较。测试中的判定条件可理解为 `绝对误差 <= atol + rtol*abs(参考值)`。若失败，应先看是哪一项输出、误差多大，再查输入和分支；直接放宽容限无法说明原因。

## 5. 从上游测试选具体 case

[`tests/test_flash_attn.py`](<../../../../../flash-attention/tests/test_flash_attn.py>) 的参数化很多。参数化 node id 将形状与功能组合绑定到具体测试；`--collect-only` 展示这些组合：

```bash
python3 -m pytest flash-attention/tests/test_flash_attn.py --collect-only -q -k test_flash_attn_output
```

`tests/...py::test_name[...]` 标识一组确定参数。不同测试承担的边界如下：

| 测试 | 要理解的边界 |
|---|---|
| `test_flash_attn_output` | dtype、形状、前向与梯度 |
| `test_flash_attn_varlen_output` | 扁平 token、累计长度、padding 对照 |
| `test_flash_attn_varlen_causal` | Q/K 长度不同的 causal 语义 |
| `test_flash_attn_kvcache` | cache 追加、cache 长度、分页和位置编码 |
| `test_flash_attn_varlen_deterministic` | 确定性反向的重复结果 |

FA3 使用 [`hopper/test_flash_attn.py`](<../../../../../flash-attention/hopper/test_flash_attn.py>)，FA4 使用 [`tests/cute/`](<../../../../../flash-attention/tests/cute>)。它们对应不同安装和接口，不要仅凭测试名字相似就替换运行路径。

## 6. 正确计时：等待 GPU，而不只测 CPU 提交

下面片段接在第 4 节示例后，只计固定输入的前向；先将输入 detach，避免构造反向图：

```python
qi, ki, vi = [x.detach() for x in (q, k, v)]
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
repeats = 100
with torch.inference_mode():
    for _ in range(10):
        flash_attn_func(qi, ki, vi, dropout_p=0.0, causal=False)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        flash_attn_func(qi, ki, vi, dropout_p=0.0, causal=False)
    end.record()
end.synchronize()
print("forward milliseconds:", start.elapsed_time(end) / repeats)
```

warmup（预热）使初始化等一次性成本尽量移出计时；FA4 还要考虑首次配置的编译缓存。这个循环包括调用期间分配输出等相关开销，短 kernel 也可能受提交间隙影响；它适合初步比较，不能直接当作纯 Tensor Core 计算时间。

记录 GPU 型号、软件版本、输入 shape/dtype、mask、是否反向、warmup 和计时方法。长度 256→1024→4096 会改变访存与并行颗粒度；没有相同形状和测量边界，结果不能直接与论文峰值相除。

## 7. 出问题时按层缩小范围

JIT（Just-In-Time，即时编译）在运行时遇到程序与输入特征后生成代码，因此首次使用某个形状可能包含编译时间；稳态计时与冷启动计时应分开。

| 现象 | 优先检查 | 源码入口 |
|---|---|---|
| `ImportError` / 符号加载失败 | 包路径、扩展、PyTorch/CUDA 构建是否匹配 | Python import 与 setup |
| `TORCH_CHECK` 报错 | dtype、device、末维 stride、head 比例 | `flash_api.cpp::mha_fwd` |
| 只有 causal 非方阵不一致 | 参考是否右下对齐 | `flash_attn_func` docstring |
| 只有 varlen 不一致 | `cu_seqlens`、实际长度、max length | `BlockInfo` |
| 输出正确、梯度不一致 | scale、dO、dropout/RNG、允许误差 | backward 与上游测试 |
| 修改同步后随机错或挂住 | 数据就绪、缓冲区覆盖、屏障参与者 | 主循环与 pipeline |
| 第一次调用特别慢 | 初始化、JIT 编译、缓存 | FA4 `interface.py` / `cache_utils.py` |

如果将来要实际修改上游 kernel，请先遵循子模块自己的 [`AGENTS.md`](<../../../../../flash-attention/CLAUDE.md>) 调试要求。GPU 数据竞争、挂起、非法地址的诊断需要可复现输入和相应工具证据，单靠阅读公式无法证明某条同步可以删除。

## 8. 从数学等价到执行正确

Python 的 `[B,N,H,D]` 经指针和 stride 进入 C++，分发选择 tile 与 grid，线程块再按布局搬运 Q/K/V。在线 Softmax 的 m/l/a 是算法状态，线程片段和跨线程归约则是实现状态，两者需要在正确时刻对应。

LSE 让反向按块重建概率，split KV 用带权合并处理短 Q，流水中的等待保护数据就绪与缓冲区复用。算法结果相同、接口约束满足、并发执行正确和端到端更快是四层不同的判断。任一层缺少证据，都不能由另一层的结果补齐。
