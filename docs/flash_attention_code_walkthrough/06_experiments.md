# 06 · 动手实验、验证与排错

本节分为 CPU 数学实验和 GPU 实现验证。二者检查的内容不同：CPU 实验能验证分块公式，GPU 实验才覆盖真实 CUDA 指令、布局与同步。

## 1. 无需 GPU：先运行在线归一化

在 slime 仓库根目录执行：

```bash
python3 docs/flash_attention_code_walkthrough/examples/online_attention.py
```

[`online_attention.py`](examples/online_attention.py) 只依赖 Python 标准库。输入是单个 head 的二维列表，使用 Python 浮点数计算。它会输出：

```text
row=0, keys=[0, 1], m=0.693147, l=1.500000, a=[2.5]
row=0, keys=[2, 3], m=2.079442, l=1.875000, a=[6.125]
Output = 3.266666667; expected 49/15 = 3.266666667
```

可按以下顺序读函数：

| 函数 | 对应知识点 |
|---|---|
| `dense_attention` | 一次性算完整 score 行、softmax、加权输出 |
| `online_attention` | 逐块维护 m/l/a，无需保存完整 score 行 |
| `visible` | 非方阵 causal mask 的右下对齐 |
| `combine_partials` | 通过 LSE 合并多个局部输出 |
| `reference_backward` | 第 04 节的梯度公式 |
| `self_check` | 与独立参考、有限差分对照 |

检查覆盖 64 组分块前向组合，包括 Q/K 长度不同、causal 与非 causal、正序与逆序、tile 大于或不能整除 K 长度。还检查 split 合并、大 score 稳定性和反向有限差分；`N_q>N_k` 的 causal case 包含完全遮蔽的行。

这些检查在本次文档编写环境中已运行通过，梯度有限差分最大绝对误差约 `2.13e-10`。这是 Python 数学实现的结果，不是上游 GPU kernel 的实测结果。脚本没有实现 GPU 线程、FP16 舍入、dropout、GQA 或 paged KV。

正常运行时末尾应出现前向与反向两行 `PASS`，并正常退出；`AssertionError` 表示某项比较失败。**有限差分**是从输入两侧各挪一点，估算变化率：`[loss(x+epsilon)-loss(x-epsilon)]/(2*epsilon)`。脚本把这个独立估计与反向公式的结果比较，因此它能发现公式错误，而不是仅仅检查某个函数有没有返回数组。尚未读第 04 章时，可以先只关注前向检查。

## 2. 做三个能解释结果的修改

在脚本副本上分别尝试：

1. 将 `tile_size` 改为 1、3、8。答案应在浮点误差内一致，但处理轮数不同。
2. 设置 `reverse=True`。答案应在误差内一致；第 02 节的 CUDA 前向也会逆序遍历 K tile。
3. 将旧 `a` 的 `correction * a[t]` 错改成 `a[t]`。手算案例会失败。用第 01 节的四个 score 解释：第二块让最大值改变，旧输出没有切换到新基准。

第三步只是在副本里制造一个明确错误，以检查自己是否理解缩放的必要性。修复后应再次通过原有检查。

前两项不需要编辑脚本。在仓库根目录的终端中直接执行下面整段命令，即可改变分块大小与扫描方向：

```bash
python3 - <<'PY'
import math
import runpy

# 载入教学文件中的函数，不执行它的 main()。
lesson = runpy.run_path("docs/flash_attention_code_walkthrough/examples/online_attention.py")
online_attention = lesson["online_attention"]
q = [[1.0]]
k = [[math.log(x)] for x in (1, 2, 4, 8)]
v = [[1.0], [2.0], [3.0], [4.0]]
for tile in (1, 2, 3, 8):
    for reverse in (False, True):
        print(f"tile={tile}, reverse={reverse}")
        out, _ = online_attention(q, k, v, tile_size=tile, reverse=reverse, trace=True)
        print(f"output={out[0][0]:.9f}")
        assert abs(out[0][0] - 49 / 15) < 1e-12
PY
```

**先预测再运行：**所有配置的输出都应约为 `3.266666667`；tile=1/2/3/8 分别处理 4/2/2/1 轮。tile=3 最后一块只有一个 key。tile=2、reverse=True 的第一条状态应为 `m≈2.079442、l=1.5、a=[5.5]`，与第 01 章的反向扫描手算一致。

再做一个不依赖 GPU 的线程练习：按第 03 章的公式算线程 38。它的 `warp=1、lane=6`，首次搬 Q 的局部行 4、列 48～55；在 `m_block=n_block=1` 时，score 行仍是 `145、153、209、217`，每行 key 列变成 `132、133、140、141、…、252、253`。若你把搬运坐标当成 score 坐标，这两份答案会混在一起。

## 3. GPU 环境：先确认调用的是哪份实现

以下步骤面向受上游支持的 Linux/NVIDIA CUDA 环境。当前文档编写机器没有 `nvcc`，本次没有编译或执行这些 GPU 示例。版本要求以固定子模块的 [README](../../flash-attention/README.md) 和 [`setup.py`](../../flash-attention/setup.py) 为准。

如果目标是测试本仓库固定的 FA2 源码，可在已具备 PyTorch、CUDA 和构建依赖的独立 Python 环境中执行：

```bash
git submodule update --init flash-attention
git -C flash-attention submodule update --init csrc/cutlass
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 python3 -m pip install ./flash-attention --no-build-isolation
```

`FLASH_ATTENTION_FORCE_BUILD=TRUE` 来自本快照的 setup 逻辑，用于强制源码构建；`MAX_JOBS=4` 限制构建并发。先准备 PyTorch 和 CUDA 构建环境，再执行安装，不能指望 `--no-build-isolation` 自动解决所有依赖。

安装后检查实际加载路径：

```python
import torch
import flash_attn
import flash_attn_2_cuda

print("torch:", torch.__version__, "torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
print("Python package:", flash_attn.__file__)
print("CUDA extension:", flash_attn_2_cuda.__file__)
```

安装包和仓库中的同名源码可能同时存在。记录实际导入路径，才能把测试结果对应到正确版本。

## 4. 用显式参考计算检查前向和反向

下面是**安装成功后可独立运行的 Python 示例**。参考路径显式计算 score、softmax、PV，并用 FP64 减少参考舍入误差。这里使用小尺寸、相同 Q/K 长度和无 mask 的基础场景；不要把它直接当作非方阵 causal 测试。

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

[`tests/test_flash_attn.py`](../../flash-attention/tests/test_flash_attn.py) 的参数化很多。先收集对应测试，再选一个完整 node id，避免初次就运行大矩阵全集：

```bash
python3 -m pytest flash-attention/tests/test_flash_attn.py --collect-only -q -k test_flash_attn_output
```

在配置好其测试依赖后，从输出复制一个 `tests/...py::test_name[...]` 节点标识交给 pytest。源码中值得顺序阅读的测试有：

| 测试 | 要理解的边界 |
|---|---|
| `test_flash_attn_output` | dtype、形状、前向与梯度 |
| `test_flash_attn_varlen_output` | 扁平 token、累计长度、padding 对照 |
| `test_flash_attn_varlen_causal` | Q/K 长度不同的 causal 语义 |
| `test_flash_attn_kvcache` | cache 追加、cache 长度、分页和位置编码 |
| `test_flash_attn_varlen_deterministic` | 确定性反向的重复结果 |

FA3 使用 [`hopper/test_flash_attn.py`](../../flash-attention/hopper/test_flash_attn.py)，FA4 使用 [`tests/cute/`](../../flash-attention/tests/cute/)。它们对应不同安装和接口，不要仅凭测试名字相似就替换运行路径。

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

记录 GPU 型号、软件版本、输入 shape/dtype、mask、是否反向、warmup 和计时方法。再一次只改一个变量，例如长度 256→1024→4096，观察内存和时延，而不是直接把结果与论文里的峰值数字相除。

## 7. 出问题时按层缩小范围

| 现象 | 优先检查 | 源码入口 |
|---|---|---|
| `ImportError` / 符号加载失败 | 包路径、扩展、PyTorch/CUDA 构建是否匹配 | Python import 与 setup |
| `TORCH_CHECK` 报错 | dtype、device、末维 stride、head 比例 | `flash_api.cpp::mha_fwd` |
| 只有 causal 非方阵不一致 | 参考是否右下对齐 | `flash_attn_func` docstring |
| 只有 varlen 不一致 | `cu_seqlens`、实际长度、max length | `BlockInfo` |
| 输出正确、梯度不一致 | scale、dO、dropout/RNG、允许误差 | backward 与上游测试 |
| 修改同步后随机错或挂住 | 数据就绪、缓冲区覆盖、屏障参与者 | 主循环与 pipeline |
| 第一次调用特别慢 | 初始化、JIT 编译、缓存 | FA4 `interface.py` / `cache_utils.py` |

如果将来要实际修改上游 kernel，请先遵循子模块自己的 [`AGENTS.md`](../../flash-attention/AGENTS.md) 调试要求。GPU 数据竞争、挂起、非法地址的诊断需要可复现输入和相应工具证据，单靠阅读公式无法证明某条同步可以删除。

## 8. 最后把整条链讲出来

完成学习后，尝试不用查看文档，依次说明：

1. Python 的 `[B,N,H,D]` 张量如何通过指针与 stride 交给 C++。
2. C++ 如何选择 tile 配置，grid 的三个轴分别代表什么。
3. 一个 block 如何协作加载 Q/K/V，并形成 score 的线程片段。
4. 在线 Softmax 为什么需要 m/l/a，以及何时跨线程合并。
5. 为什么保存 LSE 可以帮助反向重算，为什么短 Q 可能需要 split KV。

如果其中一步说不清，就回到对应源码的一个具体变量，写下它的形状、存储位置和拥有者。这通常比继续往后读更多文件有效。
