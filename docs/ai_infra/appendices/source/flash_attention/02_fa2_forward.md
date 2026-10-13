# 02 · FA2 前向：把一次调用逐段展开


## 1. 我们追踪的调用与执行链

下面是 GPU 环境中的完整调用片段，Q/K/V 的轴顺序为 `[batch, token, head, feature]`：

```python
import torch
from flash_attn import flash_attn_func

q = torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
```

我们保持无 dropout、无位置偏置、无 softcap，先追常规前向。`flash_attn_func` 把参数交给 `FlashAttnFunc.apply`，PyTorch 再执行该类的 `forward`。它向下调用扩展，经 C++ 选择一个已经编译的 kernel，最后让多个 CUDA block 并行工作。

上面的 `randn` 生成随机测试数据，`randn_like(q)` 生成与 Q 同形状、同类型、同设备的新数据。`device="cuda"` 指数据放在 GPU；`dtype=torch.float16` 指每元素采用 FP16。`forward` 是从输入算输出；`backward` 是训练时从输出梯度算输入梯度。`dropout` 是训练中的随机丢弃操作，参数 0 表示关闭；`softcap` 使用平滑函数限制 score 幅度，本例同样关闭。

### 1.1 先看本轮张量的形状与存储位置

本例选中的配置是 128 行 Q × 128 行 K/V、每个特征向量 64 维。`QK` 在正文中简称 Q 与 K 转置的乘法；**GEMM** 指矩阵乘，**MMA** 指硬件矩阵乘加操作，**Tensor Core** 是执行这类运算的 GPU 计算单元。

| 本轮数据 | 整个 block 对应的逻辑形状 | 在哪里 |
|---|---|---|
| Q tile | `128×64` | 显存输入 → shared → 寄存器操作数 |
| K tile | `128×64` | 显存输入 → shared → 寄存器操作数 |
| `Q @ K.T` | `(128×64) @ (64×128) = 128×128` | 分布在各线程的 `acc_s` 寄存器片段中 |
| 指数权重 tile | `128×128` | 原地覆盖 score；再转成 `rP` |
| V tile | `128×64` | 显存输入 → shared → 寄存器操作数 |
| `rP @ V` 的累加结果 | `(128×128) @ (128×64) = 128×64` | 分布在各线程的 `acc_o` 中 |
| m/l | 每个 query 行各一个逻辑统计量 | 各线程片段；l 在中途是部分和 |

矩阵乘中两边的中间维度要相等：QK 沿 64 个特征求和，PV 沿 128 个 key 求和。表中的逻辑形状属于**整块协作结果**，每条线程只拥有其中的 fragment（片段）。CuTe 是帮助表达这些布局和分工的库；CuTe `Tensor` 可以表示视图或线程片段，不能直接按一个新分配的 PyTorch 大张量理解。

Python 包装和 C++ 分发逻辑由 CPU 执行，CUDA kernel 的函数体由 GPU 线程执行。Python 调用的 padding 等张量操作也可能提交自己的 GPU 工作；看到 `.py` 文件不等于其中调用的计算都在 CPU。

```text
flash_attn_func → FlashAttnFunc.forward → _flash_attn_forward
→ flash_attn_2_cuda.fwd → mha_fwd → run_mha_fwd
→ run_mha_fwd_hdim64 → run_flash_fwd
→ flash_fwd_kernel → compute_attn → compute_attn_1rowblock
```

下面按这个顺序解释，读者无需离开文档才能拼出主要逻辑。

## 2. Python 的 `forward`：scale 与 padding 的顺序有意义

源码摘录：[flash_attn_interface.py，845–854 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/flash_attn_interface.py#L845-L854)。

<!-- source: flash_attn/flash_attn_interface.py:845-854 -->
```python
is_grad = is_grad_enabled and any(
    x.requires_grad for x in [q, k, v]
)
if softmax_scale is None:
    softmax_scale = q.shape[-1] ** (-0.5)
head_size_og = q.size(3)
if head_size_og % 8 != 0:
    q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
    k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
    v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
```

`is_grad_enabled` 表示外部是否启用了求导，`requires_grad` 表示输入是否需要梯度。两者都满足才需要保存反向状态。我们的调用没有要求输入求导，所以这里是 false；计算 O 仍照常进行。

`q.shape[-1] ** (-0.5)` 得到 $1/\sqrt d$。本例 d=64，结果为 0.125。`head_size_og` 记录进入函数时的维度，以便最终裁剪回用户期望的形状。

注意先求 scale，再补零。假设 d=62，`8 - 62%8 = 2`，Q/K/V 从 62 维补到 64 维，补出的分量是零，不改变原先的 QK 点积；scale 仍应为 $1/\sqrt{62}$。如果补完后再取 `q.shape[-1]`，就错误地换成了 $1/\sqrt{64}$。

这段是在满足 kernel 对末维对齐的要求，不是在给序列长度补 token。`pad(q,[0,2])` 的两个数字只作用于最后一维。

## 3. Python 包装怎样跨到 C++

源码摘录：[flash_attn_interface.py，98–114 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/flash_attn_interface.py#L98-L114)。

<!-- source: flash_attn/flash_attn_interface.py:98-114 -->
```python
q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
    q,
    k,
    v,
    None,
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    None,  # unused generator slot, kept for backwards-compat arg positioning
)
return out, softmax_lse, S_dmask, rng_state
```

`maybe_contiguous` 只在 `stride(-1) != 1` 时调用 `.contiguous()`，让最后一维可连续读取。对本例连续张量不产生拷贝。

`flash_attn_gpu` 在 NVIDIA 路径是 `flash_attn_2_cuda` 扩展模块。`fwd` 的第 4 个实参是 `None`，对应可选的预先分配输出；这里让 C++ 分配。末尾 `None` 是保留的旧 generator 参数位置，不能按名字推断仍可传自定义 generator。

LSE（log-sum-exp）是 log(∑exp(xᵢ))，可写成 m+log(∑exp(xᵢ−m))，其中 m=max(xᵢ)。它把指数归一化的分母放到对数域计算，减最大值可避免大正数的指数溢出。

RNG（Random Number Generator，随机数生成器）具有随抽样推进的内部状态。仅保存初始 seed 不能恢复已推进的位置，重计算和恢复训练通常还需保留状态及对应的消费顺序。

扩展返回四项：O、逐行 LSE、测试用途的 `S_dmask`、RNG（随机数生成器）状态。`alibi_slopes` 是可选位置偏置的参数，本例为 `None`，无需进入该分支。默认调用不会把四项返回值全交给用户。上层的处理是：

源码摘录：[flash_attn_interface.py，868–878 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/flash_attn_interface.py#L868-L878)。

<!-- source: flash_attn/flash_attn_interface.py:868-878 -->
```python
if is_grad:
    ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
    ctx.dropout_p = dropout_p
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.window_size = window_size
    ctx.softcap = softcap
    ctx.alibi_slopes = alibi_slopes
    ctx.deterministic = deterministic
out = out_padded[..., :head_size_og]
return out if not return_softmax else (out, softmax_lse, S_dmask)
```

`save_for_backward` 保存的是反向需要的张量引用；`ctx.softmax_scale` 等保存非张量配置。若输入被补到 64 维，`out_padded[..., :head_size_og]` 把输出裁回 62 维。`return_softmax` 控制返回形式，但测试用矩阵的生成还受上游 `return_softmax and dropout_p > 0` 的条件约束；不要把它当成始终可用的标准概率矩阵。

## 4. C++ 把张量拆成地址与步长

`mha_fwd` 先检查 dtype、设备、形状和末维 stride，再分配 O/LSE。随后 `set_params_fprop` 为 GPU 准备参数：

源码摘录：[flash_api.cpp，73–91 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/flash_api.cpp#L73-L91)。

<!-- source: csrc/flash_attn/flash_api.cpp:73-91 -->
```cpp
// Reset the parameters
params = {};

params.is_bf16 = q.dtype() == torch::kBFloat16;

// Set the pointers and strides.
params.q_ptr = q.data_ptr();
params.k_ptr = k.data_ptr();
params.v_ptr = v.data_ptr();
// All stride are in elements, not bytes.
params.q_row_stride = q.stride(-3);
params.k_row_stride = k.stride(-3);
params.v_row_stride = v.stride(-3);
params.q_head_stride = q.stride(-2);
params.k_head_stride = k.stride(-2);
params.v_head_stride = v.stride(-2);
params.o_ptr = out.data_ptr();
params.o_row_stride = out.stride(-3);
params.o_head_stride = out.stride(-2);
```

`params = {}` 清零参数结构，避免未启用的可选指针带有垃圾值。`q.data_ptr()` 是 Q 第一个元素的地址，没有把 Q 复制到结构里；结构只保存描述信息。

`q.stride(-3)` 是 token 轴步长，`q.stride(-2)` 是 head 轴步长。对 `[2,256,4,64]` 连续张量：

| 字段 | 本例值，单位为元素 | 用途 |
|---|---:|---|
| `q_row_stride` | 256 | token 编号加 1 时跨过 4 个 head |
| `q_head_stride` | 64 | head 编号加 1 时跨过一个 64 维向量 |
| `q_batch_stride` | 65536 | batch 编号加 1 时跨过整个样本 |
| 末维步长 | 1 | 特征编号加 1 时移动一个元素 |

C++ 指针类型转成 `Element*` 后，`ptr + 256` 已经按元素大小移动地址；不能再把这里的 stride 乘 2 后传进去，否则 FP16 地址会被重复放大。

同函数还设置 scale 与 dropout：

源码摘录：[flash_api.cpp，134–149 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/flash_api.cpp#L134-L149)。

<!-- source: csrc/flash_attn/flash_api.cpp:134-149 -->
```cpp
} else{
    // Remove potential NaN
    params.softcap = 0.0;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
}

// Set this to probability of keeping an element to simplify things.
params.p_dropout = 1.f - p_dropout;
// Convert p from float to int so we don't have to convert the random uint to float to compare.
// [Minor] We want to round down since when we do the comparison we use <= instead of <
// params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
// params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
params.rp_dropout = 1.f / params.p_dropout;
params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
```

本例走 `else`，`scale_softmax=0.125`，`scale_softmax_log2=0.125×log2(e)`，用于把自然指数改写成 `exp2`。`params.p_dropout` 是保留概率，值为 1；`rp_dropout` 为其倒数，仍为 1。参数名相似，却与 Python 的“丢弃概率 0”不是同一含义。

## 5. 分发函数选择哪个实例

源码摘录：[flash_api.cpp，261–274 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/flash_api.cpp#L261-L274)。

<!-- source: csrc/flash_attn/flash_api.cpp:261-274 -->
```cpp
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}
```

从外往里读：`FP16_SWITCH` 选择元素类型；`HEADDIM_SWITCH` 选择编译时 head dimension；`BOOL_SWITCH` 把 causal 变成编译时布尔值；最后根据 split 参数选择普通前向还是拆分 K/V 的前向。

`BOOL_SWITCH` 不是在每个 GPU 元素上判断。它在 CPU 分发阶段选择 `some_function<true>` 或 `<false>` 的编译版本，宏内部的大括号和 `[&]` 是 C++ lambda 写法，用于包装这段选择逻辑。

本例 d=64、FP16、non-causal，对应的桥接实例很短：

源码摘录：[flash_fwd_hdim64_fp16_sm80.cu，9–12 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu#L9-L12)。

<!-- source: csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu:9-12 -->
```cpp
template<>
void run_mha_fwd_<cutlass::half_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64<cutlass::half_t, false>(params, stream);
}
```

模板参数 `<cutlass::half_t,64,false>` 已把类型、维度、causal 固定。接着无 dropout 分支选择：

源码摘录：[flash_fwd_launch_template.h，210–210 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_launch_template.h#L210-L210)。

<!-- source: csrc/flash_attn/src/flash_fwd_launch_template.h:210-210 -->
```cpp
run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
```

依次是 head dimension=64、Q 块高=128、KV 块高=128、4 个 warp、不开启该 Q 常驻寄存器策略、不让 Q/K 共用同一共享内存区域、元素类型 T。对应 `kNThreads=4×32=128`。

`set_params_splitkv` 的启发式对本例按 K 长度 256、split 参考块长 256 只得到一个 K 块，选择普通前向。后面所有具体线程数字都针对这个分支。

## 6. grid 的三个轴如何成为输出所有者

`run_flash_fwd` 建立 grid：

源码摘录：[flash_fwd_launch_template.h，64–68 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_launch_template.h#L64-L68)。

<!-- source: csrc/flash_attn/src/flash_fwd_launch_template.h:64-68 -->
```cpp
dim3 grid(num_m_block, params.b, params.h);
const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
const bool is_even_K = params.d == Kernel_traits::kHeadDim;
const bool return_softmax = params.p_ptr != nullptr;
BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
```

整数表达式 `(n+b-1)/b` 做向上取整。代入本例得到 `num_m_block=2`，grid 为 `(2,2,4)`，总计 16 个 block。`is_even_MN=true` 表示固定长度且 Q/K 长度整除所选 tile，编译器可以去掉部分越界保护。

启动语句随后是 `kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params)`。每个 block 128 线程、动态共享内存 48 KiB。kernel 包装调用 `compute_attn`，后者按如下方式读编号：

源码摘录：[flash_fwd_kernel.h，1084–1088 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L1084-L1088)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:1084-1088 -->
```cpp
const int m_block = blockIdx.x;
// The block index for the batch.
const int bidb = blockIdx.y;
// The block index for the head.
const int bidh = blockIdx.z;
```

我们挑 `(blockIdx.x,blockIdx.y,blockIdx.z)=(1,0,2)`。这个 block 拥有 `out[0,128:256,2,:]`。其他 block 负责不同 Q 行或 batch/head，因此普通前向无需让两个 block 同时累加这个输出 tile。

## 7. 先确定本块要读哪些 K 行

源码摘录：[flash_fwd_kernel.h，87–94 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L87-L94)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:87-94 -->
```cpp
const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
if (Is_causal || Is_local) {
    n_block_max = std::min(n_block_max,
                           cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
```

`binfo` 给出当前样本实际 Q/K 长度。第一条 `return` 的条件对整个 block 相同：若块起始行已经越界，就没有有效输出需要计算。

`n_block_min/max` 表示 K 块范围，max 是**不包含的上界**。普通 non-causal 本例得到 `[0,2)`，即 K 块 0、1。local attention 可以提高下界；causal/local 可以降低上界，从而整块跳过不可能参与的 K。

举一个相同长度的 causal 变体：对 `m_block=0`，Q 是 0～127，最大允许 K 不超过 127，因此上界裁到 1；对 `m_block=1`，Q 是 128～255，上界为 2。块范围只做粗裁剪，仍要在块内对每一行执行精确 mask。

## 8. Q/K/V 的地址视图怎样建立

源码摘录：[flash_fwd_kernel.h，145–156 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L145-L156)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:145-156 -->
```cpp
Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                      + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                        make_shape(binfo.actual_seqlen_q, params.h, params.d),
                        make_stride(params.q_row_stride, params.q_head_stride, _1{}));
Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                       make_coord(m_block, 0));  // (kBlockM, kHeadDim)
Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                      + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                        make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                        make_stride(params.k_row_stride, params.k_head_stride, _1{}));
Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                       make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
```

`reinterpret_cast<Element*>` 先告诉编译器“这段地址按 FP16 元素访问”。`binfo.q_offset(...)` 定位样本起点；`make_shape` 和 `make_stride` 给它一个 `[token,head,feature]` 的逻辑视图。接着 `mQ(_,bidh,_)` 固定 head，`local_tile` 再取 Q 行块。

对 block `(1,0,2)`，`gQ(0,0)` 对应 `q[0,128,2,0]`，元素偏移为 `128×256+2×64=32896`。`gQ(1,0)` 的偏移是 33152，差值仍为 256；固定 head 不会让原张量的 token 行距自动缩小。

K 的 `make_coord(_,0)` 保留了沿序列分块的轴，所以 `gK` 是一串 `128×64` tile 的视图。`bidh / params.h_h_k_ratio` 是 GQA（多个 Q head 共享 KV head）的映射；本例比例为 1，仍选择 head 2。建立这些视图不会触发数据搬运。

## 9. 首轮之前：放好 Q 和最后一块 K

真正的拷贝在这里出现：

源码摘录：[flash_fwd_kernel.h，256–259 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L256-L259)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:256-259 -->
```cpp
// We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                   binfo.actual_seqlen_q - m_block * kBlockM);
if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }
```

`gmem_tiled_copy_QKV` 规定搬运方式；`tQgQ` 是当前线程负责的显存源片段，`tQsQ` 是共享内存目标片段；`tQcQ` 提供坐标，`tQpQ` 是特征维度的有效性条件。最后一个实参告诉 helper 本块还剩多少有效 Q 行。本例为 128。

本例 `Is_Q_in_regs=false`，这一处不提交单独的 Q 拷贝组。紧接着：

源码摘录：[flash_fwd_kernel.h，274–278 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L274-L278)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:274-278 -->
```cpp
int n_block = n_block_max - 1;
// We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                                   binfo.actual_seqlen_k - n_block * kBlockN);
cute::cp_async_fence();
```

`n_block=1`，所以先读 K 的 128～255 行。Q 和 K 的拷贝都在随后这次 `cp_async_fence()` 提交的组内。循环开头等它完成，才能读取共享内存中的 Q/K。

注意 `Softmax` 对象的 `row_max/row_sum` 并非靠构造函数统一初始化成 `-inf/0`；它依赖首轮的 `Is_first=true` 覆写初值。输出累加器则显式 `clear(acc_o)`。这解释了为什么首轮有专门分支，第 01 节的源码精读会继续展开。

## 10. 第一轮：从 QK 到 PV，每个调用改变了什么

本例 `n_masking_steps=1`，先进入带边界处理能力的首轮循环：

源码摘录：[flash_fwd_kernel.h，305–324 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L305-L324)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:305-324 -->
```cpp
constexpr int n_masking_steps = (!Is_causal && !Is_local)
    ? 1
    : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
#pragma unroll
for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    FLASH_NAMESPACE::cp_async_wait<0>();
    __syncthreads();

    // Advance gV
    if (masking_step > 0) {
        FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
    } else {
        // Clear the smem tiles to account for predicated off loads
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
    }
    cute::cp_async_fence();
```

先分配当前线程的 `acc_s` 片段并清零，因为 GEMM 执行的是累加。`cp_async_wait<0>()` 等前面提交的 Q/K 搬运，`__syncthreads()` 让 block 内线程可以读取彼此搬来的数据。

随后发起 V 的同一行块 128～255 的搬运。首轮传 `Clear_OOB_MN=true`，在实际尾块不足整块时清除无效 V 行，防止后续 `0×NaN` 等情况污染乘积。本例长度整齐，所有 V 行都有效。提交 V 后不立刻等待，而是利用这段时间算 QK：

源码摘录：[flash_fwd_kernel.h，326–337 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L326-L337)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:326-337 -->
```cpp
FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
    acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
    smem_thr_copy_Q, smem_thr_copy_K
);
// if (cute::thread0()) { print(acc_s); }
if constexpr (Is_softcap){
    FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
}

mask.template apply_mask<Is_causal, Is_even_MN>(
    acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
);
```

`acc_s` 是被修改的输出；`tSrQ/tSrK` 是寄存器操作数，`tSsQ/tSsK` 是其共享内存来源；剩下的对象描述 MMA 与搬运布局。`Is_Q_in_regs=false` 使 helper 按需要从共享内存加载 Q 的寄存器片段。此时 `acc_s` 仍是未乘 alpha 的 QK 点积。

本例 `Is_softcap=false`，跳过 tanh 修饰。`mask.apply_mask` 接收 K 块列起点、本线程负责的 Q 行起点，以及行步幅。虽然调用写在源文件里，本例的 mask 条件全关闭且长度整齐，内部 `if constexpr(Need_masking)` 可被编译器消掉。

接下来等 V 并准备下一块 K：

源码摘录：[flash_fwd_kernel.h，339–351 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L339-L351)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:339-351 -->
```cpp
FLASH_NAMESPACE::cp_async_wait<0>();
__syncthreads();
if (n_block > n_block_min) {
    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
    // This cp_async_fence needs to be in the if block, otherwise the synchronization
    // isn't right and we get race conditions.
    cute::cp_async_fence();
}

// TODO: when we have key_padding_mask we'll need to Check_inf
masking_step == 0
    ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
    : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);
```

这次 wait 对应本轮 V 的搬运；同步后，本轮 QK 已结束，可安全用下一块 K 覆盖 sK。由于 `n_block=1 > n_block_min=0`，线程发起 K 块 0 的搬运，并提交新的拷贝组。随后计算本轮 softmax/PV 时，这次 K 搬运可以继续推进。

`masking_step==0` 选择 `Is_first=true`：从本块 score 初始化最大值和部分和，将 `acc_s` 原地改为指数权重。此时 `acc_o` 还是 0，不需要缩放已有贡献。

最后两条关键操作是：

源码摘录：[flash_fwd_kernel.h，353–354 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L353-L354)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:353-354 -->
```cpp
// Convert acc_s from fp32 to fp16/bf16
Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
```

源码摘录：[flash_fwd_kernel.h，370–374 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L370-L374)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:370-374 -->
```cpp
// Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
// if (cute::thread0()) { print(tOrP); }
FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
```

`convert_type` 将 FP32 指数权重转换为 FP16 矩阵乘输入；`convert_layout_acc_Aregs` 把前一个 MMA 的结果片段视图改成后一个 MMA 的 A 操作数视图。这里的 `rP` 还没有最终归一化。

`gemm_rs` 做 `acc_o += rP @ V`。第一轮退出后：m/l 描述 K 块 1，`acc_o` 保存块 1 的未归一化贡献；sV 是块 1，sK 正在或已经被更新为块 0。

## 11. 第二轮：复用状态，不能再清空输出

首轮结束的 `--n_block` 把它改为 0。普通循环再次等待 K0、加载 V0、计算新的 `acc_s`。区别在于 `acc_s` 每轮清零，而 `acc_o` 在循环外初始化后一直保留。

源码摘录：[flash_fwd_kernel.h，410–416 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L410-L416)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:410-416 -->
```cpp
mask.template apply_mask</*Causal_mask=*/false>(
    acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
);

softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
```

`Is_first=false` 告诉 softmax：旧统计和旧输出已有贡献。若 K0 出现更大 score，先让旧 l 和旧 `acc_o` 乘修正因子，再累计 K0 的指数权重。随后第二次 `gemm_rs` 加上 K0 的 PV 贡献。

`n_block=0` 时没有更早的 K 块，`if (n_block > n_block_min)` 不再发起 K 拷贝，也不提交那个组。一个完整的状态记录如下：

| 时刻 | `n_block` | `acc_s` 含义 | `acc_o` 含义 |
|---|---:|---|---|
| 首轮 QK 后 | 1 | K1 的原始 score | 0 |
| 首轮 softmax 后 | 1 | K1 的指数权重 | 0 |
| 首轮 PV 后 | 1 | 本轮权重，可丢弃 | K1 的贡献 |
| 第二轮 QK 后 | 0 | K0 的原始 score | 仍有 K1 的贡献 |
| 第二轮 softmax 后 | 0 | K0 在共同新基准下的权重 | K1 贡献已按需要缩放 |
| 第二轮 PV 后 | 0 | 本轮权重，可丢弃 | K0 与 K1 的合并贡献 |

## 12. 输出写回：为什么又绕一次 shared memory

源码摘录：[flash_fwd_kernel.h，440–449 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L440-L449)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:440-449 -->
```cpp
Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

// Convert acc_o from fp32 to fp16/bf16
Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
// Partition sO to match the accumulator partitioning
auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
```

`normalize_softmax_lse` 完成跨线程 l 合并、`acc_o /= l`，并返回逐行 LSE。`rO` 是 FP16 输出片段。`sO` 使用 `sQ.data()`，说明 Q 在这里已经不再需要，其共享内存空间可被输出复用。

`smem_tiled_copy_O` 按 MMA 结果布局将 rO 写到 sO。接下来代码换成适合显存连续写入的搬运布局：

源码摘录：[flash_fwd_kernel.h，464–472 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L464-L472)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:464-472 -->
```cpp
typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

__syncthreads();

Tensor tOrO = make_tensor<Element>(shape(tOgO));
cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
```

计算用的线程片段布局未必适合连续写显存。共享内存在这里提供一次重排的中转：原来的计算线程写入各自结果，屏障之后，负责显存存储的线程按另一种分片方式读出到 `tOrO`。这说明 `.retile_*`、`partition_*` 和真正的 `copy` 有不同职责。

最后写输出时关闭越界区域的清零：

源码摘录：[flash_fwd_kernel.h，497–500 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_kernel.h#L497-L500)。

<!-- source: csrc/flash_attn/src/flash_fwd_kernel.h:497-500 -->
```cpp
// Clear_OOB_K must be false since we don't want to write zeros to gmem
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
    gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
);
```

读输入时可把无效值置零，写输出时却不能向无效地址写零；越界写同样是错误。`Clear_OOB_MN=false`、`Clear_OOB_K=false` 配合有效坐标条件，只写该 block 真正拥有的输出位置。

因此最终这个 block 写回 `out[0,128:256,2,:]`，同时为这 128 行写 LSE。到这里，一次真实前向的输入、循环状态和输出已在文中闭合。

## 13. 局部改动为什么会破坏不变量

- 把 `clear(acc_o)` 移进 K 循环：**每轮丢失旧块贡献，结果错误。**
- 先 pad，再按 pad 后的 d 计算默认 scale：**非对齐 head dimension 的缩放错误。**
- 把 Q 行距由 256 改成 64：**跨 token 寻址会读到其他 head 数据。**
- 把所有 softmax 调用都设为 `Is_first=true`：**重置统计，无法正确合并 K 块。**
- 在显存写回时给无效坐标也写零：**可能越界或覆盖其他数据。**

下一节继续展开 `partition_*`、mask 坐标和 GEMM helper 的内部实现。
