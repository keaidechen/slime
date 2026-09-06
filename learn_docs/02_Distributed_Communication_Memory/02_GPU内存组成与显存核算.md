# GPU 内存组成与显存核算

<!-- learning-position -->
> **学习定位**：A1/A2 · 必修。
> **前置**：[基础课程](<../00_Foundations/README.md>)。
> **首读/二读**：参数/梯度/优化器/activation/KV，区分 allocated/reserved/device used。
> **进度与实验**：[学习清单](<../学习清单.md>) · [总入口](<../README.md>)。
<!-- /learning-position -->

<a id="beginner-example"></a>

## 入门例子：用 1B 模型建立一张显存账本

先假设模型有 10 亿参数，BF16 参数 2 GB、FP32 梯度 4 GB、FP32 主权重 4 GB、Adam 两个矩 8 GB，共 18 GB。这里 GB=10^9 字节，换成 GiB 要除以 2^30。实际实现可能没有某些副本或使用不同梯度 dtype，必须对着代码改账本。

| 时刻 | 除常驻状态外增加什么 | 为何峰值会变化 |
|---|---|---|
| forward | 输入、中间 activation、算子 workspace | backward 尚未发生，保存状态逐层增加 |
| backward | 梯度与临时反向结果 | activation 按生命周期释放，梯度逐渐就绪 |
| optimizer | 更新 workspace、可能的参数聚合 | 取决于分片、精度和 overlap |
| rollout | KV、graph pool、推理 workspace | 与训练阶段不同，共置也不意味着自动共享全部内存 |

练习：只改变序列长度，再只改变 microbatch，分别测每阶段峰值。若常驻状态没变但峰值增长，不能把新增显存都归到参数。验收是能解释预测与实测差额，而不是恰好凑出一个数字。

## 机制与实现

## `nvidia-smi` 的 used memory 不是 tensor 总和

GPU High Bandwidth Memory（HBM，高带宽内存）中通常同时存在：

| 类别 | 生命周期 | 典型内容 |
|---|---|---|
| 模型状态 | 整个 step/任务 | 参数、梯度、optimizer states、master weights |
| 前反向中间值 | 若干 layer 到 backward | activation、saved tensor、attention workspace |
| 推理状态 | 请求存活期间 | KV Cache、sequence metadata、prefix cache |
| 通信资源 | communicator 存活期间 | NCCL internal buffer、registered user buffer |
| 运行时资源 | 进程/context 存活期间 | CUDA context、library handle、kernel module |
| 临时工作区 | 算子或图执行期间 | cuBLASLt/CUDNN workspace、排序与路由 buffer |
| allocator 保留 | 被缓存分配器持有 | 空闲 block、碎片、graph-private pool |

PyTorch 的 `memory_allocated()` 是活跃 tensor 使用量；`memory_reserved()` 是 caching allocator 从 CUDA 保留的总量；`nvidia-smi` 还会包含 context、非 PyTorch allocation 与其他库。因此 `reserved - allocated` 不全是“泄漏”，而 OOM 也可能发生在仍有总空闲量时，因为缺少足够大的连续 block 或合适 segment。

## 训练显存的第一版心算

设模型参数量为 $P$，数据并行度为 $D$。以混合精度 Adam 为例，实际字节取决于框架是否保存 FP32 master parameter、梯度 dtype 与 optimizer 实现。一个常见但必须标注假设的估算是：

| 状态 | 每参数字节（示例） |
|---|---:|
| BF16 参数 | 2 |
| BF16 梯度 | 2 |
| FP32 master 参数 | 4 |
| Adam 一阶、二阶矩 | 8 |
| 合计 | 16 B/parameter |

这 16 B 不是定律。若无 master weights、梯度为 FP32、使用 8-bit optimizer 或量化参数，结果都会变化。普通 Data Parallel（DP，数据并行）每 rank 复制全部模型状态，静态状态约为 $16P$；理想 ZeRO-3/FSDP full shard 可将这部分降至约 $16P/D$，但 gather buffer、activation、碎片和峰值瞬态不会自动除以 $D$。

### 例：70B 参数、64-way shard

按 16 B/parameter，未分片静态状态约 $70\times10^9\times16=1.12$ TB；理想分到 64 rank 后每 rank 约 17.5 GB。真实峰值还需加上 layer unshard、activation、通信 buffer 和 allocator headroom，所以“17.5 GB < 80 GB”不代表一定能训练。

## Activation 为什么难以只靠参数量估计

Activation 近似随 micro-batch $B$、序列长度 $S$、隐藏维 $H$ 和层数 $L$ 增长：

$$
M_{act}\propto BSLH
$$

但 attention 是否保存 $S\times S$ 矩阵、MLP 中间维、fused kernel 保存哪些 tensor、sequence parallel、FlashAttention 与 checkpoint 边界都会改变常数甚至主导项。可靠方法是：先符号核算，再通过 PyTorch memory snapshot 或 profiler 对真实 graph 验证。

## 推理 KV Cache 核算

对 decoder-only Transformer，若每层有 $n_{kv}$ 个 KV heads，每个 head 维度 $d_h$，层数 $L$，缓存 dtype 占 $b$ 字节，则每 token 的 KV 约为：

$$
M_{KV/token}=2\times L\times n_{kv}\times d_h\times b
$$

乘以当前所有请求的已缓存 token 数，再除以 KV 被 tensor parallel 分片的程度。前面的 2 分别对应 Key 与 Value。Multi-Query Attention（MQA，多查询注意力）或 GQA 通过减少 $n_{kv}$ 显著降低容量和读取流量。

例：$L=80,n_{kv}=8,d_h=128,b=2$，每 token 为 $2\times80\times8\times128\times2=327{,}680$ B，约 320 KiB；一条 32K token 请求约 10 GiB（尚未算 block padding 和 metadata）。这解释了长上下文服务为何常由 KV 而非权重限制并发。

## 碎片的两种含义

- 内部碎片：分配到 block/page 但未使用的尾部。例如序列只用了 KV block 的 1 个 token。
- 外部碎片：总空闲量足够，但分散为多个无法满足大请求的小洞。

PagedAttention 主要通过固定大小 block 和逻辑—物理映射避免为每个请求预留连续最大长度，显著减少外部碎片和过度预留；它不会让真实 KV 字节凭空消失。

## 建议的显存账本

每次容量评估至少报告：

1. steady-state 与 peak 两套数值；
2. per-rank 的参数、梯度、optimizer、activation、KV、communication、workspace；
3. allocated、reserved 与 device used 三种口径；
4. 最大 shape、并发和失败时刻；
5. 至少 10%–15% headroom，且通过故障恢复/动态图峰值验证。

## 资料

- [PyTorch CUDA semantics：Memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [PyTorch Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
