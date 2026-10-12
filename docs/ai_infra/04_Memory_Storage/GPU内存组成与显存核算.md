# GPU 内存组成与显存核算


前面计算了一个 Kernel 的数据搬运，本篇把视角扩大到整个模型：哪些数据必须同时放在显存中？**容量**以字节计，决定能否装下；**带宽**以字节/秒计，决定搬得多快，两者不能混用。

本篇先维护单设备账本；把容量扩展到多设备或远端资源时，继续看[超节点与远端内存池](超节点与远端内存池.md)。跨请求长期占用还依赖状态驻留时间，见[负载与资源需求](../design/负载状态与资源需求.md)。

这里需要提前认识几个状态名：BF16（bfloat16）每元素 2 字节，FP32 每元素 4 字节；主权重（master weights）是用于精细更新的高精度参数副本。Adam 是优化器，保存梯度及梯度平方的平滑历史，称为一阶矩和二阶矩，各需要一份与参数等大的状态；第 10 篇会解释其计算。1B 参数指 $10^9$ 个数，并不是 1 GB。

## 入门例子：用 1B 模型建立一张显存账本

先假设模型有 10 亿参数，BF16 参数 2 GB、FP32 梯度 4 GB、FP32 主权重 4 GB、Adam 两个矩 8 GB，共 18 GB。这里 GB=10^9 字节，换成 GiB 要除以 2^30。实际实现可能没有某些副本或使用不同梯度 dtype，必须对着代码改账本。

| 时刻 | 除常驻状态外增加什么 | 为何峰值会变化 |
|---|---|---|
| forward | 输入、中间 activation、算子 workspace | backward 尚未发生，保存状态逐层增加 |
| backward | 梯度与临时反向结果 | activation 按生命周期释放，梯度逐渐就绪 |
| optimizer | 更新 workspace、可能的参数聚合 | 取决于分片、精度和 overlap |
| 推理生成 | 历史 KV、运行时保留空间、推理 workspace | 与训练阶段不同，共用设备也不意味着自动共享全部内存 |

序列长度和 microbatch 可以改变激活与临时 workspace，而常驻参数保持不变。因此峰值的增长不能全部归入参数，也不能仅凭理论总量除以卡数解释。理论账本与物理测量之间的差额来自生命周期重叠、buffer 和分配器等额外状态。

## `nvidia-smi` 的 used memory 不是 tensor 总和

`nvidia-smi` 是 NVIDIA 的设备状态查看工具。Allocator 是分配/回收内存的软件；buffer 是一块存放数据的缓冲区；context 是进程使用 GPU 所需的运行环境；NCCL 是后文介绍的多 GPU 通信库。表中的 graph-private pool 指为重复执行 GPU 操作图保留的内存池，先将它算作运行时开销即可。

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

PyTorch 是常用的张量计算和训练框架，OOM（Out Of Memory）表示内存不足。PyTorch 的 `memory_allocated()` 是活跃 tensor 使用量；`memory_reserved()` 是 caching allocator 从 CUDA 保留的总量；`nvidia-smi` 还会包含 context、非 PyTorch allocation 与其他库。因此 `reserved - allocated` 不全是“泄漏”，而 OOM 也可能发生在仍有总空闲量时，因为缺少足够大的连续 block 或合适 segment。

## 训练显存的第一版心算

设模型参数量为 $P$，数据并行度为 $D$。以混合精度 Adam 为例，实际字节取决于框架是否保存 FP32 master parameter、梯度 dtype 与 optimizer 实现。一个常见但必须标注假设的估算是：

| 状态 | 每参数字节（示例） |
|---|---:|
| BF16 参数 | 2 |
| BF16 梯度 | 2 |
| FP32 master 参数 | 4 |
| Adam 一阶、二阶矩 | 8 |
| 合计 | 16 B/parameter |

这张表的梯度改用 BF16，因此比开头 FP32 梯度的例子每参数少 2 字节：1B 模型从 18 GB 变成 16 GB。两张表采用不同精度假设，不能混用。

下面的跨卡估算是预览：rank 是通信参与进程的编号，常见配置是一进程管理一张 GPU；shard 是切分后的一片状态。DP 将数据分给多个模型副本；ZeRO-3/FSDP 则可把参数、梯度和优化器状态分片保存，使用前临时收集（gather），细节会在并行章节展开。

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

例：$L=80,n_{kv}=8,d_h=128,b=2$，每 token 为 $2\times80\times8\times128\times2=327{,}680$ B，约 320 KiB；一条 32K（这里 K=1024，即 32,768）token 请求约 10 GiB（尚未算 block padding 和 metadata）。这解释了长上下文服务为何常由 KV 而非权重限制并发。

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
5. 单独列出 headroom（未分配的容量余量），用最大形状、恢复和动态图峰值验证余量是否足够；不存在适用于所有工作负载的固定安全比例。

## 把账本放到时间线上

假设某次训练的常驻状态按上例为 18 GB，前向末尾还有 6 GB 激活；随后反向释放 4 GB 激活，并新建 3 GB 临时工作区。两个时刻分别是 24 GB 和 23 GB，而不是把所有出现过的对象相加得到 27 GB。峰值来自**同一时刻仍存活的数据**。

这是简化例子，假设梯度空间已计入常驻状态，并忽略分配器和运行时开销。真实账本应沿时间线记录每项的创建、最后一次使用和释放，最后再与设备观测对照。

## 资料

- [PyTorch CUDA semantics：Memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- [PyTorch Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)

---

<!-- main-route -->

一张卡装不下时，状态需要跨卡分布。下一篇先解释 AllReduce、AllGather 等操作到底让每张卡得到什么。

[上一篇：GEMM 的分块与复用](<../05_Kernels/GEMM与高性能Kernel模式.md>) · [主线 7/26](<../guides/系统阅读路线.md>) · [下一篇：通信原语与 Collective](<../07_Communication/通信原语与Collective.md>)

<!-- /main-route -->
