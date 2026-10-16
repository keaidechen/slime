# 05 · Hopper 与 CuTe DSL：FA3/FA4


**阅读顺序：§1–4 → §8 → §9.1–9.4。** FP8 的 §5 和 Blackwell 的 §7、§9.5 是后续专题。Hopper、Blackwell 是 GPU 架构名称；文件名中的 `sm90`、`sm100` 是目标架构标识，不表示这个 GPU 恰好有 90 或 100 个 SM。CuTe DSL 也有 SM80 路径，因此下面比较 FA4 的 Softmax 时会引用 `flash_fwd.py`，而不只引用新硬件专用文件。

## 1. 分块算法已经正确，接下来优化等待

一个 K/V tile 要经历搬运、QK、Softmax、PV。它们之间有数据依赖，但不同 tile 的部分阶段可以同时进行。例如计算当前 tile 时，可以准备下一 tile 的数据。

FA2 已有 `cp.async` 搬运流水。Hopper 提供更适合大块张量搬运与异步矩阵乘的硬件接口，FA3 进一步重排协作关系。[FA3 论文](https://arxiv.org/abs/2407.08608)讨论数据搬运与计算重叠、矩阵乘与 Softmax 交错，以及 FP8 低精度下的精度控制。以下通过当前代码解释具体机制，不将论文峰值性能套用于任意输入。

先沿用上一章的时间模型：每个 tile 搬运耗时 $T_L=3$、计算耗时 $T_C=5$，单位可以理解为假想的微秒。处理 n 块时，串行总时间是 $n(T_L+T_C)$；若搬运和计算完全重叠、有足够槽位且没有其他瓶颈，则第一块先加载，之后稳态间隔由较慢阶段决定：

$$T_{\mathrm{pipeline}}\approx T_L+T_C+(n-1)\max(T_L,T_C).$$

n=4 时，串行为 32，流水为 `3+5+3×5=23`。最初的加载叫填充，最后的计算叫排空；块数很少时，这两段成本占比高。n 很大时，这个两阶段理想模型的加速比趋于 `(3+5)/5=1.6`，不会无限增长。真实 attention 还有 QK→Softmax→PV 的依赖、资源共享和同步成本，因此这只是解释收益上限的模型。

## 2. 先认识四个新词

| 名称 | 这里的含义 | 解决的问题 |
|---|---|---|
| TMA | Tensor Memory Accelerator，张量搬运机制 | 用描述符等信息提交张量搬运，减少逐元素地址处理负担 |
| WGMMA | warp group 级的矩阵乘加指令接口 | 让一组协作线程提交异步矩阵乘 |
| warp group | 这里由 4 个 warp、128 个线程组成的协作单位 | 匹配 Hopper 的相关矩阵乘执行方式 |
| warp specialization | 让不同线程组长期承担不同职责 | 搬运者和计算者可以各自推进工作 |

“异步”意味着发起后无需立即阻塞到完成，**不意味着可以提前读结果**。消费结果之前仍要等待对应操作完成，复用存储之前仍要确认先前读取结束。

TMA 描述符记录张量地址、形状、stride 和相关布局信息，使搬运硬件能按一个张量区域组织传输；发出请求的 GPU 线程不必逐元素重复完成同样的地址工作。WGMMA（Warpgroup Matrix Multiply-Accumulate）则接受协作线程组提交的矩阵乘任务。CPU 负责启动 kernel，kernel 内的线程安排搬运和计算，硬件执行相应操作；三个层次不能混成“Python 调一次函数就已把数据搬完”。

在本章选取的路径中，可以把关键依赖写为 `K 就绪 → 提交 QK → QK 完成 → Softmax → P 就绪`，以及 `P 与 V 都就绪 → 提交 PV → PV 完成`。同一 tile 的 Softmax 不能越过它依赖的 QK 完成点。所谓交错，是让不依赖该结果的其他 tile、其他 warp group 或搬运工作在空隙中前进，而不是删除数学依赖。

## 3. FA3 的 producer/consumer 代码

[`hopper/flash_fwd_kernel_sm90.h`](<../../../../../flash-attention/hopper/flash_fwd_kernel_sm90.h>) 用 `warp_group_idx == 0` 区分职责，后面的精读会逐段展开。producer 分支负责组织加载；后面的 consumer 分支负责矩阵乘、softmax 与输出。producer 减少寄存器额度、consumer 增加额度的语句也在这里，因为搬运与累加所需寄存器量不同。

**producer（生产者）**准备数据，**consumer（消费者）**使用数据计算。下面的“槽”是一块可重复使用的 shared 缓冲区；`stage` 表示流水中的阶段/槽位组织，`WG` 是 warp group 的缩写。这里的角色由同一个 GPU kernel 内不同线程承担，不能把 producer 当成 CPU、consumer 当成 GPU。

```text
producer：拿到可写槽 → 提交 K/V 搬运 → 发出就绪信息 → 继续准备后续块
consumer：等待就绪 → 做 QK/Softmax/PV → 释放已消费槽 → 处理下一块
```

这是**概念时序**，具体分支不一定让整个 producer warp group 的所有线程都执行相同搬运指令。源码的 `Use_TMA_KV`、`SingleProducerWarp` 等条件会改变参与者；读代码时要跟随当前实例条件。

| 要观察的内容 | 源码落点 |
|---|---|
| Python 入口与保存状态 | [`hopper/flash_attn_interface.py`](<../../../../../flash-attention/hopper/flash_attn_interface.py>)：`FlashAttnFunc` |
| 参数校验与分发 | [`hopper/flash_api.cpp`](<../../../../../flash-attention/hopper/flash_api.cpp>) |
| producer/consumer 与寄存器分配 | [`flash_fwd_kernel_sm90.h`](<../../../../../flash-attention/hopper/flash_fwd_kernel_sm90.h>) |
| 搬运、MMA 与等待的主循环 | [`mainloop_fwd_sm90_tma_gmma_ws.hpp`](<../../../../../flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp>) |
| 输出转换和写回 | [`epilogue_fwd.hpp`](<../../../../../flash-attention/hopper/epilogue_fwd.hpp>) |
| 工作 tile 怎样被领取 | [`tile_scheduler.hpp`](<../../../../../flash-attention/hopper/tile_scheduler.hpp>) |

FA2 中 `blockIdx.x` 很直接地决定 Q tile；到了带持久化调度的实现，一个 block 可以继续领取多个工作 tile。看见 scheduler 时，先找“谁领取下一个 tile”，再找“这个 tile 的数学计算”。

**scheduler（调度器）**分配尚未处理的 tile；“持久化”在这里表示 block 做完一份工作后继续领下一份，从而复用执行资源。

比如总共有 6 个 Q tile，启动 2 个持久化 block，调度器让它们在各自完成当前任务后继续领取剩余 tile。逻辑 tile 编号便不再等于 `blockIdx.x`；输出地址要从领取的工作坐标得到。领取新 tile 时，其 m/l/a 必须重新建立，不能把前一个 query 块的归一化状态带过去。持久化可能改善工作分配，但也增加调度状态；每个 tile 的实际耗时受 causal 范围、长度等影响，不能只按 tile 个数判断负载是否均匀。

## 4. 环形缓冲区为什么需要阶段信息

假设共享内存只有两个槽，producer 依次装 K0、K1、K2、K3：

| 逻辑数据 | 槽位 | 本轮复用关系 |
|---|---:|---|
| K0 | 0 | 第一次使用槽 0 |
| K1 | 1 | 第一次使用槽 1 |
| K2 | 0 | 必须等 K0 被消费后才能覆盖 |
| K3 | 1 | 必须等 K1 被消费后才能覆盖 |

只知道“槽 0 就绪”不够，因为它可能表示旧 K0 或新 K2。因此 pipeline 还维护阶段/轮次信息，区分一次次复用。理解这个约束后，再看 barrier（同步对象）和 pipeline state，代码会更具体。

把槽 0 的一生按事件列出来，可以发现有两种不同的“完成”：

| 时刻 | 搬运者/硬件的动作 | 计算者允许做什么 |
|---|---|---|
| 槽为空 | producer acquire 获得可写资格 | 不能读取，里面还不是 K0 |
| 搬运在途 | producer 提交 K0，硬件逐步写入 | 等待本轮数据就绪 |
| 搬运完成 | 与该轮绑定的完成条件满足 | 可提交读取 K0 的 QK |
| QK 在途 | producer 可以推进其他空槽 | 不能仅因指令已提交就释放槽 0 |
| QK 完成 | consumer release 宣告旧读取结束 | producer 才可用同一槽写 K2 |

“已到达”保护读者，“已释放”保护下一位写者。若 producer 太快，槽会全部占满，它必须等待；这叫背压。增加槽数能让它提前准备更多数据，却占用更多 shared memory，可能降低驻留 block 数。若计算阶段已一直繁忙，增加槽数也不会使 Tensor Core 本身算得更快。

在 FA4 的 [`pipeline.py`](<../../../../../flash-attention/flash_attn/cute/pipeline.py>) 中，`PipelineStateSimple` 用 `_phase_index` 推导槽位 `index` 和阶段 `phase`。这是便于阅读的实现例子；Hopper C++ 路径使用自身的 pipeline 类型，不能把 Python helper 当成其直接调用关系。

## 5. FP8 的速度与精度要一起看

FP16/BF16 每个元素通常为 16 位，FP8 为 8 位。更低精度可能减少数据量、匹配更高吞吐的矩阵乘，但其可表示范围和精度需要缩放策略配合。

FA3 接口中的 `q_descale`、`k_descale`、`v_descale` 是寻找反量化尺度的入口；`flash_fwd_kernel_sm90.h` 中能看到 Q/K descale 合并进 softmax scale 的代码。直观上，如果输入存储的是按尺度缩小的数，计算 score 和输出时要补回相应尺度。

论文中的块量化与 incoherent processing（用于改善量化误差分布的处理）属于完整精度方案。一个函数接受 FP8 dtype，并不能证明调用者已经完成了论文里的所有前处理。应沿数据准备、descale 传入和误差对比一起核查，而不只看矩阵乘的类型。

以无 softcap 的 FP8 分支为例，真实的尺度补偿是：

源码摘录：[flash_fwd_kernel_sm90.h，408–415 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/hopper/flash_fwd_kernel_sm90.h#L408-L415)。

<!-- source: hopper/flash_fwd_kernel_sm90.h:408-415 -->
```cpp
float softmax_scale_log2 = params.mainloop.softmax_scale_log2;
if constexpr (Is_FP8 && !Has_softcap) {
    int const bidh = get<1>(block_coord);
    int const bidh_kv = !PackGQA ? params.mainloop.qhead_per_khead_divmod.divide(bidh) : bidh;
    float const q_descale = params.mainloop.ptr_q_descale == nullptr ? 1.0f : params.mainloop.ptr_q_descale[bidb * get<0>(params.mainloop.stride_q_descale) + bidh_kv * get<1>(params.mainloop.stride_q_descale)];
    float const k_descale = params.mainloop.ptr_k_descale == nullptr ? 1.0f : params.mainloop.ptr_k_descale[bidb * get<0>(params.mainloop.stride_k_descale) + bidh_kv * get<1>(params.mainloop.stride_k_descale)];
    softmax_scale_log2 *= q_descale * k_descale;
}
```

若存储的量化值为 `Q8/K8`，对应实数近似为 `Q8*q_descale`、`K8*k_descale`，则 score 为 `alpha*(Q8@K8.T)*q_descale*k_descale`。所以这里把两个 descale 的乘积并进 Softmax 的换底尺度，无需先在显存里生成一整份反量化 Q/K。空指针表示采用默认尺度 1；`bidh_kv` 和 stride 表明尺度按 batch/KV head 寻址。

例如原 `softmax_scale_log2=alpha*log2(e)`，两个 descale 为 0.5 和 0.25，那么新尺度是原来的 0.125。V 的 descale 则在主循环尾声传给 `softmax.finalize(v_descale)`，参与最终输出的尺度恢复。Q/K 的尺度影响概率，V 的尺度影响被加权的向量，两处职责不同。

### 5.1 从存储值到注意力输出的尺度手算

为只观察尺度，取一个 query、两个 key、`d=d_v=1`：存储数 `Q8=[[2]]`、`K8=[[0],[4]]`、`V8=[[2],[4]]`，反量化尺度分别为 `s_Q=1/2、s_K=1/4、s_V=5`。它们对应 `Q=[[1]]、K=[[0],[1]]、V=[[10],[20]]`。暂不考虑进一步的量化舍入，alpha=1，则：

```text
低精度存储值做点积： Q8 @ K8.T = [0, 8]
补 Q/K 尺度：        S = [0, 8] * (1/2) * (1/4) = [0, 1]
Softmax：            P = [1/(1+e), e/(1+e)]
先汇总存储的 V8：    A = (2+4e)/(1+e)
恢复 V 尺度：        O = 5*A = (10+20e)/(1+e) ≈ 17.310586
```

若漏掉 Q/K descale，Softmax 会对 `[0,8]` 归一化，几乎只选第二个 key；这无法靠最后给 O 乘一个常数修复。若只漏掉 V descale，概率仍正确，但 O 少乘了 5。这区分了“影响概率的尺度”和“影响输出向量的尺度”。真实接口需要满足相应 head dimension 等约束，此处是代数算例。

实际量化还包含舍入或饱和：$\widehat Q=s_Q\,\operatorname{cast}_{FP8}(Q/s_Q)$，其中 cast 表示转换到有限的 FP8 表示集合。太小的尺度可能使大值超出范围，太大的尺度又可能使小值难以分辨；乘回 descale 只能恢复尺度，不能撤销这种信息损失。上述理想例子故意选简单存储值，以免把尺度错误与量化误差混在一起。

论文中的 incoherent processing 可从一个保内积变换理解：若正交矩阵 R 满足 $RR^T=I$（I 为单位矩阵），则 $(QR)(KR)^T=QRR^TK^T=QK^T$。适当的共同变换可将集中在少数特征上的大值分散，改善后续低精度表示；但变换、量化和补偿都需要成本，且量化后的误差并不因此严格为零。仅传入 FP8 张量和 descale，不足以证明调用者采用了这套额外方案。

## 6. FA4 用 Python 写 kernel，代码仍在 GPU 执行

[`flash_attn/cute/interface.py`](<../../../../../flash-attention/flash_attn/cute/interface.py>) 的公开函数接收 PyTorch 张量，内部 `_flash_attn_fwd` 选择架构和配置，然后经 `cute.compile` 编译，结果保存在编译缓存中。**DSL** 是领域专用语言：这里使用 Python 语法描述 GPU 计算，由工具生成 GPU 代码，不是用普通 Python 循环逐元素算 attention。

这和 FA2 安装时编译 C++/CUDA 扩展的路径不同。FA4 首次遇到某些配置可能产生 JIT（即时编译）开销；计时必须区分首次编译和缓存后执行。

| FA4 文件 | 建议带着的问题 |
|---|---|
| [`interface.py`](<../../../../../flash-attention/flash_attn/cute/interface.py>) | 输入是否合法？选择哪一架构实现？缓存键包含哪些配置？ |
| [`flash_fwd_sm90.py`](<../../../../../flash-attention/flash_attn/cute/flash_fwd_sm90.py>) | Hopper 如何分配 producer/MMA 线程和流水？ |
| [`flash_fwd_sm100.py`](<../../../../../flash-attention/flash_attn/cute/flash_fwd_sm100.py>) | Blackwell 的 TMEM 和 CTA 协作如何组织？ |
| [`softmax.py`](<../../../../../flash-attention/flash_attn/cute/softmax.py>) | `row_max`、`row_sum`、`rescale_O` 如何对应在线公式？ |
| [`mask.py`](<../../../../../flash-attention/flash_attn/cute/mask.py>) | causal/local/custom mask 怎样落到 tile？ |
| [`cache_utils.py`](<../../../../../flash-attention/flash_attn/cute/cache_utils.py>) | 编译结果怎样复用？ |

`softmax.py` 的基础 `Softmax.online_softmax` 是跨版本比较的好入口；专用实现还可能有更复杂的缩放策略，应按实际使用的类继续追踪。

## 7. Blackwell 不能只套用“所有累加器都在寄存器”

第 03 节的寄存器片段图针对 FA2。FA4 的 SM100 代码还使用 **TMEM（Tensor Memory）**，这是与相关 Tensor Core 操作配合的专用存储。`flash_fwd_sm100.py` 中可以搜索 `tmem_s_offset`、`tmem_o_offset`，观察 score 和输出区域如何安排。

同文件的 `use_2cta_instrs` 控制某些配置下的双 CTA 协作：`cta_group_size` 为 2 时，矩阵乘覆盖两个 CTA 协作的范围，每个 CTA 持有自己的一部分 Q 行。它增加了跨 CTA 的资源与同步要求。这是特定实现条件下的机制，不能据此把第 00 节的普通 block 局部同步推广为任意 block 间同步。

FA4 在本系列作为**当前源码演进**讲解；FA1/2/3 论文不能替代对这些具体代码分支的验证。硬件、head dimension、dtype 与功能组合的支持范围应查看固定快照的接口校验与测试。

**CTA（Cooperative Thread Array，协作线程阵列）**在这里就是 CUDA thread block。TMEM 是 Tensor Core 配套的专用片上存储，不能当成另一份可由普通 global 指针随意访问的 HBM。理解它时仍用同一套问题：哪条矩阵乘写 score，哪些线程在完成通知后读取 score 做 Softmax，后续 PV 何时更新 O，哪一步确认旧读者结束后才复用区域。双 CTA 协作还要按该机制约定协调两个参与块；一个块的 `__syncthreads()` 仍只能证明自己块内的会合。

## 8. 源码精读：Hopper 的搬运者如何与计算者交接

这里选 `Use_TMA_KV=true、PagedKVNonTMA=false、HasQv=false` 理解主干，并使用代码中 `No intra-WG overlap` 分支讲清同步。更激进的交叠分支会把等待进一步后移；不能把下面时序说成所有 Hopper 配置的唯一实现。

### 8.1 角色和等待条件先在 pipeline 中登记

源码摘录：[flash_fwd_kernel_sm90.h，219–226 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/hopper/flash_fwd_kernel_sm90.h#L219-L226)。

<!-- source: hopper/flash_fwd_kernel_sm90.h:219-226 -->
```cpp
PipelineParamsK pipeline_params_k;
pipeline_params_k.role = warp_group_idx == 0
    ? MainloopPipelineK::ThreadCategory::Producer
    : MainloopPipelineK::ThreadCategory::Consumer;
if constexpr (Use_TMA_KV) {
    pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    pipeline_params_k.is_leader = warp_group_thread_idx == 0;
    pipeline_params_k.num_consumers = !LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
```

`role` 告诉 pipeline 当前线程是生产者还是消费者；`transaction_bytes` 告诉搬运完成机制这一阶段预期有多少字节。`is_leader` 选出组内负责相应操作的线程，`num_consumers` 登记消费参与者数量。这些信息用于判断“数据是否到齐”“何时可以重新写槽位”，并不是 attention 的数学参数。

例如一个 `64×64` 的 FP16 K tile 有 `64×64×2=8192` 字节，若这一阶段只搬这一份 tile，完成协议就要与这次 8192 字节的事务匹配。线程到达屏障与异步事务完成是不同条件；少报、漏报或将完成通知绑定到错误轮次，都可能导致过早使用或一直等待。实际 `TmaTransactionBytesK` 应从该配置计算，不能把此教学尺寸直接填到源码中。相关硬件语义见 [PTX mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)。

分工以后，两类线程采用不同寄存器额度：

源码摘录：[flash_fwd_kernel_sm90.h，308–309 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/hopper/flash_fwd_kernel_sm90.h#L308-L309)。

<!-- source: hopper/flash_fwd_kernel_sm90.h:308-309 -->
```cpp
if (warp_group_idx == 0) {  // Producer
    cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
```

源码摘录：[flash_fwd_kernel_sm90.h，359–366 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/hopper/flash_fwd_kernel_sm90.h#L359-L366)。

<!-- source: hopper/flash_fwd_kernel_sm90.h:359-366 -->
```cpp
    mainloop.load_tail(pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write, shared_storage, work_idx);
} else {  // Consumer
    cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

    // Initialize matmul objects.
    TiledMmaPV tiled_mma_pv;

    PipelineState smem_pipe_read;
```

producer 主要安排搬运，consumer 需要保存 score、Softmax 状态和 O 的累加值，所以后者需要更多寄存器。`warpgroup_reg_dealloc/alloc` 调整这类资源分配，不是在 CPU 上申请或释放一块显存。

### 8.2 `producer_acquire` 取得的是一个可以安全覆写的槽位

源码摘录：[mainloop_fwd_sm90_tma_gmma_ws.hpp，749–755 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L749-L755)。

<!-- source: hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:749-755 -->
```cpp
auto load_K = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
    pipeline_k.producer_acquire(smem_pipe_write);
    if constexpr (!PagedKVNonTMA) {
        auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_K_TMA();
        copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
            tKgK_TMA(_, n_block_idx, bidb_kv_idx), tKsK_TMA(_, smem_pipe_write.index()));
    } else {
```

逐步展开：

1. `producer_acquire(smem_pipe_write)` 等待该阶段对应的槽可以写。若 consumer 仍在使用旧 K，producer 不能覆盖它。
2. `get_indices_for_K_TMA()` 得到 TMA 所需的块和样本/页索引。
3. `.with(*producer_get_barrier(...), ...)` 把搬运和完成屏障关联起来。
4. `tKsK_TMA(_, smem_pipe_write.index())` 选目标 shared 缓冲区的当前槽。
5. `copy(...)` 提交搬运；数据到达后通过相关完成机制让 consumer 的等待条件满足。

注意 TMA 分支这里没有紧跟一个普通 `producer_commit` 调用；下方非 TMA 分支才显式结合 `cpasync_barrier_arrive` 提交。不同搬运机制的完成通知方式不同，不能仅凭“没看到 commit”断定未提交。

### 8.3 K 到了不代表 QK 已结束：两种等待管不同的事

消费端的等待 helper：

源码摘录：[mainloop_fwd_sm90_tma_gmma_ws.hpp，1053–1056 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L1053-L1056)。

<!-- source: hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1053-1056 -->
```cpp
auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
};
```

`consumer_try_wait` 先获取当前阶段的等待状态，再交给 `consumer_wait` 完成等待。接着提交 QK：

源码摘录：[mainloop_fwd_sm90_tma_gmma_ws.hpp，1261–1269 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L1261-L1269)。

<!-- source: hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1261-1269 -->
```cpp
auto smem_pipe_read_prev = smem_pipe_read;
if constexpr (!Is_first_iter) { ++smem_pipe_read; }
Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
consumer_wait(pipeline_k, smem_pipe_read);
flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
if constexpr (!HasQv) {
    warp_scheduler_barrier_arrive();
    warpgroup_wait<0>();
    pipeline_k.consumer_release(smem_pipe_read);  // release K
```

这一小段有四个不同动作：

| 语句 | 保证什么 |
|---|---|
| `consumer_wait(pipeline_k, ...)` | shared 中的 K 已到达，可以被读取 |
| `gemm<zero_init=true, wg_wait=-1>` | 提交本轮 QK，初始化该轮 score 累加器；这里不立即等待全部完成 |
| `warpgroup_wait<0>()` | 等待相应已提交矩阵乘完成，结果可以消费，旧 K 不再被它读取 |
| `consumer_release(...)` | 通知 producer 这个 K 槽可以复用 |

如果在 `warpgroup_wait` 之前释放 K，producer 可能在矩阵乘尚未读完时覆盖它。反过来，如果永远不 release，producer 走完一圈缓冲区后就没有可写槽，流水可能卡住。这就是源码中“释放”语句与正确性直接相关的原因。

### 8.4 Softmax 到 PV：哪些中间值留存，哪些可以覆盖

源码摘录：[mainloop_fwd_sm90_tma_gmma_ws.hpp，1281–1294 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L1281-L1294)。

<!-- source: hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1281-1294 -->
```cpp
scoremod_premask_fn(tSrS);
mask_fn(tSrS, n_block);
Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
if constexpr (LargeHeadDimV && !Is_first_iter) { store_scales(scores_scale, smem_pipe_read_prev.index()); }
softmax.template online_softmax</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
convert_type_out(tOrP_acc, tOrP);
if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
if constexpr (!Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
if constexpr (!MmaPV_is_RS && !MmaPV_use_RS_WG1) { arrive_on_P_write_barrier(); }
if constexpr (!HasQv) { consumer_wait(pipeline_v, smem_pipe_read); }
```

先修改/遮蔽 score，再由 `max_get_scale` 得到旧 O 需要乘的校正因子；`online_softmax` 更新逐行统计量并把 score 变成指数权重。之后 `convert_type_out` 把 FP32 权重转为矩阵乘使用的 Element 类型。

`MmaPV_is_RS` 表示这里选择的 PV 操作数路径：P 可以直接作为寄存器操作数，其他配置则需要 `write_P_to_smem`。因此“P 从不落到 HBM”不等于“P 在所有实现中都只待在寄存器”。

非首轮先执行 `softmax.rescale_o(tOrO, scores_scale)` 修正旧输出，再等待 V，就能提交 PV。对应分支最后执行：

源码摘录：[mainloop_fwd_sm90_tma_gmma_ws.hpp，1303–1304 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L1303-L1304)。

<!-- source: hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1303-1304 -->
```cpp
warpgroup_wait<0>();
pipeline_v.consumer_release(smem_pipe_read);  // release V
```

这与 K 的规则对称：V 搬完才能用，PV 读完才能释放。槽位流水中的 `acquire/wait/release` 管存储所有权；Softmax 中的 `scores_scale` 管数学基准变更。两套机制共同保证“并行后仍然算对”。

## 9. 源码精读：FA4 的 Python kernel 并未省掉这些状态

### 9.1 在线更新返回的 `row_scale` 必须真正作用到 O

下面是 `Softmax.online_softmax` 的非首轮分支。为便于对照前文，`row_max_prev` 是旧 m，`row_max_cur` 是新 m，`scale_log2=alpha*log2(e)`：

源码摘录：[softmax.py，234–250 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/softmax.py#L234-L250)。

<!-- source: flash_attn/cute/softmax.py:234-250 -->
```python
    else:
        row_max_cur_scaled = row_max_cur * scale_log2
        acc_S_row_exp = cute.math.exp2(
            acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
        )
        # row_scale[r] = cute.math.exp2(row_max_prev * self.scale_log2 - row_max_cur_scaled)
        row_scale[r] = cute.math.exp2(
            (row_max_prev - row_max_cur) * scale_log2, fastmath=True
        )
        acc_S_row_sum = utils.fadd_reduce(
            acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch
        )

    row_sum[r] = acc_S_row_sum
    acc_S_mn[r, None].store(acc_S_row_exp)

return row_scale
```

逐句翻译成数学：

```text
acc_S_row_exp = exp(alpha * (当前 raw score - 新 m))
row_scale    = exp(alpha * (旧 m - 新 m))
新 row_sum   = sum(本线程的新指数权重) + 旧 row_sum * row_scale
acc_S 的存储 = 本轮指数权重
返回 row_scale
```

`fadd_reduce` 的 `init_val` 不是随便选的初始数；它把经新基准校正后的历史分母接到当前局部和上。注意 helper 还没有收到 O 参数，因此仅调用它不会自动修正 O。

FA4 的 SM80 路径 `flash_fwd.py` 有一段清晰的调用点：

源码摘录：[flash_fwd.py，1185–1191 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/flash_fwd.py#L1185-L1191)。

<!-- source: flash_attn/cute/flash_fwd.py:1185-1191 -->
```python
if const_expr(mask_fn is not None):
    mask_fn(acc_S, n_block=n_block)
row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
softmax.rescale_O(mma_params.acc_O, row_scale)
rP = cute.make_fragment_like(acc_S, self.dtype)
rP.store(acc_S.load().to(self.dtype))
tOrP = layout_utils.reshape_acc_to_frgA(rP)
```

下一步调用 `sm80_utils.gemm_rs` 做 PV。顺序因此是：mask → 更新 m/l → 修正旧 O → 将本轮 P 转输入 dtype → PV 累加。`rescale_O` 的核心是：

源码摘录：[softmax.py，298–301 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/softmax.py#L298-L301)。

<!-- source: flash_attn/cute/softmax.py:298-301 -->
```python
acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
for r in cutlass.range(cute.size(row_scale), unroll_full=True):
    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
```

`r,None` 表示选第 r 行、保留该行的全部列；`load/store` 在这里操作 kernel 的片段，不是普通 Python 从文件读取、保存。这一行让同一 query 的所有输出特征列乘同一个修正因子。

例如旧最大值 `ln(2)`、新最大值 `ln(8)`，取 alpha=1，返回的 scale 为 `1/4`。旧 O 分子若为 `2.5`，必须先变成 `0.625`，再加本轮 PV。漏掉 `rescale_O`，即便最终除以正确分母，输出仍会错。

### 9.2 finalize 会把 `row_sum` 原地改成 LSE

源码摘录：[softmax.py，263–265 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/softmax.py#L263-L265)。

<!-- source: flash_attn/cute/softmax.py:263-265 -->
```python
# quad reduction for row_sum as we didn't do it during each iteration of online softmax
row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
row_scale = cute.make_fragment_like(row_max, Float32)
```

和 FA2 一样，前面只保存线程内部分和，最后才通过 width=4 的归约得到整行分母。无 sink 时，后续核心如下：

源码摘录：[softmax.py，276–288 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/softmax.py#L276-L288)。

<!-- source: flash_attn/cute/softmax.py:276-288 -->
```python
    # if row_sum is zero or nan, set acc_O_mn_row to 1.0
    acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
    row_scale[r] = (
        cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
    ) * final_scale
    row_sum_cur = row_sum[r]
    LN2 = math.log(2.0)
    row_sum[r] = (
        (row_max_scaled + cute.math.log2(row_sum_cur, fastmath=True)) * LN2
        if not acc_O_mn_row_is_zero_or_nan
        else -Float32.inf
    )
return row_scale
```

它返回 `final_scale / row_sum` 供最终 O 归一化；与此同时，`row_sum` 的存储改写成自然对数单位的 LSE：

```text
(row_max * alpha * log2(e) + log2(sum)) * ln(2)
= alpha * row_max + ln(sum)
```

所以 finalize 之后不能再把 `row_sum` 当分母使用。这里空行的 LSE 选择 `-inf`，与前面 FA2 常规输出使用 `+inf` 哨兵的实现约定不同；跨实现比较中间状态时必须核对用途，不能只根据同名变量假设完全相同。

### 9.3 轮次不是另一个数组：由一个整数推导

源码摘录：[pipeline.py，56–76 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/pipeline.py#L56-L76)。

<!-- source: flash_attn/cute/pipeline.py:56-76 -->
```python
@property
def index(self) -> Int32:
    if const_expr(self._stages == 1):
        return Int32(0)
    else:
        return self._phase_index % self._stages

@property
def phase(self) -> Int32:
    # PTX docs say that the phase parity needs to be 0 or 1, so by right we need to
    # take modulo 2. But in practice just passing the phase in without modulo works fine.
    if const_expr(self._stages == 1):
        return self._phase_index
    else:
        return self._phase_index // self._stages

def advance(self):
    if const_expr(self._stages == 1):
        self._phase_index ^= 1
    else:
        self._phase_index += 1
```

对于两槽 pipeline，消费者从 `_phase_index=0` 开始，连续推进得到：

| `_phase_index` | `index`：槽 | `phase`：当前代码返回的轮次 |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 0 | 1 |
| 3 | 1 | 1 |
| 4 | 0 | 2 |

第 0 次与第 2 次都使用槽 0，但 phase 不同，因此等待者能区分不同轮次的通知。源码注释说明 PTX 接口定义的是 phase parity；当前 helper 对多槽情况返回整除后的整数，不能把它转述为“函数始终只返回 0/1”。单槽则专门用 `^=1` 翻转。

`parity` 指奇偶性，PTX 是 NVIDIA 的底层指令表示。这里不能把 phase 想成永不重复的唯一任务编号：相位会复用，安全性还依赖 acquire/release 保证生产者不会越过尚未消费的阶段。

生产者的起始状态也不是照搬消费者：

源码摘录：[pipeline.py，86–95 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/pipeline.py#L86-L95)。

<!-- source: flash_attn/cute/pipeline.py:86-95 -->
```python
def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
    """
    if type is PipelineUserType.Producer:
        return PipelineStateSimple(stages, Int32(stages))
    elif type is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."
```

生产者从 `stages` 开始，对应槽 0、phase 1；消费者从 0 开始，对应槽 0、phase 0。初始化的相位差与“最初缓冲区是空的”这一约定配合。这是此 Python pipeline helper 的协议，不能直接拿这些值替换 Hopper C++ 的 pipeline state。

### 9.4 `cute.compile` 与执行是分开的两步

接口先构造缓存键，开头包含 dtype、Q/K head dimension、V head dimension、GQA 比例、causal 等配置：

源码摘录：[interface.py，1198–1206 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/interface.py#L1198-L1206)。

<!-- source: flash_attn/cute/interface.py:1198-1206 -->
```python
compile_key = (
    dtype,
    head_dim,
    head_dim_v,
    qhead_per_kvhead,
    nheads_per_kv,
    causal,
    score_mod_hash,
    mask_mod_hash,
```

这只是键的开头，后面还包含 mask、变长、分页、架构和 tile 等特征。缓存不存在才进入编译分支：

源码摘录：[interface.py，1266–1267 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/interface.py#L1266-L1267)。

<!-- source: flash_attn/cute/interface.py:1266-1267 -->
```python
if compile_key not in _flash_attn_fwd.compile_cache:
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
```

在常规无 Qv 的分支，准备参数后调用：

源码摘录：[interface.py，1554–1555 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/interface.py#L1554-L1555)。

<!-- source: flash_attn/cute/interface.py:1554-1555 -->
```python
compile_args.append(current_stream)
_flash_attn_fwd.compile_cache[compile_key] = cute.compile(*compile_args, options="--enable-tvm-ffi")
```

实际输入的执行则在后面的非 `fake_mode` 分支发生：

源码摘录：[interface.py，1651–1651 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/interface.py#L1651-L1651)。

<!-- source: flash_attn/cute/interface.py:1651-1651 -->
```python
_flash_attn_fwd.compile_cache[compile_key](*call_args)
```

缓存保存的是**可执行的编译结果**，不是某次 attention 的数值输出。同配置换一批 Q/K/V，仍要重新执行计算；改变需要专门编译的配置，则可能产生新 kernel。由此也能理解计时为什么要先 warmup：首轮可能混入编译、初始化等工作。

### 9.5 SM100 的 tile 与 TMEM 偏移也可以直接代入计算

源码摘录：[flash_fwd_sm100.py，200–209 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/flash_fwd_sm100.py#L200-L209)。

<!-- source: flash_attn/cute/flash_fwd_sm100.py:200-209 -->
```python
self.cta_group_size = 2 if self.use_2cta_instrs else 1
# cta_tiler M includes only 1 CTA, the scheduler will take into account the cluster shape
self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
# With 2CTA, the MMA tiler M covers both CTAs, so it's cta_group_size * m_block_size.
# Each CTA owns m_block_size rows; the 2CTA MMA instruction spans both.
self.mma_tiler_qk = (self.cta_group_size * m_block_size, n_block_size, self.head_dim_padded)
self.mma_tiler_pv = (self.cta_group_size * m_block_size, self.head_dim_v_padded, n_block_size)
self.qk_acc_dtype = Float32
self.pv_acc_dtype = Float32
self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
```

`cta_group_size` 是 1 或 2。以 `m_block_size=128` 为例，单 CTA 的 QK MMA 高度为 128；启用双 CTA 指令则为 256，由两个 CTA 各负责 128 行。`cta_tiler` 还包含 `q_stage`，描述调度中的 Q 阶段，不能与单次 MMA 的形状混为一谈。

TMEM 空间的前半段规划如下：

源码摘录：[flash_fwd_sm100.py，349–355 行](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/flash_fwd_sm100.py#L349-L355)。

<!-- source: flash_attn/cute/flash_fwd_sm100.py:349-355 -->
```python
self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
self.tmem_o_offset = [
    self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
    for i in range(self.q_stage)
]  # e.g., 256, 384
self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
assert self.tmem_total <= self.tmem_alloc_cols
```

取 `n_block_size=128、head_dim_v_padded=128、q_stage=2`：两份 score 区域起点为 0、128，输出区域起点为 256、384，总计到 512。这里计的是该 TMEM 布局的**列偏移**，不是字节数，也不是 512 个普通标量。`tmem_alloc_cols` 是可用列容量，assert 检查规划是否放得下。

这段代码只规划位置，尚未搬运或计算。它同时给出一个具体的资源约束：改变 tile 或 V 的维度，会改变 TMEM 占用；即便数学公式相同，配置也可能因片上容量不够而无法使用。


## 10. 常见误解与边界

- producer 已提交搬运，consumer 能否立刻读？**不能，须等对应完成条件。**
- 为什么要跟踪缓冲区的轮次？**同一槽位被重复使用，必须区分新旧数据。**
- FA4 文件是 `.py`，是否意味着计算在 CPU？**不意味着，DSL 会编译 GPU kernel。**
- 能否把 FA2 的寄存器布局图完全照搬到 SM100？**不能，那里还使用 TMEM 与不同的协作机制。**

最后一篇解释数学等价、接口约束与执行时间之间的证据边界。
