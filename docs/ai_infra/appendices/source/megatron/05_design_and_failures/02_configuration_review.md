# Megatron 配置背后的状态、拓扑与失败边界

一套训练配置描述的不只是 GPU 数量，而是模型状态放在哪里、每次更新消费什么数据，以及哪些依赖必须在更新前完成。global batch 固定时提高 DP，会减少每卡样本或 microbatch 数；性能变化因此包含调度和算子颗粒度的变化。

## 模型与批量共同决定执行形状

MBS（Micro Batch Size）是每个数据并行参与者一次前反向处理的样本数；GBS（Global Batch Size）是一次逻辑更新的全局样本数。固定均匀批次时 GBS=MBS×DP 大小×累积次数。

这里的样本批量与推理时的动态 batch 不同；变长训练还需独立记录有效 token 分母。

参数量、层数、hidden、heads/GQA、FFN、词表、seq length 与 dense/MoE 定义模型；MBS、GBS、DP、microbatch 数和 token budget 定义工作量。参数、梯度、归约精度、loss scaling 和 FP8 recipe 决定数值路径。源码 commit、镜像、GPU、driver、CUDA/NCCL/PyTorch/TE 限定实际实现。

以一个贯穿评审的小方案为例：8 张 GPU、dense decoder、4 层、`H=4`、2 个 attention heads、`S=4`，`TP=2,PP=2,CP=1,MBS=1,GBS=8`，先关闭 dropout 并采用 BF16 参考。H 与 head 数能按 TP 划分、每 stage 两层；dense DP=`8/(2×2)=2`，microbatch 数 `M=8/(1×2)=4`。每副本消费四个样本，总共八个；若每个样本四个有效目标，分母为 32 token。

若只将 DP 从 2 提到 4、保持 GBS/MBS，M 会降到 2。PP=2 的理想 bubble 从 `1/(4+1)=20%` 升到 `1/(2+1)≈33.3%`。增加数据副本可能提高总并发，也同时减少流水线的填充工作，不能仅凭 GPU 更多就预测线性加速。此比较还需要提供额外设备或改变其他并行维度，固定 W=8 时不能独立改变 DP。

## 两套网格怎样映射到硬件

ETP（Expert Tensor Parallelism，专家张量并行）把同一个专家的矩阵拆给多个参与者；它与把不同专家分给不同设备的 EP 是两个切分维度。

EDP（Expert Data Parallelism，专家数据并行）表示同一专家参数分片的多个数据副本，需合并各副本处理不同样本得到的梯度。它不能替代专家矩阵内部的 ETP 合并。

在本文的标准 decoder 布局中，dense 网格为 `W=TP×CP×DP×PP`，expert 网格为 `W=ETP×EP×EDP×PP`。整除条件与两套 PP groups 的相容性决定组是否有效。TP、PP、CP、EP、ETP、DP/EDP 是否跨节点，以及 rank 到 host/GPU/NIC 的映射，决定通信路径。

未指定 ETP 时需要按当前参数解析的最终值理解，不能默认 1。ETP 变大还可能改变 EP/EDP。每 stage 层数、VPP chunks 和 microbatch 依赖决定 warmup 与 bubble；消息大小、频率和可重叠区间决定通信成本。

小方案若采用变化顺序 `tp-pp-dp`，两台机器各四卡，global ranks 0..3 在机器 A、4..7 在 B，则 TP groups `{0,1}`、`{2,3}`、`{4,5}`、`{6,7}` 均在节点内；PP groups 如 `{0,2}` 也在节点内；DP groups `{0,4}` 等跨节点。这个放置只是示例，刻意让频繁 TP 留在高速互联域。实际用 `RankGenerator` 打印出来的成员与 host/NIC 映射才是验收依据。

若在同一 world 增加 MoE，设 `ETP=1,EP=2,PP=2`，EDP=2；设 ETP=2 则 EDP=1。dense DP 仍按原网格解释，expert 参数却少了一份副本。评审表必须分开记录 dense 与 expert 的梯度组，不能把 EDP=1 读成“所有参数都没有数据并行”。

## 稳态分片与瞬时峰值

参数、gradient、master/moment、activation、通信 workspace、graph/offload buffer 与 allocator 余量在不同阶段重叠。构模、forward、backward、optimizer/param gather 的峰值可能来自不同对象。DDP+Distributed Optimizer 与 FSDP 也具有不同参数生命周期和 checkpoint state dict，不能由“都分片”推断恢复方式相同。

把“预算足够”写成可检查的不等式：`峰值状态+峰值时刻的激活/通信/工作区+运行余量≤设备可用容量`，其中前几项必须在同一时刻核算。假设测得常驻 12 GiB、backward 激活 5 GiB、同时存在的通信 buffer 2 GiB，预留 2 GiB，共 21 GiB；若目标可用 24 GiB，余量只有 3 GiB。开启需额外 4 GiB 的参数预取，会使预算超限，即使静态参数分片比例没变。

运行余量应来自目标环境的测量和波动，不能把这里的 2 GiB 当通用推荐。首次 optimizer state 创建、checkpoint staging 与图捕获也要纳入不同阶段，不能只测稳态 forward。

## 数据身份是正确性的组成部分

manifest、blend、split、cache 与 tokenizer metadata 定义样本。mock data 提供计算路径的参考，真实数据增加预处理和 data-wait；二者不具有相同供给成本。loss、grad norm 和短程 hash 可以揭示变化，但恢复还需保持样本位置、LR、参数和目标拓扑之间的关系。

同一个小方案的正确性记录可以明确为：保存八个样本 ID、token/label/mask、全局初始权重；在单副本参考中按全局有效 token mean 更新一次；并行运行用相同逻辑样本，重建各 shard 后比较 loss、参数梯度和新权重。拓扑变化时先用容差比较，随机性和低精度另作增量实验。这样“loss 差不多”才能落到具体目标与误差范围。

## 性能与失败怎样共同限制配置

MFU（Model FLOPs Utilization，模型浮点运算利用率）通常按“模型约定的每步 FLOPs ÷ 步耗时 ÷ 全部设备的对应精度峰值”计算。重计算通常不计入模型有效工作量，比较前必须固定 FLOPs 与峰值口径。

max-rank step time 的 p50/p95、tokens/s/GPU、明确口径的 MFU、allocated/reserved 峰值、网络与 PP bubble 描述不同成本。累计 CUDA/NCCL 时长与关键路径的暴露区间分开核算；少量稳态 trace 用于归因，不能替代无 profiler 的分布。

rank 异常、数据慢节点、checkpoint 写失败、恢复时 DP 改变、极端路由和 NaN/Inf 分别破坏不同前提。合理的失败边界包括及时传播第一因、清理资源、阻止坏 checkpoint 发布，以及明确的 fallback。作业退出本身不能证明恢复正确。

[数学、实现与性能的边界](<03_parallelism_validation.md>) · [并行方案原理](<../../../parallelism/并行方案的约束与取舍.md>)
