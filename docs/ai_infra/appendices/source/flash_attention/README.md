# FlashAttention：从加权求和到 GPU 流水

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../README.md>)。

只会写循环和操作数组，也可以从这里开始。我们要回答一个具体问题：给每个查询计算一组权重并汇总信息，为什么会占用大量显存，又怎样在不改变注意力定义的前提下减少读写？FlashAttention（下文简称 FA）是一组围绕这个问题设计的算法与 GPU 实现。

先读 01 的 §1–7，亲手算出一个输出；再读 00，认识执行这些计算的线程和存储；随后按 02 → 03 阅读前向代码。04 将同一算法用于求梯度和生成文本，05 解释硬件变化为何要求重新安排流水，06 则区分公式验证与 GPU 实测。每篇后半部的源码精读可以留到第二遍。

- [00 · 读 FA2 启动代码：线程、存储与模板约定](<00_cuda_basics.md>)
- [01 · 从普通 Attention 到在线 Softmax](<01_algorithm_and_online_softmax.md>)
- [02 · FA2 前向：把一次调用逐段展开](<02_fa2_forward.md>)
- [03 · 读懂 CuTe 布局、线程分工与流水](<03_layout_and_pipeline.md>)
- [04 · 反向、变长与 KV cache](<04_backward_varlen_kvcache.md>)
- [05 · Hopper 与 CuTe DSL：FA3/FA4](<05_hopper_and_cute.md>)
- [数学等价、数值误差与执行成本](<06_numerics_and_performance.md>)

阅读中会交替使用两种尺度的例子。数学例子取一个 query、四个 key，score 为 `[0, ln(2), ln(4), ln(8)]`，V 为 `[1,2,3,4]`，最终输出是 `49/15`；它用于检查分块与合并是否算对。执行例子取 `[batch,token,head,feature]=[2,256,4,64]`，用于推导真实线程和地址。手算的 `d=1` 不代表 GPU kernel 支持直接以该配置运行；真实配置也不应缩成几条线程后冒充原始硬件指令。

全系列使用从 0 开始的下标，区间 `[a,b)` 包含 a、不包含 b。`N_q/N_k` 是 query/key 的个数，`d/d_v` 是每行的特征数，`B/H` 是 batch/head 数；这些是元素数量，不是字节。数学值通常无物理单位，存储量才使用字节、KiB（1024 字节）和 MiB（1024 KiB）。张量的轴顺序在每个接口处单独标明，不能仅凭维度数相同就相乘。

源码摘录对应本仓库 `flash-attention` 子模块的固定提交 [`e9cf2c1`](https://github.com/Dao-AILab/flash-attention/tree/e9cf2c1651d2303191eb40a739a3c135fda00999)。FA2 的具体 tile、FA3 的条件分支和这里称为 FA4 的 CuTe DSL 路径均以这个快照为准。手算、简化时序和假设资源预算会明确标出；它们解释机制，不代替对应 GPU 的性能测量。

可在仓库根目录运行下面的纯 CPU 例子，不需要安装 CUDA。它输出在线状态并检查前向、split 合并和梯度；结果的适用范围见 06。

```bash
python3 docs/ai_infra/appendices/source/flash_attention/examples/online_attention.py
```
