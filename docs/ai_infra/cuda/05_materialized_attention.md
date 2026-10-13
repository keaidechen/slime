# 5. 先实现一份普通 Attention

[上一章](<04_row_softmax.md>) · [目录](<README.md>) · [下一章](<06_online_attention.md>)

这一章将已有的矩阵乘和 Softmax 组合起来。完整实现仍在 [attention_lab.cu](<examples/attention_lab.cu>)，顺序为 `qk_scores → row_softmax → pv_output`。先固定一个 batch、一个 head，使用连续行优先 FP32 数据，无 dropout、无反向传播。

## 5.1 不跳过形状与下标

| 张量 | 形状 | 一个元素的含义 |
|---|---|---|
| Q | `[Nq,d]` | query i 的特征 c |
| K | `[Nk,d]` | key j 的特征 c |
| V | `[Nk,dv]` | key j 对应的 value 通道 c |
| S、P | `[Nq,Nk]` | query i 对 key j 的分数、概率 |
| O | `[Nq,dv]` | query i 汇总后的通道 c |

首先算 `S[i,j]=sum_c(Q[i,c]*K[j,c])/sqrt(d)`，对每行 S 做 Softmax 得到 P，然后 `O[i,c]=sum_j(P[i,j]*V[j,c])`。公式中的 K 转置是为了让每个 Q 行与每个 K 行做点积；代码使用 `k[j*d+c]`，不需要先复制转置矩阵。

用最小例子 `Q=[[1]]`、`K=[[0],[ln(3)]]`、`V=[[10],[20]]`：

```text
S = [0, ln(3)]
P = [1/4, 3/4]
O = [1/4*10 + 3/4*20] = [17.5]
```

Q 的行数决定输出多少行，V 的列数决定输出多少列。理解这一点后，再看 `Nq=3,Nk=7,d=5,dv=3` 的非方阵算例，避免所有维度碰巧相同而掩盖索引错误。

## 5.2 三个 kernel 分别拥有哪份输出

`qk_scores` 的线程线性下标 `index` 拆为：

```cpp
int row = index / nk;
int key = index % nk;
for (int c = 0; c < d; ++c)
    value += q[row * d + c] * k[key * d + c];
```

这个线程拥有一个 S 元素，沿 d 做点积。第 2 章的 tiled GEMM 可以帮助理解后续复用；这里先保留最直接的标量实现，让张量之间的连接清晰。

`row_softmax` 的一个 block 拥有 S 的一行，最终写 P 的一行。`pv_output` 的一个线程拥有一个 O 元素，沿 Nk 累加。不同 kernel 在同一默认 stream 中依次启动，因此后者读取前者结果时，前者已经在设备执行顺序上完成；每两次 launch 之间不需要加一次 host 全设备同步。

```text
Q、K ── qk_scores ──> S（global）── row_softmax ──> P（global）
                                                    │
V ───────────────────────────────────────────────> pv_output ──> O（global）
```

这是数据依赖图；V 是 `pv_output` 的另一个输入，并不参与生成 P。

## 5.3 mask 是分数计算的一部分

为了统一教学接口，`limits[i]` 表示第 i 行可见的 key 前缀长度：`0 <= j < limits[i]` 可以参与，其余 score 写为负无穷。

| 配置 | limits 的取法 |
|---|---|
| 全部可见 | 每行都是 Nk |
| 等长 causal | Nq=Nk，第 i 行为 i+1 |
| 全 mask 行 | 对应值为 0，约定 O 为零 |

例：长度 3 的 causal 对应 `[1,2,3]`，第 0 行只看 key 0。不能先对整行完成 Softmax，再把未来位置的概率直接清零：那样剩余概率的和一般不为 1。应先 mask，再确定 m/l。

这里的 prefix 接口不表达任意稀疏 mask。**长度不等的 causal 对齐也不在本例约定内**；真实 FlashAttention 的右下对齐由现有 [变长与 KV cache 导读](<../appendices/source/flash_attention/04_backward_varlen_kvcache.md>)解释。

## 5.4 三个 kernel 为什么还不是 FlashAttention

S 和 P 各有 `Nq*Nk` 个 FP32 数，各占 `4*Nq*Nk` 字节。取 Nq=Nk=4096，各为 64 MiB，两张共 128 MiB，尚未计 Q/K/V/O。这个数字与 FP16 的 32 MiB/张不同，讨论容量时必须标明 dtype。

除了存储容量，还存在 S 写出和读入、P 写出和读入。我们希望把当前小块 S 用完即丢，而不是存整张矩阵；障碍在于 Softmax 的分母依赖整行。下一章解决的正是“没有同时拿到整行，怎样仍算出相同的归一化结果”。IO 出发点可参考 [FlashAttention 论文](https://arxiv.org/abs/2205.14135)。

## 5.5 输出的不变量与参考实现

[CPU 参考](<examples/attention_cpu.py>)给出这些关系的数值对照：

```bash
python3 docs/ai_infra/cuda/examples/attention_cpu.py
```

有 GPU 时使用上一章的 `attention_lab.cu` 编译命令。CUDA 示例中的 CPU 参考用 double 累加每一行分数并一次性归一化，分别核对普通与在线两条 GPU 路径，避免只比较两份可能犯同一个错误的 kernel。

| 条件 | 输出关系 |
|---|---|
| 前文两个 key 的算例 | 输出 17.5 |
| 将 Q 行重复一次 | 对应 O 行也重复 |
| 改变 V，保持 Q/K 不变 | P 不变，O 改变 |
| 给所有 V 元素加常数 c | 每个非空输出通道加 c，全 mask 行仍按约定为零 |
| 读取 `k[key*d+c]` 而非 `k[c*Nk+key]` | K 实际按 `[Nk,d]` 行优先保存，数学转置不等于更改实际 stride |

这些关系分别约束形状、归一化与地址计算。下一章保持相同数学语义，改变中间状态的存储方式。
