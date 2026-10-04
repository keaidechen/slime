# 六种性能现象背后的因果关系

从一个矩阵乘到一轮 RL，分析尺度会变，推理方法却相通：明确实际工作与时间边界，找出决定完成时间的依赖，再判断某种改变是否作用于这条依赖。下面沿六个层次展开，不将某次测试的数字当成普遍结果。

## 一、矩阵乘的 CPU 时间为何可能很短

`torch.mm(x,y)` 返回时，GPU 工作可能只是入队。CPU 的 `time.perf_counter()` 差值因此可能主要是提交开销。若在开始前排空旧工作、结束后等待目标完成，计时窗口才包括本次 GPU 执行，但也包括主机相关开销。CUDA Event 则在设备时间线上定义另一种边界。

```python
# 示意同一调用的两种时间边界；并非一组实测结果。
t0 = time.perf_counter()
torch.mm(x, y)
enqueue_time = time.perf_counter() - t0

torch.cuda.synchronize()
t0 = time.perf_counter()
torch.mm(x, y)
torch.cuda.synchronize()
completion_time = time.perf_counter() - t0
```

预热排除初始化、编译与缓存建立，重复统计区分稳定成本与波动。例如对 2048、4096、8192 的方阵，绝对耗时与有效 FLOP/s 回答不同问题：更大的矩阵耗时更长，同时可能更充分利用计算资源。30 次记录的 median/p95 是一种示例统计口径，不能保证所有工作负载都已稳态。

## 二、一行 Python 为什么对应多个名字

对 `(relu(x @ w).sum()).backward()`，Python 表达式、ATen 的 `matmul/mm`、ReLU、reduction、反向算子和 GPU kernel 分别属于不同层。`record_function` 提供业务语义，`record_shapes`、`profile_memory`、`with_stack` 提供形状、分配和调用来源；self CUDA time 与包含子调用的 total time 不能重复相加。


下面的代码把这个表达式放到一个明确的语义区间中。五轮预热用于隔离初始化；每轮清空 `x.grad/w.grad`，避免把梯度累积状态混进参考。CPU/CUDA activity、shape、memory 与 stack 各自增加一种观测维度，也各自增加采集开销。

```python
import torch
from torch.profiler import ProfilerActivity, profile, record_function

x = torch.randn(4096, 4096, device="cuda", requires_grad=True)
w = torch.randn(4096, 4096, device="cuda", requires_grad=True)

for _ in range(5):
    (torch.relu(x @ w).sum()).backward()
    x.grad = None
    w.grad = None

torch.cuda.synchronize()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("forward_backward"):
        (torch.relu(x @ w).sum()).backward()
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
prof.export_chrome_trace("/tmp/forward_backward_trace.json")
```

表格将 self CUDA time 排序，trace 则保留算子、launch 和设备执行的关系。`row_limit=20` 只是展示上限，不代表只有二十个事件。这个片段说明证据如何产生，本文没有据此写入 GPU 实测数据。

CPU、CUDA、Autograd、Autocast 的 dispatcher key 不必指向同一实现。内部接口 `torch._C._dispatch_dump_table("aten::mm")` 可以展示注册关系，但语法和输出依 PyTorch 版本变化。[执行与内存篇](02_pytorch.md)进一步解释这种映射。

同样，`memory_allocated()`、`memory_reserved()` 和 `max_memory_allocated()` 分别是当前张量占用、缓存分配器保留与窗口峰值。矩阵尺寸增大后，这三个数字不同并不自动意味着泄漏。Profiler 能找到哪个算子占时间，单凭算子表却不能解释 kernel 内部的 stall。

## 三、系统时间线与 Kernel 报告为什么要连起来

如果 GPU 在两个 kernel 之间长期空闲，根因可能在 CPU、数据供给或依赖；如果 kernel 连续执行，才进一步讨论计算、内存和指令限制。累计耗时最高的 kernel 可能调用很多次，单次最慢的 kernel 则可能不在关键路径，两者都需要上下文。

Nsight Systems 的 `cuda,nvtx,osrt` 轨道连接主机、设备与语义区间。Nsight Compute 的 kernel-name 过滤和 launch-count 限定分析对象，basic/full 等采集集合影响成本。完整训练的所有 kernel 都做 replay，会改变被分析系统并产生大量数据。

DRAM 与 SM/compute throughput、occupancy、warp stall、算术强度一起限定解释。低 occupancy 可能仍足以饱和带宽，不能单独判错。shape 或 dtype 改变后，系统总时长与对应资源信号是否按假设变化，决定该解释是否成立。详见[时间线与底层算子](03_cuda_kernels.md)。

## 四、训练的通信占比为何不能直接相加

一次 step 包括前向、反向、梯度与参数通信、PP send/recv、优化器、数据等待和气泡。它们可能重叠。timer 的 min/max 与多 rank trace 可以区分最慢参与者、阶段不均衡和到达偏斜，rank 0 的平均值没有这个能力。

固定 rollout dump 或训练数据、序列、global batch 和精度，才能解释配置差异。TP 增大可能让 GEMM 变小并增加高频通信；PP 改变会引入不同气泡；CP 和 EP 的开销还分别受到上下文与路由分布影响。固定总卡数时 TP/PP 改动可能连带改变 DP，不能把全部变化归给一个局部参数。

microbatch、activation recomputation 和 overlap 分别改变调度粒度、计算与保存的交换、通信暴露。稳态 iteration 的 median/p95、samples/s、tokens/s、框架 TFLOPS 口径、每 rank 峰值、loss 与 grad norm 一起限定判断。短 trace 负责解释依赖，无 profiler 的分布负责衡量性能。见[训练性能](04_megatron.md)。

## 五、吞吐趋平时，延迟为什么仍会上升

考虑一个明确的教学 workload：输入长度 512 或 1024，输出目标 128 或 256，200 个请求，固定长度范围、EOS 策略、sampling 与 prefix cache。并发从 1、2、4、8、16、32 到 64 是不同负载点，不是系统必然支持的容量。

并发增加可能提高 decode batch 和设备效率；接近资源上限后，增加的请求主要转成排队，于是 output tokens/s 趋平而 p95 TTFT/TPOT 上升。request/s、input/output tokens/s、p50/p95/p99 TTFT、TPOT、ITL、E2E、错误率、KV 和利用率共同描述这条曲线。

拐点两侧可能变化的机制包括 prefill 对 decode 的阻塞、scheduler/GPU gap、Graph 命中、KV retract/eviction、collective 和专家不均衡。SGLang 的 profile 接口与 vLLM 的 profiler 配置只是观测入口，具体参数依版本。生产容量还需要质量、延迟 SLO 和余量，错误或超时请求不能算有效产出。见[推理容量](05_inference.md)。

## 六、生成与训练各自很快，RL 为什么仍然慢

RL 还有奖励、工具、环境、过滤、优势计算、权重转换、传输、加载和角色等待。`--debug-rollout-only` 的数据保存与 `--debug-train-only` 的回放可以定义角色独立路径，但全链路的资源争用和同步边界仍然存在。

一种教学假设是：`wait_time_ratio` 中位数为 0.42，最长样本的非生成阶段占 rollout 的 35%，对应窗口 GPU 空闲。这支持远程 reward/tool 长尾的候选解释，尚不能证明 decode kernel 有问题；这些数字不是本仓库实测。sample 时间线和角色依赖决定下一轮训练到底在等谁。

增加 rollout GPU、并发、reward concurrency、microbatch 或改变并行配置，作用于不同限制。max response length、update frequency 还可能改变任务语义、样本年龄和质量，因此不能只比较吞吐。有效 tokens/s、samples/hour、step p50/p95、wait ratio、GPU-hours、reward、KL、长度和错误率需要一起解释。

生成正常文本、算法预期的 actor/reference 概率与 KL、loss/grad norm、第二轮换权后的状态、truncated 和重试情况，定义了正确闭环。若两侧独立性能正常而全链路慢，候选原因应转向 orchestration、同步、内存切换和共置争用。见 [RL 性能](06_slime.md)。

## 六种现象共享的推理边界

原始日志、配置和 trace 使论证可复核；单变量对照或明确说明的组合变化使因果关系可讨论。没有实际运行证据时，文章能给出机制、模型和预测，不能凭教学代码补出加速比。最终评价落在固定资源、正确性和质量下的有效工作。

[性能系列](README.md) · [证据与假设](证据与因果.md)

<!-- main-route -->

性能结论依赖状态与时序，故障也会破坏这些条件。最后一篇区分重试、恢复与一致性边界。

[上一篇：从观测到因果结论](<证据与因果.md>) · [主线 25/26](<../guides/系统阅读路线.md>) · [下一篇：故障模型与恢复](<故障模型与恢复.md>)

<!-- /main-route -->
