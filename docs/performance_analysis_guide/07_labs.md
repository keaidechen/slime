# 07 循序渐进实验课

这六个实验按依赖关系排列。前 3 个单卡即可完成；后 3 个需要能够运行对应框架。每个实验都要求保存原始结果，不以截图代替数据。

建议为每次练习建立独立目录：

```bash
mkdir -p /tmp/perf_labs/{lab1,lab2,lab3,lab4,lab5,lab6}
```

## 实验 1：学会正确计时和做可信基线

### 目标

亲眼看到 CUDA 异步执行如何让普通 Python 计时失真，并完成第一次单变量 A/B。

### 步骤 1：确认环境

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available(), "本实验需要 CUDA GPU"
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
PY
```

### 步骤 2：运行错误计时和正确计时

```bash
python - <<'PY'
import time
import statistics
import torch

device = "cuda"
x = torch.randn(4096, 4096, device=device)
y = torch.randn(4096, 4096, device=device)

for _ in range(10):
    torch.mm(x, y)
torch.cuda.synchronize()

wrong = []
correct = []
for _ in range(30):
    t0 = time.perf_counter()
    torch.mm(x, y)
    wrong.append((time.perf_counter() - t0) * 1000)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.mm(x, y)
    torch.cuda.synchronize()
    correct.append((time.perf_counter() - t0) * 1000)

print("unsynchronized median ms:", statistics.median(wrong))
print("synchronized median ms:", statistics.median(correct))
PY
```

预期：未同步时间通常只测到 launch，而不是 GPU 完成时间。数值关系因机器而异，重点是理解测量边界。

### 步骤 3：用 CUDA Event 复测

按 [PyTorch 章节](./02_pytorch.md) 的 CUDA Event 示例重测，并记录 30 次的中位数和 p95。把矩阵改为 2048 和 8192，各跑一次。

### 交付物

- `/tmp/perf_labs/lab1/environment.txt`：环境信息；
- 三种矩阵尺寸的 median/p95 表格；
- 一句话说明普通计时为什么错；
- 一个结论：矩阵尺寸变化后，是 latency 还是有效 FLOP/s 的变化更重要。

### 完成标准

你能解释 warmup、CUDA synchronize、重复测量和单变量原则，且不会用一次 `time.time()` 判断 GPU 算子快慢。

## 实验 2：从 PyTorch 调用追到 ATen 算子和 CUDA kernel

### 目标

建立 `Python API -> ATen operator -> CUDA kernel` 的映射，并学习读 trace。

### 步骤 1：创建最小工作负载

将下面代码保存为你自己的临时脚本，或直接在项目实验分支中创建脚本：

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
    with record_function("lab2_forward_backward"):
        (torch.relu(x @ w).sum()).backward()
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
prof.export_chrome_trace("/tmp/perf_labs/lab2/trace.json")
```

### 步骤 2：先读表，再读时间线

在表中找：

1. `aten::matmul`/`aten::mm` 一类算子；
2. ReLU 和 reduction；
3. backward 对应算子；
4. self CUDA time 与 total CUDA time 的区别。

把 trace 上传到 Perfetto，搜索 `lab2_forward_backward`。从 CPU operator 展开到 CUDA launch 和 kernel，记录至少一个 kernel 名称。

### 步骤 3：查看 dispatcher 注册

```bash
python - <<'PY'
import torch

print(torch._C._dispatch_dump_table("aten::mm"))
PY
```

这是内部调试接口，可能随 PyTorch 版本变化。你要回答：CPU、CUDA、Autograd、Autocast 等 key 是否都指向同一实现？

### 步骤 4：制造一个显存峰值

把矩阵尺寸逐步增大，只使用可安全运行的大小。记录：

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
```

解释 allocated 与 reserved 不同的原因。不要为了实验故意把共享机器打到 OOM。

### 交付物

- `trace.json`；
- top 20 operator 表；
- 一条完整映射：Python 行、ATen op、CUDA kernel；
- allocated/reserved/peak 表。

### 完成标准

你能指出 trace 中 CPU 发射、GPU 执行和同步分别在哪里，并知道 PyTorch Profiler 适合“哪个算子”，不直接回答“kernel 内部为什么慢”。

## 实验 3：Nsight Systems 定位，再用 Nsight Compute 下钻

### 目标

学会先系统、后 kernel 的两阶段流程。

### 步骤 1：抓一个极短 Systems trace

对实验 2 的脚本运行：

```bash
nsys profile \
  -t cuda,nvtx,osrt \
  -s none \
  -o /tmp/perf_labs/lab3/matmul_systems \
  python /path/to/lab2.py
```

用 GUI 或 `nsys stats` 查看：

```bash
nsys stats /tmp/perf_labs/lab3/matmul_systems.nsys-rep
```

先回答：

- GPU 上有哪些空洞？
- memcpy 和 kernel 是否重叠？
- CPU 是否在每个 kernel 前长时间停顿？
- 哪一个 kernel 总时间最高，哪一个单次时间最高？

### 步骤 2：只 profile 一个 kernel

先列出 kernel 名，再根据本机名称选择过滤表达式：

```bash
ncu --list-sets

ncu \
  --set basic \
  --kernel-name regex:YOUR_KERNEL_REGEX \
  --launch-count 1 \
  -o /tmp/perf_labs/lab3/matmul_compute \
  python /path/to/lab2.py
```

若没有匹配到，先去掉 `--kernel-name`，把工作负载缩短到只执行一次，再从输出中复制真实名称。不要对长训练任务使用 `--set full`。

### 步骤 3：做 Roofline 判断

使用 Nsight Compute GUI 的 Speed of Light 和 Roofline section，记录：

- DRAM throughput 相对峰值；
- SM/compute throughput 相对峰值；
- occupancy；
- warp stall 主要类别；
- arithmetic intensity 的大致位置。

不能只根据 occupancy 低就宣判 kernel 有问题。结合是否 memory-bound、compute-bound、launch-bound 或 latency-bound 写出假设。

### 步骤 4：验证一个假设

只改变矩阵尺寸或 dtype 中的一个变量，再比较 nsys 总时长和关键 ncu 指标。结果可能否定假设，这同样是有效实验。

### 交付物

- `.nsys-rep` 和 `.ncu-rep`；
- 一张系统时间线截图；
- 一张关键 kernel 指标表；
- “事实—假设—A/B 结果—结论”四行记录。

### 完成标准

你不会一看到整体慢就直接对所有 kernel 跑 ncu，并能说明 nsys 和 ncu 的职责边界。

## 实验 4：Megatron 训练 step 分解

### 目标

把分布式训练 step 分成计算、通信、pipeline bubble、optimizer 和数据等待，并找出最慢 rank。

### 前提

选择当前集群已经能跑通的最小 Megatron/Slime `--debug-train-only` 配置。不要为了学习 profiler 从零启动生产规模模型。

### 步骤 1：保存固定输入和基线

优先复用 Slime 的 rollout dump；若运行纯 Megatron，固定训练数据、sequence length 和 batch。关闭 profiler，运行至少 10 个稳态 iteration，记录：

- iteration time 的 median/p95；
- samples/s、tokens/s、TFLOPS（如框架提供）；
- 每个 rank 的显存峰值；
- loss、grad norm；
- TP/PP/CP/EP/DP 和 micro/global batch。

### 步骤 2：开启框架 timer

使用当前 Megatron 版本支持的 timer/log 参数，至少分出 forward、backward、optimizer、all-reduce/reduce-scatter/all-gather、pipeline send/recv。参数名变化较快，先运行启动脚本的 `--help`，并参考 [Megatron 章节](./04_megatron.md)。

### 步骤 3：抓两个 iteration

用 Slime training profiler 或 Megatron Bridge 的 profiler 配置抓 1～2 个稳定 iteration。打开所有 rank trace，回答：

1. 最慢 rank 是谁？
2. PP stage 是否存在明显首尾 bubble？
3. collective 是否被计算覆盖？
4. TP/CP 是否产生高频小通信？
5. MoE 时，各 EP rank 的工作量是否接近？
6. GPU 空洞前，CPU、数据或其他 rank 在做什么？

### 步骤 4：只做一个并行配置 A/B

可选择一项：

- micro-batch size；
- TP 大小；
- PP 大小；
- activation recomputation；
- communication overlap 开关。

总 GPU、global batch、sequence length、模型和数据必须相同。若改 TP 会连带改变 DP，应在报告中明确，并计算总有效 token/s，不只看单 rank 时间。

### 交付物与完成标准

交付基线表、rank 对比、trace 和 A/B 结论。完成标准是能用时间线证明“慢在什么等待关系”，而不是只说 NCCL 百分比高。

## 实验 5：SGLang 或 vLLM 容量曲线

### 目标

找到吞吐—延迟曲线的容量拐点，并对一个并发点抓 profile。

### 步骤 1：定义 workload

选择本地可用的小模型，固定：

- input length 512 或 1024；
- output length 128 或 256；
- 固定随机长度范围；
- ignore EOS 策略；
- 200 个请求；
- 相同 sampling 和 prefix cache 设置。

### 步骤 2：预热并 sweep

使用 [推理章节](./05_inference.md) 的命令，依次测最大并发：

```text
1, 2, 4, 8, 16, 32, 64
```

资源不足时提前停止。每个点记录 request/s、input/output tokens/s、p50/p95/p99 TTFT、TPOT、ITL、E2E、错误率、KV Cache 和 GPU 利用率。

### 步骤 3：找容量拐点

画两张图：

1. x=concurrency，y=output tokens/s；
2. x=concurrency，y=p95 TTFT 和 TPOT。

吞吐趋平、排队与延迟开始快速上升的位置是候选拐点。错误率非零的点不算有效容量。

### 步骤 4：对拐点前后各抓一次短 profile

SGLang 使用 `/start_profile` 或 `--profile`；vLLM 使用当前 `--profiler-config` 和 benchmark `--profile`。每次只抓几个请求。比较：

- decode batch 是否变大；
- scheduler 和 GPU gap；
- prefill 是否阻塞 decode；
- CUDA Graph 命中；
- KV Cache/retract/eviction；
- collective 或 expert 不均衡。

### 交付物与完成标准

交付原始 benchmark JSON/JSONL、两张曲线、两个短 trace 和建议生产并发。建议必须说明质量/延迟 SLO 和预留余量，不能直接采用实验中的最高吞吐点。

## 实验 6：Slime 端到端瓶颈闭环

### 目标

把前五个实验串成一次真实的 RL 系统分析。

### 步骤 1：先跑正确性检查

使用小规模配置确认：

- rollout 是正常文本；
- 第一步 actor 与 reference 的 log-prob/KL 符合当前算法预期；
- loss、reward、grad norm 正常；
- 第二轮权重更新后仍可正常生成；
- 没有错误、重试风暴或隐性 truncated 激增。

### 步骤 2：收集无 profiler 基线

跑至少 10～20 个稳态 rollout，记录第 6 章列出的 `perf/*`、有效 token 数、reward 和 GPU 曲线。填写 [性能报告模板](./performance_report_template.md) 的前半部分。

### 步骤 3：分离两侧

1. `--debug-rollout-only` 保存至少一轮数据；
2. `--debug-train-only` 回放同一数据；
3. 分别获得 rollout 和训练吞吐；
4. 把独立耗时与全链路耗时比较，找额外 orchestration/sync 成本。

### 步骤 4：只对疑似瓶颈 profile

- rollout 慢：SGLang profile + sample timeline；
- train 慢：Slime training profile；
- OOM：memory snapshot；
- 两侧都不慢但端到端慢：短 nsys 全链路；
- update 慢：分阶段记录权重转换、传输和 load。

### 步骤 5：提出一个可证伪假设

合格示例：

> 测量事实：稳态 `wait_time_ratio` 中位数为 0.42，最长样本的非生成时间占 rollout 的 35%，rollout GPU 在该窗口空闲。假设：远程 reward/tool 长尾控制了下一轮训练启动，而不是 decode kernel。

不合格示例：

> GPU 利用率不高，所以 SGLang 很慢。

### 步骤 6：做单变量 A/B

根据假设只改变一项，例如 rollout GPU 分配、并发、max response length、reward concurrency、micro-batch、并行配置或 update frequency。比较：

- 端到端有效 tokens/s 或 samples/hour；
- p50/p95 step time；
- `wait_time_ratio`；
- GPU-hours/固定时间内有效产出；
- reward、KL、长度和错误率。

### 最终完成标准

报告必须让另一个不了解本次任务的人能够：

1. 用完整命令复现 workload；
2. 找到原始日志和 trace；
3. 区分事实、假设和验证；
4. 重跑 A/B；
5. 确认性能提升没有改变正确性或训练目标。

完成这一步后，你已经建立了从单算子到 RL 系统的完整性能分析闭环。遇到新问题时，重复这套方法，而不是更换一套“玄学参数”。
