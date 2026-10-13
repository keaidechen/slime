# 01 建立环境清单与可信基线

<details>
<summary>文章导航</summary>

- [1. 原始证据也有生命周期](<#1-原始证据也有生命周期>)
- [2. 记录软件与硬件环境](<#2-记录软件与硬件环境>)
- [3. 保存完整启动命令和 workload](<#3-保存完整启动命令和-workload>)
- [4. 先用 nvidia-smi 看“生命体征”](<#4-先用-nvidia-smi-看生命体征>)
- [5. 同时观察 CPU、内存、磁盘和网络](<#5-同时观察-cpu内存磁盘和网络>)
- [6. 正确做 warmup 和测量](<#6-正确做-warmup-和测量>)
- [7. 基线结果必须包含正确性](<#7-基线结果必须包含正确性>)
- [8. 配置变化如何连接到假设](<#8-配置变化如何连接到假设>)
- [9. 什么时候进入下一层工具](<#9-什么时候进入下一层工具>)
- [结论的适用边界](<#结论的适用边界>)
- [理解 GPU 遥测而不误读利用率](<#理解-gpu-遥测而不误读利用率>)
- [参考资料](<#参考资料>)

</details>

相关基础：[必要基础](<../foundations/README.md>)。


环境、启动条件、workload、端到端指标和系统资源曲线共同定义基线。它们解释为什么相同代码在不同窗口会表现不同，也是后续归因的前提。


## 1. 原始证据也有生命周期

环境、命令、指标和 trace 只有对应同一个运行身份，才能组成基线。时间戳或 run ID 用来避免不同运行互相覆盖；证据目录需要足够的容量和 inode，因为 trace 可能达到 GB 级。`df -h` 与 `df -i` 分别反映这两种约束。

后面的命令片段用 `${ARTIFACT_DIR}` 表示这一位置。`/tmp` 只是临时示例；长期证据需要持久化存储，不能把进程成功退出等同于数据永久保存。


## 2. 记录软件与硬件环境

把下面命令的输出保存到实验目录。`tee` 既显示也保存：

```bash
git rev-parse HEAD | tee "${ARTIFACT_DIR}/git_commit.txt"
git status --short --branch | tee "${ARTIFACT_DIR}/git_status.txt"

python - <<'PY' | tee "${ARTIFACT_DIR}/python_env.txt"
import os
import platform
import torch

print("platform:", platform.platform())
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("cuda available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, p.name, p.total_memory, p.major, p.minor)
PY

nvidia-smi -q | tee "${ARTIFACT_DIR}/nvidia_smi_q.txt"
nvidia-smi topo -m | tee "${ARTIFACT_DIR}/gpu_topology.txt"
lscpu | tee "${ARTIFACT_DIR}/lscpu.txt"
numactl -H 2>&1 | tee "${ARTIFACT_DIR}/numa.txt"
```

straggler（慢参与者）是明显晚于其他参与者到达同步点或完成工作的进程／设备。同步操作受最晚到达者影响，原因可能是数据、计算、资源配额或网络。

多节点任务每个节点都要记录。不要假设同一集群的节点完全同构；驱动、GPU clock、NIC mapping 或故障状态差异都可能制造 straggler。

如果使用容器，还要记录镜像 digest，而不只是可变 tag。


## 3. 保存完整启动命令和 workload

至少保存：

- 完整命令与所有参数；
- 环境变量；
- 模型、checkpoint、数据集及其版本；
- seed；
- batch/micro-batch/sequence length；
- 并行配置和 rank placement；
- 推理的输入/输出长度分布、并发或 RPS；
- 是否启用 cache、CUDA Graph、compile、quantization。

环境变量中可能有 token、密码、代理凭据。保存前只筛选与性能有关的变量：

```bash
env | sort | rg '^(CUDA|NCCL|TORCH|PYTORCH|OMP|MKL|RAY|SGLANG|VLLM)_' \
  | tee "${ARTIFACT_DIR}/performance_env.txt"
```

不要把完整 `env` 直接上传到 issue 或 PR。


## 4. 先用 nvidia-smi 看“生命体征”

### 4.1 一次性查询

```bash
nvidia-smi --query-gpu=timestamp,index,name,uuid,pci.bus_id,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv
```

关注：

- 某张卡利用率明显低于其他卡；
- 功耗和 SM clock 被限制；
- 温度过高导致降频；
- 显存分布不符合 PP/TP/EP 预期；
- GPU UUID 与 rank placement 不一致。

### 4.2 连续采样

```bash
nvidia-smi dmon -s pucvmet -d 1 \
  | tee "${ARTIFACT_DIR}/nvidia_dmon.txt"
```

不同驱动支持的 `dmon` metric group 可能不同，先运行 `nvidia-smi dmon --help`。官方文档说明了 `p/u/c/v/m/e/t` 等组的含义：[nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。

`nvidia-smi` 的一秒采样会掩盖毫秒级空洞，所以它只能判断“哪张卡或哪个阶段可疑”，不能替代时间线 profiler。

### 4.3 DCGM：长任务与集群监控

如果机器安装 DCGM，可以低开销持续观察 SM、Tensor Core、DRAM、PCIe/NVLink 等指标。先查询本机支持字段：

```bash
dcgmi profile --list --entity-id gpu:0
```

示例字段必须按查询结果选择；官方示例用 1002（SM activity）和 1005（DRAM activity）：

```bash
dcgmi dmon --entity-id gpu:0 --field-id 1002,1005 --delay 1000
```

DCGM 的 profiling counter 可能与 Nsight 冲突。使用 Nsight 前暂停、完成后恢复：

```bash
dcgmi profile --pause
# 运行 nsys 或 ncu
dcgmi profile --resume
```

参见 [DCGM Profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)。


## 5. 同时观察 CPU、内存、磁盘和网络

GPU 空闲不一定是 GPU 问题。

### 5.1 进程级 CPU 和 I/O

```bash
PID=<目标进程PID>
pidstat -dur -p "${PID}" 1
top -H -p "${PID}"
```

解释：

- 单线程 100% 且 GPU 有空洞：Python、tokenizer、scheduler 或数据预处理可能串行；
- 大量 context switch：线程/进程过多或锁竞争；
- `iodelay`/读写持续高：数据或 checkpoint 可能在关键路径上。

### 5.2 系统级内存和磁盘

```bash
vmstat 1
iostat -xz 1
```

关注 swap、page fault、磁盘 `%util`、await 和队列。不要只看磁盘带宽；很多小文件会受 IOPS/metadata 限制。

### 5.3 网络与拓扑

```bash
nvidia-smi topo -m
ip -s link
```

多节点还应使用集群提供的 InfiniBand/RoCE 工具。通信慢时，先用 `nccl-tests` 建立硬件/网络基线，再分析框架。官方示例：

```bash
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

多节点构建和 `busbw` 解释见 [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)。


## 6. 正确做 warmup 和测量

第一次运行常包含：

autotune（自动调优）实际试运行多个候选分块或执行配置，再为指定形状选择较快的一种。搜索本身有成本，其结果受形状、硬件和测量条件影响。

- CUDA context 初始化；
- 权重 page-in；
- JIT/Triton/Inductor 编译；
- cuBLAS/cuDNN autotune；
- CUDA Graph capture；
- PyTorch allocator 扩容；
- 文件系统 cache 冷启动。

除非目标就是冷启动，否则把这些步骤与稳态分开报告。

推荐：

```text
初始化
-> warmup 5~20 次（直到 step time 稳定）
-> 测量 30~100 次
-> 报告 p50/p95/min/max
```

长训练不必人为跑 100 步，但应剔除启动阶段，并覆盖正常的数据长度变化。


## 7. 基线结果必须包含正确性

性能优化可能让结果变错。每个基线保存：

- loss/grad norm 或输出 logits 摘要；
- 生成 token 和 EOS 行为；
- 有效样本/token 数；
- 错误、超时、丢请求数；
- 若改精度，保存误差阈值和比较方式。

只有“更快且正确”才是有效优化。


## 8. 配置变化如何连接到假设

MBS（Micro Batch Size）是每个数据并行参与者一次前反向处理的样本数；GBS（Global Batch Size）是一次逻辑更新的全局样本数。固定均匀批次时 GBS=MBS×DP 大小×累积次数。

假设 `TP=8、PP=1、MBS=1` 时，时间线上出现暴露的 NCCL 区间和小 GEMM。候选解释是 TP 颗粒度过细。改成 `TP=4、PP=2` 保持八卡预算，却同时改变张量与流水并行；这可以比较两套完整方案，不能独立证明 TP 的单项因果效应。

GBS、token budget、模型、数据与精度固定后，loss/grad norm 用于界定正确性，稳态 tokens/s/GPU 和 p95 step 用于界定性能。预热 10 步、测量 50 步、检查前 20 步数值、要求吞吐至少提高 5% 且尾部不退化，只是一组示例判据，适用窗口应由实际波动与初始化行为决定。若条件不成立，原并行布局仍是明确的回退方案。


## 9. 什么时候进入下一层工具

完成本章后，先根据基线分流：

| 观察 | 下一步 |
|---|---|
| GPU 低、CPU 高 | 第 2 章 PyTorch/Python；必要时 py-spy |
| GPU 高但吞吐低 | 第 2 章找 op，再到第 3 章找 kernel 原因 |
| 卡间差异大 | 第 4 章 rank timer、NCCL、拓扑 |
| OOM/显存上涨 | 第 2 章 Memory Snapshot |
| 推理随并发崩坏 | 第 5 章容量 sweep、scheduler/KV 指标 |
| Slime 训练在等待 | 第 6 章先分析 rollout |


## 结论的适用边界

环境身份、命令与性能相关环境变量限定执行；workload 和预热窗口限定工作与阶段；端到端分布、低开销 GPU/CPU/内存/I/O 曲线和正确性结果共同限定比较。缺少其中一层时，下一步是一个有待证据支持的假设，不能直接宣布根因。

## 理解 GPU 遥测而不误读利用率

#### 1. 三种常被混淆的“利用率”

MFU（Model FLOPs Utilization，模型浮点运算利用率）通常按“模型约定的每步 FLOPs ÷ 步耗时 ÷ 全部设备的对应精度峰值”计算。重计算通常不计入模型有效工作量，比较前必须固定 FLOPs 与峰值口径。

| 名称 | 大致含义 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| GPU utilization | 采样窗口内是否有 kernel 在执行 | GPU 是否经常完全空闲 | kernel 是否高效、Tensor Core 是否吃满 |
| SM activity/throughput | Streaming Multiprocessor（SM，流式多处理器）的活跃/吞吐 | 执行单元忙碌程度 | 单独不能区分有用工作与低效指令 |
| MFU | 有效模型 FLOPs / 理论峰值 FLOPs | 端到端模型计算效率 | 依赖 FLOPs 口径，跨报告未必可比 |

因此出现 `GPU-Util=100%`、Model FLOPs Utilization（MFU，模型浮点运算利用率）仅 25% 并不矛盾：GPU 一直在执行，但可能在运行小 kernel、访存等待、低效 layout conversion 或通信 kernel。


#### 3. DCGM：集群级 GPU 遥测

DCGM（Data Center GPU Manager，数据中心 GPU 管理器）提供 health、diagnostics、statistics 与 profiling metrics。`dcgm-exporter` 把指标暴露给 Prometheus。Exporter 从 hostengine 已 watch 的缓存样本读取，所以 Prometheus scrape interval 不是硬件 counter 的真实采样周期。

推荐四组指标：

| 组 | 例子 | 用途 |
|---|---|---|
| 资源 | GPU/显存利用率、framebuffer used | 容量与空闲检测 |
| 计算/内存 | SM active、Tensor active、DRAM active、PCIe/NVLink throughput | 判断工作类型和链路压力 |
| 热与功耗 | clocks、power、temperature、throttle reason | 发现降频和散热问题 |
| 可靠性 | Xid、ECC、NVLink error、row remap | 发现硬件/driver 异常 |

ECC = Error-Correcting Code（纠错码）。Xid 是 NVIDIA driver 报告的 GPU 错误事件编号；它是定位入口，不应脱离具体编号和上下文笼统解释。


#### 4. 从“低功耗”推断时要谨慎

可能路径：

```mermaid
flowchart TD
    A["功耗低"] --> B{"GPU timeline 是否空闲？"}
    B -->|是| C["查 CPU / data / sync / communication"]
    B -->|否| D{"Memory throughput 高吗？"}
    D -->|高| E["可能 memory-bound"]
    D -->|低| F["查小 kernel、依赖、低 occupancy 或降频"]
```

Memory-bound kernel 可能占满时间线但功耗低于大型 Tensor Core GEMM。反过来，高功耗也不证明模型有效吞吐高。


#### 5. 时钟与降频

记录 SM clock、memory clock、power limit、temperature 与 throttle reason。性能突然出现平台期时：

- power cap：功耗达到上限，时钟被限制；
- thermal throttle：温度导致降频；
- application clock/locked clock 配置不一致；
- 不同节点 GPU firmware 或 power policy 不一致。

锁定时钟适合可控 benchmark，但会改变能效和生产策略，必须记录并在实验后恢复。新 GPU 上具体命令与支持状态应以对应 `nvidia-smi` 手册为准。


#### 6. 采样粒度与 aliasing

若每 100 ms 有一次 5 ms 空洞，而遥测每 1 秒采一个均值，它可能完全不可见。这叫 aliasing（混叠）：采样频率不足导致真实周期被误表示。

长期监控用秒级 metric；短时 bubble 用微秒级 timeline。不要试图用 Prometheus 替代 profiler，也不要把 30 秒 trace 当线上容量趋势。


#### 7. Counter 资源冲突

DCGM profiling metrics、Nsight Systems GPU metrics 与 Nsight Compute 可能使用同一组硬件性能计数器。若 profiler 提示 counter unavailable，可用：

```bash
dcgmi profile --pause
# 运行短时 profiler
dcgmi profile --resume
```

暂停会影响集群遥测，必须在维护窗口执行并确保恢复。不同 GPU 支持的 field group 不同，可先用 `dcgmi profile --list` 查询。


#### 8. 一个最小 dashboard

每 GPU 至少同时画：

1. job/rank identity 与 GPU UUID；
2. GPU、SM、Tensor、DRAM activity；
3. framebuffer used 与 allocator-reported memory；
4. PCIe、NVLink、NIC throughput；
5. power、clock、temperature、throttle；
6. Xid/ECC/NVLink error；
7. 应用 step/TTFT/queue/KV-cache 指标。

只有硬件图没有应用 phase，常常只能看到“慢了”，看不到“哪一步慢了”。


## 参考资料

- [nvidia-smi 官方文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [NVIDIA DCGM Profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)
- [PyTorch Benchmark Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
