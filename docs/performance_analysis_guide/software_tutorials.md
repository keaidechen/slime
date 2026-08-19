# 常用性能分析软件：安装、采集与界面操作教程

本章写给第一次使用性能工具的人。你不需要一次安装所有软件。先根据问题选择一个工具，完成“检查安装 → 采集最小数据 → 打开结果 → 回答一个问题”这四步。

## 1. 先理解两台机器的分工

大模型通常运行在远程 Linux GPU 服务器，而图形界面可以运行在你的本地电脑：

```text
GPU 服务器（target）
  -> 用 CLI 采集 trace/report
  -> 生成 .json、.nsys-rep、.ncu-rep、.pickle 等文件
  -> 安全复制到本地

本地电脑（host）
  -> 用浏览器、nsys-ui、ncu-ui 打开
  -> 搜索、缩放、比较和截图
```

不必为了看 GUI 给训练服务器安装桌面环境。Nsight Systems 官方支持在 Linux target 采集，再在 Windows、Linux 或 macOS host 查看；具体 host/target 组合以当前 [安装指南](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html) 为准。

Trace 可能包含源码路径、kernel 名、主机名、进程参数、prompt shape，甚至业务数据。复制或上传前先脱敏。Perfetto 和 PyTorch Memory Viz 的官方网页查看器默认在浏览器本地处理文件，但仍应遵守所在组织的数据规则。

## 2. 五分钟安装检查

先运行，不要看到缺失就全部安装：

```bash
python --version
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'

nvidia-smi
which nsys || true
which ncu || true
which tensorboard || true
which py-spy || true
which memray || true
which dcgmi || true
```

记录结果：

| 命令 | 缺失时是否阻塞入门 |
|---|---|
| `python`、PyTorch、`nvidia-smi` | 是，先修好运行环境 |
| `nsys` | 不阻塞 PyTorch Profiler；到系统时间线时再装 |
| `ncu` | 不阻塞系统分析；确定要看单 kernel 时再装 |
| `tensorboard` | 可改用 Perfetto；需要看训练曲线时再装 |
| `py-spy`、Memray | 只有 CPU/主机内存问题时需要 |
| DCGM | 单机学习可先用 `nvidia-smi`；集群监控再部署 |

不要擅自在共享集群安装驱动、修改性能计数器权限或启动系统服务。驱动、DCGM、容器权限通常由管理员负责。

## 3. `nvidia-smi`：先看 GPU 生命体征

### 3.1 它能回答什么

- 进程是否真的占用了 GPU；
- 显存用了多少；
- utilization、功耗、温度、频率是否随 workload 变化；
- 哪张卡明显不同。

它不能告诉你具体哪个 PyTorch 算子或 CUDA kernel 慢。

### 3.2 第一次使用

终端 A 启动 workload，终端 B 执行：

```bash
nvidia-smi
nvidia-smi dmon -s pucvmet
```

按 `Ctrl+C` 停止 `dmon`。不同驱动版本支持的 metric group 可能不同，先看：

```bash
nvidia-smi dmon --help
```

### 3.3 保存结构化数据

```bash
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv \
  -l 1
```

阅读时逐列回答：

1. workload 稳态时 GPU utilization 是否持续有工作；
2. memory.used 是否只增不减；
3. power.draw 和 clocks.sm 是否异常偏低；
4. 多卡是否有一张卡长期不同；
5. 指标变化与日志中的 step 是否在同一时间窗口。

`utilization.gpu=100%` 只表示采样窗口内 GPU 忙，并不等于 kernel 已达到硬件峰值。

官方字段说明：[NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。

## 4. DCGM：多卡和长期 GPU 遥测

DCGM（Data Center GPU Manager）适合集群遥测、健康检查和 Prometheus exporter。安装和 host engine 通常需要管理员处理。

### 4.1 确认环境已提供

```bash
dcgmi discovery --list
dcgmi dmon --list
```

如果命令不存在，先把 [DCGM 安装文档](https://docs.nvidia.com/datacenter/dcgm/latest/installation/index.html) 发给管理员，不要随意改变生产节点服务。

### 4.2 第一次观察两个字段

当前官方入门示例中，150 是温度，155 是功耗：

```bash
dcgmi dmon --field-id 150,155 --count 5
```

若输出 `N/A`，它可能代表设备不支持、无权限、尚未采样或值不可用，不能直接解释为 0。字段与命令以 [DCGM 系统管理员入门](https://docs.nvidia.com/datacenter/dcgm/latest/learn/getting-started-for-system-administrators/index.html) 为准。

### 4.3 什么时候从 `nvidia-smi` 升级到 DCGM

- 要连续观察数小时或跨多节点；
- 需要 NVLink/NVSwitch、健康和诊断；
- 要让 Prometheus 定期抓 GPU 指标；
- 需要把 GPU 指标与 job/rank 对齐。

## 5. PyTorch Profiler + Perfetto：最适合第一次看 trace

### 5.1 生成一个最小 trace

```python
import torch
from torch.profiler import ProfilerActivity, profile, record_function

x = torch.randn(2048, 2048, device="cuda")
y = torch.randn(2048, 2048, device="cuda")

for _ in range(5):
    torch.mm(x, y)
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with record_function("my_matmul"):
        torch.mm(x, y)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
```

### 5.2 用 Perfetto 打开

1. 浏览器访问 [Perfetto UI](https://ui.perfetto.dev/)；
2. 点击 **Open trace file**；
3. 选择 `trace.json`；
4. 等待解析完成；
5. 用 `W/S` 缩放，`A/D` 左右移动；
6. 用 `Ctrl+P` 搜索 track；
7. 点击 `my_matmul`，在底部查看起止时间和参数；
8. 按 `F` 把选中事件居中，再按一次让它适配窗口；
9. 展开 CPU thread 和 GPU stream，寻找 CPU op、CUDA launch、kernel 的对应关系。

若 `.trace.json.gz` 无法直接打开，复制一份后解压：

```bash
gzip -dk your_trace.trace.json.gz
```

Perfetto 支持 legacy JSON trace。完整快捷键和界面说明见 [Perfetto UI 官方文档](https://perfetto.dev/docs/visualization/perfetto-ui)。

### 5.3 第一次必须回答的四个问题

1. CPU operator 在哪条线程？
2. GPU kernel 在哪条 stream？
3. 中间是否有明显 CPU/GPU 空洞？
4. 一个 ATen op 对应一个还是多个 kernel？

不要一开始打开几十 GB 的多 rank trace。先学会读一个算子，再读两个 step。

## 6. TensorBoard：训练曲线和 PyTorch trace

### 6.1 安装

在运行训练的同一个虚拟环境中：

```bash
python -m pip install tensorboard
tensorboard --version
```

### 6.2 启动

假设日志在 `/tmp/slime_train_profile`：

```bash
tensorboard \
  --logdir /tmp/slime_train_profile \
  --host 127.0.0.1 \
  --port 6006
```

在本机任务上打开 `http://127.0.0.1:6006`。远程服务器不要直接暴露端口，使用 SSH tunnel：

```bash
ssh -L 6006:127.0.0.1:6006 user@server
```

然后在本地浏览器打开同一地址。

### 6.3 页面怎么读

- **Scalars/Time Series**：loss、reward、step time、tokens/s；先统一横轴是 train step 还是 rollout step。
- **Profiler/Trace**：若目录由 `torch.profiler.tensorboard_trace_handler` 生成，可选择 worker/rank 和 trace。
- **Histogram**：看分布变化，不用它代替数值正确性检查。

第一次操作：

1. 只选一个 run；
2. 关闭 smoothing 或记录 smoothing 值；
3. 横轴选择 step；
4. 同时显示 `perf/step_time` 与一个吞吐指标；
5. 框选稳态区域；
6. 再叠加 A/B 两个 run；
7. 检查两者的 step 语义和 workload 相同。

官方入口：[PyTorch TensorBoard](https://docs.pytorch.org/docs/stable/tensorboard) 和 [TensorBoard Get Started](https://www.tensorflow.org/tensorboard/get_started)。

## 7. PyTorch Memory Viz：看 CUDA allocator

### 7.1 生成 snapshot

```python
import torch

torch.cuda.memory._record_memory_history(max_entries=100000)

# 只运行准备分析的少量代码
run_your_workload()

torch.cuda.memory._dump_snapshot("snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

这些是调试接口，执行前核对当前 PyTorch 文档。长时间记录会产生很大的文件。

### 7.2 打开和阅读

1. 访问 [PyTorch Memory Viz](https://pytorch.org/memory_viz)；
2. 把 `snapshot.pickle` 拖入页面；
3. 先打开 **Active Memory Timeline** 找显存峰值；
4. 点击峰值附近的 block，看分配栈；
5. 再看 **Allocator State History**，区分 segment 和 block；
6. 记录 active、inactive、reserved 的关系；
7. 对比 `nvidia-smi`，判断是否存在 PyTorch allocator 不可见的显存。

Memory Viz 只看到 PyTorch allocator 管理的分配；NCCL 和直接 CUDA API 分配可能不可见。官方教程：[Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)。

## 8. Nsight Systems：读 CPU/GPU/NCCL 全局时间线

### 8.1 安装与确认

Nsight Systems 可来自 CUDA Toolkit，也可单独安装。让管理员根据 target 架构和用于查看的 host OS 选择包。安装后：

```bash
nsys --version
nsys-ui --version
```

只有 CLI 也可以在服务器采集，再把 `.nsys-rep` 复制到装有 GUI 的电脑。

### 8.2 第一次采集

```bash
nsys profile \
  -t cuda,nvtx,osrt \
  -s none \
  -o /tmp/first_nsys \
  python your_short_script.py
```

输出通常为 `/tmp/first_nsys.nsys-rep`。先在服务器看统计表：

```bash
nsys stats /tmp/first_nsys.nsys-rep
```

采集必须足够短。长任务应使用 NVTX/cudaProfilerApi capture range 或框架自带 profile window，见 [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。

### 8.3 在 GUI 打开

启动 `nsys-ui`，选择 **File → Open**，打开 `.nsys-rep`。界面中先找：

1. **Analysis Summary**：核对采集时间、机器和启用的 trace 类型；
2. **Timeline View**：展开 Processes；
3. 展开目标 PID/TID；
4. 找 CUDA API、CUDA HW、GPU Context/Stream；
5. 多卡任务再找 NCCL/NVTX；
6. 用鼠标拖选一个完整 iteration；
7. 在底部 Function Table 按 Total Time 排序；
8. 双击表中条目，让时间线跳到对应事件；
9. 找 GPU 空洞，并查看空洞前 CPU 或其他 stream/rank 在等待什么。

### 8.4 新手最容易误读的地方

- API 调用时间不是 kernel 执行时间；
- 不同 stream 上重叠的时间不能直接相加；
- NCCL 总时间高不代表都在关键路径；
- 某个 rank 最后进入 collective，其他 rank 的 NCCL 会表现为等待；
- 采集只有 driver、没有 worker，通常是 multiprocess capture 范围不对。

## 9. Nsight Compute：解释一个 kernel 为什么慢

### 9.1 前提

先通过 PyTorch Profiler 或 Nsight Systems 找到关键路径上的 kernel。不要把完整训练直接交给 Nsight Compute，因为 metric collection 会 replay kernel，开销很高。

### 9.2 第一次采集

```bash
ncu --version
ncu --list-sets

ncu \
  --set basic \
  --launch-count 1 \
  -o /tmp/first_ncu \
  python your_one_kernel_script.py
```

若脚本有很多 kernel，先用 `--kernel-name` 过滤；过滤语法和真实 kernel 名以当前 `ncu --help` 为准。

### 9.3 GUI 阅读顺序

启动 `ncu-ui`，通过 **File → Open** 打开 `.ncu-rep`：

1. 左侧选择目标 kernel launch；
2. 看 **Summary** 的规则提示，但不要直接把提示当结论；
3. 看 **GPU Speed of Light Throughput**：compute 与 memory 哪边更接近峰值；
4. 看 **Memory Workload Analysis**：DRAM/L2/shared memory；
5. 看 **Launch Statistics/Occupancy**：block、register、shared memory 是否限制并发；
6. 看 **Warp State Statistics**：主要 stall 类别；
7. 有 line info 时再看 **Source**，把指标对应到代码；
8. 保存 baseline，把第二份 report 加入比较，确认改动影响。

官方 Quickstart 说明 `ncu` 是 CLI、`ncu-ui` 是 GUI，并提供 report 比较功能：[Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)。

## 10. py-spy：Python 进程卡在哪里

### 10.1 安装和三种模式

```bash
python -m pip install py-spy

py-spy top --pid 12345
py-spy dump --pid 12345
py-spy record -o profile.svg --pid 12345
```

- `top`：实时看热点函数；
- `dump`：卡死时立刻打印所有 Python thread 的栈；
- `record`：采样一段时间并生成火焰图。

启动并跟踪一个新程序：

```bash
py-spy record -o profile.svg -- python your_script.py
```

多进程程序可查 `--subprocesses`。在 Linux 上 attach 现有进程可能被 ptrace 权限阻止；不要为了方便绕过集群安全策略。完整说明见 [py-spy 官方仓库](https://github.com/benfred/py-spy)。

### 10.2 火焰图怎么读

- 横向宽度代表采样占比，不是时间轴；
- 纵向是调用栈；
- 顶部宽框通常是实际热点函数；
- 大量线程停在同一锁/queue，可形成等待线索；
- Python 火焰图不会自动解释 GPU kernel 内部。

## 11. Memray：Python/native 主机内存

### 11.1 安装与采集

```bash
python -m pip install memray
python -m memray run -o allocations.bin your_script.py
```

生成报告：

```bash
python -m memray flamegraph allocations.bin
python -m memray tree allocations.bin
```

打开生成的 HTML。默认 flame/icicle 图中，框宽表示在所选峰值存活的分配量，不表示函数执行时间，也不直接代表调用次数。

### 11.2 什么时候用它

- Python worker RSS 持续增长；
- tokenizer、dataset、request object 或序列化对象泄漏；
- C/C++ extension 的 host memory 需要 native stack；
- CUDA 显存正常，但进程被主机 OOM killer 杀死。

它不是 CUDA 显存工具。官方入门：[Memray Getting Started](https://bloomberg.github.io/memray/getting_started.html) 和 [Flame Graph Reporter](https://bloomberg.github.io/memray/flamegraph.html)。

## 12. Prometheus + Grafana：持续观测，而不是单次 profile

Profiler 用于短窗口下钻；Prometheus/Grafana 用于长期趋势、容量和告警。

### 12.1 先确认服务暴露 metrics

以 SGLang 为例，启用 metrics 后检查：

```bash
curl http://127.0.0.1:30000/metrics
```

先读原始输出，确认 metric 名、label 和单位。不要从其他版本 dashboard 猜 metric 名。

### 12.2 最小 Prometheus 配置

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: sglang
    static_configs:
      - targets: ["127.0.0.1:30000"]
```

启动并检查 target：

```bash
prometheus --config.file=prometheus.yml
```

打开 `http://127.0.0.1:9090/targets`，必须先看到 target 为 UP。然后在 query 页面输入一个真实 metric 名。计数器通常使用 `rate(metric_total[5m])`，gauge 通常直接绘制；具体类型看 `/metrics` 的 `TYPE` 注释。

### 12.3 Grafana 第一个 dashboard

1. 打开 Grafana；
2. **Connections → Data sources → Add data source**；
3. 选择 Prometheus，填入 Prometheus 地址；
4. 点击 **Save & test**；
5. 进入 **Explore**，先执行一条能返回数据的查询；
6. 再 **Dashboards → New dashboard → Add visualization**；
7. 创建 GPU utilization、queue、running requests、KV Cache、TTFT/TPOT、错误率等 panel；
8. 所有 panel 使用同一时间范围；
9. 避免把 request ID、prompt 等作为高基数 label。

官方教程：[Prometheus First Steps](https://prometheus.io/docs/introduction/first_steps/) 和 [Grafana + Prometheus](https://grafana.com/docs/grafana/latest/fundamentals/getting-started/first-dashboards/get-started-grafana-prometheus/)。生产部署还需要认证、TLS、持久化、保留周期和权限管理，本教程中的本地配置不等于生产配置。

## 13. 软件学习顺序

按以下顺序练习，不要颠倒：

| 周期 | 软件 | 完成动作 |
|---|---|---|
| 第 1 次 | `nvidia-smi` | 对齐一次训练 step 与 GPU 曲线 |
| 第 2 次 | PyTorch Profiler + Perfetto | 从 Python op 找到 CUDA kernel |
| 第 3 次 | TensorBoard | 比较两个 run 的稳态吞吐和正确性 |
| 第 4 次 | Nsight Systems | 找到一个 GPU gap 的直接前因 |
| 第 5 次 | Nsight Compute | 解释一个热点 kernel 的瓶颈类型 |
| 第 6 次 | Memory Viz / Memray | 分清 CUDA allocator 与主机内存 |
| 第 7 次 | Prometheus/Grafana/DCGM | 建一个跨分钟的容量 dashboard |

每学一个软件，只要求回答一个问题并保留一个证据文件。能够正确选择工具，比会点完所有菜单更重要。

## 14. 常见安装与界面问题

### 14.1 “权限不足，读不到 GPU performance counters”

这是安全策略，不应自行绕过。把错误、GPU/driver、工具版本和官方权限说明发给管理员。PyTorch Profiler 和 Nsight Systems 的基础 CUDA timeline 往往仍可提供线索，具体取决于环境。

### 14.2 “GUI 装在服务器上打不开”

采用 target CLI 采集、host GUI 打开的方式。确认本地 GUI 版本能读取服务器生成的 report；必要时安装匹配或更新版本。

### 14.3 “trace 打开后什么都没有”

1. 核对采集窗口是否覆盖 workload；
2. 看命令是否只抓到了 launcher/driver；
3. 多进程是否需要 child/fork 追踪；
4. CUDA activity 是否启用；
5. 程序是否在 profile active window 内调用了 `prof.step()`；
6. 先回到一个单进程、单 kernel 脚本验证工具链。

### 14.4 “trace 太大或 UI 卡死”

- 减少 active step；
- 先关闭 stack、shape、memory；
- 只采关键 activity；
- 减少 rank；
- 先用 CLI summary；
- 不要把初始化到退出的完整生产任务全部录下来。

### 14.5 “开 profiler 后任务慢很多”

这是正常扰动，特别是 stack、shape、memory 和 Nsight Compute metric replay。性能基线必须在 profiler 关闭时测；profile 只负责解释瓶颈。
