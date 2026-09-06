# Linux 与运行环境：先知道程序运行在哪里

> 前置：会运行 Python。目标：把一次实验的机器、进程、软件和输入记录完整。预计 2–3 小时，命令以 Linux GPU 机器为目标。

## 1. 从你输入的命令追到 GPU

执行 `python train.py` 后，shell 创建 Python 进程。它加载 Python 包，框架再通过 CUDA runtime/driver 向 GPU 提交工作。容器提供用户态文件、依赖和部分隔离；GPU 驱动与实际设备来自宿主机。

```text
终端 / shell
  → Python 解释器与进程
  → PyTorch、训练或推理框架
  → CUDA 用户态库 / GPU 驱动
  → 当前进程可见的设备
```

因此，“我的终端能看到 GPU”和“这份 Python 使用了正确的 GPU 包”是两个检查。`nvidia-smi` 成功，但 Python 里 `torch.cuda.is_available()` 为假，仍可能是解释器或框架安装不匹配。

## 2. 先建立环境身份

在准备运行的容器或服务器内执行以下只读命令，保存输出。不要用本地 Windows 的结果替代远端 H20 的结果。

```bash
pwd
which python
python --version
python -m pip show torch
nvidia-smi
nvidia-smi topo -m
git rev-parse HEAD
```

再用同一个解释器检查框架：

```python
import os
import torch

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, p.total_memory / 2**30, "GiB")
```

`nvidia-smi` 展示的 CUDA 版本通常描述驱动可支持的 CUDA 版本上限，不是当前 Python 包实际使用的 runtime 版本。核对时同时保留 driver、`torch.version.cuda` 和镜像身份。

## 3. 进程、端口与文件是三种不同资源

| 资源 | 查什么 | 常见误判 |
|---|---|---|
| 进程 | PID、父进程、命令、退出码 | HTTP 启动器还活着，就认为 GPU worker 健康 |
| 端口 | 哪个地址/端口在监听，属于哪个 PID | localhost 在不同机器上指同一个地方 |
| GPU | 可见编号、物理设备、进程显存 | `cuda:0` 永远是机器物理第 0 卡 |
| 文件 | 路径、挂载、剩余空间、权限 | 容器里的路径在所有节点都存在 |

```bash
ps -eo pid,ppid,stat,etime,args
ss -ltnp
df -h
```

`ps` 的进程状态可以帮助区分运行、睡眠和退出后的状态，但不会直接给出 Python 栈或 GPU 等待原因。`ss` 的进程信息可能受权限限制。先识别目标 PID，再缩小排查范围。

## 4. 四类运行失败分别找谁

| 现象 | 第一层检查 | 下一步证据 |
|---|---|---|
| `ModuleNotFoundError` | 解释器、包环境、工作目录 | `which python` 与 `python -m pip show` |
| CUDA 不可用 | driver、设备暴露、框架构建 | 上面的 Python 环境记录 |
| 服务连接被拒绝 | 监听地址、端口、worker 是否退出 | `ss`、服务日志、退出码 |
| 程序启动后一直等待 | 资源申请、进程组会合、模型加载 | 每个进程最后的日志阶段 |

在分布式任务中，把日志至少标上时间、节点、PID、rank 和角色。只有一句“初始化中”，很难区分下载权重、构建通信组和等待 Ray 资源。

## 5. 一份可复现的实验目录

```text
run_001/
  environment.txt    机器、驱动、框架与源码版本
  command.txt        完整启动命令和显式配置
  workload.json     模型、数据、长度、batch、采样参数
  stdout.log        带时间和进程身份的输出
  metrics.csv       原始测量点，保留单位
  report.md         假设、改动、结果与正确性检查
```

环境变量只记录与实验相关的白名单，避免把密钥写入报告。相同脚本在不同目录下可能加载不同文件，因此工作目录也属于环境身份。

## 6. 动手与完成标准

1. 选一台可用 H20 机器，记录实际 GPU 数、显存、节点和互联；不预设它是几卡节点。
2. 用上述 Python 代码记录框架与可见设备。
3. 对一个已有的小任务，找到启动进程、worker 和输出目录。
4. 写清“终端在本地，计算在远端，trace 在哪里，怎样取回”。

完成时，你应能解释：驱动与 runtime 的区别；逻辑设备与物理卡的区别；进程存活、端口监听、模型可服务为什么不是同一件事。

下一课：[PyTorch 张量与自动微分](02_PyTorch张量与自动微分.md)。具体采集软件安装见[软件教程](../../docs/performance_analysis_guide/software_tutorials.md)，完整实验记录见[可信基线](../../docs/performance_analysis_guide/01_baseline.md)。
