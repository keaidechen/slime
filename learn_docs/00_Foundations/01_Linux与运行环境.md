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

这些命令的输入和输出可以按下表理解。“输入”是你在终端键入的命令；“输出”是命令执行后打印的结果。

| 输入 | 作用 | 输出怎么看 |
|---|---|---|
| `pwd` | 显示当前工作目录 | 输出一个绝对路径，例如 `/workspace/slime` |
| `which python` | 查找 shell 将要执行的 `python` | 输出 Python 可执行文件的路径；无输出通常表示没有在 `PATH` 中找到 |
| `python --version` | 查看 Python 版本 | 例如 `Python 3.11.9` |
| `python -m pip show torch` | 让当前这个 Python 查询 PyTorch 包 | `Version` 是版本，`Location` 是安装目录；提示 `Package(s) not found` 表示该 Python 环境未安装 |
| `git rev-parse HEAD` | 查看当前代码版本 | 输出一串 commit ID；复现实验时应保存它 |

`which python` 和 `python -m pip ...` 要配对使用。直接运行 `pip show torch` 有可能查到另一个 Python 环境的包。一些 Linux 发行版只提供 `python3` 命令；若 `which python` 找不到，但 `which python3` 能输出路径，可在后续命令中一致地把 `python` 换成 `python3`。启用项目的虚拟环境后，命令路径还可能再次改变。

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

### 2.1 `nvidia-smi`：查看 GPU 和驱动

`nvidia-smi` 的主表中，`Driver Version` 是 NVIDIA 驱动版本，`GPU-Util` 是采样窗口内 GPU 处于忙碌状态的时间比例，`Memory-Usage` 是“已用显存 / 总显存”，`Temp` 是温度，`Pwr:Usage/Cap` 是“当前功耗 / 功耗上限”。底部 `Processes` 表通常会列出使用 GPU 的 PID、进程名和显存用量。

`nvidia-smi` 顶部展示的 `CUDA Version` 通常描述当前驱动能支持的 CUDA 版本上限，不是当前 Python 包实际使用的 CUDA runtime 版本。核对时同时保留 `Driver Version`、`torch.version.cuda` 和镜像身份。

`nvidia-smi topo -m` 输出 GPU、CPU 与网卡之间的拓扑矩阵。交叉单元格表示两个设备之间的连接路径：`X` 是同一设备，`NV#` 是 # 条 NVLink，`PIX`/`PXB` 是经过一个或多个 PCIe switch，`PHB` 会经过 PCIe host bridge，`NODE`/`SYS` 表示路径更远。它用来看相对位置，不直接等于实测带宽。

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
netstat -lntp
df -h
```

### 3.1 怎么拆解一条 Linux 命令

以 `ps -eo pid,ppid,stat,etime,args` 为例：`ps` 是命令名，`-e` 和 `-o` 是选项（option），`pid,ppid,stat,etime,args` 是 `-o` 选项接收的参数。`-eo` 是 `-e -o` 的紧凑写法。选项用来改变命令的行为，但不要假设所有命令的字母选项都有相同含义；不确定时可用 `命令 --help` 或 `man 命令` 查看本机帮助。

一条命令还有三种需要区分的结果：标准输出是正常结果，标准错误用来显示警告或错误，退出状态是一个数字。按惯例，退出状态 `0` 表示命令成功，非 `0` 表示失败；命令没打印任何文字不代表它失败。在 bash/zsh 中，紧接着运行 `echo $?` 可查看上一条命令的退出状态。

### 3.2 `ps`：查看进程

`ps` 可理解为 process status（进程状态）。下面这条命令列出系统中的所有进程，并指定要显示的列：

```bash
ps -eo pid,ppid,stat,etime,args
```

| 输入部分 | 含义 |
|---|---|
| `-e` | 选中所有进程，而不只是当前终端的进程 |
| `-o` | 自定义输出列，列名用逗号分隔 |

示例输出：

```text
    PID    PPID STAT     ELAPSED COMMAND
      1       0 Ss      02:15:30 python -m server
   2345       1 Sl         03:42 python train.py --port 8000
   2410    2345 R+         00:01 python worker.py
```

| 输出列 | 含义 | 新手常用解读 |
|---|---|---|
| `PID` | Process ID，进程的数字标识 | 可用它继续查日志、端口或 GPU 占用 |
| `PPID` | Parent Process ID，启动该进程的父进程 PID | 相同 PPID 的进程往往由同一个启动器管理 |
| `STAT` | 进程当前状态，可由多个字符组成 | `R` 运行/等待 CPU，`S` 可中断睡眠，`D` 不可中断等待（常见于 I/O），`T` 已停止，`Z` 僵尸进程 |
| `ETIME` / `ELAPSED` | 自进程启动以来的时间 | 格式可为 `mm:ss`、`hh:mm:ss` 或 `天-hh:mm:ss`；它不是 CPU 累计计算时间 |
| `ARGS` / `COMMAND` | 启动命令及参数 | 用来区分多个 Python worker 分别在做什么 |

`STAT` 后面还可能有附加字符：`s` 表示 session leader，`l` 表示多线程，`+` 表示位于前台进程组。例如 `Sl` 不是新的一种状态，而是“睡眠中，且是多线程进程”。进程长时间显示 `S` 往往只是在等待任务，不能单凭它判定程序卡死。`Z` 表示进程已经退出，但父进程还没有回收其退出状态。

另一个常见写法是 `ps -ef`，它也列出所有进程，但使用预设的完整格式。排查时要查找某个关键字，可以使用 `ps -ef | grep python`；其中 `|` 会把左边命令的输出交给右边的 `grep` 做文本筛选。

### 3.3 `netstat` 和 `ss`：查看端口与网络连接

`netstat` 可理解为 network statistics（网络统计）。查看正在监听的 TCP 端口及所属进程，可运行：

```bash
netstat -lntp
```

| 选项 | 含义 |
|---|---|
| `-l` | 只显示正在监听（listening）的 socket |
| `-n` | 直接显示数字 IP 和端口，不把它们解析成主机名或服务名 |
| `-t` | 只显示 TCP socket |
| `-p` | 显示占用 socket 的 PID 和程序名，可能需要更高权限 |

这些单字母选项可合并，所以 `-lntp` 等价于 `-l -n -t -p`。示例输出：

```text
Proto Recv-Q Send-Q Local Address           Foreign Address         State       PID/Program name
tcp        0      0 127.0.0.1:8000          0.0.0.0:*               LISTEN      2345/python
tcp6       0      0 :::22                   :::*                    LISTEN      812/sshd
```

| 输出列 | 含义 |
|---|---|
| `Proto` | 协议，例如 `tcp` 或 `tcp6` |
| `Recv-Q` | 接收队列情况；对监听 socket，它与当前等待建立的连接有关 |
| `Send-Q` | 发送队列情况；对监听 socket，它通常反映队列上限，不是“已发送字节数” |
| `Local Address` | 本机程序绑定的地址和端口 |
| `Foreign Address` | 对端地址；监听状态下通常是 `*`，表示尚未指定某个对端 |
| `State` | TCP 状态；`LISTEN` 表示等待新连接，`ESTABLISHED` 表示连接已建立 |
| `PID/Program name` | 占用该端口的进程 PID 和程序名 |

`Local Address` 最容易误读：

- `127.0.0.1:8000`：只能从同一个网络命名空间的本机访问；其他机器通常访问不到。
- `0.0.0.0:8000`：监听所有 IPv4 网卡；是否能从外部访问还取决于防火墙、端口映射和网络路由。
- `:::22`：在 IPv6 未指定地址上监听；它是否同时接受 IPv4 取决于系统配置。

`netstat` 来自较旧的 `net-tools` 工具包，一些新 Linux 环境默认不安装它。现在更常用 `ss`（socket statistics）：

```bash
ss -ltnp
```

`ss` 这里的 `-l`、`-t`、`-n`、`-p` 与上面的用途一致。它的常见输出列为 `State`、`Recv-Q`、`Send-Q`、`Local Address:Port`、`Peer Address:Port` 和 `Process`。只想查 8000 端口时，可使用 `ss -ltnp | grep ':8000'`。查不到结果表示当前网络命名空间里没有 TCP 程序在该端口上监听，不能由此断定另一台机器或宿主机也没有监听。

若要同时看监听和已建立的 TCP 连接，可把 `-l` 换成 `-a`：`ss -antp` 或 `netstat -antp`。`-a` 表示 all。此时还常会看到 `TIME-WAIT`（连接关闭后的正常等待阶段）、`CLOSE-WAIT`（对端已关闭，本地程序尚未完成关闭）等状态。不能看到一个 `TIME-WAIT` 就当作故障；若 `CLOSE-WAIT` 持续大量增长，再检查程序是否正确关闭了连接。查 UDP 时将 `-t` 换成 `-u`，例如 `ss -lunp`。

### 3.4 `df`：查看文件系统剩余空间

```bash
df -h
```

`df` 用于查看已挂载文件系统的使用量，`-h` 表示用 K、M、G 等更易读的单位显示。输出中 `Size` 是总容量，`Used` 是已用容量，`Avail` 是可用容量，`Use%` 是使用率，`Mounted on` 是挂载点。查某个目录所在文件系统，可用 `df -h /path/to/directory`。`df` 回答“这个文件系统还剩多少空间”；它不回答“某个目录占了多少空间”，后者通常用 `du -sh /path/to/directory` 查看。

`ps` 的进程状态可以帮助区分运行、睡眠和退出后的状态，但不会直接给出 Python 栈或 GPU 等待原因。`ss` 和 `netstat` 证明的是“某个 socket 存在”，不能单独证明服务能正确处理请求。它们的进程信息也可能受权限限制。先识别目标 PID，再结合日志和实际请求缩小排查范围。

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
