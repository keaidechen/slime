# Ray、队列与背压：从远程调用走到 RL 调度

> 前置：[四种异步](../03_Hardware_Systems/进程线程与四种异步.md)、进程与设备映射。先在 CPU 上观察调度，不需要加载大模型。


## Task、Actor 与 ObjectRef

Task 是一次远程函数执行；Actor 是持有状态、能被远程调用的对象，通常由专用 worker 进程承载。ObjectRef 表示远程结果的引用；调用 `.remote()` 提交工作，返回引用并不代表工作完成。

在 RL 中，trainer 要长期保存模型、优化器和通信组，因此适合用 Actor 表达；某个无状态的评分函数可以表达为 Task。具体框架可能有不同封装，应沿本仓库调用链确认。

```python
import ray

ray.init(address="local", num_cpus=2)

@ray.remote(num_cpus=1)
class Counter:
    def __init__(self):
        self.value = 0

    def add(self, n):
        self.value += n
        return self.value

c = Counter.remote()
refs = [c.add.remote(1) for _ in range(3)]
print(ray.get(refs))  # 默认同步 actor：这里得到 [1, 2, 3]
ray.shutdown()
```

这个例子演示有状态调用，不演示并行 GPU 训练。Actor 的方法并发与排序还会受 async/threaded actor、调用方及配置影响，不应从此例推广成所有 Ray 调用都严格串行。[Ray Actors](https://docs.ray.io/en/latest/ray-core/actors.html)


## 逻辑资源声明不等于硬件隔离

`num_gpus=1` 表达调度资源需求；它不是对显存、SM 占用或带宽的硬隔离。两个角色共享一张 GPU 时，需要框架自己安排内存和执行阶段。

Placement group 把若干资源需求作为一组进行预留，具体 bundle 到节点的放置仍受策略和可用资源约束。读 slime 的 placement 时要分别记录：资源声明、实际节点/设备、rank group，不能把三者当成同一张表。


## 队列里积压的究竟是什么

| 队列内容 | 主要成本 | RL 风险 |
|---|---|---|
| 待生成 prompt | 请求元数据、等待时间 | 长尾、取消、超时 |
| 未就绪 ObjectRef | 调度状态和未来结果 | 无限制提交、依赖堆积 |
| 已生成 trajectory | token、logprob、reward、存储 | 显存/内存增长、policy lag |
| 等待换权的任务 | 版本依赖与资源占用 | 混合版本或错误地提前放行 |

同样是 100 个样本，短回答和多轮工具轨迹的字节数、生成时间并不相同。容量控制可以同时设置数量、token、字节或时间预算。


## 用 ray.wait 控制在途工作

完整脚本：[ray_backpressure.py](../examples/ray_backpressure.py)。需要已安装 Ray 的环境；它不连接既有集群，也不申请 GPU。

```bash
python docs/ai_infra/examples/ray_backpressure.py --max-pending 4
```

`python` 是解释器，`docs/ai_infra/examples/ray_backpressure.py` 是要执行的脚本，`--max-pending 4` 是传给脚本的参数，表示最多保留 4 个已提交但尚未被消费的 ObjectRef。可用 `python docs/ai_infra/examples/ray_backpressure.py --help` 只查看参数帮助，不运行实验。

正常输出有两部分：`completion order` 是 12 个任务的完成顺序，`elapsed seconds` 是本次运行的总耗时。完成顺序不一定是 0、1、2……，每次耗时也会波动；这正是脚本要展示的并发现象，不是结果错了。`--max-pending` 小于 1 时，脚本会打印参数错误并以非 0 状态退出。

脚本最多保留指定数量的未消费引用；达到上限时等一个结果就绪，消费后再提交。`ray.wait` 给出 ready 与 remaining 两组引用，`ray.get` 读取已就绪结果。这既限制积压，也避免一定按提交顺序等待最慢的第一项。[Ray wait](https://docs.ray.io/en/latest/ray-core/api/doc/ray.wait.html)

这个例子限制的是提交窗口；它没有解决所有问题，例如结果落盘速度、字节预算和跨系统取消。对于真实流水线，还需要为生产、存储、消费分别设边界。


## 吞吐、队列与陈旧度的一个算例

假设 rollout 每秒生产 12 个有效样本，trainer 每秒消费 10 个。缓冲未满时，60 秒后约新增 120 个样本。增加 buffer 只能延后溢出，并未消除速率差。

反过来，rollout 每秒 8 个、trainer 可消费 10 个，则 trainer 会等待。目标是结合 GPU 成本、长度分布和 policy freshness 调整资源，不能只让队列始终很长。

稳定系统中可用 Little 定律的直觉：平均在途数量约等于平均到达率乘平均驻留时间。它要求长期稳定且口径一致，不能直接用到不断积压、丢弃未计数或启动阶段的流水线。


## 对照 slime 时维护三份账

1. **资源账**：哪个 Actor 在哪台机器、用哪些 GPU，何时 onload/offload。
2. **样本账**：sample id、状态、token 数、生成版本、排队时间、消费次数。
3. **依赖账**：谁在等待生成、reward、训练或换权完成，哪个阶段可以安全重叠。

依次读 [slime 编排](../13_Source_Analysis/slime/01_architecture_and_ray_orchestration.md)、[rollout](../13_Source_Analysis/slime/02_rollout_sglang_server_mode.md)、[数据与异步](../13_Source_Analysis/slime/03_data_buffer_partial_rollout_async.md)。独立数据平面的比较放在 [TransferQueue](../13_Source_Analysis/slime/10_transferqueue.md)，不作为起步要求。


## 窗口、长尾与重复消费

窗口为 1 时，生产者等当前任务结束才继续；窗口为 4 时，最多四项在途工作可以互相覆盖等待。资源不足或存在串行依赖时，这不产生四倍加速。按提交顺序等待还会让一个慢任务挡住已经完成的后续结果，按完成顺序消费改变的是这种队头阻塞。

生成、排队、消费和换权分别可能成为限制。增加 worker 只能缓解可并行且资源充足的阶段；若消费者或发布边界更慢，新增生产能力只会增加队列与样本年龄。失败后的重试还必须区分未完成工作和已经计入训练的样本，否则吞吐增加可能来自重复消费。

## 完整推演

[异步队列与权重发布推演](<权重样本与异步一致性.md#异步rl推演队列样本身份与权重发布边界>)：在途任务、完成队列与实际源码背压。


---


<!-- series-nav -->

[上一篇：RL 数据与数值正确性：系统优化前先确定等价关系](<RL数据与数值正确性.md>) · [专题目录](<README.md>) · [下一篇：权重、样本与异步系统的一致性](<权重样本与异步一致性.md>)

<!-- /series-nav -->
