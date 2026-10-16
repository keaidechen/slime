# 5.3 可靠性、容量保护与发布运维

相关基础：[Ray、队列与前置系统](<../../../../06_Post_Training/27_权重发布异步与资源调度.md#s4>)。


## 入门例子：取消请求后的资源账本

请求从 running 进入 canceled 后，客户端 future、scheduler 队列、request slot、KV 引用和缓存锁可能由不同组件处理。看见 HTTP 结束并不证明这些都已回收。

正常完成与主动取消都会减少 active request，但可用 slot、locked/evictable KV 和内存池对应不同的所有权边界。可复用 prefix 留在缓存中是预期行为；只有仍被错误引用或永远无法重新分配的状态，才支持泄漏的解释。设备显存没有下降本身不足以证明泄漏。

下文的故障注入矩阵是目标行为清单，不表示当前实现对每项都提供原子性、无死锁或自动恢复保证。对换权中途失败，应停止错误发布、记录受影响 rank，并按实际恢复协议重建；不能只把 version 字符串改回去。

## 机制与实现

## 1. Admission control

SLO（Service Level Objective，服务等级目标）是服务承诺的可量化目标，例如“至少 99% 的请求首 token 延迟小于 1 秒”；阈值、统计窗口、失败请求和分母都属于定义。

过载时继续接收请求会让 queue、TTFT 和超时同时恶化。保护策略可以按并发请求、waiting tokens、预计 KV、租户配额和 SLO 分级拒绝或排队。拒绝必须快速、可观测并返回明确错误。

### 为什么同时要限制请求数和 token 数

假设池还可分配 12 个 token slot，请求 A/B 的新 KV 需求分别是 3/10；“允许再接两条请求”会接纳 13 个 slot 的需求，立即超预算。反过来，一千条极短请求虽然 KV 不大，也可能耗尽 CPU 对象、grammar 编译或输出队列。容量保护至少区分 request slot、KV slot、单轮工作量和下游输出缓冲。

可以给排队设置有限上限：教学假设稳定处理率为 10 req/s、当前有 20 条同等工作量待处理，则新请求仅排队就约需 2 s。若它的剩余首 token 预算只有 1 s，继续无界接收只会把可预期失败推迟。真实调度需考虑缓存与长度，不能把这个简单除法当作每个请求的准确预测。

Backpressure（背压）是让下游容量不足反馈给上游，例如限制并发或暂停投递；admission 是在入口决定是否接受。若重试者收到超时就立刻同时重发，会放大流量，形成 retry storm（重试风暴）。应将有限重试、退避和请求已输出状态纳入协议，不能把重试当作免费的恢复。

## 2. 资源生命周期

每条异常路径检查：request slot、KV page、radix lock、grammar state、LoRA ref、CUDA event、PD transfer buffer 和响应 future 是否释放。silent leak 往往只在重复 abort 或 timeout 后出现。

用总容量 12 个物理 slot 的玩具池说明：初始 4 个 slot 被无活跃引用的缓存持有、8 个 free；A 引用其中 2 个共享 slot，并新增 3 个私有 slot，实际占用变为 7、free 为 5。A 取消后，若把这 3 个 slot 直接释放，free 回到 8；若策略保留其可缓存部分，则 free 仍可能为 5，但这部分应变为可驱逐。后续驱逐 3 个应能把 free 恢复到 8。

可检查守恒关系 `总物理容量=free+已占用`；已占用再按互斥归属统计为受保护、可驱逐、请求私有或传输预留等。两个请求引用同一个共享 slot，只计一份物理存储，不能把每条请求的逻辑长度相加当作物理占用。清理动作还应幂等：相同 abort 重复到达时不会二次 free 或把引用计数减成负数。

清理顺序从依赖出发：停止新增消费者 → 等待或取消在途计算/传输 → 解除引用并处理缓存 → 释放映射与请求身份。具体函数可以交错，但在 GPU 仍读写某 slot 时把它发给新请求，会把资源泄漏问题变成更难发现的数据污染。

## 3. 故障注入矩阵

| 故障 | 预期 |
|---|---|
| client disconnect/abort | 停止无用计算并释放状态 |
| worker/rank crash | 实例退出 ready，整组失败 |
| collective timeout | 无死锁，记录 rank 与 operation |
| OOM | 请求明确失败或受控 retract，无错误 token |
| KV transfer 失败 | ownership 清晰、双方回收 |
| 权重更新失败 | 不发布混合版本 |
| 对象存储失败 | readiness 不提前成功 |

## 4. 配置管理

CLI 参数是显式实例配置；环境变量常控制 backend、实验开关和调试路径。将两者纳入版本化启动清单，在日志中打印安全白名单，并确认自动探测、环境变量和 CLI 的优先级。

参数和默认值变化快，权威来源是当前 parser、`server_arguments.mdx` 和 `environment_variables.mdx`。

## 5. 模型加载与预热

把下载、校验、反序列化、权重重排、GPU 搬运、JIT/compile 和 graph capture 分开计时。readiness 只在必要预热完成后成功。生产镜像与远端 artifact 使用 digest/revision，避免重启后漂移。

Liveness（存活）用于判断进程是否还可响应，readiness（就绪）用于判断能否接新业务。进程在下载权重时可以活着却未 ready；某个 TP rank 出错后，HTTP 进程也可能仍活着，但完整实例已不能 forward。重建时间可分成 `下载+加载+预热+必要检查` 的关键路径，能重叠的部分不能重复相加。把冷启动作为单独容量指标，才能解释故障后多久恢复可用副本。

## 6. 发布与回滚

Canary 同时观察延迟、吞吐、错误、输出漂移、cache 行为和显存。回滚条件在发布前定义；回滚覆盖模型、代码、kernel、路由、LoRA 和配置，而不是只切换容器 tag。

## 7. 值班排障顺序

1. 确认影响范围、版本和最近变更；
2. 保护流量并保存现场；
3. 用四层指标定位 API、scheduler、model/communication 或系统；
4. 对比健康实例和上一基线；
5. 回滚或最小化缓解；
6. 用故障注入补回归测试和告警。

## 8. 官方入口

- `sglang/docs_new/docs/advanced_features/server_arguments.mdx`
- `sglang/docs_new/docs/references/environment_variables.mdx`
- `sglang/docs_new/docs/references/faq.mdx`
- `sglang/docs_new/docs/developer_guide/release_process.mdx`
