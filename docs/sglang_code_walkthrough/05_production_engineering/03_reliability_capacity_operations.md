# 5.3 可靠性、容量保护与发布运维

## 1. Admission control

过载时继续接收请求会让 queue、TTFT 和超时同时恶化。保护策略可以按并发请求、waiting tokens、预计 KV、租户配额和 SLO 分级拒绝或排队。拒绝必须快速、可观测并返回明确错误。

## 2. 资源生命周期

每条异常路径检查：request slot、KV page、radix lock、grammar state、LoRA ref、CUDA event、PD transfer buffer 和响应 future 是否释放。silent leak 往往只在重复 abort 或 timeout 后出现。

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

