# 性能分析报告模板

复制本文件，为每次分析创建一份独立报告。不要删除“正确性”和“原始证据”部分；性能数字没有 workload 与正确性约束就不可解释。

## 1. 一句话结论

> 在【固定 workload】下，把【唯一改动】从 A 改为 B 后，【主指标】由 X 变为 Y（提升 Z%）；【质量/正确性指标】无显著退化。证据支持/不支持“【瓶颈假设】”。

若尚未完成验证，写“当前只有假设，无优化结论”。

## 2. 问题与边界

- 现象：
- 首次出现时间/commit：
- 影响范围：
- 本次要回答的具体问题：
- 不在本次范围内的部分：
- 性能目标或 SLO：

## 3. 环境清单

| 项目 | 值 |
|---|---|
| 日期/时区 | |
| 主机/集群 | |
| GPU 型号、数量、显存 | |
| Driver/CUDA | |
| CPU/NUMA/RAM | |
| 网络/拓扑 | |
| OS/kernel/container | |
| Python/PyTorch | |
| Slime commit | |
| Megatron commit/版本 | |
| SGLang/vLLM commit/版本 | |
| NCCL/Transformer Engine/Triton | |
| 功耗/频率设置 | |

附环境命令原始输出路径：

## 4. Workload 与完整命令

- 模型/revision：
- dtype/quantization：
- input/output/sequence 长度分布：
- 样本数、batch、micro/global batch：
- request rate/concurrency：
- TP/PP/CP/EP/DP：
- rollout/train GPU 分配：
- cache/CUDA Graph/compile：
- reward/tool/environment：
- warmup 与测量窗口：
- 随机种子和固定数据：

完整启动命令或配置文件链接：

```bash
# 粘贴可复现命令；删除 token、密码、私有 URL 等秘密
```

## 5. 正确性与质量门槛

| 指标 | 基线 | 实验 | 可接受范围 | 通过？ |
|---|---:|---:|---:|---|
| loss | | | | |
| reward | | | | |
| KL/log-prob | | | | |
| grad norm | | | | |
| 有效 token/样本数 | | | | |
| output length/truncated | | | | |
| 错误/超时/重试 | | | | |
| 任务特定质量指标 | | | | |

## 6. 无 profiler 基线

至少排除初始化和 warmup，报告 median、p95、样本数，不只报告均值。

| 指标 | median | p95 | 单位 | 备注 |
|---|---:|---:|---|---|
| 端到端 step/E2E | | | s/ms | |
| tokens/s 或 samples/s | | | | |
| TTFT/TPOT/ITL | | | ms | 推理任务填写 |
| rollout_time | | | s | Slime 填写 |
| actor_train_time | | | s | Slime 填写 |
| update_weights_time | | | s | Slime 填写 |
| wait_time_ratio | | | % | Slime 填写 |
| GPU 利用率/功耗/显存 | | | | |

原始日志：

## 7. 阶段分解与测量事实

| 阶段/rank | 时间或占比 | 数据来源 | 是否关键路径 |
|---|---:|---|---|
| | | | |

只写直接观测到的事实，例如：

- rank 7 比其他 rank 的 backward 晚 180 ms；
- decode 每步之间存在约 60 μs CPU gap；
- 35% rollout 时间位于非生成 tool 阶段。

不要在本节写“因为 NCCL 有 bug”这类未经验证的解释。

## 8. 假设

> 如果【原因】是瓶颈，那么只改变【一个变量】后，应该观察到【可量化预测】；若没有出现，则该假设被否定或证据不足。

- 支持假设的现有证据：
- 与假设冲突的证据：
- 可能的替代解释：

## 9. A/B 实验设计

唯一改变的变量：

保持不变的变量：

重复次数和统计方法：

预期结果：

回滚方式：

## 10. A/B 结果

| 指标 | A：基线 | B：实验 | 变化 | 是否通过 |
|---|---:|---:|---:|---|
| 主性能指标 | | | | |
| p95/p99 延迟 | | | | |
| GPU-hours/有效产出 | | | | |
| 显存峰值 | | | | |
| 正确性/质量 | | | | |
| 错误率 | | | | |

变化率统一使用：

```text
吞吐提升 = (B - A) / A
延迟下降 = (A - B) / A
```

## 11. 原始证据

- benchmark JSON/JSONL：
- PyTorch trace：
- `.nsys-rep`：
- `.ncu-rep`：
- memory snapshot/Memray：
- TensorBoard/W&B 链接：
- 系统监控：
- 脚本/配置 diff：

记录文件大小、rank、采集窗口和工具版本。上传 trace 前检查 prompt、路径、主机名、环境变量和业务数据是否需要脱敏。

## 12. 结论与下一步

- 假设：支持 / 否定 / 证据不足；
- 最终可合入的改动：
- 适用范围与已知副作用：
- 是否需要长期回归监控：
- 下一条最有价值的假设：
- 若回归，如何回滚：
