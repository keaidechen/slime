# 08 工程化与可观测性：调试、容错、CI 与性能分析

> 对应综述（`00_rl_infra_survey.md`）§2.10「工程化」。
> README 里有句纲领："RL bug 往往不会立刻报错"——权重没同步、logprob 对不上、数据错位，训练照跑，只是模型悄悄变差。slime 把正确性设施当一等公民。本篇盘点这些设施及对应代码。

---

## 1. 分离调试：把 RL 拆成两个可独立运行的半系统

最常用的调试手段（`slime/ray/placement_group.py:100-117` 的资源布局分支）：

| 参数 | 效果 |
|---|---|
| `--debug-rollout-only` | 只起 rollout 侧（SGLang + 数据流），不建训练 actor——验证数据生成、reward、partial rollout |
| `--debug-train-only` | 只起训练侧，rollout 数据从落盘的 dump 读——验证 loss、梯度、checkpoint |

配套机制：

- **rollout 数据落盘**：`_save_debug_rollout_data`（`slime/ray/rollout.py:663`）把每轮 rollout 数据 dump 到磁盘（`--dump-details` 等参数控制）；train-only 模式回放这份数据，实现"rollout 一次、训练调一百遍"；
- **`forge_load.py`**（`slime/rollout/forge_load.py`）：加载落盘数据但**不跳过 SGLang**——server、router、权重更新、offload 照常运行，专门用来测真实显存占用；
- **`sleep_rollout.py`**：rollout 侧彻底占位（死循环打日志），纯训练侧联调；
- **save/load 对称**：`MegatronTrainRayActor.save_model`（`slime/backends/megatron_utils/actor.py:542-564`）在 `debug_rollout_only` 时直接跳过；`update_weights`（actor.py:567-569）在两种 debug 模式下都短路。

**工作流建议**：新任务先 `--debug-rollout-only` 把数据和 reward 跑对并落盘 → 再 `--debug-train-only` 回放数据把训练跑对 → 最后全链路小规模 → 放量。

---

## 2. 正确性校验设施

### 2.1 权重同步对账

`--check-weight-update-equal`（train.py:29-30）：

- 启动时 `check_weights(action="snapshot")` + `reset_tensors`（placement_group.py:246-248）在引擎侧记录基准；
- 首轮权重同步后 `check_weights(action="compare")` 逐张量比对训练侧与引擎侧；
- disk 模式还有**版本号对账**（`slime/ray/actor_group.py:254-265`）：CI 中校验每个引擎的 `get_weight_version` 必须等于训练侧发布的版本，不匹配直接 `RuntimeError`。

### 2.2 训推数值监控（持续）

不是一次性校验，而是每步记录（05 篇 §5）：

- `train_rollout_logprob_abs_diff`：同参数下训练侧与推理侧 logprob 的绝对差，是 FP8、kernel 差异、routing 差异的"体温计"（loss.py:1073-1077）；
- `tis / tis_clipfrac / tis_abs / ois`：TIS 修正的强度与截断比例；
- `examples/train_infer_mismatch_helper/`：专门的离线诊断工具。

### 2.3 不变量断言

`Sample` 的长度校验（03 篇）、`add_samples` 的组长度断言（data_source.py:206-209）、`start_rollout_id` 全组一致断言（placement_group.py:216）——把"数据错位"这类 bug 尽量变成启动即报的错误。

---

## 3. 容错

- **健康监控**：`RolloutManager.__init__` 可选启动 `RolloutHealthMonitor`（`slime/ray/rollout.py:470-477`），定期检查 SGLang server 存活；`health_monitoring_pause/resume`（rollout.py:620-627）在权重同步等敏感窗口暂停监控避免误判；
- **故障恢复**：`RolloutServer.recover`（rollout.py:346）重建挂掉的引擎；训练侧下次 `update_weights` 时 `recover_updatable_engines`（rollout.py:601）+ 检测到 `num_new_engines > 0` 后把新引擎加进 NCCL 组（`slime/backends/megatron_utils/actor.py:597-607`）——**引擎热替换不断训**；
- **故障注入测试**：`_try_ci_fault_injection`（rollout.py:479）在 CI 里故意制造故障，验证恢复路径真的可用——"未经测试的容错等于没有容错"的实践；
- **checkpoint 恢复**：训练侧 `start_rollout_id` 对齐 + 数据源游标恢复（03 篇）+ `rollout_manager.load.remote(start_rollout_id - 1)`（placement_group.py:221-222），三层状态一起回到断点。

详细运维文档见 `docs/zh/advanced/fault-tolerance.md` 与 `docs/zh/advanced/reproducibility.md`。

---

## 4. Trace 与 Profiling

- **trace**：`slime/utils/trace_utils.py` 提供 `trace_function` / `trace_span` 装饰器与上下文管理器。rollout 链路的关键节点都埋了点——`generate_and_rm`（sglang_rollout.py:223）、`generate_and_rm_group`（289）、`sglang_generate`（201）、`reward_model`（273/283）——导出后可用 Chrome trace viewer 这类工具可视化"一步里每个样本的时间线"，定位长尾的利器；`build_sglang_meta_trace_attrs` 把 SGLang 返回的 meta（排队时间、prefill/decode 耗时）也纳入 trace。文档：`docs/zh/developer_guide/trace.md`；
- **timer**：`@timer` 装饰器（如 actor.py:566 的 `update_weights`）自动记录各阶段耗时；
- **性能指标**：`_log_rollout_data` / `compute_perf_metrics_from_samples`（rollout.py:1292-1359）计算 token 吞吐、按 `non_generation_time` 拆分"生成耗时 vs 环境耗时"（agentic 场景归因工具执行开销），以及 SGLang 请求级性能指标（`_compute_sglang_request_perf_metrics`）；
- **nsys 等深度 profiling**：`docs/zh/developer_guide/profiling.md`。

---

## 5. CI：把"正确性"变成回归测试

README 概括了 CI 的三层：

1. **CPU 单测**：数据流、参数、转换逻辑的纯 CPU 测试（`tests/` 下大量 `test_*.py`）；
2. **customization hook contract test**：保证 07 篇那些 `--*-path` 钩子的签名契约不被破坏；
3. **GPU e2e 测试**：真实 Megatron + SGLang 跑通 dense/MoE、checkpoint、数值精度、async rollout、OPD、PPO workflow、debug rollout-then-train replay。

CI 里的特色设施：`--ci-test` 参数触发严格断言（如 generate 里 `assert isinstance(sample.prompt, str)`，sglang_rollout.py:155-156）、磁盘权重更新的版本对账（04 篇）、故障注入（§3）。文档：`docs/zh/developer_guide/ci.md`。

---

## 6. 可复现性

- **种子链路**：`--rollout-seed` 驱动数据集 shuffle（`Dataset.shuffle` 用 `seed + epoch_id`，保证同 epoch 同排列，03 篇）；`sglang_enable_deterministic_inference` 时组内每个采样用 `rollout_seed + i` 的固定采样种子（sglang_rollout.py:110-112、317-319）；
- **确定性排序**：rollout 收集结果按 `sample.index` 排序（sglang_rollout.py:451-454），fully async 同样按 index 排序（fully_async_rollout.py:233-241）——乱序完成 ≠ 乱序训练；
- **调试文档**：`docs/zh/developer_guide/debug.md`、`docs/zh/advanced/reproducibility.md`。

---

## 7. 系列收尾：学习路线建议

至此本系列 8 篇已覆盖 slime 的全部主干。推荐的动手顺序：

1. 跑通 `docs/zh/get_started/quick_start.md` 的最小示例，对照 01/02 篇理解主循环；
2. 用 `--debug-rollout-only` + `--dump-details` 观察 `Sample` 的真实内容（03 篇）；
3. 改 `--advantage-estimator` 与 TIS 开关，观察 wandb 指标差异（05 篇）；
4. 写一个 `--custom-generate-function-path` 的 two-turn 玩具示例（07 篇 §1.2）；
5. 读 `examples/search-r1`，然后读 `examples/fully_async`，最后挑战 `examples/coding_agent_rl`；
6. 有余力可对照综述 §3 读 verl（HybridEngine/AgentLoop）与 AReaL（staleness-aware）源码，体会不同架构取舍。
