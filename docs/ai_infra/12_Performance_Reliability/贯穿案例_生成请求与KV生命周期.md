# 贯穿案例二：一个生成请求如何穿过队列、GPU与KV生命周期

> 证据：SGLang本地快照源码已核对，未启动生成服务或采集GPU trace。基线为普通文本、单个TP组、PP=1、无PD/推测/LoRA/Mamba；先关闭overlap观察，再只打开overlap比较。高级路径不能直接套用基线。

## 请求身份比batch行号更稳定

一条请求会经历HTTP/TokenizerManager、进程间传递、Scheduler、设备执行和输出返回。batch成员可以每轮变化，因此“第0行”不是请求身份。用rid关联生命周期，再记录每轮batch的request-to-row映射。

| 源码入口 | 原理对应 | 建议观察 |
|---|---|---|
| [TokenizerManager](../../../sglang/python/sglang/srt/managers/tokenizer_manager.py)，`generate_request` | 接收、规范化、分发、等待响应 | rid、tokenized输入、采样参数、接收/返回事件 |
| [Scheduler](../../../sglang/python/sglang/srt/managers/scheduler.py)，`handle_generate_request` | 请求进入可调度状态 | 队列及请求状态 |
| 同文件`get_next_batch_to_run` | 动态组批和状态选择 | batch序号、forward mode、成员及长度 |
| 同文件`run_batch` | 下发当前设备工作 | CPU提交时刻、设备关联ID |
| [结果处理器](../../../sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py) | 消费结果、追加输出、判断结束 | copy_done、output_ids、finished/retracted |
| [KV释放入口](../../../sglang/python/sglang/srt/mem_cache/common.py)，`release_kv_cache` | 完成请求的缓存归属与回收 | committed/allocated长度、请求slot和树缓存 |
| [RadixCache](../../../sglang/python/sglang/srt/mem_cache/radix_cache.py)，`cache_finished_req` | 请求结束后前缀可能仍缓存 | 缓存插入、引用/锁、可淘汰容量 |

字段的实际所在对象以此快照为准，例如KV状态通过`req.kv`管理；本页表格不是承诺已有同名日志列。

## 从一个5-token prompt推到两个输出

用真实tokenizer选定一个输入，保存其实际token IDs；只有长度确实为5时才使用下面数字。设普通非chunked prefill、没有缓存命中：

| 逻辑事件 | 已计算KV覆盖的token数 | 已采样输出数 | 下一次需要什么 |
|---|---:|---:|---|
| 接收请求 | 0 | 0 | 入队并准入 |
| prefill模型执行完成 | 5 | 0 | 用末位置logits采样 |
| 采出g0 | 5 | 1 | 把g0作为下一次forward输入 |
| decode处理g0并采出g1 | 6 | 2 | 若未结束，再处理g1 |
| 此时结束 | 6 | 2 | 完成输出与资源清理 |

这张表描述逻辑覆盖长度。真实实现可以预分配slot、推测性增加批次长度或稍后提交元数据，因此某个内部计数在CPU提交时就可能改变。应同时观察“分配”“计划执行”“已完成并提交”，不能用单个字段在任意时刻强行匹配表格。

若新生成的g1触发停止条件，不代表g1已经写入KV。类似地，cache命中会减少本次实际计算量；输入总长仍是5，不必与本次extend token数相同。

## normal和overlap的差别落在哪里

`event_loop_normal`明确按接收请求、选batch、`run_batch`、`process_batch_result`和更新last_batch推进。`event_loop_overlap`维护`result_queue`，让CPU处理与设备计算在受控依赖下重叠。

在当前decode结果处理里，若`result.copy_done`存在，会先调用`synchronize()`，再规范化输出并追加到`req.output_ids`。overlap路径还跳过已经finished或retracted的请求。这个检查意味着回传结果不必对应“此刻仍活跃的请求”。

因此应分开记录：host调用返回、GPU计算完成、结果复制完成、请求状态提交、用户收到字节。这五个时刻不必相等。采集profile时，也不能用一个Python函数的持续时间代替设备Kernel的持续时间。

## 哪些证据现成可用，哪些需要补点

slime已有sample trace，可记录`sglang_generate`和奖励过程，并附带SGLang返回的元信息。它通常覆盖客户端视角的请求区间，不能还原所有GPU内部状态。现有说明见[Trace指南](../../zh/developer_guide/trace.md)。

若保存了自己的rollout dump，可以生成离线页面：

```bash
python tools/trace_timeline_viewer.py /tmp/slime-case2/rollout_0.pt --no-serve
```

已有[profile_rollout.py](../../../tools/profile_rollout.py)可以向运行中的引擎请求profile。地址必须替换为本次实验的router，下面不负责启动服务：

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action start --num-steps 3
```

在开始采集后发送预定请求，再保留工具报告的实际trace路径。使用方法见[Profiling指南](../../zh/developer_guide/profiling.md)。为复原KV账本，还需在隔离调试运行中用调试器或临时日志观察上述Scheduler/KV入口。当前sample trace没有完整页表，所以不能仅运行viewer就宣称页表已经验证。

## 事件记录如何组织

每条记录保留run_id、rid、进程/rank、batch序号、forward mode、clock_domain，以及相关长度和状态。记录可人工整理，不能把示意事件伪装为现有trace导出格式。

| 事件 | 可用于哪个结论 | 不能单独证明 |
|---|---|---|
| tokenizer接受rid | 请求开始进入服务端 | 已获得GPU执行资格 |
| scheduler选中rid | 本轮计划执行 | Kernel已经完成 |
| GPU区间结束 | 该次设备计算结束 | 输出已发给用户 |
| result copy完成 | CPU结果可消费 | 整条请求已结束 |
| finish处理与cache释放入口 | 请求清理已进入实现路径 | 所有物理KV已归零 |
| 客户端收到首token/结束 | 用户延迟 | 服务器内部是哪一段造成等待 |

跨进程clock需要校准或显式关联；跨机器不能直接相减未同步的单调时钟。没有足够事件时，应把原因标为未定位。

## 完成请求为什么不等于物理池占用归零

`release_kv_cache`先依据effective committed长度调用`cache_finished_req`，随后处理多分配尾部、请求slot及相关状态。缓存插入可能保留可复用前缀，所以请求数下降而缓存仍占用并不自动是泄漏。

核对时分别记录：请求slot占用、活跃KV、缓存保留/可淘汰量、allocator余量。对照[六页手工推演](<../10_Inference_Serving/推理内存系统.md#数值展开-kv分页前缀共享分叉与取消>)理解守恒关系，但不要把教学页号p0到p5映射成引擎的真实物理地址。

## 等待中的取消与执行中的取消

当前实现对 waiting queue 中的请求可以删除匹配项；已经进入运行的请求则设置 `to_finish=FINISH_ABORT()`，沿结果处理路径清理。区别来自在途 batch：CPU 接口返回时，GPU 可能仍持有对应地址。

因此“abort 已返回”不能证明资源已释放。rid、batch 身份、finish 状态与最后消费者一起定义释放边界。当前匹配包含 rid 前缀或 abort_all 语义，前缀重叠会扩大匹配范围。若把一轮 batch 的行号误绑定到下一轮复用同一行的新 rid，即使行号合法，请求身份也已经错了。

## 功能组合算例：前缀复用与 adapter 身份

另设一个独立教学负载：每页 16 token，4 个请求各有 64-token 公共文本前缀、16-token 私有 prompt 和 32-token 输出；每次完整执行的 KV 共 7 页。假定前缀页对齐且可命中，不计元数据、碎片、图池和在途空间。

| 条件 | 公共前缀页 | 私有 prompt + 输出页 | 总页数 |
|---|---:|---:|---:|
| 相同 adapter，未共享 | 4×4=16 | 4×3=12 | 28 |
| 相同 adapter，共享一个前缀 | 4 | 12 | 16 |
| 两个 adapter，各有两个请求，各共享一份前缀 | 2×4=8 | 12 | 20 |

最后一行不能按文本相同直接算成 16 页：adapter 改变计算，KV 身份必须包含其版本。多模态内容也不能只凭文本 placeholder 判定相同。命中节省的是可复用前缀的执行和缓存，不自动改变 decode 步数；chunk 与 graph 影响调度和执行形状，不再重复计算同一份容量收益。

核对实现时记录 adapter identity、cache key、hit token 数、物理页与引用数，以及 Graph bucket 的 padding 和额外常驻内存。支持组合需另核对，页数表不能证明特定引擎已支持 Multi-LoRA + prefix + graph。通用身份规则见[高级服务功能](../10_Inference_Serving/结构化生成与多模型服务.md)，组合约束见[执行优化](../10_Inference_Serving/执行优化与功能组合.md)。

## 生命周期与实际时间的区别

同一 rid 的接收、准入、执行、输出与结束形成因果链。KV 逻辑长度和分配长度不同，结束后状态也可能转为可复用缓存，而不是立即归零。只有活跃引用失去消费者之后，回收才安全。

normal 与 overlap 的源码顺序不能直接当成实测时间线。本文解释控制流与状态，没有提供 GPU trace 或服务延迟测量；固定请求与采样下的数值对齐，也不能单独证明时延改善。

[跨层案例](贯穿案例索引.md) · [推理原理](../10_Inference_Serving/README.md)

<!-- series-nav -->

[上一篇：贯穿案例一：一组Rollout如何变成一次参数更新](<贯穿案例_从Rollout到参数更新.md>) · [专题目录](<README.md>) · [下一篇：贯穿案例三：一次分布式训练究竟在等待什么](<贯穿案例_分布式训练等待与通信重叠.md>)

<!-- /series-nav -->
