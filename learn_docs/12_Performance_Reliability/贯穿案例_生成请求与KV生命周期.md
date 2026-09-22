# 贯穿案例二：一个生成请求如何穿过队列、GPU与KV生命周期

> 证据：SGLang本地快照源码已核对，未启动生成服务或采集GPU trace。基线为普通文本、单个TP组、PP=1、无PD/推测/LoRA/Mamba；先关闭overlap观察，再只打开overlap比较。高级路径不能直接套用基线。

## 请求身份比batch行号更稳定

一条请求会经历HTTP/TokenizerManager、进程间传递、Scheduler、设备执行和输出返回。batch成员可以每轮变化，因此“第0行”不是请求身份。用rid关联生命周期，再记录每轮batch的request-to-row映射。

| 源码入口 | 原理对应 | 建议观察 |
|---|---|---|
| [TokenizerManager](../../sglang/python/sglang/srt/managers/tokenizer_manager.py)，`generate_request` | 接收、规范化、分发、等待响应 | rid、tokenized输入、采样参数、接收/返回事件 |
| [Scheduler](../../sglang/python/sglang/srt/managers/scheduler.py)，`handle_generate_request` | 请求进入可调度状态 | 队列及请求状态 |
| 同文件`get_next_batch_to_run` | 动态组批和状态选择 | batch序号、forward mode、成员及长度 |
| 同文件`run_batch` | 下发当前设备工作 | CPU提交时刻、设备关联ID |
| [结果处理器](../../sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py) | 消费结果、追加输出、判断结束 | copy_done、output_ids、finished/retracted |
| [KV释放入口](../../sglang/python/sglang/srt/mem_cache/common.py)，`release_kv_cache` | 完成请求的缓存归属与回收 | committed/allocated长度、请求slot和树缓存 |
| [RadixCache](../../sglang/python/sglang/srt/mem_cache/radix_cache.py)，`cache_finished_req` | 请求结束后前缀可能仍缓存 | 缓存插入、引用/锁、可淘汰容量 |

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

slime已有sample trace，可记录`sglang_generate`和奖励过程，并附带SGLang返回的元信息。它通常覆盖客户端视角的请求区间，不能还原所有GPU内部状态。现有说明见[Trace指南](../../docs/zh/developer_guide/trace.md)。

若保存了自己的rollout dump，可以生成离线页面：

```bash
python tools/trace_timeline_viewer.py /tmp/slime-case2/rollout_0.pt --no-serve
```

已有[profile_rollout.py](../../tools/profile_rollout.py)可以向运行中的引擎请求profile。地址必须替换为本次实验的router，下面不负责启动服务：

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action start --num-steps 3
```

在开始采集后发送预定请求，再保留工具报告的实际trace路径。使用方法见[Profiling指南](../../docs/zh/developer_guide/profiling.md)。为复原KV账本，还需在隔离调试运行中用调试器或临时日志观察上述Scheduler/KV入口。当前sample trace没有完整页表，所以不能仅运行viewer就宣称页表已经验证。

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

核对时分别记录：请求slot占用、活跃KV、缓存保留/可淘汰量、allocator余量。对照[六页手工推演](../10_Inference_Serving/推演_KV分页共享与取消.md)理解守恒关系，但不要把教学页号p0到p5映射成引擎的真实物理地址。

## 可控取消实验

在独立服务上做两种取消，并各保留rid：第一种在waiting queue，第二种在已经开始decode之后。当前`abort_request`会从等待队列删除匹配请求；运行中的请求则设置`to_finish=FINISH_ABORT()`，再沿已有结果处理路径清理。

不能以“abort接口已返回”为唯一释放证明。应观察最后在途batch、finish状态与资源回收边界。当前匹配使用rid前缀或abort_all语义，因此实验rid应避免彼此前缀重叠，不要随手取消其他实验请求。

离线反例：把同一batch的行号错绑到下一轮复用该行的新rid，应被rid与batch映射检查识别。运行时反例不要直接制造非法KV写入；取消实验已能安全暴露错误的即时释放假设。

## 完成标准与结论边界

成功记录同一rid的接收、准入、两轮执行、两次输出与结束/取消；解释KV逻辑长度与分配长度的区别；确认清理后活跃引用消失或转为有依据的缓存保留。对比normal/overlap时固定请求和采样条件，数值与输出对齐先通过，再报告时延变化。

本轮已核对上述控制流；没有请求事件文件、GPU trace或服务延迟测量。取得这些文件后才能填写本案例的实测结论，而不是把源码顺序当实际时间线。

---

[四个贯穿案例](贯穿案例索引.md) · [推理运行时](../10_Inference_Serving/README.md)
