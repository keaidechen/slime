# 02 Rollout 子系统：Server 模式的 SGLang 推理是如何生成训练数据的

> 对应综述（`00_rl_infra_survey.md`）§2.2「Rollout 架构：Engine 模式 vs Server 模式」。
> slime 是纯 server 模式框架：推理是独立的 HTTP 服务，训练侧通过 router 访问。本篇逐层解读这条链路。

---

## 1. 三个层次

```
编排层  slime/ray/rollout.py        RolloutManager / RolloutServer / ServerGroup
        ─ 起停 server、起 router、取数据、打包训练 batch、容错
函数层  slime/rollout/sglang_rollout.py
        ─ 默认 rollout 函数：异步生成循环、RM 打分、过滤、abort 回收
引擎层  slime/backends/sglang_utils/
        ─ SGLang server 进程封装、server_control（abort/显存）、引擎配置
```

数据流向：`RolloutManager.generate()` → （函数层）对 router 发 HTTP `/generate` → （引擎层）SGLang server 采样 → 带 token 级 logprob 的响应 → 填回 `Sample` → 攒够 `rollout_batch_size` 组 → 转换成训练数据。

---

## 2. 编排层：server 与 router 的启动

### 2.1 启动总入口

`RolloutManager.__init__`（`slime/ray/rollout.py:430` 起）里调用：

- `start_rollout_servers(args, pg)`（rollout.py:1090）：解析 `--sglang-config`（`_resolve_sglang_config`，rollout.py:1232，支持异构 server group、multi-model serving 的 YAML 配置），为每个模型启动 router，为每个 ServerGroup 创建引擎 actor；
- `_start_router(args, ...)`（rollout.py:1019）：用 `multiprocessing.Process` 启动 **sglang_router** 进程。router 是所有推理 server 前的负载均衡入口——训练侧只认 `sglang_router_ip:sglang_router_port` 一个地址；
- `ServerGroup.start_engines()`（rollout.py:150-259）：创建 `SGLangEngine` Ray actor 并 `engine.init.remote(...)`，每个 actor 内部用 `launch_server_process` 拉起 `sglang.http_server` 子进程。

**server 模式的关键收益**在这里体现：server 崩溃只影响一个 actor（有 `recover()`，rollout.py:346）；server 数量可以独立于训练规模扩缩；router 让多实例对训练侧透明。

### 2.2 RolloutManager 的远程接口

driver 与训练侧看到的是这组方法（rollout.py）：

| 方法 | 行号 | 用途 |
|---|---|---|
| `generate(rollout_id)` | 552 | 产一批训练数据（核心） |
| `eval(rollout_id)` | 567 | 评测集生成 |
| `save/load` | 578-583 | 数据源游标持久化 |
| `offload/onload/onload_weights/onload_kv` | 584-600 | 显存错峰（04 篇） |
| `get_updatable_engines_and_lock` | 533 | 权重同步时取引擎列表 + 互斥锁 |
| `recover_updatable_engines` | 601 | 容错恢复 |
| `check_weights` | 628 | 权重一致性校验（08 篇） |

---

## 3. 函数层：默认 rollout 主循环

`RolloutManager.generate` 最终调用 `args.rollout_function_path` 指定的函数，默认是：

```617:639:slime/rollout/sglang_rollout.py
def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    ...
    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output
```

注意两个设计：(1) `data_source.get_samples` 作为**函数**传入，rollout 函数决定何时取多少数据；(2) `aborted_samples`（本轮被截断的半成品）**回写**进数据源——这是 partial rollout 的关键一环（03 篇详解）。

### 3.1 异步生成主循环 `generate_rollout_async`

```408:448:slime/rollout/sglang_rollout.py
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()
            ...
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                ...
                continue
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)
    ...
    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)
```

机制解读：

- **over-sampling**：每次取 `over_sampling_batch_size` 组提交（sglang_rollout.py:411），在飞任务可以多于目标数 `rollout_batch_size`——先完成先收，收够即停，多出来的请求直接 abort。这就是综述 §2.4 讲的"主动超投抗长尾"（APRIL 同款思路的轻量版）。
- **`asyncio.wait(..., FIRST_COMPLETED)`**：乱序完成，不被最慢样本阻塞。
- **dynamic filter**：每组完成后过 `--dynamic-sampling-filter-path`（如 DAPO 的"全对全错组丢弃"），被丢弃的组会扣减 `remaining_batch_size` 触发补采（sglang_rollout.py:429-433）。
- 收够目标后调用 `abort` 清尾（见 §3.4）。

### 3.2 全局状态单例 `GenerateState`

```84:118:slime/rollout/sglang_rollout.py
class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        ...
        self.semaphore = asyncio.Semaphore(args.sglang_server_concurrency * get_rollout_num_engines(args))
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            ...
        )
```

三个值得注意的点：

1. **并发上限**：`sglang_server_concurrency × 引擎数` 的信号量，防止把 server 打爆；
2. **top-p 回放**：`rollout_top_p != 1.0` 时自动加 `custom_params={"return_top_p_token_ids": True}`（sglang_rollout.py:107-108）——让 SGLang 返回每步采样时的候选核（nucleus），训练侧算 logprob 时只在相同候选集内计算，**保证训推 logprob 可比**（05 篇的 `_build_topp_keep_mask`）；
3. **`dp_rank_context`**（sglang_rollout.py:120-130）：SGLang data-parallel 多 DP rank 时的简单均衡——记录每个 DP rank 的在飞请求数，随机挑一个最空闲的。

### 3.3 单样本生成链路

`generate_and_rm_group`（294-333）→ `generate_and_rm`（224-286）→ `generate`（153-220），层层分工：

- `generate_and_rm_group`：一个 prompt 的 N 个采样（GRPO 的 group）并发执行；若 `--group-rm`（如需要整组比较的比赛型 RM），在这里统一对整组打分；
- `generate_and_rm`：处理 partial rollout 续写样本的 loss mask（231-232 行，见 03 篇）；已完成的样本直接短路返回；分发到自定义生成函数或默认 `generate`；生成后调 RM（`async_rm` / `batched_async_rm`）；
- `generate`：拼 HTTP payload 发给 router：

```174:202:slime/rollout/sglang_rollout.py
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    ...
        payload["input_ids"] = prompt_ids
    ...
    with trace_span(sample, "sglang_generate", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}) as span:
        output = await post(url, payload, headers=headers)
```

- `return_logprob: True` 是 RL 与普通 serving 的核心区别：SGLang 会在 `meta_info["output_token_logprobs"]` 里返回**每个生成 token 的 logprob**，训练侧用它计算重要性采样比（05 篇）；
- 多模态样本改发 `image_data + text`（183-188 行）；
- **session affinity**：`sample.session_id` 存在且 router 策略为 `consistent_hashing` 时，请求头加 `X-SMG-Routing-Key`（195-199 行）——multi-turn agent 的多轮请求因此落到同一台 server，命中本地 KV cache。

响应回填：

```205:218:slime/rollout/sglang_rollout.py
    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    ...
    sample.append_response_tokens(
        args,
        tokens=new_response_tokens,
        log_probs=new_response_log_probs,
        trainable=True,
        meta_info=output["meta_info"],
        text=output["text"],
    )
```

`append_response_tokens`（`slime/utils/types.py:253-314`）会同步维护 `tokens / loss_mask / rollout_log_probs / top-p replay / 状态`，并做严格的长度校验——token 级数据对齐是 RL 正确性的生命线，03 篇详解 `Sample`。

### 3.4 中断：`abort`

```336:372:slime/rollout/sglang_rollout.py
async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []
    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True
    ...
    await abort_servers_until_idle(urls)
    ...
    if args.partial_rollout:
        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            ...
            aborted_samples.append(group)
```

三步：(1) 置全局 `aborted` 标志，新的 `generate_and_rm` 进入时直接标记 ABORTED 返回（245-247 行）；(2) `abort_servers_until_idle`（`slime/backends/sglang_utils/server_control.py`）对所有 server 调 `/abort_request` 并等待排空——SGLang 会中断在飞请求并**返回已生成前缀**；(3) 若开了 `--partial-rollout`，把这些半成品打上 `start_rollout_id` 元数据收集起来，由 `generate_rollout` 回写 buffer，下一轮用新权重**续写**（03 篇）。

---

## 4. 引擎层：训练侧看不见的 SGLang

`slime/backends/sglang_utils/` 中：

- `sglang_engine.py`：`SGLangEngine` Ray actor。`init` 里拼 SGLang 命令行（把 `--sglang-*` 参数透传）拉起 server 子进程；并提供 RL 专用远程方法：`update_weights_from_distributed`（加入 NCCL 组收权重）、`update_weights_from_tensor`（CUDA IPC）、`pause_generation/continue_generation`、`release_memory_occupation/resume_memory_occupation`（offload）、`flush_cache`、`abort_request` 转发等。这些端点是 SGLang 为 RL 系统专门提供的（官方文档「SGLang for RL Systems」）。
- `server_control.py`：跨引擎的批量控制，如 `abort_servers_until_idle`。
- `sglang_config.py`：`--sglang-config` YAML 的解析（异构 server group、per-group 参数覆盖、PD 分离部署）。
- `external.py`：外部托管引擎（`--rollout-external`）的适配。

**为什么 slime 敢只做 server 模式？** 因为 SGLang 把 RL 需要的一切控制面（热更新、显存释放、中断、logprob 回传、top-p/routed-experts 回放）都经 HTTP 暴露了，engine 模式的内存直传优势被抹平，剩下全是 server 模式的结构性收益（隔离、弹性、负载均衡、多模型）。

---

## 5. 评测路径

`eval_rollout` / `eval_rollout_single_dataset`（sglang_rollout.py:473-614）与训练路径共用 `generate_and_rm`，差异：

- 数据来自 `--eval-*` 配置的 `EvalDatasetConfig`，每个数据集可有自己的采样参数与 per-sample `generate_function_path`（568-569 行）；
- 用 `EVAL_PROMPT_DATASET` 全局缓存（470 行）避免每个 rollout 重复加载；
- 结果按数据集聚合 rewards/truncated，由 `_log_eval_rollout_data`（rollout.py:1259）记录。

---

## 6. 小结

- server 模式 = 训练侧只面对一个 router URL + 一组 HTTP 控制端点；
- 生成主循环 = over-sampling 提交 + 乱序完成 + 动态过滤 + abort 清尾 + 半成品回 buffer；
- RL 相对普通 serving 的三个增量：`return_logprob`、top-p/routed-experts 回放、会话亲和与中断续写。
- 下一篇（03）看这些数据如何被组织成 `Sample`、如何在 buffer 中流转、以及 fully async 如何进一步解耦。
