# PyTorch Profiler

PyTorch Profiler 连接 Python/module、ATen operator、autograd、CUDA runtime 与 device kernel。它比系统级 trace 更懂 framework 语义，比 Nsight Compute 更接近模型代码，是“哪一段模型触发了哪些 kernel”的桥梁。

ATen 是 PyTorch 的底层张量算子库。

## 1. 推荐的定时采样模板

```python
import torch

schedule = torch.profiler.schedule(
    wait=2, warmup=2, active=4, repeat=1
)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./trace"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()
```

`wait` 不记录，`warmup` 记录但丢弃以稳定 profiler，`active` 输出 trace；它们不是替代模型本身的 warm-up。`prof.step()` 必须与 schedule 的逻辑 step 对齐。

## 2. 给模型加语义边界

```python
with torch.profiler.record_function("data/h2d"):
    batch = move_to_device(batch)

with torch.profiler.record_function("train/forward"):
    loss = model(batch)

with torch.profiler.record_function("train/backward"):
    loss.backward()
```

命名建议使用层级前缀和稳定 ID：`train/forward/layer_12/attention`。避免把 request ID、完整 prompt 等高基数或敏感内容写进 range。

## 3. 表格怎么读

```python
print(prof.key_averages(group_by_input_shape=True).table(
    sort_by="self_cuda_time_total", row_limit=30
))
```

| 字段 | 含义 |
|---|---|
| CPU total | operator 及其 children 的 CPU 时间 |
| Self CPU | 扣除 children 后自身 CPU 时间 |
| CUDA total | 归属该 operator 的 GPU kernel 总时间 |
| Self CUDA | 扣除 nested operator 后的 device 时间 |
| Calls | 调用次数；可暴露碎片化小算子 |
| Input Shapes | 区分同名算子的不同工作量 |

异步归因、overlap 与 nested event 会让“各行相加”等于 wall time的直觉失效。表格用于找候选热点，最终关键路径仍回到 timeline。

## 4. Trace 中优先看什么

1. `ProfilerStep*` 是否包含完整、稳定 step；
2. CPU op 到 CUDA launch 再到 GPU kernel 的 flow；
3. GPU 空洞前 CPU 在 data、Python、GC 还是同步；
4. operator 是否展开成大量 pointwise/reformat kernel；
5. communication 与 backward 是否重叠；
6.不同 rank 的 phase boundary 是否错位。

GC = Garbage Collection（垃圾回收）。

## 5. Memory profiling

`profile_memory=True` 提供 operator 相关的 allocation/deallocation，但完整 OOM 根因更适合 CUDA memory snapshot：

```python
torch.cuda.memory._record_memory_history(max_entries=100000)
# run workload
torch.cuda.memory._dump_snapshot("snapshot.pickle")
```

snapshot 可在 PyTorch memory visualizer 中查看 allocation timeline 与 allocator segments。注意它主要看 PyTorch allocator 已知的分配；NCCL、CUDA context 或自定义 library 直接分配的显存可能不可见。

Mosaic memory profiler 可离线组合多 worker snapshot，适合“只有某些 rank OOM”或长时间运行后的泄漏/峰值分析。官方教程提到它曾用于 LLaMA 405B 规模任务的 OOM 调试。

## 6. `torch.compile` 特别注意

编译前后 operator 名与 kernel 数量会改变：fusion 后多个 eager operator 对应一个 generated kernel。应分开记录：

- compile/tracing time；
- steady-state execution time；
- graph break 与 recompile 次数；
- generated kernel 的 shape specialization；
- eager/compiled correctness 与 peak memory。

动态 shape 导致的偶发 recompile 可能成为 p99 抖动，而平均稳态看不出来。

## 7. 控制开销

- 只采少数 step；
- 先关 `with_stack`、`record_shapes`、`profile_memory` 得到轻量 trace，再按问题打开；
- 多 rank 先采代表 rank；
- trace 写本地 SSD，再异步归档；
- 不在同时段叠加多个基于 CUPTI（CUDA Profiling Tools Interface，CUDA 性能分析工具接口）的 profiler。

## 8. 工具分工

| 问题 | 首选 |
|---|---|
| 哪个 PyTorch op 最贵 | PyTorch Profiler |
| CPU/GPU/通信怎样重叠 | Nsight Systems |
| 某 kernel 为何慢 | Nsight Compute |
| 显存 allocation 演化 | CUDA memory snapshot / Mosaic |
| 集群长期趋势 | DCGM + Prometheus |

## 资料

- [PyTorch Profiler API](https://docs.pytorch.org/docs/stable/profiler.html)
- [Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Accelerator Profiler Integration](https://docs.pytorch.org/docs/stable/accelerator/profiler.html)
- [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [Mosaic Memory Profiling](https://docs.pytorch.org/tutorials/beginner/mosaic_memory_profiling_tutorial.html)
