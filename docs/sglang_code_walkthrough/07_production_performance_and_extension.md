# 07. 生产性能、可观测性、排障与扩展

## 1. Benchmark 必须描述 workload

`python -m sglang.bench_serving` 支持 request rate、concurrency、数据集、streaming 和多种指标。可靠实验至少固定：

- 模型、dtype/quantization、context；
- GPU/互联/驱动/CUDA/版本；
- 输入/输出长度分布与共享前缀比例；
- arrival process（burst、Poisson、trace replay）；
- request rate/max concurrency；
- sampling/structured/spec 配置；
- warmup、持续时间、重复次数；
- TTFT/ITL/TPOT/E2E 分位数、吞吐和失败率。

离线 all-at-once 测的是饱和吞吐，不代表在线 SLO。

### 找容量拐点

逐步提高 request rate，画：

```text
x: offered load
y1: achieved throughput
y2: p99 TTFT
y3: p99 ITL
y4: queue depth / KV occupancy
```

吞吐开始平台化而延迟陡增处就是过载区。生产 admission limit 应留余量。

## 2. 观测四层

### API/路由层

QPS、状态码、client cancel、queue、rate limit、路由/cache affinity。

### Scheduler 层

waiting/running requests、prefill/decode tokens、retract、cache hit、KV free、batch size、forward occupancy。

### Model/通信层

forward latency、attention/MoE/sampling、CUDA graph hit、NCCL/A2A、kernel fallback。

### 系统层

GPU SM/HBM、host CPU、RSS/pinned memory、NIC/RDMA、文件描述符、模型存储。

OpenTelemetry trace 要用 request id 连接 tokenize→queue→prefill→KV transfer→decode→detokenize，而不是只包 HTTP handler。

## 3. 常见症状

| 症状 | 优先假设 |
|---|---|
| TTFT 高，ITL 正常 | prefill queue、长 prompt、prefix miss、tokenizer |
| TTFT 正常，ITL 抖 | prefill 干扰、decode batch 变化、CPU scheduler、通信 |
| KV free 持续下降 | abort/finish/retract 漏释放、radix lock 泄漏 |
| GPU 低利用率且 queue 高 | CPU scheduling/tokenize、过小 batch、graph miss、同步 |
| 吞吐高但用户慢 | batch 过大、只优化均值、排队失控 |
| 只在 TP/DP hang | collective 次序、空 rank、phase 决策不一致 |
| 输出偶发错误 | KV index/复用 race、overlap hazard、cache namespace |
| graph 开启后 OOM | capture pool/workspace、capture size 过密 |

## 4. profiling 顺序

1. 请求 stage timestamps 确定慢在控制面还是 GPU；
2. scheduler iteration 日志看 batch/token/KV；
3. PyTorch profiler 看 operator/CPU gap；
4. Nsight Systems 看 stream、graph、NCCL；
5. 锁定 kernel 后才用 Nsight Compute；
6. 内存问题用 allocator/pool invariants，不只看 `nvidia-smi`。

短 trace、稳定 workload、明确 request ids 比长时间全量 trace 更有用。

## 5. 新模型接入

常见入口在 `srt/models/` 与 model registry。步骤：

1. 识别 HF config architecture；
2. 实现模型/layer forward，复用 SGLang parallel Linear、attention、norm；
3. weight loader 映射 HF key 到 TP/EP shard；
4. 配置 attention backend、RoPE、GQA/MLA、sliding window；
5. 注册 architecture；
6. greedy logits 与 HF reference 对齐；
7. 测 TP、CUDA graph、quantization、logprob、prefix cache；
8. 做 accuracy 与 serving benchmark。

### weight loader 的关键

fused QKV/gate-up 常把多个 checkpoint tensor pack 到一个参数。TP shard axis 可能因 tensor 类型不同：

- column-parallel：通常切 output axis；
- row-parallel：切 input axis；
- Q/K/V 在 GQA 下每 rank 数量不对称；
- MoE 还按 expert id/EP rank 过滤。

不要先 load 完整权重再切，超大模型会瞬间 OOM；应流式读取目标 shard。

## 6. 新 attention backend 接入

必须定义 metadata 生命周期与所有 forward mode，建立 reference：

- PyTorch eager 小 shape；
- 不同 batch/seq/page；
- causal/sliding/packed；
- MHA/GQA/MLA；
- prefill/decode/spec verify；
- graph/eager；
- KV FP16/BF16/FP8；
- TP/DP。

数值容差要按 dtype 和累加路径设定，并覆盖长序列误差。

## 7. 生产可靠性

- readiness 等 model load、warmup、graph capture、distributed init；
- watchdog 能区分慢 batch 与死锁；
- client abort 贯穿所有进程和 PD transfer；
- worker crash 后路由停止新流量并清理/重启；
- weight update 有版本与 barrier，不能一半 rank 新权重；
- rollout/RL 场景更新权重前处理在途请求；
- cache key 包含影响 KV 语义的模型/adapter/version；
- canary 同时比较 token correctness 与性能。

## 8. 最终实战

选择一个 7B 模型完成：

1. 建立短请求、长 prompt、共享前缀三套 workload；
2. 画 baseline latency-throughput 曲线；
3. 分别评估 radix cache、chunked prefill、CUDA graph、speculative；
4. 压满 KV 并注入 10% abort；
5. 用 trace 解释 p99；
6. 给出容量、admission、监控和回滚方案；
7. 对一次真实性能回归定位到具体 scheduler 决策、collective 或 kernel。

做到“能解释、能复现、能回滚”，才算完成从使用者到 infra 工程师的转变。

## 9. Benchmark 命令拆解

示例：

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 256 \
  --random-range-ratio 0.2 \
  --num-prompts 2000 \
  --request-rate 20 \
  --max-concurrency 128 \
  --output-file result.jsonl \
  --output-details
```

- `request-rate` 控制 offered load；`inf` 是 burst/offline 风格；
- `max-concurrency` 是客户端在途上限，不等同 server running batch；
- `random-range-ratio` 避免所有 shape 完全固定但也要记录真实分布；
- `output-details` 用于按长度分桶，不能只保存聚合均值；
- streaming 关闭时 TTFT 语义会变化。

先测低负载 latency floor，再逐步加压到饱和。直接 burst 只能得到吞吐上界。

## 10. 指标因果链

```text
TTFT =
  API/tokenize
  + scheduler queue
  + prefix lookup/restore
  + prefill forward
  + sample/D2H
  + detokenize/network

ITL =
  decode queue/schedule
  + decode forward/collective
  + sample
  + stream processing
```

把 TTFT 只归因于 prefill kernel 会遗漏高负载下最大的 queue time。trace 中每段要用 monotonic clock，并避免跨主机未校时的 wall clock 直接相减。

## 11. 新模型权重映射例

HF 可能存：

```text
q_proj.weight [q_out, hidden]
k_proj.weight [kv_out, hidden]
v_proj.weight [kv_out, hidden]
```

SGLang 参数可能 fused：

```text
qkv_proj.weight [q_out + k_out + v_out, hidden] 的 TP shard
```

loader 不能简单先 concat 再 `chunk(TP)`，因为 GQA 下 Q/K/V 输出大小不同。正确流程：

1. 分别按各自 head/group 规则求本 TP rank slice；
2. load q local、k local、v local；
3. 写入 fused parameter 的对应 shard_id；
4. 对 bias/scale 同样映射；
5. 检查 global coverage。

用一个小 tensor 填连续整数，可以肉眼验证每 rank 取到的区间。

## 12. correctness 分层

### 层 1：算子

小 shape 与 PyTorch reference，覆盖 dtype/page/mask。

### 层 2：模型

固定 prompt greedy logits/token 与 HF 或可信 backend 对齐。

### 层 3：runtime

continuous batch、prefix hit、abort、graph/eager 下同请求语义一致。

### 层 4：分布式

TP/EP/PD 与单卡/统一部署对齐。

### 层 5：API

chat template、stop、stream、usage、logprob、错误码符合契约。

性能优化必须通过全部相关层，而不是只看 benchmark 无 crash。

## 13. 线上容量保护

常用 admission signals：

- waiting token budget；
- running KV token；
- predicted request total length；
- TTFT/ITL SLO debt；
- PD transfer queue；
- per-replica cache affinity；
- host pinned/remote cache pressure。

只按 request count 限流会把 1-token prompt 与 100k prompt 等价。更合理的是 token-aware cost，并为 decode 已运行请求预留容量，防止新 prefill 抢空 KV。

## 14. 回归二分

版本回归时固定 workload，逐层关闭：

```text
speculative
structured output
radix cache
overlap scheduler
CUDA graph / torch.compile
custom attention/MoE backend
TP/EP
```

若 `--disable-overlap` 恢复正确性，重点查 batch snapshot、future token 和 stream barrier；若 `--disable-radix-cache` 恢复，查 extra_key、lock/free 和 page alignment；若 eager 正确 graph 错，查 fixed buffer/dummy slot/shape selection。

## 15. 一份合格的性能报告

必须附：

- 两端 commit/container/driver；
- 完整 server 与 client 命令；
- 模型权重/quantization；
- hardware topology；
- workload 原始长度与到达 trace；
- warmup/运行时长/失败数；
- p50/p95/p99 TTFT/ITL/E2E；
- input/output/total tok/s；
- KV/cache/graph/backend 指标；
- 至少一张系统 trace 支持结论；
- correctness/accuracy guard。

## 16. 本章延伸阅读

- [Bench Serving 官方指南](https://docs.sglang.io/docs/developer_guide/bench_serving)：参数、指标与 JSONL schema。
- [Benchmark and Profiling](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling)：官方 profiling 工作流。
- [Production Metrics](https://docs.sglang.io/docs/references/production_metrics)：Prometheus 指标定义。
- [SGLang Llama Serving 博客](https://www.lmsys.org/blog/2024-07-25-sglang-llama3/)：离线/在线、模型大小与系统优化收益的对比。
- [SGLang v0.3 博客](https://www.lmsys.org/blog/2024-09-04-sglang-v0-3/)：从新架构/torch.compile 接入看性能验证。
- [Little's Law](https://en.wikipedia.org/wiki/Little%27s_law)：用到达率、在途量、停留时间做容量 sanity check。
