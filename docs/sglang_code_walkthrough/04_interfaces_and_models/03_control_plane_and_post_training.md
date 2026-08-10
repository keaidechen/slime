# 4.3 控制面、在线权重更新与后训练

## 1. 控制面不是普通推理流量

Native API 中的 health、server/model info、flush cache、更新权重和 LoRA 管理会改变服务状态：

- health 与 readiness 区分“进程存活”和“可以接请求”；
- flush cache 必须与在途请求和 radix lock 协调；
- 权重/LoRA 更新必须保证所有 rank 版本一致；
- 失败不能留下部分 rank 已更新的状态；
- model info 应用于客户端能力协商，不能靠猜测。

## 2. 后训练版本闭环

```text
policy version N
 -> rollout requests
 -> tokens / logprobs / rewards
 -> optimizer update
 -> weight transfer or reload
 -> policy version N+1
```

每条 rollout 记录 policy version、tokenizer/chat template、采样参数、截断和取消原因。同一训练 batch 混入不同版本样本会静默破坏 on-policy 假设。

## 3. 更新边界与一致性

权重更新有三层一致性：

1. **参数完整性**：名称、shape、dtype 和分片匹配；
2. **分布式一致性**：所有 rank 成功后才发布新版本；
3. **请求一致性**：单个请求不能跨两个 policy version 执行。

失败路径要能回滚或让整个实例退出 ready，不能继续以未知混合版本服务。

## 4. Cache 与 Graph 的派生状态

更新权重后，旧 KV 是否可复用取决于模型语义；通常不能把旧 policy 的 KV 当成新 policy 的结果。CUDA Graph、编译缓存、LoRA slot 和量化派生 buffer 也需要明确失效或重建策略。

## 5. RL 请求生命周期

RL rollout 常包含高并发、批量 abort、外部 reward、部分 rollout 和动态权重更新。除正常 finish 外，必须测试 trainer 取消、engine timeout、worker crash、版本切换和权重传输失败，并确认 request slot、KV 和传输 buffer 全部回收。

## 6. 延伸阅读

- `sglang/docs_new/docs/advanced_features/sglang_for_rl.mdx`
- `sglang/docs_new/docs/advanced_features/checkpoint_engine.mdx`
- `sglang/docs_new/docs/references/post_training_integration.mdx`
- `docs/code_walkthrough/02_rollout_sglang_server_mode.md`
- `docs/code_walkthrough/04_weight_sync_and_memory.md`

