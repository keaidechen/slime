# 06. 推测解码、约束输出与采样正确性

## 1. 推测解码的目标

decode 每步受模型权重读取和 launch/通信 latency 限制。用更便宜的 draft 一次提出多个 token，再让 target 一次验证，可减少 target forward 次数。

典型流程：

```text
draft 提议 d1...dk
target 并行计算这些位置的分布
按算法接受最长合法前缀
在第一个拒绝位置重采样
提交 accepted tokens 与一个 target token
回滚未接受部分的临时 KV/状态
```

加速约为“平均每次 target forward 接受的 token 数”与额外 draft/verify 开销的平衡，不是 draft steps 越多越快。

## 2. greedy 与 sampling 的验收差异

greedy 下，draft token 等于 target argmax 就接受，直观简单。

随机采样下，为保持 target 分布，不能用“相等则接受”草率处理。经典 rejection sampling 对 draft 分布 `q`、target `p`：

```text
accept probability = min(1, p(x) / q(x))
```

拒绝后从校正分布（与 `max(p-q,0)` 相关）采样。不同 EAGLE/MTP/tree verification 实现细节不同，但验收底线是输出分布与非 speculative target 等价或满足明确近似。

## 3. 树形验证与 KV

EAGLE 等可构建候选树而非单链。target attention 需让每个候选只看到其祖先路径，随后从验证结果选接受路径。

系统难点：

- tree token 的 position；
- tree attention mask；
- 候选 token→KV slot；
- 接受路径 materialize；
- 未接受 branch 的 KV 释放；
- radix cache 使用 bigram/extra key 时的 namespace；
- CUDA graph capture shape；
- batch 内每请求接受长度不同。

若只修 token 输出而漏清 KV，错误通常在后续若干步才出现。

## 4. SGLang 代码入口

`srt/speculative/` 下按算法组织 worker，例如 EAGLE、MTP、ngram、DFlash。普通 `TpModelWorker` 与 speculative worker 都向 scheduler 提供 generation/verify 结果，但 result metadata 更丰富：

- draft tokens；
- verified token ids；
- accepted length；
- temporary KV positions；
- hidden states 或 draft model inputs。

学习时先跑 ngram/简单线性 proposal，再读 EAGLE V2；不要从最大模型、多层 draft、DP attention 组合起步。

## 5. acceptance 指标

至少报告：

- mean accepted length；
- 每请求/长度桶分布；
- draft time、target verify time、sampling time；
- target forward calls/output token；
- 端到端 TPOT 与 TTFT；
- accuracy/distribution regression；
- 额外 KV/graph memory。

高 acceptance 不等于高性能：draft 太贵或 verify batch 太稀时仍可能变慢。

## 6. Structured Output

JSON schema、regex、EBNF 等被编译成 FSM/grammar state。每步：

1. 根据当前状态得到允许 token；
2. 对 logits 应用 mask；
3. sampling；
4. 用选中 token 推进状态。

实现位于 `srt/constrained/`，支持不同 backend。难点不是字符串正则本身，而是 tokenizer token 可能跨字符/byte 边界，一个 token 可让 FSM 前进多步或非法。

## 7. grammar、overlap 与 speculative 的交叉

下一轮 allowed-token mask 依赖上一轮实际接受 token。overlap scheduler 若在 result 未处理前构造下一批，会用旧 FSM state。因此源码在特定 grammar/spec 场景加同步或禁用 overlap。

树形 speculative 下，每个 candidate branch 有自己的 grammar state；可提前剪枝非法候选，但必须与最终接受路径状态一致。

## 8. sampling 的可复现性

动态 batching 会改变请求在 batch 中的位置与 kernel 执行形态。若 RNG 只按 batch slot 推进，其他请求到达/结束会改变某请求输出。更稳健的设计按 request 维护 seed/offset 或 counter-based RNG。

严格 deterministic 还受：

- reduction 顺序；
- attention backend；
- TP world size；
- FP8/量化；
- speculative proposal；
- prefix cache 命中路径；
- CUDA graph/eager；
- torch/NCCL 版本。

生产应声明可复现范围，不能笼统承诺“同 seed 必然字节级相同”。

## 9. 正确性测试

- greedy：spec on/off token 完全一致；
- sampling：大量样本做分布/统计检验；
- EOS/stop/max tokens 恰在 draft 中间；
- grammar 在候选分叉处；
- abort/retract 后临时 KV 全释放；
- prefix cache 命中与未命中；
- batch 中 accepted length 分别为 0、1、k；
- TP/DP attention/CUDA graph 组合。

## 10. 线性 speculative 的完整状态表

假设 committed 序列是 `[A,B]`，draft 提议 `[C,D,E]`：

| 阶段 | token 状态 | KV 状态 |
|---|---|---|
| 开始 | AB committed | KV(AB) committed |
| draft | CDE temporary | draft KV 临时 |
| target verify | 并行评估 CDE | target 临时 slots |
| 接受 C,D，拒绝 E | ABCD committed | KV(CD) materialize |
| correction token F | ABCDF committed | KV(F) 按算法产生/下轮产生 |
| cleanup | E branch 删除 | E 临时 slots free |

`output_ids` 只应追加最终提交的 C、D、F，不能先 append CDE 再原地把 E 改成 F，因为 `Req` 契约是 append-only。

## 11. acceptance 的概率例

draft 对 token x 给 `q(x)=0.5`，target 给 `p(x)=0.2`：

```text
accept prob = min(1, 0.2/0.5) = 0.4
```

若 target 更偏好 x，例如 p=.6、q=.3，则必接受。拒绝后必须从校正后的 residual 分布采样，才能保持 target 边缘分布。

工程实现通常在 GPU 上融合 top-k、概率、随机数和 tree verify；对照 reference 时用小词表 CPU 实现验证统计，而不是只测 greedy。

## 12. tree attention mask

候选树：

```text
        C
      /   \
     D     X
     |
     E
```

E 可看 committed prefix、C、D，不能看 sibling X；X 可看 prefix、C，不能看 D/E。若把候选简单拼成 `[C,D,E,X]` 用 causal mask，X 会错误看到 D/E，需要显式 tree mask/position mapping。

接受 C→D 后，只 materialize 该路径 KV；X/E 未接受分支释放。radix key 若使用 EAGLE bigram view，也需按 accepted path 更新。

## 13. grammar token mask 的实现成本

词表 150k 时，每 request 每 step 构造 dense boolean mask 很贵。常见优化：

- FSM 状态缓存允许 token 集；
- compressed FSM；
- GPU token bitmask；
- batch 内相同 grammar state 共享；
- jump-forward：确定字符串片段无需逐 token 模型采样。

SGLang v0.4 团队博客介绍了 XGrammar 集成与结构化输出加速。性能测试要区分 grammar compile time 与 steady decode time。

## 14. RNG 的 request-local 设计

理想 counter-based RNG key：

```text
(request_seed, committed_token_index, sample_substep)
```

这样 batch slot 从 3 变 7 不改变随机数。speculative 又会消耗 draft/acceptance/correction 随机数，必须定义哪些 substep 属于 target 输出语义。

对 deterministic test，记录：

- request seed；
- 每步 RNG offset；
- accepted length；
- graph/eager path；
- backend；
- batch membership。

## 15. speculative 性能模型

令：

```text
T_base = 每个 target decode step 时间
T_draft(k) = 产生 k 个候选
T_verify(k) = target 一次验证
A(k) = 平均接受 token 数
```

近似每输出 token：

```text
T_spec/token ≈ (T_draft(k)+T_verify(k)) / (A(k)+correction contribution)
```

k 增大时 A 通常上升但边际下降，verify token 和临时 KV 增加。最优 k 随 batch、模型、prompt domain 变化。

## 16. 本章延伸阅读

- [Speculative Decoding 论文](https://arxiv.org/abs/2211.17192)：严格 rejection sampling 与分布保持。
- [EAGLE 论文](https://arxiv.org/abs/2401.15077)：feature-level autoregression 与 tree candidates。
- [SGLang Speculative Decoding 官方文档](https://docs.sglang.io/docs/advanced_features/speculative_decoding)：当前算法与参数。
- [SGLang v0.4 博客](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)：XGrammar 与 overlap scheduler 的交互背景。
- [XGrammar 论文/仓库](https://github.com/mlc-ai/xgrammar)：结构化输出的 grammar compilation 与高效 mask。
- [DSpark SGLang 博客](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/)：最新 variable-length verify 的调度性能模型。
