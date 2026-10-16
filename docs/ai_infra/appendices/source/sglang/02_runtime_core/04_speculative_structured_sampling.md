# 2.4 推测解码、约束输出与采样正确性

相关基础：[Transformer 与 KV](<../../../../05_Inference/19_生成执行与采样.md#s1>)。


普通自回归生成的困难是串行依赖：只有得到 token 1，才知道 token 2 的模型输入。推测解码先猜一段后缀，把“猜测条件下可并行计算的概率”交给目标模型验证；语法约束则解决另一个问题——哪些 token 在当前输出格式下允许出现。两者都必须维护临时状态与真正已接受状态的边界。

本文的 draft 指低成本草稿生成器，target 指定义最终输出规则的目标模型；greedy（贪心）每次选概率最大项，sampling（随机采样）按概率抽取。分布等价意味着重复抽样的概率规律相同，不保证固定 seed 下两种算法逐 token 相同，因为消耗随机数的顺序可能改变。

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


## 2. greedy 与 sampling 的等价性差异

greedy 下，draft token 等于 target argmax 就接受，直观简单。

随机采样下，为保持 target 分布，不能用“相等则接受”草率处理。经典 rejection sampling 对 draft 分布 `q`、target `p`：

```text
accept probability = min(1, p(x) / q(x))
```

MTP（Multi-Token Prediction，多 token 预测）在训练中增加对后续多个位置的预测目标或模块。它改变训练计算与状态；是否用于推理草稿生成，是另一个实现选择。

拒绝后从校正分布（与 `max(p-q,0)` 相关）采样。不同 EAGLE/MTP/tree verification 实现细节不同，但验收底线是输出分布与非 speculative target 等价或满足明确近似。

### 从概率质量看为什么必须校正

固定已经接受的历史，p(x) 和 q(x) 分别是 target 与 draft 在有限词表上对 token x 的条件概率，两者非负、总和均为 1，无单位。草稿先按 q 抽 x，再以 `a(x)=min(1,p(x)/q(x))` 接受；实际被抽中的候选满足 `q(x)>0`。由“抽到 × 接受”得到直接输出 x 的概率质量：

```text
q(x) × a(x) = min(q(x), p(x))
```

尚缺的 target 质量为 `p(x)-min(q(x),p(x))=max(p(x)-q(x),0)`。把全词表缺口相加得 `Z=sum_x max(p(x)-q(x),0)`，也恰好是总拒绝概率。拒绝后按 `r(x)=max(p(x)-q(x),0)/Z` 抽取，就补齐缺口：

```text
最终 P(x) = 直接接受质量 + 拒绝总概率 × 拒绝后选中 x 的概率
          = min(q(x),p(x)) + Z × r(x)
          = p(x)
```

若 Z=0，则 p=q，拒绝事件不会发生，无需计算除以零的 residual。这个推导解释经典[精确推测采样算法](https://arxiv.org/abs/2211.17192)；实际 p/q 必须对应当前历史与实际使用的温度、截断、grammar 等规则，不能拿原始 softmax 和处理后的草稿概率混算。

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

FSM 是 Finite State Machine（有限状态机），用“当前状态＋输入符号→下一状态”描述规则。一般语法 backend 也可能需要栈等更强的解析状态，并非所有 JSON/EBNF（扩展巴科斯范式）约束都只是一个小型 FSM。以只允许字符串 `ab` 为例：状态 s0 需 `a`，s1 需 `b`，s2 完成。若词表含 token `a`、`ab`、`b`，s0 允许前两项；token `ab` 会一次跨过两个字符状态，而 `b` 在 s0 非法。

对某步 logits `[2,1,3]` 施加允许集合 `{a,ab}`，得到 `[2,1,-∞]`，softmax 约为 `[0.731,0.269,0]`。mask 的作用是把非法概率压为零，再在合法集合内采样；它不证明生成内容事实正确。若所有 token 都被屏蔽，softmax 没有合法归一化分母，应按 backend 的错误/终止协议处理，不能悄悄放开约束。

## 7. grammar、overlap 与 speculative 的交叉

下一轮 allowed-token mask 依赖上一轮实际接受 token。overlap scheduler 若在 result 未处理前构造下一批，会用旧 FSM state。因此源码在特定 grammar/spec 场景加同步或禁用 overlap。

树形 speculative 下，每个 candidate branch 有自己的 grammar state；可提前剪枝非法候选，但必须与最终接受路径状态一致。

## 8. sampling 的可复现性

RNG（Random Number Generator，随机数生成器）在 draft、接受判断和 correction 中都可能被消费。相同初始 seed 不保证两种算法使用同一随机数序列来产生最终 token；分布等价与逐次样本相同是两个目标。

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

- greedy：在模型、输入、tie-breaking 与可复现数值条件一致时，检查 spec on/off token 一致；浮点近并列差异需定位后单独解释；
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

Target 验证仍遵守“位置 i 的 logits 预测 i+1”。评价 C 的概率来自历史 AB 的末位置输出，评价 D 来自假设 ABC 的末位置，评价 E 来自假设 ABCD 的末位置。输入一段候选后可以并行计算这些条件分布，但首个候选的分布仍需由历史末位置提供；具体 worker 可能携带上一轮尚未物化的 token 一起验证，不能把“候选 C 的 hidden state”误当作“在 C 出现前预测 C 的分布”。

## 11. acceptance 的概率例

取三个 token `[x,y,z]`，草稿 `q=[0.5,0.3,0.2]`，目标 `p=[0.2,0.5,0.3]`。逐项计算：

| token | 草稿抽到 q | 接受概率 a | 直接接受质量 q×a | 目标还缺多少 |
|---|---:|---:|---:|---:|
| x | 0.5 | 0.4 | 0.2 | 0 |
| y | 0.3 | 1 | 0.3 | 0.2 |
| z | 0.2 | 1 | 0.2 | 0.1 |

总接受概率为 0.7，拒绝概率 Z=0.3；拒绝后分布 `r=[0,2/3,1/3]`。最终 x/y/z 的概率是 `[0.2,0.3+0.3×2/3,0.2+0.3×1/3]=[0.2,0.5,0.3]`。如果拒绝后直接再按 p 抽，x 的概率会变为 `0.2+0.3×0.2=0.26`，已经偏离目标。这就是“看上去合理的重抽”仍可能错误的反例。

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

用候选顺序 `[C,D,E,X]` 写出“行 Query 可读列 Key”的候选内 mask；1 表示可见，0 表示屏蔽，各行还都可读已提交前缀：

```text
       C D E X
C      1 0 0 0
D      1 1 0 0
E      1 1 1 0
X      1 0 0 1
```

mask 形状为 `[4,4]`，加上长度 L 的共享前缀后逻辑范围是 `[4,L+4]`。C 的位置为 L，D/X 都是 L+1，E 是 L+2；数组中的索引 3 不代表 X 应有位置 L+3。GPU 可以并行计算这些“假定路径”上的结果，但 CPU/验证 kernel 选出路径并完成必要搬移后，只有对应 slot 能进入后续正式历史。非接受分支仍被验证 kernel 读取时，不能提前回收。

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

先固定单链算法、相同负载，并暂不考虑 EOS 截断。设每轮提出 k 个候选，A 是本轮接受的草稿 token 数，C 是本轮额外提交的 correction/bonus token 数；经典完整轮通常 C=1。基础 decode 每 token 用时为 `T_base`，推测轮的草稿、验证、采样及清理总耗时为 `T_round`，单位均为 ms。多轮总时间除以总输出数给出：

```text
平均每 token 时间 ≈ E[T_round] / E[A+C]
加速比 ≈ T_base × E[A+C] / E[T_round]
```

E 表示多轮平均，不能把“每轮时间/token 的算术平均”随意代替总量比值。假设普通 decode 为 10 ms/token，草稿 4 ms、验证 12 ms、整理 2 ms，平均 A=2、C=1，则推测是 `18/3=6 ms/token`，加速约 1.67 倍。若平均只接收 0.5 个草稿，变成 `18/1.5=12 ms/token`，反而变慢。

更理想地假定每个位置在此前都接受的条件下，以相同概率 α 被接受，则 `P(A≥i)=α^i`。利用非负整数期望 `E[A]=sum_{i=1}^k P(A≥i)`，得到 `E[A+1]=1+α+…+α^k`；α=0.5、k=3 时为 1.875。增加第 4 个候选只多贡献 0.0625 个预期输出，却继续增加 draft、verify 和临时 KV 成本。真实接受率随历史变化，树算法和提前结束也会改变 C，所以应从实际提交数估计收益。

## 16. 本章延伸阅读

- [Speculative Decoding 论文](https://arxiv.org/abs/2211.17192)：严格 rejection sampling 与分布保持。
- [EAGLE 论文](https://arxiv.org/abs/2401.15077)：feature-level autoregression 与 tree candidates。
- [SGLang Speculative Decoding 官方文档](https://docs.sglang.io/docs/advanced_features/speculative_decoding)：当前算法与参数。
- [SGLang v0.4 博客](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)：XGrammar 与 overlap scheduler 的交互背景。
- [XGrammar 论文/仓库](https://github.com/mlc-ai/xgrammar)：结构化输出的 grammar compilation 与高效 mask。
- [DSpark SGLang 博客](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/)：最新 variable-length verify 的调度性能模型。
