# 5.2 模型与 Backend 扩展、正确性验证

相关基础：[对应系统的基础实践](<../../../../README.md>)。


## 入门例子：先定义一个 backend 的输入输出契约

更换 attention backend 时，数值公式相同仍可能在 mask、KV layout、page size、causal 边界和 dtype 上不兼容。列出输入 shape/stride、有效长度、缓存 owner、输出形状，以及是否写入 KV。

验证顺序：小 tensor 与参考实现对照 → 单请求 prefill/decode → 混合长度 batch → finish/abort → 多卡与特殊特性。误差阈值要匹配 dtype、累积与任务需求，不用一个固定阈值覆盖所有层。

失败的意义取决于具体输入与状态；性能回归还取决于 backend、输入与环境是否相同。数值正确并不能替代内存释放和跨请求隔离检查。

## 机制与实现

## 1. 新模型接入层次

```text
config/registry
 -> model architecture
 -> weight loader and shard mapping
 -> forward/logits or task head
 -> TP/EP partition
 -> quantization/backend/graph
 -> serving protocol
```

先实现最小 eager、单卡、标准精度路径，再逐项开启优化。一次同时修改模型、kernel、并行和量化，会让错误无法归因。

## 2. 权重映射不变量

- checkpoint 名称到参数名称一一可解释；
- fused QKV/gate-up 的拼接顺序正确；
- TP/EP shard axis 和 rank offset 正确；
- tied weight 保持共享语义；
- 未消费、重复消费和 shape 不匹配都应报错；
- dtype conversion 与量化 scale 明确。

用小 tensor 和可预测数值先验证映射，再加载完整 checkpoint。

例如存储权重 `W:[4,2]=[[1,0],[2,0],[3,0],[4,0]]`，计算约定为 `y=xWᵀ`。输入 `x=[1,0]`，完整输出应为 `[1,2,3,4]`。按输出维分给两个 rank 后，rank 0 应得到前两行、rank 1 得到后两行，结果拼接仍是原顺序。若两 rank 都错加载了前两行，shape 检查通过，结果却变成 `[1,2,1,2]`。给行加上可识别数值可以直接看出 shard offset 错误，比随机张量只得到一个“大误差”更容易定位。

## 3. 新 Attention Backend 契约

Backend 不只是一个 kernel wrapper。它要处理 prefill/decode、paged KV 地址、GQA/MLA、causal/sliding-window mask、position/RoPE、不同 dtype、graph capture 和并行布局。

每一种 forward mode 都与参考 backend 比较输出，并覆盖非对齐长度、page boundary、不同 batch、KV 复用和 graph replay。

### 一份能手算的 Attention 验收样例

先使用一个请求、一个 head、每头维度 d=1，三位置 prefill。取 `Q=[[0],[0],[0]]`、`K=[[1],[2],[3]]`、`V=[[2],[4],[8]]`，形状都为 `[3,1]`。因果 Attention 的位置 0 只能看到第一项，输出 2；位置 1 均匀平均前两项，输出 3；位置 2 平均三项，输出 `14/3`。参考输出为 `[[2],[3],[14/3]]`。

然后只缓存前两个位置，把第三位置作为 decode 输入，Q 形状变为 `[1,1]`，历史加当前 KV 仍为 `[3,1]`，结果必须仍为 `14/3`。再把物理 KV 改放在 slot `[7,2,9]` 并同步更新映射，逻辑输出应保持不变。若得到 2，可能只读了一个位置；若 prefill 第一位置得到 `14/3`，说明未来 token 没被屏蔽。

这个例子覆盖因果边界和地址映射，但 Q 全零无法发现 K 的数值错误，因此还要加一组非零 Q 测试。测试必须知道自己的盲区，不能把一个漂亮的手算例当成全部正确性证明。

Backend 元数据可以跨层复用，但 workspace 只能在最后使用它的 kernel 完成后重用；graph replay 还要求地址与捕获契约相容。数值通过之后要再验证“取消 A、复用其 slot 给 B、A 的旧结果晚到”这种生命周期用例。

## 4. 正确性分层

| 层 | 检查 |
|---|---|
| 算子 | shape、dtype、reference tensor、边界输入 |
| 模型 | logits/hidden/embedding、权重加载、长上下文 |
| Runtime | batch、KV、stream、abort、prefix cache |
| 分布式 | TP/EP/PP/PD 与单卡结果对齐 |
| API | token id、finish reason、usage、错误码 |

准确率数据集是系统级回归，不替代低层单元测试；最终文本相同也不证明 logits、token 或资源生命周期正确。

## 5. 数值容差

容差由 dtype、backend、reduction 顺序和量化决定。报告绝对/相对误差、top-k token 排名、生成准确率和随机种子，不使用一个宽松阈值掩盖系统性偏差。

逐元素比较的常见条件是 `|y-y_ref|≤atol+rtol×|y_ref|`。`atol` 是绝对容差，与输出数值同尺度；`rtol` 是无单位的相对容差。假设 `atol=0.001,rtol=0.01`：参考值 2 的允许误差为 0.021，观测 2.015 可通过；参考值 0 的允许误差只有 0.001，观测 0.015 则失败。接近零时单独看相对误差容易失真。

也不能只看最大误差很小：两项 logits 原为 `[1.000,0.999]`，约 0.002 的扰动就可能交换 argmax，让后续自回归序列分叉。应同时检查首次差异位置、候选概率间隔、总体任务质量。对严格等价的 greedy 优化，在固定可复现条件下希望 token 一致；对浮点路径或近似量化，要声明容差和任务门槛，而不是把任何 token 差异都直接判成缓存错。

## 6. 回归二分

一次只切换一个维度：eager/graph、单卡/多卡、参考/新 backend、FP16/量化、cache off/on、普通/speculative。保留最小失败输入和完整环境，直到定位到第一个产生差异的层。

例如只有“graph 开＋prefix 命中”失败，先固定同一 token 序列与采样，形成四格对照：eager/cache off、eager/cache on、graph/cache off、graph/cache on。前三格通过会把问题缩到 replay 输入更新、固定地址与缓存元数据的交互。再把 batch 从 2 缩到 1、长度缩到页边界附近，找到最早偏离的张量；直接用长文本准确率反复测试很难提供这种证据。

## 7. 官方入口

- `sglang/docs_new/docs/supported-models/support_new_models.mdx`
- `sglang/docs_new/docs/developer_guide/evaluating_new_models.mdx`
- `sglang/docs_new/docs/developer_guide/development_jit_kernel_guide.mdx`
- `sglang/docs_new/docs/developer_guide/quantization_contribution_guide.mdx`
