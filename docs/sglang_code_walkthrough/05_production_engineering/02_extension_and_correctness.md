# 5.2 模型与 Backend 扩展、正确性验证

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

## 3. 新 Attention Backend 契约

Backend 不只是一个 kernel wrapper。它要处理 prefill/decode、paged KV 地址、GQA/MLA、causal/sliding-window mask、position/RoPE、不同 dtype、graph capture 和并行布局。

每一种 forward mode 都与参考 backend 比较输出，并覆盖非对齐长度、page boundary、不同 batch、KV 复用和 graph replay。

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

## 6. 回归二分

一次只切换一个维度：eager/graph、单卡/多卡、参考/新 backend、FP16/量化、cache off/on、普通/speculative。保留最小失败输入和完整环境，直到定位到第一个产生差异的层。

## 7. 官方入口

- `sglang/docs_new/docs/supported-models/support_new_models.mdx`
- `sglang/docs_new/docs/developer_guide/evaluating_new_models.mdx`
- `sglang/docs_new/docs/developer_guide/development_jit_kernel_guide.mdx`
- `sglang/docs_new/docs/developer_guide/quantization_contribution_guide.mdx`

