# 模型谱系、转换与 Tokenizer

相关基础：[通信与 tensor 基础](<../../../communication/两卡通信与torchrun.md>)。


## 入门例子：模型接入是六项契约

模型名字相同仍可能有不同词表、位置编码、QKV 布局或权重命名。先核对 config、tokenizer/template、state_dict 名字、shape/dtype、分片布局、forward 输出六项。

用一小段固定文本验证 token ids；用一层固定输入核对输出；再做整模型少量 logits 对齐。随机采样生成文本相似不足以证明权重转换正确。

一项 QKV 权重从 HF shape 到本 rank local shape，再导回全局布局，必须保持同一 head 语义。GQA 的 query 与 KV head 数不同，按三个相同矩阵盲切会破坏这种对应。

## 1. 模型不是一组启动参数

Megatron Core 的主要模型族包括 decoder-only GPT/LLaMA/Qwen/DeepSeek 类、encoder-only BERT、encoder-decoder T5，以及 Mamba/hybrid 和多模态组件。共同基础是可组合的 `TransformerConfig`、`TransformerBlock/Layer`、`ModuleSpec` 与 parallel-aware layer。

源码入口：`megatron/core/models/`。新模型应先识别它改变的是层结构、attention、position embedding、normalization、MLP、loss 还是 pipeline placement，再决定扩展 spec、config 或专门 model class。

这些模型族首先在“允许看到什么”和“预测什么”上不同。decoder-only 自回归模型用因果 mask，从前缀预测下一 token；BERT 类 encoder 常允许输入位置双向互看，再预测被遮住的位置；T5 类 encoder-decoder 先编码源序列，decoder 再同时读取源表示和目标前缀。即使 hidden size、层数相同，改变 attention mask 或目标构造也会改变函数，不能只按参数 shape 判断模型等价。

## 2. 外部模型转换

官方推荐通过 Megatron Bridge 与 Hugging Face 格式互转。转换的本质不是重命名 key，还包括 TP/PP shard 合并拆分、QKV/MLP layout、GQA head、词表 padding、共享权重和 dtype。转换验收至少包含 config 对齐、参数数目/shape、固定输入 logits、短生成或短训练 loss。

### QKV 合并为什么不是简单 concat

设 hidden H=4，Q heads=4，KV heads=2，每个 head 宽 d=1。按 PyTorch weight 存储约定，外部权重 `W_Q:[4,4]`、`W_K:[2,4]`、`W_V:[2,4]`。按 Q、K、V 拼接可以形成 `[8,4]`，但目标实现可能按 query group 组织成：

```text
group0: Q0,Q1,K0,V0
group1: Q2,Q3,K1,V1
```

这与 `[Q0,Q1,Q2,Q3,K0,K1,V0,V1]` 的行顺序不同。前者若按 TP=2 分组，每卡得到 `[4,4]`，包含两个 Q heads 和与它们配套的一组 KV；直接切后者前四行，则 rank0 只有 Q，没有对应 KV。真实布局需由转换器与目标模块共同确认，本例说明为什么必须先解释行的语义再切 shard。

验证可以分两步：用唯一编号标记每一行，检查转换→导出后顺序是否还原；再给定相同 hidden input，比较 Q、K、V 及 attention 输出。单纯 round-trip 还不充分：一对互为逆的错误转换也能还原原文件，却让训练模型执行错误函数。因此固定输入的中间张量对齐是独立证据。

## 3. Tokenizer 是 checkpoint 契约的一部分

统一 `MegatronTokenizer` API 可从 metadata 选择 Hugging Face、SentencePiece、TikToken、Megatron legacy 或 Null backend。metadata 应与模型配置、数据 preprocessing 和 checkpoint 一起版本化。

Null tokenizer 适合 mock benchmark、无外部文件的测试、以及已经预分词的 `.bin/.idx` 数据；此时仍必须提供正确 `vocab_size/eod/pad`。它不是“随便跳过 tokenizer”。

例如旧词表 `{猫:0,狗:1}`，新词表 `{狗:0,猫:1}`，两者词表大小完全相同。若原样加载 embedding 的两行，输入“猫”会取到旧“狗”的向量；output layer 和预处理数据也随之错位。词表大小一致只保证索引范围合法，不保证 ID 的语言含义一致。

词表 padding 是另一个概念：有效词表 V=10，为满足 TP=4 的分片要求可填充到 V_pad=12，每卡存 3 行。多出的两行是布局需要，不是自动新增了两个真实 token。数据中的合法 ID、输出 loss 是否包含 padding 类别，以及转换时如何保留/去掉填充，都要按实现约定处理。把 padding 行直接删掉可能破坏 shard shape；把它们当真实 token 又会改变任务。

chat template（对话模板）把角色、分隔符和消息正文组织成待分词文本。同一句正文若前后加了不同的角色标记，会得到不同 ID 序列，也会改变哪些位置需要 loss。模板、special token、loss mask 与 checkpoint 应作为一份共同的数据契约记录。

## 4. 训练前检查

- encode/decode 与 special token；
- tokenizer vocab 与 padded vocab；
- embedding/output layer 是否共享；
- 数据中的 token ID 是否都在有效范围；
- 转换前后相同文本是否产生相同 IDs；
- chat template 是否属于训练数据契约，而非运行时临时配置。

<a id="5-源码阅读路线"></a>

## 5. 源码职责索引

`megatron/core/tokenizers/` → `megatron/training/global_vars.py:_build_tokenizer` 的应用启动 → preprocessing tool → `GPTDataset` → `GPTModel` embedding。沿这条线能解释“文本如何最终成为某个 vocab shard 上的 embedding lookup”。
