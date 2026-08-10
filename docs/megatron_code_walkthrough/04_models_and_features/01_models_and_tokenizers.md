# 模型谱系、转换与 Tokenizer

## 1. 模型不是一组启动参数

Megatron Core 的主要模型族包括 decoder-only GPT/LLaMA/Qwen/DeepSeek 类、encoder-only BERT、encoder-decoder T5，以及 Mamba/hybrid 和多模态组件。共同基础是可组合的 `TransformerConfig`、`TransformerBlock/Layer`、`ModuleSpec` 与 parallel-aware layer。

源码入口：`megatron/core/models/`。新模型应先识别它改变的是层结构、attention、position embedding、normalization、MLP、loss 还是 pipeline placement，再决定扩展 spec、config 或专门 model class。

## 2. 外部模型转换

官方推荐通过 Megatron Bridge 与 Hugging Face 格式互转。转换的本质不是重命名 key，还包括 TP/PP shard 合并拆分、QKV/MLP layout、GQA head、词表 padding、共享权重和 dtype。转换验收至少包含 config 对齐、参数数目/shape、固定输入 logits、短生成或短训练 loss。

## 3. Tokenizer 是 checkpoint 契约的一部分

统一 `MegatronTokenizer` API 可从 metadata 选择 Hugging Face、SentencePiece、TikToken、Megatron legacy 或 Null backend。metadata 应与模型配置、数据 preprocessing 和 checkpoint 一起版本化。

Null tokenizer 适合 mock benchmark、无外部文件的测试、以及已经预分词的 `.bin/.idx` 数据；此时仍必须提供正确 `vocab_size/eod/pad`。它不是“随便跳过 tokenizer”。

## 4. 训练前检查

- encode/decode 与 special token；
- tokenizer vocab 与 padded vocab；
- embedding/output layer 是否共享；
- 数据中的 token ID 是否都在有效范围；
- 转换前后相同文本是否产生相同 IDs；
- chat template 是否属于训练数据契约，而非运行时临时配置。

## 5. 源码阅读路线

`core/tokenizers/` → `training/tokenizer/` 的应用适配 → preprocessing tool → `GPTDataset` → `GPTModel` embedding。沿这条线能解释“文本如何最终成为某个 vocab shard 上的 embedding lookup”。

