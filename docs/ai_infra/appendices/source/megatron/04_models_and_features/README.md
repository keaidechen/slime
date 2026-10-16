# 04_models_and_features资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../../README.md>)。

这一组沿三个问题展开：外部权重怎样保持原来的数学语义；MoE/MLA/MTP 怎样改变层内计算；多模态、混合层和在线生成怎样改变数据与状态。最后再分析这些结构与 CP、低精度、offload 的组合。

不用先记住全部模型名。对每个新结构，先找一个小输入，列出新增张量和损失，再看谁负责保存、路由或同步。读完应能区分“参数更少”“每 token 计算更少”“历史缓存更小”和“端到端更快”这四种不同结论。

- [模型谱系、转换与 Tokenizer](<01_models_and_tokenizers.md>)
- [MoE、MLA、MTP：高级架构如何进入训练系统](<02_moe_mla_mtp.md>)
- [Hybrid、Mamba、多模态、Energon 与 RL 生态](<03_hybrid_multimodal_rl.md>)
- [Context Parallel、MoE/EP 与低精度训练深读](<04_context_moe_precision_deep_dive.md>)
