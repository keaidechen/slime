# 01_foundations资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../../README.md>)。

这一组回答“单次训练究竟做了什么”。先从文本怎样变成 token 开始，再认识进程、GPU 和通信，随后沿训练循环走到一次参数更新，最后拆开模型内部的计算。读完后，应能解释为什么打印出 loss 还不代表参数更新成功，以及 `[B,S]` 的整数 token 怎样变成 `[S,B,H]` 的连续特征。

可以随手保留一张账本：每遇到一个函数，写下输入形状、输出形状、执行设备、保留给反向的数据。暂时不打开任何并行优化，也能建立后面各章共同使用的基线。

- [Megatron 的程序边界、运行环境与数据管线](<00_ecosystem_install_and_data.md>)
- [Megatron 框架与执行边界](<01_framework_overview.md>)
- [从入口到一次参数更新](<02_training_mainline.md>)
- [模型、Transformer 组件与 ModuleSpec](<03_model_and_transformer.md>)
