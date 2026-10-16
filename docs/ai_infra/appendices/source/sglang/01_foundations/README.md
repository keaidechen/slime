# 01_foundations资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../../README.md>)。

第一次阅读按下面三篇的顺序进行。概念课用 `[10,11,12]` 的短 prompt 建立 prefill/decode 与 KV 的区别；框架概览解释各组件的所有权；请求链路再把这些状态落实到进程、消息和源码对象。

读完后应能解释：为什么首 token 在 prefill 后已经产生，却还没有自己的 KV；为什么 `rid` 与 batch 行号不同；为什么取消 HTTP 不等于立即释放 GPU 内存。后续 Runtime 章节将继续使用这些边界。

- [SGLang Runtime：请求、执行与 KV 三种状态如何协作](<00_runtime_concepts.md>)
- [SGLang 框架与状态概览](<01_framework_overview.md>)
- [1.2 一条 request 进入 SGLang 后如何流转](<02_process_topology_and_request_path.md>)
