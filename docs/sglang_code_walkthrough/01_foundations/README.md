# 第一部分：基础与请求链路

这一部分先建立性能、内存和状态机心智模型，再跟踪一个请求穿过 HTTP、Tokenizer、Scheduler、模型执行和 Detokenizer 的完整路径。

1. [Infra 工程师的学习地图](01_learning_map.md)
2. [进程拓扑与端到端请求路径](02_process_topology_and_request_path.md)

读完后应能画出进程与 IPC 拓扑，区分 TTFT、ITL、TPOT、E2E、throughput 和 goodput，并说清 request、batch、KV 三种生命周期。

