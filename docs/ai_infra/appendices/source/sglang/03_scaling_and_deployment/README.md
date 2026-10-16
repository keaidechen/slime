# 03_scaling_and_deployment资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../../README.md>)。

模型放不下与请求处理不过来是两个问题：第一篇用小矩阵和 KV 传输解释 TP/PP/DP/EP/PD；第二篇把 rank 组成可路由实例；第三篇再核对硬件、软件和 backend 的相容条件。

每做一次容量计算，先写明是“每 rank”“每副本”还是“整个资源池”，并统一 byte、KiB、GiB 与秒。读完应能说明多卡增加了哪种通信、PD 省掉的干扰能否覆盖传输、命中更多前缀的实例为什么仍可能更慢。

- [3.1 分布式执行与 Prefill-Decode 解耦](<01_distributed_and_pd_disaggregation.md>)
- [3.2 部署拓扑、多节点与路由](<02_deployment_topology_and_routing.md>)
- [3.3 硬件平台与安装选择](<03_hardware_platforms.md>)
