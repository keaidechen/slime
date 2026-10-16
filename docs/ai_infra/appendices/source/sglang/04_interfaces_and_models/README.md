# 04_interfaces_and_models资料

本目录用于查询正文之外的推导、实现和历史资料；全书的连续阅读顺序见[总目录](<../../../../README.md>)。

从用户输入走向共享状态：4.1 把协议默认值与 GPU 采样连起来，4.2 解释模型注册、权重映射和媒体张量展开，4.3 处理换权、LoRA 与后训练版本交接。

本组的三个关键算例是三项词表的概率过滤、4×4 图像变成 patch/placeholder、两 rank 换权只有一侧成功。读完应能区分格式合法与内容正确、形状兼容与语义兼容、请求隔离与全局原子更新。

- [4.1 API、请求规范化与采样](<01_api_request_and_sampling.md>)
- [4.2 模型支持、多模态与 Transformers Fallback](<02_model_support_and_multimodal.md>)
- [4.3 控制面、在线权重更新与后训练](<03_control_plane_and_post_training.md>)
