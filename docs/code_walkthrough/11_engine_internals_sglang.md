# 11 引擎内部实现（SGLang 篇）：RL 专用端点在服务端发生了什么

> 本页为兼容入口，正文已整合到下面的主章节；学习时直接进入主章节即可。

服务端机制统一由 SGLang 系列维护；slime 客户端见 02/04。修正旧稿：abort+换权不是跨 rank 事务保证，回放字段也不等于完整随机性或严格 on-policy。字节桶例子已合入控制面章节。

- [换权、通信组、IPC、缓存与失败边界](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)
- [采样、logprob 与返回数据](<../sglang_code_walkthrough/04_interfaces_and_models/01_api_request_and_sampling.md>)
- [Router 与会话亲和](<../sglang_code_walkthrough/03_scaling_and_deployment/02_deployment_topology_and_routing.md>)
- [abort 后的 Sample 与 partial rollout](<03_data_buffer_partial_rollout_async.md>)


<details>
<summary>旧章节定位（供历史链接使用）</summary>

<a id="11-引擎内部实现sglang-篇rl-专用端点在服务端发生了什么"></a>

11 引擎内部实现（SGLang 篇）：RL 专用端点在服务端发生了什么 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="1-三层调用链总览"></a>

1. 三层调用链总览 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="2-update_weights_from_distributednccl-广播的服务端"></a>

2. `update_weights_from_distributed`：NCCL 广播的服务端 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="21-http-路由与请求体"></a>

2.1 HTTP 路由与请求体 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="22-tokenizermanager并发闸门"></a>

2.2 TokenizerManager：并发闸门 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="23-scheduler--tpworker"></a>

2.3 Scheduler / TpWorker → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="24-modelrunnerweightupdater建组与接收"></a>

2.4 ModelRunner.WeightUpdater：建组与接收 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="25-深入拆解flattenedtensorbucketweight_synctensor_bucketpy19-89为什么要把多个张量拼成一个再传"></a>

2.5 深入拆解：`FlattenedTensorBucket`（`weight_sync/tensor_bucket.py:19-89`）——为什么要把多个张量拼成一个再传 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="3-update_weights_from_tensorcuda-ipc-的服务端"></a>

3. `update_weights_from_tensor`：CUDA IPC 的服务端 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="4-releaseresume_memory_occupation显存错峰的服务端"></a>

4. `release/resume_memory_occupation`：显存错峰的服务端 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="5-abort_request-与已生成前缀的保留"></a>

5. `abort_request` 与已生成前缀的保留 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="6-logprob-与回放数据的产生"></a>

6. logprob 与回放数据的产生 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="7-router会话亲和的实现"></a>

7. Router：会话亲和的实现 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="8-其余-rl-端点速查"></a>

8. 其余 RL 端点速查 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

<a id="9-小结"></a>

9. 小结 → [进入主章节](<../sglang_code_walkthrough/04_interfaces_and_models/03_control_plane_and_post_training.md>)

</details>
