# 08｜TensorRT-LLM Runtime 架构详解：从“高性能 Kernel”到 NVIDIA 推理执行系统

> 本页为兼容入口，正文已整合到下面的主章节；学习时直接进入主章节即可。



- [TensorRT-LLM 架构对比](<../runtime_comparisons/TensorRT-LLM.md>)


<details>
<summary>旧章节定位（供历史链接使用）</summary>

<a id="08tensorrt-llm-runtime-架构详解从高性能-kernel到-nvidia-推理执行系统"></a>

08｜TensorRT-LLM Runtime 架构详解：从“高性能 Kernel”到 NVIDIA 推理执行系统 → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#08tensorrt-llm-runtime-架构详解从高性能-kernel到-nvidia-推理执行系统>)

<a id="1-先明确-tensorrt-llm-是什么层"></a>

1. 先明确 TensorRT-LLM 是什么层 → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#1-先明确-tensorrt-llm-是什么层>)

<a id="2-tensorrttensorrt-llmtriton-inference-server-不要混"></a>

2. TensorRT、TensorRT-LLM、Triton Inference Server 不要混 → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#2-tensorrttensorrt-llmtriton-inference-server-不要混>)

<a id="tensorrt"></a>

TensorRT → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#tensorrt>)

<a id="tensorrt-llm"></a>

TensorRT-LLM → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#tensorrt-llm>)

<a id="triton-inference-server"></a>

Triton Inference Server → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#triton-inference-server>)

<a id="3-为什么-tensorrt-llm-也必须有-scheduler"></a>

3. 为什么 TensorRT-LLM 也必须有 Scheduler？ → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#3-为什么-tensorrt-llm-也必须有-scheduler>)

<a id="4-in-flight-batching-是什么"></a>

4. In-Flight Batching 是什么？ → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#4-in-flight-batching-是什么>)

<a id="5-为什么-in-flight-batching-与-paged-kv-cache-必须配合"></a>

5. 为什么 In-Flight Batching 与 Paged KV Cache 必须配合？ → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#5-为什么-in-flight-batching-与-paged-kv-cache-必须配合>)

<a id="6-runtime-里的-scheduler-大概在决定什么"></a>

6. Runtime 里的 Scheduler 大概在决定什么？ → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#6-runtime-里的-scheduler-大概在决定什么>)

<a id="7-executor-是什么"></a>

7. Executor 是什么？ → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#7-executor-是什么>)

<a id="8-模型到底怎么跨多-gpu"></a>

8. 模型到底怎么跨多 GPU？ → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#8-模型到底怎么跨多-gpu>)

<a id="tensor-parallelism"></a>

Tensor Parallelism → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#tensor-parallelism>)

<a id="pipeline-parallelism"></a>

Pipeline Parallelism → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#pipeline-parallelism>)

<a id="expert-parallelism"></a>

Expert Parallelism → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#expert-parallelism>)

<a id="9-runtime-与-kernel-是什么关系"></a>

9. Runtime 与 Kernel 是什么关系？ → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#9-runtime-与-kernel-是什么关系>)

<a id="10-tensorrt-llm-为什么高度强调-quantization"></a>

10. TensorRT-LLM 为什么高度强调 Quantization？ → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#10-tensorrt-llm-为什么高度强调-quantization>)

<a id="11-为什么-paged-attentionifbscheduler-是一组功能"></a>

11. 为什么 Paged Attention、IFB、Scheduler 是一组功能？ → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#11-为什么-paged-attentionifbscheduler-是一组功能>)

<a id="12-chunked-context--chunked-prefill-在这里做什么"></a>

12. Chunked Context / Chunked Prefill 在这里做什么？ → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#12-chunked-context--chunked-prefill-在这里做什么>)

<a id="13-tensorrt-llm-当前为何同时存在编译引擎和-pytorch-workflow-的概念"></a>

13. TensorRT-LLM 当前为何同时存在“编译引擎”和 PyTorch workflow 的概念？ → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#13-tensorrt-llm-当前为何同时存在编译引擎和-pytorch-workflow-的概念>)

<a id="14-tensorrt-llm-与-vllm-最大的气质差异是什么"></a>

14. TensorRT-LLM 与 vLLM 最大的气质差异是什么？ → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#14-tensorrt-llm-与-vllm-最大的气质差异是什么>)

<a id="tensorrt-llm-1"></a>

TensorRT-LLM → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#tensorrt-llm>)

<a id="vllm"></a>

vLLM → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#vllm>)

<a id="15-tensorrt-llm-与-flashattention--flashinfer-的关系"></a>

15. TensorRT-LLM 与 FlashAttention / FlashInfer 的关系 → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#15-tensorrt-llm-与-flashattention--flashinfer-的关系>)

<a id="16-tensorrt-llm-与-distserve--p-d-分离的关系"></a>

16. TensorRT-LLM 与 DistServe / P-D 分离的关系 → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#16-tensorrt-llm-与-distserve--p-d-分离的关系>)

<a id="17-一个请求在-tensorrt-llm-中的概念路径"></a>

17. 一个请求在 TensorRT-LLM 中的概念路径 → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#17-一个请求在-tensorrt-llm-中的概念路径>)

<a id="18-ai-infra-最值得记住的-8-个-insight"></a>

18. AI Infra 最值得记住的 8 个 Insight → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#18-ai-infra-最值得记住的-8-个-insight>)

<a id="19-推荐进一步读源码文档的顺序"></a>

19. 推荐进一步读源码/文档的顺序 → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#19-推荐进一步读源码文档的顺序>)

<a id="主要参考资料"></a>

主要参考资料 → [进入主章节](<../runtime_comparisons/TensorRT-LLM.md#主要参考资料>)

</details>
