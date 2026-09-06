# 09｜vLLM V1 Runtime 架构详解：Scheduler、KV Cache Manager、Engine Core 与 GPU Worker 如何协作？

> 本页为兼容入口，正文已整合到下面的主章节；学习时直接进入主章节即可。



- [vLLM V1 架构对比](<../runtime_comparisons/vLLM-V1.md>)


<details>
<summary>旧章节定位（供历史链接使用）</summary>

<a id="09vllm-v1-runtime-架构详解schedulerkv-cache-managerengine-core-与-gpu-worker-如何协作"></a>

09｜vLLM V1 Runtime 架构详解：Scheduler、KV Cache Manager、Engine Core 与 GPU Worker 如何协作？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#09vllm-v1-runtime-架构详解schedulerkv-cache-managerengine-core-与-gpu-worker-如何协作>)

<a id="1-先区分vllm-论文时代和今天的-vllm"></a>

1. 先区分“vLLM 论文时代”和“今天的 vLLM” → [进入主章节](<../runtime_comparisons/vLLM-V1.md#1-先区分vllm-论文时代和今天的-vllm>)

<a id="2-v1-的高层进程结构"></a>

2. V1 的高层进程结构 → [进入主章节](<../runtime_comparisons/vLLM-V1.md#2-v1-的高层进程结构>)

<a id="3-api-server-做什么"></a>

3. API Server 做什么？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#3-api-server-做什么>)

<a id="4-engine-core-是-vllm-v1-的心脏"></a>

4. Engine Core 是 vLLM V1 的心脏 → [进入主章节](<../runtime_comparisons/vLLM-V1.md#4-engine-core-是-vllm-v1-的心脏>)

<a id="5-vllm-v1-scheduler-的调度单位不是request而是token-budget"></a>

5. vLLM V1 Scheduler 的调度单位不是“Request”，而是“Token Budget” → [进入主章节](<../runtime_comparisons/vLLM-V1.md#5-vllm-v1-scheduler-的调度单位不是request而是token-budget>)

<a id="6-continuous-batching-在-vllm-里怎么体现"></a>

6. Continuous Batching 在 vLLM 里怎么体现？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#6-continuous-batching-在-vllm-里怎么体现>)

<a id="7-kv-cache-manager-是另一个核心心脏"></a>

7. KV Cache Manager 是另一个核心心脏 → [进入主章节](<../runtime_comparisons/vLLM-V1.md#7-kv-cache-manager-是另一个核心心脏>)

<a id="8-paged-kv-cache-到底如何帮助-scheduler"></a>

8. Paged KV Cache 到底如何帮助 scheduler？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#8-paged-kv-cache-到底如何帮助-scheduler>)

<a id="9-prefix-caching-在-v1-中是什么"></a>

9. Prefix Caching 在 V1 中是什么？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#9-prefix-caching-在-v1-中是什么>)

<a id="10-prefix-cache-和-sglang-radixattention-是不是一样"></a>

10. Prefix Cache 和 SGLang RadixAttention 是不是一样？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#10-prefix-cache-和-sglang-radixattention-是不是一样>)

<a id="11-gpu-worker-是什么"></a>

11. GPU Worker 是什么？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#11-gpu-worker-是什么>)

<a id="12-model-runner-又是什么层"></a>

12. Model Runner 又是什么层？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#12-model-runner-又是什么层>)

<a id="13-多-gpu-时进程结构怎么理解"></a>

13. 多 GPU 时进程结构怎么理解？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#13-多-gpu-时进程结构怎么理解>)

<a id="14-chunked-prefill-为什么自然融入-v1-scheduler"></a>

14. Chunked Prefill 为什么自然融入 V1 Scheduler？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#14-chunked-prefill-为什么自然融入-v1-scheduler>)

<a id="15-speculative-decoding-为什么也能塞进同一-runtime"></a>

15. Speculative Decoding 为什么也能塞进同一 Runtime？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#15-speculative-decoding-为什么也能塞进同一-runtime>)

<a id="16-pd-disaggregation-在-vllm-里怎么进入架构"></a>

16. P/D Disaggregation 在 vLLM 里怎么进入架构？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#16-pd-disaggregation-在-vllm-里怎么进入架构>)

<a id="17-vllm-v1-为什么强调简化核心架构"></a>

17. vLLM V1 为什么强调“简化核心架构”？ → [进入主章节](<../runtime_comparisons/vLLM-V1.md#17-vllm-v1-为什么强调简化核心架构>)

<a id="18-vllm-与-flashinfer-的关系"></a>

18. vLLM 与 FlashInfer 的关系 → [进入主章节](<../runtime_comparisons/vLLM-V1.md#18-vllm-与-flashinfer-的关系>)

<a id="19-一个请求完整生命周期"></a>

19. 一个请求完整生命周期 → [进入主章节](<../runtime_comparisons/vLLM-V1.md#19-一个请求完整生命周期>)

<a id="20-ai-infra-最值得记住的-9-个-insight"></a>

20. AI Infra 最值得记住的 9 个 Insight → [进入主章节](<../runtime_comparisons/vLLM-V1.md#20-ai-infra-最值得记住的-9-个-insight>)

<a id="21-推荐源码阅读顺序"></a>

21. 推荐源码阅读顺序 → [进入主章节](<../runtime_comparisons/vLLM-V1.md#21-推荐源码阅读顺序>)

<a id="主要参考资料"></a>

主要参考资料 → [进入主章节](<../runtime_comparisons/vLLM-V1.md#主要参考资料>)

</details>
