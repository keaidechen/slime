# 10｜SGLang Runtime 架构详解：TokenizerManager、Scheduler、ModelRunner、Radix Cache 如何组成一条推理流水线？

> 本页为兼容入口，正文已整合到下面的主章节；学习时直接进入主章节即可。



- [SGLang Runtime 概念入门](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md>)


<details>
<summary>旧章节定位（供历史链接使用）</summary>

<a id="10sglang-runtime-架构详解tokenizermanagerschedulermodelrunnerradix-cache-如何组成一条推理流水线"></a>

10｜SGLang Runtime 架构详解：TokenizerManager、Scheduler、ModelRunner、Radix Cache 如何组成一条推理流水线？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#10sglang-runtime-架构详解tokenizermanagerschedulermodelrunnerradix-cache-如何组成一条推理流水线>)

<a id="1-先明确sglang-不只等于-radixattention"></a>

1. 先明确：SGLang 不只等于 RadixAttention → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#1-先明确sglang-不只等于-radixattention>)

<a id="2-srt-的最小请求路径"></a>

2. SRT 的最小请求路径 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#2-srt-的最小请求路径>)

<a id="3-tokenizermanager-为什么单独存在"></a>

3. TokenizerManager 为什么单独存在？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#3-tokenizermanager-为什么单独存在>)

<a id="4-scheduler-是-sglang-runtime-的核心控制面"></a>

4. Scheduler 是 SGLang Runtime 的核心控制面 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#4-scheduler-是-sglang-runtime-的核心控制面>)

<a id="5-radix-cache-为什么直接影响-scheduler"></a>

5. Radix Cache 为什么直接影响 Scheduler？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#5-radix-cache-为什么直接影响-scheduler>)

<a id="6-cache-aware-scheduling-的直觉"></a>

6. Cache-Aware Scheduling 的直觉 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#6-cache-aware-scheduling-的直觉>)

<a id="7-modelrunner-是什么"></a>

7. ModelRunner 是什么？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#7-modelrunner-是什么>)

<a id="8-attention-backend-为什么在-sglang-中特别值得关注"></a>

8. Attention Backend 为什么在 SGLang 中特别值得关注？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#8-attention-backend-为什么在-sglang-中特别值得关注>)

<a id="9-flashinfer-在-sglang-里处于哪里"></a>

9. FlashInfer 在 SGLang 里处于哪里？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#9-flashinfer-在-sglang-里处于哪里>)

<a id="10-detokenizermanager-为什么还要独立出来"></a>

10. DetokenizerManager 为什么还要独立出来？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#10-detokenizermanager-为什么还要独立出来>)

<a id="11-sglang-如何体现-continuous-batching"></a>

11. SGLang 如何体现 Continuous Batching？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#11-sglang-如何体现-continuous-batching>)

<a id="12-chunked-prefill-在-sglang-中解决什么"></a>

12. Chunked Prefill 在 SGLang 中解决什么？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#12-chunked-prefill-在-sglang-中解决什么>)

<a id="13-scheduler-与-kv-memory-pool-的关系"></a>

13. Scheduler 与 KV Memory Pool 的关系 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#13-scheduler-与-kv-memory-pool-的关系>)

<a id="14-为什么-sglang-的-prefix-cache-更像-runtime-的一等公民"></a>

14. 为什么 SGLang 的 Prefix Cache 更像 runtime 的一等公民？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#14-为什么-sglang-的-prefix-cache-更像-runtime-的一等公民>)

<a id="15-sglang-的-pd-disaggregation-怎么理解"></a>

15. SGLang 的 PD Disaggregation 怎么理解？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#15-sglang-的-pd-disaggregation-怎么理解>)

<a id="16-kv-transfer-backend-为什么成为新组件"></a>

16. KV Transfer Backend 为什么成为新组件？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#16-kv-transfer-backend-为什么成为新组件>)

<a id="17-router--model-gateway-是更上一层什么东西"></a>

17. Router / Model Gateway 是更上一层什么东西？ → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#17-router--model-gateway-是更上一层什么东西>)

<a id="18-三层调度一定要分清"></a>

18. 三层调度一定要分清 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#18-三层调度一定要分清>)

<a id="1-cluster--router-层"></a>

1. Cluster / Router 层 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#1-cluster--router-层>)

<a id="2-engine-scheduler-层"></a>

2. Engine Scheduler 层 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#2-engine-scheduler-层>)

<a id="3-gpu-kernel-层"></a>

3. GPU Kernel 层 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#3-gpu-kernel-层>)

<a id="19-sglang-和-vllm-runtime-的相似与不同"></a>

19. SGLang 和 vLLM Runtime 的相似与不同 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#19-sglang-和-vllm-runtime-的相似与不同>)

<a id="相似"></a>

相似 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#相似>)

<a id="历史基因不同"></a>

历史基因不同 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#历史基因不同>)

<a id="20-请求生命周期完整图"></a>

20. 请求生命周期完整图 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#20-请求生命周期完整图>)

<a id="21-ai-infra-最值得记住的-10-个-insight"></a>

21. AI Infra 最值得记住的 10 个 Insight → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#21-ai-infra-最值得记住的-10-个-insight>)

<a id="22-推荐源码阅读顺序"></a>

22. 推荐源码阅读顺序 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#22-推荐源码阅读顺序>)

<a id="主要参考资料"></a>

主要参考资料 → [进入主章节](<../sglang_code_walkthrough/01_foundations/00_runtime_concepts.md#主要参考资料>)

</details>
