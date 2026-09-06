# 00｜共享基础：GPU、显存与 LLM 推理系统硬件基础

> 本页为兼容入口，正文已整合到下面的主章节；学习时直接进入主章节即可。



- [GPU 执行与内存基础](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md>)
- [Transformer 与 KV 基础](<../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md>)
- [Hopper 异步硬件（专项）](<../../learn_docs/01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md>)


<details>
<summary>旧章节定位（供历史链接使用）</summary>

<a id="00共享基础gpu显存与-llm-推理系统硬件基础"></a>

00｜共享基础：GPU、显存与 LLM 推理系统硬件基础 → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md>)

<a id="1-先建立一张总地图llm-最终是在一台什么机器上跑"></a>

1. 先建立一张总地图：LLM 最终是在一台什么机器上跑？ → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#1-先建立一张总地图llm-最终是在一台什么机器上跑>)

<a id="2-cpu-和-gpu-为什么不一样"></a>

2. CPU 和 GPU 为什么不一样？ → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#2-cpu-和-gpu-为什么不一样>)

<a id="3-什么是-hbm为什么论文天天讨论-hbm"></a>

3. 什么是 HBM？为什么论文天天讨论 HBM？ → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#3-什么是-hbm为什么论文天天讨论-hbm>)

<a id="4-gpu-memory-hierarchy显存并不是只有一层"></a>

4. GPU Memory Hierarchy：显存并不是只有一层 → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#4-gpu-memory-hierarchy显存并不是只有一层>)

<a id="41-register"></a>

4.1 Register → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#41-register>)

<a id="42-shared-memory--sram"></a>

4.2 Shared Memory / SRAM → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#42-shared-memory--sram>)

<a id="43-l2-cache"></a>

4.3 L2 Cache → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#43-l2-cache>)

<a id="44-hbm"></a>

4.4 HBM → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#44-hbm>)

<a id="5-带宽和延迟到底是什么"></a>

5. “带宽”和“延迟”到底是什么？ → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#5-带宽和延迟到底是什么>)

<a id="带宽-bandwidth"></a>

带宽 Bandwidth → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#带宽-bandwidth>)

<a id="延迟-latency"></a>

延迟 Latency → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#延迟-latency>)

<a id="6-threadwarpblockctasmcuda-的执行层次"></a>

6. Thread、Warp、Block/CTA、SM：CUDA 的执行层次 → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#6-threadwarpblockctasmcuda-的执行层次>)

<a id="61-thread"></a>

6.1 Thread → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#61-thread>)

<a id="62-warp"></a>

6.2 Warp → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#62-warp>)

<a id="63-thread-block--cta"></a>

6.3 Thread Block / CTA → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#63-thread-block--cta>)

<a id="64-sm"></a>

6.4 SM → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#64-sm>)

<a id="7-tensor-core-是什么为什么矩阵乘法特别快"></a>

7. Tensor Core 是什么？为什么矩阵乘法特别快？ → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#7-tensor-core-是什么为什么矩阵乘法特别快>)

<a id="8-kernel-是什么"></a>

8. Kernel 是什么？ → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#8-kernel-是什么>)

<a id="9-kernel-fusion-为什么有效"></a>

9. Kernel Fusion 为什么有效？ → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#9-kernel-fusion-为什么有效>)

<a id="10-什么是-tiling"></a>

10. 什么是 Tiling？ → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#10-什么是-tiling>)

<a id="11-arithmetic-intensity--operational-intensity"></a>

11. Arithmetic Intensity / Operational Intensity → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#11-arithmetic-intensity--operational-intensity>)

<a id="程序-a"></a>

程序 A → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#程序-a>)

<a id="程序-b"></a>

程序 B → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#程序-b>)

<a id="12-compute-bound-与-memory-bound"></a>

12. Compute-bound 与 Memory-bound → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#12-compute-bound-与-memory-bound>)

<a id="compute-bound"></a>

Compute-bound → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#compute-bound>)

<a id="memory-bound"></a>

Memory-bound → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#memory-bound>)

<a id="13-roofline-model-的直觉"></a>

13. Roofline Model 的直觉 → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#13-roofline-model-的直觉>)

<a id="14-occupancy-是什么"></a>

14. Occupancy 是什么？ → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#14-occupancy-是什么>)

<a id="15-为什么-llm-推理必须理解-prefill-与-decode"></a>

15. 为什么 LLM 推理必须理解 Prefill 与 Decode？ → [进入主章节](<../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md#15-为什么-llm-推理必须理解-prefill-与-decode>)

<a id="prefill"></a>

Prefill → [进入主章节](<../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md#prefill>)

<a id="decode"></a>

Decode → [进入主章节](<../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md#decode>)

<a id="16-kv-cache为什么一个缓存能成为系统核心"></a>

16. KV Cache：为什么一个“缓存”能成为系统核心？ → [进入主章节](<../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md#16-kv-cache为什么一个缓存能成为系统核心>)

<a id="17-为什么-kv-cache-会产生内存碎片"></a>

17. 为什么 KV Cache 会产生“内存碎片”？ → [进入主章节](<../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md#17-为什么-kv-cache-会产生内存碎片>)

<a id="18-virtual-memory-与-paging为什么-vllm-会借鉴操作系统"></a>

18. Virtual Memory 与 Paging：为什么 vLLM 会借鉴操作系统？ → [进入主章节](<../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md#18-virtual-memory-与-paging为什么-vllm-会借鉴操作系统>)

<a id="19-internal-fragmentation-与-external-fragmentation"></a>

19. Internal Fragmentation 与 External Fragmentation → [进入主章节](<../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md#19-internal-fragmentation-与-external-fragmentation>)

<a id="internal-fragmentation"></a>

Internal Fragmentation → [进入主章节](<../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md#internal-fragmentation>)

<a id="external-fragmentation"></a>

External Fragmentation → [进入主章节](<../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md#external-fragmentation>)

<a id="20-copy-on-write-为什么会出现在-llm-serving"></a>

20. Copy-on-Write 为什么会出现在 LLM Serving？ → [进入主章节](<../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md#20-copy-on-write-为什么会出现在-llm-serving>)

<a id="21-prefix-cache-与-radix-tree-的直觉"></a>

21. Prefix Cache 与 Radix Tree 的直觉 → [进入主章节](<../../learn_docs/00_Foundations/05_Transformer执行与KV基础.md#21-prefix-cache-与-radix-tree-的直觉>)

<a id="22-四篇论文分别在硬件系统哪一层"></a>

22. 四篇论文分别在硬件/系统哪一层？ → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#22-四篇论文分别在硬件系统哪一层>)

<a id="23-读论文时遇到这些词可以直接这样翻译"></a>

23. 读论文时遇到这些词，可以直接这样翻译 → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#23-读论文时遇到这些词可以直接这样翻译>)

<a id="24-推荐阅读顺序"></a>

24. 推荐阅读顺序 → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#24-推荐阅读顺序>)

<a id="25-最值得先记住的-8-个硬件-insight"></a>

25. 最值得先记住的 8 个硬件 Insight → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#25-最值得先记住的-8-个硬件-insight>)

<a id="主要参考资料"></a>

主要参考资料 → [进入主章节](<../../learn_docs/00_Foundations/04_GPU执行与内存基础.md#主要参考资料>)

<a id="26-hopperh100-时代的新硬件概念为-flashattention-3-做准备"></a>

26. Hopper/H100 时代的新硬件概念：为 FlashAttention-3 做准备 → [进入主章节](<../../learn_docs/01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md#26-hopperh100-时代的新硬件概念为-flashattention-3-做准备>)

<a id="261-同步执行和异步执行有什么区别"></a>

26.1 同步执行和异步执行有什么区别？ → [进入主章节](<../../learn_docs/01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md#261-同步执行和异步执行有什么区别>)

<a id="262-tma-是什么"></a>

26.2 TMA 是什么？ → [进入主章节](<../../learn_docs/01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md#262-tma-是什么>)

<a id="263-warp-group-是什么"></a>

26.3 Warp Group 是什么？ → [进入主章节](<../../learn_docs/01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md#263-warp-group-是什么>)

<a id="264-wgmma-是什么"></a>

26.4 WGMMA 是什么？ → [进入主章节](<../../learn_docs/01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md#264-wgmma-是什么>)

<a id="265-什么是-warp-specializationwarp-专职化不同-warp-group-承担不同流水线职责"></a>

26.5 什么是 Warp Specialization（warp 专职化：不同 warp group 承担不同流水线职责）？ → [进入主章节](<../../learn_docs/01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md#265-什么是-warp-specializationwarp-专职化不同-warp-group-承担不同流水线职责>)

<a id="266-double-buffer--multi-stage-pipeline-是什么"></a>

26.6 Double Buffer / Multi-stage Pipeline 是什么？ → [进入主章节](<../../learn_docs/01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md#266-double-buffer--multi-stage-pipeline-是什么>)

<a id="267-为什么-tensor-core-变得越快softmax-反而越值得优化"></a>

26.7 为什么 Tensor Core 变得越快，Softmax 反而越值得优化？ → [进入主章节](<../../learn_docs/01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md#267-为什么-tensor-core-变得越快softmax-反而越值得优化>)

<a id="268-fp16bf16fp8-为什么会影响-kernel-设计"></a>

26.8 FP16、BF16、FP8 为什么会影响 kernel 设计？ → [进入主章节](<../../learn_docs/01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md#268-fp16bf16fp8-为什么会影响-kernel-设计>)

<a id="269-把-hopper-硬件能力映射到-flashattention-3"></a>

26.9 把 Hopper 硬件能力映射到 FlashAttention-3 → [进入主章节](<../../learn_docs/01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md#269-把-hopper-硬件能力映射到-flashattention-3>)

<a id="hopper-部分参考资料"></a>

Hopper 部分参考资料 → [进入主章节](<../../learn_docs/01_Kernel_GPU_Programming_Compiler/11_Hopper异步硬件专题.md#hopper-部分参考资料>)

</details>
