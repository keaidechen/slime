# 6.2 并行、Backend、缓存与量化

相关基础：[GPU、tensor 与通信基础](<../../../foundations/README.md>)。


## 入门例子：把缓存与量化的收益写成可比较条件

减少采样步数、复用中间结果和降低精度都会影响执行成本，但也可能改变输出质量。比较时固定 prompt、种子、尺寸、步数与评价方式，并明确哪些量是实验变量。

并行可以切组件、batch 或内部计算，通信成本与显存收益不同；先画 latent/条件张量在哪里，再讨论多卡。缓存则要说明何时可复用、何时因条件变化必须失效。

速度、显存与质量需要在相同适用条件下共同解释。LLM 的 token/s 与扩散的请求/图像或采样步耗时不能直接比较。

## 机制与实现

## 1. 并行和解耦

CFG（Classifier-Free Guidance，无分类器引导）组合有条件与无条件等预测来加强生成条件的影响。它可能增加网络计算，也会改变生成分布，具体执行次数与组合公式需声明。

DiT（Diffusion Transformer）使用 Transformer 组织扩散生成中的去噪计算。一次输出通常需多个采样时间步，所以一次网络前向不能直接当作完整生成耗时。

Sequence Parallel 切分 token/patch 序列，适合高分辨率图像和长视频；CFG parallel 可分离条件/无条件分支；TP 切分权重；组件解耦把 text encoder、DiT 和 VAE 放入不同资源池。

选择并行度时同时测阶段计算、collective、pipeline 空洞和跨阶段传输。Diffusion 解耦传输 tensor/latent，LLM PD 传输 KV，容量和失败模型不同。

### CFG：两份预测怎样合成一份

常见 CFG 公式是 `v_guided=v_uncond+w(v_cond-v_uncond)`。`v_cond` 是给定文本条件的预测，`v_uncond` 是无条件或约定负条件的预测，均与 latent 同形状；w 是无单位引导强度，三者必须处于相同 prediction space。例如两元素预测分别为 `[1,0]` 和 `[2,2]`，w=2 时组合为 `[3,4]`；w=1 恢复有条件预测，w=0 使用无条件预测。

这相当于先求“条件带来的变化”再放大，并非求两个图像的平均。传统两分支 CFG 每步需两份网络计算，可以在一张卡顺序算、拼 batch 算，或把分支分到两卡后合并。分支并行减少关键路径，但会复制部分权重/状态并新增合并通信；蒸馏 guidance 等模型未必执行两分支，不能一律把步骤成本乘二。

### 序列并行不是切完各算各的

沿用 6.1 的 4 个 patch，隐藏张量为 `[1,4,d]`。两卡各持 2 个 Query patch，即 `[1,2,d]`；若模型是全局 self-attention，每卡的 Query 仍需要全部 4 个位置的 K/V。可以收集 K/V，或让 K/V 分块在 rank 间流转并累积 Attention。仅各算自己的两个 patch 会丢掉跨片信息，改变模型。

这里的 N 是 patch 数。全局 Attention 的分数逻辑规模是 N×N，即使融合 kernel 不把完整矩阵写入内存，计算仍依赖这些位置对。序列并行降低每卡激活压力、分摊计算，但通信规模和固定延迟会限制收益；小 N 或慢互联下可能更慢。Ring（环形）交换还要求各 rank 按一致轮次收发，某条请求取消时不能让一个 rank 单独跳出通信。

## 2. Attention Backend

选择受硬件、dtype、attention 变体、head/sequence shape、并行方式和 mask 影响。对每个主要 shape 记录实际 kernel、warmup 后耗时、workspace、峰值显存、参考误差和平台限制。

Sliding Tile Attention 等稀疏方法可能改变输出质量，不能与严格等价优化混在同一组结论中。

## 3. 缓存加速

Cache-DiT、TeaCache 利用相邻 timestep 特征变化有限，近似跳过或复用计算；它们不是 LLM prefix cache。

| LLM prefix cache | Diffusion feature cache |
|---|---|
| 相同 token 前缀复用 KV | 相邻 step 近似复用特征 |
| 通常不改变语义 | 可能引入质量误差 |
| key 是离散 token | 判定依赖特征差异、step 和模型 |

报告阈值、命中/跳步率、固定 seed 质量和加速比。

### 用特征变化解释“近似”在哪里

残差是网络块输出与其输入的差值，缓存残差可用来近似本步某段块计算。TeaCache 的本地说明先比较相邻 step 的调制输入 u：`δ=mean(|u_now-u_prev|)/mean(|u_prev|)`。分子分母是同尺度的平均绝对值，δ 无单位；实际实现还需处理分母很小等数值边界。

手算 `u_prev=[2,4]`、`u_now=[2.1,4.2]`，分子是 `(0.1+0.2)/2=0.15`，分母是 3，故 δ=0.05。官方机制还使用模型相关的多项式重标定并累积变化；若教学上暂把重标定看成恒等映射，累计值原为 0.02、阈值为 0.1，则本次累计 0.07，可选择复用；下一步再增加 0.04 就达到 0.11，应重算并重置。这个阈值和恒等假设仅帮助理解，不是推荐参数或 TeaCache 的完整实现。

关键区别在于：`u_now≠u_prev`，缓存残差只是估计当前残差，所以会引入误差。必须保持模型、条件、shape、step 以及 CFG 分支身份正确；正负分支不能互用缓存。复用减少网络块计算，却增加检测与缓存读写，缓存 tensor 还要驻留到下一次消费。

若可跳过块每步成本为 8 ms，命中率 h=0.5，检测/管理每步 0.5 ms，则该部分平均成本约 `0.5+(1-h)×8=4.5 ms`，理想节省 3.5 ms；h 很低时检测反而可能净增成本。端到端还含未跳过层、VAE 等，且必须同时测质量，不能只报告命中率。

## 4. Progressive Resolution

早期 timestep 使用较低空间/时间分辨率，后期恢复完整分辨率。收益依赖模型、输出尺寸、切换策略和 VAE 占比。这是近似优化，必须与 full-resolution 基线做同 seed 对比。

例如固定 patch 大小时，latent 空间边长减半，patch 数从 N 降到 N/4。线性投影/MLP 的 token 相关工作约降到 1/4，全局 Attention 的位置对从 N² 降到 N²/16。若原来四步都用全分辨率，现在前两步低分辨率、后两步全分辨率，只按位置对估计工作是 `2×N²/16+2×N²=2.125N²`，相对 `4N²` 约为 53.1%。

这不是整个管线必然加速 1/0.531 倍：尺寸转换、位置编码重建、网络其他层和解码仍有成本。切换后的 latent 不是凭空恢复出此前没计算的高频细节，因此生成质量和模型是否适应该路径才是最终边界。

## 5. 量化

先明确量化组件、weight/activation/accumulation dtype、敏感层跳过规则、硬件 kernel，以及与并行、LoRA、cache 和 graph 的兼容性。图像/视频质量评估不能只看数值误差。

以最简单的对称整数标量量化解释误差来源：`q=clip(round(x/s),q_min,q_max)`，重建 `x_hat=sq`。s 是正 scale，单位与 x 相同，q 是整数编码；round 取最近整数，clip 把结果限制在表示范围。取 x=0.7、s=0.25、范围 [-7,7]，得到 q=3、重建 0.75，误差 0.05。若 x=3，则 q 截到 7、重建 1.75，发生饱和，误差变为 1.25。

组大小为 G 的 n 个权重，b 位编码及每组 s_bytes 字节 scale 的理想存储约为 `n×b/8+ceil(n/G)×s_bytes` 字节，另有对齐等成本。例如 n=16、b=4、G=8、scale 每组 2 字节，需要 `8+2×2=12` 字节，相比 BF16 的 32 字节约压缩 2.67 倍，而非 4 倍。FP8/FP4 等格式不等同于上述均匀整数公式，这里只说明“编码＋scale＋重建”的成本。

降低权重精度主要节省权重存储与搬运，不自动减少采样步数，也不自动缩小所有 activation；累积仍可能使用较高精度。多次迭代中误差会传播，因此要检查中间 latent 和最终质量。没有高效低精度 kernel、反量化占主导或瓶颈在 VAE/网络输出时，压缩也未必加速。

官方量化章节涵盖 FP8、FP4/MXFP4、Nunchaku/SVDQuant、ModelOpt 和 ModelSlim；具体支持以当前兼容矩阵为准。

## 6. 优化分类

| 类别 | 示例 | 正确性与质量条件 |
|---|---|---|
| 尽量等价 | backend、并行、通信优化 | 数值/视觉对齐 |
| 精度变化 | 量化 | 质量门槛与性能 |
| 近似计算 | feature cache、progressive resolution | 同 seed 质量与收益曲线 |

## 7. 官方入口

- `attention_backends.mdx`
- `ring_sp_performance.mdx`、`disaggregation.mdx`
- `caching-acceleration.mdx`、`cache_dit.mdx`、`teacache.mdx`
- `progressive_resolution.mdx`、`quantization.mdx`

以上路径均位于 `sglang/docs_new/docs/sglang-diffusion/`。
