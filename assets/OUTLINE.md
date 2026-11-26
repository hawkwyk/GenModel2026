# 一、先确定一条可执行的技术路线

建议项目题目定为：

> **面向文生图扩散模型的采样器与特征缓存协同加速研究——速度、生成质量及组合效应评估**

核心思路是：

- **Scheduler 加速**：减少求解扩散 ODE/SDE 所需的步数，即减少网络前向次数。
- **Cache 加速**：不减少名义采样步数，而是在部分步骤复用历史特征，降低单步计算成本。
- **组合实验**：研究二者能否叠加，以及少步采样是否会让缓存误差更严重。

不过，你目前列出的候选方法**不能直接在一个标准模型上组成完整的 3×3 实验矩阵**：

| 模型 | DDIM / DPM-Solver++ / UniPC | DeepCache | TeaCache | D2Cache |
|---|---|---|---|---|
| **Stable Diffusion 1.5，U-Net** | 标准支持 | 官方支持 | 无官方对应实现 | 无官方对应实现 |
| **FLUX.1-dev，DiT / Flow Matching** | 不能作为标准 `diffusers` 即插即用替换 | 不适用其 U-Net 高层特征复用机制 | 官方支持 | 需要自行从视频模型迁移 |

依据：

- [DeepCache 官方实现](https://github.com/horseee/DeepCache)明确支持 SD、SDXL，并利用 U-Net 的高低层特征复用。
- [TeaCache4FLUX](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/README.md)提供 FLUX 实现，通过 timestep 相关信号决定是否复用 Transformer residual。
- 如果你指的是 CVPR 2026 的 **D2Cache: Second-Order Delta Caching**，其[官方仓库](https://github.com/VG-Huai/D2Cache)目前提供的是 Wan2.1、LTX-Video、Latte、Open-Sora 等**视频扩散**实现，没有现成的 FLUX 文生图版本。
- [diffusers 的 Scheduler 指南](https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/schedulers.md)建议 FLUX 等 Flow Matching 模型使用兼容的 FlowMatch Scheduler，不能直接沿用 SD 的 Scheduler 配置。

因此，我推荐采用以下双主线：

### 主线 A：完整、低风险的 Scheduler × Cache 组合实验

**SD 1.5 + DDIM / DPM-Solver++ / UniPC + DeepCache**

这是课程报告的核心，保证实验完整、可比、可复现。

### 主线 B：现代 DiT 模型上的 Cache 对比与迁移

**FLUX.1-dev + FlowMatchEuler + TeaCache / D2Cache-FLUX**

其中：

- TeaCache 复现官方结果。
- D2Cache-FLUX 是自行迁移的方法，必须标注为 **“our adaptation”**，不能写成官方支持。
- 如果时间或显存不足，D2Cache 只保留机制分析与初步实验，不强行做大规模评测。

这条路线既能完成组合实验，又可能形成一个小创新点：**研究二阶缓存迁移到文生图 DiT 后是否仍然有效**。

---

# 二、研究问题与预期贡献

建议围绕四个研究问题组织项目，而不是只罗列运行结果。

## RQ1：不同 Scheduler 的少步生成能力有何差异？

考察：

- 相同步数下，DDIM、DPM-Solver++、UniPC 的质量差异。
- 达到相近质量时，各自需要多少步。
- 少步设置下，高阶求解器是否更有优势。

## RQ2：Cache 的速度—质量折中如何变化？

考察：

- Cache 越激进，实际跳过多少完整网络前向？
- 加速比是否随缓存阈值单调变化？
- FID、CLIPScore 是否足以反映缓存造成的细节损失？
- 相同 Scheduler、相同种子下，LPIPS 是否随缓存强度增加？

## RQ3：Scheduler 与 Cache 是否存在交互作用？

重点验证：

- 在 50 步时表现良好的缓存参数，到了 10 步是否仍然可用。
- DPM-Solver++、UniPC 的多步历史状态是否对缓存更敏感。
- 不同 Scheduler 的 timestep 分布是否改变缓存命中的位置。
- 组合加速比是否低于两个独立加速比的乘积。

## RQ4：缓存机制能否跨模型架构泛化？

通过 FLUX 实验考察：

- TeaCache 在文生图 DiT 上的速度—质量曲线。
- D2Cache 的二阶差分思想能否迁移到 FLUX。
- 在**相同实际计算量或相同延迟**下，TeaCache 与 D2Cache-FLUX 谁更接近无缓存结果。

最终贡献可以写成：

1. 建立统一的 Scheduler 与 Cache 评测框架。
2. 给出 SD 1.5 上完整的组合实验和速度—质量 Pareto 前沿。
3. 分析采样步数、timestep 分布与缓存误差之间的关系。
4. 实现并初步评估 D2Cache 的 FLUX 文生图迁移版本。

---

# 三、模型、数据集与 Benchmark 选择

## 3.1 模型配置

### 模型一：Stable Diffusion 1.5

建议作为主实验模型：

- 分辨率：`512×512`
- 精度：FP16
- `batch_size=1`
- CFG scale：固定为 `7.5`
- negative prompt：所有配置保持一致，可使用空字符串
- 基线：**DDIM，50 steps，无 Cache**
- Scheduler：
  - DDIM，`eta=0`
  - DPM-Solver++：使用 `DPMSolverMultistepScheduler`，`algorithm_type="dpmsolver++"`、二阶、确定性版本
  - UniPC：二阶，其余参数继承 checkpoint 配置

不要把随机 SDE 版本和确定性版本混在一起，否则相同种子的配对分析会更难解释。

### 模型二：FLUX.1-dev

作为扩展模型：

- 分辨率：`1024×1024`
- 精度：BF16 或 FP16，但所有实验必须统一
- Scheduler：`FlowMatchEulerDiscreteScheduler`
- 基线：**FlowMatchEuler，28 steps，无 Cache**
- 可补充 50 steps 检查步数影响
- Guidance：固定，例如常用的 `3.5`
- TeaCache：复现官方 `rel_l1_thresh` 设置
- D2Cache：实现 `D2Cache-FLUX`

官方 TeaCache FLUX 示例使用 28 步，并给出 `0.25 / 0.4 / 0.6 / 0.8` 等阈值。这些阈值可以作为搜索起点，但**官方加速比不能直接当作你的实验结果**。

> 如果使用 CPU offload、量化或显存优化，必须在所有 FLUX 配置中保持一致。否则测到的可能是 CPU—GPU 传输时间，而不是 Cache 的收益。正式计时时最好不使用 offload。

---

## 3.2 提示词数据集

### A. 开发集：用于调参和排错

从 MS-COCO 中固定选择：

- 100～200 个 prompt
- 每个 prompt 使用 2 个固定种子
- 只用于选择 Cache 参数、检查崩溃和观察失败案例

建议建立确定性的 prompt manifest：

```json
{
  "prompt_id": "coco_000000000009_01",
  "image_id": 9,
  "caption_id": 123,
  "prompt": "...",
  "seeds": [0, 1]
}
```

选择规则要固定，例如：

1. 按 `image_id` 排序；
2. 每张图选择 `annotation_id` 最小的 caption；
3. 保存实际使用的 ID 列表。

不要使用 Python 内置 `hash()` 生成种子，因为它默认不具备跨进程稳定性。

### B. 主评测集：MS-COCO

推荐课程项目采用：

- **COCO 2017 val2017 的 5,000 张图像**
- 每张图选择一个固定 caption
- 生成 5,000 张图像
- 计算 FID-5K、CLIPScore 和效率指标

注意：

- FID-5K 的绝对值不能与论文中的 FID-30K 直接比较。
- 若要严格做 COCO-30K，需要另行固定 30,000 个 COCO 2014 validation 样本及其真实图像参考集，并公开 ID、种子和预处理流程。
- 课程时间有限时，做好统一的 FID-5K，比不规范地声称“FID-30K”更可靠。

### C. DrawBench：定性与人评

使用 200 个 DrawBench prompt：

- 展示复杂文本、空间关系、计数、文本渲染等案例。
- 用于同一 prompt、同一 seed 下的视觉对比。
- 从中抽取 30～50 个 prompt 做人工成对比较。

### D. GenEval：组合能力评测

[GenEval 官方工具](https://github.com/djghosh13/geneval)包含 553 个对象中心 prompt，官方生成方式是每个 prompt 生成 4 张图，可评估：

- 单对象与双对象；
- 数量；
- 颜色；
- 空间位置；
- 颜色归属。

不必对所有组合都跑 GenEval。只选择：

- 无加速基线；
- 最优 Scheduler；
- 最优 Cache；
- 最优组合；
- 一个激进加速配置；
- D2Cache-FLUX / TeaCache 对照。

---

# 四、实验矩阵：分四个阶段进行

不要一开始就跑所有模型、所有参数和 5,000 个 prompt。应按以下顺序逐步扩大。

## E0：正确性与基线复现

目标：

- 所有方法都能生成正常图像。
- 相同配置重复运行时结果和延迟稳定。
- Cache 关闭时与原始 pipeline 一致。
- 每张图生成后 Cache 状态完全重置。
- 保存实际 timestep 序列、完整前向次数和缓存命中次数。

先用 20 个 prompt，每个 1～2 个种子。

---

## E1：纯 Scheduler 实验

在 SD 1.5 上运行：

| Scheduler | Steps |
|---|---|
| DDIM | 10、20、30、50 |
| DPM-Solver++ 2M | 10、20、30、50 |
| UniPC | 10、20、30、50 |

共 12 个基础配置。

分析两个维度：

1. **固定步数比较质量**：例如全部使用 20 步。
2. **固定质量比较成本**：例如找到各 Scheduler 达到某个 CLIPScore 或视觉质量水平所需的最少步数。

Scheduler 在相同步数下通常不会自动带来成倍的端到端加速；它的加速主要来自“用更少步数达到可接受质量”。

---

## E2：单独 Cache 参数搜索

### DeepCache

先固定一个 Scheduler，例如 DPM-Solver++，搜索：

- `cache_interval ∈ {2, 3, 5, 10}`
- `cache_branch_id ∈ {0, 1}`
- steps 先固定为 20 或 50

其中官方常用起点是：

```text
cache_interval = 3
cache_branch_id = 0
```

然后在开发集上选择两档参数：

- **moderate**：质量损失较小；
- **aggressive**：速度更快，但质量损失更明显。

不要直接根据单一示例图选择参数。

### TeaCache

在 FLUX 上搜索：

```text
rel_l1_thresh ∈ {0.25, 0.4, 0.6, 0.8}
```

记录每个阈值实际产生的：

- 完整 Transformer 前向次数；
- Cache 命中次数；
- 平均连续跳过步数；
- 延迟；
- LPIPS；
- CLIPScore。

### D2Cache-FLUX

不要直接沿用 TeaCache 的阈值数值。应当：

1. 先匹配相同完整前向次数；
2. 再匹配相同实际延迟或加速比；
3. 比较对应质量。

因为不同方法中的阈值没有相同量纲，**“阈值都是 0.4”不代表公平比较**。

---

## E3：Scheduler × DeepCache 组合实验

在 SD 1.5 上执行核心矩阵：

| 因子 | 取值 |
|---|---|
| Scheduler | DDIM、DPM-Solver++、UniPC |
| Steps | 10、20、50 |
| Cache | Off、DeepCache-moderate、DeepCache-aggressive |

共：

\[
3 \times 3 \times 3 = 27
\]

个组合。

建议：

- 开发阶段：每个配置 100～200 prompt。
- 组合主表：每个配置 500 个固定 prompt、1 个固定种子。
- 最终质量评测：挑选 6～8 个代表配置扩展到 COCO-5K。

30 steps 可以只保留无 Cache 的 Scheduler 曲线，减少计算量。

---

## E4：FLUX Cache 与 D2Cache 迁移实验

核心矩阵为：

| Scheduler | Cache |
|---|---|
| FlowMatchEuler | None |
| FlowMatchEuler | TeaCache，多档阈值 |
| FlowMatchEuler | D2Cache-FLUX，多档计算强度 |

如果希望进一步做 Scheduler × TeaCache，可以额外引入**明确支持 Flow Matching 的 DPM/UniPC 实现**，但必须单独验证正确性。不要把 SD 的 Scheduler 直接塞进 FLUX 后，仅因“能输出图片”就认为实验有效。

---

## E5：消融与失败案例分析

至少完成以下消融：

### DeepCache

- 不同 `cache_interval`
- 不同 `cache_branch_id`
- 不同采样步数
- 缓存开始和结束阶段：前期缓存、中期缓存、后期缓存

### TeaCache

- 不同累计误差阈值
- 是否强制首尾步完整计算
- 最大连续缓存步数限制
- 固定间隔缓存与动态缓存的比较

### D2Cache-FLUX

- 去掉二阶项
- 固定二阶缩放系数与动态缩放
- 一阶预测与二阶预测
- 相同缓存步数下的质量比较

这些消融比单纯增加更多模型更容易形成有说服力的报告。

---

# 五、效率指标与计时规范

## 5.1 必须记录的指标

- 单图端到端延迟，秒/张
- Denoising loop 延迟
- 文本编码与 VAE 解码时间，可单独统计
- 吞吐率，images/s，可选
- 峰值显存
- 完整 U-Net/Transformer 前向次数
- Cache 命中率
- 名义采样步数与实际模型调用次数
- 加速比

主实验固定 `batch_size=1`，因为动态 Cache 和大 batch 的调度行为可能不同。

## 5.2 两种加速比必须分开

### 总加速比

相对于统一基线：

\[
S_{\mathrm{total}}
=
\frac{T_{\mathrm{baseline}}}
{T_{\mathrm{method}}}
\]

例如：

- SD：DDIM-50、无 Cache；
- FLUX：FlowMatchEuler-28、无 Cache。

### Cache 局部加速比

固定 Scheduler 和步数：

\[
S_{\mathrm{cache}\mid s,N}
=
\frac{T_{\mathrm{no\ cache},s,N}}
{T_{\mathrm{cache},s,N}}
\]

它才表示 Cache 本身带来的收益。

组合实验还可以定义：

\[
\eta
=
\frac{S_{\mathrm{actual}}}
{S_{\mathrm{scheduler}}\times S_{\mathrm{cache}}}
\]

用于观察组合是否低于理论乘积。由于文本编码、VAE、Python 调度和缓存判断都有固定开销，实际总加速比通常不会简单相乘。

## 5.3 计时流程

每个配置：

1. 模型加载完成后再开始测试。
2. 使用相同分辨率和生成参数预热 3～5 次。
3. 清空上一轮 Cache 状态。
4. CUDA 计时前后执行 `torch.cuda.synchronize()`。
5. 正式测量至少 20 次。
6. 报告中位数、均值、标准差，必要时增加 P95。

模型下载、加载、图像写盘不计入端到端生成延迟；但文本编码和 VAE 解码应计入。另报 denoising-only 时间，帮助解释 Cache 的收益。

硬件环境必须记录：

- GPU 型号与显存；
- CPU、内存；
- CUDA、PyTorch、diffusers 版本；
- 模型 revision；
- 是否启用 `torch.compile`、xFormers、FlashAttention、offload；
- 是否有其他进程占用 GPU。

---

# 六、生成质量指标

建议将质量指标分成三类。

## 6.1 数据集级质量与文本对齐

| 指标 | 方向 | 用途 |
|---|---:|---|
| FID | 越低越好 | 生成分布与真实图像分布的差异 |
| CLIPScore | 越高越好 | 文本—图像语义一致性 |
| ImageReward / PickScore | 越高越好 | 人类偏好代理指标，可选 |
| GenEval | 越高越好 | 对象、计数、颜色、位置等组合能力 |

FID 实现建议统一使用 [CleanFID](https://github.com/GaParmar/clean-fid)，并固定 resize、量化、Inception 权重和 `mode`。图像统一保存为 PNG，避免 JPEG 压缩引入额外变量。

## 6.2 Cache 保真度

对于相同模型、Scheduler、steps 和 seed：

\[
I_{\mathrm{ref}}
=
G_{\mathrm{no\ cache}}(p,z)
\]

\[
I_{\mathrm{cache}}
=
G_{\mathrm{cache}}(p,z)
\]

计算：

- LPIPS，越低越好；
- PSNR / SSIM，可作为辅助；
- 必要时记录 latent 或模型输出误差曲线。

其中 **LPIPS 很适合评估 Cache 对原轨迹的偏离**，但不适合单独判断不同 Scheduler 谁的生成质量更高。

## 6.3 人工评价

从 DrawBench 选取 30～50 个 prompt：

- 随机打乱方法顺序和左右位置；
- 分别评价“视觉质量”和“文本一致性”；
- 使用成对偏好：A 胜 / B 胜 / 平局；
- 最好有 3 位评价者；
- 报告胜率、平局率和评价者一致性。

---

# 七、推荐的项目结构

```text
diffusion-acceleration/
├── README.md
├── requirements.txt
├── configs/
│   ├── models/
│   │   ├── sd15.yaml
│   │   └── flux_dev.yaml
│   ├── schedulers/
│   │   ├── ddim.yaml
│   │   ├── dpm_solver_pp_2m.yaml
│   │   ├── unipc.yaml
│   │   └── flowmatch_euler.yaml
│   ├── caches/
│   │   ├── none.yaml
│   │   ├── deepcache.yaml
│   │   ├── teacache.yaml
│   │   └── d2cache_flux.yaml
│   └── experiments/
│       ├── e1_schedulers.yaml
│       ├── e2_cache_sweep.yaml
│       ├── e3_sd15_combination.yaml
│       └── e4_flux_cache.yaml
├── src/
│   ├── pipeline_factory.py
│   ├── scheduler_registry.py
│   ├── caches/
│   │   ├── base.py
│   │   ├── deepcache_adapter.py
│   │   ├── teacache_adapter.py
│   │   └── d2cache_flux.py
│   ├── runner.py
│   ├── seeding.py
│   └── instrumentation/
│       ├── timer.py
│       ├── memory.py
│       └── cache_stats.py
├── scripts/
│   ├── prepare_prompts.py
│   ├── generate.py
│   ├── run_matrix.py
│   ├── evaluate_fid.py
│   ├── evaluate_clip.py
│   ├── evaluate_lpips.py
│   ├── evaluate_geneval.py
│   └── analyze_results.py
├── third_party/
│   └── ...                 # 官方代码，记录 commit 与本地 patch
├── data/
│   └── prompts/
│       ├── dev.jsonl
│       ├── coco5k.jsonl
│       └── drawbench.jsonl
├── outputs/
│   └── <run_id>/
│       ├── resolved_config.yaml
│       ├── environment.json
│       ├── metrics.jsonl
│       ├── cache_events.jsonl
│       └── images/
└── reports/
    ├── figures/
    └── tables/
```

每次运行都保存完整的 `resolved_config.yaml`，不要只保存命令行。至少包括：

- 模型和 revision；
- Scheduler 全部参数；
- 实际 timesteps；
- Cache 参数；
- seed；
- 精度、分辨率和 CFG；
- Git commit；
- 依赖版本；
- GPU 环境。

这样后面的表格和图都可以从 `metrics.jsonl` 自动重建。

---

# 八、D2Cache-FLUX 的迁移方式

这部分可以作为项目的进阶创新，但要控制工作量。

## 8.1 先复用 TeaCache 的工程外壳

TeaCache FLUX 已经解决：

- 在 Transformer forward 中插入缓存判断；
- 保存 previous residual；
- 首尾步强制完整计算；
- 累计相对误差；
- 每个样本结束时重置状态。

D2Cache-FLUX 可以复用这些工程逻辑，只替换：

- 历史模型输出的维护；
- 一阶、二阶差分的构造；
- 跳过步骤时的预测方式；
- 基于 timestep embedding 的二阶项动态缩放。

## 8.2 必须完成的正确性检查

- Cache 禁用后，输出应与原 pipeline 在数值容差内一致。
- 新 prompt 不会读取上一张图的 residual。
- 记录的一阶、二阶差分确实来自真实完整前向。
- 二阶项关闭时能退化到指定的一阶版本。
- 所有缓存索引与实际 scheduler step 对齐。
-  batch、CFG 或 guidance 状态变化时不会错误复用。

## 8.3 公平比较

TeaCache 与 D2Cache-FLUX 至少采用两种比较协议：

1. **相同完整前向次数**：比较 LPIPS、CLIPScore、FID。
2. **相同实测延迟**：比较生成质量。

如果二阶迁移效果不理想，这仍是有效结果。可以分析：

- 文生图步数少于视频，历史差分是否不足；
- FLUX 的 residual 动态范围是否不同；
- 二阶外推是否在少步或高噪声阶段过冲；
- 视频中的时间平滑性是否是原方法有效的重要前提。

---

# 九、结果展示与报告大纲

## 9.1 主结果表

建议统一为：

| Model | Scheduler | Steps | Cache | Full Calls | Latency ↓ | Speedup ↑ | FID ↓ | CLIP ↑ | LPIPS ↓ | VRAM |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| SD1.5 | DDIM | 50 | None | 50 | … | 1.00× | … | … | 0 | … |
| SD1.5 | UniPC | 20 | None | 20 | … | … | … | … | — | … |
| SD1.5 | UniPC | 20 | DeepCache | … | … | … | … | … | … | … |

LPIPS 必须写清楚参考对象；跨 Scheduler 时没有统一的同轨迹参考，不应随意填值。

## 9.2 必做图表

1. Scheduler 的 steps—latency—quality 曲线。
2. Cache 参数的速度—质量 Pareto 曲线。
3. Scheduler × Steps × Cache 的加速比热力图。
4. 不同 Scheduler 下缓存命中位置的 timestep 分布图。
5. 同 prompt、同 seed 的视觉对比网格。
6. 典型失败案例：物体丢失、结构改变、纹理模糊、文本错误。
7. 理论乘积加速比与实际组合加速比对比图。

## 9.3 报告章节

1. 研究背景与问题定义  
2. 扩散采样与 Scheduler 原理  
3. DeepCache、TeaCache、D2Cache 机制  
4. 统一实验框架与兼容性设计  
5. 数据集、指标与计时协议  
6. Scheduler 单因素实验  
7. Cache 单因素与参数消融  
8. Scheduler–Cache 组合实验  
9. FLUX 与 D2Cache 迁移实验  
10. 失败案例、局限性与结论  

---

# 十、六周实施安排

| 周次 | 任务 | 验收标准 |
|---|---|---|
| 第 1 周 | 阅读论文、搭建环境、固定 prompt 与 seed | SD/FLUX 基线可运行，配置完整保存 |
| 第 2 周 | 完成 SD 三种 Scheduler 实验 | 得到 steps—质量—延迟曲线 |
| 第 3 周 | 接入 DeepCache，复现 TeaCache-FLUX | Cache 状态、前向次数、计时均正确 |
| 第 4 周 | 完成 SD 组合矩阵，开发 D2Cache-FLUX | 27 个核心组合和初步迁移结果 |
| 第 5 周 | COCO-5K、GenEval、DrawBench、人评 | 代表配置具有完整质量指标 |
| 第 6 周 | 消融、统计分析、绘图与报告 | 所有表格可复现，结论有证据支持 |

如果只有一张消费级显卡，建议优先级为：

1. **SD 1.5 的完整 Scheduler × DeepCache 实验**；
2. 小样本 TeaCache-FLUX；
3. D2Cache-FLUX 原型；
4. 最后才考虑 COCO-30K 或更多模型。

最终最重要的不是跑出“最大加速比”，而是建立一条可信的**速度—质量 Pareto 前沿**，并解释为什么某些 Scheduler 与 Cache 的组合有效、另一些组合会放大误差。