# EDG / CENS 新颖性核验报告（2026-09-01）

## Proposed Method

候选方法最初称为 Evidence-Deficit Grounding（EDG）：在 V99 主 scorer 内联合 candidate-local 高分辨率点证据、同类干扰物对比证据与 target/attribute/relation 子句证据，并在训练时逐支路移除证据，约束 GT 与困难负样本之间的排序 margin。

查新后只保留一个可能成立的窄核心，改称 **Candidate-pair Counterfactual Evidence Necessity Supervision（CENS）**：对每个 GT--hard-negative candidate pair，监督哪一类证据对区分这一对候选是必要的；不把三类常规 evidence expert、本地高分辨率点或统一单次推理本身写成创新。

## Core Claims

| Claim | Novelty | Closest prior work | Verdict |
|---|---|---|---|
| C1：candidate-local raw/high-resolution point rehydration | LOW | PV-Ground、TSP3D、3D-SPS、Refer-It-in-RGBD | 已有工作已经用高分辨率 point/voxel、文本引导采样、局部 proposal refinement 缓解下采样损失；不能作为创新。 |
| C2：same-class distractor contrast capsule | LOW | EG-3DVG、TransRefer3D、CORE-3DVG、3DVG-Transformer | 同类候选消歧与同场景同类别困难负样本已是成熟路线；“capsule”命名没有技术差异。 |
| C3：target/attribute/relation/anchor clause evidence | LOW | EDA、G3-LQ、ViewSRD、ORD、Fine-Grained Spatial and Verbal Losses | 文本分解、属性/位置线索、multi-anchor 拆解与 predicate decoupling 已被直接覆盖。 |
| C4：candidate-pair-specific counterfactual evidence necessity | MEDIUM-LOW | Multi-Attribute Interactions Matter 的 counterfactual causal analysis，以及通用 branch/modality dropout | 尚未发现完全相同的“对某一 GT--困难负样本对，移除必要证据必须降低 margin，而移除无区分力证据应保持排序”的 3DVG 公式；但与反事实注意力、feature dropout 和 margin regularization 非常接近。只有严格 pair-specific 定义和直接实验增益才能保留。 |
| C5：C1--C4 合入同一主 scorer、无 post-hoc reranker、三数据集统一 | LOW | MCLN、G3-LQ、TSP3D、PV-Ground、EG-3DVG、ViewSRD、ORD | 单次主模型与跨 ScanRefer/Nr3D/Sr3D 评估是常见范式；这是工程合同，不是新颖性贡献。 |

## Closest Prior Work

| Paper | Year / Venue | Overlap | Remaining delta |
|---|---|---|---|
| [EG-3DVG](https://openaccess.thecvf.com/content/CVPR2026/html/Park_EG-3DVG_Expression_and_Geometry_Aware_Grounding_Decoder_for_3D_Visual_CVPR_2026_paper.html) | 2026 / CVPR | expression/geometry-aware decoder、同类混淆、expression-aware contrastive learning | CENS 若成立，差异只能是 pair-specific evidence necessity target，而非 contrastive/local geometry 本身。 |
| [PV-Ground](https://openaccess.thecvf.com/content/CVPR2026/html/Shang_PV-Ground_Text-Guided_Point-Voxel_Interaction_for_3D_Visual_Grounding_CVPR_2026_paper.html) | 2026 / CVPR | 高分辨率 point--voxel 交互、文本引导 keypoint sampling | C1 没有独立新颖性。 |
| [ORD](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_ORD_Object-Relation_Decoupling_for_Generalized_3D_Visual_Grounding_CVPR_2026_paper.html) | 2026 / CVPR | target--anchor relation decoupling、predicate-only alignment、anchor-guided regression | C3 没有独立新颖性。 |
| [ViewSRD](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_ViewSRD_3D_Visual_Grounding_via_Structured_Multi-View_Decomposition_ICCV_2025_paper.html) | 2025 / ICCV | multi-anchor query 分解、single-anchor statements、多视角融合 | clause/anchor 分解不能作为新贡献。 |
| [TSP3D](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Text-guided_Sparse_Voxel_Pruning_for_Efficient_3D_Visual_Grounding_CVPR_2025_paper.html) | 2025 / CVPR | 文本引导稀疏体素 pruning 与 completion-based addition | local/detail recovery 高重叠。 |
| [Fine-Grained Spatial and Verbal Losses](https://openaccess.thecvf.com/content/WACV2025/html/Dey_Fine-Grained_Spatial_and_Verbal_Losses_for_3D_Visual_Grounding_WACV_2025_paper.html) | 2025 / WACV | candidate spatial offset 与 word-span 细粒度监督 | C3 与普通细粒度 loss 叙事接近。 |
| [Multi-Attribute Interactions Matter](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Multi-Attribute_Interactions_Matter_for_3D_Visual_Grounding_CVPR_2024_paper.html) | 2024 / CVPR | 多属性交互、counterfactual attention、causal effect supervision | C4 最危险的直接近邻；必须证明不是 branch-level counterfactual attention 的改写。 |
| [G3-LQ](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_G3-LQ_Marrying_Hyperbolic_Alignment_with_Explicit_Semantic-Geometric_Modeling_for_3D_CVPR_2024_paper.html) | 2024 / CVPR | fine-grained language-guided queries、geometry/semantic alignment | C3/C5 高重叠。 |
| [Distilling Coarse-to-Fine Semantic Matching](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Distilling_Coarse-to-Fine_Semantic_Matching_Knowledge_for_Weakly_Supervised_3D_Visual_ICCV_2023_paper.html) | 2023 / ICCV | Top-K 候选逐个 masked-keyword reconstruction | candidate-specific language verification 不是空白方向。 |
| [3D-SPS](https://openaccess.thecvf.com/content/CVPR2022/html/Luo_3D-SPS_Single-Stage_3D_Visual_Grounding_via_Referred_Point_Progressive_Selection_CVPR_2022_paper.html) | 2022 / CVPR | referred point progressive selection、文本引导 proposal/keypoint | C1 的早期直接近邻。 |

## Strongest Rejection Argument

EDG 把 2023--2026 年已经拥挤的四条路线放在一起：高分辨率 point/voxel evidence、same-class distractor contrast、target/attribute/relation/anchor decomposition，以及 counterfactual/causal auxiliary loss。若 C4 没有由 GT--hard-negative pair 明确定义的 evidence-necessity target，整套方法就是“已有 evidence modules + branch dropout + margin loss”，组合本身不足以构成论文创新。

## Overall Novelty Assessment

- Score：**3/10（原 EDG）**；收缩后的 CENS 仅为 **有条件继续**。
- Recommendation：**PROCEED WITH CAUTION**，并把未通过证伪实验时的终态预注册为 ABANDON。
- Key differentiator：只有“candidate-pair-specific evidence necessity”可能是差异；C1、C2、C3、C5 均不得作为独立创新点。
- Main risk：审稿人会引用 Multi-Attribute Interactions Matter、EG-3DVG、ORD、ViewSRD、PV-Ground 与 TSP3D，将其判为反事实注意力/feature dropout 在 3DVG 上的增量组合。

## Minimal Defensible Formulation

对固定的 GT 候选 (q^+) 与当前模型高分错误候选 (q^-)，三个常规 evidence expert 输出局部几何、同类对比和语言子句证据。核心监督不是随机丢支路，而是：

1. 若 expert (k) 在该候选对上提供主要可区分信息，移除 (k) 后的 (s(q^+)-s(q^-)) 必须显著下降；
2. 若 expert (k) 对该候选对不具区分力，移除 (k) 不应改变正负排序；
3. 推理只执行完整 evidence 的一次主 scorer 前向，不使用 gate、threshold、parent switch 或 post-hoc reranker。

论文若继续，只能把贡献写成 CENS；三类 expert 是实现载体和消融对象。

## One Bounded Falsification Experiment

当前只允许形成实验合同，不自动授权 GPU：

- 使用新的、未消费的 Nr3D train-only scene-disjoint mini-fold；不访问 7,899-row formal validation，不调 threshold；
- 冻结 V99 backbone，或只训练小 evidence heads；固定 hard negatives 为 protected V99 中 proposal-present ranking failures；
- 同参数直接比较：`C1-C3 concat`、`C1-C3 + random branch dropout/普通 counterfactual loss`、`C1-C3 + CENS`；
- 只评估单次前向 Top-1，不使用 gate、switch 或 fallback；总预算不超过 2 GPU 小时。

同时满足才算通过：

1. CENS 相对 concat baseline 的 held-out Overall Acc@0.25 至少 `+0.8pp`；
2. proposal-present ranking failures 的净修复至少为 baseline ranking failures 的 `3%`；
3. 原本容易/已正确样本下降小于 `0.2pp`；
4. CENS 必须同时优于 random branch dropout/普通 counterfactual loss；auxiliary attribution 改善但 Top-1 不改善不算通过。

任一条件不满足，C4 归类为普通 regularizer，路线封存；不得转入完整 Nr3D、Sr3D 或 ScanRefer 重训。

## Search Audit

每个 claim 均用至少三类检索式覆盖 CVF、arXiv 和 OpenReview，年份重点为 2024--2026：

- C1：`candidate local high resolution point features`、`ROI point rehydration`、`raw point local proposal refinement`；
- C2：`same class distractor contrastive candidate`、`intra-class hard negative`、`same-category relational contrast`；
- C3：`clause target attribute relation decomposition`、`phrase slots anchor relation`、`structured language multi-anchor query`；
- C4：`leave-one-evidence-out margin`、`evidence dropout causal intervention`、`leave-one-modality-out consistency`；
- C5：`unified ScanRefer Nr3D Sr3D scorer`、`single-forward no reranking`、`cross-dataset unified model`。

独立 GPT-5.5 xhigh reviewer 的最终 verdict 为 **CONDITIONAL，接近 NO-GO**。完整 reviewer prompt/response 保存在项目本地 `.aris/traces/novelty-check/2026-09-01_run01/`，该目录按 trace 协议不提交 Git。
