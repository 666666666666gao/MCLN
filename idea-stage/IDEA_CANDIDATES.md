# MCLN 正交方法候选清单

**方向**：在不复活 A‑V4/FPR Gate、不使用数据集 ID 或 GT anchor sidecar 的前提下，提高 Nr3D/Sr3D，并保持与 ScanRefer 相同的总体网络接口。

**证据起点**：Nr3D 正式最好 `4475/7899=56.6527%`；Top‑2/5/16 oracle 为 `61.6787/69.0974/80.3013%`；有 `2068` 条排序失败与 `1556` 条 proposal failure。A‑V4 Fold‑4 完整训练后 @.25 净 `-20`、@.50 净 `-151`，已经封存。

## 机械可行性门

十个候选均可在现有接口上实现，单个短审计预计不超过 2 GPU 小时，没有因算力或数据不可用而淘汰。当前代码已提供：

- `point_clouds [B,50000,6]` 与原始 superpoint 映射；
- `seed_xyz/seed_features`（约 1024 点）；
- `query_points_feature [B,288,256]` 与 query xyz；
- `text_memory`、`proj_tokens`；
- target/entity/attribute/relation/anchor structured slots；
- detector-valid axis 与 V99 `selected_source_scores`。

## 候选与已知近邻

| ID | 候选 | 实际改动 | 对应失败 | 主要近邻与新颖性风险 |
|---|---|---|---|---|
| A | Same-class Distractor Capsule | 为每个候选聚合同类邻居的相对尺寸、颜色、点密度、支撑面、距离和序数统计，再在主 scorer 前融合 | Top‑K 内同类实例选错 | EG‑3DVG、Multi‑Attribute Transformer、InstanceRefer、TransRefer3D；重叠高 |
| B | Text-conditioned Proposal Query Seeding | 保留 256 个原 Query，额外生成 4–8 个 target/attribute/anchor 条件查询 | 1556 proposal failures、小目标 | 3D‑SPS、TSP3D；重叠高，可能只是增量实现 |
| C | Multi-frame Relation Evidence Tensor | 在 room、speaker、anchor-local、candidate-local 坐标系并行编码 candidate–anchor 几何 | viewpoint、left/right/front/back | Viewpoint‑Aware、ViewSRD、ORD；重叠很高 |
| D | Candidate-local Evidence Rehydration | 仅对 Top‑K box 内、1.5×、2×范围重新读取原始点/超点，生成局部 micro-geometry token | 稀疏、小体积、细粒度属性 | CFA、TSP3D、PointRCNN 式 RoI refinement；中高重叠 |
| E | Clause-complete Latent Anchor Matching | 对 target/attribute/relation 子句分别打分，每个关系对预测 anchor 集做潜变量匹配，以最弱未满足子句约束候选 | 长句、多 anchor、部分匹配 | CSVG、ViewSRD、ORD、NS3D；高重叠 |
| F | Phrase-slot Dual-stream Evidence | target、attribute、relation、anchor 四路独立 cross-attention，再融合 | 13+ token 长句 | EDA、Fine-Grained Spatial/Verbal Losses、DASANet；中高重叠 |
| G | 2D Appearance Evidence Bank | 将 ScanNet 多视角 RGB 的冻结 appearance feature 投影到 3D proposal | color/material/state 同类消歧 | 多模态 2D–3D 3DVG 很多；额外模态破坏简洁统一性 |
| H | Room-layout Landmark Tokens | 从点云提取墙、门、角点、地面、Manhattan axes，作为 layout anchor tokens | 以房间/入口/墙为参照的描述 | Viewpoint/room-frame/ORD 有近邻；仍可能存在 layout-token 差异 |
| I | Support-surface Relation Evidence | 构造接触、包含、上下支撑图，并与 on/under/inside 等关系对齐 | 小物体依赖支撑上下文 | 关系图与手工几何近邻多；可能被认为规则工程 |
| J | Scene-native Structured Pre-alignment | 用训练场景中可验证的类别、相对位置、支撑、序数事实做短结构 prompt 预对齐，再正常 REC | relation/attribute 监督稀疏 | AugRefer、scene-verified negatives；数据构造重叠高 |

## 当前文献地图

- [EG-3DVG, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Park_EG-3DVG_Expression_and_Geometry_Aware_Grounding_Decoder_for_3D_Visual_CVPR_2026_paper.html)：expression/geometry-aware decoder 与 intra-class confusion。
- [ORD, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_ORD_Object-Relation_Decoupling_for_Generalized_3D_Visual_Grounding_CVPR_2026_paper.html)：anchor-centric relation decoupling。
- [ViewSRD, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_ViewSRD_3D_Visual_Grounding_via_Structured_Multi-View_Decomposition_ICCV_2025_paper.pdf)：多 anchor 分解与多视角交互。
- [TSP3D, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Text-guided_Sparse_Voxel_Pruning_for_Efficient_3D_Visual_Grounding_CVPR_2025_paper.html)：文本引导稀疏体素与小/薄目标信息补全。
- [Fine-Grained Spatial and Verbal Losses, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Dey_Fine-Grained_Spatial_and_Verbal_Losses_for_3D_Visual_Grounding_WACV_2025_paper.html)：视觉 offset 与文本 span 监督。
- [Multi-Attribute Interactions Matter, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Multi-Attribute_Interactions_Matter_for_3D_Visual_Grounding_CVPR_2024_paper.html)：多属性交互与因果分析。
- [Viewpoint-Aware 3DVG, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shi_Viewpoint-Aware_Visual_Grounding_in_3D_Scenes_CVPR_2024_paper.html)：显式说话者视角预测。
- [CFA, Neurocomputing 2024](https://www.sciencedirect.com/science/article/pii/S0925231224009664)：高分辨率局部点聚合，重点缓解小目标下采样损失。
- [3D-SPS, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Luo_3D-SPS_Single-Stage_3D_Visual_Grounding_via_Referred_Point_Progressive_Selection_CVPR_2022_paper.html)：文本引导关键点渐进选择。

## 不可复活项

- A‑V4/FPR 的 parent switching、counterfactual parent、阈值/margin/Top‑K/loss 变体；
- 已消费 Fold‑4 上的任何调参；
- fair baseline reproduction、旧 Section 7、旧 Section 8、E0–E7；
- dataset ID、Unique/Multiple 标签、GT-derived anchor sidecar；
- 未经独立查新和人工决策自动启动 GPU 实验。

## 待独立评审问题

1. 十个候选中是否存在高于“把已有模块接到 MCLN”的真实技术差异？
2. 若新颖性均有限，哪一个仍最直接对应现有失败证据、最值得做一次有界诊断？
3. 能否把候选重构为一个更小、可证伪、三数据集共用的统一机制？

## 2026-09-01 独立查新结论

完整报告见 `idea-stage/NOVELTY_CHECK_EDG_20260901.md`。结论为 **CONDITIONAL，接近 NO-GO**：

- A--J 中的局部点重读、同类干扰物对比、文本子句/anchor 分解与统一单次 scorer 均已有直接近邻，不能单独写成创新；
- 唯一可能保留的是 `candidate-pair-specific counterfactual evidence necessity supervision`：在固定 GT--困难负样本对上，监督移除真正必要的 evidence 会降低 margin，而移除无区分力 evidence 不改变排序；
- 该机制必须用新的 train-only scene-disjoint mini-fold，对比同特征 concat 与 random branch dropout/普通 counterfactual loss；没有 `+0.8pp`、至少 `3%` ranking-failure 净修复且易样本退化小于 `0.2pp`，立即封存；
- 当前只完成查新和证伪合同，**没有授权 GPU 实验**。
