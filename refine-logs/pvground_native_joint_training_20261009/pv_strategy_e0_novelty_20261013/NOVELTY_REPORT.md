# 当前原生 A/B/C 新颖性与基线重叠审查

审查日期：2026-10-13。结论：**PROCEED WITH CAUTION，5.5/10，通过早期新颖性门槛。** 没有找到一篇已公开论文完整包含当前 A/B/C 的具体结果，因此没有 ABANDON 的依据；但现有证据也不能支持“三项已独立有效的创新”。这是继续检验具体机制的许可，不是论文成熟度或性能保证。

本审查仅针对已实现源代码。未把六面分布回归、新注意力、teacher、质量排序器或尚未实现的责任统一机制算入贡献。请求路线为全新上下文 `gpt-6-astra / max`；实际模型与 effort 没有运行时 attestation，均记为 **UNATTESTED**。`review_independence=same-family`，`acceptance_status=provisional`。

## 判定与证据边界

| 当前主张 | 机制判定 | 能否称为独立有效贡献 |
|---|---|---|
| A：融合前候选—成员 Query-Mask 残差 | 4.5/10，PROCEED WITH CAUTION | 尚缺隔离 A 的同条件正式效果；普通残差 MLP 与已有双 Mask 特征之间的增量较薄 |
| B：真实极值来源条件的逐轴两框混合 | 5.5/10，PROCEED WITH CAUTION | 学习混合有端点证据；“极值来源优于整体支持”没有，strict 反而少 1 |
| C：部署所选、原未匹配、双 Mask 合格候选的几何补充监督 | 6/10，PROCEED（仅机制） | 当前正常训练正式结果未知；不能借旧版 +3 或 control 下滑证明 |

评分的 5 分附近表示有明确近邻，仍存在可核验的增量；不是录用概率。A 的薄处是常规条件残差尚缺状态特征必要性证据；B 的薄处是最针对性的同容量对照未胜；C 的机制组合尚未被所查论文覆盖，效果未出不应被改写成“重复已有论文”。

科学基线始终是 **PV-Ground**。实际实验起点 **PV-Ground + 历史 G task-observation reader** 已经修改，必须独立披露。不能把该起点或下表的 native regression endpoint 称作 original PV。

PV-Ground 原论文的 §3.2–3.3 给出 PVI/VSA 多尺度聚合与 TGS，§3.4 明确采用 MCLN 解码器和多任务预测头，并指向 MCLN 的损失设计。这些均为继承项；MCLN 的 RSA/ASA、REC/RES 协作及 Text/own Query Mask 自适应融合也不是当前创新。[PV-Ground 原论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Shang_PV-Ground_Text-Guided_Point-Voxel_Interaction_for_3D_Visual_Grounding_CVPR_2026_paper.pdf)，[MCLN §3](https://arxiv.org/html/2407.05363v1)。

已核验本地作者/CVF PDF 的题名、六位作者、摘要和方法 §3.1–3.4；查看了题页、图 2 和 §3.4 页面，并非依赖旧第三方摘要。PDF SHA256 为 `ad96d075ca310f5cc48d875074cf5dc6af594938904811be3d734df0416287d2`；两个已下载副本一致。未声称逐页通读所有实验和附录。:codex-file-citation{path="C:/Users/gb/.codex/tmp/pvground_novelty_20261009/PV-Ground.pdf" purpose="source"}

## 三个可核验主张

### A：融合前候选条件的成员 logit 校正

**可保留的一句话增量：**在现有 Text/Query 融合之前，用候选 Query、其原生支持成员及两 Mask 的当前状态，预测 own Query-Mask 的成员 logit 残差。

[实际 A 源码](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/mask_support_corrector.py) 有 27,841 个参数。Query 与支持特征分别由 288 降到 32 维；输入包括 Text、own、fused 概率，Mask 分歧、原 alpha、实际成员计数及原生 member mapping。输出修正自身 Mask，再进入原融合。**初始化的 geometry flag 为 False，9 个显式几何输入恒零**；不能写成已实现几何感知 A。上游内容特征可能含空间信息，不改变这个显式分支未启用的事实。

修正结果写回 `sp_last_pred_masks`，原 segmentation losses 因而监督它；这不是新损失。正常路径中 core/G/backbone/A/B 可训练，RoBERTa 冻结。Mask extent 来自 detached 概率、硬阈值成员和 min/max，不能宣称 extent 计算端到端可微；这也不等于 A 的其他监督路径没有梯度。

MCLN 已有候选 Mask 与自适应双 Mask 融合，3DRefTR 已复用 Query 和 superpoint 特征做 REC/RES；RefineMask 已展示用已有 Mask 与内容特征细化边界。它们没有因此完整覆盖上述候选状态残差，但排除了“首次联合定位分割”“新 Mask 融合”这些宽泛写法。[MCLN](https://arxiv.org/html/2407.05363v1)、[3DRefTR §3](https://arxiv.org/pdf/2308.11887)、[RefineMask §3](https://arxiv.org/html/2104.08569v1)。

**薄处与证据要求：**常规残差 MLP 本身承载的机制新意有限。材料没有足以分离 A 独立最终定位收益的正式对照，B 的端点表不能替代它。若要把状态条件作为贡献，应有同 seed 2027、同原生协议/初始化/预算下 A 的定位与 own-Mask 改善，并说明分歧/候选条件确实有作用。若只能证明额外容量有用，应按轻量候选 Mask 校正定位，避免拔高。

### B：真实极值来源编码与有界逐轴混合

**可保留的一句话增量：**用实际定义预测 Mask 六个边界面的全部极值成员证据，为每个轴学习原回归框与 Mask extent 的共同中心/尺寸混合系数。

[实际 B 源码](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/extremal_span_mixer.py) 有 29,793 个参数。先在预测 fused-Mask 支持集合内寻找各轴 min/max 面，保留所有达到同一极值的并列 superpoints；其内容、概率、实际点数和相对几何组成 face token，两个相对面成对产生三个轴系数。每轴 `clamp[0,1]` 的同一系数同时混合中心与尺寸，等价于该轴两端使用相同混合系数，保持有序几何。它只在两个框端点间插值。

Mask 改善 box 已有 Mask DINO 先例；Not Every Side Is Equal 已有六面几何特征、SoI pooling、面分布和质量估计；D-FINE 已做边界分布细化。B 的差别是**真正达到预测极值的来源成员选择 + 成对逐轴两框混合**。当前 B 没有六个独立分布、teacher、新注意力或质量排序头，不能借那些更强机制的文献叙事扩大实际实现。[Mask DINO §3](https://arxiv.org/html/2206.02777v3)、[Not Every Side Is Equal §3](https://arxiv.org/html/2312.10390)、[D-FINE §4](https://arxiv.org/html/2410.13842v1)。

保留原 256 个候选和原 bbs 选择，没有新增 score。参数少不等于已经测得低运行成本，本轮无 GPU 或时间开销测试。

**薄处与证据要求：**最关键的反证不是“领域拥挤”，而是同容量 whole-support 为 5676/4921，extremal 为 5677/4920：strict 少 1。现有结果支持学习混合相对固定混合的作用，不能把它归因于极值 provenance。若没有同容量优势及与极值/边界错误相连的定位证据，应将来源编码作为实现选择，将已证主张限定为学习混合；不能计为已独立有效的第二创新。六面独立系数旧试验已失败，不是尚未尝试的自然升级。

### C：给部署所选、支持合格的未匹配候选补几何责任

**可保留的一句话增量：**每表达最多为原 native-bbs 所选、未被任何原 Hungarian 匹配使用且 own/fused 两 Mask 均合格的一项候选，增加最终框的原生几何监督。

[实际 C 源码](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/source_overlay_v2/selected_support_geometry_objective.py) 无推理参数。选择索引 stop-gradient；仅考虑原 bbs 第一项。所有原 Hungarian matched Queries 都排除，既不重匹配，也不把 C 资格传播为 CE/NCE 正标签。资格是训练期 own Mask 与 fused Mask 的 GTIoU **都严格大于 0.5**；**没有 final BoxIoU 资格条件**，所以不能描述为专门筛选“好 Mask、坏 Box”。

对该一项的最终框复用原生 box loss：中心 L1 加 0.5 倍尺寸 L1，和 GIoU，系数 10/2，再除 7，并以实际 batch size 作原生归一化。推理没有 GT、teacher、候选剪枝或新 score。旧 selected-mask extra 在当前对照中关闭；官方/G/A/B 按当前同源初始化重新加载，优化器重置，开关 C 是当前正常对照的区别。

MS-DETR、Co-DETR，以及近期 PaQ-DETR/MDS-DETR 已使额外正 Query 监督和一对多训练成为明确先例。3DRP-Net 的 BoxIoU 软标签也排除了宽泛“根据几何质量重新监督候选”的首创主张。但所读方法没有包含当前 **部署 top-1 + 双 Mask 资格 + 原匹配全部保护 + 仅最终几何** 的完整限制组合。HYDRA 已研究候选池与部署选择的缺口，却训练选择器并校准输出动作；当前 C 保留 bbs，不能把两者混同。[MS-DETR](https://arxiv.org/html/2401.03989v1)、[Co-DETR](https://arxiv.org/html/2211.12860v6)、[PaQ-DETR](https://arxiv.org/html/2603.06917)、[MDS-DETR](https://arxiv.org/html/2605.23507)、[3DRP-Net](https://aclanthology.org/2023.emnlp-main.656.pdf)、[HYDRA](https://arxiv.org/html/2609.20283)。

**机制可继续，效果待定。**当前正常 C 没有正式训练后评测；旧广泛额外候选版的冻结头 strict +3 不能移植。应先以正在进行的正常同条件对照的正式端点判断 C，而不是再开新机制。相对下降 control 的恢复与超过 retained parent 必须分开。若要声称这些资格约束本身必要，需要对应的监督覆盖和定位改善证据；若只有等价于一般额外 box loss 的收益，应把机制解释收窄。本报告没有启动或授权新试验。

## 最接近文献及实际差异

下表只把方法层面相关之处作为重叠；主题接近、并发或竞争本身不构成否决。逐篇阅读范围见 JSON 的 `closest_work[].read_scope`。

| 原始论文/作者来源 | 已有机制或重叠 | 当前实际差异及边界 |
|---|---|---|
| [PV-Ground](https://github.com/AaNnWwTt/PV-Ground) (CVPR 2026) | PVI/VSA 多尺度点—体素特征及 TGS；§3.4 明确采用 MCLN 解码器、多任务预测头，并指向其损失设计。 | A/B/C 均位于现有特征与候选体系之上；不能把原多尺度或 TGS 再算创新。 |
| [MCLN](https://arxiv.org/html/2407.05363v1) (arXiv v1（本轮使用版本）) | RSA、REC/RES 分支、ASA 与自适应 Text/Query Mask 融合。 | A 只增加融合前候选—成员残差；B 再作用于最终框坐标；这些增量才是审查对象。 |
| [3DRefTR](https://arxiv.org/pdf/2308.11887) (arXiv v2（本轮使用版本）) | 复用 REC Query 和 superpoint 特征产生候选 Mask，并联合 REC/RES。 | 复用 Query/支持特征不是 A 的新增点；A 的状态条件残差及 B/C 接口不同。 |
| [Mask DINO](https://arxiv.org/html/2206.02777v3) (arXiv v3（CVPR 2023 方法）) | Mask 生成更好的 box 初始化；class/box/mask 联合匹配。 | B 在输出端做来源条件的逐轴两框混合；C 保留原匹配。把 matcher 改成 own-mask 本身无首创性。 |
| [RefineMask](https://arxiv.org/html/2104.08569v1) (arXiv v1（CVPR 2021 方法）) | 用实例/语义特征和已有 Mask 多阶段细化边界。 | A 在原 superpoint 粒度，用 Text/own Mask 状态修正候选 logit；当前没有多分辨率或边界专属细化。 |
| [Not Every Side Is Equal](https://arxiv.org/html/2312.10390) (ICCV 2023) | 六面几何证据、SoI pooling、面位置分布与质量估计。 | B 选真正达到预测极值的成员（保留全部并列者），再按轴混合两框；无六面分布、质量头或 teacher。 |
| [D-FINE](https://arxiv.org/html/2410.13842v1) (arXiv v1（本轮使用版本）) | 边界分布残差细化和定位自蒸馏。 | 当前 B 仅三个有界混合系数；当前 A 的 logit residual 不是边界坐标分布。 |
| [MS-DETR](https://arxiv.org/html/2401.03989v1) (arXiv v1（CVPR 2024 方法）) | 同一解码器 Query 的额外一对多监督，按分类与 box IoU 选正样本。 | C 仅给 native-bbs 第一且原未匹配、双 Mask 合格的一项最终几何损失，排除原匹配项。 |
| [Co-DETR](https://arxiv.org/html/2211.12860v6) (arXiv v6（ICCV 2023 方法）) | 辅助一对多头和定制正 Query 增强训练监督。 | C 不引入辅助预测头或新 Query；只改变现有部署候选的几何责任。 |
| [3DRP-Net](https://aclanthology.org/2023.emnlp-main.656.pdf) (EMNLP 2023) | 方向关系注意力；以 BoxIoU 排名构造 referring-score 软标签。 | C 不改分数标签，也不用最终 BoxIoU 准入；不是该软标签策略的复述。 |
| [From Alignment to Fusion in 3D Vision-Language](https://arxiv.org/html/2609.28222) (arXiv 2609.28222) | 对齐后融合、多表示协作；最新相近任务工作。 | 它对图像/点/体素特征做正交映射和全局可学习权重；不是候选 Mask 残差或极值框混合。 |
| [PaQ-DETR](https://arxiv.org/html/2603.06917) (arXiv 2603.06917) | 质量条件的一对多正样本数量与选择；终层保留一对一。 | 它按 box IoU/分类置信度在中间层扩展监督；C 在最终部署 Query 上用双 Mask 资格，几何单项补充。 |
| [MDS-DETR](https://arxiv.org/html/2605.23507) (arXiv 2605.23507) | 单解码器一对多/一对一监督及按置信度排序。 | 它以置信因果 Mask 自注意力抑制重复；C 没有此结构或新增排序模块。 |
| [HYDRA / Queries Knew More Than We Thought](https://arxiv.org/html/2609.20283) (arXiv 2609.20283) | 指出候选池与部署选择间的缺口，使用 Mask 状态/几何等候选特征。 | 它学习冻结候选 selector 并校准替换动作；A 改 Mask，B 改坐标，C 保留原 bbs 只补训练几何。 |
| [Detector-Augmented SAMURAI](https://openaccess.thecvf.com/content/WACV2026W/RWS/papers/Lenhard_Detector-Augmented_SAMURAI_for_Long-Duration_Drone_Tracking_WACVW_2026_paper.pdf) (WACV 2026 Workshop) | 融合检测框和 Mask 派生框以校正输出。 | 该方法采用追踪/检测条件整合；没有证据表明它含 B 的真实极值成员编码及逐轴学习混合。 |

2026 年 From Alignment to Fusion 在方法 §IV-B–IV-C 中处理图像/点/体素特征的正交对齐，并以全局可学习权重融合；该权重不是候选条件的 Mask/框操作。因此是需要引用的近邻，不能把“adaptive fusion”同名当成覆盖 A/B 的证据。[方法原文](https://arxiv.org/html/2609.28222)。

## 现有数值支持什么

同 9508、seed 2027、无 ensemble。以下为父级提供的冻结父模型正式保存端点、独立 CPU 重计数；本审查只复核计数关系，未重新运行推理或亲自重计全量预测。此表不等于正常联合训练的贡献拆分。

| 冻结父模型中的输出方案 | Acc@0.25 命中 | Acc@0.5 命中 |
|---|---:|---:|
| native regression endpoint（修改起点） | 5615 | 4495 |
| 校正 fused-Mask extent | 5606 | 4881 |
| 固定 50/50 | 5675 | 4857 |
| 学习 whole-support，同容量 | 5676 | 4921 |
| 学习 extremal-support，当前保留 | 5677 | 4920 |

extremal 比固定混合为 +2/+63，比 whole-support 为 +1/−1。不能把前者的收益写成极值来源本身独立有效。

当前 retained 为 **59.7076/51.7459**，未达到 **60/52**。同 9508 的最低达标整数计数是 **5705/4945**，尚差 **28/25**。也未完成同完整架构的 Nr3D/Sr3D 独立原生训练目标。

其他历史与当前状态必须分开：

- 当前正常 control strict 为 4920→4620→4596，连续两次下降后停止。这既不证明 C 受益，也不证明失败由某种责任冲突导致。
- 当前正常 C 的正式终点未知；任务开始时首个 epoch 在运行，本审查没有查询实时状态或推断结果。
- 六个独立 face 系数正常试验为 4920→4599→4625→4476，不能列为未试方向。
- 旧 broad 高 Mask/低 Box 候选额外监督仅在冻结头上 +3 strict，不代表当前 C。
- 父级补充的普通参考保持 λ0/1 正式结果均为 5593/4832，起点 5598/4848；本轮未独立调取这项全部收据，只作历史排除。不能把普通退化惩罚重新包装为全新未试机制。

## 源码接口事实、历史检查与未证假设

当前 R4 的 [pv_ground.py](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/source_overlay_v2/models/pv_ground.py:549) 将一个 Text mask 广播到 256 Query 的 `pred_masks`；[matcher](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/source_overlay_v2/models/losses.py:294) 的 mask cost 权重为 0.0002，实际读取该字段。因此对同一个 GT 列，mask cost 无法区分候选行；class/GIoU 仍为 Query-specific（1/2），不能说整个 Hungarian 匹配无效。own `sp_pred_masks` 在另一条原损失路径训练，不能说 own Mask 未训练。源码与绑定说明 SHA 一致。

历史 own-mask matcher **已经试过有限真实检查，但当前未启用**。[发布记录](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/terminal_matcher_closeout_publication_20261011.json)及所引的 [PAIR_ANALYSIS](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/terminal_matcher_closeout_20261011/MATCHER_PAIR_ANALYSIS.json)、[Text 收据](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/terminal_matcher_closeout_20261011/MATCHER_TEXT_RECEIPT.json)、[Query 收据](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/terminal_matcher_closeout_20261011/MATCHER_QUERY_RECEIPT.json)显示：两种模式各跑相同 2 个增强 batch、各 16 行/2 次原生更新；两模式共 4 步，GT 匹配对变化均为 0，mask cost 数值改变，观察行全部单 GT。不是 32 个独立输入，也没有后续长训练或正式准确率。该结果只覆盖这些输入与步骤，不证明全数据无收益。Mask-aware matching 又已有 Mask DINO 先例；替换字段本身不能作为原创性依据。[Mask DINO hybrid matching](https://arxiv.org/html/2206.02777v3)。

历史 G 对未匹配且 final BoxIoU>.5 的候选替换 last CE 的 no-object 项；原 contrast 仍按 not-mentioned 处理。C 用双 Mask 条件补几何且不改这些语义目标。这是**可直接核验的职责差异**，不是测得的梯度冲突。当前 R4 没有本审查所得的运行时成本矩阵或梯度夹角证据。

因此，匹配/语义/对比/几何资格是否一致，只能作为**源码引出的待诊断机会**单列；不存在已实现的统一责任分配新模块，也不能作为第四贡献，更不能由此解释 control 的下降。上述有限历史 matcher 检查必须随该机会同时披露。

当前初始化及 G/C 有显式 ScanRefer 约束，原 loss 的语义系数等也随数据集协议不同。不能仅修改数据集名称就声称 Nr3D/Sr3D 已完成同架构原生验证。遵守用户单 seed 2027 要求，不提出多 seed；跨数据集本身亦非新颖性。

## 检索、可复核性与建议定位

父级原始检索覆盖每条主张至少三种表述、2024–2026 与最近 6 个月，并含 Scholar/Semantic Scholar 网页索引。本轮另查候选 logit 残差、Mask-derived box fusion、extremal/side-aware geometry、unmatched/selected Query supervision，并核查 arXiv、CVF、OpenReview、PMLR/ICML 与 NeurIPS 相关索引；补读上表方法，而非只按标题判定。最近窗口为 2026-04-13 至 2026-10-13；PaQ-DETR 早于该窗口，仍因方法相关纳入。

没有可用的专用 Scholar/Semantic Scholar 数据库工具；网页索引检索不是全数据库穷举，“本轮未检出完整重复”不等于绝对不存在。部分作者页直开失败时注明具体已读范围；PV-Ground 后续已由本地原 PDF 完成方法核验。HYDRA 的索引日期和编号月份不完全一致，未据此推断优先权。Detector-Augmented SAMURAI 只依据作者论文被索引的 §3.2 作宽泛框融合先例，没有声称它含 B 的逐轴学习。

建议定位为：**在 PV-Ground 及明确披露的历史 G 起点上，研究融合前候选 Mask 校正、真实极值来源驱动的逐轴框混合，以及部署所选未匹配候选的有限几何监督；保持原候选池与 bbs 规则，并分别检验其作用。** 三项机制可以继续接受后续证据检验；现在不能写成“三项各自显著有效”，也不能承诺达到 60/52。

本轮没有 SSH、GPU 调用、神经网络 forward、训练状态查询、源码编辑、checkpoint 变更或新训练。源码/证据 SHA、完整请求、补充消息、原始检索响应和报告冻结摘要均保存在本目录。[结构化报告](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/novelty_native_abc_20261013/NOVELTY_REPORT.json)、[来源 SHA 清单](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/novelty_native_abc_20261013/REVIEW_SOURCE_MANIFEST.json)、[审查 trace](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/novelty_native_abc_20261013/.aris/traces/novelty-check/2026-10-13_run01/001-native-abc.request.json)。

