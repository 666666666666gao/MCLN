# PV-Ground当前研究目标（2026-10-06T23:26:37.316629+08:00）

用户本次目标替代“接近V99即可进入Nr/Sr”的旧准入条件，当前状态未完成。

1. ScanRefer同一完整模型、完整9508条原生验证：Acc@0.25严格超过59.1%，Acc@0.5严格超过50.1%，即至少5620/4764个命中。不得拼接不同检查点的两列指标。当前保留5598/4848，宽松项还差22条，严格项已达本项门槛。
2. 三项有效模块支撑论文：每项职责明确，有匹配的直接消融和实际输出/学习贡献；不将零输出、不使用或未验证分支写成有效创新。候选方向为预测支撑空间参考、真实边界观测精修、参考约束的跨任务几何学习，后两项仍待验证，不能预先计为三项已成立。原PV和已有G机制如实披露，已有前作不重新包装。
3. 前两项满足后固定完整模型，再分别加载作者对应的Sr3D／Nr3D预训练权重，接入同一最终三模块结构独立训练；对应原版baseline采用相同作者核心起点。三个基准报告同一最终方法版本；不把Scan权重直接测试结果称为Nr/Sr训练结果。既有Nr/Sr接口及R2预检源码作为准备保留，正式训练现在不启动。

PV-Ground为baseline；单一路径last/bbs、保留全部256候选，同一Query输出Box和Mask，不恢复V99双源/侧链。固定seed2027，不做多seed搜索、挑选或集成。A100任务串行，按实测估算时间，到了约定时间再查，接近结束复查间隔180～300秒。

保留当前Scan最佳及必要官方PV/原G/V99依赖，及时清理用户已授权范围内的自生成、闭合、非最佳权重。历史负实验的文本和指标仍保留。当前最优50.9886%来自本轮零更新Mask空间参考，不称学习式精修/蒸馏增益。

## Historical contract through2026-10-02 (superseded; evidence preserved)

# Current research contract — 2026-10-02

PV-Ground is the paper baseline; verified original G5615/4495 is the common strong start.
The joint and semantic-only P2 adaptations are completed negative results: formal
5613/4419 and5588/4439 versus same-budget G5600/4452. Do not reuse their endpoints.
Raw-point candidate-aligned P3 is running at fixed3723 updates; complete6887 E0
REC/input equivalence is verified, Mask differences remain unresolved. No trained
P3 terminal or9508 result exists yet; current source/configuration stay unchanged.

User priority is ScanRefer Acc@0.5, aiming4754/9508(50.0%) while recording
Acc@0.25 and native Mask costs. Preserve original G loose5615 and strict4495.
Next prepare the same-tail raw-versus-predicted-fused-Mask support comparison in
docs/PVG_FUSED_SUPPORT_PLAN_2026-10-02.md. Keep6D residual/nativebbs; no new
V99 inference chain, general P2 repeat, teacher, quality loss or boundary-distribution
change in that first comparison. Planned mechanisms are not completed contributions.
Each result requires actual GT evaluation, same-start/same-budget control, full
fit-order evidence and independent integrity review. After success freeze the
method for independent Nr3D/Sr3D training. No new cross-benchmark result exists.

## Historical contract (superseded experiment; evidence remains archived)

User priority: return to protected MCLN/V99 family; improve Nr3D beyond native
MCLN while preserving ScanRefer and Sr3D protected results. No multi-seed or
long baseline reproduction. Mask is diagnostic, not a promotion gate.

Current bounded experiment is specified in
`docs/NR3D_SEMANTIC_ASSIGNMENT_2026-09-21.md`. Its proposed claim is that
training-only final-layer root-token replacement for geometrically qualified,
unmatched Nr3D candidates can improve native grounding. This is unproven.

Evidence required: real native-loss integration and optimizer preflight,
same-start/same-budget/same-seed native-versus-replacement training, complete
7899-row REC with unchanged inference, separate candidate coverage and selection
analysis. Synthetic checks alone do not establish a training or accuracy claim.

Protect root/other-GT matched Queries and joint detection prompts. Do not
introduce GT at inference or claim box IoU proves semantic identity. Preserve
the original geometric assignment and all other losses. E57 has no optimizer
state, so this is fresh adaptation from protected weights.

Current native Nr3D reference is 59.82/51.38; strict exceed requires 4726/4059
hits. Protect historical MCLN 4475/3759; obtain the actual same-source start
separately. Do not combine single-metric bests from different models.

The bridge skill's referenced research-contract template is absent from the
installed skill bundle; this record states the applicable project contract
directly rather than blocking execution on that missing template.
