# 候选自身 Mask 参与原生末层匹配：SOURCE_ONLY 审查

**结论：WARN；阻断项 0；限定源码变更评估为 PASS。** 两份修改正确实现了计划中的成本来源对照，未发现需要修改实现的阻断缺陷。新的 query 模式尚未实际运行；本报告不构成 CPU、GPU、训练或最终方法准入。

审查日期为 2026-10-11。审查使用独立上下文，归属 `same-family`，接受状态为 `provisional`。请求模型为 `gpt-6-astra`，请求推理强度为 `max`；没有可验证的实际模型／强度身份证明，故记录 `actual_identity_attestation=UNATTESTED`，不将请求值当作实际凭证。

下文 `source/...` 与计划文件相对于 `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/query_mask_assignment_20261011`；基准为 `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source`。全部 48 个审查输入的绝对路径与完整 SHA256 见 `SOURCE_REVIEW.json`。个人会话记忆不作为研究审查输入。

本次仅进行本地源码读取、SHA256 比对和文本 diff。项目导入、准备脚本／matcher／criterion／模型执行、真实数据读取、反向传播、优化器更新、SSH、联网、GPU 查询和活动训练查询均为 0。只新增本目录报告，没有修改实现、PLAN 或活动训练。

**实际修改与接线。** 独立核对 14 对源码，28 个摘要均匹配对应 manifest。恰好 `main_utils.py`（+4/-1）和 `models/losses.py`（+4/-2）变化，其余 12 份字节一致；合计 +8/-3，无新增 Python 文件。准备脚本只在不存在的隔离目录生成文件（`C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/prepare_query_mask_assignment_source_20261011.py:8`、该文件 `:19`、`:49`），没有部署或训练操作。

`source/main_utils.py:39` 声明 `text/query` 与默认 `text`；`source/train_dist_mod.py:40` 的 TrainTester 继承原工厂，`source/main_utils.py:327` 调用 get_criterion，`source/main_utils.py:284` 将选项交给 matcher。正常包的 [__init__.py:10](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/runtime_binding/model_source/models/__init__.py:10) 导出同一实现，其摘要与 `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/NATIVE_SOURCE_PORT.json:160` 相符。默认和显式 text 保留原成本表达式；这尚不是运行等价的实证。

| 数据 | 正常 forward 的形状 | matcher 读取 |
|---|---|---|
| Text Mask | 长度 B 列表；各项 `1×256×SP_b` | `pred_masks[b].squeeze(0)` |
| Query Mask | 长度 B 列表；各项 `256×SP_b` | `sp_pred_masks[b]` |
| 超点映射 | 每样本 50000 个超点索引 | 扩展为 `256×50000` 后 gather |
| GT Mask | 每样本 `T_b×50000`，仅有效槽 | 拼接作点级 L1，最后按各批 GT 切分 |

形状依据为 `source/models/pv_ground.py:330`、`:537`、`:549`、`:552`、`:559`。Text 实际执行 `expand(1,256,P)`，其注释不能被误读为已经去掉首维。支持修正器保持 Query 二维形状（`source/mask_support_corrector.py:24`、`:54`），在 `source/models/pv_ground.py:573` 更新同一输出键。criterion 在 `source/models/losses.py:913` 同时传入两路；新增分支在 `source/models/losses.py:334` 正确直接读取 Query 张量。

**成本与映射。** `source/models/losses.py:336`、`:337`、`:338`、`:344` 保留 `>0` 硬二值化、原超点到点映射、点级 `torch.cdist(...,p=1)`。不同样本的 SP 数可以不同，因为各自先映射到共同点数才拼接。没有改用融合 Mask、alpha 或 CE/Dice 匹配成本。工厂系数仍为分类 1、框 L1 0、GIoU 2（`source/main_utils.py:284`），Mask 仍为 0.0002（`source/models/losses.py:294`）；softmax/positive_map、三维 GIoU、最终代价加权和按批切片保留于 `source/models/losses.py:325`、`:347`、`:369`、`:375`、`:382`。

**六个早期 prefix。** 当前六层 decoder（`source/main_utils.py:49`）对应 proposal_、0head_ 至 4head_ 与 last_。已核验准确绑定的 [modules.py:135](C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground/models/modules.py:135) 与 `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/NATIVE_SOURCE_PORT.json:201` 摘要一致；head 在 [modules.py:173](C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground/models/modules.py:173) 仅写框、语义等输出，没有 Mask 键。PV 分割发生在 decoder 循环后（`source/models/pv_ground.py:473`、`:528`）。criterion 每个 prefix 重新创建 output（`source/models/losses.py:893`），正常 last-only 路径仅 last_ 加入两路 Mask（`source/models/losses.py:909`）。六个早期 prefix 不会进入新的 query 读取路径（`source/models/losses.py:329`），固定相同输出／GT 时成本、分配和损失不变。这不是不同训练轨迹的早期预测不变保证，也不是其他模型多层 Mask 路径的兼容性审查。

**全部有效 GT 与实例职责。** `source/models/losses.py:874` 仍按 box_label_mask 统一筛选各 GT 字段；`source/models/losses.py:382` 仍按每批 GT 数切分，再对本批 `c[i]` 作一对一分配。没有跨样本配对、root-only 重映射或由 matcher 新增正例。准确数据副本在 [dataset:62](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/normal_controls_20261010/normal_source_review/private/audited_inputs/runtime_binding/dataset_source/src/joint_det_dataset.py:62) 定义上限 132，在 [dataset:1086](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/normal_controls_20261010/normal_source_review/private/audited_inputs/runtime_binding/dataset_source/src/joint_det_dataset.py:1086) 保留列表目标与可选 anchor 的独立槽，在 [dataset:1100](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/normal_controls_20261010/normal_source_review/private/audited_inputs/runtime_binding/dataset_source/src/joint_det_dataset.py:1100) 构造各对象点 Mask，在 [dataset:1116](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/normal_controls_20261010/normal_source_review/private/audited_inputs/runtime_binding/dataset_source/src/joint_det_dataset.py:1116) 标记有效槽；检测目标列表按 [dataset:1279](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/normal_controls_20261010/normal_source_review/private/audited_inputs/runtime_binding/dataset_source/src/joint_det_dataset.py:1279) 生成。候选为 256，故该接口有效输入下可覆盖全部有效 GT 且 query 不重复。数据副本摘要匹配 `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/NATIVE_SOURCE_PORT.json:564`。

这里保留的是目标身份和匹配结构，原先哪个 Query 负责哪个对象可能改变。当前正常入口还明确要求 ScanRefer 且 not joint_det（`source/train_dist_mod.py:406`、`:408`）。没有执行混合检测行或 Nr3D/Sr3D，不能因此宣布完整检测／跨数据集训练链已经验证。先前读到的 `runtime_binding/dataset_source/models/modules.py` 与该目录的 `models/__init__.py` 是其他历史副本，JSON 已标记，未用作正常模型的绑定证据。

**损失与反向。** SetCriterion 保留同一 indices 分发以及按匹配数计算 num_boxes（`source/models/losses.py:819`、`:833`、`:844`）。位置语义损失（`source/models/losses.py:471`）、框损失（`:523`）、Mask 损失（`:578`）与对比损失（`:668`）沿用原索引；Query Mask 按 matched query 取值，GT 按对应 matched target 取值，Text 仍用广播副本。监督分母与最终损失权重没有改动（`source/models/losses.py:951`）。

G 与 selected-query 目标源码字节不变，本次没有新增额外正例；G 仍排除全部已匹配 Query（`source/pvground_semantic_assignment.py:19`）。两开关继续由 native_init_spec 设置（`source/native_model_initialization.py:64`），所以 PLAN 的 selected=false 仍是后续实际执行配置约束。分配改变还可能改变 G 看到的未匹配集合，不能把“规则未改”理解为“作用的 Query 一定未变”。

matcher 保持 `@torch.no_grad()`（`source/models/losses.py:299`）和离散 Hungarian；新增成本本身没有可微反向路径。原训练仍对依赖真实匹配索引的预测张量计算损失后 backward（`source/main_utils.py:463`）。分配变化可能影响框、语义、对比、Mask 的监督对象，不能说它只改变几何梯度，也不能保证原已匹配实例不受损。限定 CPU 框损失检查不能被写成完整 criterion 通过；完整原生 Mask loss 仍含 CUDA 调用（`source/models/losses.py:622`）。

**前向与 fallback。** 12 份模型、初始化、训练入口及已有模块源码与基准字节相同。新字段是 matcher 普通字符串（`source/models/losses.py:295`），没有 Parameter；get_model 不读取它（`source/train_dist_mod.py:100`）。模型输入接口与 last/bbs 评测接口保留（`source/train_dist_mod.py:163`、`:252`），没有新增推理 GT 输入、候选裁剪或第二排名。相同模型参数／输入下结构不变，不等于不同训练后的预测相同。

合法 CLI 两值范围内是明确的来源选择：query 直接索引 sp_pred_masks，没有缺键时退回 Text 的逻辑，也没有新增 try/except 或兼容层。原有离散 Mask 几何及无效支撑处理仍在 `source/native_mask_geometry.py:13`、`:20`；Query Mask 对最终框的间接作用仍由 `source/models/pv_ground.py:593`、`:604` 保留。不能把整个原系统称为不使用 Query Mask 或没有任何既有回退。

**实证、因果与新颖性。** 旧 `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/matcher_mask_role_cpu_20261011/ACTUAL_CHECK_SUMMARY.json:13` 标明 simulation_only，`:24` 标明未持久保存完整成本矩阵，`:36` 标明当时没有实现新 matcher 策略。九次旧检查只支持原策略的受控合成观察，不能充作本次 query 模式已通过 CPU 的证据。

`EXPERIMENT_PLAN.md:5`、`:18`、`:20`、`:30`、`:36` 已保留掉点因果、监督改变、新颖性、共同起点历史及实证限制。本次没有发现把该来源改动宣称为已成立论文贡献的表述。计划明确承认 Mask 匹配已有前作；本次遵守纯本地范围，未独立联网核验这些引用。

本报告不证明精度增益、训练下降原因、原实例性能不降或三项有效机制成立。共同 E0 已含既往 C 适配历史，不能称为从未训练 C 的独立消融。9508 条双阈值目标、既有最好 5677/4920、三项有效机制及完整模型 Nr/Sr 要求都没有由本报告重新验证或关闭。历史本地源绑定也不是远端此刻状态的重新核验。

保留的 WARN 为：

- W01：新 query 策略没有本次实际 CPU/GPU／模型／criterion／训练证据；本报告不授予执行准入。
- W02：匹配职责结构保留不等于 Query 身份或性能不变；没有新增精度、掉点因果或实例无损保证。
- W03：当前 normal 入口不是混合检测／Nr/Sr 入口；后续同起点、同预算和 G/selected 标志须实际落实。
- W04：仅有本地历史依赖绑定，未重查远端活动状态，也未做外部文献核验。

下一步检查只沿已有计划建议：若主流程另行落实执行范围，限定 CPU 检查应覆盖默认／text 等价、固定框和语义时 query 成本依赖候选 Mask、B>1／不同超点数／多 GT 的批间对应、六个早期 prefix 不变，以及原生框损失梯度落到实际匹配 Query。真实模型前后向、数据、保存恢复、同起点 GPU 预检及正式同预算训练仍须后续关口；本次没有排队、启动或授权这些操作。

