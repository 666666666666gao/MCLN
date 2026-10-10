# C-off 原生正常训练控制：独立源码审查

结论：**WARN / SOURCE_ONLY，发现 0 个阻断性的源码缺陷；C-off 当前尚未获得实际启动准入。** 控制配置及接线确实只关闭 `use_selected_mask_supervision`，其余被审查的计算配方不变。下面的启动门槛属于尚未完成的实际工作，不能把本结论改写成 GPU 预检通过或正式实验完成。

本审查由独立新上下文 Codex reviewer 直接读取输入完成。`review_independence: same-family`，`acceptance_status: provisional`。请求路由为 `gpt-6-astra` / `max`；宿主未提供实际后端模型/effort 的运行时证明，因此实际身份记为 `UNATTESTED`，不把请求参数当作证明。未 SSH、查询 GPU、导入项目运行模块、加载权重、执行神经计算、启动训练或删除文件；全部写入均限于本审查目录。

下列路径除特别注明外，均相对 `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/`。行号对应封存的原始字节。`runtime_binding/…` 及空间估算文件相对上一层 `pvground_native_joint_training_20261009/`。

**源码核对结果**

1. **唯一计算配置差异为 C 的布尔值。** `normal_controls_20261010/init.json:13` 为 false，`init_manifests/extremal_support.json:13` 为 true；逐键比较确认两个对象键集合相同，其余字段相同。官方、G、d06、f989 的路径与身份相同（两个 manifest 的第 4–11 行），G 保持 true（第 12 行），seed2027 与 ScanRefer 相同（第 2–3 行），span 来源仍为 extremal_support（第 26 行）。控制 protocol 只新增明确的 `native_init_spec` 路径（`normal_controls_20261010/NORMAL_NATIVE_RUN_PROTOCOL.json:108`），其余字段和完整 `common_arguments` 与主臂 protocol 完全相同。

2. **控制 manifest 确实进入原生入口，没有被默认 manifest 覆盖。** 完整接线为控制 protocol 第 108 行 → `normal_controls_20261010/normal_joint_controller.py:33` → `source/main_utils.py:37` 的 required 参数 → `source/train_dist_mod.py:129` → `source/native_model_initialization.py:19` 读取该路径 → 第 65 行赋值 → `source/models/pv_ground.py:617` 写入 end_points → `source/models/losses.py:965` 控制额外损失。控制器与主臂控制器进行文本比较后，仅第 33 行的路径表达式不同。

3. **网络构造与参数形状保持相同；实际张量相等仍待运行证据。** `source/native_model_initialization.py:29–59` 采用相同父链，严格加载官方/G/A/B；第 44、54 行无条件构造相同 A/B 模块；第 60 行要求 1295 个 state 项。C 布尔值在上述构造/加载之后才于第 65 行赋值，不控制参数创建、形状或 RNG。第 61–62 行保留原核心可训练性及原 RoBERTa 冻结。C 的值也写入 `native_training_architecture` 元数据（第 73 行），因此两臂的**结构元数据整体对象**并不字面相等，但网络和参数形状不因此改变；这不是额外的结构消融。恢复 C-off 检查点时应使用本控制 manifest，因为 `source/main_utils.py:135` 比较包含该值的架构元数据；无需兼容层。

4. **C 只增加训练期目标，推理评分不变。** `source/models/pv_ground.py:615` 将 `self.training` 传给 criterion，`source/models/losses.py:959–970` 仅在训练态启用 G/C；C-off 不改第 960–964 行的 G correction，也不改第 945–958 行原有损失。`source/selected_query_mask_objective.py:8–9` 用 detached 原生 bbs 分数选唯一赢家，第 18–24 行排除所有末层匹配 Query，第 25–32 行使用 root 的实际 GT Mask，按原系数 5/1/10/2 监督自身与融合 Mask；没有用另一个模型的预测代替 GT。`source/native_root_bbs.py:5–9` 与实际 evaluator 的 root bbs 公式一致（`runtime_binding/model_source/src/grounding_evaluator.py:253–286`）。C 不增加另一套部署评分。

5. **seed、有效 batch、预算及重置保持相同。** 两臂 common argv 都是 seed2027、B8、3 轮、start_epoch1、核心/骨干 LR1e-6、A/B LR1e-5、WD5e-4、clip0.1、step scheduler 280/340、无 warmup（两个 protocol 第 20–56 行）。控制器固定 world1/GPU0（第 23–24 行），排除 `--checkpoint_path` 和 `--frozen`（第 35 行）。入口固定 Python/NumPy/Torch seed、关闭 TF32、固定 cudnn 设置（`source/train_dist_mod.py:405–426`）。`source/main_utils.py:292–306` 新建同一 **AdamW** 分组，主函数第 327–331 行新建 optimizer/scheduler；仅在有 checkpoint_path 时才恢复其状态（第 350–352 行）。每 batch 一次 zero_grad/backward/step（第 460–470 行），没有累积，故 world1 下有效 B8。第 371 行覆盖 E1–E3；预计每轮 4583 次更新来自 36665/8 向下取整。新增主臂实际恢复回执记录 E3 的 820 个 optimizer 状态和 step13749（`normal_terminal_recovery_20261010/attempt2/RECOVERY_RESULT.json:33–40`），与 3×4583 相符；这不替代 C-off 自身的实际加载数量/更新数记录，也不证明每个可训练参数均有非零变化。

6. **数据与评估入口相同，best 规则未变。** 同一 data_root 与 ScanRefer 参数进入 `source/train_dist_mod.py:68–96` 的 train/val 构造；无 joint_det/detect_intermediate/augment_det 开关，不能冒称作者混合检测配方。`source/main_utils.py:244–267` 保持训练 drop_last、验证不 drop；第 367–385 行运行 E0 和每轮正式验证，保留 E0 参与 best 候选。`source/train_dist_mod.py:237–257` 从同一个模型的 last/bbs 计算两阈值并要求 9508 行；evaluator 从真实 `center_label`/`size_gts`/`gt_masks` 取 GT（`runtime_binding/model_source/src/grounding_evaluator.py:535–550`、第 879–892 行）。`source/main_utils.py:167–169` 的顺序准确保持为 `(both gates, strict gate, wide hits, strict hits)`，第 382 行只有严格优于才替换 best，E3 另存 latest。不得改成仅比较 Acc@0.5，亦不得拼接不同 epoch 的两列成绩。

**仍阻断实际启动的事项（不是新增源码缺陷）**

- **主臂终态、冷恢复和限定范围的结果审查已有证据；仍应依据退化决定控制准入。** `normal_epoch3_boundary_observation/normal_status.json:125–126` 的 pending 是生成该不可变记录时的历史状态。随后 `normal_terminal_recovery_20261010/attempt2/RECOVERY_RESULT.json:2–40` 记录 best/E0 与 latest/E3 的各 1295 项状态精确恢复；第 69 行为实际 exit0，身份与原 status 两份权重相同。`terminal_actual_review_20261010/EXPERIMENT_AUDIT.json:3–7` 已给出 WARN；第 264–281 行确认终态冷恢复，限定描述性报告及状态恢复主张均没有阻断项。因此本轮不再把“主臂恢复未做/结果未审查”列为缺口。该审查第 291 行及第 317–329 行明确更新后退化、E3 未过双门槛；计划第 48 行仍要求利用已有结果检查退化并决定是否进行 C-off，不能仅据 E0 达标自动准入。所给审查没有把退化归因于 C，本审查也不作这种归因。计划第 40 行要求的固定 E3 与各自 best 都须报告。
- **C-off 实际接线/初始化/E0 等价尚未验证，也没有控制 admission。** `normal_controls_20261010/CONTROL_PREPARATION.json:43–49` 明确为来源准备且实际 GPU 验证 pending、尚未训练；本地不存在该控制目录的 `admission.json`。控制器第 11–19 行要求实际已审查 M0、未承接预检状态及 helper/environment 身份。主臂的 admission 是主臂历史证据，其 helper 身份属于旧控制器/protocol（`normal_epoch3_boundary_observation/admission.json:6–20`），不能直接代填成 C-off 的 admission。应完成现有契约要求的控制实际构造/短工程验证及相同初始模型状态、E0 输出核对，并审查证据，再从同一父链重新创建正式 optimizer/scheduler。仅两个 E0 总命中数相等不足以宣称逐样本输出/完整张量相等。
- **当前空间、GPU、父依赖和现场源码/数据/环境未由本次审查确认。** 计划第 87 行要求至少 2,605,151,860 字节三槽及余量，同时保留全局最佳、两臂所需 best/latest 和官方/G/d06/f989。现有正常启动实现要求可用空间严格大于该值、无 compute-app、唯一 GPU0 为 A100且空闲显存占用低于500 MiB，并在资源锁下启动（`launch_normal_authorized.py:83–104`）；其父身份/环境/现场源码核对位于第 53–64 行。这些是已有准入标准，旧 admission 第 25 行的 03:19 空间记录不是当前空间证明。C-off 必须使用属于控制目录的实际准入与独立输出；本次未执行这些步骤。

**非阻断事项与结果边界**

- 计划第 31、44、80、94 行及原 status 的第 125–126 行是历史进度措辞。保留原始文件，在当前交接记录里指向后续实际 recovery/audit 回执即可；不应改写不可变原始状态或继续把已完成的主臂恢复称为 pending。新增审查本身仍为 same-family/provisional（`terminal_actual_review_20261010/EXPERIMENT_AUDIT.json:25–26`），其第 267–272、282–287 行没有宣称重新评估精度、逐行 IoU 独立复算或完整 benchmark 复现。状态恢复成功不能扩张成这些主张；本次只核对所给回执与原终态身份，没有自行加载权重。
- C-off 只能检验**共同、已经接受 C 相关适配的父历史之后，三轮正常继续训练中启用 C 的增量效果**。d06 的 11169 次更新、f989 的 3723 次更新仍继承（两个 manifest 第 17、27 行）；计划第 40、92 行和准备记录第 42 行已经正确限制主张。它不检验从未使用 C 的完整训练历史，不证明 C 的独立从头训练价值，也不能替代 A/B 的独立贡献证据。未获得 C-off 结果，不能报告 C 有效/无效。
- 后续 C-off 终态必须包含实际退出、E0/E1/E2/E3 四次完整 9508 验证、best/latest、完整冷恢复与实际结果审查。控制器第 43–61 行已经区分训练 complete 和恢复/审查 pending。Mask 汇总由既有 evaluator 输出（其第 194–206 行及第 646–653 行）；`native_metrics.jsonl` 的结构化返回仅包含框计数（`source/train_dist_mod.py:252–257`）。机制所需的逐表达修复/破坏证据应按计划第 69–71 行复用或在确有缺失时补齐，不能用训练 loss 或聚合框数冒充。

**本次实际验证**：13 个指定输入在首次读取至封存期间 SHA 一致；首轮 29 份、补充 2 份，共 31 份原始输入字节已封存；两个 manifest 逐键差异、两个 protocol、两个 controller 均完成直接比较；14 份模型源码均与 CONTROL_PREPARATION 和 NATIVE_SOURCE_PORT 的对应条目一致，native port 含 116 个条目，但本次没有重新核验远端全部 116 文件；实际 evaluator 本地字节与 port 身份一致；19 份 Python 源码通过 AST 解析。只执行了隔离的纯 `native_metric_order` 函数来核对已有文本记录的 best 规则。第一次 PATH 中的 Python 启动失败为 `No pyvenv.cfg file`；随后使用本机已有 CPython3.10.19 完成上述 stdlib 检查，未安装包或修改运行环境。

完整请求、此响应、归属说明、输入 SHA/字节快照及静态检查结果均保存在本目录。源码阻断项为零；主臂限定范围的终态审查/恢复已有实际回执，C-off 实际准入及其未来结果接纳仍 pending。

补充记录：本版本纳入父代理随后提供的两份主臂完成证据；首轮报告及 001 请求/响应原样保留。002 请求/响应保存此次补充指令与完整修订报告，避免把新增证据倒写成首轮已知信息。
