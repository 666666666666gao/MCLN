# Nr3D/Sr3D 作者输入协议入口：R1 源码审查

**结论：WARN；入口修改本身 PASS；execution_scope: SOURCE_ONLY。** 本次入口修正没有源码准备阻塞项。该结论只接受所审查的两条断言及其限定说明，不是训练、GPU、完整模型或精度准入。评估来源必须按下文限定；Box/Mask 同 Query 要求仍未完成。

请求模型为 `gpt-6-astra`、effort 为 `max`；实际模型与 effort 均为 `UNATTESTED`。本审查使用 fresh context，`review_independence: same-family`，`acceptance_status: provisional`，无跨模型家族接受声明。

## 已确认正确的内容

- 对比前一作者核心目录的全部 14 个 Python 文件，只有 `source/train_dist_mod.py` 改动：原第 410 行的一条断言替换为现第 410–411 行两条断言。其余 13 个文件和 Nr3D/Sr3D 两份 `init.json` 均保持原字节；实际 diff 与 `ENTRY_CHANGE.diff` 一致。将这两条 AST 节点还原后，整份入口 AST 与旧版一致。
- `butd_cls/joint_det/detect_intermediate=True`，`butd/butd_gt/augment_det=False` 与四份保存的作者 train/test 脚本，以及真实数据 CPU receipt 的两组 `author_flags` 一致。当前 parser 对六个开关均使用 `store_true`。断言位于 `parse_option()` 之后、CUDA/NCCL 设置之前。
- `TrainTester.get_datasets` 的 AST 与保存的作者入口相同：训练使用本 benchmark 加十次 ScanNet 数据；验证按 `test_dataset` 载入。`butd_cls` 传入数据类，模型仍通过 `butd or butd_gt or butd_cls` 启用对象输入流。数据源码中该模式使用场景实例框和 `cls_results` 类别；`augment_det=False` 不关闭原生训练增强。
- 两份初始化文件分别指向 `PV-Ground_NR3D.pth` 与 `PV-Ground_SR3D.pth`，seed 为 2027，新模块为 `fresh`，三个附加 checkpoint 均为 null。配置未照抄历史 Sr3D 脚本中的 Nr3D/ScanRefer 路径，也未加入 ScanRefer 终点权重。本审查未加载或重核验远端权重字节；未来命令仍须配对相应 benchmark 和 init 文件。
- 116 项 warm 清单与已有数据检查清单的键集合相同，仅 `train_dist_mod.py` 的值变化；14 项准备源码均与清单匹配。没有将其余继承项误报为全部重新核验。
- 对准备源码与旧源码各 14 个文件完成 `ast.parse(feature_version=(3, 7))`，全部通过。实际解析器为 Python 3.12.6；这是 Python 3.7 语法模式的静态解析，不是 Python 3.7 运行验证。

主要依据：准备入口第 55–96、121、128–129、208–213、406–434 行；`main_utils.py` 第 37、57–76、126–130 行；数据类第 99–119、240–258、1361–1372 行；四份作者脚本第 9–15 行；两份 init 第 2–12 行。所有原始输入快照和具体路径见私有 manifest。

## 阻塞项

`blocking_findings: []`，仅针对本次入口源码准备。正式执行准入仍为 false。

## 非阻塞发现与后续准入约束

**NB-01 / WARN：PLAN 中的“当前项目评估器”不是本次 warm 清单对应的评估器。**

`WARM_SOURCE_HASHES.json` 的 `src/grounding_evaluator.py` 为 `39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677`，与 `runtime_binding/model_source/src/grounding_evaluator.py`、保存的作者 upstream 文件逐字节相同。它在作者源码第 240–249 行对任意有效输入场景框检查 IoU > 0.25，第 281–286 行将不合格候选分数乘零后排序。

`runtime_binding/dataset_source/src/grounding_evaluator.py` 则为 `b77376d16f0210ae18f067655552a806040614d7a21385477ffdb48590775af8`，通过第 1697–1735 行的有效性掩码排除候选，全部无效时计 miss；其依赖 `models/rec_evaluator_filter.py` 不在这份 warm 清单内。该文件只能证明另一份本地源码的行为。PLAN 第 13–15 行对两种算法的描述正确，但不能把后一种行为归到本次 warm 执行来源上，更不能据此声称已经执行了哪一种评估。

最小处理：在后续说明中明确“R1 的 warm 来源仍为作者乘零版本；dataset_source 新版仅作对照”，并使正式 baseline 与新模型使用同一明确评估版本。两种排序并不保证等价：原分数包含减去 `other_entity` 项，可能为负；全部不合格时的处理也不同。本审查没有测量真实发生率或精度影响，没有修改任何评估器。

**NB-02 / WARN：Box 与普通 Mask 仍分别选择 Query，同 Query 交付要求尚未落实。**

这项限制也存在于本次实际 warm 的作者评估器：`bbs` 在第 283–286 行乘对象资格后排序，`mask_pos` 在第 609 行令 `is_correct=None`，第 641–646 行直接按未过滤分数选第一名。上述本地新版同样存在普通 Mask 的独立排名（第 2052、2084–2094 行）；准备入口没有启用它的其他联动选择模式。因此 `butd_cls=True` 下不能从当前源码推出 Box/Mask 永远同 Query。

另外，已开启的 `selected_query_mask_objective.py` 第 8–9 行也使用未过滤 `native_root_bbs` 排名。其文档中的“deployed Query”不能自动等同于经对象资格处理后的 REC 胜者。它是否保留取决于最终方法/C 开关；这里报告选择路径事实，不把未经运行的潜在样本差异当作观测结果。

最小后续修正应保留确定的 `bbs` 排名及其对象过滤，将普通 Mask 报告绑定到该同一 Query；若保留 selected-mask 监督，再明确其选择语义。父任务已说明另建 R2 准备目录，本审查不评价尚未写出的 R2 实现，不修改 R1 或活动源码。不得用独立 Mask 选择拼接 REC 结果。

## 证明范围

已有 CPU receipt 只支持有限真实数据接口背景：4 次构造、10 次取样、2 次训练批次合并；6 个 benchmark 上下文选择对应 5 个原始来源记录/场景。它调用了原生 CPU NLP 预处理，不包含 PV 模型、原生训练入口、criterion 或 optimizer 验证。本审查没有重跑数据检查。

本轮未执行项目模块或入口、SSH、网络、GPU、数据构造、模型前向、loss、优化器更新、权重保存或训练状态查询。未修改所依赖的输入文件。准备代码的 AST 与文件一致性不证明依赖导入成功、真实 GPU 前向、完整损失和模块更新、checkpoint/AdamW/scheduler/RNG 保存恢复、完整数据规模、训练预算或 REC 精度。

PLAN 已正确说明 batch/LR/epoch 并未完整复现作者配方，最终完整结构及 C 开关仍待确定。相应真实输入、对应作者核心、完整模型状态恢复与相同起点预算仍需后续完成；本报告不授予任何训练准入，也不宣称总目标完成。

## 私有溯源

`source_review/.aris/traces/experiment-bridge/2026-10-10_run01/` 保存 59 个依赖输入的精确副本、原始 SHA256、完整请求/回复、确定性检查结果和实际 diff。哈希仅用于本次审查溯源，没有新增训练期机制。检查脚本只使用标准库读取/解析源码，未 import 项目模块。
