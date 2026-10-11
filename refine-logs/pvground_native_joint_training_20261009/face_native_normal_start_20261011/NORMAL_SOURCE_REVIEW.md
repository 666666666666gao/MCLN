# PV-Ground Face V2 正常训练源码审查

结论：**SOURCE_ONLY_PASS；总体 WARN；0 个源码阻断；正常训练 NOT_ADMITTED**。WARN 表示实际运行证据尚未闭合，不是发现源码缺陷。审查为 fresh-context、same-family、provisional；模型与推理强度身份 **UNATTESTED**。

本次核对 REVIEW_INPUTS.json 的 19 个文件，并经主任务授权补充 4 个精确哈希匹配的依赖源码。全部 23 个输入绑定在 SOURCE_REVIEW.json；两个正常训练协议的 118 项源码目录与 SOURCE_HASHES.json 完全相同。17 份 Python 源码和 3 段内嵌远程脚本完成 AST 解析。没有执行被审查脚本、导入 Torch、调用 SSH/GPU、进行神经网络运行或修改源码。

- **正常入口和预算正确。** normal_joint_controller.py:34 调用隔离 PV-Ground/train_dist_mod.py，后者 :430 进入 TrainTester.main。main_utils.py:365 先完整 E0 验证，再运行 E1–E3；原生 B8/drop_last、36665 条训练表达对应每轮 4583 次更新、36664 次呈现，合计 13749/109992。预检代码 :66 对真实 loader 数量设有断言；这些仍是待运行预算。
- **保留已学习 G/A/B，六面新增零输出。** native_face_model_initialization.py:14 先严格载入 1295 个原模型状态，再包裹保留的 extremal_support 轴先验，增加 6 个 Face 状态、23425 个参数，总计 1301 个模型状态。face_residual_span_mixer.py:51–63 分别生成六面残差，向旧中心/尺寸添加移动量，最终采用 abs(raw_size)。新增最终尺寸下限已经移除，原有 native/reference 定义保留；没有删除处理已实测独立面反转的 abs。实际零输出浮点相等性由预检 :174–187 验证，源码检查不代替该运行。
- **两模式同容量，控制范围准确。** 两份 normal protocol 只差 face_residual_mode/native_init_spec；init 只差 face_residual_mode。without_additional_source 仅将新增 Face 输入中的 extreme-source token/count fraction 置零（face_residual_span_mixer.py:37），老轴先验及 G/core 的来源信息仍保留。不能称为所有来源有无对照，也不声称跨进程逐位配对。
- **匹配与损失符合方案。** main_utils.py:282 与 losses.py:327 使用原生 Text pred_masks 匹配；last 的 native bbox/GIoU、语义和 mask 损失保留。losses.py:959 在训练中加入 G correction，C 的额外 selected-query mask 分支由两份 spec 的 false 禁用。GT 在 model.forward 之后加入 loss；几何前向没有读取 GT。
- **联合训练与学习率正确。** main_utils.py:292 覆盖原可训练 core、backbone、G 以及 A/B/Face；RoBERTa 参数按原生规则冻结。core/backbone 为 1e-6，A/B/Face module 组为 1e-5，AdamW WD=0.0005、clip=0.1、seed=2027。仅 ScanRefer、关闭 joint_det/detect_intermediate/augment_det；协议明确不等同作者完整混合 ScanNet detection recipe。
- **完整保存和恢复责任明确。** main_utils.py:133–164 保存并恢复模式架构、完整模型、Adam、scheduler、retained metrics、Python/NumPy/Torch/CUDA RNG；best/latest 原子替换。train_dist_mod.py:128 在 checkpoint 恢复时只构建架构，再严格加载状态。native_face_preflight.py:285–350 对 1301 个模型状态、826 个 Adam 状态、scheduler/RNG 逐项核对。正式训练后的冷恢复仍列为 pending。
- **指标来自数据集 GT，目标是同一完整模型。** grounding_evaluator.py:232、:289、:544 使用预测框和 center_label/size_gts 计算 last/bbs top-1；train_dist_mod.py:252 要求两阈值分母均为 9508。native_metric_order 优先保留同时达到 5705/4945 的单模型，约为 60.0021%/52.0088%；没有拼接检查点。5677/4920 是保留起点，不是本次新结果。
- **正式准入链存在，不能承接两步拟合状态。** launch_normal_authorized.py:20–49 要求两组真实完成、16 个相同输入、Face 梯度、1301/826 完整恢复、无预检正式分数、actual_pair integrity audit 和绑定正常入口的 SOURCE_ONLY 审查；:65–120 再核对远程状态、stage 源哈希、环境与原始依赖权重、单 A100 空闲、2610000000 字节磁盘 reserve 和 GPU lock。stage_source_authorized.py:55–58 固定 118 个源文件；两协议已与该目录静态对齐。normal controller :38 禁止 checkpoint_path，正式初始化重新加载官方/G/A/B，不读取预检 best.pth/Adam。

本次没有读取或验收 actual_pair 工程结果、实际模型权重、远程进程或磁盘状态。任务交接明确旧版预检在 optimizer update 前失败，修订版 GPU 尚未启动；旧失败记录保留，本报告不将其改写为通过。两组修订版真实预检及其完整性审查未闭合前，**NOT_ADMITTED** 必须保留。完整 E0+三轮结果、训练后冷恢复和独立结果审计完成前，不得声明达到目标或提升精度。

补充只读绑定：models/losses.py（00b86273…）、src/grounding_evaluator.py（39c8de92…）、extremal_span_mixer.py（5ab81a56…）、native_mask_geometry.py（8386111d…）；精确绝对路径与完整 SHA-256 均在 JSON。这里只审查这些源码体和原始 19 个输入，没有声称语义复审全部 118 个文件。
