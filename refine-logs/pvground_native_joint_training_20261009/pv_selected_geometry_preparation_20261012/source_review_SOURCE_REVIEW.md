# 所选候选几何责任：源码审查

**Verdict: FAIL；scope: SOURCE_ONLY；scoped_blockers: 1。** 当前隔离 overlay 有可复现的 Python 语法错误，不能通过源码准备门禁。损失设计的逻辑核对与限定 CPU kernel 检查通过；它们不抵消实际源码无法导入的阻断。完整正常训练入口与 GPU 接入仍是另行待完成的部署条件，不作为模型精度失败。

本次为 fresh-context、same-family/provisional 审查，请求路由 gpt-6-astra/max；实际模型及推理档位均为 **UNATTESTED**。只读本地源与已保存证据，在本 source_review 目录写报告/检查文件；0 次远端查询、模型前向、GPU 调用、优化器更新及生产源修改。未认证当前 MCLN 进度，也未创建或轮询其观察器。

## 唯一源码阻断：SRC-01

生成器 prepare_source_overlay.py:16-24 先读取并解码原文件的原始 CRLF 字节，再于**第24行**调用 output.write_text(text, encoding='utf-8')。Windows 文本写入再次转换 LF，形成 CRCRLF，给反斜杠续行插入空行。第21行的 ast.parse(text) 只检查写入之前的字符串，因此漏过生成后的错误。

指定解释器的实际回执：

| 当前生成文件 | Python解析位置 | 实际错误 | CRCRLF数量 |
|---|---:|---|---:|
| source_overlay/models/pv_ground.py | 65 | IndentationError: unexpected indent | 653 |
| source_overlay/models/losses.py | 254 | SyntaxError: invalid syntax | 988 |

pv_ground.py:63-65 的 pairwise_locs 减法续行与 losses.py:253-257 的 rho2 加法续行均可直接看到中间空行。表中位置按 Python/通用换行读取计数；直接按 LF 字节计数的工具会显示不同编号。原两份正常源的 CRCRLF 数量均为0。当前文件 SHA 与 SOURCE_OVERLAY.json 完全一致，说明 manifest 记录的就是这个不可解析版本。

**最小修复**仅需将生成器第24行改为：

~~~python
output.write_bytes(text.encode('utf-8'))
~~~

保留当前失败版本，在另一隔离修订中重新生成实际 overlay 与现有 manifest，再检查写入后的源码。无需修改模型逻辑、引入兼容层或回退。审查仅在内存按这一个写入修复生成三个候选文本，三者全部通过 Python 3.7.6 解析；**没有写入修复源码，也没有关闭 SRC-01**。

CPU_TOOL_RECEIPT.json 保留原始 exit 1 与异常。CPU_TOOL_RECEIPT_02.json 对实际语法错误作汇总后继续独立 kernel 检查；exit 0 表示审查脚本完成，其状态为 FAIL_SOURCE_SYNTAX_CPU_KERNEL_PASS，不能读成源码 PASS。

## 按计划核对的逻辑

以下结论针对当前文件的可读逻辑及原生依赖关系。overlay 逻辑差异已与生成器 substitutions 逐项核对；由于 SRC-01，不能称新模块在真实工厂中执行成功。

| 检查 | 结论和依据 |
|---|---|
| 原生末层框与损失 | selected_support_geometry_objective.py:14,62-64 读取最终 last_center/last_pred_size，调用真实 criterion.loss_boxes；原源 models/losses.py:518-544 是中心 L1 + 0.5×尺寸 L1 与对角配对 GIoU。新增系数为 (10*L1+2*GIoU)/7，符合当前六层 decoder 的七输出平均；无 DFL、替代 box metric 或额外几何头。 |
| 真实 GT 与当前匹配 | 原源 models/losses.py:857-886 从本批 center_label/size_gts/gt_masks 按 box_label_mask 同序压紧 targets；:917-919 保存本次末层 criterion 返回的 last_indices。新增调用直接使用这些 target/last_indices，正式训练不从冻结父缓存取分配。SetCriterion.forward 每次重新匹配（:817）。 |
| root 与其他实例保护 | 新目标:19-30 断言 ScanRefer、有效 GT 槽0为 root，将 compact target index 映射回 valid 槽位。所选 Query 已被任一有效 GT 匹配即不给额外项，不重复 root 或覆盖 matched_other；不修改原 indices。 |
| 单一 Query 与点 Mask 资格 | :12-13 在完整256候选上按原生分数降序取一项；:33-48 读取同一 Query 的 own/text logits，经该行 superpoint 映射，与 targets[bid]['masks'][0] 比较。两路均使用整数 2*intersection > union；恰好0.5与空支撑不合格，不把超点数当点成员数。每条表达至多一个辅助索引。 |
| Box IoU 不作资格 | :32-56 资格仅依赖 own/fused Mask 与当前未匹配状态，没有 Box IoU 筛选。框跨过0.5不会仅因此退出几何项；若变成原匹配，原定位损失接管。 |
| 分母与原损失 | :11,62-64 使用本次 indices 长度，即真实 B，不是新增候选数。原 SetCriterion 的有效匹配数分母与各 prefix 损失未改，新项在原总损失之后加一次。范围为计划单卡配置，不认证其他分布式配方。 |
| bbs 与推理 | native_root_bbs.py:4-9 与原 evaluator root position/bbs 公式一致：主 token map 二值化，modify/pron/rel相加、other相减，降序排序。原评价 only_root、butd_cls=False；无新增候选裁剪、二次评分或空间 GT 推理。仍使用 baseline 文本 annotation maps，不能称完全不使用注释信息。 |
| 旗标及 eval | 生成器:29-54 将 spec 旗标接到 configure_native_model、PVGround 默认属性及 forward end_points，损失分支位于原 native_joint_training 条件内；原 forward 令该值=self.training，评价入口 model.eval()。逻辑上 eval 跳过新项；完整 factory/forward/get_loss 未执行，当前语法错误必须先修复。 |
| 两组初始化和参数 | 两份初始化 JSON 唯一差异为 use_selected_geometry_supervision=False/True；两组 G=True、旧 selected-mask=False。其他字段与 extremal_support 父 manifest 相同，无新增参数，也不改变加载、core可训练与RoBERTa冻结规则。1295状态和 checkpoint 加载一致性仅为静态预期，本轮未实例化/加载权重。 |

GT 来源直接查看保存的 runtime_binding/dataset_source/src/joint_det_dataset.py:1086-1119,1387-1392：目标来自注释 target_id、对象点成员与对象 bbox，root先放；训练增强时有原生 box jitter。训练支撑资格与定位目标均是真实数据派生标签，jitter后的训练框不能当作未增强验证框。train_dist_mod.py:163-175 的网络输入未传空间 GT；main_utils.py:450-455、522-533 在前向后才合入 batch GT 用于损失/评价。本轮未运行数据加载器或重新认证远端原始数据。

原 main_utils.py:292-306 保留 optimizer 分组，core/backbone与A/B按正式参数共同更新；这不证明新框项直接穿过全部 A/Mask 分支。native_mask_geometry 的硬支撑参考含 detach，不能称框损失通过完整 Mask 路径端到端优化。

## 实际 CPU 证据与边界

指定 E:\anaconda\python.exe -B，实际 Python **3.7.6**、Torch **1.13.1+cpu**；未安装依赖。记录47项输入的路径、字节数与SHA，检查结束全部未改。18份 Python 输入中16份可解析，另两份为 SRC-01；三份 patch manifest SHA、新目标两份拷贝、两组初始化唯一差异均核对通过。

原 check_saved_responsibility_cpu.py 仅在 AST 中移除第124-125行输出写入/print 后重放，其余执行内容不变；完整结果与原 CPU_RESPONSIBILITY_CHECK.json **逐项相等**。两保存步骤分别4和1个新增直接梯度 Query；额外与原匹配框梯度接收者不重叠，原 indices 保持，选择与资格输入不接收该项梯度。原结果未重写。

一个独立合成 fixture 使用 B6/Q256/full-shaped mask tensors、非均匀点映射，确认仅 (batch=0, Query=17) 有新增框输出梯度；已匹配 root 与映射回实际 GT 槽3的 other 受保护；own或fused恰好0.5均拒绝、空支撑为零；单个合格候选损失按B6平均。它是 simulation_only 工程检查，无模型或数据集精度含义。

保存 witness 仍为**同一增强batch两步、16次观测、8个唯一训练行**。CPU脚本使用保存的 Mask 几何端点，缓存没有这些观测的最终 B 框；own/text 仅取保存的所选 Query slices，字典索引代替完整网络输出。不能称16个独立样本、最终B误差重放、完整factory/forward/criterion接通、参数更新、GPU预检或精度通过。保存score只被读取及核对排序，没有 native semantic score forward。限制与原CPU JSON、保留诊断及其审查一致。

## G/no-object 与贡献边界

新增框责任与既有语义职责仍需实测。pvground_semantic_assignment.py:9-23,34-47 的 G 只对 **Box IoU >0.5 且未匹配**候选替换末层 eos-weighted position/token CE；新几何资格不依赖 Box IoU。Mask合格但Box未过0.5时，同一 Query 可获得 root定位目标及原no-object token目标。Box过0.5后 G 可更换该CE，但新几何项仍在，原 contrastive/not-mentioned 路径及其他层职责也未消失。不能称全部任务冲突已解除，也不能无实证断言这些职责必然造成退化。

本实现沿用现有 selected_query_mask_loss 的部署 Query/root保护结构，将额外责任换成原生框损失。目前是针对问题的训练策略差异，0新增推理参数；直接控制完成前不计第三个有效模块。计划已限制“首次正例/首次联合分割定位”等novelty声明，本轮没有新的外部文献认证。历史 §20.376.147–163 的匹配、C、Face、BN负结果及局部检查边界继续保留；未因源码/CPU结果推翻历史，也未把历史叙述当成本轮运行证据。

## 独立部署待办

以下为计划明确尚未完成的门禁，不额外计成当前损失逻辑缺陷或模型失败：

1. 修复 SRC-01，对新实际文件完成定点复核，保留本轮失败文件与回执。
2. 完成隔离 normal loop、最多8轮、E0参与的Acc@0.5连续两次严格逐轮下降停/持平或上升重置、strict优先再loose的同一best/latest保存、唯一A100队列。原 main_utils.py:167-169 仍为旧门槛/loose优先排序，:365-386 没有新下降早停；不能原样宣称符合新计划。
3. 明确完整原生依赖和运行参数：ScanRefer-only/joint_det=False、256候选/六层decoder、B8/seed2027、计划LR/WD/clip、实际36665行与4583更新/完整轮。四个overlay文件不是完整运行仓库。
4. 保持当前MCLN任务和19:05前不轮询要求；实际结束且资源/锁释放后才做计划中的两步GPU native检查，核实新旗标、完整loss钩子、当前匹配/GT、参数梯度及保存恢复。本文未查询或认证当前状态。
5. 两组从同源初始化分别验证真实状态、fresh optimizer/update history及完整9508条E0；工程两步更新状态不承接正式训练。随后同时对比同预算控制和强起点，不把追回控制退化算作新能力。

完整任务仍是同一完整模型 ScanRefer60.0%/52.0%（5705/4945）、三个有直接证据的有效贡献，以及对应Nr3D/Sr3D作者权重验证和同结构完整训练。保留5677/4920是计划/历史上下文，本审查没有新正式精度或当前最佳认证；CPU PASS不完成该目标。

## 产物与追踪

SOURCE_REVIEW.json 保存唯一 scoped_blocker 和独立 deployment_pending_conditions。CPU_VERIFICATION.json、CPU_SAVED_REPLAY.json、verify_source_and_cpu.py 保存实际限定结果与复现脚本。.aris/traces/experiment-bridge/2026-10-12_run01/ 保存完整收到的请求、完整报告/最终回复、路由与身份元数据、47项输入前后哈希、原始失败及后续CPU工具回执和逻辑diff。diff去除CR仅供阅读，错误判断来自未改字节的Python解析。


