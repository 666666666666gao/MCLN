# PV-Ground BatchNorm buffer 配对诊断源码审查

**NON-BLOCKING / WARN；0 项 BLOCKING。** 配对 replay、冷恢复和两阶段准入未发现必须修改后才能开展计划内 sanity 的源码缺陷。存在一项本地启动解释器说明。此结论仅限 SOURCE_ONLY，不是 native sanity/full9508 通过，也不证明训练退化原因。

requested model=gpt-6-astra，reasoning=max；实际身份无法独立证明，identity_attestation=UNATTESTED。fresh context、同一模型家族，review_independence=same-family、acceptance_status=provisional。

## BLOCKING

无。

## NON-BLOCKING

**NB-01：显式沿用项目已有本地 Python 3.11 启动 launcher。** `launch_buffer_diagnostic_authorized.py:91` 使用 `shlex.join`；当前裸 `python` 是 `E:\anaconda\python.exe`（3.7.6），该 API 不存在，确实会在发起 SSH 前报 AttributeError。但项目现有 `normal_training/NORMAL_source_conditioned_OBSERVER_OWNER.json:4` 已明确使用下列 3.11.14 解释器。本轮在该解释器确认 API 存在，并完成完整嵌入式命令的 POSIX 引用往返检查。无需添加 fallback 或兼容层，实际启动命令写明该解释器即可：

`C:\Users\gb\AppData\Roaming\uv\python\cpython-3.11.14-windows-x86_64-none\python.exe`

`STATIC_PREPARATION.json` 的 Python 3.7 语法通过，不等于本地 launcher 的 3.7 标准库运行兼容。原始失败观察和随后3.11检查均保留在 private trace。

## 已核对正确的实现

- **初始化与完整 E3 加载。** replay:53–92 不给 checkpoint_path；实际 `train_dist_mod.py:129` 因而令 initialize_weights=True。官方/G/A/B 先按现有 SHA、严格状态加载，Face 工厂保留 axis_prior 层级并扩展至1301项。捕获 E0 BN 后，检查 E3 epoch、完整 architecture、全部 module. 前缀，再去前缀并 strict 加载全部训练状态，逐项相等检查；E0 捕获不会覆盖后来的训练参数。
- **仅替换 BN buffers。** helper 只枚举实际 `_BatchNorm` 的 running_mean/running_var/num_batches_tracked，核对集合、shape、dtype 并原位 copy，不写 affine/Linear/LayerNorm。replay 使用同一 eval 模型、no_grad，最后恢复 trained buffers 并对全部 state_dict 与 E3 CPU 状态逐项比较。已有 CPU 报告仅是构造模型的不变量证据，未在本轮重跑。
- **输入与 RNG 配对。** 原生 get_loaders 使用 ScanRefer val、seed2027、B8、world size1；断言9508行、1189 batches和顺序 sampler。每个 batch 在 A/B 循环外取出，分支独立 deepcopy；在含 voxel 预处理的 `_main_eval_branch` 前恢复 Python/NumPy/Torch CPU/全部 CUDA RNG，下一 batch 继续 A 结束后的流。worker/generator只生成一份 cached 输入，不需要分支重取数据。
- **Mask 形状与融合。** 额外读取并绑定实际部署 `native_joint_v2/source/models/pv_ground.py:537–573`，证实 Text Mask `[1,256,S]`、QueryMask `[256,S]`；选中单一 Text Query 的 pred_scores 经 squeeze 后 alpha 为标量。replay:129–141 的 `[0,selected_query]`、QueryMask索引、scalar融合和superpoints映射正确，均与实际 root gt_masks[bid,0] 比较。QueryMask 是当前模型经 support corrector 更新后的 own Mask。
- **last/bbs 原生口径。** evaluator 使用指定部署 SHA `39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677`。native_root_bbs 与其 softmax、root positive-map 二值化、modify/pron/rel加项、other减项及排序一致；butd_cls=False，无筛框。root GT、center/size转换与IoU helper相同，阈值均为严格 `>`；结尾强制 native gts/逐行命中数分别等于预期行数和 native dets。
- **交叉 IoU 与 transitions。** cross字段分别索引 B 输出中的 A所选槽位、A 输出中的 B所选槽位，方向正确。repair/break把边界相等视为失败，与严格 `>` 一致，并校验净变化=B−A。所有256 Query保留，full256_max_iou是GT上的候选几何上界，没有新增部署排名或裁剪。这里“同 Query”仅指同一槽位；BN改变可使特征、Gumbel权重和几何变化，不能将交叉值声称为独立隔离出的评分因果效应。
- **Mask 指标范围。** 逐行own/fused Mask都对应本分支bbs所选Query；native_mask_miou_pos是position/bbs选择，native_mask_miou_sem是semantic/bbf选择。native_semantic_alignment_mask_hits025/050来自原生bbf的overall_mask，不能称为bbs Mask命中。mIoU为0–1比例；框acc字段明确为百分比。
- **冷恢复正确沿用原生入口。** 与提供的旧C-off helper直接比较，变化限于1301状态、best真实选模epoch、Face状态名及相关断言。max(native_metric_order)保留最早同分项，与正常训练严格 `>` 的best规则一致。真实工厂/loader严格恢复模型、Adam、scheduler、epoch和四类RNG，逐项比较，无forward/step/权重保存。replay本身不创建optimizer。
- **执行门槛和 SSH 引用。** launcher先检查normal complete/exit0/[0,1,2,3]、child退出、E3 identity、现有manifest、环境SHA、GPU无计算进程，使用与正常训练相同的GPU锁经flock --no-fork -n串行运行。sanity先冷恢复best/latest，再首批8行；full须另一次显式调用并要求同检查点sanity和恢复receipt完成。没有timer、活动训练改写或自动插入训练。SSH采用参数列表、POSIX shlex.join、stdin JSON和固定known_hosts/StrictHostKeyChecking，静态引用检查通过。

## 输入绑定与范围

JSON的 audited_input_hashes 使用junction解析后的实际D盘路径，包含原17项、补充3项和全部使用过的辅助文件。protocol的118项源码SHA与staging receipt相同；本轮绑定的11项实际部署源码和init asset均匹配。准备版protocol/staging receipt的历史状态字段不作为当前训练状态。

必须区分两个参考快照：指定runtime_binding/model_source/models/losses.py为920051cc…，真实部署overlay为00b86273…；本轮额外审查后者，差异为保存last indices和训练态额外监督，eval分支关闭，四个框转换/IoU helper AST相同。model_source下dataset为ee4ebfac…，不匹配部署；本轮额外找到并直接读取匹配3afebfc6…的runtime_binding/dataset_source/src/joint_det_dataset.py，确认root GT、point Mask与返回字段，不把旧快照当作部署源。

本轮只有本地读取、SHA/AST/命令引用检查，没有执行PV harness、SSH、GPU、DataLoader、训练、评估、安装或模型前向。实际模型类型/数值/显存及动态断言仍须在首批8行sanity验收后才能开展full9508。输入不承诺与历史E3点采样逐位相同；即使B改善，也只是固定检查点下BN buffers的前向影响，不替换正式成绩、不晋升最好权重、不证明正常训练配方修复。
