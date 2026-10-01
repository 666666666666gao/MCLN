# PVG G/P2 离线分析脚本审查

- Date: 2026-10-02
- Reviewer: `gpt-6-astra`, reasoning effort `max`
- review_independence: `same-family`
- acceptance_status: `provisional`
- Verdict: **PASS**
- Scope: `scripts/analyze_pvground_g_p2.py`。本次是 fresh-agent 代码审查，不是外部模型家族独立验收，也不是完整结果验收。

BLOCKING: 无。

NON-BLOCKING: 无。当前实现正确，无需修改。

## 核对依据

直接阅读了 `scripts/run_pvground_g_p2.py` 的原生逐行记录和 receipt 写入逻辑、`scripts/run_pvground_g_p2_pair.py` 的串行完成协议、`docs/PVG_G_P2_PLAN_2026-10-02.md`，以及实际部署 spec 的本地原件：

- `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/g_control_spec.json`
- `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/g_p2_spec.json`
- `refine-logs/pvg_g_p2_20261002/g_control_initial_receipt.json`

两份实际 spec 除 `root` 和 `p2` 外完全一致，固定 seed2027、batch8、fit_passes1、core/backbone LR1e-5，并指向同一 G 起点。写入端另外断言 weight decay5e-4、clip0.1，重新建立优化器。

## 正确性结论

1. 分析脚本第82–102行在生成输出前要求 pair 完成，四个阶段顺序完整，每臂均完成3723更新、29778行单次消费，且分别存在6887行 initial/terminal 和9508行 formal。状态和字段名称与实际 controller/writer 一致。训练 receipt 还须证明 fresh optimizer、冻结参数不变及同一 G 起点；spec 的成对比较保留了共同协议。
2. 第13–50行验证实际 rows 文件与既有 receipt 的校验值、行数、唯一 row_id、有限 IoU、oracle 数组，并重新计算 REC、mask hits、mask IoU 总和与均值后逐项核对。写入端第418–423行保存的 mask 总和来自相同逐行值，因此当前容差没有与原生 evaluator 的批次累加方式混淆。
3. 逐行 REC IoU 来自写入端第383–407行的原始数据集 root GT box；mask IoU 使用 `gt_masks`。模型输出没有被用作 GT。写入端第415–416行还核对了原生 evaluator 的 REC hit counts。分析采用相同的严格 `>0.25`、`>0.5` 命中规则。
4. 第53–74行逐行检查两臂相同 row_id、scan_id、target_id、GT root_box 和 point_sha256，再计算 repairs、damages，并验证净增量等于命中数之差。候选覆盖数组对应写入端的排名前16/32/64/256个候选；完整256候选可覆盖错误及两类修复的计数正确。解释明确承认框与评分均会改变，没有把修复归因为固定框重排。
5. `bbs` 为固定主口径，`bbf` 单独报告；初始、终态模块留出和开发验证分别呈现。6887行曾被作者预训练见过、9508行属于开发验证、G既有适配历史需额外披露、单seed没有跨seed显著性估计等限制均明确。
6. 开发目标准确使用宽松命中数至少5615、严格命中数至少4754；4754/9508正好是50%。历史G的4495严格命中只作为显式历史参考。`same_budget_strict_increment_preserving_loose` 精确表达宽松净增量非负且严格净增量为正；它未被标成双阈值均严格增加或跨数据集成功。Nr3D/Sr3D新方法结果仍明确不可用。

## 实际检查与边界

- Python AST解析与 `compile(..., 'exec')` 通过；`--help` 正常退出。未导入训练依赖，也未触发模型执行。
- 通过本地实际 JSON 比较确认两份部署 spec 只有 `root`、`p2` 不同。已有 initial receipt 的字段与分析器契约一致；其中bbs6176/5602属于6887模块留出，不能称为新9508正式成绩。
- 完整六组逐行结果尚未在本次审查中提供，因此未执行完整分析或声称结果已验收。没有构造或使用合成正式精度，没有访问SSH、修改运行中代码或启动GPU任务。

本次PASS仅允许该离线分析实现进入后续实际完成结果的核对流程；P2收益仍须等待真实两臂终点及两次9508评估完成。
