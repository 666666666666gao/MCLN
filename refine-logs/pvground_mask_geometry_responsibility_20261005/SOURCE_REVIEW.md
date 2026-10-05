# Mask/Box 几何责任探针源码审查

**Verdict: PASS — SOURCE_ONLY。阻断问题：无。**

审查包含计划、生成器、探针主体、生成 runner、spec、controller 和 launcher，以及实际 loader、原生 criterion 和相关复用 helper。没有修改实验源码，没有执行远程命令、GPU 或模型。

审查请求为 `gpt-6-astra` / `max` / `fork_turns=none`；实际 backend 未获独立证明，按 `same-family` / `provisional` 记录。

已确认：

- official PV → original G → 4506 distribution checkpoint 的恢复链明确；fresh R 的输出层为零，探针冻结全部参数并核对同次前向的 Query 恒等。
- loader 身份与实际历史 imports、input manifest 一致。沿用固定 seed 2027、B8、shuffle、2 workers、训练点/检测框增强和 train voxel preprocessing；逐批比较历史前 64 个行号，不按结果筛选。
- 原生 criterion 的顺序确为 proposal → last → 0head…4head，第二次 matcher 的真实训练 targets 用于探针；压缩 GT 索引正确映射回 valid slots。
- Box 叶子梯度调用原生 `(10×L1+2×GIoU)/7`，boundary-logit 叶子梯度调用现有 `distribution_loss/7`。两者都使用真实 final assignment，显式断言未匹配候选的直接梯度为零。
- 256 个候选全部保留。融合 Mask 公式及 signed native bbs 与原生 evaluator 一致；IoU 按每个 superpoint 的实际点数统计交并集，并对 selected query 作点展开校验。
- runner 无 optimizer、model backward 或 checkpoint 保存调用；model forward 与 native criterion 在 `no_grad` 下执行，结束时核对参数梯度缺失和完整 state_dict 不变。controller/launcher 只启动该有界探针，使用原有 runtime/flock，检查 protected parent 文件未变。

本地静态核验通过：Python 3.7 语法解析、生成源码与 prepare recipe 一致、已有 generation/helper 身份一致、实际 loader/native source lineage 一致。历史日志前 8 批确有 64 个互异行号；新探针尚未运行。

结论边界：这些将是当前冻结 checkpoint 上 64 个增强训练输入的诊断，不是历史每步匹配或验证集比例。直接输出梯度不代表共享参数或其他 loss 的总影响；boundary 梯度字段仅为 distribution 项。Mask/Box 达标不是语义或物理身份判据。须按结果报告实际 GT slots；若只有 root，不得声称验证了非空 multi-GT 保护。未审查另行添加的 observer/collector/CPU recount，也未宣称这些已完成。

逐项源码依据、准确 reviewed_files 路径与 SHA256 见同目录 `SOURCE_REVIEW.json`。本次未要求修改。
