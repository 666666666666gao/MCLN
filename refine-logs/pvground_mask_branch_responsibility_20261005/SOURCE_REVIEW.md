# Query / Text / fused Mask 源码审阅

**Verdict: PASS — SOURCE_ONLY。阻塞问题：无。**

审阅了新增探针、生成脚本、spec、controller、launcher、observer、collector 和 CPU recount，以及相关未改动 loader、模型构造、原生 criterion/evaluator。没有修改实验源码，没有执行远端、模型、GPU、optimizer 或新结果分析。

请求配置为 `gpt-6-astra` / `max` / `fork_turns=none`；实际 backend 未获独立证明。结论按 `same-family` / `provisional` 记录。

已核对：

- 原生 Text logits 为 `[1,256,S]` 的同一份展开，Query logits 为 `[256,S]`，alpha 为每行标量。索引、单分支 sigmoid 阈值与原生 logits 融合公式一致。
- 按 superpoint 的输入点数统计三条分支的 root Mask 交并比；保存全部256候选的整数交并计数。选中 Query 的三条 Mask 另行展开到输入点并核对。
- 原生 matcher 调用顺序为 proposal、last、0head…4head；第二次匹配与 last boxes 一致。`valid[targets]` 正确恢复 GT 槽号，新增统计排除所有已匹配候选。
- signed native bbs 的正/负文本证据计算一致。只读检查旧64行归档分数：顶部分数无并列，新增 argmax 与旧 native selected Query 全部一致。
- official PV → original G → protected4506 构造未改；fresh R 零输出、全模型 frozen/eval。新 forward 内核对 R 前后 Query 相等，不要求独立 CUDA forward 位级复现。
- seed2027、B8、两worker、shuffle、训练 preprocessing 和点/检测框增强沿用已闭合探针；逐行检查旧64的 row、scan、点内容及 root box。
- runner 无 optimizer、梯度重放、backward 或权重保存；forward/criterion 在 no_grad 下运行，结束时核对 state_dict 未改及无参数梯度。
- CPU recount 用已存整数计数重算资格和候选分组，并记录与旧探针的 Box/融合Mask 阈值、匹配差异；没有把这些差异改写成旧实验结果。
- launcher/controller 使用已有 runtime/flock 和前序闭合检查，observer 首查300秒/后续240秒，collector 只收诊断产物且拒收权重。

本地静态检查通过：8份新 Python 文件按3.7语法解析；runner 与生成配方、spec 与继承配方一致；generation 摘要、17份沿用 helper 摘要和实际 imports/source lineage 一致。没有发现需修改的源码问题。

边界：新64行实际执行结果和终审尚未产生。GT 资格是训练样本上的诊断代理，不能证明物理对象身份或准确率提升。旧64行实际均只有 root GT，多目标槽映射仅做源码核对。本审阅不批准新几何正样本池或新训练目标。完整文件路径、SHA256、读取范围和检查记录见 `SOURCE_REVIEW.json`。
