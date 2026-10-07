# ANALYSIS_AND_INTAKE_SOURCE_REVIEW — SOURCE_ONLY PASS

同上下文、同系列、临时接受；实际后台/model/effort 均为 **UNATTESTED**。阻塞项0、非阻塞项0。

收集器与旧face-support版本逐字节相同。它在连接前要求真实fit_wait闭合、controller不存活、exit0和complete；远端再次核对退出码。归档只遍历隔离root，排除实际产生的.pth/.pt/.tmp权重及临时文件，以manifest/tar流和逐文件大小/SHA校验收集文本、源码及完整数组，不删除或加载权重。

分析器字段已与本轮eval/train/save/restore逐项核对。两次9508完整验证各1189批，NPZ实际键为row_ids/root_gt/scores/parent/content/box_conditioned；每表达保留256候选。CPU独立重算三类Box共14604288个IoU并记录阈值/oracle差异，已存原生selected IoU的命中、修复/破坏及summary另行对账。Mask仅重计每组已存所选IoU，不声称重建原始Mask。

跨遍历及历史输入保护使用row/scan/target/GT/point摘要对齐；没有要求Query或父Box逐位相同。父Query、全部256父Box/score的实际漂移独立记录。同次initial零头与父输出的比较不被写成历史增益。

最佳选择沿用旧F：仅历史protected与两个真实trained terminal可入选，joint5620/4764优先，再strict、wide，完全相同保留protected。零头initial复评不可入选。分析只记录候选，不执行权重保留/删除，不宣称三项贡献或完整目标完成。

14个实际文件绑定见JSON。两个新脚本及一段嵌入程序Python3.7 AST通过；8份当前runner和98文件port摘要保持一致。此次没有查询当前训练/日志/进度、检查当前fit_wait是否存在、读取本轮输出、执行collector/analyzer、SSH、Torch/NumPy或权重加载。实际closed0、归档及结果核验仍须在原观察器关闭后完成。
