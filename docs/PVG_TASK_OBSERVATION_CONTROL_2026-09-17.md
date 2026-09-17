# D：共享候选身份的语义／几何观测读取

固定设计起点为1229025及已封存C结果。C对同预算A/B的REC分别+20/+9、+3/+65，是研究读取分工的有限正证据；自身起点-8/-3，正式已跳过，不能追认晋级。本方案尚无性能结果。

## 唯一主要机制

在最后层保留原自注意力、完整文本、对象交互和原生视觉交互。观测内容与状态Key/Value、六路投影、attention和输出映射全部共享。对同一实例query q，新增两个零初始化288×288矩阵，得到q_sem=q+W_sem q和q_geo=q+W_geo q；将两组query沿查询轴合并，在一次共享Key/Value构建中读取同一6144-token记忆，再拆回原候选顺序。

两路分别经过原norm_v/FFN/norm2，参数共享，三处dropout分别使用同一个掩码；前五层保持原执行。这样改变的是注意力读取问题，不是固定任务常量或最终两个MLP。增加165888参数，读取模块923616参数，比C多一个state tensor（37新增state总数）。在C原安装后复用其所有实际参数，再创建零矩阵，不增加初始化随机抽样。

输出责任：semantic进入最后层soft-token类别评分和对比投影；geometry进入最后层center/size与Query Mask。原Text Mask、alpha、Hungarian匹配、全部原生损失、同一候选编号和最终一次选择保持。任务分工不声称统计独立或消除了梯度冲突；Key/Value及主干仍共享梯度。没有硬规定粗细来源的用途。

## 工程验收

1. 复用真实C安装参数；零初始化task矩阵不改变RNG。实际CUDA上，初始输出与C的数值差及RNG分别记录，不能外推全量逐位一致。
2. 模块非零输出层下，相同task变换应得到相同两路；改变语义变换不能直接改变几何读取，反之亦然。两路都对特征与各自矩阵有有效梯度，不能只验证零输出。
3. 共享dropout尾部在训练态和C单路对同一输入逐项比较，结束RNG一致；相同两路得到相同输出。真实模型检查语义分数、对比投影、center/size及Query Mask是否接到指定路径。
4. 正确接口真实fixture上的完整forward、原生loss和两次临时更新，新增task矩阵梯度与变化、冻结参数、保存和严格CPU恢复。无正式验证，不以梯度通过宣称精度增益。
5. batch8容量测量后才能启动固定训练；记录额外查询attention与尾部计算成本，不声称零计算开销。若不适配原容量，先报告实测原因，不静默改batch/点数/预算。

## 后续固定实验

不加载失败C终点续训。从同一官方ScanRefer epoch81、修复mesh/增强/VSA接口、seed2027起步，沿用C同一29778 fit/6887 holdout、batch8、3723步、LR与backbone LR1e-5、原损失和bbs主输出；bbf仅诊断。先完整零更新，再固定终点，不用中间checkpoint择优。比较自身起点和已有同预算C/A/B，保留已知跨进程数值差异，验证训练行顺序。

本轮不新增状态维度、Key-only/Value-only训练、边界分布、质量头、教师、关系图、额外主干、输入采样或多seed。模块holdout通过既定起点不退化筛选后才按固定正式流程评估9508。ScanRefer仍需同一系统REC至少5572/4797与Mask58.70/50.70/44.72，达到后尽快Nr/Sr REC，不恢复Nr/Sr Mask门槛。

## 来源与命名

这是C内部的任务读取对照，不声称首创任务attention。直接复用本仓库C及PV-Ground的预训练模块，保留原许可证。主张仅在正常数据、同预算配对增量后讨论。C工程修复、V99成绩以及D尚未产生的结果不得混记。

## Fixed endpoint outcome, 2026-09-17 15:20 CST

Completed exactly3723 updates. Main bbs6136/5547, own start-11/-2; matched C-3/+1. Integrity audit passed; frozen quality gate failed, formal skipped. D is sealed as no independent dual-threshold gain over C; no task-disentanglement or new-scene claim. Full results and original comparisons are in master §20.213.
