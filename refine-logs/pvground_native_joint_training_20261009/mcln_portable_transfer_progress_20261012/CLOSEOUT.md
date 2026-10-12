用户新增要求把当前最好版本的有效改进迁移到加载MCLN baseline，观察跨baseline效果。PV主线与完整60.0%/52.0%、三个有效贡献及Nr/Sr独立验证训练目标保留，不把本次试验视为PV目标完成或主线取消。

实际隔离MCLN原PointNet++及语言/候选/Mask核心，使用此前真实验证的E71 checkpoint，严格核验794125833B、SHA3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208；只移除9项旧SourceChoice，严格加载1135核心状态，不恢复V99/旧CS/R侧链。迁移当前已训练的A候选支撑校正27841参数和B极值来源轴向范围混合29793参数及模型内Mask空间参考，两者使用对应原Query和超点288维接口。PV专属六源G读取没有MCLN对应来源，未迁移G核心；零输出Face分支不计入。不能把本实验描述成整个PV参数原封不动替换baseline。

fresh源码审查SOURCE_ONLY WARN/0阻断，同family/provisional、实际Astra/max身份UNATTESTED。修复实际逐行评分与native主token map二值化不同的问题，改为调用native_parse_gt和排序；补齐工厂必需的类别嵌入及解析器资产，使用warm已编译PointNet扩展，不增加fallback、不构建新环境。736源码/资产清单核对；实际CPU工厂08:47:51完成0，原核心不变、1159完整状态和A/B严格加载、原生冻结RoBERTa策略保留。

真实8条GPU三路配对08:51:50完成0：native6/5、Mask-reference5/3、A+B迁移5/3，属于工程sanity，不是正式精度或完整方法失败。输入每batch只读一次，Python/NumPy/TorchCPU/CUDA RNG配对，语义分数逐位相同，全部1159模型状态不变，无optimizer更新或新权重。GT仅在前向之后合入原生criterion/evaluator，推理模块不读GT、不删除256候选、不增加第二排名。

完整9508三路last/bbs已于08:58:59实际启动controller266328，尚无终态或全量MCLN指标。由8条三路耗时6.846秒保守估计11:16:39完成，唯一观察器10953/PID46664首次11:13:39，之后如确需查看则240秒；未提前重复查询。由于首batch包括warmup，估时不是完成承诺。当前仅测直接迁移效果，尚未在MCLN上训练A/B；待结果决定正常训练的适配及预算，不把加载其他骨干的已训练参数称作MCLN训练完成。

此前PV固定E3参数BN诊断实际08:36:54完成0，9508配对：trained统计5549/4482、E0统计5529/4490，替换统计净−20/+8，严格修复579/破坏571，8045条Query槽位变化。只改变252BNbuffers，参数相同、输入/RNG配对、完整状态恢复exact。逐行9,139,234B已核对并CPU重算，fresh实际完整性WARN/0阻断；只支持固定checkpoint前向作用，不能解释/追回原训练−444条或称为正常训练修复。历史正式E3仍5551/4476，不被独立重放覆盖；不晋升混合状态。

best与negative latest完整归档SHA核对。唯一已归档的negative latest841964750B在诊断关闭后09:02:38远端清理；最佳、MCLN E71、PV/G/A/B/V99和数据保留。本地仅清理590807040B中断重复partial，原故障文字与归档完整状态保留。MCLN试验本身没有写新权重。

当前PV保留最好5677/4920=59.7076%/51.7459%，目标5705/4945仍差28/25。三项有效贡献、最终正常训练改进及对应Nr/Sr完整验证训练未完成，goal ACTIVE_UNMET。