# PV-Ground ScanRefer固定终点的正式评估接续

此计划在原定3723步微调仍运行、终态结果未知时固定。它不修改当前训练，也不新增Loss、Epoch或模式选择。

## 启动条件与固定模型

等待当前训练及独立CPU终态复核完整退出0。核对训练spec/source/plan/terminal SHA、29778行恰好一次、6887行对应与重计一致。固定主bbs REC两项相对原生起点均不退化，才启动本次9508-row评估；否则记录模块筛选负结果并跳过，不扫描其他权重或模式。

同一PV-Ground架构依序评估两个固定状态：官方ScanRefer epoch81完整预训练、当前3723步terminal。由同一父权重加终点增量恢复完整state后strict=True加载；验证增量只覆盖训练时允许的参数和buffer，冻结RoBERTa原值保持。两次评估之间不训练、不挑选输出。最终候选固定为terminal，官方起点只作同协议参考。

## 固定输入与输出

- 正确`DATA_ROOT_mcln_meshsp`，312份val superpoint与既定SHA清单逐一核对；ScanRefer原val列表共9508条表达、141个表达场景，顺序及annotation SHA绑定已完成的输入检查。
- 使用当前已经核验的625文件Dataset来源及固定官方PV-Ground模型/loss/Evaluator来源，记录实际导入路径；不将两套`models`模块混用。
- XYZ/RGB仍50000点，BUTD预测检测框；`butd_cls=False`、`butd_gt=False`，不加入GT实例框、GT Mask或GT Anchor作为推理输入。GT仅在模型输出后用于原GT损失和Evaluator。
- 两模型均batch8、两worker、seed2027、无训练增强。使用已通过batch8检查的相同设置；它与旧V99的历史batch12/seed0不同，不能称为本次同步重跑V99的配对结果。比较V99时使用受保护正式历史值，并明确其完整系统身份。
- 主输出固定soft-token position/bbs，contrastive/bbf独立记录完整Box和Mask结果；尺寸沿作者clamp(min=1e-6)，不新增对象重叠过滤。Mask按同模式选中的Query输出，不跨模式挑最佳Mask。
- 每行记录身份、点SHA、root GT、选中Query/框/IoU/Mask IoU，保留256个原框和两源分数。调用原作者完整Evaluator并核对逐行汇总。

独立CPU正式审计重新检查所有候选有限性、实际最高分选择、选中框与原框对应、轴对齐IoU、两模型输入与GT对应、阈值计数及修复/破坏。Mask审计重计已保存的逐行IoU，不冒称重新读取未落盘的二值Mask。

## 固定晋级规则

只看terminal的bbs完整指标：REC@0.25至少5572/9508，REC@0.50至少4797/9508，Scan Mask@0.25/@0.50/mIoU至少58.70%/50.70%/44.72%。不要求在正式集上逐样本支配预训练起点，不新增子群门槛。59%/51%是伸展目标，未达到不阻止已达底线的结构接续Nr/Sr REC。

达到这些规则且独立审计通过后，按用户要求尽快训练同结构Nr3D/Sr3D REC；它们的Mask不作为晋级门。未达到则如实封存本次完整结果，不改变seed、batch、模式或终点重新选择。PV-Ground为引用的外部预训练基础，不作为本项目原创模块。

## 执行与资源

CPU依赖队列接近预计完成时间开始查看，然后每300秒观察。只有依赖终态成立后才取得既有GPU锁并启动正式模型进程；原训练不被停止或热修改。原进程异常时记录失败，不因观察超时重复启动。

只在独立正式目录保存结果和小型回执，不写新checkpoint；原始候选NPY留在服务器，代码、计划、退出记录及汇总及时同步远端主文档、GitHub和Desktop主文档。此文档是预先固定的执行计划，不代表9508条已经评估或指标已经达到。

## 本地结果收集

`scripts/observe_pvground_scanrefer_pipeline.py`在已有训练观察器上补齐正式阶段：同时收集训练终点、CPU终态审计、正式队列决策、published_parent/fit_terminal各自回执、正式总回执与独立审计退出码，并记录三个任务目录对应的实际进程。只读取远端文件并保存本地快照，不启动、停止或改写服务器任务，不下载候选NPY与权重。

训练`controller.exit`只代表训练控制器终止。正式单臂回执、正式总回执和独立审计必须分别读取；只有全部对应退出成功且晋级规则经独立审计成立，才认为本次正式结果完成并可接续Nr/Sr。新版收集器已按实际writer输出文件名核对并做语法检查，第一次远端执行安排在接近10:30；本次增加收集入口本身不代表正式评估已经开始。
