# 稀疏局部记忆的条件式原生正式评估入口

2026-09-06 08:39 CST：评估入口及状态读取已实现，在训练原Python3.7.11/Torch1.10.2+cu111环境
完成19项CPU检查和CLI编译/帮助入口检查。没有加载训练终态，没有原生模型GPU forward，
没有新增正式指标或正式评估manifest。正在运行的V3训练目录和预注册质量标准未修改。

## 比较对象与进入条件

本入口保留三个状态，在同一批正式7899行输入上分别评估：

| 状态 | 作用 |
|---|---|
| protected | 当前冻结源码下的受保护parent，给出当次原生控制 |
| native | 同预算更新16个原生Mask投影张量的终点 |
| sparse | 同预算更新上述16张量并加入17张量稀疏局部记忆的终点 |

需要两个参照，才能区分原生Mask更新与新增局部证据的贡献。正式比较以当次protected
结果为准，不把已知历史4475/3759和当前源码4478/3763的差异计作新方法收益。

入口先从完整训练receipt、6172起点/终点逐行记录和实际fit点hash重新计算原质量门，
要求sparse相对native终点、受保护起点均mIoU至少+0.2个百分点且Mask25/50命中不下降。
还要求独立权重/AdamW实物审计PASS，并绑定其hash和实际两份终点artifact。训练质量
不通过时断言退出，不创建正式GPU评估，不读取一个可手工改写的PASS字段代替复算。

未来正式manifest须在实际终态通过后另建，锁定源代码、正式数据、原生历史CLI/config、
parent、训练manifest/receipt、AdamW审计和两份终点artifact。精确稀疏模块、点映射与
训练汇总脚本必须匹配V3已冻结文件字节。当前未创建符合条件的manifest或GPU作业。

## 保持原生评估含义

复用TrainTester.get_loaders()、_main_eval_branch()、evaluate_one_epoch()与原生
GroundingEvaluator。B16、4个worker、原7899行val顺序、butd_cls、源分数、size处理、
合法框过滤以及REC/Mask各自选择路径都沿用历史配置；配置只允许输出/权重路径标识改变。

单个原生模型依次临时写入native和sparse的16个Mask张量，sparse额外挂接训练时相同的
局部记忆；随后恢复protected的16张量并返回其原生输出。每批开始和恢复后检查受保护
投影精确一致，整个epoch前后检查完整parent和稀疏模块状态。没有优化器更新或checkpoint写入。

三个状态分别使用独立原生evaluator。同一输入上的Box、size、selected_source_scores
必须逐位一致；每行REC Query、Mask Query、合法最佳框Query及过滤前后Top-K覆盖必须
一致。Mask输出可以改变。保存实际输入点hash、选择/条件Mask和Top16/32/64/256信息，
逐行REC/Mask命中及IoU总和必须与各自原生evaluator完全对应。

最终报告sparse相对protected、native两组修复/破坏/净命中、scene聚类bootstrap区间，
另报固定REC Query和固定好框集合上的Mask变化。GT辅助最佳框仅供诊断，不进入网络分数。
不会把模块筛选阈值自动变成正式结果的投稿门槛，也不会自动覆盖受保护权重。
本结构REC冻结，正式Mask改善本身不能完成Nr3D REC>60%的目标；共享身份推理仍需后续独立验证。

## 已执行检查与限制

19项CPU检查中，11项覆盖新增入口/状态切换，8项复用既有原生evaluator一致性及固定
Mask质量规则。所有数值fixture为合成数据；检查了错误父权重、错误步数/分支、形状、
非有限值、非Mask额外参数及文件内容变化的拒绝；连续两批临时切换后完整模型恢复。
原生evaluator例子覆盖REC与Mask选择不同、完全没有合法REC框、将其他分支指标误归属。
真实稀疏模块仅CPU构造并核对17张量/267936参数，未执行其训练终态或正式原生forward。

v1隔离CPU目录的19项检查通过，但测试夹具导入依赖额外tests/__init__.py；最终改为
仓库实际非package的tests目录导入形式，v2移除该额外标记后19项仍通过。保留两份回执，
v2是当前代码的检查证据。原Python3.7编译和直接脚本--help通过，612份冻结源码hash一致。
CPU检查不覆盖完整DDP/native epoch执行；如果训练通过，仍须先在fit输入完成完整接口
复核，再运行已锁定的正式三状态比较。实际耗时和显存须届时测量，当前不作性能开销结论。

代码：scripts/run_nr3d_sparse_native_formal.py、scripts/nr3d_sparse_formal_state.py。
当前证据：refine-logs/sparse_native_formal_cpu_20260906_v2/，服务器同名mcln目录。
训练观察会话08:20确认有效；最近实际GPU观察仍是07:53:40两臂192/6687，约10:53下次查询。
本条记录不冒充新的训练进度。三个正式受保护结果保持，整体目标继续active。
