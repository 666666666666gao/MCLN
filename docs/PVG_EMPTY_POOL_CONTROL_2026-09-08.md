# Fixed ScanRefer empty-neighborhood pooling control

This is a zero-new-parameter control, not a validated coverage-aware method.
The corrected PV-Ground VSA source and aligned mesh/detection inputs remain fixed.
For each actual StackSAModuleMSG radius, keep the real CUDA empty-neighborhood mask
and zero its pooled vector after the original MLP/BN/ReLU/max pooling. Supported
neighborhoods retain the original computation. No source-ranking or candidate gate
is introduced. BN behavior, native REC and Mask losses, evaluator, and data remain native.

Evidence before launch: the 16-fit-scene census found actual augmented sparse-source
empty neighborhoods while raw-point neighborhoods retained support; zero grouped
input can produce nonzero pooled output through the pretrained MLP. This may encode
absence and is not by itself a numerical bug. Real CUDA supported/empty tests,
nonempty full-model exact replay, and two ephemeral full-native-loss updates passed.

Start from the official ScanRefer epoch81 parent, never from ephemeral check weights
or a failed endpoint. Use seed2027, 29778 fit expressions, 6887 fixed module-holdout
expressions, batch8, LR/backbone LR1e-5, one fit pass/3723 updates. No seed search,
extra epochs, loss changes, or branch-specific hyperparameter search. The prior
completed VSA-corrected native trial is the same-budget control (terminal bbs6119/5537).
Report both the new arm's own initial-to-terminal change and its difference from that
control; initial differences must not be described as training gain. Holdout scenes
were seen during upstream pretraining and do not establish unseen-scene generalization.

Primary output remains native bbs; bbf is diagnostic. Independent CPU audit recounts
all 6887 rows/candidate arrays. Existing own-initial two-threshold nonregression gates
the queued formal evaluation; formal acceptance still requires V99 REC5572/4797 of9508
and ScanRefer Mask58.70/50.70/44.72 percent. Nr/Sr training waits for Scan acceptance;
Nr/Sr Mask is not an acceptance gate but its native training loss is retained.

The control's boolean is not a state_dict entry. Train spec, checkpoint, receipt and
formal evaluator must bind the module SHA and explicit architecture flag. Formal
published-parent arm disables this control; terminal arm enables it. State keys and
parameter shapes remain exact. Inference still uses no MCLN Parent/Geometry/V99 sidecars.

Disk: reuse all parent weights and source directories. Reserve 1.25GiB for two peak
331MB delta checkpoints plus holdout/formal arrays and logs, and 1GiB free reserve.
Only one latest checkpoint is overwritten atomically; preserve protected parents and
fixed endpoints. No duplicate full parent or optimizer history files are created.
Queue checks every300s after estimated milestones; never start duplicate jobs.

Runtime is warm-reused under the unchanged env_spec SHA966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c.
No packages, compiled kernels, or base runtime files change. The module preserves its
OpenPCDet Apache2.0 source attribution (see licenses/OpenPCDet-LICENSE).


## Execution evidence (20.192)

当前用户上传材料记录的47/3723步已过时：VSA顺序修正后的原生微调已在17:21:55完整收尾，6887条bbs6147/5549→6119/5537（−28/−12），bbf6174/5583→6153/5550（−21/−33），独立审计通过但科学门失败，正式队列按原规则跳过，0正式行。这个结果保持负结论，不继续延训或改LR，也不声称正确接口自动涨点。

本回合将20.191的空邻域控制接入完整native REC/Mask损失。18:28:38启动、18:29:15完成，耗时36.34秒；固定训练行[0,173,237,455]，2 eval forward、2 train forward及2次临时AdamW更新，783可训练张量27959611参数、199冻结张量保持不变。非零原生REC及Mask损失参与反向；没有输出新权重文件、没有正式评估。模块SHA1eb336f59a1824fa74e778c2e0c293b2357fefbd23bdbfa66f5988cf2767c803，正确VSA源码/运行环境复用。临时更新权重不会进入长训起点。

18:36:29启动固定ScanRefer empty-pool控制，训练controller20870、screen mcln_pvg_scan_emptypool_v1。仍从官方Scan epoch81 parent严格加载；29778 fit、6887模块留出，seed2027、batch8、LR及backbone LR均1e-5，3723步/一遍fit，native全部REC和Mask损失不变。唯一机制变化是保存真实CUDA空邻域mask，在原MLP/BN/ReLU/maxpool后逐半径清零。没有额外参数、学习式coverage权重、候选Gate、复杂多尺度读取、语义/几何解耦或V99 sidecar。比较自己的initial→terminal与已完成同预算native终点6119/5537，不能将初始差异写成训练收益。

18:36:33排队独立CPU审计，controller20878；18:37:42排队正式接续controller20981，首次检查21:22:35 CST，之后300秒间隔。正式recount/evaluator函数用既有6887输出做CPU一致性检查通过，0 GPU forward/更新/正式行；这不是正式质量结果。预计当前长训及留出评估约3小时，按实际吞吐修正ETA；不高频轮询或启动重复作业。

本控制不改变state_dict键，因此仅载入权重不能保证恢复同一模型。训练spec、delta checkpoint及receipt显式记录empty_pool_mask与模块SHA；正式evaluate严格验证这些字段。published_parent臂明确关闭mask，fit_terminal臂明确开启；参数键、native bbs/diagnostic bbf、候选规则与9508行正式输入契约保持一致。独立审计通过且own-initial两个REC阈值不退化才进入既定正式流程；最终仍需V99的5572/4797及Scan Mask58.70/50.70/44.72底线。跨数据集只在Scan正式通过后继续，不恢复Nr/Sr Mask门。

磁盘启动检查可用2690916352字节（约2.51GiB）。按已有331MB增量权重实际大小，预留1.25GiB产物预算及1GiB空闲，不再复制830MB父权重或源目录。仅原子覆盖一个latest并保留固定terminal；本回合没有删除受保护文件。此前已经核验清理的Sr本地传输副本及旧native中间latest不重复删除。runtime环境未变，无包/编译修改，不重建环境。

所有通过目前仍是训练接口、数据和执行完整性证据，不是新REC增益。V99/V109/V113及Nr/Sr正式受保护结果不变；没有将新PV控制、原生输出或模块留出数字拼到完整系统表里。

本次实际观察时间：2026-09-08T18:41:05.797336+08:00；末尾日志：
```text
- This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
PVG_FINETUNE_DATASET_LOADING 2026-09-08T18:36:52.218668+08:00
Loading train files, take a breath!
Begin text decoupling......
```


## Capacity and paired output analysis (20.193)

上一回合已启动empty-pool固定训练及评估接续；本回合核实原controller20870及train20872持续存活，完成全量训练输入和实际batch8容量验证，新增不改变训练的跨控制分析队列，属于progress。无重复训练、无训练参数/规则变更。

18:44:07容量检查通过：完整原生损失13.558786、clip前梯度范数37.601467、峰值allocated16434629120字节（15.31GiB），一次batch8前后向22.88秒，0 optimizer update。随后严格恢复官方父状态并重置seed，进入initial6887评估。18:47:24实时观察为initial1536/6887、累计146.42秒；当前预计initial18:55—18:56完成并自动进入训练。此时尚未有完整initial指标或任何终态结果，容量loss不能当作质量增益。

新增独立CPU跨控制分析scripts/compare_pvground_empty_pool_control.py：将已完成VSA-corrected native与empty-pool各自的initial/terminal逐行对齐，核对父权重、输入契约、正确源码、seed、LR、预算和6887条point SHA/GT/身份字段；报告bbs/bbf修复破坏、IoU三档迁移、按物理场景统计，以及原始Full256 GT oracle。该oracle明确是全部raw候选、不冒充合法过滤后的recall，也不用于部署。终态另外核对两轮实际3723步训练行顺序，并分别报告初始差、终点差和训练变化之差；后者是描述性差分，不声称独立因果效应。输入点哈希一致不等于全部内部浮点逐位一致，场景留出仍是主干见过的训练场景。

18:47:11分析队列screen mcln_pvg_empty_compare_v1/controller21426已启动，依赖现有independent audit20878的initial_audit.json/终态audit.json及精确receipt SHA；18:56首次检查、后续300秒。不占用GPU、0模型forward/更新/正式行，不修改现有own-initial筛选或正式晋级门槛。代码已编译并按真实归档字段检查，实际数值比较须待完整initial审计后产生；当前不能声称该比较已经通过或给出增益。
