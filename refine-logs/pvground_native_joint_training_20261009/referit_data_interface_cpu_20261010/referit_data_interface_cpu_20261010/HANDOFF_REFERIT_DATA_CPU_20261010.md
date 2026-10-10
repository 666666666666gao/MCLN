## §20.376.142 Nr3D/Sr3D真实数据接口通过有限CPU检查；旧候选数组完成归档与清理

承接§20.376.141，本节不重复ScanRefer架构和正式指标。活动C-off正常训练及唯一观察器保持原样，未提前查询训练。当前保留最好5677/4920；三项有效机制与完整模型的Nr3D/Sr3D训练仍待完成。

### 作者输入与实际数据项

21:11:34.938921—21:13:33.334021，独立CPU子进程复用真实Joint3DDataset，串行构建Nr3D、Sr3D各自train/val。使用已保存作者配置：颜色输入，butd_cls/joint_det/detect_intermediate=true，butd/butd_gt/height/multiview=false。对象输入是标注场景实例框＋cls_results预测类别；训练框沿用原生增强和抖动，不能称为纯检测框或GT类别。原数据类在butd_cls覆盖前仍读取GroupFree文件。

直接使用overfit=true，仅将每种来源的注释限制到前128条，保留真实train split与增强；没有使用会把训练split换为val的debug入口。每组有限train为1408条（128条指代＋10×128条检测注释），val为128条。这不是完整基准长度或正式fit预算。场景pickle与超点仍按完整split加载；本次峰值RSS为15298314240字节，不能外推为GPU显存。

两个基准各读取训练指代、训练检测、验证指代三种接口，合计10次getitem（包含collation的再次读取），两次真实DataLoader/default collate，训练批次XYZRGB为2×50000×6。六个按基准上下文选择的数据项共用同一个ScanNet场景检测注释；按原始来源计为四个指代表达记录＋一个共享检测场景记录，不能将原结果中的distinct_selected_annotations=6说成六份互不重复的原始注释。执行源、原始回执和结果字段均保留，计数口径在审查中限定。

实际单条GT Mask为132×50000，文本positive_map为132×256；验证root槽0的Mask/点标签一致、有效且非空，以及目标ID、proposal来源与预测类别接口。检测行分别保留language_dataset=nr3d/sr3d、sample_dataset=scannet；两条检测接口各有7个有效GT，Sr3D验证接口有2个有效GT。只检查root对应和接口，没有逐一重算所有辅助GT，也没有运行多GT损失保护。

源码与实际审查结论及具体范围以同目录审查JSON/Markdown为准，原始payload、stdout、stderr和源码哈希门禁分别封存。请求审查身份为gpt-6-astra/max，实际服务身份UNATTESTED，按same-family/provisional报告。原生预处理使用本地spaCy/sng_parser CPU模型和离线RoBERTa tokenizer；本次没有构建PV网络、PV前向、criterion、优化器更新、GPU训练、保存权重或候选数组，没有REC精度结果。

### 后续跨基准还缺什么

此前作者核心真实CPU加载已通过，本次补上了有限真实数据接口；二者不等于正常训练入口已完整准入。既有隔离Nr/Sr入口的main仍断言butd=true且butd_cls/joint_det=false，与本次作者数据协议不一致。最终模型确定后，应在新的隔离入口中根据明确协议作最小修改，不改活动ScanRefer源码；再完成实际GPU前向、原生loss/各模块更新、完整模型与优化器恢复，才启动各自独立训练。

Nr/Sr原生评估的butd_cls相关过滤也需要与输入协议一并核对和公开；本次没有执行评估器，不能宣称完整评估流程已验证。两组作者权重仍可作为对应起点，新增模块独立初始化，不直接将ScanRefer终点测Nr/Sr冒充各自训练。保持单seed2027、单A100串行及最终统一模型版本。

### 一份旧数组的精确清理

原传输session51919在21:03:05.349796完成，耗时7603.887秒；308574336字节的旧EG-Sr3D候选数组已完整归档至C:\Users\gb\.codex\archives\pvg_retired_candidate_arrays_20261010\eg_sr3d\candidates.npy，SHA256为8621cc084725688fecbe9ea8e2a9655728e7b14b922cb2c10774158d89d75937。没有观察超时重启或重复复制。

21:05:28.034661再次核对本地与远端字节/SHA后，仅删除/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1/formal/candidates.npy这一份已结束实验生成的数组，释放308574336字节；删除后数据盘实测可用2326523904字节。这是当时磁盘快照，不能当作服务器当前余量。最佳/作者/G/A/B/V99权重、日志、数据集与运行环境均保留，未删除权重。

后续首先等原C-off观察器21631/PID48772于10月11日01:03:05检查完整结果，再根据正常训练对照决定六面残差与最终方法。不得将本次数据接口工程结果当作新增有效贡献或跨基准涨点。
