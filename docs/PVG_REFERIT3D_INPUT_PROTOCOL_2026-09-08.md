# PV-Ground Nr3D / Sr3D 接续输入核对

本项只准备 ScanRefer 正式过线后的接入，不启动 Nr3D/Sr3D 训练或模型评估。
2026-09-08 08:46:30 CST，实际服务器 CPU 审计退出0，用时9.16秒，未导入Torch。
当前 ScanRefer 的3723步训练和既有两条评估接续队列保持原样。

## 固定来源与实际数据

PV-Ground 固定提交为 `262e2592589baec7bb83a0d46aae6542d4ccedfb`。
从当前隔离 runtime 读取了四份 Nr/Sr two-stage 训练/评估脚本、四份场景划分及相关数据/评估源码，逐文件与原下载包 SHA 比较，17份均一致。
对照使用当前 Scan 训练所用的625文件数据层快照，manifest SHA 为
`190d0011bc5bfefab4a3965d972606f4dc21a946f68b2382f5a90837b3a4c4ef`。

CPU直接执行两个固定源码中的原始 annotation list comprehension，未实例化完整Dataset、未加载点云或执行场景图解析。
两版本在实际CSV上选出的原始 annotation 字典和顺序完全相同：

| 数据集 | 训练表达 / 场景 | 评估表达 / 场景 |
|---|---:|---:|
| Nr3D | 32919 / 511 | 7899 / 130 |
| Sr3D | 65846 / 1018 | 17726 / 255 |

四份声明场景列表一致，各数据集 train/test 场景集合不相交。两版本的 `data/cls_results.json` 文件SHA也一致，覆盖所有上述表达场景。
CSV实际路径、SHA、逐集合原始身份顺序SHA及场景清单保存在回执中；完整CSV未复制或提交。
身份SHA按 `[dataset, train_or_test, selected_row_index, scan_id, target_id, raw_utterance]` 的UTF-8 JSONL计算，保留重复表达的行位置。

Nr3D test只保留 `correct_guess=True`；train不按该字段过滤，也不删去 `mentions_target_class=False` 的表达。
Sr3D train/test均要求 `mentions_target_class=True`。`val`按作者代码映射到`test`。
本次没有更改筛选、目标解析或推理输入协议。

## 对象输入、训练混合与部署候选

四份上游脚本均开启 `butd_cls`、`self_attend`、六层Decoder、RGB、soft-token和contrastive对齐。
`butd_cls`在Dataset中将对象输入换成现有实例框和有效槽位mask，类别来自预测文件；因此这是实例框先验协议，不能写成纯预测检测框协议。
原始有效槽位可能不连续，继续使用现有mask与类别绑定；不加入GT instance mask清洗。

作者Evaluator在Nr/Sr开启对象重叠过滤：候选与有效对象框的最大IoU严格大于0.25时保留其得分，否则分数乘0；随后仍在256个Query上按作者`argsort`选择。
后续入口需保留这一具体操作并核对实际排序，不能自行改成删除Query或负无穷屏蔽。
这与当前ScanRefer `butd_cls=False` 的规则不同，后续不能直接复制Scan不带过滤的评价入口。
原生bbs/position仍作为同结构主输出候选；bbf独立记录，不能按数据集逐项拼接最好分数。

上游 `joint_det` 将1199条现有有效ScanNet检测场景annotation重复10次，另加语言表达。
1199来自已完成的原生输入/对象槽位审计，本轮没有重载点云重新计算。
由此当前原生联合训练行数为 Nr44909、Sr77836；这是数据组成，不是本轮已执行训练量。
后续训练预算、batch和起点应在正式接入前固定，不能自动执行上游100/140 epoch脚本。

## 视角增强差异的准确范围

当前原生数据层的 `_augment_nr3d` 复用小写、正则分词的视角词检测；上游仍按带空格的原字符串匹配。
两份源码在进入场景图解析前的caption规范化AST完全一致。
在同一文本处理阶段比较两个纯函数，得到：

| Nr3D集合 | 原始CSV检测差异 | 既有规范化后、解析前检测差异 |
|---|---:|---:|
| train32919 | 2155 | 327 |
| test7899 | 482 | 65 |

train原始文本被限制大旋转的行数：当前12307、上游10152；规范化后为12307与11980。
这些计数没有执行parser可能附加的前缀、随机增强或Dataset `__getitem__`，不能当成实际错误增强次数。
test中的函数比较也不表示验证启用了训练增强。
后续继续采用已修复的原生数据层，不为使用PV预训练而恢复旧视角检测。

## 预训练文件需要显式绑定

| 上游脚本 | 脚本中的checkpoint文件名 |
|---|---|
| train_nr3d.sh | PV-Ground_NR3D.pth |
| test_nr3d.sh | PV-Ground_NR3D.pth |
| train_sr3d.sh | PV-Ground_NR3D.pth |
| test_sr3d.sh | PV-Ground_ScanRefer.pth |

这些是固定脚本的实测路径，不足以证明作者发布的Sr评估日志使用了哪个文件，也不能自动当作本项目的起点选择。
现有官方模型库固定revision `cf4a8b1eed045f1309e6a691ea948d1d6c54448e` 单独提供：

- `PV-Ground_NR3D.pth`：829830168字节，SHA `d2d9afaf9c293c54977f3555a46c7bb2a72d9f80163a3f602dd8426032d7fa5d`。
- `PV-Ground_SR3D.pth`：829830168字节，SHA `a4a14b0090947177a648703ad6de094246891d89174fffa56fe454f730dbe3dc`。

这两份尚未下载。Scan正式过线后，按固定模型文件验证来源和config/state兼容性，再定义同一结构的预训练微调。
不因输出同为288维而直接载入旧MCLN的1144-state权重；现有PV模型为不同的1234-state结构。
也不把官方Nr/Sr日志当作本项目指标。Nr3D的上游严格阈值日志仍低于本项目MCLN底线。

## 执行回执与剩余工作

审计入口：`scripts/audit_pvground_referit3d_inputs.py`。
证据：`refine-logs/pvground_referit3d_input_protocol_20260908_v1/`。
远端审计目录：`/root/autodl-tmp/mcln_pvground_referit3d_input_protocol_20260908_v1`。
执行源码SHA `15dcec3676d3cb8969b9451feaeca1d200dbe07d80c1ecf2255f5706b683c609`；
回执SHA `49c32f59d5c38471885b96160f677deb0eabbe3d9c7cf1d1f4c5c48c58d7f9f9`。

本项模型前向0、更新0、正式评估0、权重下载0。它没有复活旧OpenShape外观分支及其条件队列。
真实Nr/Sr模型strict load、体素输入、候选过滤、完整前向/反向与训练尚待Scan正式晋级后执行。
当前三个数据集的正式保护指标不变。


## 2026-09-08 Nr3D父权重准备接续

上文“尚未下载”是原CPU协议审计的时点状态。现在仅Nr父权重已完成本地下载及829830168字节/SHA校验，正在传向独立服务器目录；Sr仍未下载。此次准备先于训练，不改变Scan正式晋级后才启动Nr/Sr训练的顺序。
远端清点和CPU strict-load已设置依赖接续，当前尚无完成态；并未执行真实Nr/Sr网络前向/反向或正式评估。详见统一交接§20.178和 `refine-logs/pvground_nr_checkpoint_inspection_20260908_v1/`。
权重不进入Git；完成远端校验和CPU检查后清理本地中转副本。原runtime包与spec不变。


## Nr3D实际父权重已核验（15:44 CPU完成态）

固定Nr文件已经传到服务器，SHA与829830168字节均一致；实际epoch25。保存配置是butd=False、butd_cls=True、butd_gt=False，作者用三者的OR启用模型对象流。首版清点把butd也要求True而失败，属于检查脚本假设错误，已保留v1异常；不能把合法的butd_cls协议误判为不使用对象输入。

Nr状态共1235项，比Scan多一个确定性的RoBERTa position_ids；其余共有形状一致。CPU strict-load使用同一修正VSA模型源，保留Nr实际存储的buffer，1235状态逐项精确加载，通过记录在v2目录。没有strict=False、删键或重新训练。可学习参数规模与Scan一致，不把序列化buffer多一项视为新架构。

本地中转副本已在成功校验后清理；远端父权重位于/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1/PV-Ground_NR3D.pth。实际Nr数据前向、反向、训练与完整REC依然未执行；Sr准备状态未改变。作者保存的学习率和优化器不是本项目已经批准执行的续训配置。

## Nr3D真实CPU预检批次已冻结（16:32完成态）

已沿用既有selection manifest ef0d6d75...的Nr条目，固定取索引0/1/2/3和12/13/14/15，组成4条Nr表达＋4条ScanNet联合检测的增强训练批次，seed2027一次顺序采样。它是接口/容量预检批次，不代表44909条完整联合训练的混合比例，也不用于挑选模型或评估成绩。

数据源复用已审计的model_source与manifest190d0011...（既有Nr getitem/voxel输入路径），模型尚未实例化。使用butd=False、butd_cls=True，实际模型输入为协议实例框＋预测类别；检测框及合法槽与all_bboxes/all_bbox_label_mask逐项相同。目标中心/尺寸、positive_map和Mask标签只在独立native_batch中保存，不进入模型inputs字典，不能据此称完全无对象GT先验。

CPU作者体素处理后400000个原始点、299785个体素；原始点顺序、RGB和batch序号检查通过，对象输入[8,132,6]、superpoint[8,50000]。inputs与native_batch保存后全部张量和非张量字段回读一致。gt_masks精确二值转bool只用于存储，后续原生损失前必须恢复float32；Nr Mask不作晋级门不意味着此次擅自删除原生Mask训练损失。

文件/root/autodl-tmp/mcln_pvground_nr_native_batch_20260908_v1/native_batch.pt，135447612字节，SHA7974b2bd74a4c8990501d6ea2264440cc985a2145090aa04efc515394d813f98。原始批次留服务器，不推送GitHub；代码和receipt/日志在同名refine-logs目录。16:32:26完成，用时68.49秒；16:34:53观察退出0、准备进程已结束，磁盘3701506048字节。0模型前向/参数更新/正式行/权重加载，CUDA未初始化。后续仍须实际Nr模型前向、原生损失、反向和候选读出检查，不能把输入成功视为训练或REC通过。


## Nr3D真实批次索引和文本标签绑定检查（16:55记录）

Nr3D固定真实批次CPU绑定检查通过：预测类别索引0—362，位于实际父权重485行类别表内；离线RoBERTa得到[8,47]输入，词表50265行，八行有效长度26/16/15/34/47/47/47/47。六类positive map均为[8,132,256]、有限且非负；各行EOS之后至254无非零标签，255槽也均为零。本批次没有发现类别/词表索引越界或文本标签超出编码长度。

检查绑定已冻结native_batch、Nr父权重SHA和1235项状态清单；没有重新加载权重或实例化模型，0模型前向/更新/正式行，CUDA未初始化，执行耗时2.0956秒。gt_masks存储[8,132,50000] bool，原生损失前恢复float32的要求不变。证据在refine-logs/pvground_nr_batch_bindings_20260908_v1，脚本audit_pvground_nr_batch_bindings.py。本检查只能说明这八行的索引范围与token/标签尺寸一致，不能证明类别本体映射、损失/梯度或模型精度正确；真实Nr模型前向与训练仍待执行。


## 原生损失标签及Mask类型接续说明（17:01记录）

对同一真实Nr预检批次完成CPU原生损失输入检查，全部14个所需标签字段存在；8行有效目标数1/1/1/1/7/8/6/8，共33个。有效目标中心及尺寸有限，三轴尺寸均为正；所有33个目标Mask均非空，文本map在有效目标索引下形状一致。四条语言样本与四条联合检测样本的language_dataset均为nr3d，因此原作者按batch首项决定的分类/对比系数在本批次为1.0。sem_cls_label属于原生训练标签，不与前项对象输入485类预测表混为同一索引空间。

同时读取当前修正VSA源的models/losses.py与main_utils.py，SHA记录在回执：criterion启用boxes/labels/masks/contrastive_align，共proposal、last及五个中间prefix。Mask分支在选出匹配目标后调用.float()，再scatter_mean；matcher的Mask cdist也明确转float。因此§20.184—185的“损失前恢复float32”应准确理解为沿用原生内部转换，无需另将[8,132,50000]整个bool数组提前展开成float32；当前Scan使用bool存储也不能据此误判为Mask计算错误。不改变任何已运行训练或原生Mask权重。

本项只完成真实标签选择、尺寸/有限值及源接口核对，未执行criterion、模型前向或反向，0优化器更新/正式行/权重加载，耗时1.7276秒；不能标记为完整Nr训练接口通过。证据目录refine-logs/pvground_nr_loss_inputs_20260908_v1，源码audit_pvground_nr_loss_inputs.py；原始批次仍留服务器，未上传Git。


## Nr3D真实前向/反向入口准备（17:08，未执行）

已准备check_pvground_nr_native_backward.py，复用实际Nr父权重、修正VSA源和固定8行真实批次。入口先读取正式Scan controller.exit、9508行独立audit及其receipt SHA，要求现行晋级检查全部通过，才实例化Nr模型。没有启动或排队该入口；当前Scan不受影响。

预检定义为一次train-mode前向＋完整原生criterion＋一次反向，0optimizer.step、0新权重文件。严格加载1235状态，保持butd_cls的OR启用方式及Nr实际position_ids，核对783可训练/199冻结张量；记录七个关键模块梯度、全部有限loss、峰值显存、缺失梯度和耗时。参数必须未改变，BN等已序列化buffer在检查后恢复父状态，父文件SHA保持；bool Mask仍沿原生匹配/损失内部float转换。运行时需由同一GPU锁下的持久控制器执行，且排在已存在支持采集之后，不与之竞争；具体launch尚未建立。

本地Python3.13和服务器既有runtime Python3.7.11编译语法通过，源码SHA0b4bc6a214cc2a90671ce12e427801b50ebffc343547a190e6d18461c044c2e3。证据目录refine-logs/pvground_nr_backward_preparation_20260908_v1。该状态严格为prepared_syntax_only，不能称真实前向/反向通过、满数据容量通过或Nr训练成功。环境spec966235b2未改，没有重建包或改环境台账；Scan正式未晋级前不执行此GPU预检。
