# Mask几何训练终点的 Nr3D / Sr3D 初始化接口

本页对应当前`native_gt_mask_geometry`固定候选。它是后续训练接续的格式准备，不代表ScanRefer已经晋级，也不代表Nr3D/Sr3D已获得新结果。

2026-09-07 22:06终态更新：该固定候选模块REC筛选失败，完整系统6677/6448低于起点的@.25及GT控制的@.25。本页代码及CPU准备仅作为历史可复用接口保留；失败终点不导出、不启动Nr/Sr。原生辅助默认关闭，未因代码准备通过而获得训练晋级。详见[终态报告](SCANREFER_MASK_GEOMETRY_TERMINAL_RESULT_2026-09-07.md)。

当前Scan终点保存84项core参数和实际optimizer；原生`main_utils.load_checkpoint`需要完整、带`module.`前缀的模型state。新增`export_mask_geometry_initialization.py`以保护E71补齐1144项，只覆盖`decoder.5`、`prediction_heads.5`、`x_query`、`x_mask`、`rel_encoder`的84项参数。注意力的`in_proj_weight/in_proj_bias`明确包含在内，全部BN运行均值、方差和计数器保持E71值。导出不携带旧optimizer/scheduler，使用现有`--model_only_initialization`，无需改原生加载器。

导出CLI先独立重算既有正式9508行及晋级条件，核对正式审计、输入与训练终点SHA；仅接受固定候选`native_gt_mask_geometry`、2482步和正确E71来源。不会根据控制臂更好而换成控制，也不重新运行正式评估。

正式Scan审计通过后，在远端项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/bdetr/bin/python \
  -m scripts.export_mask_geometry_initialization \
  --formal-directory /root/autodl-tmp/mcln_scanrefer_mask_geometry_official_20260907_v1 \
  --output /root/autodl-tmp/mcln_scanrefer_mask_geometry_official_20260907_v1/native_model_initialization.pth
```

原生后续训练使用上述权重并显式设置`--model_only_initialization --checkpoint_start_epoch 1 --start_epoch 1`。这是初始化接口，完整Nr/Sr训练的实际终点绑定、数据批次、梯度和监督接入仍须在Scan正式晋级后核验；本次没有提前启动它们。Mask几何辅助属于训练损失，初始化文件本身不会自动在新数据集训练中启用该损失。

## 已完成的执行证据

- 原Python3.7/Torch1.10.2环境8项格式测试通过，覆盖冻结state/BN保存、旧优化器去除、注意力输入投影，以及错误来源、缺失/额外参数、shape/dtype或控制臂拒绝。
- 以真实E71构造明确标记的84项合成差值，实际调用Nr3D及Sr3D原生模型工厂、优化器工厂和checkpoint加载器。每个模型的1144项state逐值一致；84项真实参数正确装入，其他1060项不变，原生named_parameters集合也与84项名单一致。
- 两个原生加载器都保持optimizer为空、scheduler不变、start_epoch为1。CPU加载阶段约6.83秒；未执行GPU前向、数据集训练或正式评估。
- 临时完整fixture为599066465 bytes，核对路径和SHA后删除；剩余测试权重0。受保护E71和当前训练文件未修改。
- 证据归档：`refine-logs/mask_geometry_initialization_preparation_20260907_v1/receipt.json`。真实训练终点尚未产生，正式通过后的CLI和真实终点加载尚未执行。

初始化准备不能证明新原生网络达到完整V99的REC，也不能证明同一辅助在Nr/Sr有效。后续仍按原生与完整系统分开报告，并以Nr3D59.82/51.38、Sr3D68.43/57.30为REC底线；Nr/Sr Mask不设当前晋级门。

## 原生训练监督入口

原生`train_dist_mod.py`现可显式使用`--native_mask_geometry_supervision`。默认关闭；启用时在原生全部GT损失之后加入当前固定的Mask几何辅助，系数1、分位数0.005、中心/尺寸和GIoU公式均与Scan配对一致，没有新增可调权重。

该入口直接读取`compute_hungarian_loss`已有的最后一层匹配结果，不重新匹配，也不永久绑定旧Query编号。Nr/Sr的`joint_det`会加入`scannet`纯检测样本，因此使用已有`sample_dataset`元数据筛选指代表达行，辅助在这些行上取均值；纯检测行不贡献辅助。全检测批次保留原生损失与梯度，不强行给第一个检测框赋予指代身份。当前数据按root目标在GT槽0的约定核验。已有四种只训练专用输出头的提前返回模式不接受该开关，防止请求的监督被跳过。

原Python3.7/Torch1.10.2环境11项CPU检查通过，实际调用原生`BaseTrainTester._compute_loss`、`HungarianMatcher`和`SetCriterion`，使用合成点云/Mask张量验证：Nr/Sr混合检测批次使用最终层而非proposal层匹配；纯检测行辅助梯度为0；全检测批次保持原损失和梯度；默认关闭不调用辅助、不新增输出字段；指定批次子集与单行目标相同。沿用的两项辅助损失测试、CLI/损失转发及旧density默认关闭检查也通过。首次CPU测试包遗漏被旧测试直接按路径读取的density模块，未完成测试收集；补齐原文件后通过，这不是模型质量负结果。

本次没有GPU前向、优化器更新、权重写入或正式评估，当前Scan运行文件逐项SHA保持不变。完整证据位于`refine-logs/mask_geometry_native_loss_cpu_20260907_v2`。这完成原生损失接入与CPU集成核验；正式Scan晋级后的真实84项终点导出、Nr/Sr真实数据GPU梯度与训练仍未执行。不得在当前运行中开启此新入口，否则会与专用配对循环重复计算辅助；后续原生训练使用该入口时仅由原生criterion加入一次。

发布时发现旧canonical远端目录中的`models/losses.py`比Git当前版本缺少已有接口，因此本轮保留其原文件，未直接覆盖。新`main_utils.py`和`models/losses.py`已发布GitHub，并保存于已检查的远端`/root/autodl-tmp/mcln_mask_geometry_native_loss_cpu_20260907_v2`隔离目录。后续正式训练须绑定新源码快照及完整依赖，不能直接在旧canonical目录使用新CLI；当前Scan的冻结快照不受影响。

## 完整原生源码快照

上述隔离入口已进一步整理为完整的622文件源码目录：

```text
/root/autodl-tmp/mcln_native_mask_geometry_source_preparation_20260907_v1/model_source
```

以已核验的618文件原生依赖为基础，覆盖Git提交`9c7c9f44ecfdada79cddfcfedfe2cfd47957b935`中的`main_utils.py`、`models/losses.py`及当前辅助/测试文件。全部文件SHA记录在`native_source_manifest.json`，其SHA为`dbdc1da768fb0689f9b7ce1b140bfafc5cc2646d99fc3ed69b1ada5a27693030`。这是一份明确组成的快照，不声称其全部文件来自同一个Git提交。

2026-09-07 20:34:41 CST在原Python3.7/Torch1.10环境完成普通入口检查：`train_dist_mod.py --help`成功；Nr3D/Sr3D参数分别构造实际原生criterion，当前辅助开启、旧local/extent关闭；7个关键模块均从新目录正常导入，无测试包`__path__`重写。该完整目录内的6项现有原生辅助集成测试通过，使用合成点云/Mask，不是6项新增实验。

首次路径检查因Python命名空间把同一个`scripts`目录列出三次而失败；实际模块来源均在新快照内。仅将检查器从列表长度相等改为解析后的目录集合相等，未改变任何模型源码或增加兼容加载逻辑。初次日志、路径诊断与修正后的结果均保留。

这一步没有GPU前向、优化器更新、新权重或正式评估；运行中Scan源码前后逐项不变。后续使用此完整目录绑定正式通过的真实84项初始化，再做实际Nr/Sr数据批次和训练预算核验。旧range预检仍属于失败extent版本，不能复用其启动入口；`nr_contract.json`中的历史评估参数（含`--eval`、旧E57路径和`max_epoch=240`）也不是获准的新训练命令。当前未启动Nr/Sr或恢复长期baseline重训。
