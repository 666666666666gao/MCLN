# 完整预训练几何组件的资源核验

本次目的是检查可用的完整预训练点—体素几何组件，减少后续从随机初始化训练的成本。当前MCLN/V99未替换，没有新增训练、正式评估或性能晋级。固定OpenShape外观方案已封存；本次也不重新运行失败的Mask专项稀疏分支。

## 上游来源与公开结果

已固定PV-Ground官方GitHub提交`262e2592589baec7bb83a0d46aae6542d4ccedfb`和官方Hugging Face revision `cf4a8b1eed045f1309e6a691ea948d1d6c54448e`。模型库实际提供ScanRefer、Nr3D、Sr3D的single/two-stage共六份权重及六份评估文本；保存了原始元数据、文件大小、LFS SHA和下载来源。附加源码同时验证Git blob SHA及SHA256。

以下只是作者发布日志中`last`、position、top1的REC数字，不是本项目运行结果，也未完成协议等价复核：

| 官方预训练模型日志 | 表达数 | REC@0.25 | REC@0.50 |
|---|---:|---:|---:|
| PV-Ground ScanRefer，two-stage | 9508 | 59.560% | 47.402% |
| PV-Ground Nr3D，two-stage | 7899 | 61.899% | 50.519% |
| PV-Ground Sr3D，two-stage | 17726 | 68.914% | 58.653% |

ScanRefer严格阈值仍低于本项目保护V99，Nr3D严格阈值仍低于原MCLN底线。因此不能把下载这些权重写成三数据集已经达标，也不能拼接作者其他选择路径的单项指标。公开日志的月/日时间戳未给出年份，不作为本项目当前运行时间。

- 官方仓库：[PV-Ground](https://github.com/AaNnWwTt/PV-Ground/tree/262e2592589baec7bb83a0d46aae6542d4ccedfb)。
- 固定模型库：[六份权重和日志](https://huggingface.co/AaNnWwTt/PV-Ground/tree/cf4a8b1eed045f1309e6a691ea948d1d6c54448e)。
- 对应日志：[ScanRefer](https://huggingface.co/AaNnWwTt/PV-Ground/blob/cf4a8b1eed045f1309e6a691ea948d1d6c54448e/PV-Ground_ScanRefer.txt)、[Nr3D](https://huggingface.co/AaNnWwTt/PV-Ground/blob/cf4a8b1eed045f1309e6a691ea948d1d6c54448e/PV-Ground_NR3D.txt)、[Sr3D](https://huggingface.co/AaNnWwTt/PV-Ground/blob/cf4a8b1eed045f1309e6a691ea948d1d6c54448e/PV-Ground_SR3D.txt)。

只选择ScanRefer two-stage权重进入接入检查。其官方大小为830036622字节，SHA256为`6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3`。未下载其余五份，避免无用磁盘占用。

## 实际代码接口

已读取原始`pv_ground.py`、`pv_backbone.py`、`pv_utils.py`、`prepare_data.py`、`train_dist_mod.py`和实际加载的`wandb_config.yaml`，并保存固定版本。此配置不是仅作记录：训练入口确实调用`cfg_from_yaml_file('wandb_config.yaml', model_cfg)`。

上游路径为`MeanVFE → VoxelBackBone8x → HeightCompression → VoxelSetAbstraction`，输出1024个关键点、288维特征，再进行多模态编码和六层Decoder。配置为2cm体素、每体素最多5点、最多50000体素，坐标范围`[-8,-8,-0.2,8,8,3.8]`。VSA读取BEV、四级稀疏特征和原始点；这与本项目此前REC冻结的Mask局部分支不同，但差异尚未成为本项目精度证据。

输入实际需要`voxels`、`points`、`voxel_coords`、`voxel_num_points`、`batch_size`，并保留文本、对象输入和50000点superpoint映射。`points`是批次号加XYZ/RGB；稀疏坐标代码明确使用`[batch,z,y,x]`，不能按训练入口某处的`bxyz`注释重排。Mask路径仍从关键点分组到superpoint，不能声称完整体素组件自动消除了旧邻域限制。

作者ScanRefer two-stage脚本启用`--butd`；single脚本是不同对象输入协议。实际点坐标平移、RGB归一化、体素范围截断、采样行与superpoint对应尚未通过真实样本前向验证。不能仅因最后维度也是288就直接将新特征插入旧V99。

## 运行环境与后续边界

2026-09-08 06:00 CST只读检查：服务器为A100 40GB，驱动550.90.07；当前`bdetr`是Python3.7.11、Torch1.10.2+cu111、Transformers4.17.0，系统nvcc11.6。默认Python路径没有spconv、cumm、OpenPCDet或torch_scatter。上游README推荐Python3.12、Transformers4.40.0及spconv-cu124，并需要OpenPCDet扩展。这是实际依赖差异；本次未向保护环境安装或升级包。

随后06:02 CST实际导入了已保留的独立环境`/root/autodl-tmp/mcln_sparse_runtime_20260906_v3/venv/bin/python`：Torch1.10.2、spconv2.3.6和cumm0.4.11均可导入。该环境中OpenPCDet、torch_scatter和skimage仍缺失；不能把基础环境未安装spconv误写成服务器没有可用稀疏算子。当前仅完成导入，未重新执行GPU算子测试。

下一步检查缺失的OpenPCDet局部聚合算子、权重与源码形状，然后在隔离工作区做输入与前向检查。仅在这些检查成立之后，才定义ScanRefer短周期对照。此处没有启动取消的长期baseline训练，也没有授权将未验证的上游日志作为本项目成绩。

所有上游代码和版权头保持原样；模型卡标MIT，复用文件含继承来源的许可证说明，最终使用按各文件实际许可证核验并保留引用。直接移植属于预训练基础或比较方法，不能改名声称本项目原创。

## 权重传输与CPU读取

服务器首次HTTPS获取在写入权重前失败，错误为`Cannot assign requested address`；随后的IPv4连接也在25秒超时。原`run.log/controller.exit=1`完整保留，这不是训练失败。之后改由本机从固定官方URL下载，140.76秒完成，并验证830036622字节和官方SHA。

06:05:01 CST，SFTP传输完成，服务器再次计算SHA一致；传输耗时733.01秒。06:05:06 CST，原Torch1.10.2 CPU读取完整结束，独立`cpu_inventory.exit=0`。`model`中1234个张量全部有限，总元素152640527（包含状态buffer，并非可训练参数计数）。其中视觉主干204个张量、3492130个元素；编码器、六层Decoder、预测头和语言模型均在权重中。负载另有config、epoch、optimizer、scheduler、save_path。

CPU清单SHA为`2ed19ee241fe3e1b3063e53c787d1aa8c8400d48e5868cc41b8f0154a0a4a250`；本地重新核对文件SHA、张量数和元素总和通过。只读取state，不代表已实例化网络、完成strict load或执行前向。GPU前向、优化步数、正式评估行数均为0。

服务器保留唯一已下载的官方权重：`/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1/PV-Ground_ScanRefer.pth`。06:06:31 CST，在远端回执和本地SHA都匹配后，删除本机同名中转文件，释放830036622字节；本地可用空间恢复到6488969216字节。只删除这一份副本，保护E71、V99/Nr/Sr及当前有用权重未改动。权重不进入Git或Desktop交接目录。

证据目录：`refine-logs/pvground_pretrained_resources_20260908_v1/`及`refine-logs/pvground_scan_checkpoint_inspection_20260908_v1/`。前者的原始`receipt.json`记录05:39时尚未下载权重的事实，后续下载/传输/读取使用独立回执，不覆盖旧状态。
