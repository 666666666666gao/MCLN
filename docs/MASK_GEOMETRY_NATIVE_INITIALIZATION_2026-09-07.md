# Mask几何训练终点的 Nr3D / Sr3D 初始化接口

本页对应当前`native_gt_mask_geometry`固定候选。它是后续训练接续的格式准备，不代表ScanRefer已经晋级，也不代表Nr3D/Sr3D已获得新结果。

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
