# Nr3D稀疏局部记忆：完整起点一致，配对训练已开始

07:53:40 CST实际观察PID28416正常运行，GPU10855MiB，两臂各完成192/6687步。
已有128/192步日志，loss及梯度范数均有限；稀疏输出投影范数从0开始，至192步为
0.10415036976337433。这证明更新已实际执行，不作为局部几何有效或泛化收益证据。

完整6172条/98扫描起点复核已经通过，并由本地独立逐行读取再次核验：

- 行ID、扫描ID、实际采样点hash、Box/score hash及原生输出，与受保护归档全部一致。
- 同行native与sparse的REC/Mask选择、IoU、合法Box oracle及条件Mask记录全部一致。
- 起点评估阶段0更新；这是被多轮方案使用、且主干见过场景的模块留出，不是正式验证。
- 当前baseline_rows SHA：5f02c1bb8fd20c94c103cda35c44fcee1709b12aed97bbf1ee6ecc6408c8f2b1。
- 归档参考SHA：884d72879a7a9485309ec9dadc588357a8ea69dbe847854e4af6934a4eef84ef。

本实验保留PointNet++/Decoder/候选及REC决策；新17张量/267936参数由相同50000点的
连续坐标偏移与RGB、2cm稀疏邻域及逆映射构造细特征，直接按实际superpoint成员池化，
经零初始化投影加入Mask特征。只额外优化原生x_query/x_mask/rel_encoder，与原生同预算
对照比较。它是独立Mask证据实验，尚未验证共享Box/Mask任务查询或三数据集统一增益。

结构、输入、loss、LR、每臂6687步及验收条件沿用冻结计划。相对原生终点和受保护起点，
均须mIoU至少+0.2个百分点，两个Mask命中阈值不下降。原生REC保持原规则；不挑中间最好。

最近64个配对步骤约1.47644秒/步，预计余下训练加终态评估约11071.61秒，终态初估
10:58:12 CST。观察器已安排约10:53:12再查询，随后240秒间隔；这是按短段实测速率的
外推，可能随场景复杂度变化。当前没有终态权重、完整学习质量或新的正式指标。

终态后先独立读取16/33张量与AdamW实物，再重算逐行Mask净修复/破坏、固定条件Mask
及scene聚类区间。质量失败封存，通过才按计划注册进一步机制与正式验证。
Nr3D/Sr3D/ScanRefer受保护结果未更新，原整体目标仍active。

代码与检查入口见scripts/run_nr3d_sparse_point_pair.py、summarize_nr3d_sparse_point_pair.py、
verify_nr3d_sparse_point_artifacts.py。原始字节源码归档和完整回执位于
refine-logs/sparse_point_pair_20260906_v3/；V1/V2失败及数值检查修订证据保留。
