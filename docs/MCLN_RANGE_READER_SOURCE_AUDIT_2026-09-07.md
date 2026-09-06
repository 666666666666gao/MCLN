# 范围读取的来源与当前实验边界

2026-09-07。仅核对公开原始实现与当前代码，不改变正在运行的 ScanRefer 配对。来源文件的提交、SHA256 和定位行记录在 `refine-logs/range_source_audit_20260907_v1/source_receipt.json`；未把第三方完整源码复制进仓库。

## 与已有实现的关系

**PV-RCNN 已有有序 RoI 网格。** OpenPCDet 的 `roi_grid_pool` 在候选范围内建立网格、从关键点聚合邻域特征，再将空间网格展平用于分类与回归。默认 KITTI 配置是 6×6×6 网格、两尺度半径与各16个邻居。这与本轮64槽位预算不同，不能将本轮简单称为 PV-RCNN 复现或公平性能对比。[固定源码](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/models/roi_heads/pvrcnn_head.py#L56)，[配置](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/cfgs/kitti_models/pv_rcnn.yaml)

**空邻域识别也是已有机制。** OpenPCDet 的 CUDA ball query 对无邻居返回标记，Python 分组层在取索引后将该区域的相对坐标与特征置零；非空但未满的邻域会重复首个命中点，搜索按输入顺序而非距离排序。本轮则按距离选择不同的观测点，对不足的槽位使用有效性掩码。[CUDA](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/ops/pointnet2/pointnet2_stack/src/ball_query_gpu.cu#L43)，[分组层](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_utils.py#L30)

本轮另外在输出投影后屏蔽整个空候选；上游是分组输入置零后再经过 MLP、BN 和池化。两者掩码位置不同，不足以宣称上游存在已测出的错误，也没有性能证据说明本轮做法更优。[上游聚合层](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_modules.py#L70)

**Box-DETR 的代理点是动态且按注意力头生成的。** 官方实现用 Decoder 表征预测每头二维偏移，以框宽高缩放后形成代理位置，再构造该头的空间查询。这不是本轮八个固定三维支撑位置，也不等于硬取64个点。[固定源码](https://github.com/tiny-smart/box-detr/blob/053bd1f65159e431db7a0ab17a12413db1c7b8ae/models/box_detr/transformer.py#L157)，[论文](https://arxiv.org/abs/2307.08353)

## 本轮实际验证什么

`CandidateRangeVisual` 两臂共享全部145008个参数、同一有效窗口和分区注意力。center 选窗口中最近64点，随后按八象限分组；extent 在每象限的固定支撑位置附近最多选8个不同观测点。64是槽位上限，实际有效点数可以不同。因此，本轮检验的是固定预算下的空间取点分配，不是“有分区”对“无分区”，也不是相同有效点数的对照。

由两份代码公式直接推得：本轮支撑位置为 `c ± 0.5 * max(size/2, 0.05m)`。只有各轴5厘米半尺寸下限未生效时，它才等于常规2×2×2 RoI网格中心 `c ± size/4`。支撑位置不等于六个框面；本轮没有实现六面边界token或边界回归分布。

区域注意力读取现有 Decoder Query，语言影响经已有多模态 Query 间接传入；读取器没有新增完整文本 token 接口、明确的关系槽或区域语义监督。不能把本轮称为已实现的“身份—范围解耦”或完整的表达条件边界学习。

当前可报告的是工程预检和实际覆盖统计。新增读取的 REC 贡献仍等待完整训练、终态审计及固定正式评估；空间覆盖扩大、空区域置零或梯度连通均不是精度证据。即使本轮通过，也需要进一步对照才可判断超出普通 RoI 网格的贡献；本次查阅不是完整新颖性检索。没有为这项说明新增训练、修改运行源码或改变晋级规则。
