# 对象外观原生配对：完整性PASS，固定质量FAIL

两臂均完成1024更新，终态6172行完整，controller.exit=0。98个模块留出场景未参与新增
参数训练，但主干以前见过；这不是正式验证，也不是整个系统未见场景泛化。

| 版本 | REC hits@.25 | REC hits@.50 | Mask hits@.25 | Mask hits@.50 | Mask mIoU |
|---|---:|---:|---:|---:|---:|
| 受保护起点 | 6005 | 5306 | 5767 | 5057 | 68.881519715% |
| native终点 | 6002 | 5329 | 5762 | 5061 | 68.890362068% |
| appearance终点 | 6004 | 5337 | 5758 | 5060 | 68.880898540% |

| appearance相对 | REC25修复/破坏/净变化 | REC50修复/破坏/净变化 | Mask25修复/破坏/净变化 | Mask50修复/破坏/净变化 | mIoU差，百分点 |
|---|---:|---:|---:|---:|---:|
| native终点 | 3/1/+2 | 19/11/+8 | 1/5/-4 | 6/7/-1 | -0.009463527 |
| 受保护起点 | 4/5/-1 | 50/19/+31 | 2/11/-9 | 18/15/+3 | -0.000621175 |

训练前固定门槛：appearance相对native终点和受保护起点均需REC@.25净增至少10，
且REC@.50、Mask@.25、Mask@.50命中与Mask mIoU均不下降。严格IoU>阈值。
两参考均需通过，结论：**FAIL**。封存本设定，不续训、不调学习率或门槛，不启动本设定正式验证；细尺度空间记忆的运行条件单独准备。

| 对照 | REC25差95% scene CI，百分点 | Mask mIoU差95% scene CI，百分点 |
|---|---:|---:|
| native终点 | [-0.034118, +0.122508] | [-0.041919, +0.027791] |
| 受保护起点 | [-0.098576, +0.068377] | [-0.078378, +0.080694] |

以上CI为2000次配对场景聚类bootstrap，不是固定行数下的命中数区间。

| 阶段 | 合法Full256 oracle hits@.25 / @.50 | 无合法REC候选行 |
|---|---:|---:|
| 起点 | 6147 / 5931 | 0 |
| native | 6147 / 5952 | 0 |
| appearance | 6148 / 5966 | 0 |

本实验更新最后Decoder对象注意力及其LayerNorm，允许末层Query、候选框、语义分数与
Query Mask共同变化，合法性按各自实际框计算；不是固定候选重排。上述oracle仅为GT诊断。
appearance相对native终点，REC Query索引相同行6150，其中框IoU变化6042行；索引相同不等于已核验真实实例身份。
appearance相对起点，REC Query索引相同行6123，其中框IoU变化6018行；索引相同不等于已核验真实实例身份。

两臂同2048 fit/262场景、两轮B4、原完整loss/Hungarian，fresh AdamW lr1e-5、wd.0005、
clip.1。native6张量/333504参数，appearance另加5张量/41472参数，总374976，多12.44%；
不是等容量控制。原模型eval，其余权重和buffer冻结，末层Box/Mask/语义writer不直接训练。
所有前五层Query、采样、Text Mask及alpha在两臂/起终逐行一致，保留原REC/Mask选择协议。
输入同50000点/原生RGB与butd_cls框，不扩框、不用GT实例mask清洗，无其他实验artifact混入。

完整6172起点两臂与受保护C1起点逐行相同。4096 fit顺序恰为两轮2048行，1024个实际
点batch/前五层Query/Text Mask/alpha摘要齐全。两份实物权重及AdamW的有限性、形状、
1024步和参数改变已核验；6个共享权重均独立核对相对父权重有真实变化，新增输出矩阵非零。
冻结状态、612源码/724输入清单/parent前后未改变。当前实验始终使用旧模块4b3dad...，
Sr3D零轴修复2a5600...未注入已冻结运行目录，避免混合代码版本。

本结果只检验这个局部更新预算和接入位置。不能据此宣称所有局部外观、体素主干或
共享身份/任务查询设计已被完整验证或被整体否定；也没有新的跨基准泛化结论。

manifest SHA：`5f467d81155c03fa939f898a8d15a5263a98b407c31c342a72223b931b5a844f`。
terminal receipt SHA：`2c1311f5c1f6ee8ef60bf5570c1177cdeef0d7a2c8fd43413be321572159a86a`。
native artifact：4007215 bytes，SHA `07043e3a61f112145a0ce44eb19687351ec6f1a5ce981e23b8123e2aa50c576b`。
appearance artifact：4509058 bytes，SHA `034d09e7da14c19ab550cedb12592e55b63efd6626df48b5460667eac9900a7e`。
本次记录总elapsed 4406.068秒、峰值GPU allocated 5765650944 bytes；这不是等工作量推理开销对比。

独立复算scripts/summarize_nr3d_object_appearance_pair.py；原始逐行/实物核验在
refine-logs/object_appearance_pair_20260906_v2/，远端/root/autodl-tmp/mcln_object_appearance_pair_20260906_v2。
0正式验证行，保护模型未升级；Nr3D4475/3759、Sr3D12139/10335与ScanRefer5572/4797保持原记录。
三数据集完整目标仍未完成。
