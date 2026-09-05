# Sr3D输入协议、物理空间分组与对象裁剪结果

本次是训练输入与协议检查，0原生模型forward、0GPU、0优化器更新、0正式模型评估行。
不恢复缺失Sr3D权重，不将Nr3D权重改名为Sr3D受保护起点，不启动Sr3D学习。

## 输入协议与未来模块留出

按当前Joint3DDataset有效过滤口径核对CSV、split metadata、预测类别、superpoint和
Group-Free框文件。Nr3D训练保留列入扫描的表达，评估应用correct_guess；Sr3D训练和
评估均应用mentions_target_class。代码中的val对应metadata的test。

| 数据 | 有效表达 | 扫描 | 物理空间 |
|---|---:|---:|---:|
| Nr3D train | 32919 | 511 | 511 |
| Nr3D test | 7899 | 130 | 130 |
| Sr3D train | 65846 | 1018 | 490 |
| Sr3D test | 17726 | 255 | 123 |

物理空间采用scene前缀：ScanNet官方定义scene<spaceId>_<scanId>，同一spaceId可有多次
扫描。见[官方数据说明](https://github.com/ScanNet/ScanNet#data-organization)。
两基准各自的train/test在扫描和物理空间上均无交集；Nr train与Sr test、Sr train与
Nr test也无交集。同split共享扫描为train468、test116；Sr新增扫描分别550、139。
Sr训练490个物理空间中483个也在Nr训练；这不是对完整checkpoint训练来源的证明。
全部Sr有效train/test扫描均找到既定预测类别、superpoint和Group-Free框输入。

额外诊断仅模拟未来新增模块划分，沿用Nr实验salt，不创建实际Sr训练split：

| 分组键 | fit表达/扫描/空间 | holdout表达/扫描/空间 | 两侧共有物理空间 |
|---|---:|---:|---:|
| 完整scan_id哈希 | 52102 / 815 / 443 | 13744 / 203 / 177 | 130 |
| 物理space_id哈希 | 54492 / 836 / 404 | 11354 / 182 / 86 | 0 |

因此未来Sr模块留出及场景聚类统计应按物理空间分组。同一salt下直接取CSV前2048个fit
表达，在物理空间分组中仅覆盖50扫描/38空间，不能照抄Nr的2048行并宣称覆盖相同。
这是新方案准备中的真实分组风险；既定官方split未改变，也没有发生一次被证实泄漏的
Sr训练。Nr当前扫描对应不同物理空间，其既有实验划分不受该问题影响。

04:15再次核对，文档记录的Sr受保护E26路径仍不存在。Nr已知权重为evaluation_only，
无优化器，完整历史训练来源尚未恢复。现有协议审计不能替代Sr父权重或来源证明。
协议receipt SHA82ac488ed6164d2fe069e74b7b826ee48b1b3e0a0cb2d10787e78ad5154959ef；
证据目录refine-logs/referit3d_protocol_20260906。

## 完整训练裁剪审计

每个Sr训练扫描取过滤CSV首条表达，经原生数据集读取50000个XYZ/原生RGB点、既定
butd_cls对象框和有效槽，augment=False。实际覆盖1018扫描、34865有效对象槽。

- 468共同扫描逐字节核对点、框、有效槽和完全相同旧模块后，复用已完成Nr裁剪计数。
- 550新增扫描实际运行Torch box_crop_mask，与NumPy显式AABB逐点一致。
- 0空裁剪；2个对象含非正尺寸，实际均为三个尺寸全0，各裁到1点。
- 重叠框允许重复计数，累计裁剪成员68702391；不能称为独立场景点数。
- 数据锁定2081文件，其中显式覆盖全部2036个Sr训练SP/GF扫描文件。
- 排除数据集初始化的审计耗时183.812884569秒，结束后源码与数据哈希一致。

| 过滤后train行 | 扫描 | 对象槽 | 三轴尺寸 | 裁剪点数 |
|---:|---|---:|---|---:|
| 18112 | scene0561_01 | 21 | 0 / 0 / 0 | 1 |
| 21602 | scene0673_05 | 89 | 0 / 0 / 0 | 1 |

旧模块要求正半尺寸，故原input_contract_pass=false真实成立并保留。该结果不是完整
Sr性能失败，也不表示单点对象能够恢复缺失几何。Nr完整训练审计没有非正轴，因此
正在运行Nr对象外观配对无需改变已冻结代码。

裁剪审计manifest SHA06c73ea67f2dbe36a93ef4c36e57d18bc158282c028e9d3e8b430a3ceaa29b31；
原始证据refine-logs/sr3d_object_crop_inputs_20260906_v1，远端同名mcln目录。
此v1使用旧模块SHA4b3dadf6a25508a3453e81416e415b4262cd11fd044ab3324ab76b20d0b54f9b，
不能以当前已修正模块替换冻结文件后仍声称是原审计复现。

## 最小修复及验证边界

只将零长度轴的归一化坐标定义为0。显式裁剪已经要求此轴坐标等于中心，因此把该轴
除数置1即可；正长度轴仍除以真实半尺寸，包括很薄的正尺寸。框范围、crop函数、点、
RGB、5参数张量/41472参数不变，无epsilon、扩框或GT实例mask清洗。

本地7测试与服务器原Py3.7.11/Torch1.10.2环境7测试均PASS。新增两项覆盖真实异常框
几何（单元测试RGB明确为合成数据）与2e-6薄正尺寸精确归一化。服务器原环境测试结果
refine-logs/object_point_appearance_cpu_20260906_v3。修复后模块SHA
2a560042da5605f6368d12f6ba95c11333bb0797026ab1c2d32457d51fffa3b0。

真实两扫描特征检查已按SR3D_ZERO_EXTENT_INPUT_PROBE_PLAN_2026-09-06.md完成，04:54:48
观察到controller退出0、新receipt PASS。实际覆盖27+125=152个有效槽：150个正尺寸
对象的MLP实际归一化输入与旧公式精确相等，两个单点对象的位置输入全0且实际RGB未改。
零初始化输出全0，固定矩阵扰动后两个异常对象输出范数分别0.0008818682、0.0008668081；
各自单独产生的损失都使5个参数张量梯度有限且非零。4次外观CPU forward、2次CPU
backward，结束后权重恢复、梯度清空；不含原生MCLN或GPU forward，不含优化器更新。

两扫描实际点/框/槽/裁剪计数与原审计一致；新旧crop函数AST相同。原FAIL receipt、
锁定源码/数据与运行中Nr模块前后哈希一致。耗时27.541743040秒，不含数据集初始化。
新manifest SHA2b6fba33e1904e1cc61be2d9e8a31a6b77803d7559279df5a8c83c2b10e75b3a；
完整证据refine-logs/sr3d_zero_extent_inputs_20260906_v1。

这是已观测异常的输入修复验证，不证明Sr3D REC或Mask改善。完整训练裁剪计数支持
所有34865槽非空及仅两处非正轴；新特征forward只实际执行这两个扫描的152槽，不能
写成全部34865槽都已执行新编码器，也不能改写原正尺寸合同FAIL。
