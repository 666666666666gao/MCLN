# C1 原生输入预检：固定16行、12次forward、零更新

B已完成并未通过固定质量筛选。C1从原受保护Nr3D parent启动，使用原生superpoint
记忆，独立验证 `MaskQueryMemoryReadout`；不加载B的16/19张量训练artifact。

输入沿用M3的16条fit、16个不同训练场景，原始点hash、36个数据文件、612个冻结源码
文件和checkpoint写入清单。只使用原有点云、候选框与文本；GT仅用于原生loss计算。

每批4条，共4批。每批依次：

1. 原生无新增模块forward，记录Query/seed/框/分数、raw Query Mask、Text Mask和alpha。
2. 附加零输出C1，forward及原生Mask loss梯度；所有快照必须一致，输出矩阵有有限
   非零梯度，其他新增参数在零输出矩阵下为零梯度。
3. 将输出矩阵固定为.001倍矩形单位阵，forward及梯度；raw Query Mask必须改变，
   REC张量、原始Decoder Query、Text Mask和alpha必须不变。

扰动后所有梯度必须有限；每个新增张量在这16行中至少观察到一次非零梯度。
该检查证明梯度路径可用，不要求每个样本都激活全部参数，不构成质量筛选。
不调用optimizer，不保存checkpoint，不读取模块留出或正式验证行。

每批移除attachment并将输出矩阵恢复零；最后核验parent参数/buffer、全部新增参数、
源码、数据及原始点hash。预期为12次原生forward、0更新、0checkpoint写入。
计时中的原生no-grad与新增分支autograd模式不同，不能直接解释为部署开销比。

入口 `scripts/run_nr3d_mask_query_memory_preflight.py`。已有4项CPU合成测试通过；
本预检运行前单独确认原Python3.7编译和清单hash，使用现有GPU锁。
通过后才固定C1学习对照；失败先定位实际错误，不运行训练或调整质量门槛。
此预检不能完成REC目标，三数据集正式目标继续保留。
