# 点细节到 superpoint 的真实 CUDA 预检通过

L1 退出后于 2026-09-06 00:09:34 CST 启动，00:13:41 已确认完成并释放 GPU。
16 条 fit 表达、16 个场景、12 次原生 forward、0 次 optimizer 更新、0 checkpoint
写入、0 holdout 或正式验证行。它证明接入及梯度路径可用，尚无质量收益结论。

新增模块 54,144 参数，读取实际 SA1 `(B,128,2048)` 特征、相同 50,000 点坐标/RGB，
插值至点后按真实 superpoint 成员池化，残差加到原生 Mask 局部特征。

通过内容：

- 原生 CUDA 三邻居插值在真实重合采样点上有限且正确。
- 零残差严格复现 Query、seed、框、分数、原始 Query/Text Mask 和融合 alpha。
- 固定 .001 输出扰动使四批原始 Query Mask logits 最大变化约 .0258–.0348；
  同时 Query、seed、框和分数保持完全一致。
- 零输出时输出矩阵梯度范数为 1.64–6.56，点编码器首次梯度为零符合链式求导。
  固定扰动后，全部三个新参数张量的梯度均有限且非零。
- parent 参数/buffer、输入点 hash、612 个源码文件、36 个数据文件保持一致；
  新模块恢复初始状态，未写训练权重。

不含数据集初始化耗时 31.4054 秒，峰值 CUDA allocation 1,833,081,344 字节。
原生 no-grad forward 与带新增分支的 autograd forward 在不同梯度模式下测量，
不能直接用二者比值宣称部署开销。

manifest SHA：`74103cf7c92dfc62716ee7830386570c4c1bfed86f2cf000067be114a53aad2a`。
receipt SHA：`76cbec3ba81e7f62fa6d79918959131a095068997ded1b22eeec365286a0761d`。
完整证据 `refine-logs/point_detail_preflight_20260905_v1/receipt.json`。

下一步按照 `NR3D_POINT_DETAIL_PAIR_PLAN_2026-09-06.md` 执行独立配对短训，
先判断真实局部输入增量是否有可复核收益，再决定后续体素或任务查询设计。
