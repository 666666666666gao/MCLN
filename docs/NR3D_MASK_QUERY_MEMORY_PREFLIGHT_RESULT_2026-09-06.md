# C1 真实输入预检通过；尚无质量结果

固定16条fit/16个不同训练场景，4批B4、12次真实CUDA forward，0 optimizer更新、
0 checkpoint写入、0模块留出或正式验证行。controller.exit=0。

`MaskQueryMemoryReadout` 的74,880个参数、8个张量只更新候选Mask Query的局部记忆读取。
使用原受保护Nr3D parent与原生superpoint特征，未加载B的训练artifact。

- 零输出初始化下，原始Decoder Query、seed/候选/框/分数、raw Query Mask、Text Mask
  和alpha逐bit一致。
- 固定输出矩阵扰动后，4批raw Query Mask均改变；REC、原始Decoder Query、Text Mask
  和alpha仍逐bit不变。
- 原生Mask loss对零输出矩阵有有限非零梯度；其他新增参数初始零梯度符合计算图。
  扰动后8个参数张量均在这16行中观察到有限非零梯度。
- 全部parent参数/buffer、源文件、数据、同一50,000点输入hash均通过核验；新增参数
  和attachment最终恢复，没有训练或永久修改parent。

峰值CUDA allocation：1695122432字节；计时29.329秒，不含数据集初始化。
原生forward使用no-grad，新增分支使用autograd，不能把计时比当作推理开销。

这是接入完整性和梯度检查，不证明Mask质量改善、任务冲突减少或REC提升。
真实质量需先固定学习对照、更新预算和质量门槛，再运行独立配对实验。
现有Box/身份路径保留，因此本项本身不能完成REC目标。

manifest SHA：`26542f87e372a22e2710418f767782fbb1a10bc7c0aa557f4c614dea9a65febf`

receipt SHA：`02e27807057b24054a5e6b5ba3b0dcdfa9e052a87883852bbe79afc4b1ba9214`

逐批扰动、梯度与计时记录见 `refine-logs/mask_query_memory_preflight_20260906_v1/`。
入口 `scripts/run_nr3d_mask_query_memory_preflight.py`；固定计划
`docs/NR3D_MASK_QUERY_MEMORY_PREFLIGHT_PLAN_2026-09-06.md`。
