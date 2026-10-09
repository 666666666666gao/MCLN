Nr3D / Sr3D 的作者权重和数据分区已在旧准备记录中定位，本次仅本地读取缓存，没有启动新训练或查询服务器。

Nr3D正式接口7899条，Sr3D17726条；两者旧接口为原生butd_cls实例框和预测类别，与ScanRefer的检测对象输入须分别报告。Sr3D的文本权重与ScanRefer/Nr3D不同，Nr/Sr原生语言损失倍率1，ScanRefer为0.5；联合检测行通过sample_dataset保护，不能只改language_dataset名字。

后续从各自作者预训练核心开始，不用ScanRefer G完整权重覆盖Nr/Sr核心。完整最终模块固定后复用已准备接口，按同起点、同预算、seed2027比较。当前缓存的两步spec不等于新模型训练计划或结果；远端当前可用性还未在本次验证。先完成ScanRefer目标和三项有效贡献要求。
