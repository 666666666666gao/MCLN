# Nr3D / Sr3D 实际数据文件准备检查

只读读取当前已核验native数据源码、CSV和官方场景划分，按源码中的真实过滤条件统计训练／验证表达数，并检查相应场景的超点和GroupFree预测文件是否存在。只检查已知输入路径，不扫描全盘、不加载完整scans pickle、不构造模型、不导入CUDA、不读取当前训练状态、不执行优化器、不改活动源码或数据。

Nr训练不按correct_guess过滤；Nr验证按correct_guess=true过滤。Sr训练／验证均按mentions_target_class=true过滤。验证读取官方test场景清单，运行数据split仍为val。统计是元数据条件下的资格数，不是实际Joint3DDataset/DataLoader长度，也不是REC精度。

沿用当前准备入口的预测GroupFree对象输入（butd=true，butd_cls/butd_gt=false），并同时记录其与作者权重保存的butd_cls场景GT提议协议不同。最终Nr/Sr正式协议需与对应baseline一致，并在选定完整模型后另行真实loader、GPU和冷恢复检查；本轮不会启动Nr/Sr训练，不把作者预训练可加载等同于作者原协议已复现。

输出仅小型JSON和文本，缺失文件如实记录，不增加fallback、不生成文件修补。完整权重链及单卡ScanRefer正常训练不变，唯一下一次训练观察仍为12:01:45.656476。
