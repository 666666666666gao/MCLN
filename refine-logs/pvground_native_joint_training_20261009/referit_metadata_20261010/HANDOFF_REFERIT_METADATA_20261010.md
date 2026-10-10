### 20.376.136 Nr3D／Sr3D真实元数据与预测提议文件检查；不是模型训练结果

本轮只读检查于服务器记录的2026-10-10 08:35:31附近完成，传输退出码0。按已核验native数据源码的实际CSV过滤规则和官方场景清单统计，不构造Joint3DDataset，不读取完整scans pickle，不导入CUDA或模型，不执行优化器，不读取当前ScanRefer训练状态，也没有写远端文件。

| 数据集与split | 过滤后元数据表达数 | 实际涉及场景 | 缺少所需超点文件 | 缺少所需GroupFree提议文件 |
|---|---:|---:|---:|---:|
| Nr3D train | 32919 | 511 | 0 | 0 |
| Nr3D val | 7899 | 130 | 0 | 0 |
| Sr3D train | 65846 | 1018 | 0 | 0 |
| Sr3D val | 17726 | 255 | 0 | 0 |

Nr3D训练不按correct_guess过滤，验证保留correct_guess=true；Sr3D两种split均保留mentions_target_class=true。val读取官方test场景清单，超点与预测提议仍使用运行split的val目录。两数据集的train/test场景清单分别没有交集。native CSV路径解析函数直接从已核验源码提取使用，没有新增路径fallback、数据过滤修补或下载。

这些是元数据资格数，不能替代实际Joint3DDataset/DataLoader长度、scans成员与实例对应检查，也不证明NPY/PTH内容有效。若最终沿用当前准备入口的benchmark-only、batch8、drop_last=true配方，按这些元数据数推算Nr3D每轮4114次更新、Sr3D8230次更新；这不是实际训练预算确认。若采用作者joint_det混合ScanNet配方，暴露量和更新数将改变，必须依据实际构造结果重新记录。

本轮检查的是预测GroupFree文件。作者Nr／Sr权重保存的butd_cls对象输入会用场景GT提议加预测类别，与目前准备入口的butd=true预测提议协议不同；权重能严格加载不代表两种输入协议等价。最终baseline与完整模型需使用同一清楚标注的协议，不能将协议变化当成模块增益。允许使用各自作者预训练核心，不用ScanRefer的完整G权重覆盖Nr／Sr核心。

源代码复核PASS，无阻断；真实输出复核已关闭，总结为WARN、限定元数据检查PASS、0阻断。唯一提醒是服务器观察时间比本地传输结束时间晚5.239782秒，两端时钟同步未获确认；原始时间各自保留，不据此推断跨主机精确先后关系。初始换行符疑问经AST/literal检查被证伪，没有修改代码或失败执行。审查身份只按实际可证明范围记录为UNATTESTED、同系列、临时结论。没有REC精度、GPU准入或Nr／Sr正式训练。

活动ScanRefer正常训练源码、3轮预算和best/latest不变，下一次唯一观察仍为12:01:45.656476北京时间。完整跨基准训练继续等待选定同一最终完整架构、真实loader/batch/criterion/GPU检查和冷恢复；三个有效贡献和总目标尚未完成。
