# 固定D终点的运行统计干预

目的：检查训练后BatchNorm运行统计是否是当前D输出变化的实质因素。CPU实测84组/252个running_mean、running_var、num_batches_tracked张量全部改变；这本身不是错误或有害证据。前轮梯度诊断未检查这类不可训练buffer。

固定D3723步终点、官方Scan epoch81父、正确mesh/detection/VSA接口。仍使用前两轮预先固定的128个fit物理场景各一表达；seed2027、batch8、无增强、eval模式。128场景是预训练见过的训练数据，不是正式集。

每批仅三种前向：D正常、D正常重复、D参数不变但全部84组BN运行统计恢复父值。保持BN affine weight/bias及其他所有参数为D；不恢复一部分来源、不扫描组合、不混合比例。前向前恢复同一Python/NumPy/CPU/CUDA RNG状态，各臂重新构造输入；下一批继续正常臂的RNG状态。模型始终eval，不在测试数据上重估BN。共48次no_grad forward，0 backward、0 optimizer、0训练checkpoint、0正式行。

先按实际模块类型核对BN buffer集合与CPU census一致，逐项载入预先定义状态；每批后恢复D，全部结束校验全state_dict与D一致。.grad必须全部None。原输入SHA/文本/行序与128行记录一致。保留每个候选的原始center/size、bbs score、root GT、所选编号及IoU。IoU沿原生正尺寸clamp和严格>0.25/>0.5规则；raw256 oracle仅诊断，不称合法召回。未评估Mask，不得据此通过Scan完整要求。

比较正常重复的数值和选择波动，报告父BN相对正常的两阈值修复/破坏、全部候选变化、均值与所选变化。主判据仅为是否存在超出同状态重复波动的因果敏感性；任何训练收益仍需另外的固定训练对照。若两个阈值均无净正变化，停止恢复父BN这一直接路线；不能由此证明BN完全无影响。若有双阈值改善，也只允许设计保留/适配运行统计的后续训练对照，不能把训练集干预指标当模型增益或直接进入9508。

预计5分钟以内、产物不超过32MiB、无新权重。沿用已核验环境spec，不重建环境。先确认GPU空闲与旧任务退出，使用原GPU锁和独立screen/controller；第一次约300秒取证，失败保留日志、不因观测超时重启。
