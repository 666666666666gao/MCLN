用户要求将当前最好PV框架的可迁移机制加载MCLN baseline试验，原PV60.0%/52.0%、三项有效贡献、Nr/Sr对应初始化验证和完整模型训练目标保留。PV最好仍5677/4920，距5705/4945差28/25；不替换主线最佳。

完整9508条、141场景、native last/bbs三路配对于2026-10-12 10:14:55实际完成，11:13:49采集终态；耗时4556.04秒，0更新/0新权重。native5515/4409=58.003786%/46.371477%；Mask参考5459/4767=57.414809%/50.136727%；加载PV已训练A/B的full transfer5473/4777=57.562053%/50.241902%。native到Mask参考宽松138修复/194破坏净−56，严格694/336净+358；native到A/B宽松138/180净−42，严格695/327净+368；A/B相对参考宽松18/4净+14，严格23/13净+10。这是直接跨baseline迁移，尚非MCLN正常训练精度，也不单独证明三项模块。

所有9508所选Query相同，原生语义分数和完整1159模型状态通过实际逐batch/终态断言保持，使用同一输入和恢复的RNG。Full256严格覆盖7886/8159/8165，仅GT几何诊断不可部署。PV六源G和G标签策略不迁入；原MCLN已有REC/RES、Query/Text Mask和adaptive融合，不能包装为本项目创新。新A校正候选自身支撑，B读极值来源和回归先验，均在forward实际输出路径中，无第二排名或候选截断。

15,766,406字节配对JSONL SHA9562bb3a4bea8fef04fc2bd7adb9aa3110224c1193bf2b4f6bc81436582ea417已保存本地，未把巨大数组加入Git。9508所选框IoU由真实保存GT/预测坐标CPU独立重算最大差0，命中和修复/破坏一致。fresh实际完整性审查WARN/0阻断，同family/provisional，实际模型身份UNATTESTED。两项记录限制保留：保存预测尺寸已经原生clamp到1e-6，旧raw非正尺寸0字段只能解释为保存后的统计，截断前数量不可用；native第2982条三轴均达到下限。CPU admission引用的旧源码审查SHA对应原始报告未在当前包找到；sanity/full admission与留存最终报告一致，不重建旧报告，不据此改历史成绩。

CPU附加诊断显示native到Mask参考的194条宽松破坏中，79条融合MaskIoU仍>.5，预测/GT体积比中位数约4.082；336条严格破坏中203条Mask仍>.5，体积比约2.012。支持继续研究范围过大/分割到框失配，但不证明具体极值点是错误来源，不把GT重叠当真实实例身份。这些诊断只读保存输出，不重跑NN。

MCLN原生两步真实训练检查于13:46:27启动，13:56:17结束0，实际16条两次更新；原生训练数据48655，joint_det/detect_intermediate/augment_det/butd均True。A/B内部及输出、原可训练核心均有非零梯度，变化27265/29793参数元素；RoBERTa按原生冻结。实际保存793579034B完整1159状态、731个Adam状态以及scheduler/全局RNG并同进程文件恢复exact；冷启动未验证、正式精度为空。峰值allocator17.069GB只代表这两步。工程权重不选模、不承接正式初始化，可在关闭后清理。

工程控制器R1未应用已有env变量，R2最小复用原env/PointNet路径/CUDA rank设置并SOURCE WARN0阻断；保留R1/R2版本。首个本地观察timer消失，工具句柄不可用；到时只读观察又暴露缺stage字段，已保留失败并最小修正，13:59:50采集真实终态，未重启NN。数据工厂构造未使用val对象，但两步只遍历train，不宣称没有访问任何val文件。

后续正常训练入口复用原生完整train_one_epoch/evaluate_one_epoch，最多8轮，每轮完整9508 E0/epoch验证，严格相邻两轮下降停、持平/回升重置；原核心+A/B共同更新，core/PointNet1e-6、A/B1e-5、B8、seed2027。loader generator和train sampler显式同seed2027，避免原native固定0与seed声明不一致。最佳选Acc@.5首要、同strict才比较Acc@.25，不拼指标，保留best和必要latest/optimizer/RNG。正式启动状态以本快照CLOSEOUT.json/LAUNCH为准，SOURCE通过不等于启动或有新准确率。工程状态不承接，原E71+A/B重新加载。单独迁移训练不能分离结构与续训因果；若有新增能力再做对应同起点同预算控制。

正常训练与三个有效贡献/Nr/Sr仍未完成。目标ACTIVE_UNMET；Mask参考硬极值仍detach，不写成Box损失穿过完整Mask路径端到端优化。

首次正式启动因4GiB空间门槛被拒绝，未建立训练目录、未删工程权重、未启动NN，原失败保留。重核SHA后仅清理两份本地完整归档的旧native/control best远端重复副本，共1230047504B；当前最高模型及official/E71/G/A/B不动。CPU-only打开已完成工程状态发现optimizer755参数已有731状态，24项尚未创建，不能直接用现文件大小当未来上界。实际model tensor598498856B、冻结RoBERTa498586640B、非RoBERTa99912216B，所有group amsgrad=False。按全部非RoBERTa张量（包括buffer，保守）预留两份Adam moment，再为各权重留64MiB metadata；根据latest/best加单个原子替换峰值max(2L+B,L+2B)及256MiB日志余量，需要2664907480B，而非任意降低阈值。新预算需经修订源码门禁；实际启动及工程清理以本快照LAUNCH/关闭证据为准。容量计量0NN/GPU/新权重，不算精度实验。
