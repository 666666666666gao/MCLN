## 20.376.126 PV-Ground正常训练接入：配置隔离修复与保留模块初始化

本节记录原生训练接入的工程状态，不新增精度结果。用户要求改进尽量进入正常训练网络及loss，避免推理后处理；继续采用同一9508条ScanRefer、单seed2027、单一原生last/bbs输出。目标仍为同一完整模型Acc@0.25>59.5%、Acc@0.5>51.0%，并有直接证据支持三个有效贡献，再独立训练Nr3D/Sr3D；允许对应作者预训练权重。

### 已知最好结果与当前证据边界

保留的extremal_support终点为5677/4920，即59.7076%/51.7459%，完整原始候选数组已归档、CPU重算及终点构建/头部恢复已审查。它来自冻结父模型的适配，不是本轮普通联合训练结果。whole_support为5676/4921，两者+1/-1不能证明极值来源有独立增益；三个有效贡献仍未成立。完整所需作者PV、G及支持模块依赖和Nr/Sr作者权重保留。

### 两次原工程失败保留

第一次工程预检实际完成两次原生训练更新后，在逐步梯度验收停止；原FAIL不改。

第二次工程预检同样实际完成两次更新。四组参数更新已记录，完整1295状态及Adam状态已保存，随后第二次模型构建在原始1234状态断言失败，尚未完成cold恢复；extremal组未运行，原正式队列关闭，未开始epoch训练。第二步whole门值6144项均负，内部模块两步梯度为零，证据仅适用于已执行的whole初始化，不外推未执行组。

### 最小实现修复及真实CPU见证

实际PV VSA构造器会原地增加传入MLPS的层定义，TrainTester原样复用同一model_cfg；每次ModelClass构建改为copy.deepcopy(self.model_cfg)，仅新增import copy及该一处参数修改。其余13个计算文件、模型数学、原生loss及优化器不改，1234/1271/1295、strict加载、完整Adam/scheduler和四类RNG断言保留。

真实暖环境CPU构建发现，共享配置两次VSA状态由126变186；独立复制两次均126且key/shape一致。未forward、未更新、CUDA未初始化；这证明配置污染及结构隔离，不能冒充GPU完整恢复。

独立源目录为/root/autodl-tmp/pvground_native_joint_training_20261010/PV-Ground，116项来源逐项SHA核对。原失败源码与结果保持。旧whole的完整工程权重841675936字节，SHA256为588cb2100985ab27baea57592b5d4b3470e857a34972401891b5620fec8b5b6b；新工程先只补该状态的cold恢复，0新optimizer.step，不重复成功的两次训练。

### 新初始化与正常训练范围

extremal工程及后续正常训练加载当前保留的f9898c5dc7a5e97bf152421efc9072c23ef255618c64477a4bfacf996f2b3b19模块终点，绑定d06支持父权重。该模块已有3723次更新，支持模块此前11169次更新及G适配历史须披露；这是单独的初始化选择，不把它归因于配置修复涨点。优化器/scheduler重新初始化，不使用工程预检状态继续拟合。

正常训练通过train_dist_mod原入口运行：全36665条ScanRefer训练表达，batch/effective batch8、drop_last实际每轮36664行/4583更新，3轮，共13749更新；原可训练骨干、Decoder/G表示与新增模块联合更新，原生RoBERTa冻结策略保留。核心/骨干LR1e-6，新增模块LR1e-5，WD5e-4、clip0.1。此为ScanRefer适配，没有复用作者联合ScanNet检测的joint_det/detect_intermediate/augment_det配方。

改进在forward中产生唯一最终框，原criterion监督实际交付框，训练期G与selected-query Mask监督保留；推理不读取GT、不按当前低分裁掉256候选，不增加双源排名或评估器改框。离散硬Mask参考及极值来源选择仍不是全程可微，不能声称所有路径都由框loss更新Mask。

GPU工程先验证旧whole严格恢复，再对加载保留头的ext进行两个真实batch联合更新及完整恢复；内部/输出模块均检查任务梯度和实际更新。初始检查对应实际混合公式，不再要求预训练头输出零。仅实际成功并由同一原fresh integrity任务续审绑定收据SHA后，才启动一个extremal正常训练臂。工程无精度分数，普通训练E0和每轮都需完整9508验证；旧5677/4920不能直接填入新E0。

原SSH入口一次返回255/零stdout，未取得GPU启动收据；实际ssh -G显示继承本地ProxyCommand。仅本项目SSHargv显式ProxyCommand=none，默认SSH配置不改；原失败字节保留。后续入口仍检查新engineering目录不存在，避免重复启动。

实时启动、唯一观察节点及是否实际完成，见同目录ENGINEERING_LAUNCH/ENGINEERING_OBSERVATIONS/ENGINEERING_INTAKE及NORMAL_LAUNCH/normal_status原始收据。没有这些终态就不写通过或新精度。长任务首次观察按预估节点，后续240秒，不增加并行轮询器。

工程无分数权重仅在严格恢复、实际审查及SHA检查后清理；保留最佳模型、必需父权重、活动恢复及唯一文本记录。研究目标仍ACTIVE_UNMET。
