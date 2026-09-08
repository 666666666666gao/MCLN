# 普通分源多尺度Query读取控制

当前是待测控制，不是覆盖感知方法或已成功网络。承接20.197，empty-pool负结果封存。

## 固定结构

- 复用正确VSA排序、正确mesh/检测框增强与官方Scan epoch81父权重；不载入失败终点，不安装empty-pool清零。
- VSA融合前1024通道依次为BEV/raw/conv1/conv2/conv3/conv4，宽度128/128/128/128/256/256；显式保存[B,1024,1024]记忆，坐标仍是原FPS关键点。
- 每一路用对应的预训练VSA Linear列切片初始化独立288维投影，再分别LayerNorm；没有复制融合后特征、假造新增原始观测或额外体素化。
- 六路按来源依次拼成6144个token，各路重复同一份既有关键点位置编码。该位置编码只计算一次，与原encoder共用，不额外更新其BN。
- 仅最后Decoder层增加普通跨注意力：Query为已有语言/对象交互后的Query加原Query位置；Key为分源特征加位置，Value为分源特征。覆盖状态、empty mask、GT和预测IoU不进入此控制。
- 注意力复制该层预训练cross_v，新增注意力dropout为0，避免新增随机抽样改变原训练RNG流。随后288→288线性残差为零初始化，加入原cross_v输出、位于原dropout_v/norm_v之前。原全局cross_v、自注意力、语言/对象路径、Box/Mask头、匹配及损失保持。
- 投影来自真实源，但分源LayerNorm与注意力并不等价于原拼接融合；只有零输出残差保证插入时整体函数预期一致，必须实测。首步新增内部投影梯度可为0，输出层应收到梯度；更新输出层后内部路径应能得到非零梯度。
- 模型架构及新增权重必须随spec/checkpoint绑定；严格加载原生父权重后才安装reader，新checkpoint需要安装同结构后恢复全部状态。禁用开关只用于配对执行检查，不作为部署选择器。

## 当前检查与后续固定预算

首先在已有4条fit输入上检查完整原生loss、禁用/启用零更新重放、两次临时更新、参数覆盖及冻结文字不变；不保存临时权重。检查仅证明接口和梯度，不证明精度。随后batch8容量检查复用旧流程。

通过后从同一官方Scan parent与固定新增初始化开始，seed2027，29778fit/6887模块留出、batch8、一遍3723更新、LR/backbone LR均1e-5，原weight decay0.0005/clip0.1/native REC和Mask损失。容量check后恢复完整起点、重置RNG；不重选epoch或追加训练预算。对照为已结束同预算正确接口native，主bbs、bbf仅诊断；新增参数/算力单列，不能把对native增益自动归因于覆盖机制。

当后续研究覆盖条件时必须同读取位置、同源和同预算与本控制比较；关键点mask不能当Gumbel连续Query位置处的实际支持，某尺度空不能删除整个候选。当前不实现覆盖权重、语义/几何双分支或新质量loss。

own-initial主REC两阈值筛选及正式V99 REC5572/4797、Scan Mask58.70/50.70/44.72门保持；正式通过后立即Nr/Sr REC，不等59/51，不恢复Nr/Sr Mask门。6887场景主干已见，不作新场景泛化成绩。空间不足时先核实实际文件与保留用途；不得删除受保护权重来启动本控制。

## 来源与创新边界

PV-Ground/OpenPCDet提供预训练骨干、VSA及原Decoder，本次只增加显式分源输出和项目读取模块；保留上游来源与许可证。分源多尺度注意力是普通控制，不声称首次多源/多尺度Query方法。预训练切片初始化与保留原路径用于减少适配成本，是否带来严格定位净收益待完整实验。


## 当前执行证据（20.198）

上一回合空邻域控制终态/审计/清理已全部收尾；本回合实现并验证新的普通分源读取控制，随后启动固定Scan训练，属于progress。没有重启旧负结果、加载其终点或改变目标。RGBD项目继续忽略。

### 新结构已实际实现，尚无精度收益

独立源码目录/root/autodl-tmp/mcln_pvground_source_query_source_20260908_v1/PV-Ground；保持原修正源码与runtime完整不变。仅在pv_utils.py暴露真实融合前1024通道，pv_ground.py显式传递它和已有关键点位置，BiDecoderLayer在最后层接新reader。原位置编码只计算一次，继续与encoder共用；不额外改变其BN调用次数。

models/pvground_source_query.py读取BEV/raw/conv1/conv2/conv3/conv4真实源，128/128/128/128/256/256通道分别投影为288；投影从已严格加载的官方VSA对应列切片复制，独立LayerNorm。六路6144个token使用相同关键点坐标位置；最后层Query经过既有语言和对象交互后读取它们，新增attention从原cross_v复制、dropout=0，输出线性零初始化残差加入原cross_v输出、进入原dropout_v/norm_v。原全局/语言/对象/自注意力、预测、匹配、REC与Mask损失保留。新增714528参数，可训练合计28674139参数/807张量；不是学习式coverage、empty-mask、几何分支或末端重排。

源特征在原FPS关键点处，非预测框中心重采样；不把关键点mask混同Gumbel连续Query位置处的覆盖。真实分源与已有普通多尺度attention是控制，不能作为已验证新颖性或新成绩。模块SHA e0fcc6a30402ebeba7a9ffa1be907515350b64e408602a8b47b702e83a5d05ab，准确源文件SHA清单归档于source_port.json。

### 三次检查的结果与边界，保留原失败记录

v1在eval全张量逐位断言处停止，0更新；差异首先显示为last_pred_mask_seeds，不能把它跳过并宣布整网完全一致。独立disabled→disabled→enabled三前向诊断发现，原生重复本身即有superpoint中心最大2.384e-6米、Mask logits最大6.58e-5差异；所有新增残差为0，REC张量与结束RNG一致。说明Mask数值重放不是此处新增模块的独立检验；不据此宣称已完全定位某个CUDA算子缺陷。

代码审查指出缺少训练模式与结束RNG检查，已补。v2通过两个eval批次后，在完整train重放的source_features逐位断言处停止，仍0更新。该特征位于新reader之前；v3先让reader连续两次禁用，实际测得上游source_features仍差2.384e-6。故训练模式严格检验改为同一次前向捕获的真实最后层输入：恢复全部参数/缓冲区及Python/NumPy/CPU/CUDA随机状态，分别禁用/启用reader，最后层完整输出和结束RNG精确一致；新增残差两次均0。没有修改模型、容差或训练门，仅区分整网数值重放与新增层功能等价的证据。整网train及Mask不作逐位一致声明。

v3在22:07:40退出0，实际4次full eval、4次full train、额外2次最后层forward；2次临时AdamW更新、不存权重。eval九个REC/来源字段及结束RNG精确一致；完整原生损失和两阈值/Mask evaluator手工选择核对通过。step1仅零初始化输出层收非零梯度，step2所有六路投影及新增attention梯度非零，全部现有梯度有限，冻结文字参数未变。batch2两次峰值allocated约5.24/5.39GB；不能当batch8容量通过。loss14.8402/13.0991只是接口数据，不是精度。

标准轴无实际规范违规；规格轴最初的训练重放缺口经代码修订和上述限定验证处理。生产训练桥接的独立只读审查无阻塞发现：先strict父权重再安装reader，initial包含新增state，容量检查后完整恢复，全部新增参数进入optimizer/delta。独立审查是同模型家族、暂定意见，不声称独立跨模型验收。

### 固定Scan长训已启动

22:11:12训练controller25348、screen mcln_pvg_scan_sourcequery_v1启动；22:11:14独立审计controller25357启动。root分别为/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_sourcequery_v1及对应endpoint_audit目录。22:12:13再次核实两个原进程仍存活，训练处于Loading train files；GPU报告2426MiB，没有capacity.json/initial完整receipt或实际长训更新证据。不能把launch说成已完成batch8检查或已产生新训练终点。

官方Scan epoch81起点、corrected mesh/detection、seed2027、29778fit/6887留出、batch8、3723固定更新、LR/backbone LR1e-5、weight decay0.0005/clip0.1/native全部损失。临时检查权重不进入训练；新增weights初始单独快照，容量backward无update后strict恢复整套状态并重置RNG，initial后也恢复预定随机起点。checkpoint绑定新增架构、模块SHA、port SHA、parent和spec。一次latest原子覆盖加固定terminal，不存多epoch历史。

本轮复用env_spec966235b2...及已经通过的真实算子/文档执行环境，不安装包、不重编译、不改环境文档。磁盘启动空闲2224848896字节；依据实际331MB native delta及新增参数/Adam两份状态估计新delta339755998字节，两个并存delta加300MiB导出/日志预算994084796字节，另保留1GiB。检查总需求2067826620字节小于实际空闲；未进一步删除任何权重或复制父权重。运行时继续检查磁盘。

预计本轮训练加留出约3—4小时，尚待batch8实测吞吐更新。CPU审计首次22:31附近、随后300秒按原队列；下一回合核验同进程容量/initial并完成对应正式接续准备，当前新的9508正式接续尚未启动，不能说已经排队。own-initial bbs两阈值筛选不变；最终Scan正式V99 REC5572/4797、Mask58.70/50.70/44.72，过线立即Nr/Sr REC，不等59/51、不恢复Nr/Sr Mask门。未通过则保留负结果，不中途延训扫LR。当前0正式行，V99/Nr/Sr正式保护成绩不变，三数据集总目标仍未完成。
