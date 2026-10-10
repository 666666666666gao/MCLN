## §20.376.147 — 原生匹配中的公共Text Mask成本与候选自身支撑

等待正常C-off第2轮既定观察期间，完成一次限定的源码与实际原生matcher CPU检查。没有重新读取活动训练日志或GPU状态，没有改网络、criterion、优化器、数据或当前配方。

当前正常PV源码在选取Text Mask后，将其expand给全部256个Query；候选自身Query Mask另存于sp_last_pred_masks。末层criterion同时收到两路Mask，但HungarianMatcher的直接Mask成本只读取公共Text Mask，做硬二值判断、原始点映射和L1距离；cost_masks系数0.0002。实际criterion构建仍为分类1、框L1为0、GIoU为2。

因此，固定候选框和token分数时，直接Mask成本在每一个GT列内是相同常量。原生接口最多132个有效GT，候选为256，全部有效GT各匹配一次时，该项给全部完整分配增加相同的总成本，未提供区分候选自身支撑的依据。不能把这一点写成“原生没有Mask成本”。

实际CPU检查从当前warm model_source加载未修改的models/losses.py；三份主源码、两份utils导入文件以及运行环境规范均按实际原生manifest完整SHA核对，并记录真实导入路径。使用原生Torch1.10.2+cu111及SciPy assignment，CUDA隐藏、单CPU线程，不执行PV网络、criterion、真实数据或优化器。

九组受控输入覆盖单GT、两GT和混合batch，均为256个Query、256个token、四个合成点。固定框和token logits，替换候选自身Query Mask后，实际原生成本矩阵完全相同，分配不变；更换公共Text Mask或去掉Mask键时，其成本差为GT列常量，保留浮点舍入量摘要。完整矩阵仅在内存捕获，回执保留差值摘要和分配，不声称保存完整矩阵数组。

这一结果属于simulation_only工程行为检查：九次matcher调用、零模型前向、零criterion调用、零优化器更新、零GPU执行、零真实表达，formal_accuracy=null。没有新的ScanRefer、Nr3D或Sr3D指标，不能把合成GT当作正式验证，也没有证明这是正常续训掉点的原因。

必须保留间接路径：候选自己的Query Mask会通过native_mask_geometry和candidate_span_mixer改变last_center/last_pred_size，最终框再进入GIoU匹配。故固定框的检查只隔离直接Mask成本，不能说整个模型的Query Mask不影响匹配。原生匹配索引随后同时用于定位、语义及Query Mask损失。

源码与实际检查均经新上下文Codex审查；同模型家族、provisional，实际模型身份/推理档位无外部凭证，不宣称跨家族或外部独立验收。报告与完整字节摘要保留，不把执行者自己的判断改写成独立审查。

后续先完成活动C-off三轮规定预算。若需要调整几何学习责任，优先用明确的同起点、同预算对照检验候选自身支撑参与末层训练分配的增量，保留真实有效GT、其他已匹配实例与联合检测行的职责；当前没有实施这一新策略。Mask参与匹配已有前作，这次发现不构成“首次Mask匹配”或第三个有效论文模块的证明。

保留最好5677/4920及所有必要父权重；Nr/Sr正式训练尚未启动。唯一原观察器52851/PID51540仍按2026年10月11日05:00:40检查正常第2轮。完整目标继续ACTIVE_UNMET。
