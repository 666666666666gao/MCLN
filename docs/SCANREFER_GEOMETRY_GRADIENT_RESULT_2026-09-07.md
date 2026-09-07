# ScanRefer：Mask几何输出与评分梯度的实际边界

## 固定输入与检查范围

复用保护E71、Parent、Geometry、V99及614-file冻结来源，正确mesh训练superpoint目录。使用既有split_protocol中预先固定的16条fit表达，B4、model.eval；没有训练集重划分、optimizer、权重写入或正式验证。该批训练场景曾被预训练主干使用，不能将其命中数解释为新场景性能。

17:44:52启动实际Python62155；17:46:17完成，controller.exit=0。原始1144项模型state及三个固定读出逐值不变，峰值GPU分配1607.82MiB。17:47:59实际进程已结束，GPU回到1MiB。

为追踪原生参数链，仅设置末层Decoder、末层预测头、x_query、x_mask、rel_encoder的requires_grad；没有更改参数。梯度目标是forward直接产生的last_pred_masks/sp_last_pred_masks原始tensor，以及原始center/size tensor与这些原生参数，未用重建的fused view或人工leaf替代。

## 实际结果

每种几何对相同root框计算有效候选坐标L1，仅用于查询梯度路径，不是新训练目标或匹配策略。16×16个候选Query对应以下七变体：

| 变体 | 有效候选数 | 坐标L1→原始Text/Query Mask梯度 | 坐标L1→原生参数 |
|---|---:|---|---|
| regressed | 256 | 均未连接 | 有非零梯度 |
| fused_t0_exact | 251 | 均未连接 | 零 |
| fused_t0_q0.005 | 251 | 均未连接 | 零 |
| query_t0_q0.005 | 251 | 均未连接 | 零 |
| fused_t0.5_q0.005 | 250 | 均未连接 | 零 |
| blend_regressed_fused_q0.005 | 251 | 均未连接 | 通过原生框有非零梯度 |
| blend_regressed_query_q0.005 | 251 | 均未连接 | 通过原生框有非零梯度 |

同一forward的已有读出GT损失和原生GT损失均向16/16条样本的原始Text/Query Mask传递非零梯度。四批读出损失到Query Mask的范数为0.018164、0.015453、0.012380、0.008526；原生GT损失对应0.023593、0.022784、0.014327、0.012422。梯度范数只说明连接，不是作用优劣或增益。

原因可以在实际算子逐行定位：`point_logits > threshold`生成离散成员集合，min/max/nanquantile只读取固定输入点坐标。前景logit均值、标准差等评分特征则仍可微。因而“Mask到评分器有梯度”与“框坐标损失可以直接纠正Mask成员边界”不是一回事。混合变体对原框可微，也不能推断对Mask支撑集合可微。

本地独立NumPy从保存框重算全部IoU与L1，最大差分别1.20175e-6、1.62982e-8，均通过2e-6容差。该重算验证几何数值和保存的梯度记录一致，不冒充第二次独立原生autograd执行。

## 能与不能得出的结论

- 当前joint/frozen_gt的连续读出监督没有直接优化硬Mask几何边界的梯度路径；它仍可通过连续评分特征、原生框和共享参数间接改变最终结果。
- 这不是将所有Mask梯度都切断的bug，也不证明它导致历史全部REC下降。跨越阈值仍会改变实际框；离散成员在局部的autograd断开不等于输入对框没有因果影响。
- V44已有多候选focal/Dice Mask监督。不能把再次给多个Query加普通Mask损失视为新机制。
- 不能据此直接把硬阈值换成sigmoid或声明端到端优化已成功。连续概率在大量背景点上累积，可能改变有效支撑及框范围，必须先检查实际前向。

## 连续概率几何原型的实际检查

实现隔离原型`prototype_probability_geometry.py`：固定输入坐标排序，以sigmoid Mask质量形成CDF，并在既定0.005/0.995分位点线性插值读出边界。它直接检验可微坐标读取的前向代价；没有插入MCLN、没有替换七变体或Pareto、没有新增训练损失。

这是普通加权CDF插值，不是新颖性主张，也不是[可微排序OT方法](https://papers.neurips.cc/paper_files/paper/2019/hash/d8c24ca8f23c562a5600876ca2a550ce-Abstract.html)的复现。类似地，[BoxInst](https://openaccess.thecvf.com/content/CVPR2021/html/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.html)已有Mask/Box投影监督原则；不能只把二维轴改成三维轴即宣布原创。

原型已在同一16条fit输入完成；独立核验点SHA、Query映射、原硬框逐项完全相同。对硬框本来有效的251个Query，只比较同Query的fused_t0_q0.005与未训练连续概率框：

| 检查项 | 原硬Mask框 | 连续概率框 |
|---|---:|---:|
| 候选IoU>0.25数量 | 210 | 211 |
| 候选IoU>0.50数量 | 208 | 210 |
| 16行候选oracle@0.25/@0.50 | 16/16 | 16/16 |
| 有效候选平均IoU | 0.71459236 | 0.71281146 |

这些是候选级比较和GT辅助oracle，不是部署REC或泛化增益。不能因+1/+2就晋级，也不能因为均值略降宣称所有连续几何无效。真实坐标L1对原始Query Mask的四批梯度范数为0.0171761、0.000475928、0.0000868858、0.0000505885；对原生可训练参数也均有限且非零，说明新的直接几何路径确实存在。

被原硬阈值排除的点，其累计概率占全部概率的均值2.4856%、中位数1.3157%，155/251候选超过1%。这不是GT背景比例；它说明soft与hard支撑不同，需要同时报告两种几何，不能只优化soft损失。数学反例也显示大量小背景概率可撑大框；实际16条未出现候选过线数总体崩溃，因此没有把理论风险误写成已发生的普遍失败。

原环境7项CPU检查通过：原硬框梯度与阈值变化3项，CDF数值、有限差分梯度、点置换/仿射几何、低概率累积4项。原型receipt SHA `df24466ed0b6b318407d85dc52bc460cdb9c265026aa41cf5f77f0747eb9fb3d`，NumPy独立IoU重算最大差1.20175e-6。原生模型、读出、权重、部署方式不变，optimizer/正式行/新权重均0。

首次启动观察误用了上一探针名称，导致本地观察assert失败；实际新Python62328正常运行。随后只修正只读ps查询核对原进程，没有重启。17:59:33确认该进程自然结束、controller.exit=0、GPU1MiB。

## 下一项训练机制的最小定义

隔离训练辅助`native_mask_geometry_supervision.py`使用当次原生Hungarian的root匹配Query，读取该Query的fused Mask概率几何，以root GT监督中心/尺寸/GIoU。GT只在损失与匹配处使用；不固定旧Query编号、不模仿教师、不改变原生匹配、不把全部Top16都作为root正例。损失沿用原MCLN的5×(中心L1+0.2×尺寸L1)+GIoU系数。

它改变的是Mask几何的训练责任，区别于仅训练原生框头的教师辅助、V44逐点focal/Dice或冻结读出的质量标签。当前仍是独立实验辅助，未改生产训练或推理接口，也不构成已验证原创网络。

16条实际fit梯度与两臂各2次一次性更新已完成，controller.exit=0、9项CPU检查通过。四批梯度余弦0.0566、0.2052、0.7435、0.1180，不能外推全数据集。第一批两步后控制/候选几何loss为0.686834/0.676409（共同起点0.701149）；这里只说明目标可执行，不是REC效果。允许84项参数中两臂各82项改变，其余state和读出不变，checkpoint0、formal0，峰值GPU2431.24MiB。

训练probe receipt SHA `699b74ef2b79a5c26fd4d51c49a973f2c5e79942ac3624e6ea26b4cfbd8ac26b`。固定配对见`SCANREFER_MASK_GEOMETRY_GT_PLAN_2026-09-07.md`；长配对尚未启动，须完成新runner/终态审计，不能套用旧16项框头保存接口。终态同时记录原生REC、完整V99、硬/软几何、原始Mask，不能以soft损失下降替代REC晋级。

## 证据

- `refine-logs/scanrefer_geometry_gradient_probe_20260907_v1/input_manifest.json`：SHA ad27833662348540e6721d887e7478274d5d699d5776cbee30d83cfe35772405。
- `receipt.json`：SHA 3f88249cdd4cb29a33efb310b69c951af4a493ae9cc3320f10a691b735ed994c。
- `independent_recount.json`：本地独立算术核验。
- `scripts/probe_scanrefer_geometry_gradient.py`、`scripts/audit_scanrefer_geometry_gradient.py`。
- `tests/test_mask_geometry_gradient_contract.py`：硬成员路径、连续统计梯度、跨阈值有限变化三个检查。

三个数据集正式结果和保护线均未变化；完整Goal仍未达成。ScanRefer正式过现行REC/Mask底线后即接Nr/Sr REC，不等待59/51，Nr/Sr Mask仍不设晋级门槛。
