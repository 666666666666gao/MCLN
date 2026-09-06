# ScanRefer 预训练读出联合训练：首轮计划

2026-09-06。验收优先级遵循 `docs/SCANREFER_REC_FIRST_TRAINING_SCOPE_2026-09-06.md`。
本计划先验证已有学习模块能否在在线训练中与最后一层 Decoder 协同，不重新设计主干或
增加关系图、体素主干、Selector/Gate。它是已有系统的训练方式改进，尚无新质量结果。

## 核心问题与最小对照

当前完整 V99 的最终排名不携带梯度。Parent、Geometry、V99 的连续输出可训练，但
过去在冻结缓存上拟合，无法通过这些目标更新主网络表示。现在复用全部已有权重，把
连续读出监督接入同一训练图，检查这一连接是否改善 REC。

| 分支 | 原生 GT 损失 | Parent/Geometry/V99 连续监督 | 读出损失更新最后一层 Decoder |
|---|---|---|---|
| detached 对照 | 保留 | 在线计算并更新已有读出参数 | 否，只阻断这条梯度 |
| joint 候选 | 保留 | 同一目标、同一权重、同一数据 | 是 |

两臂从完全相同的 E71、Parent、Geometry、V99 参数开始，同批输入、顺序、预算、参数
容量和候选规则。训练后的候选数值允许变化，不能声称两臂仍有完全相同的候选框。
两臂均更新同一组最后一层 Decoder/预测头参数和已有的 42 个读出张量；不是一个臂
多训练整个主干。读出共 544396 参数，没有新建一套随机评分头。

主要待验证结论：读出到核心网络的梯度连接是否带来 REC 净收益。
需排除的解释：收益仅来自继续更新原生模型或再拟合已有读出。
即使有效，也不能仅凭模块封装声称已经去掉所有几何后处理或获得新关系建模能力。

## 固定实现与训练配置

- 起点：四份已核验的 ScanRefer 保护权重，身份见当前验收文件。新 checkpoint 同时
  保存核心网络、三份读出当前参数、归一化及新训练来源，不覆盖原 E71/V99 文件。
- 核心可训练范围：`decoder.5.*`、`prediction_heads.5.*`；其余主网络参数和所有
  running buffers 保持。实际连接/未连接张量由零更新原生检查记录，不伪报全部有梯度。
- 两臂的学习模块使用 eval 模式进行短程继续训练，保留固定 buffers 和原推理模式，
  通过 requires_grad/optimizer 更新指定参数；不以 train/eval 模式区别实验臂。
- 新建 AdamW。核心学习率 `1e-6`，三份读出 `1e-5`，weight decay `5e-4`，
  全部可训练参数梯度裁剪 `0.1`。E71 的旧 optimizer 不用于新的参数组。
- 总损失：原生 MCLN GT loss + `(Parent loss + Geometry loss + V99 loss) / 3`。
  三份读出使用已有连续质量/排名目标。不存在 IoU>0.25 候选的行不产生排名正例；
  Parent/Geometry 仍接受连续质量和阈值标签。V99 的局部变体排名仅监督有合格变体的 Query。
- GT 仅进入 loss，forward 只读原输入、Query/Mask 特征。保留原 parser、mesh-derived
  superpoint、候选生成与最终决策规则；不重新搜索 V99 margin 或 Mask 阈值。
- 首轮仅 ScanRefer 训练表达，不增加辅助 ScanNet 检测样本。seed=0，B12，末批自然
  余数，0 worker，不做旋转/翻转等输入增强。此阶段是有明确范围的继续训练，不宣称
  与旧完整 baseline 重训使用同一训练日程。
- 每臂遍历 fit 表达一次，步数为 `ceil(fit_rows/12)`；不挑中间最好，不根据中途 loss
  改学习率、延长步数或切换方法。成功后的多 seed 用于最终结果可信度，不作为当前首轮前置。

## 数据与证据的含义

使用原 ScanRefer 36665 条训练表达。按
`sha256("scanrefer_joint_readout_v1" + "\0" + physical_space_id)[:8] % 5`
划分；0 为本轮 holdout，其余为 fit。physical_space_id 是 scan_id 去掉最后的扫描后缀。
全量行 ID、扫描/物理空间数及 SHA 在实际协议检查后写入训练 manifest，训练前锁定。

E71 与原 V99 已经见过这些训练场景/表达；holdout 只表示没有参与本轮更新，属于开发
检查，不能写成新场景泛化。16 行原生预检从 fit 中按原顺序选不同扫描的第一条表达，
不按 GT/质量筛选；保留各扫描的全部表达进行原生 distractor 构造，再取预检行。

正式 ScanRefer 的 9508 行不参与训练、归一化或参数选择。先在开发 holdout 报告受保护
起点和两臂终点的 REC、ScanRefer Mask、净修复/破坏及场景聚类区间。joint 的 REC 两项
相对起点及 detached 对照均不下降后，才注册固定终点的正式验证；开发检查没有达到此条件
时封存本次连接方案，不用正式验证集挑另一个中间 epoch。

正式晋级依据：ScanRefer REC 不低于 V99 的 5572/4797 命中，且 Mask 三项不低于
原论文 58.70/50.70/44.72%。通过即尽快转 Nr3D/Sr3D 的 REC 训练，不等 59/51 争取目标。
Nr3D/Sr3D Mask 不进入后续门槛。未通过的方案不替换 ScanRefer 受保护最好结果。

## 顺序、时间和当前状态

1. 已完成：实际三份读出权重的 CPU 合成检查，原路径与可求导路径输出一致；42 张量
   梯度有限非零；阻断对照仅切断到视觉预测的梯度；单份内存 checkpoint 往返一致。
   没有完整主网络 forward、真实训练数据、优化器更新或正式质量结果。
2. 已排队：16 行 ScanRefer fit、B12+4 的原生 GPU 检查，验证严格加载、原输出/evaluator
   一致、真实读出梯度到最后一层 Decoder，并测量显存与耗时。0 更新。复用 GPU 独占锁，
   在旧 Nr3D Mask V3 释放后自动执行，未并发争抢 GPU。
3. 原生预检通过后，先完成下述 512-row 冻结教师诊断，结合用户最新的教师路线建议
   明确近期训练采用的路径，再锁定实际 fit/holdout 清单与训练 manifest。上述联合读出
   是已准备好的性能接续对照，不自动等同于训练期教师或原生网络内化。
   `scripts/run_scanrefer_joint_readout_pair.py` 已通过原 Python 3.7 的编译和入口检查；
   尚未创建训练 manifest、启动训练或更新优化器。暂估一次遍历配对训练与开发评估
   2–5 GPU 小时，待实际步速修正；该范围不是已观测耗时。
4. 首段固定进度用于估时，随后接近预计结束前五分钟查询，再按 240 秒观察。中途指标
   只用于发现运行故障，不用于改变科学配置。
5. 通过 ScanRefer 正式底线后，优先启动 Nr3D，再做 Sr3D。Sr3D 旧保护 checkpoint 当前
   尚未恢复；可使用核验过的现有预训练网络继续训练，但须明确来源，不能写成恢复旧 Sr 权重。

实现：`scripts/scanrefer_joint_readout.py`。原生预检：
`scripts/probe_scanrefer_joint_readout_native.py`。
CPU 证据：`refine-logs/scanrefer_readout_gradient_probe_20260906_v1/`、
`refine-logs/scanrefer_joint_readout_cpu_20260906_v1/`。
GPU 队列证据：`refine-logs/scanrefer_joint_native_probe_20260906_v1/queue.json`。

## 最新补充：先明确教师可迁移的收益

用户最新建议优先评估冻结 E71+Parent+Geometry+V99 作为训练期教师，GT 为主监督，
学生推理不依赖教师。这里明确区分：在线更新已有读出保留原几何规则；教师内化要求
原生学生自己的分数和框恢复相应收益。将原模块移入 forward 不能替代后者的验证。

为避免直接用较差教师框覆盖已有 GT 监督，固定读取 512 条 ScanRefer fit 表达：在
上述 fit 行序列中等间隔取位置 `floor(k * fit_rows / 512)`，k=0..511，不按质量选样。
保留各所选扫描的全部表达构造 distractor，然后取诊断表达；B12、无增强、0更新、
0正式行。数据以前被 E71/V99 见过，仅解释教师信号，不证明学生或新场景泛化。

逐行比较原生部署选择、原 Hungarian 的 root 匹配 Query、完整256框中最高IoU、
V99选择的变体框及该变体的原始Query框。同时按当前框与教师框的几何重叠重建对应，
记录教师框过线但对应学生框不过线的数量，避免将所有变体收益误称为排名收益。
教师的错误、修复和破坏均保留，不先过滤成只有正收益的审计数据。

10:36:19 已排入 `mcln_scanrefer_teacher_transfer_20260906_v3`，screen 31455，等待
16-row 原生预检成功，然后获取同一 GPU 锁。依赖检查间隔240秒，最多512行；没有
训练、没有自动启动正式评估。CPU合成检查通过错误教师记账、排序/几何收益分离、
Query重新排列后几何对应不变。原入口检查确认 `eval_use_selector_choice_scores=False`，
因此原生对照使用实际 Default 分数。v1/v2仅是部署准备错误，不是模型质量失败。

候选局部视觉读取是后续独立结构研究：当前 `BiDecoderLayer.cross_v` 使用1024个
seed作Key/Value，Query已受文本/对象交互影响，但没有显式候选框邻域读取。下一项
结构应改变这里读取的空间证据，并单独比较原路径；本轮不把已有读出梯度接通当作
新增局部视觉信息，也不同时修改主干、matcher和多套loss。

论文依据只用于界定问题：Rank-DETR §3.3 将分类置信度与GIoU质量联系，并强调匹配；
EG-3DVG提出几何一致视觉聚合。它们不证明本项目的缩小版本必然有效，也不提供
当前ScanRefer达到SOTA的证据。
来源：https://arxiv.org/html/2310.08854 ，
https://openaccess.thecvf.com/content/CVPR2026/html/Park_EG-3DVG_Expression_and_Geometry_Aware_Grounding_Decoder_for_3D_Visual_CVPR_2026_paper.html 。
