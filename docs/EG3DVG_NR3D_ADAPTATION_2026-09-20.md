# EG-3DVG Nr3D 单轮原生适配控制

固定协议于2026-09-20 18:13在当前Nr零更新迁移正式结果完成前写入。

## 研究职责

使用已验收的完整作者ScanRefer epoch69权重进行有限Nr3D适配。保持EG网络、原生预测规则和完整原生损失；不加入C/D/G、旧V99或额外关系/质量头。它是之后新增机制的原生训练控制，不是作者240轮Nr3D专用训练复现。

ScanRefer原权重和V99受保护权重保持独立。此次只训练Nr3D；不能把它写成同一权重已经超过三个数据集基线。

## 固定设置

| 项目 | 设置 |
|---|---|
| 初始化 | 作者ScanRefer epoch69，全1276项严格加载 |
| 数据 | Nr3D32919条 + ScanNet1199条重复10次，共44909条 |
| 更新 | 完整一次遍历，batch8，末批5条保留，共5614步 |
| 种子/顺序 | seed2027；独立RandomState预先保存44909条排列及SHA |
| 优化器 | 新AdamW，weight_decay0.0005，不恢复旧optimizer/epoch/scheduler |
| LR | 常数：主网络1e-4，PointNet++1e-5，text_encoder组1e-5；RoBERTa仍按作者requires_grad冻结 |
| 梯度 | 原生clip_norm0.1；loss/梯度非有限立即停止 |
| 损失 | 原生Hungarian框、Mask、soft-token、对比对齐、query生成及ECL负表达损失，原系数 |
| 预检 | 两批共16条实际训练样本、完整双forward/loss/backward/AdamW；两步权重丢弃，正式从作者权重重新加载 |
| 选择规则 | 固定5614步终点，不按中间loss或验证成绩选epoch、seed、输出头 |
| 验收 | 固定Nr7899、last/bbs、作者对象输入及候选过滤；bbf与未过滤路径只诊断；Mask不设门槛 |

## 仅训练副本的方向词修正

实际作者解析后的32919条Nr训练表达，原 `_augment_nr3d` 对句首/大小写/标点的处理不一致。使用原十个方向词，通过小写字母分词统一 `_is_view_dep` 与 `_augment_nr3d`。全量检查325条从允许大角度旋转变为禁止，0条反向变化；view分类标记另外有274条变化。训练样本正文和图解析不改，验证源码不改。

原训练Dataset源SHA5f771e1d55e0663ba4f84240866bbf3c4d4a94e3033bea26fce72b5523793a30；修正后a425943b8abdb82c0b8aae6cd8c20868b9596ff6045e8c410c472f87640578d9。八个实际例子及325行清单见view_fix.json/view_fix_rows.json。

这属于训练标签一致性修正，不作为网络创新。适配后的增量不能全部归因于该修正，因为本轮同时进行了原生网络更新，也没有运行旧错误增强的配对训练。

## 执行与恢复

- 独立运行目录：`/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v1`。
- 状态存储：`/root/mcln_eg3dvg_nr3d_adapt_states_20260920_v1/latest.pth`。启动前核验该分区空闲超过4GB；每512步及终点原子替换单一恢复文件，包含模型、优化器、RNG、步数和spec SHA。
- 完整训练日志逐步写JSONL。保存恢复状态不代表按它选择模型；只有固定终点接正式评估。
- controller4951于18:13:03启动，先每180秒等待Nr零更新评估controller.exit0及独立audit通过，再获取同一GPU锁，运行训练预检、正式fit、终点8条前向、7899正式和CPU复算。任一步失败即停止；不会在旧任务仍占卡时启动新forward。
- spec SHA `f5608e579a52b4730cbeca465beddec908cbe38b5825c333eecfd367f45110ff`。
- 沿用已通过完整EG的Torch1.12+cu116环境，环境SHA81a835e8144f7070d66feb0afa460911dec610052ced6b96c98097352cd022fd不变。

18:13:58实际检查4951处于等待，4408仍执行零更新迁移，日志6144/7899。此时没有训练前向或适配性能结果。GPU训练是否通过、显存与耗时须由真实预检提供，不能由Python编译或CPU数据读取替代。
