# PV-Ground Nr3D/Sr3D：论文 Accuracy 与原生 REC 的边界

**结论：不能证明论文 Table 2 的 Accuracy 与当前固定 `last/bbs` 的任何 REC 计数器完全等价。** 原始 ReferIt3D 的 referential accuracy 判断所选候选对象是否为真实目标；PV-Ground 发布评估器判断所选 query 的连续预测框是否命中目标框 IoU 阈值，没有 candidate instance ID 的正确/错误比较。两者的判定定义不同。论文未指明 Table 2 使用哪个发布计数器，也未给出预测框到对象 ID 的规则。[PV-Ground §4.1/Table 2](https://openaccess.thecvf.com/content/CVPR2026/papers/Shang_PV-Ground_Text-Guided_Point-Voxel_Interaction_for_3D_Visual_Grounding_CVPR_2026_paper.pdf)、[原始 ReferIt3D evaluator](https://raw.githubusercontent.com/referit3d/referit3d/eccv/referit3d/models/referit3d_net_utils.py)。

本轮为静态 primary-source 调查。SSH/远程进度查询、GPU、模型/数据 forward、训练更新、队列/计时器操作、当前源码修改和旧证据修改均为0。以下作者日志不是本项目新增测量结果。

## 1. 论文数值、发布日志和实际分母是三件事

论文 Table 2 标为 Accuracy：two-stage Nr/Sr 为62.1/68.9，single-stage为51.3/56.5。§4.1描述候选目标框选择/分类正确率，没有给 ReferIt3D 的 IoU 阈值或 `bbs/bbf` 映射。P1:429–457；PDF印刷页38663、本地第6页。

作者模型库 revision `cf4a8b1eed045f1309e6a691ea948d1d6c54448e` 的四份原始文本（L1–L4:1–5）如下。这里“dataset长度”只引用日志第1行，不等于已证明的accuracy分母。

| 设置 | 日志dataset长度 | last/bbs REC025 | last/bbs REC050 | last/bbf REC025 | last/bbf REC050 | 论文Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Nr3D two-stage | 7899 | 61.899% | 50.519% | 62.089% | 50.684% | 62.1% |
| Sr3D two-stage | 17726 | 68.914% | 58.653% | 68.857% | 58.518% | 68.9% |
| Nr3D single-stage | 7899 | 51.247% | 38.144% | 51.336% | 38.106% | 51.3% |
| Sr3D single-stage | 17726 | 56.493% | 42.503% | 56.510% | 42.463% | 56.5% |

四个 `bbf/REC025` 百分数保留一位小数都等于Table2；Nr的`bbs`会显示61.9/51.2，与62.1/51.3不同。Sr两条路径都能显示68.9/56.5，不能靠Sr判别。**这只证明数值兼容，支持把author-bbf称为“paper-compatible numerical reference”；不证明这些日志是制表来源，不证明对象身份Accuracy=REC025，也不授权更换当前唯一固定bbs。** [Nr two日志](https://huggingface.co/AaNnWwTt/PV-Ground/blob/cf4a8b1eed045f1309e6a691ea948d1d6c54448e/PV-Ground_NR3D.txt)、[Sr two日志](https://huggingface.co/AaNnWwTt/PV-Ground/blob/cf4a8b1eed045f1309e6a691ea948d1d6c54448e/PV-Ground_SR3D.txt)、[Nr single日志](https://huggingface.co/AaNnWwTt/PV-Ground/blob/cf4a8b1eed045f1309e6a691ea948d1d6c54448e/PV-Ground_NR3D_single.txt)、[Sr single日志](https://huggingface.co/AaNnWwTt/PV-Ground/blob/cf4a8b1eed045f1309e6a691ea948d1d6c54448e/PV-Ground_SR3D_single.txt)。

**分母的具体缺口：** 作者print_stats把整数dets/gts之比输出五位小数，但two-stage四个Nr top1值均不兼容整数hits/7899，四个Sr值也均不兼容整数hits/17726。静态十进制区间计算得到以下兼容例子，均不是实测计数：

| 日志 | 假定分母 | 顺序为bbs025/bbs050/bbf025/bbf050的兼容整数hits | 证据性质 |
|---|---:|---|---|
| Nr two | 7899 | 四项均无整数解 | 五位小数区间算术 |
| Nr two | 7900 | 4890 / 3991 / 4905 / 4004 | 兼容示例，实际分母未证实 |
| Sr two | 17726 | 四项均无整数解 | 五位小数区间算术 |
| Sr two | 17728 | 12217 / 10398 / 12207 / 10374 | 兼容示例，实际分母未证实 |
| Nr single | 7899 | 4048 / 3013 / 4055 / 3010 | 兼容示例，实际计数未recount |
| Sr single | 17726 | 10014 / 7534 / 10017 / 7527 | 兼容示例，实际计数未recount |

例如 `0.62089*7899=4904.41011`；`0.68914*17726=12215.69564`，与邻近整数比值保留五位小数均不吻合。报告不将dataset长度重命名为accuracy分母，也不反推这些整数为作者真实结果。

O14:377–378记录的是`len(test_loader.dataset)`；O14:248–257用默认 `DistributedSampler(test_dataset, shuffle=False)` 和 DataLoader(drop_last=False)。PyTorch DistributedSampler默认补齐到 `ceil(N/world_size)*world_size`；O2:174–191对各rank的dets/gts直接求和，没有表达ID去重。**world_size=4会同时解释7900/17728，但只是源码支持的推断。** 日志没有argv、rank/world-size、batch或checkpoint SHA；保存的test脚本又明确nproc1，因此实际日志运行的rank、分母、版本仍未知。[官方sampler实现v1.10.2](https://raw.githubusercontent.com/pytorch/pytorch/v1.10.2/torch/utils/data/distributed.py)、[作者loader](https://github.com/AaNnWwTt/PV-Ground/blob/262e2592589baec7bb83a0d46aae6542d4ccedfb/main_utils.py)。后续单rank全验证仍须精确7899/17726唯一表达行；不能为了凑发布小数添加重复样本。

## 2. 两条score的来源：语言map，不是类别/实例ID正确率

固定作者commit `262e2592589baec7bb83a0d46aae6542d4ccedfb`。

验证样本过滤也必须匹配：val被映射到test scan list；Sr保留mentions_target_class=true；Nr test保留correct_guess=true，不启用mentions_target_class过滤。这些是corpus选择，不是按GT类别重选答案。O4:177–240；O16/O17绑定test scan lists。同样的row total不足以证明同一批表达。O1–O14本地原始文件都重新匹配saved author source receipt的SHA256和Git blob。以下行号为本地1-based；网页工具压缩空行后的行号不同。

Nr/Sr CSV加载utterance、target_id、instance_type。`Scene_graph_parse`只把规范化caption传给parser；parser使用通用类别词表与句法，接口没有该行target_id或真实target class。parsed node0形成名词、修饰语、代词、关系词跨度，其他对象名词形成扣分跨度；跨度映射到token、归一化并累加到root slot0。Nr/Sr用这条路径。按anno真实类别名找文本的另一函数用于ScanNet detection分支，不能混称为Nr/Sr评分。O4:598–677、955–961、1209–1236、1369–1458；O12:62–73；O13:64–86。[作者dataset](https://github.com/AaNnWwTt/PV-Ground/blob/262e2592589baec7bb83a0d46aae6542d4ccedfb/src/joint_det_dataset.py)。

`_parse_gt`将主名词map正值二值化，其他组件保留归一化/累加值；only_root只保留slot0主名词map和root GT框。O2:535–553。两条score都将名词、修饰语、代词、关系词分数相加，扣除其他对象分数。

- `bbs / position alignment`：softmax(last_sem_cls_scores)与组件map汇总，O2:209–303。
- `bbf / semantic alignment`：query/text投影点积除0.07、token softmax，再与同一组件map汇总，O2:306–391。

这里semantic指embedding对齐，没有检查预测对象类别是否等于真实类别，也没有检查预测instance ID。相同query轴不保证两条score选出同一个query。[作者evaluator](https://github.com/AaNnWwTt/PV-Ground/blob/262e2592589baec7bb83a0d46aae6542d4ccedfb/src/grounding_evaluator.py)。

`target_id`从scan成员点构造slot0的GT框/GT mask，也参与distractor/分组元数据。root ranking不读取它进行对象ID选择，没有按该行GT target类别约束候选。辅助伪框代码确实读取GT目标位置；但auxi map/slot1在only_root评分中不构成额外目标得分：slot1被切掉，实际root score不加入计算出的auxi分数。这一判断只覆盖root评估路径，不声称训练无需GT。O4:703–736、778–811、680–699、1030–1062；O2:357–389、548–550。

## 3. butd_cls候选合法性与连续框命中

O4:738–774从scene所有合法类别对象生成框/mask（最多132槽）；butd_cls将这些GT场景框用于允许的proposal输入，类别值来自发布的`data/cls_results.json`，不使用本行GT target类别填写。O3:113–124/164–167开启box stream并只传text、场景框/预测类别等；O6:375–388编码这些框/类别。它提供作者允许的候选输入，没有把该表达的答案instance ID作为选择结果。

O2:240–249/337–346检查每个query预测框是否与任意合法scene框的最大IoU严格大于0.25，然后在283–284/387将分数乘0/1。它没有绑定scene object ID，没有替换成GT候选框，没有按target class筛选。**原代码是乘mask，不是严格剔除非法query**；作者复现不能未经证明换成另一过滤语义。

之后按score排序query，仍取它自身的预测center/size框。top1 correctness为`IoU(query predicted box, root GT box)>t`，t=0.25/0.5，严格大于。bbs计数O2:286–303，bbf计数485–491；O7:35–75为AABB IoU。这个标准没有对象ID判定。

原始ReferIt3D dataset把目标对象插入context并保存target_pos；原官方referential evaluator判断argmax(logits)==target_pos，类别分类accuracy单独统计。这是对象选择判定，连续框IoU判定没有由源码证明与其等价。[ReferIt3D dataset](https://raw.githubusercontent.com/referit3d/referit3d/eccv/referit3d/in_out/pt_datasets/listening_dataset.py)、[ReferIt3D evaluator](https://raw.githubusercontent.com/referit3d/referit3d/eccv/referit3d/models/referit3d_net_utils.py)。

## 4. 当前文件不能自动称为official

C1=`C:/Users/gb/.codex_pvground_cs_20261002/src/grounding_evaluator.py`包含候选轴/selector/reranker分支；bbs通过bool valid集合排序并处理无合法候选，官方O2通过分数乘mask。C1:1643–1646、1681–1707、1708–1757；C3:61–83。这是实际源码差异，不能因默认关闭部分功能就假定whole evaluator执行等价。[project evaluator](C:/Users/gb/.codex_pvground_cs_20261002/src/grounding_evaluator.py:1616)。

当前prepared entry E1:211–217传root/IoU阈值/butd_cls；245–248计算两个score四个counter，260–267仅将last/bbs写主metrics/diagnostics。它不包含完整导入runtime；现有audit E4也保留完整author validation和paper metric mapping未证实。后来必须绑定实际执行的evaluator版本。本轮不查看远程部署、不修改固定bbs。

O8/O9 two-stage脚本启用butd_cls，O10/O11 single没有。Sr two/single的checkpoint literal均为ScanRefer，Nr single literal为无single后缀Nr文件；README要求先指定checkpoint_path。后续须明确传对应dataset/stage作者权重，不能照抄literal。two-stage Nr发布LFS SHA为`d2d9afaf9c293c54977f3555a46c7bb2a72d9f80163a3f602dd8426032d7fa5d`，Sr为`a4a14b0090947177a648703ad6de094246891d89174fffa56fe454f730dbe3dc`。这是资源metadata，不是本轮remote hash实测。[作者脚本](https://github.com/AaNnWwTt/PV-Ground/blob/262e2592589baec7bb83a0d46aae6542d4ccedfb/scripts/test_sr3d.sh)。

## 5. 最小后续验证与可接受的声明

仍缺 **Table2→实际脚本/commit→对应weight SHA→annotation/preprocessing→score/阈值→各rank原始hits/total→去重/补齐处理→paper表项** 的作者来源链。还有“论文Accuracy本来是否以REC025作为操作定义”的明确说明。数值吻合不能补这些空白；对象ID等价还需要作者实际身份输出/判定规则及实例对齐。这里不发明nearest-GT投影或新metric。

既有训练结束且资源/计划允许后，最低实测是对应作者checkpoint在原允许候选协议上对完整Nr7899/Sr17726唯一表达集验证。一次forward同一份query框、token logits、text/query投影、语言map、合法候选与GT，分别记录作者原生bbs和bbf的选择query以及各自REC025/050整数hits/total；保存expression/scene/target身份与GT框，CPU按原AABB函数recount。记录rank/world-size、sampler与真正counter分母，确认无丢失/重复。**两个score分别选择，绝不混合预测、挑dataset-specific score或替换当前固定bbs。** 验证发布数字可复现仍不会自动证明paper-counter来源或对象ID等价。

| 声明 | 可诚实接受的要求 |
|---|---|
| 对应author完整验证完成 | 对应weight hash、完整实际runtime/data/flags、确切唯一表达覆盖、两个counter独立整数recount；方法E0不是unmodified baseline |
| 完整方法优于匹配部署score的author control | 相同dataset/stage/candidate/data，固定bbs，比较本项目实际测得的author-bbs与完整方法bbs同阈值hits；保留既有REC050目标 |
| 数值高于Table2 | 固定bbs百分数严格高于62.1/68.9或51.3/56.5；必须限定为numeric exceedance，paper protocol mapping unresolved |
| 同协议超过paper Accuracy | 补齐作者表项到counter的来源链，并按被证明一致的metric/selection/denominator比较；目前不能验收 |
| 对象身份Accuracy超越baseline | 作者identity判定与aligned candidate-ID输出得到证明；native REC单独不能验收 |

author-bbf可以并列解释paper-compatible数值参考；author-bbs作为固定部署score control。它们必须分别命名、保留各自预测，不叫作相同metric/predictions，不建议方法切换score。

## 6. 来源绑定与访问范围

SOURCE_BINDING.json逐文件记录绝对路径、字节数、完整SHA256、decisive line ranges。O1–O14另有独立重算并匹配的author Git blob/commit；L1–L4匹配原download receipt与HF metadata blob。P1 text SHA=`700caa56e348ba5f9a4ee909ec087f8552cac137e3ddf42d74cf8f19f54a5083`；P2 CVF PDF SHA=`ad96d075ca310f5cc48d875074cf5dc6af594938904811be3d734df0416287d2`；O2 author evaluator SHA=`39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677`。其余完整hash在JSON。

公开ReferIt3D与PyTorch源码是web-visible primary text，没有冒称原始download byte hash。PV官方仓库/readme/pinned raw可读；CVF当前html403、HF当前web不可读，论文与日志结论来自既有hash-bound primary原始档案。未读取不到的issue回复推断作者意见。实际日志分母、paper映射和当前runtime仍未实测。
