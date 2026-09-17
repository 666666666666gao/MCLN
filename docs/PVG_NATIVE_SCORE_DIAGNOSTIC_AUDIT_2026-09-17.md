# 原生 CE 与 bbs 分数诊断：独立完整性审计

静态审阅时间：2026-09-17 16:24 CST；终态追加审阅：16:35 CST。审阅者：原 native Codex 独立子代理 `/root/terminal_diagnostic_audit`；技能：`experiment-audit`。本报告直接读取指定代码、计划、原生损失源码、v1 停止证据、v2 staged 文件及完整产物；未执行远程命令，未修改 driver 或实验代码。审阅者只写入本报告。

```yaml
review_independence: same-family
acceptance_status: provisional
overall_verdict: WARN
static_code_verdict: PASS_after_native_float32_correction
execution_blocker: none_identified_in_reviewed_v2
v1_status: stopped_before_diagnostic_result
v2_runtime_evidence: verified_complete
deterministic_export_checks: pass
evaluation_type: real_gt
evaluation_subtype: fixed_trained_output_local_logit_gradient_diagnostic
```

模型和 reasoning effort 身份以父代理原始 reviewer spawn 记录为准；本报告不声称跨模型家族验收。v2 已完整结束并通过独立导出值复核。整体 WARN 保留的是已见训练样本、固定 logit 局部方向和未重算原始 GT/梯度的范围限制，不表示运行未完成或出现数值矛盾。

下文仓库相对路径以 `C:/Users/gb/.codex_mcln_g0_20260905/` 为根；原生损失和 criterion 源码以 `D:/Program Files/UserCache/gb/codex/tmp/pvg_loss_20260917/` 为根，分别记为 `native/models/losses.py` 与 `native/main_utils.py`。

## 先前阻断、停止记录与修正

v1 将 logits 转为 double，但 targets 的 token maps 保持 float32。原生 `loss_pos_align` 的 `zeros_like(logits)` 因而创建 double `target_mask/target_sim`，而高级索引赋值的右侧仍是 float32：`native/models/losses.py:491-497`。审阅者用本地 PyTorch CPU 张量复现了 `Index put requires the source and destination dtypes match`；这是原代码的确定性 dtype 不兼容。

真实 v1 没有产生此异常的实验 traceback：其归档 `run.log:4-5` 止于加载 train 和文本解析，没有 batch/完成记录。`refine-logs/pvground_native_score_diagnostic_20260917_v1/stop_receipt.json:2-5` 记录 16:20:28.899461 CST 对 child 6883 的精确停止，`controller.exit:1=143`。应记录为发现静态阻断后提前停止，不能写成原生 CE 已实测崩溃、完成诊断或得到了梯度结果。

v2 选择直接保持原生 float32：`scripts/diagnose_pvground_native_score.py:179,192-210` 的 logits、标签运算、score weights 和 EOS dtype 一致，不修改原生损失源码；解析导数容差为 `1e-6`。旧 `double_selection` 改为 `formula_selection`，准确表示同 dtype 下“合并权重后求和”与原部署逐项求和的浮点计算顺序差异；部署选择仍来自原来的逐项 bbs 公式，见 `:216-223,233,275-276`。`:241-242` 补齐 matched/best 的分数 rank；其定义为 `1 + strictly_greater_count`，并列分数共享 rank，不冒称排序中的唯一位置。

## A-F 检查

| 项目 | 状态 | 直接证据与解释 |
| --- | --- | --- |
| A. Ground truth provenance | PASS | 真实 dataset GT 在模型 forward 后供 matcher、CE 与 IoU 分析；模型输入只含原点/体素、文本、检测框/类别与 superpoint。见诊断脚本 `:163-178`。没有模型生成 GT。 |
| B. Score normalization / formula | PASS after correction | 实际调用原生 CE；目标质量未强制归一，解析式保留 `target_sim.sum`。score 的 softmax 及加减项是原 bbs 规则，没有用模型自身最大值归一化性能。见 `:185-214,217-223`。 |
| C. Result existence | PASS within inspected artifacts | v1 停止产物保留；v2 input selection、rows、NPZ、diagnostic、完整日志和 exit 0 齐全，全部关联 SHA 与导出数值重计一致。此前 pending 状态由下文终态追加取代。 |
| D. Dead code / hidden training | PASS for v2 executed path | matcher 和 loss_pos_align 实际被调用，梯度只对 detached logits 求取；model forward 在 eval/no_grad 下；无 optimizer、parameter backward 或 checkpoint 保存。16 批及完整退出支持结束时全 state_dict 与所有参数 `.grad is None` 的断言通过。见 `:106,168-190,197,254-258` 和 v2 `run.log:7-23`。 |
| E. Scope | WARN / fixed ceiling | 128 个 fit 物理场景各第一条表达，seed 2027，batch 8，共 16 次 forward；已见训练场景、单 checkpoint、固定匹配下的最后层 CE 局部 logit 方向。不能推出参数更新、总损失改进、正式精度或泛化。见计划 `:7,11-15,21-27` 和脚本 `:138-153,274`。 |
| F. Evaluation type | real_gt diagnostic | GT 框、mask、token labels 来自数据集；局部方向是分析量，没有模拟或执行真实训练步。另行本地 CPU 公式 fixture 仅属合成公式验证，不计入该 real_gt 运行。 |

## 原生 matcher 与 CE 构造

`native/main_utils.py:268-280` 创建 `HungarianMatcher(1, 0, 2, args.use_soft_token_loss)` 和 EOS=0.1 的 SetCriterion。matcher 的 mask weight=0.0002，soft-token cost 为负 token probability 与 positive_map 的乘积；联合 GIoU 与 mask 项后执行一对一匹配，见 `native/models/losses.py:290-298,324-353,363-390`。诊断以原 float32 预测先匹配，未用 bbs 所选 Query 或 GT-best Query 替代训练匹配；传入 raw boxes 后，由原生 box helper 在 GIoU 路径内部执行尺寸 clamp，见 native `:35-45`。报告应保持“原 matcher 规则”，不能写成没有任何 clamp。

ScanRefer 的匹配目标为 `t=0.6*positive+0.2*modify+0.2*pron+0.1*relation`；未匹配 Query 的目标为末 token one-hot，EOS 权重为 float32 0.1，匹配项为 1。CE 中的 `other_entity_map` 只参与未使用的 target_mask，没有直接负项；部署 bbs 则包含 other-entity 减项。见 native `:478-511`。给定匹配，CE 的 logits 梯度为 `eos/num_boxes * (softmax(z)*sum(t)-t)`；target 熵项不依赖 z，因此无梯度贡献。目标总质量不固定为 1，保留 sum(t) 是必要的。

score 权重 w 固定时，`s=sum(softmax(z)*w)` 的梯度为 `p*(w-s)`，与诊断 `:198` 一致。`v=-grad(CE) dot grad(s)` 只表示固定匹配、当前 logit 位置上单独沿最后层 CE 下降的瞬时分数方向。原总损失还对 ScanRefer CE 施加 `0.5/(6+1)` 系数，并含其他 decoder 层、框、mask 和对齐项，见 native `:943-955`。此正缩放不改变单项方向符号，但当前 velocity 大小不能被解释为实际优化器更新量。

## 固定选择与运行路径

诊断 `:114-150` 重建并核对原 fit/holdout 划分，按 fit 原始顺序选前 128 个不同物理场景，每场景一条表达，显式排除 holdout 行和 holdout 物理场景；在第一次 forward 前保存 `input_selection.json`。GT 不参与这次样本选择，也没有先按预测好坏筛样本。当前代码会解析 train annotation 集合，但只取预固定 fit 子集进行诊断；“不读 holdout 输出”不应扩大为“未接触任何 holdout annotation 元数据”。

prepare `:11-12,30` 指向独立 v2 目录，并要求 v1 `controller.exit==143`；`:40-44,62` 检查 GPU 空闲、磁盘和已有 GPU flock。`:71-79` 禁止覆盖既有目录，逐文件写入并回读比对；controller 的唯一命令是该 v2 `diagnose.py`。observer `scripts/observe_pvground_native_score_diagnostic.py:9-23` 只下载 v2 产物、核对三个结果文件的 SHA，不会重启任务。它能在尚无 exit 时打印结果，所以最终完成判断仍须独立核对 exit 0、完成标记和真实 16 批进度，不能只看 observer 打印。

v2 staged 七项文件的本地 SHA 均匹配 `spec.json:17-24`。`launch.json:2-8` 记录 16:21:56.845934 CST 的 controller 7113 和零训练/零正式行；这属于启动证据，不是完成证明。

## 本地公式 fixture 验证（不计入实验结果）

审阅者以 AST 读取并原样执行 `native/models/losses.py` 的 SetCriterion 类，在本地 PyTorch `2.14.0+cpu` 上构造 2 批、8 Query、16 token、共 5 个匹配目标的 float32 合成 fixture，同时包含未匹配 EOS 和非单位质量目标。未导入或运行网络，没有 optimizer，没有真实数据集行。

| 项目 | 结果 |
| --- | --- |
| logits / token maps / EOS dtype | 均为 torch.float32 |
| 原生 loss_pos_align 对比解析 CE 梯度最大绝对误差 | `1.1175870895385742e-08` |
| score autograd 对比解析导数最大绝对误差 | `0.0` |
| 当前容差 | `1e-6`，此 fixture 通过 |
| 网络 forward / optimizer steps / 真实 dataset rows | `0 / 0 / 0` |

这确认修正消除已复现的 dtype 阻断，并验证本地公式实现；不能替代远程 128 行梯度误差、实际输出类型、GPU 数值行为或终态恢复的运行证据。

## 已核验版本标识

| 文件 | SHA256 |
| --- | --- |
| v1 diagnose.py | `6855b69f90eefe2da2bb2ecc21f0c1a9c53db2488c206988130fa6d36b86126b` |
| v2 diagnose.py / 当前诊断源脚本 | `1d51a20431793062b041745a39965515597e3c75f3ea6fec1241792b613ad31c` |
| prepare_pvground_native_score_diagnostic.py | `507e4eafc2990599b8dffb803cebf2118590517696e2ec5948f29871d4bfdec4` |
| observe_pvground_native_score_diagnostic.py | `742c2dfac7c5b98bb569209cde87684331ecdeccfcb1dcb852ef5a7cd3e37d8c` |
| 预固定计划 / v2 plan.md | `84a84dacf65f3f4749b79b2c6da39ee0fcb70f5f73fb1d2be16b296a08778aff` |
| 原生 models/losses.py | `920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de` |
| 原生 main_utils.py | `14bd101eb78ec3d729968aff0a975684efa8388574583ca9e4b0f512877b4f5c` |
| 固定 D terminal 标识（本审阅未重新读取远程二进制） | `ce03188965491a82bcb1c5a6d26f590d3a243a01985457f220d5503c75b2fcf5` |

## 终态追加核验（2026-09-17 16:35 CST）

下述产物均位于 `refine-logs/pvground_native_score_diagnostic_20260917_v2/`。`run.log:7-22` 恰有顺序 1–16 的批次记录，累计行数为 8–128；`:23` 唯一完成记录与 diagnostic JSON 完全一致，`controller.exit:1=0`。完成时间为 16:29:19.751653 CST，耗时 443.5821259022 秒。`diagnostic.json:2-22` 记录 float32、解析误差最大 `3.725290298461914e-08 < 1e-6`、全部参数梯度为空、状态未变及严格恢复通过；这些标志有已审阅代码的实际成功退出支持。训练更新、正式行和新 checkpoint 均为 0。v1 停止历史与 float32 修正未被覆盖。

审阅者另写只读 NumPy 核验，未运行执行者的写文件分析入口，也未执行模型：直接核对 128 个唯一 row_id、128 个不同物理场景、与预存 input selection 相同的顺序/scan_id、256 个 NPZ 数组、32,768 个候选及其 finite/shape/dtype。原部署选择和 formula 选择均是每行唯一最大分数，128 行全部一致；两种浮点求和得到的分数最大差为 `1.1920928955078125e-07`。逐行重核 root 唯一匹配、target 编号、selected/matched/best 的 score/IoU/velocity、排名、float32 margin 差值、逐批目标数与 CE 记录。独立重算所有阈值计数后，同时匹配 diagnostic 和既有 recount；没有把 recount 的 PASS 字段本身当作重计依据。

| 导出值独立重计 | IoU > 0.25 | IoU > 0.50 |
| --- | --- | --- |
| 原部署选择合格 | 114 / 128 | 106 / 128 |
| Hungarian matched-root 合格 | 128 / 128 | 127 / 128 |
| raw GT-best 候选合格 | 128 / 128 | 127 / 128 |
| 有合格候选但部署选择失败 | 14 | 21 |
| 上述错误中 best 相对 selected 的 margin velocity > 0 | 14 / 14 | 21 / 21 |
| 上述错误中 matched-root 相对 selected 的 margin velocity > 0 | 14 / 14 | 21 / 21 |
| 上述错误中 matched-root 自身合格 | 14 / 14 | 21 / 21 |
| 完全未匹配的合格候选 / 其中 score velocity < 0 | 8259 / 8258 | 6802 / 6802 |

证据：`diagnostic.json:23-45`、`recount.json:5-100` 及独立 NPZ 计算。两阈值错误集合内最小 best margin velocity 均为 `0.0006657742196694016`，最小 matched margin velocity 均为 `0.004097750876098871`。两阈值下合格且匹配其他 target 的候选数均为 0；合格 root 匹配候选均未出现负 score velocity。

结果不支持“在当前样本和固定输出处，原生 CE 局部方向把已覆盖的排序错误推向更坏方向”这一假设。大量合格但完全未匹配的重复候选降分，单独不足以反驳 CE：实际合格 matched-root 的竞争 margin 在所有 covered errors 中都在增大。这个结果没有检验真实共享参数更新后的行为，也不证明 CE 永远合适；本轮没有提供据此改 loss、matcher、部署 score 或增加 quality head 的实验证据。

### 两个按固定输入顺序选取的说明行

示例分别取输入顺序中第一个 selected IoU≤0.25、以及第一个 0.25<selected IoU≤0.5 且 matched-root 通过相应阈值的行；只用于解释统计，不追加筛样本或模型执行。

| 行 / 场景 | 部署 Query：score / IoU / velocity | matched-root Query：score / IoU / rank / velocity | matched-root − selected 的 margin velocity |
| --- | --- | --- | --- |
| 173 / scene0001_00 | Q88：0.807038 / 0.055228 / -0.001529794 | Q247：0.080348 / 0.885319 / 14 / +0.011403185 | +0.012932980 |
| 848 / scene0010_00 | Q4：0.154300 / 0.494085 / -0.000501469 | Q85：0.060291 / 0.525348 / 55 / +0.009871874 | +0.010373343 |

这两行的 matched-root 也恰为 GT-best。它们当前分数落后但局部 CE 方向在提高相对分数；不能把这个方向等同于已经修复错误。证据：`rows.json:31-58,205-232`，已与 NPZ 对应数组独立逐项核对。

### 完成产物哈希与复核边界

| 产物 | 直接重算 SHA256 |
| --- | --- |
| input_selection.json | `d2fbed7e9aa4546af703f90cff47928e7950c783be03f49c9ed916b66192ca22` |
| rows.json | `42e9bff1058c09d01e831c71a354163de6f677831f71e8ee3dc17d28f8fa1db1` |
| candidate_values.npz | `7d311074f1b1cd3f18a815ba0bd03e082f7e82c82d48d81b4b7739d5429b00ed` |
| diagnostic.json | `28be9e74ae279984f53c026ee2dee5c153e3cb9e4d551f69de62578349095b77` |
| recount.json | `2c674b566ef5758d2c69200a98c33e1a163c868c789835e3d80259e8bfb7210c` |
| run.log | `7e9f246467799fafc5214232a23d65dcd275c3a21cbc72c96de5956dc7530029` |

`good_unmatched_*` 的代码语义是完全未匹配（matched_target=-1），不包括匹配其他目标的候选，见脚本 `:245-248`；本追加已分别重计三组。best-IoU 是 GT 辅助的 raw 候选上界，不能当作可部署选择。当前 NPZ 不保存 logits、token maps、原始框或完整 CE 梯度，不能仅凭它独立重算 matcher、几何 IoU 或解析梯度；本次独立核验是导出值之间的索引、排序、差值与汇总一致性，不是 raw GT 重算。公式正确性另有源码审阅、本地合成 fixture 和实际 16 批解析误差证据，各证据层级分别保留。所有结论限于已训练 D 的这 128 个训练样本，不改变已有正式晋级条件。确定性产物核验通过；语义审阅仍为 same-family / provisional。
