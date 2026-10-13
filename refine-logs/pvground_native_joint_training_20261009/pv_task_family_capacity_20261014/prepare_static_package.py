"""Package completed local-only analysis; no NN or remote progress reads."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
study = root / 'deployment_geometry_training_20261012'
audit = study / 'selected_box_family_audit_20261014'
packet = root / 'pv_task_family_capacity_20261014'
assert not packet.exists()
closure = json.loads((study / 'SELECTED_FAMILY_AUDIT_CLOSURE_20261014.json').read_bytes())
assert closure['verified_input_files'] == 47 and closure['verified_output_files'] == 28
assert closure['seal_sha256'] == 'dfa0d1d2362df562f5bc70f0d754979a1806041572cadc714da372b194776d75'
prior = json.loads((root / 'pv_initial_and_dedup_publication_20261014.json').read_bytes())
assert prior['section'] == '20.376.176' and prior['github_main_verified']

old_publisher = (root / 'publish_pv_initial_and_dedup_authorized_20261014.py').read_text(encoding='utf-8')
new_publisher = old_publisher.replace('pv_initial_and_dedup', 'pv_task_family_capacity')
new_publisher = new_publisher.replace('PV_INITIAL_AND_DEDUP', 'PV_TASK_FAMILY_CAPACITY')
new_publisher = new_publisher.replace('20.376.176', '20.376.177').replace('20.376.175', '20.376.176')
replacements = {
    "data=root/'pv_task_family_capacity_publication_20261014'": "data=root/'pv_task_family_capacity_20261014'",
    "root/'pv_terminal_span_lr_publication_20261014.json'": "root/'pv_initial_and_dedup_publication_20261014.json'",
    'Publish saved PV evidence and verified archive availability; no training/NN operations.': 'Publish sealed selected-family/source analysis; no training/NN operations.',
    '低B学习率完整起点与重复权重去重（2026-10-14）': '已选候选几何范围与训练职责复核（2026-10-14）',
    'Record complete PV E0 and equal model checkpoint deduplication': 'Record selected box family capacity and native training targets',
    'Lower_B_LR_complete_E0_5677_4920_real_first_epoch_started': 'Historical_E0_retained_5677_4920_local_GT_only_analysis_no_new_NN_metric',
}
for old, new in replacements.items():
    assert new_publisher.count(old) == 1, old
    new_publisher = new_publisher.replace(old, new)
publisher_path = root / 'publish_pv_task_family_capacity_authorized_20261014.py'
assert not publisher_path.exists()
ast.parse(new_publisher)
publisher_path.write_bytes(new_publisher.replace('\r\n', '\n').encode('utf-8'))

sources = {
    'QUALIFIED_DIAGNOSTIC_REPORT.md': study / 'task_path_and_family_capacity_20261014' / 'REPORT.md',
    'TASK_SOURCE_EXECUTION_BINDINGS.json': study / 'task_path_and_family_capacity_20261014' / 'SOURCE_AND_EXECUTION_BINDINGS.json',
    'BOX_FAMILY_CAPACITY.json': study / 'selected_family_capacity_20261014' / 'RESULT.json',
    'selected_family_capacity.py': study / 'selected_family_capacity_20261014.py',
    'generate_task_path_report.py': study / 'record_task_trace_and_capacity_20261014.py',
    'close_local_audit.py': study / 'close_selected_family_audit_20261014.py',
    'AUDIT_CLOSURE.json': study / 'SELECTED_FAMILY_AUDIT_CLOSURE_20261014.json',
    'EXPERIMENT_AUDIT.md': audit / 'EXPERIMENT_AUDIT.md',
    'EXPERIMENT_AUDIT.json': audit / 'EXPERIMENT_AUDIT.json',
    'REVIEW_COMPLETE.json': audit / 'REVIEW_COMPLETE.json',
    'INDEPENDENT_RECOUNT.json': audit / 'INDEPENDENT_RECOUNT.json',
    'SOURCE_TASK_PATH_REVIEW.md': audit / 'SOURCE_TASK_PATH_REVIEW.md',
    'MATHEMATICAL_REVIEW.md': audit / 'MATHEMATICAL_REVIEW.md',
    'INPUT_REVISION_RECORD.json': audit / 'INPUT_REVISION_RECORD.json',
    'independent_checks.py': audit / 'independent_checks.py',
    'rational_boundary_checks.py': audit / 'rational_boundary_checks.py',
    'SUPPLEMENTAL_CHECKS.json': audit / 'SUPPLEMENTAL_CHECKS.json',
    'RATIONAL_BOUNDARY_CHECKS.json': audit / 'RATIONAL_BOUNDARY_CHECKS.json',
    'EXECUTION_RECEIPTS.json': audit / 'EXECUTION_RECEIPTS.json',
    'FIRST_CPU_FAILURE.json': study / 'selected_family_capacity_20261014_attempt1_size_assertion' / 'FAILURE.json',
    'FIRST_CPU_FAILED_SOURCE.py': study / 'selected_family_capacity_20261014_attempt1_size_assertion' / 'FAILED_SOURCE.py',
    'NEXT_E1_OBSERVER_OWNER.json': study / 'OBS_lower_span_lr_e1_near_complete_OWNER.json',
    'prepare_static_package.py': Path(__file__),
    'publish_static_package.py': publisher_path,
}
snapshots = study / 'span_lr_actual_pair_review_20261014' / 'snapshots'
for name in ('058_pv_ground.py', '055_losses.py', '125_mask_support_corrector.py',
             '124_extremal_span_mixer.py', '126_native_mask_geometry.py',
             '127_native_model_initialization.py', '128_native_root_bbs.py',
             '130_pvground_semantic_assignment.py', '135_selected_support_geometry_objective.py'):
    sources[name] = snapshots / name
packet.mkdir()
bindings = {}
for name, source in sources.items():
    raw = Path('\\\\?\\' + str(source.resolve())).read_bytes()
    (packet / name).write_bytes(raw)
    bindings[name] = dict(source=str(source), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    if name.endswith('.py'):
        ast.parse(raw.decode('utf-8-sig'))
(packet / 'PACKAGE_SOURCE_BINDINGS.json').write_bytes((json.dumps(bindings, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))

summary = dict(closure)
summary.update(status='SEALED_LOCAL_GT_FAMILY_AND_SOURCE_DIAGNOSTIC_NO_NEW_TRAINED_METRIC',
    observer_pid=26676, observer_native_session=11018, observer_functions_cell='457',
    MCLN_formal_training_launched=True, MCLN_formal_training_complete=True,
    metric_promoted=False, released_weight_bytes_this_publication=0,
    raw_rows_and_independent_CSVs_retained_locally_not_copied_in_publication=True)
(packet / 'CLOSEOUT.json').write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
notes = '''本节新增已完成的只读训练路径核对，以及闭合C模型保存输出的GT离线几何范围计算。没有新NN前向、优化器更新、验证精度、早期训练查询或活动源码修改。

### 当前目标和训练保持不变

同一9508条ScanRefer模型目标为60.0%／52.0%（至少5705／4945命中），再建立三个有直接增益证据的实质机制，随后验证对应作者Nr3D／Sr3D权重并分别训练同一完整方法。最好仍是5677／4920，即59.7076%／51.7459%，还差28／25条。本节不更新成绩，也不把调学习率当作创新。

低B学习率正常组原控制器432868／子进程432871保持原配置。最近实际远端观察仍为03:20:49：完整E0为5677／4920，第1轮日志35／4583。这不是现在重新读到的进度。唯一原观察器26676／native11018，由functions457等待08:15；在此节点前不查询远端训练、日志、epoch、GPU进度，不另起观察器或训练。Acc@0.5连续两次逐轮下降停止，持平或回升重置。

### 可确认的训练路径事实

A修正的Query Mask写回原生输出并进入真实匹配索引下的Query／融合Mask监督。B保留原粗框与概率／内容路径的梯度连接，硬预测成员极值仍离散detach。不能据此说所有输出的实际梯度都非零。

G只替换未匹配、最终Box与root IoU>0.5候选的末层CE目标，仍保留eos权重0.1，不改对比目标。C至多为原bbs选中的一个未匹配、自身与融合Mask都获得训练GT确认的候选加入几何目标；不要求该Box已经合格，不改变原匹配、CE或对比。Box仍不合格的资格候选由此可能获得root几何目标，同时原语义目标仍为未匹配背景。这是源码任务责任差异，尚未证明梯度冲突、bug或下降主因；已有大集合一致性与归一化负实验不能被忽略。

原生对应Mask项的离散伪目标使用raw Query logit>0.5，部署及C资格使用logit>0（概率>0.5）。这是阈值事实，未测量实际影响，不在运行实验中修改。

### 固定同一已选Query的GT离线输出族

固定保存的已选Query、原回归框和Mask端点，只允许B形式中的三个轴系数自由取[0,1]值。GT用于选择系数，这不是可部署预测，也不是全部256候选的上界。

|闭合C快照|实际@0.25|族上界@0.25|实际@0.5|族上界@0.5|
|---|---:|---:|---:|---:|
|E0|5677|5839|4920|5379|
|E1|5679|5859|4752|5335|
|E2|5588|5793|4623|5208|

E0中494条自身及融合Mask均合格、所选Box严格不合格的表达，271条能在现有两端点三轴输出族中跨过0.5，223条不能。宽松阈值对应75条，27条可达、48条不可达。因此既有“当前参数化内仍有空间”，也有“固定端点范围不足”的错误。不能据此承诺增加459条，不能用固定快照界否定完整正常训练——正常训练会改变端点、表示和选择。

每轴上下边界随其系数线性变化。在边界与GT交叉的四个事件及0／1之间，固定其他轴的IoU为正分母线性分式，最大值可移到事件端点，最多检查6^3组合。保存float32框重算满足单侧`actual_IoU <= ideal_family_max + 1e-5`；不是数值相等或每行严格小于理想界。无阈值损失；E2有约9.69e-9的保存舍入超界。首个CPU尺寸断言失败及原源码已保留，修正计算实际所需的正尺寸条件后成功；没有NN故障或重启。

### 独立复核与适用范围

新鲜上下文审查完成于06:06:17；总体WARN、限定范围数值／数学PASS、源码路径PASS_SOURCE_ONLY，0阻断。复算28524条、142项确定性与9行精确有理数检查。最终47输入／28输出SHA及字节数已核验。请求Astra/max，实际路由UNATTESTED，same-family/provisional，不宣称另一模型已认证。封印26743字节，SHA256 `dfa0d1d2362df562f5bc70f0d754979a1806041572cadc714da372b194776d75`。

仅使用保存的dataset GT坐标并核对来源源码；没有重载原注释／点／Mask，没有重算完整256数组、实际梯度或冷恢复。三个epoch是同一seed2027轨迹，不是独立重复。源报告中“复核尚待完成”是封存前历史生成文本，最终完成状态以本节及REVIEW_COMPLETE为准；没有为了发布改写已封存报告。

发布包保留报告、计算代码、原失败、限定源快照、结果及审查。原始逐行文件和独立CSV仍留本地，未重复上传；包中清单不代表全部47／28文件均复制到发布目录。下一步仍是按08:15原观察器读取真实E1，依据正式增量、修复／破坏及停止规则决定后续，不重复计算或继续制造无关预备实验。
'''
(packet / 'CLOSEOUT.md').write_bytes(notes.replace('\r\n', '\n').encode('utf-8'))
assert len(list(packet.iterdir())) == len(sources) + 3
print(json.dumps(dict(status='LOCAL_SEALED_DIAGNOSTIC_PACKAGE_PREPARED',
    files=len(list(packet.iterdir())), curated_sources=len(sources),
    new_NN_calls=0, remote_progress_queries=0, publisher=str(publisher_path))), flush=True)
