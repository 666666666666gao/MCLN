"""Bind read-only task-path source and GT-only saved-output capacity findings."""
import hashlib
import json
from pathlib import Path

root = Path('\\\\?\\' + str(Path(__file__).resolve().parent))
snapshots = root / 'span_lr_actual_pair_review_20261014' / 'snapshots'
stage = json.loads((root / 'span_lr_runtime_staging_20261013_v2' / 'SOURCE_STAGE_RECEIPT.json').read_bytes())
paths = {
    'models/pv_ground.py': snapshots / '058_pv_ground.py',
    'models/losses.py': snapshots / '055_losses.py',
    'mask_support_corrector.py': snapshots / '125_mask_support_corrector.py',
    'extremal_span_mixer.py': snapshots / '124_extremal_span_mixer.py',
    'native_mask_geometry.py': snapshots / '126_native_mask_geometry.py',
    'native_model_initialization.py': snapshots / '127_native_model_initialization.py',
    'native_root_bbs.py': snapshots / '128_native_root_bbs.py',
    'pvground_semantic_assignment.py': snapshots / '130_pvground_semantic_assignment.py',
    'selected_support_geometry_objective.py': snapshots / '135_selected_support_geometry_objective.py',
}
bindings = {}
for name, path in paths.items():
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == stage['source_files'][name]['sha256']
    bindings[name] = {'path': str(path), 'bytes': len(raw), 'sha256': digest}
capacity_path = root / 'selected_family_capacity_20261014' / 'RESULT.json'
capacity = json.loads(capacity_path.read_bytes())
assert capacity['neural_forwards'] == capacity['remote_queries'] == 0
output = root / 'task_path_and_family_capacity_20261014'
assert not output.exists()
output.mkdir()
record = {'scope': 'SOURCE_TRACE_AND_CLOSED_C_GT_ONLY_BOX_FAMILY_ANALYSIS',
    'source_stage_bound_paths': bindings,
    'capacity_result_path': str(capacity_path),
    'capacity_result_sha256': hashlib.sha256(capacity_path.read_bytes()).hexdigest(),
    'actual_successful_CPU_native_chunk': 'f91170', 'successful_CPU_exit_code': 0,
    'preserved_failed_attempt': str(root / 'selected_family_capacity_20261014_attempt1_size_assertion'),
    'failed_CPU_native_chunk': 'f2f8ef', 'failed_CPU_exit_code': 1,
    'new_NN_performance_claim': False, 'independent_review_pending': True,
    'active_model_training_source_changed': False, 'current_low_LR_NN_status_queried': False,
    'current_low_LR_new_trained_formal_accuracy': None, 'full_goal_complete': False}
(output / 'SOURCE_AND_EXECUTION_BINDINGS.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
lines = ['# 已保存输出的几何范围与原生训练职责', '',
    '这是当前已核验源码的只读路径检查，以及已结束C模型保存输出的GT离线分析。没有新模型精度、NN前向、更新或远程训练查询。当前低B学习率正常实验配置保持不变，仍等待08:15原观察器。', '',
    '## 训练路径事实', '',
    '- 当前模型在生成原生两路Mask后调用A，将修正值写回sp_last_pred_masks。原生criterion将它作为sp_pred_masks，按真实匹配索引计算Query Mask和融合Mask损失。A没有脱离原生Mask监督；直接GT Mask目标仍由原匹配分配，不能说每个未匹配候选都获得直接Mask目标，也不能说它们在全部路径完全无梯度。',
    '- native_mask_geometry保留原预测center/size张量供B使用，没有把粗框整体detach。B把正尺寸的两个端点按三个独立轴系数混合，中心和尺寸共用每轴系数。预测Mask成员极值路径是离散detach，概率/内容读取与粗框路径另行保留梯度。',
    '- 原生CE将未匹配候选赋给最后一个no-object token，权重0.1。G只替换最终Box与root IoU>0.5且未匹配的候选文本目标，并保留原eos权重0.1，不能写成与匹配前景权重1相同。G没有替换对比目标。',
    '- C按原native_root_bbs选择一个候选，排除全部原匹配，训练GT确认该候选自身Query Mask及融合Mask IoU>0.5时加入Box/GIoU目标；不要求当前Box IoU>0.5，也不改CE、对比或原匹配。',
    '- 因此，在当前Box IoU<=0.5的C资格候选上，额外几何目标可指向root，而原未匹配CE仍为no-object，对比仍为not-mentioned。这是明确的任务责任差异，尚不是梯度冲突、实现bug或下降主因的证明。一对一主任务与辅助任务也可能有意分工。此前大集合CE/对比一致性及归一化试验不能被忽略。',
    '- 原生对应Mask项使用raw Query logit>0.5生成离散伪目标，部署及C资格使用logit>0（即sigmoid概率>0.5）。这是源码阈值事实；是否有意采用更高确认阈值、是否实际伤害边界，尚未测量，不修改活动实现。', '',
    '## 三轴Box输出族的GT离线上界', '',
    '固定每轮保存的已选Query、原回归框及Mask端点，只允许现有B输出形式中的三个轴系数在[0,1]取任意值。该GT选择不能部署，也不意味着共享网络特征能学出它。它与使用全部256候选的上界不同。', '',
    '|轮次|实际@0.25|族上界@0.25|实际@0.5|族上界@0.5|双Mask合格、严格Box错误中可达/不可达|',
    '|---|---:|---:|---:|---:|---:|']
for report in capacity['epochs']:
    a, b = report['025'], report['050']
    lines.append('|E%d|%d|%d|%d|%d|%d/%d|' % (report['epoch'], a['actual_selected_hits'],
        a['GT_only_independent_axis_family_upper_bound_hits'], b['actual_selected_hits'],
        b['GT_only_independent_axis_family_upper_bound_hits'], b['support_good_box_bad_reachable'],
        b['support_good_box_bad_not_reachable']))
lines += ['',
    '强起点E0的同Query自由三轴系数最多可使5839/5379条过阈值；实际5677/4920。494条双Mask资格合格而严格Box错误中，271条能在现有两个端点的输出族内跨过0.5，223条不能。75条双Mask资格合格而@0.25错误中，27条可达、48条不可达。',
    '这些数字不承诺可涨459条，也不证明有限网络可拟合每个GT最优系数。当前正常训练还会改变端点、表示及选择；不能用此固定快照界宣称完整正常训练无法达到目标。', '',
    '## 计算依据与验证范围', '',
    '每轴的预测上下边界随该轴系数线性变化。在上下边界与GT上下边界交叉的四个事件及0/1之间，固定其他轴后，IoU是正分母的线性分式，导数符号恒定，最大值在事件端点。逐轴可将全局最优点移到这些事件而不降低IoU，因此最多检查6^3=216个组合。无变化的边界没有交叉事件；复制端点0不增加输出。',
    '实际保存数据GT最小体积约0.000120223m³，端点尺寸均为正。保存的float32预测经float64重算，每行满足 saved_final_IoU <= ideal_family_max_IoU + 1e-5；这不是两者数值相等，也不能声称保存结果每行都严格不超过理想实数界。没有距0.25/0.5阈值1e-5内的族上界案例，也没有相对保存实际结果的阈值损失。验证另含六个可解析案例及24个随机案例的独立13^3网格下界比较。随机与解析案例只是计算检查，不计入9508条数据成绩。',
    '首个CPU执行因把float32尺寸下限与float64精确1e-6比较而断言失败；原源码与失败记录保留。修正为该计算真正要求的正尺寸后重算完成。不是NN训练故障，也没有重启NN。',
    'GT来自保存的dataset root_gt_box列及已有来源审计，本次没有重读原始数据集GT、重算点Mask、重跑模型或读取全部256候选数组。源码路径和SHA见SOURCE_AND_EXECUTION_BINDINGS.json；独立复核尚待完成。']
(output / 'REPORT.md').write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))
print(json.dumps({'status': 'SOURCE_TRACE_AND_GT_FAMILY_REPORT_SAVED', 'stage_matched_sources': len(bindings),
    'output': str(output), 'NN_calls': 0, 'remote_training_queries': 0}))
