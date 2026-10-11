"""Analyze saved native E2 results and prepare one estimated E3 observation."""
import ast
import datetime
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
task = root.parent.parent
receipt_path = root / 'NORMAL_source_conditioned_EPOCH2_COMPLETION.json'
observed = json.loads(receipt_path.read_bytes())
assert observed['controller_alive'] and observed['child_alive']
assert [r['epoch'] for r in observed['metrics']] == [0, 1, 2]
history = json.loads((task / 'normal_controls_20261010/C_OFF_TERMINAL_SUMMARY_20261011.json').read_bytes())
rows = []
for name, metrics in (
    ('historical_C_off', history['metrics'][:3]),
    ('current_Face_source_conditioned', observed['metrics']),
):
    for item in metrics:
        assert item['rows'] == 9508 and item['primary_score'] == 'last/bbs'
        rows.append(dict(model=name, epoch=item['epoch'], rows=9508,
                         hits025=item['hits025'], hits050=item['hits050'],
                         acc025_percent=item['hits025'] / 9508 * 100,
                         acc050_percent=item['hits050'] / 9508 * 100))
initial, first, second = observed['metrics']
delta_start = [second[k] - initial[k] for k in ('hits025', 'hits050')]
delta_first = [second[k] - first[k] for k in ('hits025', 'hits050')]
delta_history = [second[k] - history['metrics'][2][k] for k in ('hits025', 'hits050')]
assert delta_start == [-58, -295]
assert delta_first == [-26, 26]
assert delta_history == [4, 27]
analysis = dict(
    status='ACTUAL_FACE_EPOCH2_COMPLETE_BELOW_RETAINED_START_EPOCH3_RUNNING',
    actual_observed_cst=observed['observed_cst'],
    observation_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    rows=rows, delta_current_E2_from_E0=delta_start,
    delta_current_E2_from_E1=delta_first,
    delta_current_E2_from_historical_control_E2=delta_history,
    current_best_retained_hits=[5677, 4920], target_hits=[5705, 4945],
    gap_best_to_target=[28, 25], current_training_complete=False,
    paired_without_added_source_normal_started=False,
    historical_comparison_causal=False, only_single_seed=2027,
    degradation_cause_proven=False, formal_results_unaudited=True,
    formal_result_promoted=False, three_effective_modules_proven=False,
    full_goal_complete=False,
    conclusion='E2 strict hits recover26 fromE1 but remain295 belowE0. '
               'Historical C-off comparison is not an independent Face-source control. '
               'No degradation cause is established; finish the unchanged3epoch run.',
)
(root / 'EPOCH2_ANALYSIS.json').write_text(json.dumps(analysis, indent=2) + '\n')
table = ['| 模型 | 轮次 | @0.25命中／百分比 | @0.5命中／百分比 |',
         '|---|---:|---:|---:|']
for row in rows:
    table.append('| {model} | {epoch} | {hits025}／{acc025_percent:.4f}% | '
                 '{hits050}／{acc050_percent:.4f}% |'.format(**row))
notes = (
    '\n\n当前E2相对E0为−58／−295，相对E1为−26／+26。严格指标部分恢复，尚未保住起点。'
    '与历史C-off同轮相比+4／+27只作历史参照；本轮同容量来源隐藏控制尚未启动，不能归因于Face来源。\n\n'
    '指标为完整9508条、原生last/bbs的实际记录，未做新结果晋级或完整终态审计。'
    '保留最好5677／4920，60%／52%目标仍差28／25个命中；三项有效机制与Nr/Sr验证未完成。\n\n'
    '第3轮已经开始，保持原定配置和预算。BN对照只在本轮训练真正结束后串行执行；'
    '当前无新增GPU诊断，不认定BN、学习率或新增模块为下降原因。\n'
)
(root / 'EPOCH2_ANALYSIS.md').write_text('# 第二轮正式记录\n\n' + '\n'.join(table) + notes, encoding='utf-8')

tail = (root / 'normal_source_conditioned_epoch2_completion_observation/train_tail.txt').read_text(encoding='utf-8')
train = re.findall(r'(\d+)/4583 \[([0-9:]+)<([0-9:]+),\s*([0-9.]+)s/it\]', tail)
validation = re.findall(r'(\d+)/1189 \[([0-9:]+)<([0-9:]+),\s*([0-9.]+)s/it\]', tail)
step, elapsed, remaining, rate = train[-1]
vstep, velapsed, vremaining, vrate = validation[-1]
assert int(step) == 17 and remaining == '3:51:03'
assert int(vstep) == 1189 and velapsed == '26:40' and vremaining == '00:00'

def duration(value):
    return sum(int(part) * 60 ** i for i, part in enumerate(reversed(value.split(':'))))

train_seconds = duration(remaining)
validation_seconds = duration(velapsed)
expected = datetime.datetime.fromisoformat(observed['observed_cst']) + datetime.timedelta(
    seconds=train_seconds + validation_seconds + 60)
due = expected - datetime.timedelta(minutes=3)
estimate = dict(
    status='EPOCH3_BOUNDARY_ESTIMATED_FROM_ACTUAL_EPOCH3_PROGRESS',
    actual_observed_cst=observed['observed_cst'], saved_train_step=int(step),
    train_steps_total=4583, displayed_train_remaining_seconds=train_seconds,
    validation_seconds_estimate=validation_seconds,
    validation_estimate_basis='Actual completed E2 validation1189batches took26:40.',
    completion_write_margin_seconds=60,
    expected_next_complete_metric_cst=expected.isoformat(), due_cst=due.isoformat(),
    estimate_basis='Actual E3 step17/4583 at21:26:13 shows3:51:03 remaining; '
                   'plus actual E2 validation26:40 and60s write margin. '
                   'Firstread3minutes before estimated completion; early-step ETA is not a deadline.',
    neural_calls=0, training_restart=False, later_poll_seconds=240,
)
(root / 'EPOCH3_BOUNDARY_ESTIMATE.json').write_text(json.dumps(estimate, indent=2) + '\n')
destination = root / 'observe_normal_epoch3_boundary_authorized.py'
assert not destination.exists()
script = (root / 'observe_normal_epoch2_boundary_authorized.py').read_text(encoding='utf-8')
script = script.replace('EPOCH2_BOUNDARY', 'EPOCH3_BOUNDARY')
script = script.replace('_epoch2_boundary_observation', '_epoch3_boundary_observation')
script = script.replace('One observation at the estimated first-epoch boundary, not a fast poller.',
                        'One E3 boundary observation estimated from actual native progress.')
tree = ast.parse(script)
embedded = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'code' for target in node.targets)]
assert len(embedded) == 1
ast.parse(embedded[0])
destination.write_text(script, encoding='utf-8')
print(json.dumps(dict(status=analysis['status'], E2_hits=[second['hits025'], second['hits050']],
                      delta_from_start=delta_start, next_due_cst=due.isoformat(), neural_calls=0)))
