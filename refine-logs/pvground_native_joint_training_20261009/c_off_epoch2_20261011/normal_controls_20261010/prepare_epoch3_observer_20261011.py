"""Summarize completed E2 and schedule one final-epoch observation."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
observation_path = root / 'NORMAL_EPOCH2_FOLLOWUP_OBSERVATION.json'
observed = json.loads(observation_path.read_bytes())
assert json.loads((root / 'NORMAL_EPOCH2_FOLLOWUP_OBSERVATION_EXIT.json').read_bytes())['exit_code'] == 0
assert observed['controller_alive'] and observed['child_alive']
assert [entry['epoch'] for entry in observed['metrics']] == [0, 1, 2]
assert all(entry['rows'] == 9508 and entry['primary_score'] == 'last/bbs'
           and entry['same_complete_model'] for entry in observed['metrics'])
for entry in observed['files']:
    path = root / 'normal_epoch2_followup_observation' / entry['name']
    raw = path.read_bytes()
    assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
tail_path = root / 'normal_epoch2_followup_observation/train_tail.txt'
tail = tail_path.read_text(encoding='utf-8')
validation = re.findall(r'1189/1189 \[(\d+):(\d+)[^\]]*\]', tail)
progress = re.findall(r'(\d+)/4583 \[(\d+):(\d+)<(\d+):(\d+):(\d+),\s*([0-9.]+)s/it\]', tail)
assert validation and progress
validation_seconds = int(validation[-1][0]) * 60 + int(validation[-1][1])
done, elapsed_minutes, elapsed_seconds, left_hours, left_minutes, left_seconds, rate = progress[-1]
assert int(done) == 67
remaining_train = int(left_hours) * 3600 + int(left_minutes) * 60 + int(left_seconds)
at = datetime.datetime.fromisoformat(observed['observed_cst'])
estimate = at + datetime.timedelta(seconds=remaining_train + validation_seconds)
due = estimate - datetime.timedelta(seconds=300)
metrics = [dict(entry, acc025_percent=entry['hits025'] * 100 / 9508,
                acc050_percent=entry['hits050'] * 100 / 9508) for entry in observed['metrics']]
assert (metrics[2]['hits025'], metrics[2]['hits050']) == (5615, 4598)
summary = dict(status='C_OFF_SECOND_COMPLETE_NORMAL_EPOCH_OBSERVED_NOT_TERMINAL',
    observed_cst=observed['observed_cst'], metrics=metrics,
    epoch2_delta_vs_own_E0=[5615 - 5677, 4598 - 4920],
    epoch2_delta_vs_own_E1=[5615 - 5644, 4598 - 4604],
    historical_C_on_E2_counts=[5552, 4552],
    epoch2_delta_vs_historical_C_on_E2=[5615 - 5552, 4598 - 4552],
    historical_comparison_is_interim_not_never_C_parent_ablation=True,
    epoch3_completed_updates=int(done), updates_per_epoch=4583,
    epoch3_elapsed_seconds=int(elapsed_minutes) * 60 + int(elapsed_seconds),
    epoch3_logged_remaining_train_seconds=remaining_train,
    measured_epoch2_validation_seconds=validation_seconds,
    gpu_resource_csv=observed['gpu_resource_csv'],
    disk_free_bytes=observed['disk_free_bytes'], weight_sizes=observed['weight_sizes'],
    best_retained_counts=[5677, 4920], selected_epoch=0,
    selected_epoch_scope='retained_parent_not_new_normal_training_gain',
    observation_sha256=hashlib.sha256(observation_path.read_bytes()).hexdigest(),
    train_tail_sha256=hashlib.sha256(tail_path.read_bytes()).hexdigest(),
    source_receipts_and_logged_counts_verified=True, formal_results_unaudited=True,
    original_followup_native_session=36654, original_followup_consumed_exit_code=0,
    original_followup_wait_cell=791, new_neural_calls=0, new_training_started=False,
    source_changed=False, weight_promoted=False, terminal_result=None,
    three_effective_contributions_proven=False, full_goal_complete=False)
(root / 'NORMAL_SECOND_OBSERVATION_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
plan = dict(status='SINGLE_FINAL_EPOCH_OBSERVATION_PLANNED_FROM_MEASURED_RATE',
    due_cst=due.isoformat(), estimated_epoch3_full_validation_end_cst=estimate.isoformat(),
    advance_seconds=300, later_poll_seconds=240,
    basis='Actual E3 67/4583 with3:43:13 logged remaining, plus actual completed E2 validation duration; estimate may change.',
    actual_completed_updates=int(done), logged_remaining_train_seconds=remaining_train,
    measured_epoch2_validation_seconds=validation_seconds,
    original_first_observer_closed=True, original_epoch2_followup_observer_closed=True,
    original_epoch2_followup_native_session=36654, original_epoch2_followup_exit_code=0,
    observations=1, new_neural_calls=0, training_restart=False, source_changed=False)
path = root / 'NORMAL_EPOCH3_OBSERVATION_PLAN.json'
assert not path.exists()
path.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
source = (root / 'observe_c_off_epoch2_followup_authorized_route.py').read_text(encoding='utf-8')
source = source.replace('NORMAL_EPOCH2_FOLLOWUP', 'NORMAL_EPOCH3')
source = source.replace('normal_epoch2_followup_observation', 'normal_epoch3_observation')
source = source.replace('NORMAL_EPOCH3_PLAN.json', 'NORMAL_EPOCH3_OBSERVATION_PLAN.json')
source = source.replace('SINGLE_PLANNED_EPOCH2_OBSERVER_WAITING', 'SINGLE_PLANNED_EPOCH3_OBSERVER_WAITING')
source = source.replace('One C-off E2 boundary observation', 'One C-off E3 final boundary observation')
ast.parse(source)
helper = root / 'observe_c_off_epoch3_authorized_route.py'
assert not helper.exists()
helper.write_text(source, encoding='utf-8')
print(json.dumps(dict(summary=summary, next_observation=plan)))
