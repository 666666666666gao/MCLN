"""Use the completed original observation; schedule one E2 boundary read."""
import ast
import base64
import datetime
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
observation = json.loads((root / 'NORMAL_FIRST_OBSERVATION.json').read_bytes())
assert json.loads((root / 'NORMAL_FIRST_OBSERVATION_EXIT.json').read_bytes())['exit_code'] == 0
assert observation['controller_alive'] and observation['child_alive']
assert [(r['epoch'], r['rows'], r['hits025'], r['hits050']) for r in observation['metrics']] == [(0,9508,5677,4920),(1,9508,5644,4604)]
assert all(r['primary_score'] == 'last/bbs' and r['same_complete_model'] for r in observation['metrics'])
raw_observation = json.loads((root / 'NORMAL_FIRST_OBSERVATION_STDOUT.json').read_bytes())
for item in raw_observation['files']:
    raw = base64.b64decode(item['base64'])
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    assert (root / 'normal_first_observation' / item['name']).read_bytes() == raw
metrics_path = next(item['name'] for item in raw_observation['files'] if item['name'].endswith('/native_metrics.jsonl'))
assert [json.loads(line) for line in (root / 'normal_first_observation' / metrics_path).read_text().splitlines()] == observation['metrics']
tail = (root / 'normal_first_observation/train_tail.txt').read_bytes()
assert base64.b64decode(raw_observation['train_tail_base64']) == tail
lines = re.sub(r'\x1b\[[0-9;]*m', '', tail.decode()).splitlines()

def seconds(text):
    parts = [int(part) for part in text.split(':')]
    assert len(parts) in (2, 3)
    return sum(part * 60 ** index for index, part in enumerate(reversed(parts)))

progress = [re.search(r'(\d+)/4583 \[([^<]+)<', line) for line in lines]
progress = [item for item in progress if item]
step, elapsed = int(progress[-1].group(1)), seconds(progress[-1].group(2))
assert step == 269 and elapsed == 817
assert any('Train: [2][256/4583]' in line for line in lines)
evaluation = [re.search(r'1189/1189 \[([^<]+)<', line) for line in lines]
evaluation = [item for item in evaluation if item]
assert evaluation
validation_seconds = seconds(evaluation[-1].group(1))
observed = datetime.datetime.fromisoformat(observation['observed_cst'])
eta = observed + datetime.timedelta(seconds=(4583-step)*elapsed/step + validation_seconds)
due = eta - datetime.timedelta(seconds=300)
assert due > datetime.datetime.now().astimezone()
metrics = [dict(r, acc025_percent=100*r['hits025']/r['rows'], acc050_percent=100*r['hits050']/r['rows']) for r in observation['metrics']]
summary = dict(status='C_OFF_FIRST_ORDINARY_JOINT_EPOCH_OBSERVED_NOT_TERMINAL',
    observed_cst=observation['observed_cst'], metrics=metrics,
    epoch1_delta_vs_own_E0=[-33,-316], epoch1_delta_vs_historical_C_on_E1=[-8,8],
    historical_C_on_E1_counts=[5652,4596], historical_C_on_is_not_never_C_parent_control=True,
    epoch2_completed_updates=step, updates_per_epoch=4583, epoch2_elapsed_seconds=elapsed,
    measured_epoch1_validation_seconds=validation_seconds,
    source_receipts_and_logged_counts_verified=True, formal_results_unaudited=True,
    original_observer_native_session=21631, original_native_session_consumed_exit_code=0,
    observation_sha256=hashlib.sha256((root/'NORMAL_FIRST_OBSERVATION.json').read_bytes()).hexdigest(),
    train_tail_sha256=hashlib.sha256(tail).hexdigest(),
    selected_mask_supervision_disabled=True, best_retained_counts=[5677,4920],
    normal_optimizer_updates_observed=True, terminal_result=None,
    configuration_changed=False, three_effective_contributions_proven=False, full_goal_complete=False)
plan = dict(status='SINGLE_NEXT_OBSERVATION_PLANNED_FROM_MEASURED_RATE',
    due_cst=due.isoformat(), estimated_epoch2_full_validation_end_cst=eta.isoformat(),
    advance_seconds=300, later_poll_seconds=240,
    basis='Actual E2 269/4583 in817seconds, plus the observed completed E1 validation duration; completion is an estimate.',
    original_first_observer_closed=True, original_first_native_session=21631,
    observations=1, new_neural_calls=0, training_restart=False, source_changed=False)
for name, value in [('NORMAL_FIRST_OBSERVATION_SUMMARY.json', summary), ('NORMAL_NEXT_OBSERVATION_PLAN.json', plan)]:
    path = root / name
    assert not path.exists()
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source = (root / 'observe_normal_planned_authorized.py').read_text(encoding='utf-8')
old_due = "due=datetime.datetime.fromisoformat(launch['time_cst'])+datetime.timedelta(seconds=launch['first_observation_seconds'])"
assert old_due in source
source = source.replace(old_due, "plan=json.loads((root/'NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())\nassert plan['original_first_observer_closed']\ndue=datetime.datetime.fromisoformat(plan['due_cst'])")
source = source.replace('NORMAL_FIRST_OBSERVATION','NORMAL_EPOCH2_OBSERVATION').replace('normal_first_observation','normal_epoch2_observation').replace('NORMAL_OBSERVER_OWNER','NORMAL_EPOCH2_OBSERVER_OWNER')
source = source.replace('SINGLE_PLANNED_NORMAL_OBSERVER_WAITING','SINGLE_PLANNED_EPOCH2_OBSERVER_WAITING')
source = source.replace("estimate_basis='Prior ordinary three-epoch run took14.63349hours including initial validation; first node fivehours after launch, then use measured progress'", "estimate_basis=plan['basis']")
source = source.replace('if remaining>0:time.sleep(remaining)', "while remaining>0:\n time.sleep(min(300,remaining))\n remaining=(due-datetime.datetime.now().astimezone()).total_seconds()")
source = source.replace('One observation at the estimated first-epoch boundary, not a fast poller.','One C-off E2 boundary observation using the actual E2 rate and completed E1 validation.')
ast.parse(source,feature_version=(3,7))
path = root / 'observe_c_off_epoch2_planned_authorized.py'
assert not path.exists()
path.write_text(source,encoding='utf-8')
print(json.dumps(dict(status=summary['status'], metrics=metrics, epoch2_updates=step,
    measured_validation_seconds=validation_seconds, next_due_cst=due.isoformat(), estimated_end_cst=eta.isoformat(),
    observer_started=False, new_remote_queries=0, full_goal_complete=False)))
