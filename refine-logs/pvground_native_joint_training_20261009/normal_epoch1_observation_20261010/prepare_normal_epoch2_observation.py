"""Retain the first real observation and schedule one near the next epoch end."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
observation = json.loads((root / 'NORMAL_FIRST_OBSERVATION.json').read_bytes())
assert json.loads((root / 'NORMAL_FIRST_OBSERVATION_EXIT.json').read_bytes())['exit_code'] == 0
assert observation['controller_alive'] and observation['child_alive']
assert [row['epoch'] for row in observation['metrics']] == [0, 1]
assert all(row['rows'] == 9508 and row['primary_score'] == 'last/bbs' for row in observation['metrics'])
raw = (root / 'normal_first_observation/train_tail.txt').read_bytes()
lines = re.sub(r'\x1b\[[0-9;]*m', '', raw.decode('utf-8')).splitlines()
progress = re.compile(r'(\d+)/4583 \[([^<]+)<')
train_lines = [line for line in lines if progress.search(line)]
assert train_lines
last = progress.search(train_lines[-1])
step = int(last.group(1))

def seconds(value):
    parts = [int(part) for part in value.split(':')]
    assert len(parts) in (2, 3)
    return sum(part * 60 ** index for index, part in enumerate(reversed(parts)))

elapsed = seconds(last.group(2))
assert step == 436 and elapsed == 1286
assert any('Train: [2][384/4583]' in line for line in lines)
evaluations = [re.search(r'1189/1189 \[([^<]+)<', line) for line in lines]
evaluations = [match for match in evaluations if match]
assert evaluations
eval_seconds = seconds(evaluations[-1].group(1))
observed = datetime.datetime.fromisoformat(observation['observed_cst'])
fit_remaining = (4583 - step) * elapsed / step
eta = observed + datetime.timedelta(seconds=fit_remaining + eval_seconds)
due = eta - datetime.timedelta(seconds=300)
assert due > datetime.datetime.now().astimezone()
metrics = []
for row in observation['metrics']:
    metrics.append(dict(**row, acc025_percent=100 * row['hits025'] / row['rows'],
                        acc050_percent=100 * row['hits050'] / row['rows']))
summary = dict(status='FIRST_ORDINARY_JOINT_EPOCH_OBSERVED_NOT_TERMINAL',
    observed_cst=observation['observed_cst'], metrics=metrics,
    epoch1_hits_delta=[metrics[1]['hits025'] - metrics[0]['hits025'],
                       metrics[1]['hits050'] - metrics[0]['hits050']],
    epoch2_completed_updates=step, updates_per_epoch=4583,
    epoch2_elapsed_seconds=elapsed, mean_update_seconds=elapsed / step,
    measured_epoch1_validation_seconds=eval_seconds,
    observation_sha256=hashlib.sha256((root / 'NORMAL_FIRST_OBSERVATION.json').read_bytes()).hexdigest(),
    train_tail_sha256=hashlib.sha256(raw).hexdigest(),
    same_model_output='last/bbs', formal_results_unaudited=True,
    best_prescribed_start_retained=True, terminal_result=None,
    current_training_configuration_changed=False, full_goal_complete=False)
plan = dict(status='SINGLE_NEXT_OBSERVATION_PLANNED_FROM_MEASURED_RATE',
    due_cst=due.isoformat(), estimated_epoch2_full_validation_end_cst=eta.isoformat(),
    advance_seconds=300, later_poll_seconds=240,
    basis='Observed E2 436/4583 updates in 1286s plus completed E1 1189-batch validation duration; completion is an estimate',
    initial_observation=summary, observations=1, neural_calls=0,
    job_restart=False, source_changed=False)
for name, value in [('NORMAL_FIRST_OBSERVATION_SUMMARY.json', summary),
                    ('NORMAL_NEXT_OBSERVATION_PLAN.json', plan)]:
    destination = root / name
    assert not destination.exists()
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

source = (root / 'observe_normal_planned_authorized.py').read_text(encoding='utf-8')
source = source.replace('One observation at the estimated first-epoch boundary, not a fast poller.',
    'One E2 boundary observation, timed from the actual E2 rate and E1 validation.')
source = source.replace("due=datetime.datetime.fromisoformat(launch['time_cst'])+datetime.timedelta(seconds=launch['first_observation_seconds'])",
    "plan=json.loads((root/'NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())\n"
    "assert plan['status']=='SINGLE_NEXT_OBSERVATION_PLANNED_FROM_MEASURED_RATE'\n"
    "due=datetime.datetime.fromisoformat(plan['due_cst'])")
source = source.replace('NORMAL_FIRST_OBSERVATION', 'NORMAL_EPOCH2_OBSERVATION')
source = source.replace('NORMAL_OBSERVER_OWNER', 'NORMAL_EPOCH2_OBSERVER_OWNER')
source = source.replace('normal_first_observation', 'normal_epoch2_observation')
source = source.replace('SINGLE_PLANNED_NORMAL_OBSERVER_WAITING', 'SINGLE_PLANNED_EPOCH2_OBSERVER_WAITING')
source = source.replace("estimate_basis='Two-batch engineering steady update about3.5s times4583, plus initialization and formal evaluation; not a measured full epoch'",
    "estimate_basis=plan['basis']")
ast.parse(source)
target = root / 'observe_normal_epoch2_planned_authorized.py'
assert not target.exists()
target.write_text(source, encoding='utf-8')
print(json.dumps(dict(status=summary['status'], metrics=metrics,
    next_due_cst=due.isoformat(), estimated_end_cst=eta.isoformat(),
    observer_started=False, remote_queries=0)))
