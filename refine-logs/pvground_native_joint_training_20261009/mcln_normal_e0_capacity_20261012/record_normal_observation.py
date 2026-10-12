"""Record an already collected native observation; no SSH or neural calls."""
import argparse
import datetime
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('--node', required=True)
options = parser.parse_args()
assert options.node.isidentifier()
root = Path(__file__).resolve().parent
normal = root / 'normal_training'
prefix = 'normal_training_' + options.node
observation_path = normal / (prefix + '_OBSERVATION.json')
observed = json.loads(observation_path.read_bytes())
launch = json.loads((normal / 'LAUNCH.json').read_bytes())
assert observed['stage'] == 'normal_training'
assert launch['controller_pid'] == 271902
assert datetime.datetime.fromisoformat(observed['observed_cst']) >= datetime.datetime.fromisoformat(observed['first_due_cst'])
folder = normal / (prefix + '_observation')
names = {row['name'] for row in observed['files']}
history = []
if 'training/native_metrics.jsonl' in names:
    history = [json.loads(line) for line in (folder / 'training/native_metrics.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in history] == list(range(len(history)))
    assert all(row['rows'] == 9508 and row['primary_score'] == 'last/bbs' for row in history)
    assert all(row['acc050_percent'] == 100 * row['hits050'] / 9508 for row in history)
progress = None
if 'training/PROGRESS.json' in names:
    progress = json.loads((folder / 'training/PROGRESS.json').read_bytes())
log = (folder / 'training.log.tail.txt').read_text(encoding='utf-8', errors='replace')
fit_rows = re.findall(r'Train: \[(\d+)\]\[(\d+)/(\d+)\]', log)
latest_fit_log = dict(epoch=int(fit_rows[-1][0]), batch=int(fit_rows[-1][1]),
                      batches=int(fit_rows[-1][2])) if fit_rows else None
training_tqdm = re.findall(r'(\d+)/6081\s*\[', log)
latest_training_tqdm_batch = int(training_tqdm[-1]) if training_tqdm else None
record = dict(observation_path=str(observation_path), observed_cst=observed['observed_cst'],
              remote_status=observed['remote_status'], controller_alive=observed['controller_alive'],
              child_alive=observed['child_alive'], metrics_history=history,
              progress=progress, latest_fit_log=latest_fit_log,
              latest_training_tqdm_batch=latest_training_tqdm_batch,
              formal_fit_observed=bool(fit_rows) or bool(latest_training_tqdm_batch)
                                  or bool(progress and progress['optimizer_updates'] > 0),
              collected_files=sorted(names), new_neural_forwards=0, new_GPU_queries=0)
state = json.loads((root / 'CURRENT_PROGRESS.json').read_bytes())
now = datetime.datetime.now().astimezone().isoformat()
complete = observed['remote_status']['status'] == 'complete'
state.update(updated_cst=now, normal_training_observation=record,
             actual_formal_E0_metrics=history[0] if history else None,
             actual_formal_epoch_metrics=history[1:], formal_training_completed=complete,
             actual_formal_fit_observed=record['formal_fit_observed'],
             status=('MCLN_NATIVE_NORMAL_TRAINING_COMPLETE' if complete else
                     'MCLN_NATIVE_NORMAL_TRAINING_' + observed['remote_status']['status'].upper()),
             observer_pid=None, observer_native_session=None, next_observation_cst=None,
             PV_best_retained_hits=[5677, 4920], full_goal_complete=False)
if 'ENGINEERING_WEIGHT_RETIREMENT.json' in names:
    state['engineering_weight_retirement_execution'] = json.loads((folder / 'ENGINEERING_WEIGHT_RETIREMENT.json').read_bytes())
if 'latest_publication_section' in state:
    state['publication_section'] = state['latest_publication_section']
    state['publication_main_commit'] = state['latest_publication_main_commit']
(root / 'CURRENT_PROGRESS.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
base = Path('C:/Users/gb/.codex/tmp')
for path in (base / 'pvground_native_joint_training_20261009/native_joint_v2/NORMAL_CONTINUATION_STATE.json',
             base / 'pvground_native_joint_training_20261009/native_joint_v2/face_native_gpu_v2_20261011/CURRENT_PROGRESS.json',
             base / 'pvground_referit_mask_reference_20261006/current_research_goals.json',
             base / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'):
    data = json.loads(path.read_bytes())
    data.update(user_mcln_baseline_transfer=state, updated_cst=now, full_goal_complete=False)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=state['status'], metrics_history=history, latest_fit_log=latest_fit_log,
                     latest_training_tqdm_batch=latest_training_tqdm_batch,
                     formal_fit_observed=record['formal_fit_observed'], new_remote_queries=0)))
