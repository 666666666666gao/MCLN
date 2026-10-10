"""Schedule one overdue follow-up from the completed, actual E2 observation."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
observed = json.loads((root / 'NORMAL_EPOCH2_OBSERVATION.json').read_bytes())
assert json.loads((root / 'NORMAL_EPOCH2_OBSERVATION_EXIT.json').read_bytes())['exit_code'] == 0
assert observed['controller_alive'] and observed['child_alive']
assert [entry['epoch'] for entry in observed['metrics']] == [0, 1]
tail = (root / 'normal_epoch2_observation/train_tail.txt').read_text(encoding='utf-8')
progress = re.findall(r'(\d+)/1189 \[(\d+):(\d+)<(\d+):(\d+),\s*([0-9.]+)s/it\]', tail)
assert progress
done, minutes, seconds, left_minutes, left_seconds, rate = progress[-1]
assert int(done) == 1155
at = datetime.datetime.fromisoformat(observed['observed_cst'])
due = at + datetime.timedelta(seconds=240)
estimate = at + datetime.timedelta(seconds=60 * int(left_minutes) + int(left_seconds))
plan = dict(status='SINGLE_OVERDUE_EPOCH2_FOLLOWUP_PLANNED_FROM_ACTUAL_PROGRESS',
    due_cst=due.isoformat(), estimated_epoch2_full_validation_end_cst=estimate.isoformat(),
    actual_validation_batches_completed=int(done), total_validation_batches=1189,
    actual_validation_elapsed_seconds=60 * int(minutes) + int(seconds),
    logged_recent_seconds_per_batch=float(rate), later_poll_seconds=240,
    original_first_observer_closed=True, original_epoch2_observer_closed=True,
    original_epoch2_native_session=52851, original_wait_cell=785,
    original_epoch2_exit_code=0, observations=1,
    basis='Actual E2 validation1155/1189 at05:00:47 with40seconds logged remaining; overdue follow-up waits240seconds.',
    original_observation_sha256=hashlib.sha256((root / 'NORMAL_EPOCH2_OBSERVATION.json').read_bytes()).hexdigest(),
    train_tail_sha256=hashlib.sha256((root / 'normal_epoch2_observation/train_tail.txt').read_bytes()).hexdigest(),
    new_neural_calls=0, training_restart=False, active_source_changed=False)
path = root / 'NORMAL_EPOCH2_FOLLOWUP_PLAN.json'
assert not path.exists()
path.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
source = (root / 'observe_c_off_epoch2_authorized_route.py').read_text(encoding='utf-8')
source = source.replace('NORMAL_EPOCH2_OBSERVATION', 'NORMAL_EPOCH2_FOLLOWUP_OBSERVATION')
source = source.replace('NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER', 'NORMAL_EPOCH2_FOLLOWUP_OBSERVER_OWNER')
source = source.replace('normal_epoch2_observation', 'normal_epoch2_followup_observation')
source = source.replace('NORMAL_NEXT_OBSERVATION_PLAN.json', 'NORMAL_EPOCH2_FOLLOWUP_PLAN.json')
source = source.replace('import base64,datetime,hashlib,json,sys',
                        'import base64,datetime,hashlib,json,shutil,subprocess,sys')
anchor = "print(json.dumps(dict(status='SINGLE_SCHEDULED_NORMAL_OBSERVATION',observed_cst="
assert source.count(anchor) == 1
resources = """assert len(metric_paths)==1
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']).decode().strip()
disk_free={name:shutil.disk_usage(name).free for name in ('/','/root/autodl-tmp')}
weights=[dict(name=name,bytes=(metric_paths[0].parent/name).stat().st_size) for name in ('best.pth','latest.pth')]
"""
source = source.replace(anchor, resources + anchor)
anchor = 'train_log_bytes=size,train_tail_base64='
assert source.count(anchor) == 1
source = source.replace(anchor, 'gpu_resource_csv=gpu,disk_free_bytes=disk_free,weight_sizes=weights,' + anchor)
ast.parse(source)
helper = root / 'observe_c_off_epoch2_followup_authorized_route.py'
assert not helper.exists()
helper.write_text(source, encoding='utf-8')
print(json.dumps(plan))
