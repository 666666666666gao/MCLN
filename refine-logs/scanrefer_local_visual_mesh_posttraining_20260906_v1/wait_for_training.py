"""Wait near the measured endpoint ETA, then audit and launch the fixed formal run."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).parent
plan = json.loads((root / 'queue_plan.json').read_text())
assert hashlib.sha256((root / 'post_training.py').read_bytes()).hexdigest() == plan['post_training_sha256']
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['waiter_sha256']
training = Path(plan['training_directory'])
baseline = training / 'baseline_verification.json'
assert hashlib.sha256(baseline.read_bytes()).hexdigest() == plan['baseline_verification_sha256']
checked = json.loads(baseline.read_text())
assert checked['status'] == 'pass' and checked['observed_steps_per_arm'] >= 64
assert checked['baseline_arms_all_rows_equal'] and checked['old_new_row_identity_and_points_equal']
zone = datetime.timezone(datetime.timedelta(hours=8))
first = datetime.datetime.fromisoformat(plan['first_check_cst'])
start = {'time_cst': datetime.datetime.now(zone).isoformat(), 'pid': os.getpid(),
    'status': 'waiting_near_measured_training_endpoint', 'first_check_cst': first.isoformat(),
    'training_screen_pid': plan['training_screen_pid'], 'interval_seconds': 240,
    'queue_plan_sha256': hashlib.sha256((root / 'queue_plan.json').read_bytes()).hexdigest()}
with (root / 'wait_started.json').open('x') as stream:
    json.dump(start, stream, indent=2, sort_keys=True)
print(json.dumps(start), flush=True)
delay = (first - datetime.datetime.now(zone)).total_seconds()
if delay > 0:
    time.sleep(delay)
while True:
    process = subprocess.run(['ps', '-p', str(plan['training_screen_pid']), '-o', 'pid=,args='],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.returncode in (0, 1), process.stderr
    live = process.returncode == 0
    if live:
        assert plan['training_directory'] in process.stdout.decode()
    event = {'time_cst': datetime.datetime.now(zone).isoformat(), 'training_screen_live': live,
        'process': process.stdout.decode(), 'controller_exit_present': (training / 'controller.exit').exists()}
    print(json.dumps(event), flush=True)
    with (root / 'wait_progress.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')
    if not live:
        break
    time.sleep(240)
assert (training / 'controller.exit').read_text().strip() == '0'
subprocess.check_call([sys.executable, str(root / 'post_training.py')])
