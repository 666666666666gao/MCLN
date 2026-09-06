import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/native_range_preflight_queue_20260907_v1'
remote = '/root/autodl-tmp/mcln_native_range_preflight_queue_20260907_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
files = {}
for name in ['unit_tests.txt', 'upstream_observations.jsonl', 'queue.log']:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        raw = stream.read()
    (local / name).write_bytes(raw)
    files[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
assert '3 passed' in (local / 'unit_tests.txt').read_text()
last = json.loads((local / 'upstream_observations.jsonl').read_bytes().splitlines()[-1])
assert last['upstream_screen_pid'] == 47242 and not last['native_gpu_started']
probe = """import json,shutil,subprocess
from pathlib import Path
p=Path('/root/autodl-tmp/mcln_native_range_preflight_queue_20260907_v1')
screens=subprocess.run(['ps','-p','47112,47242,48128','-o','pid,stat,etime,args'],stdout=subprocess.PIPE,check=True).stdout.decode()
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],stdout=subprocess.PIPE,check=True).stdout.decode()
assert all(str(pid) in screens for pid in [47112,47242,48128])
print(json.dumps({'processes':screens,'gpu':gpu,'disk':shutil.disk_usage('/root/autodl-tmp')._asdict(),
 'queue_controller_exit_exists':(p/'controller.exit').exists(),
 'native_preflight_directory_exists':Path('/root/autodl-tmp/mcln_native_range_preflight_20260907_v1').exists()}))
"""
_, output, error = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(probe), timeout=30)
state = json.loads(output.read())
assert output.channel.recv_exit_status() == 0, error.read().decode()
assert not state['queue_controller_exit_exists'] and not state['native_preflight_directory_exists']
with sftp.open('/root/autodl-tmp/mcln_scanrefer_range_posttraining_20260907_v1/training_observations.jsonl', 'rb') as stream:
    raw = stream.read()
(repo / 'refine-logs/scanrefer_range_posttraining_20260907_v1/training_observations.jsonl').write_bytes(raw)
state['latest_training_observation'] = json.loads(raw.splitlines()[-1])
state.update(schema='mcln-native-range-queue-armed-v1', status='armed_waiting_for_scan',
    time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    files=files, last_upstream_observation=last, tests_passed=3, native_preflight_started=False,
    native_training_started=False)
raw = (json.dumps(state, indent=2, sort_keys=True) + '\n').encode()
(local / 'armed_receipt.json').write_bytes(raw)
with sftp.open(remote + '/armed_receipt.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(state), flush=True)
