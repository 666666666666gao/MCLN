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
prep = repo / 'refine-logs/native_range_preparation_20260907_v2'
upstream = repo / 'refine-logs/scanrefer_range_posttraining_20260907_v1'
expected = json.loads((prep / 'expected.json').read_bytes())
files = {name: (repo / name).read_bytes() for name in ['scripts/queue_native_candidate_range_preflight.py',
    'scripts/evaluate_scanrefer_range_official.py', 'tests/test_queue_native_candidate_range_preflight.py']}
files['scripts/__init__.py'] = b''
files['launch_from_local.py'] = Path(__file__).read_bytes()
manifest = {'schema': 'mcln-native-range-preflight-queue-v1', 'interval_seconds': 240,
    'upstream_directory': '/root/autodl-tmp/mcln_scanrefer_range_posttraining_20260907_v1',
    'upstream_screen_pid': 47242, 'upstream_manifest_sha256': hashlib.sha256((upstream / 'input_manifest.json').read_bytes()).hexdigest(),
    'formal_directory': '/root/autodl-tmp/mcln_scanrefer_range_official_20260907_v1',
    'preparation_directory': '/root/autodl-tmp/mcln_native_range_preparation_20260907_v2',
    'preparation_receipt_sha256': hashlib.sha256((prep / 'receipt.json').read_bytes()).hexdigest(),
    'preparation_expected_sha256': hashlib.sha256((prep / 'expected.json').read_bytes()).hexdigest(),
    'native_preflight_directory': '/root/autodl-tmp/mcln_native_range_preflight_20260907_v1',
    'queue_script_sha256': hashlib.sha256(files['scripts/queue_native_candidate_range_preflight.py']).hexdigest(),
    'evaluation_script_sha256': hashlib.sha256(files['scripts/evaluate_scanrefer_range_official.py']).hexdigest(),
    'scope': 'GPU preflight after fixed Scan promotion only; no native formal training; no change to active Scan training or posttraining queue.'}
files['input_manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
controller = '''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd ''' + remote + '''
run_queue() {
set -e
/root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_queue_native_candidate_range_preflight.py > unit_tests.txt 2>&1
/root/miniconda3/envs/bdetr/bin/python -u scripts/queue_native_candidate_range_preflight.py --manifest input_manifest.json
}
(run_queue)
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''
files['controller.sh'] = controller.encode()
local.mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
for path, digest in [(manifest['upstream_directory'] + '/input_manifest.json', manifest['upstream_manifest_sha256']),
                     (manifest['preparation_directory'] + '/receipt.json', manifest['preparation_receipt_sha256']),
                     (manifest['preparation_directory'] + '/expected.json', manifest['preparation_expected_sha256'])]:
    with sftp.open(path, 'rb') as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == digest, path
_, output, error = client.exec_command('ps -p 47112,47242 -o pid,stat,etime,args', timeout=30)
processes = output.read().decode()
assert output.channel.recv_exit_status() == 0 and '47112' in processes and '47242' in processes, error.read().decode()
sftp.mkdir(remote)
for subdir in ['scripts', 'tests']:
    sftp.mkdir(remote + '/' + subdir)
for name, raw in files.items():
    target = local / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
command = 'screen -dmS mcln_native_range_preflight_queue_v1 bash -c ' + shlex.quote('exec bash ' + remote + '/controller.sh > ' + remote + '/queue.log 2>&1')
_, output, error = client.exec_command(command, timeout=30)
assert output.channel.recv_exit_status() == 0, error.read().decode()
_, output, error = client.exec_command('screen -ls', timeout=30)
sessions = output.read().decode()
assert output.channel.recv_exit_status() == 0, error.read().decode()
matches = [line.strip() for line in sessions.splitlines() if '.mcln_native_range_preflight_queue_v1' in line]
assert len(matches) == 1, sessions
launch = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'screen_session': matches, 'manifest_sha256': hashlib.sha256(files['input_manifest.json']).hexdigest(),
    'controller_sha256': hashlib.sha256(files['controller.sh']).hexdigest(), 'upstream_processes_verified': processes,
    'native_gpu_started': False, 'purpose': manifest['scope']}
raw = (json.dumps(launch, indent=2, sort_keys=True) + '\n').encode()
(local / 'launch.json').write_bytes(raw)
with sftp.open(remote + '/launch.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(launch), flush=True)
