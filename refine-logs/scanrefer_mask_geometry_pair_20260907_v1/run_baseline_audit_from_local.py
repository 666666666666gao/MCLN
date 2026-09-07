import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_mask_geometry_pair_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1'
zone = datetime.timezone(datetime.timedelta(hours=8))
first_check = datetime.datetime.fromisoformat('2026-09-07T19:18:00+08:00')
delay = max(0., (first_check - datetime.datetime.now(zone)).total_seconds())
print(json.dumps({'phase': 'waiting_for_complete_baseline', 'first_check_cst': first_check.isoformat(),
                  'interval_seconds': 240, 'sleep_seconds': delay, 'training_pid': 62969}), flush=True)
time.sleep(delay)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
probe = '''import datetime,json,subprocess
from pathlib import Path
root=Path(REMOTE)
log=(root/'run.log').read_text()
complete=any(line.startswith('SCANREFER MASK GEOMETRY GT EVAL COMPLETE') and '"stage": "baseline"' in line for line in log.splitlines())
process=subprocess.run(['ps','-p','62969','-o','pid=,stat=,etime=,args='],stdout=subprocess.PIPE,universal_newlines=True)
exits={name:(root/name).read_text().strip() for name in ['training.exit','controller.exit'] if (root/name).exists()}
print(json.dumps({'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), 'baseline_complete':complete,'process':process.stdout.strip(),'exit_files':exits}))
'''.replace('REMOTE', repr(remote))
while True:
    _, output, error = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(probe), timeout=30)
    body = output.read(); errors = error.read().decode()
    assert output.channel.recv_exit_status() == 0, errors
    record = json.loads(body)
    print(json.dumps(record), flush=True)
    if record['baseline_complete']:
        break
    assert record['process'] and not record['exit_files'], record
    time.sleep(240)
assert 'baseline_cross_run_audit.json' not in sftp.listdir(remote)
source = (local / 'baseline_cross_run_audit.py').read_bytes()
for name, data in [('baseline_cross_run_audit.py', source),
                   ('run_baseline_audit_from_local.py', Path(__file__).read_bytes())]:
    (local / name).write_bytes(data)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(data)
_, output, error = client.exec_command('CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/bdetr/bin/python ' +
                                      shlex.quote(remote + '/baseline_cross_run_audit.py'), timeout=60)
body = output.read(); errors = error.read().decode()
assert output.channel.recv_exit_status() == 0, errors
result = json.loads(body)
for name in ['baseline_cross_run_audit.json', 'baseline_rows.json', 'baseline_metrics.json',
             'baseline_native_metrics.json', 'baseline_geometry_metrics.json']:
    size = sftp.stat(remote + '/' + name).st_size
    with sftp.open(remote + '/' + name, 'rb') as stream:
        stream.prefetch(file_size=size)
        data = stream.read()
    assert len(data) == size
    (local / name).write_bytes(data)
for name, digest in result['baseline_file_sha256'].items():
    assert hashlib.sha256((local / name).read_bytes()).hexdigest() == digest
assert hashlib.sha256(source).hexdigest() == result['audit_source_sha256']
sftp.close(); client.close()
print(json.dumps(result, indent=2), flush=True)
