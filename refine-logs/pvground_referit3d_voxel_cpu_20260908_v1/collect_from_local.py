import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

root = '/root/autodl-tmp/mcln_pvground_referit3d_voxel_cpu_20260908_v1'
archive = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_referit3d_voxel_cpu_20260908_v1')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
names = sftp.listdir(root)
for name in ['controller.pid', 'audit.log', 'rows.json', 'receipt.json', 'audit.exit', 'controller.exit']:
    if name in names:
        sftp.get(root + '/' + name, str(archive / name))
observation = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), 'files': names}
if 'controller.pid' in names:
    pid = int((archive / 'controller.pid').read_text())
    script = 'import json,os; print(json.dumps(dict(pid=' + str(pid) + ',live=os.path.exists("/proc/' + str(pid) + '"))))'
    _, out, err = client.exec_command('/root/miniconda3/bin/python -c ' + shlex.quote(script), timeout=30)
    observation.update(json.loads(out.read()))
    assert out.channel.recv_exit_status() == 0 and not err.read()
if 'receipt.json' in names:
    receipt = json.loads((archive / 'receipt.json').read_bytes())
    assert receipt['rows_sha256'] == hashlib.sha256((archive / 'rows.json').read_bytes()).hexdigest()
    assert receipt['script_sha256'] == hashlib.sha256((archive / 'audit.py').read_bytes()).hexdigest()
    assert receipt['spec_sha256'] == hashlib.sha256((archive / 'spec.json').read_bytes()).hexdigest()
    observation['receipt_sha256'] = hashlib.sha256((archive / 'receipt.json').read_bytes()).hexdigest()
    observation['status'] = receipt['status']
    print(json.dumps({key: receipt[key] for key in ['status', 'time_cst', 'elapsed_seconds', 'sampled_rows', 'batches', 'voxel_processor_calls', 'torch_cuda_initialized']}))
if 'audit.log' in names:
    print((archive / 'audit.log').read_text()[-9000:])
(archive / 'observation.json').write_text(json.dumps(observation, indent=2) + '\n')
print('CPU_OBSERVATION ' + json.dumps(observation))
sftp.close()
client.close()
