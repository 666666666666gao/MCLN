import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
remote = '/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_audit_preparation_20260906_v1'
local = repo / 'refine-logs/scanrefer_local_visual_mesh_audit_preparation_20260906_v1'
names = ['scripts/audit_scanrefer_local_visual_pair.py', 'scripts/audit_scanrefer_joint_readout_pair.py',
    'scripts/scanrefer_data_contract.py', 'tests/test_audit_scanrefer_local_visual_pair.py',
    'tests/test_scanrefer_data_contract.py']
uploads = {name: (repo / name).read_bytes() for name in names}
uploads['scripts/__init__.py'] = b''
uploads['prepare.py'] = Path(__file__).read_bytes()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
sftp.mkdir(remote)
sftp.mkdir(remote + '/scripts')
sftp.mkdir(remote + '/tests')
local.mkdir()
for name, raw in uploads.items():
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
    target = local / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
command = 'cd ' + shlex.quote(remote) + " && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=" + shlex.quote(remote) + ' /root/miniconda3/envs/bdetr/bin/python -m pytest -q tests/test_audit_scanrefer_local_visual_pair.py tests/test_scanrefer_data_contract.py'
_, out, err = client.exec_command(command, timeout=60)
stdout = out.read().decode()
stderr = err.read().decode()
code = out.channel.recv_exit_status()
(local / 'test_stdout.txt').write_text(stdout)
(local / 'test_stderr.txt').write_text(stderr)
assert code == 0, stdout + stderr
receipt = {'schema': 'mcln-scanrefer-mesh-audit-preparation-v1',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'status': 'cpu_tests_pass; actual new trained endpoint audit pending',
    'command': command, 'exit_code': code, 'stdout': stdout,
    'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in uploads.items()},
    'gpu_forwards': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0}
raw = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
with sftp.open(remote + '/receipt.json', 'wx') as stream:
    stream.write(raw)
(local / 'receipt.json').write_bytes(raw)
sftp.close()
client.close()
print(raw.decode())
