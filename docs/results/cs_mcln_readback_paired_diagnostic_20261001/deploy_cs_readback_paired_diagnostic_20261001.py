from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


main = Path(r'C:\Users\gb\.codex_mcln_g0_20260905')
isolated = Path(r'C:\Users\gb\.codex_mcln_v99_internal_20260928')
name = 'cs_mcln_readback_paired_diagnostic_20261001'
destination = main / 'docs/results' / name
destination.mkdir(parents=True, exist_ok=True)
remote = '/root/autodl-tmp/cs_mcln_readback_diagnostic_20261001_v1'
source = '/root/autodl-tmp/cs_mcln_readback_source_20260930_v1'
base = '/root/autodl-tmp/cs_mcln_source_20260923_v1'
python = '/root/miniconda3/envs/bdetr/bin/python'
launch = json.loads((main / 'docs/results/cs_mcln_scanrefer_readback_20260930/launch.json').read_text())
files = [
    isolated / 'scripts/diagnose_cs_mcln_readback.py',
    isolated / 'scripts/run_cs_mcln_readback_posttrain_diagnostic.py',
    Path(r'C:\Users\gb\.codex\tmp\check_cs_readback_paired_diagnostic_cpu_20261001.py'),
]
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
for relative, expected in launch['frozen_python_files'].items():
    with sftp.open(source + '/' + relative, 'rb') as stream:
        data = stream.read()
    assert len(data) == expected['bytes']
    assert hashlib.sha256(data).hexdigest() == expected['sha256']
_, stdout, stderr = client.exec_command('mkdir -- ' + shlex.quote(remote), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
hashes = {}
for file in files:
    data = file.read_bytes()
    target = remote + '/' + file.name
    sftp.put(str(file), target)
    with sftp.open(target, 'rb') as stream:
        assert stream.read() == data
    hashes[file.name] = hashlib.sha256(data).hexdigest()
    (destination / file.name).write_bytes(data)
command = ('cd ' + shlex.quote(base) + ' && PYTHONPATH=' + shlex.quote(source + ':' + base)
           + ' ' + python + ' -u ' + shlex.quote(remote + '/' + files[2].name))
_, stdout, stderr = client.exec_command(command, timeout=120)
out, err = stdout.read().decode(), stderr.read().decode()
code = stdout.channel.recv_exit_status()
receipt = {
    'checked_at_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
    'exit_code': code, 'stdout': out, 'stderr': err,
    'remote_diagnostic_directory': remote, 'files_sha256': hashes,
    'frozen_training_source_hashes_verified': len(launch['frozen_python_files']),
    'training_state_queried': False, 'gpu_work_launched': False,
}
if code == 0:
    receipt['result'] = json.loads(out.splitlines()[-1])
(destination / 'cpu_fixture.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
sftp.close()
client.close()
print(json.dumps(receipt, indent=2), flush=True)
assert code == 0
