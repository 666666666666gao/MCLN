import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo / 'refine-logs/pvground_rec_readout_cpu_20260908_v1'
archive.mkdir()
root = '/root/autodl-tmp/mcln_pvground_rec_readout_cpu_20260908_v1'
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
files = {'readout.py': (repo / 'scripts/pvground_rec_readout.py').read_bytes(),
         'verify.py': (repo / 'scripts/verify_pvground_rec_readout.py').read_bytes()}
for name, raw in files.items():
    compile(raw, name, 'exec')
    (archive / name).write_bytes(raw)
spec = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'runtime': runtime, 'root': root, 'input_kind': 'synthetic controlled CPU integration fixtures',
        'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
        'active_scan_source_changed': False, 'model_forwards': 0, 'optimizer_steps': 0, 'formal_rows': 0}
files['spec.json'] = (json.dumps(spec, indent=2) + '\n').encode()
(archive / 'spec.json').write_bytes(files['spec.json'])
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
sftp.mkdir(root)
for name, raw in files.items():
    with sftp.open(root + '/' + name, 'wb') as stream:
        stream.write(raw)
    with sftp.open(root + '/' + name, 'rb') as stream:
        assert stream.read() == raw
with sftp.open(runtime + '/env_spec.json', 'rb') as stream:
    runtime_spec = json.loads(stream.read())
assert hashlib.sha256(json.dumps(runtime_spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == '966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
environment = dict(runtime_spec['env'], CUDA_VISIBLE_DEVICES='')
argv = ['env'] + [key + '=' + value for key, value in environment.items()]
argv += [runtime + '/venv/bin/python', '-u', root + '/verify.py', '--runtime', runtime,
         '--readout', root + '/readout.py', '--output', root + '/receipt.json']
_, stdout, stderr = client.exec_command(' '.join(shlex.quote(arg) for arg in argv), timeout=60)
raw = stdout.read()
err = stderr.read()
code = stdout.channel.recv_exit_status()
for name, value in [('stdout.txt', raw), ('stderr.txt', err), ('audit.exit', (str(code) + '\n').encode())]:
    (archive / name).write_bytes(value)
    with sftp.open(root + '/' + name, 'wb') as stream:
        stream.write(value)
print(raw.decode(), end='')
print(err.decode(), end='')
assert code == 0, code
sftp.get(root + '/receipt.json', str(archive / 'receipt.json'))
receipt = json.loads((archive / 'receipt.json').read_bytes())
assert receipt['status'] == 'pass' and not receipt['torch_cuda_initialized']
record = {'root': root, 'exit_code': code, 'argv': argv, 'receipt_sha256': hashlib.sha256((archive / 'receipt.json').read_bytes()).hexdigest()}
(archive / 'execution.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
sftp.close()
client.close()
