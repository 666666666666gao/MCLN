import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

temporary = Path('C:/Users/gb/.codex/tmp')
bundle = temporary / 'pvground_runtime_bundle_20260908_v1.tar.gz'
proof = json.loads((temporary / 'pvground_runtime_bundle_receipt_20260908.json').read_bytes())
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == proof['bundle_sha256']
root = proof['remote_root']
remote_bundle = '/root/autodl-tmp/pvground_runtime_bundle_20260908_v1.tar.gz'
archive = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_runtime_20260908_v1')
archive.mkdir()
c = paramiko.SSHClient()
c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
check = "from pathlib import Path; import shutil; assert not Path(" + repr(root) + ").exists(); assert not Path(" + repr(remote_bundle) + ").exists(); assert shutil.disk_usage('/root/autodl-tmp').free>2500000000"
_, stdout, stderr = c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(check), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
s.put(str(bundle), remote_bundle)
bootstrap = '''import hashlib,json,tarfile
from pathlib import Path
root=Path(ROOT)
archive=Path(BUNDLE)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==SHA
root.mkdir()
with tarfile.open(str(archive),'r:gz') as packed:
    for member in packed.getmembers():
        assert member.isfile()
        destination=(root/member.name).resolve()
        assert root in destination.parents
    packed.extractall(str(root))
for name,digest in json.loads((root/'bundle_manifest.json').read_bytes()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print('RUNTIME_BUNDLE_VERIFIED')
'''.replace('ROOT', repr(root)).replace('BUNDLE', repr(remote_bundle)).replace('SHA', repr(proof['bundle_sha256']))
_, stdout, stderr = c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(bootstrap), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
print(raw.decode(), end='', flush=True)
for name in ['env_spec.json', 'source_bundle_receipt.json', 'bundle_manifest.json', 'build_runtime.py', 'setup_pvg_ops.py', 'verify_pvground_runtime.py', 'controller.sh']:
    data = (temporary / 'pvground_runtime_bundle_20260908_v1' / name).read_bytes()
    (archive / name).write_bytes(data)
(archive / 'bundle_transfer.json').write_bytes((json.dumps(proof, indent=2) + '\n').encode())
(archive / 'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
inner = 'exec bash ' + shlex.quote(root + '/controller.sh') + ' > ' + shlex.quote(root + '/run.log') + ' 2>&1'
_, stdout, stderr = c.exec_command('screen -dmS mcln_pvg_runtime_v1 bash -c ' + shlex.quote(inner), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
_, stdout, stderr = c.exec_command('pgrep -af ' + shlex.quote(r'^/root/miniconda3/envs/bdetr/bin/python -u build_runtime.py$'), timeout=30)
process = stdout.read().decode()
assert stdout.channel.recv_exit_status() == 0 and process.strip(), stderr.read().decode()
launch = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), 'process': process.strip(), 'screen': 'mcln_pvg_runtime_v1', 'root': root, 'estimate_seconds': [300, 600], 'spec_sha256': proof['spec_sha256'], 'training_steps': 0, 'formal_rows': 0}
data = (json.dumps(launch, indent=2) + '\n').encode()
(archive / 'launch.json').write_bytes(data)
with s.open(root + '/launch.json', 'wx') as stream:
    stream.write(data)
s.close()
c.close()
print('PVG_RUNTIME_BUILD_LAUNCHED ' + json.dumps(launch), flush=True)
