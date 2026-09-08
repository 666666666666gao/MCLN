"""Stage and execute one bounded CUDA module check in the unchanged warm runtime."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_empty_pool_check_20260908_v1'
archive = repo / 'refine-logs/pvground_empty_pool_check_20260908_v1'
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
c = paramiko.SSHClient(); c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
with s.open('/root/autodl-tmp/mcln_pvground_fit_support_20260908_v2/controller.exit') as f: assert f.read().strip() == b'0'
with s.open(runtime + '/env_spec.json', 'rb') as f: environment = json.loads(f.read())
env_sha = hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert env_sha == '966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
_, out, err = c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader', timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status() == 0, err.read().decode()
assert root.rsplit('/', 1)[1] not in s.listdir('/root/autodl-tmp')
env = dict(environment['env'], CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
argv = [runtime + '/venv/bin/python', '-u', root + '/check.py']
controller = ('import fcntl,json,os,subprocess\nfrom pathlib import Path\nroot=Path(__file__).parent\n'
              'env=os.environ.copy();env.update(' + repr(env) + ')\n'
              'with open("/root/autodl-tmp/mcln_v99_backbone_gpu0.lock","a") as lock:\n'
              '    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n'
              '    with (root/"run.log").open("xb") as log:\n'
              '        result=subprocess.run(' + repr(argv) + ',env=env,stdout=log,stderr=subprocess.STDOUT)\n'
              '(root/"controller.exit").write_text(str(result.returncode)+"\\n")\n'
              'raise SystemExit(result.returncode)\n')
files = {'pvground_empty_pool_mask.py': (repo / 'models/pvground_empty_pool_mask.py').read_bytes(),
         'check.py': (repo / 'scripts/check_pvground_empty_pool_mask.py').read_bytes(), 'controller.py': controller.encode()}
with s.open(runtime + '/OpenPCDet/LICENSE', 'rb') as f: license_text = f.read()
files['OpenPCDet-LICENSE'] = license_text
files['source.json'] = (json.dumps(dict(upstream_commit='233f849829b6ac19afb8af8837a0246890908755',
    upstream_paths=['pcdet/ops/pointnet2/pointnet2_stack/pointnet2_utils.py', 'pcdet/ops/pointnet2/pointnet2_stack/pointnet2_modules.py'],
    change='Retain actual ball_query empty mask, zero per-radius output after original MLP/max-pool',
    environment_sha256=env_sha), indent=2) + '\n').encode()
s.mkdir(root); archive.mkdir()
for name, raw in files.items():
    if name.endswith('.py'): compile(raw, name, 'exec')
    (archive / name).write_bytes(raw)
    with s.open(root + '/' + name, 'wb') as f: f.write(raw)
    with s.open(root + '/' + name, 'rb') as f: assert f.read() == raw
_, out, err = c.exec_command('/root/miniconda3/envs/bdetr/bin/python ' + shlex.quote(root + '/controller.py'), timeout=60)
exit_code = out.channel.recv_exit_status(); stderr = err.read().decode()
for name in ['controller.exit', 'run.log']:
    s.get(root + '/' + name, str(archive / name))
assert exit_code == 0, (stderr, (archive / 'run.log').read_text())
s.get(root + '/receipt.json', str(archive / 'receipt.json'))
receipt = json.loads((archive / 'receipt.json').read_bytes())
assert receipt['status'] == 'pass'
s.close(); c.close()
print(json.dumps(receipt), flush=True)
