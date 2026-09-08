"""Compare native/masked/disabled full models on the existing nonempty frozen batch."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_empty_pool_replay_20260908_v1'
archive = repo / 'refine-logs/pvground_empty_pool_replay_20260908_v1'
previous = '/root/autodl-tmp/mcln_pvground_support_capture_20260908_v1'
c = paramiko.SSHClient(); c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
with s.open('/root/autodl-tmp/mcln_pvground_empty_pool_check_20260908_v1/controller.exit') as f: assert f.read().strip() == b'0'
with s.open(previous + '/observed_replay/support_receipt.json', 'rb') as f: support = json.loads(f.read())
assert all(r['all_query_operator_empty'] == 0 for r in support['records'])
with s.open(previous + '/spec.json', 'rb') as f: spec = json.loads(f.read())
runtime = spec['runtime']
with s.open(runtime + '/env_spec.json', 'rb') as f: environment = json.loads(f.read())
assert hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
_, out, err = c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader', timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status() == 0, err.read().decode()
assert root.rsplit('/', 1)[1] not in s.listdir('/root/autodl-tmp')
replay = (repo / 'scripts/replay_pvground_vsa_order.py').read_text(encoding='utf-8')
before = 'model.load_state_dict(state,strict=True);model.cuda().eval().requires_grad_(False)'
assert replay.count(before) == 1
replay = replay.replace(before, 'from pvground_empty_pool_mask import install_empty_pool_mask\n    install_empty_pool_mask(model,False)\n    ' + before)
replay = replay.replace("['reset_seed_first','reset_seed_repeat','advance_rng_control']", "['native_disabled','empty_mask_enabled','disabled_repeat']")
replay = replay.replace('if index<2:seed()', 'for reader in [model.backbone_net.vsa.SA_rawpoints]+list(model.backbone_net.vsa.SA_layers):reader.mask_empty=(index==1)\n        seed()')
replay = replay.replace("scope='same frozen first batch; no original capacity backward or whole6887 replay; hooks are observational'", "scope='same frozen nonempty eight-row batch; native, empty-mask enabled, disabled repeat; no training or population equivalence claim'")
replay = replay.replace("(args.output/'receipt.json').write_text", "assert all(v['exact'] for case in results[1:] for v in case['compared_to_first'].values())\n    (args.output/'receipt.json').write_text")
env = dict(environment['env'], CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
argv = [runtime + '/venv/bin/python', '-u', root + '/replay.py', '--spec', root + '/spec.json', '--output', root + '/results']
controller = ('import fcntl,os,subprocess\nfrom pathlib import Path\nroot=Path(__file__).parent\n'
              'env=os.environ.copy();env.update(' + repr(env) + ')\n'
              'with open("/root/autodl-tmp/mcln_v99_backbone_gpu0.lock","a") as lock:\n'
              '    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n'
              '    with (root/"run.log").open("xb") as log:\n'
              '        result=subprocess.run(' + repr(argv) + ',env=env,stdout=log,stderr=subprocess.STDOUT)\n'
              '(root/"controller.exit").write_text(str(result.returncode)+"\\n")\n'
              'raise SystemExit(result.returncode)\n')
files = {'pvground_empty_pool_mask.py': (repo / 'models/pvground_empty_pool_mask.py').read_bytes(),
         'replay.py': replay.encode(), 'controller.py': controller.encode()}
spec['files'] = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
spec['scope'] = 'Native versus empty-pool mask on fixed nonempty parent batch; three full forwards, zero updates'
files['spec.json'] = (json.dumps(spec, indent=2) + '\n').encode()
s.mkdir(root); archive.mkdir(); s.symlink(previous + '/inputs', root + '/inputs')
for name, raw in files.items():
    if name.endswith('.py'): compile(raw, name, 'exec')
    (archive / name).write_bytes(raw)
    with s.open(root + '/' + name, 'wb') as f: f.write(raw)
    with s.open(root + '/' + name, 'rb') as f: assert f.read() == raw
_, out, err = c.exec_command('/root/miniconda3/envs/bdetr/bin/python ' + shlex.quote(root + '/controller.py'), timeout=60)
exit_code = out.channel.recv_exit_status(); stderr = err.read().decode()
for name in ['controller.exit', 'run.log']:
    s.get(root + '/' + name, str(archive / name))
assert exit_code == 0, (stderr, (archive / 'run.log').read_text()[-5000:])
s.get(root + '/results/receipt.json', str(archive / 'receipt.json'))
receipt = json.loads((archive / 'receipt.json').read_bytes())
assert receipt['repeated_seed_exact'] and receipt['full_forwards'] == 3
s.close(); c.close()
print(json.dumps({key: value for key, value in receipt.items() if key != 'cases'}), flush=True)
