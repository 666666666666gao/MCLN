"""Launch four frozen Scan forwards on sixteen predetermined fit scenes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_fit_support_20260908_v2'
archive = repo / 'refine-logs/pvground_fit_support_20260908_v2'
previous = '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_vsaorder_v1'
support = '/root/autodl-tmp/mcln_pvground_support_capture_20260908_v1'
c = paramiko.SSHClient(); c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
failed = '/root/autodl-tmp/mcln_pvground_fit_support_20260908_v1'
with s.open(failed + '/controller.exit') as f: assert f.read().strip() == b'1'
with s.open(failed + '/run.log', 'rb') as f:
    failed_log = f.read()
assert b'and (sizes > 0).all()' in failed_log and failed_log.rstrip().endswith(b'AssertionError')
assert 'rows.jsonl' not in s.listdir(failed)
with s.open(previous + '/controller.exit') as f: assert f.read().strip() == b'0'
with s.open(support + '/controller.exit') as f: assert f.read().strip() == b'0'
with s.open(previous + '/spec.json', 'rb') as f: previous_spec = json.loads(f.read())
keys = ['runtime', 'input_manifest', 'env_spec_sha256', 'checkpoint_sha256', 'model_source', 'source_port', 'seed']
spec = {key: previous_spec[key] for key in keys}
spec.update(scene_count=16, row_selection='first expression per fit physical scene; sixteen evenly spaced sorted scene indices',
            conditions=['native_unaugmented', 'native_augmented'], batch_size=8, optimizer_steps=0, formal_rows=0,
            model_mode='eval frozen official Scan parent', gpu_lock='/root/autodl-tmp/mcln_v99_backbone_gpu0.lock')
spec['diagnostic_repair'] = dict(previous_root=failed, previous_completed_forwards=1, previous_optimizer_steps=0,
                                 change='Replace unsupported all-positive-size assertion with native float32 clamp(min=1e-6) for IoU only')
runtime = spec['runtime']
with s.open(runtime + '/env_spec.json', 'rb') as f: environment = json.loads(f.read())
assert hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
probe = "import json,shutil,subprocess;print(json.dumps({'free':shutil.disk_usage('/root/autodl-tmp').free,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader']).decode(),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']).decode()}))"
_, out, err = c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(probe), timeout=30)
state = json.loads(out.read()); assert out.channel.recv_exit_status() == 0, err.read().decode()
assert state['free'] > 2 * 1024**3 and not state['gpu_processes'].strip(), state
assert root.rsplit('/', 1)[1] not in s.listdir('/root/autodl-tmp')
env = dict(environment['env'], CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
argv = [runtime + '/venv/bin/python', '-u', root + '/audit.py']
controller = ('import fcntl,hashlib,json,os,subprocess\nfrom pathlib import Path\nroot=Path(__file__).parent\n'
              '(root/"controller.pid").write_text(str(os.getpid())+"\\n")\n'
              'spec=json.loads((root/"spec.json").read_bytes())\n'
              'for name,digest in spec["files"].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name\n'
              'env=os.environ.copy();env.update(' + repr(env) + ')\n'
              'with open(spec["gpu_lock"],"a") as lock:\n'
              '    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n'
              '    with (root/"run.log").open("xb") as log:\n'
              '        result=subprocess.run(' + repr(argv) + ',env=env,stdout=log,stderr=subprocess.STDOUT)\n'
              '(root/"controller.exit").write_text(str(result.returncode)+"\\n")\n'
              'raise SystemExit(result.returncode)\n')
files = {'audit.py': (repo / 'scripts/audit_pvground_fit_support.py').read_bytes(),
         'pvground_support_observation.py': (repo / 'scripts/pvground_support_observation.py').read_bytes(),
         'controller.py': controller.encode()}
spec['files'] = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
files['spec.json'] = (json.dumps(spec, indent=2) + '\n').encode()
s.mkdir(root); archive.mkdir()
for name, raw in files.items():
    if name.endswith('.py'): compile(raw, name, 'exec')
    (archive / name).write_bytes(raw)
    with s.open(root + '/' + name, 'wb') as f: f.write(raw)
    with s.open(root + '/' + name, 'rb') as f: assert f.read() == raw
command = ['screen', '-dmS', 'mcln_pvg_fit_support_v2', '/root/miniconda3/envs/bdetr/bin/python', root + '/controller.py']
_, out, err = c.exec_command(' '.join(map(shlex.quote, command)), timeout=30)
assert out.channel.recv_exit_status() == 0, err.read().decode()
_, out, err = c.exec_command('ps -eo pid,ppid,args', timeout=30)
processes = out.read().decode(); assert out.channel.recv_exit_status() == 0, err.read().decode()
matches = [line.strip() for line in processes.splitlines() if len(line.split(None, 2)) == 3 and line.split(None, 2)[2] == '/root/miniconda3/envs/bdetr/bin/python ' + root + '/controller.py']
assert len(matches) == 1
record = dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), root=root,
              process=matches[0], state_before=state, optimizer_steps=0, formal_rows=0, planned_forwards=4, eta_minutes=5)
raw = (json.dumps(record, indent=2) + '\n').encode(); (archive / 'launch.json').write_bytes(raw)
with s.open(root + '/launch.json', 'wb') as f: f.write(raw)
s.close(); c.close()
print(json.dumps(record), flush=True)
