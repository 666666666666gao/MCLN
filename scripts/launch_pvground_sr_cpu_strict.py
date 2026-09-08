"""Launch one CPU strict load after the already-completed Sr inventory."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_sr_checkpoint_inspection_20260908_v1'
archive = repo / 'refine-logs/pvground_sr_checkpoint_inspection_20260908_v1'
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
c = paramiko.SSHClient(); c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
assert 'strict_load.py' not in s.listdir(root)
with s.open(root + '/inventory.exit') as f: assert f.read().strip() == b'0'
with s.open(root + '/receipt.json', 'rb') as f: receipt = json.loads(f.read())
assert receipt['status'] == 'complete' and receipt['epoch'] == 31 and receipt['model_tensor_count'] == 1235
assert receipt['object_protocol_matches_planned_butd_cls']
with s.open(runtime + '/env_spec.json', 'rb') as f: env = json.loads(f.read())
assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == '966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
argv = ['env'] + [k + '=' + v for k, v in dict(env['env'], CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1').items()]
argv += [runtime + '/venv/bin/python', '-u', root + '/strict_load.py']
controller = ('import os,subprocess,json\nfrom pathlib import Path\nroot=Path(__file__).parent\n'
              '(root/"strict_controller.pid").write_text(str(os.getpid())+"\\n")\n'
              'with (root/"strict_load.log").open("xb") as out,(root/"strict_load.stderr").open("xb") as err:\n'
              '    result=subprocess.run(' + repr(argv) + ',stdout=out,stderr=err)\n'
              '(root/"strict_load.exit").write_text(str(result.returncode)+"\\n")\n'
              'raise SystemExit(result.returncode)\n')
files = {'strict_load.py': (repo / 'scripts/strict_load_pvground_sr_cpu.py').read_bytes(),
         'strict_controller.py': controller.encode()}
for name, raw in files.items():
    compile(raw, name, 'exec')
    (archive / name).write_bytes(raw)
    with s.open(root + '/' + name, 'wb') as f: f.write(raw)
    with s.open(root + '/' + name, 'rb') as f: assert f.read() == raw
command = ['screen', '-dmS', 'mcln_pvg_sr_cpu_strict_v1', '/root/miniconda3/envs/bdetr/bin/python', root + '/strict_controller.py']
_, out, err = c.exec_command(' '.join(map(shlex.quote, command)), timeout=30)
assert out.channel.recv_exit_status() == 0, err.read().decode()
_, out, err = c.exec_command('ps -eo pid,ppid,args', timeout=30)
processes = out.read().decode(); assert out.channel.recv_exit_status() == 0, err.read().decode()
matches = [line.strip() for line in processes.splitlines() if len(line.split(None, 2)) == 3 and line.split(None, 2)[2] == '/root/miniconda3/envs/bdetr/bin/python ' + root + '/strict_controller.py']
assert len(matches) == 1
launch = dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), process=matches[0],
              argv=argv, receipt_sha256=hashlib.sha256((archive / 'receipt.json').read_bytes()).hexdigest(),
              files={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}, model_forwards=0, training_launched=False)
raw = (json.dumps(launch, indent=2) + '\n').encode()
(archive / 'strict_launch.json').write_bytes(raw)
with s.open(root + '/strict_launch.json', 'wb') as f: f.write(raw)
s.close(); c.close()
print(json.dumps(launch), flush=True)
