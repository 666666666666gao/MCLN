"""Install F and run CPU tests plus a zero-update real-batch preflight only."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_rec_competition_v1'
archive = repo / 'refine-logs/pvground_scanrefer_finetune_20260917_rec_competition_v1'
control = repo / 'refine-logs/pvground_scanrefer_finetune_20260917_task_observation_v1'
c = paramiko.SSHClient()
c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()


def read(path):
    with s.open(path, 'rb') as stream:
        stream.prefetch()
        return stream.read()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


for suffix in ['task_observation', 'fixed_memory']:
    for part in ['finetune', 'endpoint_audit', 'formal']:
        assert read('/root/autodl-tmp/mcln_pvground_scanrefer_' + part + '_20260917_' + suffix + '_v1/controller.exit').strip() == b'0'
_, out, err = c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader', timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status() == 0, err.read().decode()
_, out, err = c.exec_command('df -B1 --output=avail /root/autodl-tmp', timeout=30)
free = int(out.read().decode().splitlines()[-1])
assert out.channel.recv_exit_status() == 0 and free > 1024**3, err.read().decode()
spec = json.loads((control / 'spec.json').read_bytes())
runtime = spec['runtime']
environment = json.loads(read(runtime + '/env_spec.json'))
assert sha(json.dumps(environment, sort_keys=True, separators=(',', ':')).encode()) == spec['env_spec_sha256']
files = {'train.py': (repo / 'scripts/run_pvground_scanrefer_rec_competition.py').read_bytes(),
         'pvground_rec_competition.py': (repo / 'models/pvground_rec_competition.py').read_bytes(),
         'plan.md': (repo / 'docs/PVG_REC_COMPETITION_PLAN_2026-09-17.md').read_bytes()}
for name in ['pvground_source_query.py', 'pvground_observation_query.py', 'pvground_task_observation_query.py']:
    files[name] = (repo / 'models' / name).read_bytes()
test = (repo / 'tests/test_pvground_rec_competition.py').read_text()
old = "Path(__file__).parents[1] / 'models/pvground_rec_competition.py'"
assert test.count(old) == 1
files['cpu_test.py'] = test.replace(old, "Path(__file__).parent / 'pvground_rec_competition.py'").encode()
spec.update(root=root, rec_competition=True, competition_weight=1.0,
            competition_module_sha256=sha(files['pvground_rec_competition.py']),
            comparison_control_root=json.loads((control / 'spec.json').read_bytes())['root'],
            comparison='F actual-bbs matched-root competition versus same-budget D',
            preflight_only_at_install=True, disk_free_before=free)
spec['files'] = {name: sha(raw) for name, raw in files.items()}
files['spec.json'] = (json.dumps(spec, indent=2) + '\n').encode()
env = dict(environment['env'], CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1')
command = ['flock', '-n', '/root/autodl-tmp/mcln_v99_backbone_gpu0.lock', runtime + '/venv/bin/python', '-u', root + '/train.py', '--spec', root + '/spec.json', '--preflight-only']
controller = ('import hashlib,json,os,subprocess\nfrom pathlib import Path\nroot=Path(__file__).parent\n'
    '(root/"preflight.pid").write_text(str(os.getpid())+"\\n")\n'
    'spec=json.loads((root/"spec.json").read_bytes())\n'
    'for name,digest in spec["files"].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name\n'
    'with (root/"preflight.log").open("xb") as log:\n'
    '    result=subprocess.run(' + repr(command) + ',env=dict(os.environ,**' + repr(env) + '),stdout=log,stderr=subprocess.STDOUT)\n'
    '(root/"preflight.exit").write_text(str(result.returncode)+"\\n")\nraise SystemExit(result.returncode)\n')
files['preflight_controller.py'] = controller.encode()
assert Path(root).name not in s.listdir('/root/autodl-tmp') and not archive.exists()
s.mkdir(root)
archive.mkdir()
for name, raw in files.items():
    if name.endswith('.py'):
        compile(raw, name, 'exec')
    (archive / name).write_bytes(raw)
    with s.open(root + '/' + name, 'wx') as stream:
        stream.write(raw)
    assert read(root + '/' + name) == raw
cpu_command = "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 " + shlex.quote(runtime + '/venv/bin/python') + ' ' + shlex.quote(root + '/cpu_test.py') + ' > ' + shlex.quote(root + '/cpu_test.log') + ' 2>&1'
_, out, err = c.exec_command(cpu_command, timeout=60)
code = out.channel.recv_exit_status()
s.get(root + '/cpu_test.log', str(archive / 'cpu_test.log'))
with s.open(root + '/cpu_test.exit', 'wx') as stream:
    stream.write(str(code) + '\n')
(archive / 'cpu_test.exit').write_text(str(code) + '\n')
assert code == 0, (archive / 'cpu_test.log').read_text()
_, out, err = c.exec_command('screen -dmS mcln_pvg_rec_competition_preflight_v1 /root/miniconda3/envs/bdetr/bin/python -u ' + shlex.quote(root + '/preflight_controller.py'), timeout=30)
assert out.channel.recv_exit_status() == 0, err.read().decode()
_, out, err = c.exec_command('pgrep -af ' + shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u ' + root + '/preflight_controller.py$'), timeout=30)
process = out.read().decode().strip()
assert process and out.channel.recv_exit_status() == 0, err.read().decode()
record = dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    process=process, cpu_tests_pass=True, full_training_started=False, expected_seconds=600,
    first_check_after_seconds=300, optimizer_steps=0, formal_rows=0, disk_free_before=free)
raw = (json.dumps(record, indent=2) + '\n').encode()
(archive / 'preflight_launch.json').write_bytes(raw)
with s.open(root + '/preflight_launch.json', 'wx') as stream:
    stream.write(raw)
s.close()
c.close()
print(json.dumps(record), flush=True)
