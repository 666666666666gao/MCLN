"""Queue fixed G training behind its live preflight and measured disk requirement."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260918_semantic_assignment_v1'
audit = '/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260918_semantic_assignment_v1'
archive = repo / 'refine-logs' / Path(root).name.replace('mcln_', '', 1)
audit_archive = repo / 'refine-logs' / Path(audit).name.replace('mcln_', '', 1)
c = paramiko.SSHClient()
c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()


def read(path):
    with s.open(path, 'rb') as stream:
        return stream.read()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


raw_spec = read(root + '/spec.json')
spec = json.loads(raw_spec)
assert raw_spec == (archive / 'spec.json').read_bytes()
for name, digest in spec['files'].items():
    assert sha(read(root + '/' + name)) == digest
preflight = json.loads(read(root + '/preflight_launch.json'))
pid = int(preflight['process'].split()[0])
if 'preflight.exit' in s.listdir(root):
    assert read(root + '/preflight.exit').strip() == b'0'
else:
    assert (root + '/preflight_controller.py').encode() in read('/proc/' + str(pid) + '/cmdline')
assert 'controller.py' not in s.listdir(root)
required_free = 2 * 343_000_000 + 256 * 1024**2 + 768 * 1024**2
controller = '''import datetime,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\\n')
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
while not (root/'preflight.exit').is_file():
    process=Path('/proc/PREFLIGHT_PID/cmdline')
    assert process.is_file() and (str(root)+'/preflight_controller.py').encode() in process.read_bytes()
    time.sleep(300)
assert (root/'preflight.exit').read_text().strip()=='0'
check=json.loads((root/'preflight_receipt.json').read_bytes())
assert check['status']=='pass' and check['model_states_restored'] and check['optimizer_steps']==0
assert check['spec_sha256']==hashlib.sha256((root/'spec.json').read_bytes()).hexdigest()
assert check['script_sha256']==hashlib.sha256((root/'train.py').read_bytes()).hexdigest()
while shutil.disk_usage(root).free<REQUIRED_FREE:
    print('SEMANTIC_ASSIGNMENT_WAIT_DISK '+str(shutil.disk_usage(root).free),flush=True)
    time.sleep(300)
runtime=Path(spec['runtime'])
environment=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
start=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),disk_free=shutil.disk_usage(root).free,required_free=REQUIRED_FREE,training_steps=3723,seed=2027)
(root/'training_start.json').write_text(json.dumps(start)+'\\n')
print('SEMANTIC_ASSIGNMENT_TRAINING_START '+json.dumps(start),flush=True)
command=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',str(runtime/'venv/bin/python'),'-u',str(root/'train.py'),'--spec',str(root/'spec.json')]
result=subprocess.run(command,cwd=str(runtime/'PV-Ground'),env=dict(os.environ,**environment['env']))
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''.replace('PREFLIGHT_PID', str(pid)).replace('REQUIRED_FREE', str(required_free)).encode()
compile(controller, 'controller.py', 'exec')
(archive / 'controller.py').write_bytes(controller)
with s.open(root + '/controller.py', 'wx') as stream:
    stream.write(controller)


def launch(where, local, screen):
    inner = 'exec /root/miniconda3/envs/bdetr/bin/python -u ' + shlex.quote(where + '/controller.py') + ' > ' + shlex.quote(where + '/run.log') + ' 2>&1'
    _, out, err = c.exec_command('screen -dmS ' + screen + ' bash -c ' + shlex.quote(inner), timeout=30)
    assert out.channel.recv_exit_status() == 0, err.read().decode()
    _, out, err = c.exec_command('pgrep -af ' + shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u ' + where + '/controller.py$'), timeout=30)
    process = out.read().decode().strip()
    assert process and out.channel.recv_exit_status() == 0, err.read().decode()
    record = dict(process=process, screen=screen, time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    raw = (json.dumps(record, indent=2) + '\n').encode()
    (local / 'launch.json').write_bytes(raw)
    with s.open(where + '/launch.json', 'wx') as stream:
        stream.write(raw)
    return record


training_launch = launch(root, archive, 'mcln_pvg_semantic_assignment_v1')
old = repo / 'refine-logs/pvground_scanrefer_endpoint_audit_20260917_fixed_memory_v1'
old_root = '/root/autodl-tmp/mcln_' + old.name
queue = (old / 'audit_queue.py').read_text().replace(old_root, audit)
queue = queue.replace("verified['fixed_visual_memory_verified']", "verified['semantic_assignment_verified']")
files = {'audit.py': (repo / 'scripts/audit_pvground_semantic_assignment.py').read_bytes(),
         'base_audit.py': (repo / 'refine-logs/pvground_scanrefer_endpoint_audit_20260917_task_observation_v1/audit.py').read_bytes(),
         'audit_queue.py': queue.encode(),
         'plan.md': (repo / 'docs/PVG_SEMANTIC_ASSIGNMENT_PLAN_2026-09-18.md').read_bytes()}
audit_spec = dict(training_root=root, training_controller_pid=int(training_launch['process'].split()[0]),
    training_spec_sha256=sha(raw_spec), first_check_cst=(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))) + datetime.timedelta(minutes=20)).isoformat(),
    poll_seconds=300, formal_evaluation_launch=False, files={name: sha(raw) for name, raw in files.items()})
files['spec.json'] = (json.dumps(audit_spec, indent=2) + '\n').encode()
files['controller.py'] = (old / 'controller.py').read_text().replace(old_root, audit).encode()
assert Path(audit).name not in s.listdir('/root/autodl-tmp')
s.mkdir(audit)
audit_archive.mkdir()
for name, raw in files.items():
    if name.endswith('.py'):
        compile(raw, name, 'exec')
    (audit_archive / name).write_bytes(raw)
    with s.open(audit + '/' + name, 'wx') as stream:
        stream.write(raw)
audit_launch = launch(audit, audit_archive, 'mcln_pvg_semantic_assignment_audit_v1')
print(json.dumps(dict(training=training_launch, audit=audit_launch, required_free=required_free,
                     preflight_pid=pid, training_is_queued=True, optimizer_steps_claimed=0)), flush=True)
s.close()
c.close()
