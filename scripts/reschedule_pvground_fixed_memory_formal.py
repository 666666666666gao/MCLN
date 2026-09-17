"""Reschedule only the verified sleeping E formal waiter; retain original provenance."""
import json
import os
import shlex
from pathlib import Path
import paramiko

repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260917_fixed_memory_v1'
archive = repo / 'refine-logs' / Path(root).name.replace('mcln_', '', 1)
remote_code = r'''import datetime,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260917_fixed_memory_v1')
training=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_fixed_memory_v1')
def command(pid):return (Path('/proc')/str(pid)/'cmdline').read_bytes()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
launch=json.loads((root/'launch.json').read_bytes())
parent=int(launch['process'].split()[0])
assert parent==8872 and (str(root)+'/controller.py').encode() in command(parent)
children=subprocess.check_output(['pgrep','-P',str(parent)]).decode().split()
assert len(children)==1
child=int(children[0])
assert (str(root)+'/formal_queue.py').encode() in command(child)
assert (Path('/proc')/str(child)/'wchan').read_text().strip()=='do_select'
assert not (Path('/proc')/str(child)/'task'/str(child)/'children').read_text().strip()
for name in ['controller.exit','decision.json','evaluation.exit','terminal_restore.json','reschedule_receipt.json']:
    assert not (root/name).exists(),name
assert (root/'run.log').stat().st_size==0
spec=json.loads((root/'spec.json').read_bytes())
assert spec['first_check_cst']=='2026-09-17T20:41:11.386051+08:00'
assert spec['poll_seconds']==300
for name,digest in spec['files'].items():assert sha(root/name)==digest,name
train_command=command(8666)
assert (str(training)+'/controller.py').encode() in train_command
assert b'mcln_pvground_scanrefer_endpoint_audit_20260917_fixed_memory_v1/controller.py' in command(8673)
train_sha=sha(training/'spec.json')
assert train_sha==spec['training_spec_sha256']
before=root/'before_reschedule_20260917'
before.mkdir()
for name in ['launch.json','spec.json','run.log']:(before/name).write_bytes((root/name).read_bytes())
old_spec_sha=sha(root/'spec.json')
os.kill(parent,signal.SIGTERM)
os.kill(child,signal.SIGTERM)
time.sleep(2)
for pid in [parent,child]:
    proc=Path('/proc')/str(pid)
    assert not proc.exists() or (proc/'stat').read_text().split()[2]=='Z'
assert not (root/'controller.exit').exists()
spec['first_check_cst']='2026-09-17T19:15:00+08:00'
(root/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
screen='mcln_pvg_fixed_memory_formal_v1'
subprocess.run(['screen','-dmS',screen,'bash','-lc',
    'exec /root/miniconda3/envs/bdetr/bin/python -u '+str(root/'controller.py')+' >> '+str(root/'run.log')+' 2>&1'],check=True)
time.sleep(2)
process=subprocess.check_output(['pgrep','-af','^/root/miniconda3/envs/bdetr/bin/python -u '+str(root/'controller.py')+'$']).decode().strip()
assert len(process.splitlines())==1
new_parent=int(process.split()[0]);assert new_parent!=parent
new_children=subprocess.check_output(['pgrep','-P',str(new_parent)]).decode().split()
assert len(new_children)==1
assert (str(root)+'/formal_queue.py').encode() in command(int(new_children[0]))
assert (Path('/proc')/new_children[0]/'wchan').read_text().strip()=='do_select'
assert command(8666)==train_command and sha(training/'spec.json')==train_sha
for name,digest in spec['files'].items():assert sha(root/name)==digest,name
assert not (root/'decision.json').exists() and (root/'run.log').stat().st_size==0
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
launch.update(time_cst=now,process=process,first_check_cst=spec['first_check_cst'],superseded_waiter_pid=parent,training_changes=False)
(root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
receipt=dict(time_cst=now,old_controller_pid=parent,old_queue_pid=child,new_process=process,new_queue_pid=int(new_children[0]),
    old_first_check_cst='2026-09-17T20:41:11.386051+08:00',first_check_cst=spec['first_check_cst'],poll_seconds=300,
    old_spec_sha256=old_spec_sha,new_spec_sha256=sha(root/'spec.json'),training_spec_sha256=train_sha,
    training_controller_pid=8666,audit_controller_pid=8673,training_unchanged=True,formal_code_unchanged=True,
    formal_rows=0,reason='Measured fit and endpoint ETA 19:20-19:30; remove obsolete conservative wait to20:41')
(root/'reschedule_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
'''
compile(remote_code, 'reschedule_waiter.py', 'exec')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
target = root + '/reschedule_waiter.py'
with sftp.open(target, 'wb') as stream:
    stream.write(remote_code.encode())
_, stdout, stderr = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -u ' + shlex.quote(target), timeout=30)
output = stdout.read().decode()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
for name in ['launch.json', 'spec.json', 'reschedule_waiter.py', 'reschedule_receipt.json']:
    sftp.get(root + '/' + name, str(archive / name))
(archive / 'before_reschedule_20260917').mkdir()
for name in ['launch.json', 'spec.json', 'run.log']:
    sftp.get(root + '/before_reschedule_20260917/' + name,
             str(archive / 'before_reschedule_20260917' / name))
sftp.close()
client.close()
print(output)
