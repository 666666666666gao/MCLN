import datetime,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_fixed_memory_comparison_20260917_v1')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def command(pid):return (Path('/proc')/str(pid)/'cmdline').read_bytes()
def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)
spec=json.loads((root/'spec.json').read_bytes())
launch=json.loads((root/'launch.json').read_bytes())
parent=int(launch['process'].split()[0]);assert parent==8866
assert (str(root)+'/controller.py').encode() in command(parent)
children=subprocess.check_output(['pgrep','-P',str(parent)]).decode().split()
assert len(children)==1
child=int(children[0]);assert (str(root)+'/queue.py').encode() in command(child)
assert (Path('/proc')/str(child)/'wchan').read_text().strip()=='do_select'
assert not (Path('/proc')/str(child)/'task'/str(child)/'children').read_text().strip()
for name in ['controller.exit','terminal_D_E.json','terminal_D_E.log','terminal_D_E.exit','reschedule_receipt.json']:
    assert not (root/name).exists(),name
assert (root/'initial_D_E.exit').read_text().strip()=='0'
initial=json.loads((root/'initial_D_E.json').read_bytes())
assert initial['status']=='complete' and initial['stage']=='initial' and initial['rows']==6887
assert all(value==6887 for value in initial['identical_row_fields'].values())
initial_sha=sha(root/'initial_D_E.json')
assert spec['terminal_first_check_cst']=='2026-09-17T20:41:36.065439+08:00' and spec['poll_seconds']==300
for name,digest in spec['files'].items():assert sha(root/name)==digest,name
training=Path(spec['candidate_root']);train_sha=sha(training/'spec.json')
train_command=command(8666);assert (str(training)+'/controller.py').encode() in train_command
audit=Path(spec['candidate_audit']);assert (str(audit)+'/controller.py').encode() in command(8673)
queue=(root/'queue.py').read_text()
queue=replace_once(queue,"for stage,filename in [('initial','initial_audit.json'),('terminal','audit.json')]:",
    "assert (root/'initial_D_E.exit').read_text().strip()=='0'\n"
    "assert hashlib.sha256((root/'initial_D_E.json').read_bytes()).hexdigest()==spec['completed_initial_sha256']\n"
    "for stage,filename in [('terminal','audit.json')]:")
controller=replace_once((root/'controller.py').read_text(),"(root/'run.log').open('xb')","(root/'run.log').open('ab')")
compile(queue,'queue.py','exec');compile(controller,'controller.py','exec')
backup=root/'before_terminal_reschedule_20260917';backup.mkdir()
for name in ['queue.py','controller.py','spec.json','launch.json','run.log','controller.pid']:
    (backup/name).write_bytes((root/name).read_bytes())
old_spec_sha=sha(root/'spec.json')
os.kill(parent,signal.SIGTERM);os.kill(child,signal.SIGTERM);time.sleep(2)
for pid in [parent,child]:
    proc=Path('/proc')/str(pid)
    assert not proc.exists() or (proc/'stat').read_text().split()[2]=='Z'
assert not (root/'controller.exit').exists()
(root/'queue.py').write_text(queue);(root/'controller.py').write_text(controller)
spec['terminal_first_check_cst']='2026-09-17T19:15:00+08:00'
spec['completed_initial_sha256']=initial_sha
for name in ['queue.py','controller.py']:spec['files'][name]=sha(root/name)
(root/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
subprocess.run(['screen','-dmS','mcln_pvg_fixed_memory_comparison_v1','/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'controller.py')],check=True)
time.sleep(2)
process=subprocess.check_output(['pgrep','-af','^/root/miniconda3/envs/bdetr/bin/python -u '+str(root/'controller.py')+'$']).decode().strip()
assert len(process.splitlines())==1
new_parent=int(process.split()[0]);assert new_parent!=parent
new_children=subprocess.check_output(['pgrep','-P',str(new_parent)]).decode().split();assert len(new_children)==1
assert (str(root)+'/queue.py').encode() in command(int(new_children[0]))
assert (Path('/proc')/new_children[0]/'wchan').read_text().strip()=='do_select'
assert (root/'run.log').read_bytes()==(backup/'run.log').read_bytes()
assert sha(root/'initial_D_E.json')==initial_sha and not (root/'terminal_D_E.log').exists()
assert command(8666)==train_command and sha(training/'spec.json')==train_sha
for name,digest in spec['files'].items():assert sha(root/name)==digest,name
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
launch.update(time_cst=now,process=process,first_terminal_check_cst=spec['terminal_first_check_cst'],superseded_waiter_pid=parent)
(root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
receipt=dict(time_cst=now,old_controller_pid=parent,old_queue_pid=child,new_process=process,new_queue_pid=int(new_children[0]),
    first_terminal_check_cst=spec['terminal_first_check_cst'],poll_seconds=300,completed_initial_sha256=initial_sha,
    initial_not_repeated=True,comparison_code_unchanged=True,training_unchanged=True,formal_rules_unchanged=True,
    model_forwards=0,optimizer_steps=0,formal_rows=0,new_spec_sha256=sha(root/'spec.json'),old_spec_sha256=old_spec_sha)
(root/'reschedule_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
