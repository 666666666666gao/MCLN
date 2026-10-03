"""Read the completed control and the continuing pair; do not copy weights."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).parent
output = local/'control_complete'
assert not output.exists()
launch = json.loads((local/'pair_launch.json').read_bytes())
spec = json.loads((local/'g_control_spec.json').read_bytes())
code = r'''
import datetime,hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);controller=int(sys.argv[2]);directory=root/'g_control'
def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()
status=json.loads((root/'pair_status.json').read_bytes())
assert status['status']=='running' and status['stage']=='g_consistent/train'
assert [(x['arm'],x['mode']) for x in status['completed']]==[('g_control','train')]
train=json.loads((directory/'receipt.json').read_bytes())
assert train['status']=='complete' and train['training_steps']==3723
assert train['fit_rows']==29778 and train['fit_seen_exactly_once']
assert train['frozen_parameters_unchanged']
assert (directory/'train.exit').read_text().strip()=='0'
assert not (directory/'latest.pth').exists()
assert not (directory/'latest.pth.tmp').exists()
checkpoint=directory/'terminal.pth';digest=sha(checkpoint)
assert digest==train['terminal_sha256']
names=['spec.json','receipt.json','train.jsonl','train.log','train.exit','load.json','imports.json',
       'initial/receipt.json','initial/rows.jsonl','terminal/receipt.json','terminal/rows.jsonl']
files={name:{'bytes':(directory/name).stat().st_size,'sha256':sha(directory/name)} for name in names}
child=status['process_pid']
assert Path('/proc/%d'%controller).exists() and Path('/proc/%d'%child).exists()
print(json.dumps(dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    status=status,controller_pid=controller,controller_live=True,child_pid=child,child_live=True,
    files=files,checkpoint={'path':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256':digest},
    directory_free_bytes=shutil.disk_usage(str(root)).free,
    control=train,terminal_holdout=json.loads((directory/'terminal/receipt.json').read_bytes()))))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
_, stdout, stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python', '-c', code,
    launch['root'], launch['process'].split()[0]]), timeout=180)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
witness = json.loads(raw)
output.mkdir()
sftp = client.open_sftp()
for name, expected in witness['files'].items():
    target = output/name
    target.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(launch['root']+'/g_control/'+name, str(target))
    content = target.read_bytes()
    assert len(content) == expected['bytes']
    assert hashlib.sha256(content).hexdigest() == expected['sha256']
sftp.close()
client.close()
witness.update(collected_cst=datetime.datetime.now().astimezone().isoformat(),
               scope='completed control training and seen-scene module holdout only',
               formal9508_evaluated=False, weights_copied=0, weights_deleted=0,
               GPU_forward_executed=False, optimizer_updates_executed=0)
(output/'INTAKE.json').write_text(json.dumps(witness, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status='control_complete_verified',
    training_steps=witness['control']['training_steps'],
    terminal_holdout=witness['terminal_holdout'],
    checkpoint=witness['checkpoint'], current_stage=witness['status']['stage'],
    directory_free_bytes=witness['directory_free_bytes'],
    formal9508_evaluated=False, checkpoints_copied=0)))
