"""Read the original remote controller after its local observer handle was lost."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).resolve().parent
assert not (local/'fit_wait.json').exists()
spec = json.loads((local/'control_spec.json').read_bytes())
launch = json.loads((local/'fit_launch.json').read_bytes())
probe = '''
import datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);spec=json.loads((root/'control_spec.json').read_bytes())
pattern='^'+spec['runtime']+'/venv/bin/python -B -u '+str(root)+'/controller.py --phase fit$'
result=subprocess.run(['pgrep','-af',pattern],stdout=subprocess.PIPE)
assert result.returncode in (0,1)
process=result.stdout.decode().strip()
status=json.loads((root/'fit_status.json').read_bytes())
exitpath=root/'fit_controller.exit'
exitcode=int(exitpath.read_text().strip()) if exitpath.exists() else None
receipts={}
for arm in ('control','support_reference'):
    receipts[arm]={name:json.loads((root/arm/name).read_bytes()) for name in
        ('receipt.json','initial/receipt.json','terminal/receipt.json','formal/receipt.json')
        if (root/arm/name).exists()}
parent=Path(spec['geometry_terminal'])
assert hashlib.sha256(parent.read_bytes()).hexdigest()==spec['geometry_terminal_sha256']
print(json.dumps(dict(time_cst=datetime.datetime.now().astimezone().isoformat(),stage='fit',
    controller_alive=bool(process),process=process,status=status,exitcode=exitcode,
    receipts=receipts,protected_best_sha256_exact=True,
    data_free_bytes=shutil.disk_usage(root).free,system_free_bytes=shutil.disk_usage('/').free,
    gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']).decode().strip(),
    inference_or_optimizer_replayed=False)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
    password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
_,stdout,stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',probe,launch['root']]),timeout=90)
raw = stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record = json.loads(raw)
stamp = record['time_cst'].split('T')[1][:8].replace(':','')
path = local/('ACTUAL_EXISTING_CONTROLLER_'+stamp+'.json')
assert not path.exists()
path.write_bytes(raw)
if not record['controller_alive'] and record['exitcode'] is not None:
    closed = dict(time_cst=record['time_cst'],observer_closed=True,stage='fit',terminal=record,
        observation_count=1,inference_or_optimizer_replayed=False,
        observer='actual_one_shot_after_original_handle_missing',
        original_observer_handle=40310,original_observer_exit_code_unobserved=True,
        source_observation=str(path))
    (local/'fit_wait.json').write_text(json.dumps(closed,indent=2)+'\n',encoding='utf-8')
client.close()
print(json.dumps({'time_cst':record['time_cst'],'status':record['status'],
    'controller_alive':record['controller_alive'],'exitcode':record['exitcode'],
    'receipts':{arm:{name:value.get('metrics') for name,value in items.items() if name=='formal/receipt.json'}
        for arm,items in record['receipts'].items()},'gpu_processes':record['gpu_processes'],
    'observation':str(path)}),flush=True)
