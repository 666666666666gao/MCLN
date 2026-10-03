"""Capture actual controller/worker liveness and current checkpoint capacity."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).parent
spec=json.loads((local/'g_control_spec.json').read_bytes())
code=r'''
import json,os,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);controller=int(sys.argv[2])
status=json.loads((root/'pair_status.json').read_bytes())
stage=status['stage'];directory=root/stage.split('/')[0]
pid=status['process_pid']
log=directory/(stage.split('/')[1]+'.log')
raw=log.read_bytes()[-5000:]
records={}
for name in ('initial/receipt.json','initial_reconstruction.json','checkpoint_snapshot.json','progress.json','receipt.json'):
    p=directory/name
    if p.exists():records[name]=json.loads(p.read_bytes())
print(json.dumps(dict(status=status,controller_pid=controller,
    controller_live=Path('/proc/%d'%controller).exists(),child_pid=pid,
    child_live=Path('/proc/%d'%pid).exists(),directory_free_bytes=shutil.disk_usage(str(root)).free,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used','--format=csv,noheader'],text=True).strip(),
    compute_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).splitlines(),
    stage_log_tail=raw.decode('utf-8',errors='replace'),records=records)))
'''
launch=json.loads((local/'pair_launch.json').read_bytes())
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
               password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
_,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-c',code,
    launch['root'],launch['process'].split()[0]]),timeout=60)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
receipt=json.loads(raw)
receipt['time_cst']=datetime.datetime.now().astimezone().isoformat()
(local/'pair_live_observation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
client.close()
print(json.dumps(receipt))
