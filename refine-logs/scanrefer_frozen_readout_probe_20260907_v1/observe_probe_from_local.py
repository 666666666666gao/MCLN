import datetime
import json
import os
from pathlib import Path
import shlex

import paramiko

local=Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_frozen_readout_probe_20260907_v1')
remote='/root/autodl-tmp/mcln_scanrefer_frozen_readout_probe_20260907_v1'
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
probe="""import datetime,json,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_scanrefer_frozen_readout_probe_20260907_v1')
launch=json.loads((root/'launch.json').read_text())
pid=int(launch['screen_session'][0].split('.')[0])
p=subprocess.run(['ps','-eo','pid,ppid,stat,etime,args'],stdout=subprocess.PIPE,check=True)
rows=[line.split(None,4) for line in p.stdout.decode().splitlines()[1:]]
family={pid}
while True:
 children={int(row[0]) for row in rows if int(row[1]) in family}
 if children.issubset(family):break
 family.update(children)
result={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'process_family':[row for row in rows if int(row[0]) in family]}
with (root/'run.log').open('rb') as stream:
 stream.seek(max(0,(root/'run.log').stat().st_size-14000))
 result['log_tail']=stream.read().decode().splitlines()[-10:]
if (root/'controller.exit').exists():result['controller_exit']=(root/'controller.exit').read_text().strip()
print(json.dumps(result))
"""
_,output,error=client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(probe),timeout=30)
raw=output.read()
assert output.channel.recv_exit_status()==0,error.read().decode()
state=json.loads(raw)
sftp=client.open_sftp()
if state.get('controller_exit')=='0':
    for name in ['receipt.json','controller.exit','run.log']:
        with sftp.open(remote+'/'+name,'rb') as stream:raw=stream.read()
        (local/('run_complete.txt' if name=='run.log' else name)).write_bytes(raw)
    receipt=json.loads((local/'receipt.json').read_bytes())
    state['receipt_summary']={key:value for key,value in receipt.items() if key not in ['observations','disposable_updates','candidate_trainable_tensors']}
    state['gradient_summaries']=[{key:value for key,value in row.items() if key not in ['parameter_gradients','point_sha256','scan_ids','row_ids']} for row in receipt['observations']]
stamp=datetime.datetime.fromisoformat(state['time_cst']).strftime('%Y%m%d_%H%M%S')
(local/('observation_'+stamp+'.json')).write_text(json.dumps(state,indent=2,sort_keys=True)+'\n',encoding='utf-8')
sftp.close()
client.close()
print(json.dumps(state),flush=True)
