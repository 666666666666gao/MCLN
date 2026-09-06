import datetime
import json
import os
from pathlib import Path

import paramiko

local=Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_range_pair_20260907_v1')
remote='/root/autodl-tmp/mcln_scanrefer_range_pair_20260907_v1'
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
_,out,err=client.exec_command('ps -p 47112 -o pid,stat,etime,args',timeout=30)
process=out.read().decode()
status=out.channel.recv_exit_status()
assert status in (0,1),err.read().decode()
with sftp.open(remote+'/run.log','rb') as stream:
    stream.seek(max(0,stream.stat().st_size-64000))
    raw=stream.read()
progress={}
for line in raw.decode().splitlines():
    for label in ['SCANREFER RANGE EVAL ','SCANREFER RANGE EVAL COMPLETE ','SCANREFER RANGE TRAIN ','SCANREFER RANGE PAIR COMPLETE ']:
        if line.startswith(label+'{'): progress[label.strip()]=json.loads(line[len(label):])
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
value={'time_cst':now.isoformat(),'screen_live':status==0 and 'mcln_scanrefer_range_pair_v1' in process,'process':process,'progress':progress}
if status==1:
    with sftp.open(remote+'/controller.exit','rb') as stream: value['controller_exit']=int(stream.read().strip())
stem='progress_'+now.strftime('%Y%m%d_%H%M%S')
(local/(stem+'.json')).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
(local/(stem+'.txt')).write_bytes(raw)
sftp.close()
client.close()
print(json.dumps(value))
print('\n'.join(raw.decode().splitlines()[-6:]))
