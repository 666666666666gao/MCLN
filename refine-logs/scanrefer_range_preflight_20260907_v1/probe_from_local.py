import datetime
import json
import os
from pathlib import Path

import paramiko

local=Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_range_preflight_20260907_v1')
remote='/root/autodl-tmp/mcln_scanrefer_range_preflight_20260907_v1'
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
names=sftp.listdir(remote)
state={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'files':names}
for name in ['launch.json','controller.exit','receipt.json','coverage_rows.json']:
    if name in names:
        with sftp.open(remote+'/'+name,'rb') as stream:
            stream.prefetch(file_size=stream.stat().st_size)
            raw=stream.read()
        (local/name).write_bytes(raw)
        if name!='coverage_rows.json': state[name]=json.loads(raw)
if 'run.log' in names:
    with sftp.open(remote+'/run.log','rb') as stream:
        stream.seek(max(0,stream.stat().st_size-12000))
        raw=stream.read()
    (local/'latest_run.txt').write_bytes(raw)
    state['log_tail']=raw.decode()
_,out,err=client.exec_command("ps -eo pid,ppid,stat,etime,args | grep -E 'mcln_scanrefer_range_preflight|run_scanrefer_range_preflight|nvidia-smi' | grep -v grep",timeout=30)
state['processes']=out.read().decode()
assert out.channel.recv_exit_status() in (0,1),err.read().decode()
(local/'latest_probe.json').write_text(json.dumps(state,indent=2,sort_keys=True)+'\n',encoding='utf-8')
sftp.close()
client.close()
print(json.dumps(state))
