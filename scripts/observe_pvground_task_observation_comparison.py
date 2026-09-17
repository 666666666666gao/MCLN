"""Read the original D comparison controller and saved CPU results only."""
import datetime
import json
import os
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
folder='pvground_task_observation_comparison_20260917_v1'
archive=repo/'refine-logs'/folder
root='/root/autodl-tmp/mcln_'+folder
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();names=s.listdir(root)
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),results={})
files=['run.log','controller.exit']
files += [stage+'_'+label+'_D.'+extension for stage in ['initial','terminal']
          for label in ['A','B','C'] for extension in ['json','exit','log']]
for name in files:
    if name in names:
        assert s.stat(root+'/'+name).st_size<10*1024**2
        s.get(root+'/'+name,str(archive/name))
        if name.endswith('.json'):
            result=json.loads((archive/name).read_bytes())
            record['results'][name]={key:result[key] for key in
                ['status','stage','rows','metrics','transitions','full256_oracle','export_byte_exact']}
        elif name.endswith('.exit'):
            record['results'][name]=int((archive/name).read_text())
launch=json.loads((archive/'launch.json').read_bytes());pid=int(launch['process'].split()[0])
_,out,err=c.exec_command('ps -p '+str(pid)+' -o pid=,args=',timeout=30)
record['actual_process']=out.read().decode().strip()
assert out.channel.recv_exit_status() in [0,1],err.read().decode()
if 'controller.exit' not in names:
    assert root+'/controller.py' in record['actual_process']
record['last_log_line']=(archive/'run.log').read_text().splitlines()[-1:]
s.close();c.close()
(archive/'observation_latest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
