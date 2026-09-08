"""Read original CPU analysis/checkpoint handles; never launch or alter training."""
import datetime,json,os
from pathlib import Path
import paramiko
repo=Path(__file__).resolve().parents[1]
folders=['pvground_observation_checkpoint_audit_20260909_v1','pvground_observation_comparison_20260909_v1']
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),jobs={})
for folder in folders:
    root='/root/autodl-tmp/mcln_'+folder;archive=repo/'refine-logs'/folder
    names=s.listdir(root);entry={}
    for name in ['run.log','check.log','controller.exit','check.exit','receipt.json','initial_A_C.json','initial_B_C.json','initial_A_C.exit','initial_B_C.exit','initial_A_C.log','initial_B_C.log','terminal_A_C.json','terminal_B_C.json','terminal_A_C.exit','terminal_B_C.exit','terminal_A_C.log','terminal_B_C.log']:
        if name in names:
            assert s.stat(root+'/'+name).st_size<10*1024**2
            s.get(root+'/'+name,str(archive/name))
            if name.endswith('.json'):
                item=json.loads((archive/name).read_bytes())
                entry[name]={k:v for k,v in item.items() if k not in ['scene_transitions','added_max_absolute_change']}
            elif name.endswith('.exit'):entry[name]=int((archive/name).read_text())
    if 'run.log' in names:entry['last_log_line']=(archive/'run.log').read_text().splitlines()[-1:]
    pid=int(json.loads((archive/'launch.json').read_bytes())['process'].split()[0])
    _,out,err=c.exec_command('ps -p '+str(pid)+' -o pid=,args=',timeout=30)
    entry['actual_process']=out.read().decode().strip();assert out.channel.recv_exit_status() in [0,1],err.read().decode()
    if 'controller.exit' not in names:assert root+'/controller.py' in entry['actual_process']
    record['jobs'][folder]=entry
s.close();c.close()
(repo/'refine-logs'/folders[0]/'observation_latest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
