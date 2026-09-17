import datetime,json,os
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/pvground_task_checkpoint_check_20260917_v1'
root='/root/autodl-tmp/mcln_pvground_task_checkpoint_check_20260917_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();names=s.listdir(root)
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
for name in ['run.log','check.log','check.exit','controller.exit','receipt.json']:
    if name in names:
        assert s.stat(root+'/'+name).st_size<1024**2
        s.get(root+'/'+name,str(archive/name))
        if name.endswith('.exit'):record[name]=int((archive/name).read_text())
        if name=='receipt.json':
            result=json.loads((archive/name).read_bytes())
            record['receipt']={k:result[k] for k in ['status','step','fit_rows','added_tensors','expanded_tensors','delta_tensors','task_query_max_absolute_change','strict_restore_exact','torch_cuda_initialized','checkpoint_snapshot_bytes']}
            record['all_added_states_changed']=all(value>0 for value in result['added_max_absolute_change'].values())
pid=int(json.loads((archive/'launch.json').read_bytes())['process'].split()[0])
_,out,err=c.exec_command('ps -p '+str(pid)+' -o pid=,args=',timeout=30)
record['actual_process']=out.read().decode().strip();assert out.channel.recv_exit_status() in [0,1],err.read().decode()
if 'controller.exit' not in names:assert root+'/controller.py' in record['actual_process']
s.close();c.close()
(archive/'observation_latest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
(archive/'observe_from_local.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(record),flush=True)
