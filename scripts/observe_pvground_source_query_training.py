"""Read the original source-query trial handles and bounded result files, without launching."""
import datetime,json,os,shlex
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
roots=['/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_sourcequery_v1',
       '/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_sourcequery_v1']
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();record={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'jobs':{}}
for root in roots:
    archive=repo/'refine-logs'/Path(root).name.replace('mcln_','',1)
    names=s.listdir(root);entry={}
    for name in ['run.log','controller.exit','capacity.json','receipt.json','audit.json','audit.exit','initial_audit.json','train.jsonl']:
        if name in names:
            assert s.stat(root+'/'+name).st_size<10*1024**2
            s.get(root+'/'+name,str(archive/name))
            if name.endswith('.json'):entry[name]=json.loads((archive/name).read_bytes())
            elif name.endswith('.exit'):entry[name]=int((archive/name).read_text())
    if 'run.log' in names:entry['last_log_line']=(archive/'run.log').read_text().splitlines()[-1:]
    for stage in ['initial','terminal']:
        if stage in names and 'receipt.json' in s.listdir(root+'/'+stage):
            (archive/stage).mkdir(exist_ok=True)
            s.get(root+'/'+stage+'/receipt.json',str(archive/stage/'receipt.json'))
            entry[stage]=json.loads((archive/stage/'receipt.json').read_bytes())
    launch=json.loads((archive/'launch.json').read_bytes());pid=int(launch['process'].split()[0])
    _,out,err=c.exec_command('ps -p '+str(pid)+' -o pid=,args=',timeout=30)
    entry['actual_process']=out.read().decode().strip()
    assert out.channel.recv_exit_status() in [0,1],err.read().decode()
    if 'controller.exit' not in names:assert root+'/controller.py' in entry['actual_process']
    record['jobs'][root]=entry
_,out,err=c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader',timeout=30)
record['gpu_processes']=out.read().decode().strip();assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('df -B1 --output=avail /root/autodl-tmp',timeout=30)
record['disk_free']=int(out.read().decode().splitlines()[-1]);assert out.channel.recv_exit_status()==0,err.read().decode()
s.close();c.close()
(repo/'refine-logs/pvground_scanrefer_finetune_20260908_sourcequery_v1/observation_latest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
