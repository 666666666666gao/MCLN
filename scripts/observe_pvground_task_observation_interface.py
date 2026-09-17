"""Collect the original D interface handle and receipts without relaunching."""
import json, os
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_task_observation_interface_20260917_v1'
archive=repo/'refine-logs/pvground_task_observation_interface_20260917_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();names=s.listdir(root);record={}
for name in ['controller.pid','unit.json','unit.log','unit.exit','run.log','check.exit','controller.exit']:
    if name in names:
        s.get(root+'/'+name,str(archive/name))
        if name.endswith('.json'):record[name]=json.loads((archive/name).read_bytes())
        elif name.endswith('.exit'):record[name]=int((archive/name).read_text())
        elif name.endswith('.log'):record[name]=(archive/name).read_text()[-3000:]
pid=int(json.loads((archive/'launch.json').read_bytes())['process'].split()[0])
_,out,err=c.exec_command('ps -p '+str(pid)+' -o pid=,args=',timeout=30)
record['process']=out.read().decode().strip()
assert out.channel.recv_exit_status() in [0,1],err.read().decode()
if 'controller.exit' not in names:assert root+'/controller.py' in record['process']
if 'results' in names:
    (archive/'results').mkdir(exist_ok=True)
    for name in s.listdir(root+'/results'):
        assert s.stat(root+'/results/'+name).st_size<10*1024**2
        s.get(root+'/results/'+name,str(archive/'results'/name))
    if (archive/'results/receipt.json').exists():
        receipt=json.loads((archive/'results/receipt.json').read_bytes())
        record['receipt']={k:v for k,v in receipt.items() if k not in ['changed_parameters','changed_buffers','steps']}
s.close();c.close();print(json.dumps(record),flush=True)
