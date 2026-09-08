"""Collect the fixed interface check without launching or changing it."""
import json, os
from pathlib import Path
import paramiko

repo = Path(__file__).resolve().parents[1]
root = '/root/autodl-tmp/mcln_pvground_observation_interface_20260909_v2'
archive = repo/'refine-logs/pvground_observation_interface_20260909_v2'
c = paramiko.SSHClient(); c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
names = s.listdir(root)
for name in ['controller.pid', 'run.log', 'controller.exit']:
    if name in names:
        s.get(root+'/'+name, str(archive/name))
if 'controller.exit' in names:
    code = int((archive/'controller.exit').read_text())
    if code == 0:
        (archive/'results').mkdir(exist_ok=True)
        for name in s.listdir(root+'/results'):
            assert s.stat(root+'/results/'+name).st_size < 10*1024**2
            s.get(root+'/results/'+name, str(archive/'results'/name))
        receipt = json.loads((archive/'results/receipt.json').read_bytes())
        print(json.dumps({key: receipt[key] for key in ['status', 'optimizer_steps', 'train_forwards', 'eval_forwards', 'source_query_read', 'module_sha256', 'time_cst']}))
    else:
        print((archive/'run.log').read_text()[-6000:])
    print(json.dumps({'exit_code': code}))
else:
    _,out,err=c.exec_command('ps -p 29432 -o pid=,args=',timeout=30)
    process=out.read().decode().strip()
    assert out.channel.recv_exit_status()==0 and root+'/controller.py' in process
    print(json.dumps({'process':process,'status':'running'}))
    print((archive/'run.log').read_text()[-3000:])
s.close(); c.close()
