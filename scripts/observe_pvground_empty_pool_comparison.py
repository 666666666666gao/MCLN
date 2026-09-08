"""Read the existing CPU comparison process and its immutable stage reports."""
import datetime,json,os,shlex
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_empty_pool_comparison_20260908_v1'
archive=repo/'refine-logs/pvground_empty_pool_comparison_20260908_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();names=s.listdir(root)
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),files=names)
for name in ['controller.pid','controller.exit','run.log','initial.exit','initial.log','initial.json','terminal.exit','terminal.log','terminal.json']:
    if name in names:
        assert s.stat(root+'/'+name).st_size<10*1024**2
        s.get(root+'/'+name,str(archive/name))
        if name.endswith('.json'):
            result=json.loads((archive/name).read_bytes())
            record[name]={key:result[key] for key in ['status','stage','rows','metrics','transitions','full256_oracle','formal_rows']}
        elif name.endswith('.exit'):
            record[name]=int((archive/name).read_text())
        elif name.endswith('.log'):
            record[name]=(archive/name).read_text().splitlines()[-3:]
if 'controller.exit' not in names:
    pattern='^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'
    _,out,err=c.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
    record['live_process']=out.read().decode().strip()
    assert out.channel.recv_exit_status()==0 and record['live_process'],err.read().decode()
raw=(json.dumps(record,indent=2)+'\n').encode()
(archive/'observation_latest.json').write_bytes(raw)
s.close();c.close();print(json.dumps(record),flush=True)
