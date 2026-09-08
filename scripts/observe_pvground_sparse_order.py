"""Collect the bounded sparse-backbone diagnostic and actual process state."""
import datetime
import json
import os
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_sparse_order_20260908_v1'
archive=repo/'refine-logs/pvground_sparse_order_20260908_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();names=s.listdir(root)
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),files=names)
for name in ['controller.pid','controller.exit','run.log']:
    if name in names:
        with s.open(root+'/'+name,'rb') as f:raw=f.read()
        (archive/name).write_bytes(raw)
        record[name]=raw.decode().splitlines()[-4:] if name.endswith('.log') else raw.decode()
if 'result' in names and 'receipt.json' in s.listdir(root+'/result'):
    with s.open(root+'/result/receipt.json','rb') as f:raw=f.read()
    (archive/'receipt.json').write_bytes(raw)
    result=json.loads(raw)
    record['result']={k:v for k,v in result.items() if k not in ['observations','differences']}
_,out,err=c.exec_command('ps -eo pid,ppid,etimes,args',timeout=30)
record['processes']=[line for line in out.read().decode().splitlines() if root in line]
assert out.channel.recv_exit_status()==0 and not err.read()
(archive/'observation_latest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='run.log'}));s.close();c.close()
