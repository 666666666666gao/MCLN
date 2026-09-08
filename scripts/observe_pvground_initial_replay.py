"""Collect small initial-replay receipts and worker failures without GPU traces."""
import datetime
import json
import os
from pathlib import Path
import paramiko

c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();root='/root/autodl-tmp/mcln_pvground_initial_replay_20260908_v1'
archive=Path(__file__).resolve().parents[1]/'refine-logs/pvground_initial_replay_20260908_v1'
names=s.listdir(root)
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),files=names)
for name in ['input.log','input_controller.pid','input_controller.exit','queue.pid','controller.pid',
             'controller.exit','queue.log','comparison.json','process_a.log','process_b.log','process_a.exit','process_b.exit']:
    if name in names:
        with s.open(root+'/'+name,'rb') as f:f.prefetch();raw=f.read()
        (archive/name).write_bytes(raw)
        record[name]=raw.decode().splitlines()[-6:] if name.endswith('.log') else raw.decode()
for phase in ['inputs','process_a','process_b']:
    if phase in names and 'receipt.json' in s.listdir(root+'/'+phase):
        with s.open(root+'/'+phase+'/receipt.json','rb') as f:raw=f.read()
        (archive/phase).mkdir(exist_ok=True);(archive/phase/'receipt.json').write_bytes(raw)
        receipt=json.loads(raw)
        record[phase]={k:v for k,v in receipt.items() if k not in ['rows','cases']}
_,out,err=c.exec_command('ps -eo pid,ppid,etimes,args',timeout=30)
record['processes']=[line for line in out.read().decode().splitlines() if root in line]
assert out.channel.recv_exit_status()==0 and not err.read()
(archive/'observation_latest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));s.close();c.close()
