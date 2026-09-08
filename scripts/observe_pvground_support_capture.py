"""Read the same support-capture queue; never launch or retry a model job."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_support_capture_20260908_v1'
archive=repo/'refine-logs/pvground_support_capture_20260908_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();names=s.listdir(root)
record={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'files':names}
for name in ['controller.pid','queue.pid','controller.exit','capture.exit','queue.log','capture.log']:
    if name in names:
        with s.open(root+'/'+name,'rb') as f:raw=f.read()
        (archive/name).write_bytes(raw)
        record[name]=raw.decode()[-5000:] if name.endswith('.log') else raw.decode().strip()
pids=[record[name] for name in ['controller.pid','queue.pid'] if name in record]
if pids:
    _,out,err=c.exec_command('ps -p '+shlex.quote(','.join(pids))+' -o pid,ppid,etimes,args',timeout=30)
    record['processes']=out.read().decode();record['ps_exit']=out.channel.recv_exit_status()
    assert not err.read()
if 'observed_replay' in names:
    outputs=s.listdir(root+'/observed_replay')
    record['output_files']=outputs
    (archive/'observed_replay').mkdir(exist_ok=True)
    for name in ['receipt.json','support_receipt.json']:
        if name in outputs:
            s.get(root+'/observed_replay/'+name,str(archive/'observed_replay'/name))
            record[name]=json.loads((archive/'observed_replay'/name).read_bytes())
    if 'support_receipt.json' in outputs:
        reports=archive/'observed_replay/support';reports.mkdir(exist_ok=True)
        for name in s.listdir(root+'/observed_replay/support'):
            if name.endswith('.json'):s.get(root+'/observed_replay/support/'+name,str(reports/name))
(archive/'observation_latest.json').write_text(json.dumps(record,indent=2)+'\n')
s.close();c.close();print(json.dumps(record))
