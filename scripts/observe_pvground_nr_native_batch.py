"""Collect the existing CPU-only native batch preparation; never restart it."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_nr_native_batch_20260908_v1'
archive=repo/'refine-logs/pvground_nr_native_batch_20260908_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();names=s.listdir(root)
record={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'files':names}
for name in ['controller.pid','controller.exit','run.log','receipt.json']:
    if name in names:
        with s.open(root+'/'+name,'rb') as f:raw=f.read()
        (archive/name).write_bytes(raw)
        record[name]=json.loads(raw) if name.endswith('.json') else raw.decode()[-6000:]
probe="import json,subprocess,shutil;print(json.dumps({'free_bytes':shutil.disk_usage('/root/autodl-tmp').free,'processes':[v for v in subprocess.check_output(['ps','-eo','pid,ppid,etimes,args']).decode().splitlines() if 'mcln_pvground_nr_native_batch_20260908_v1/' in v and ' -c ' not in v]}))"
_,out,err=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(probe),timeout=30)
record['state']=json.loads(out.read());assert out.channel.recv_exit_status()==0,err.read().decode()
(archive/'observation_latest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));s.close();c.close()
