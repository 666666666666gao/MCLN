import datetime,json,os
from pathlib import Path
import paramiko
b=Path(__file__).resolve().parents[1]/'refine-logs/eg3dvg_nr3d_transfer_20260920_v1';b.mkdir(exist_ok=True);r='/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1'
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30);s=c.open_sftp();names=s.listdir(r)
rec={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
for name in ['preflight.exit','formal.exit','audit.exit','controller.exit','train_input_prep.exit','train_dataset_preflight.exit','launch.json']:
 if name in names:
  raw=s.open(r+'/'+name,'r').read().decode();rec[name]=json.loads(raw) if name.endswith('.json') else raw.strip()
for name in ['preflight.log','formal.log','audit.log','controller.log','train_input_prep.log','train_dataset_preflight.log']:
 if name in names:rec[name]=s.open(r+'/'+name,'r').read().decode().splitlines()[-3:]
for stage in ['preflight','formal']:
 if stage in names:
  for name in ['receipt.json','audit.json']:
   if name in s.listdir(r+'/'+stage):rec[stage+'/'+name]=json.loads(s.open(r+'/'+stage+'/'+name,'r').read().decode())
for key,cmd in [('process','ps -p 4408 -o pid,etime,args --no-headers'),('gpu','nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'),('memory','free -m'),('disk','df -B1 --output=avail /root/autodl-tmp')]:
 _,o,e=c.exec_command(cmd,timeout=30);rec[key]=o.read().decode().strip()
s.close();c.close();(b/'observation_latest.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))



