import datetime,hashlib,json,os,time
from pathlib import Path
import paramiko
b=Path(__file__).resolve().parent;p=b/'EG3DVG_scanrefer_official.pth';r='/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1'
assert p.is_file() and p.stat().st_size>800000000
h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
rec={'source':'https://drive.google.com/file/d/1WcXEVCWgJL6Qfmbqg7ach_2RIUnfULNZ/view','sha256':h.hexdigest(),'bytes':p.stat().st_size,'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
(b/'checkpoint_download.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();assert 'official_scanrefer.pth' not in s.listdir(r)
t=time.time();s.put(str(p),r+'/official_scanrefer.pth')
s.put(str(b/'checkpoint_download.json'),r+'/checkpoint_download.json')
rec['transfer_seconds']=time.time()-t
s.close();c.close();print(json.dumps(rec))
