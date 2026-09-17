import datetime,hashlib,json,os
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/pvground_failed_weight_offload_20260917_v1';archive.mkdir()
local=Path('D:/Program Files/UserCache/gb/codex/tmp/mcln_failed_weight_archive_20260917');local.mkdir(exist_ok=True)
items=[('B','20260908_sourcequery','ec3a08674ccaa648ce808626b88411cf5d06df7c6e64f1a116171b36ffd1d1e8'),('C','20260909_observation','cad742d7ad3efa512435426f74328e298c48263f8dd470eef51448654bff0f93')]
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30);s=c.open_sftp()
def read(p):
 with s.open(p,'rb') as f:return f.read()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
froot='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_rec_competition_v1'
fspec=read(froot+'/spec.json')
_,out,err=c.exec_command('ps -eo pid,args',timeout=30);processes=out.read().decode();assert out.channel.recv_exit_status()==0
records=[]
for label,suffix,digest in items:
 root='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_'+suffix+'_v1'
 assert root not in processes and root.encode() not in fspec
 assert read(root+'/controller.exit').strip()==b'0'
 receipt=json.loads(read(root+'/receipt.json'));assert receipt['terminal_sha256']==digest and not receipt['primary_rec_nonregression']
 dest=local/(label+'_terminal.pth');assert not dest.exists()
 size=s.stat(root+'/terminal.pth').st_size
 print('COPY_BEGIN '+label+' '+str(size),flush=True)
 s.get(root+'/terminal.pth',str(dest))
 assert dest.stat().st_size==size and sha(dest)==digest
 entry=dict(label=label,source=root+'/terminal.pth',local_archive=str(dest),sha256=digest,bytes=size,local_hash_verified=True,deleted_remote=False)
 (archive/(label+'_planned.json')).write_text(json.dumps(entry,indent=2)+'\n')
 assert s.stat(root+'/terminal.pth').st_size==size
 s.remove(root+'/terminal.pth')
 assert 'terminal.pth' not in s.listdir(root)
 entry['deleted_remote']=True;records.append(entry)
 (archive/(label+'_receipt.json')).write_text(json.dumps(entry,indent=2)+'\n')
 print('ARCHIVED_AND_REMOVED '+label,flush=True)
_,out,err=c.exec_command('df -B1 --output=avail /root/autodl-tmp',timeout=30);free=int(out.read().decode().splitlines()[-1]);assert out.channel.recv_exit_status()==0
result=dict(status='complete',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),records=records,freed_bytes=sum(r['bytes'] for r in records),disk_free=free,protected_weights_unchanged=True)
(archive/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
s.close();c.close()