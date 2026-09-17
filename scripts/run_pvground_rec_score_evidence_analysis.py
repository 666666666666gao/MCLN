"""Execute one independent CPU recount after the original forward run has exited."""
import hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
name='pvground_rec_score_evidence_20260917_v1'
root='/root/autodl-tmp/mcln_'+name;archive=repo/'refine-logs'/name
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(root+'/controller.exit','rb') as f:assert f.read().strip()==b'0'
raw=(repo/'scripts/analyze_pvground_rec_score_evidence.py').read_bytes();compile(raw,'analysis.py','exec')
with s.open(root+'/analysis.py','wx') as f:f.write(raw)
(archive/'analysis.py').write_bytes(raw)
with s.open(root+'/analysis.py','rb') as f:assert f.read()==raw
command=['env','CUDA_VISIBLE_DEVICES=','/root/miniconda3/envs/bdetr/bin/python','-u',root+'/analysis.py','--root',root]
_,out,err=c.exec_command(' '.join(map(shlex.quote,command)),timeout=60)
stdout=out.read();stderr=err.read();code=out.channel.recv_exit_status()
for name,raw in [('analysis.log',stdout),('analysis.stderr',stderr),('analysis.exit',(str(code)+'\n').encode())]:
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wx') as f:f.write(raw)
assert code==0,stderr.decode()
with s.open(root+'/analysis.json','rb') as f:raw=f.read()
(archive/'analysis.json').write_bytes(raw)
print(raw.decode());s.close();c.close()
