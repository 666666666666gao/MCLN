import hashlib,json,os,posixpath,shlex
from pathlib import Path
import paramiko
local=Path('C:/Users/gb/.codex_mcln_g0_20260905/scripts')
r='/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1'
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
python='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
pause="import os,signal;from pathlib import Path;p=Path(%r);assert b'launch_when_inspected.py' in Path('/proc/1754/cmdline').read_bytes();os.kill(1754,signal.SIGSTOP);print('acceptance_waiter_paused');assert not (p/'spec.json').exists()"%r
_,out,err=c.exec_command(python+' -c '+shlex.quote(pause),timeout=30)
stdout=out.read().decode();stderr=err.read().decode();code=out.channel.recv_exit_status();print(stdout,flush=True);assert code==0,stderr
assert 'spec.json' not in s.listdir(r)
record={'training_steps':0,'formal_started':False,'changes':'Save all 256 raw/native boxes and both native scores; independently verify selected rows against array.'}
for name,dest in [('evaluate_eg3dvg_pretrained.py','evaluate.py'),('audit_eg3dvg_pretrained.py','audit.py')]:
 raw=(local/name).read_bytes();compile(raw,name,'exec')
 path=r+'/'+dest
 with s.open(path,'rb') as f:old=f.read()
 archive=r+'/'+dest+'.before_candidates'
 assert posixpath.basename(archive) not in s.listdir(r)
 s.rename(path,archive);s.put(str(local/name),path)
 with s.open(path,'rb') as f:assert f.read()==raw
 record[dest]={'before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(raw).hexdigest()}
with s.open(r+'/candidate_export_update.json','w') as f:f.write(json.dumps(record,indent=2))
_,out,err=c.exec_command(python+" -c 'import os,signal;os.kill(1754,signal.SIGCONT)'",timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
Path(__file__).with_name('candidate_export_update.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record));s.close();c.close()
