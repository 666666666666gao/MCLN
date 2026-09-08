"""Run the CPU-only initial-output comparison once after its input receipt exists."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_vsaorder_v1'
archive=repo/'refine-logs/pvground_scanrefer_finetune_20260908_vsaorder_v1'
source=(repo/'scripts/compare_pvground_vsa_initial.py').read_bytes()
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(root+'/initial/receipt.json','rb') as f:receipt=json.loads(f.read())
assert receipt['status']=='pass' and receipt['rows']==6887
assert 'initial_vsa_comparison.json' not in s.listdir(root)
with s.open(root+'/compare_initial.py','wb') as f:f.write(source)
with s.open(root+'/compare_initial.py','rb') as f:assert f.read()==source
(archive/'compare_initial.py').write_bytes(source)
command='/root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/compare_initial.py')
_,out,err=c.exec_command(command,timeout=60)
raw=out.read();error=err.read();code=out.channel.recv_exit_status()
(archive/'compare_initial.log').write_bytes(raw+error)
(archive/'compare_initial.exit').write_text(str(code)+'\n')
assert code==0,error.decode()
with s.open(root+'/initial_vsa_comparison.json','rb') as f:result=f.read()
(archive/'initial_vsa_comparison.json').write_bytes(result)
proof=dict(script_sha256=hashlib.sha256(source).hexdigest(),result_sha256=hashlib.sha256(result).hexdigest(),
    cpu_only=True,model_forwards=0,formal_rows=0,exit_code=code)
(archive/'compare_initial_execution.json').write_text(json.dumps(proof,indent=2)+'\n')
print(result.decode());s.close();c.close()
