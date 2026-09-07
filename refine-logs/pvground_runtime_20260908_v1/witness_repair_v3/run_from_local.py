import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/pvground_runtime_20260908_v1/witness_repair_v3'
root='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
for name in ('run.log','controller.exit'):
    sftp.get(root+'/build_repair_v2/'+name,str(archive/('previous_'+name)))
sftp.mkdir(root+'/witness_repair_v3')
for name in ('env_spec.json','verify_pvground_runtime.py','finalize_runtime_v3.py'):
    sftp.put(str(archive/name),root+'/'+name)
for name in ('env_spec.json','verify_pvground_runtime.py','finalize_runtime_v3.py','voxel_reference_probe.json','spec_sha256.txt'):
    sftp.put(str(archive/name),root+'/witness_repair_v3/'+name)
code="""import subprocess
from pathlib import Path
root=Path(ROOT)
with (root/'witness_repair_v3/run.log').open('x') as log:
    result=subprocess.run(['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock','/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'finalize_runtime_v3.py')],stdout=log,stderr=subprocess.STDOUT)
(root/'witness_repair_v3/controller.exit').write_text(str(result.returncode)+'\\n')
print((root/'witness_repair_v3/run.log').read_text())
raise SystemExit(result.returncode)
""".replace('ROOT',repr(root))
_,stdout,stderr=client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(code),timeout=60)
raw=stdout.read();error=stderr.read();exitcode=stdout.channel.recv_exit_status()
(archive/'stdout.log').write_bytes(raw);(archive/'stderr.log').write_bytes(error)
print(raw.decode(),end='');print(error.decode(),end='')
for name in ('run.log','controller.exit'):
    sftp.get(root+'/witness_repair_v3/'+name,str(archive/name))
if exitcode==0:
    for name in ('build_receipt.json','kernel_receipt.json','base_packages_after.json'):
        sftp.get(root+'/'+name,str(archive/name))
(archive/'run_from_local.py').write_bytes(Path(__file__).read_bytes())
sftp.close();client.close();assert exitcode==0
