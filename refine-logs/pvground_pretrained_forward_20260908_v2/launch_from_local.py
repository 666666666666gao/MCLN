import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
fixtures='/root/autodl-tmp/mcln_pvground_train_fixtures_20260908_v2'
root='/root/autodl-tmp/mcln_pvground_pretrained_forward_20260908_v2'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(fixtures+'/controller.exit','r') as stream:assert stream.read().decode().strip()=='0'
with sftp.open(runtime+'/build_receipt.json','rb') as stream:build=json.loads(stream.read())
with sftp.open(runtime+'/agent_kernel_receipt.json','rb') as stream:agent=json.loads(stream.read())
assert build['status']==agent['status']=='pass'
archive=repo/'refine-logs/pvground_pretrained_forward_20260908_v2';archive.mkdir()
source=(repo/'scripts/verify_pvground_pretrained_forward.py').read_bytes()
spec={'root':root,'runtime':runtime,'fixtures':fixtures+'/fixtures','env_spec_sha256':build['spec_sha256'],
    'script_sha256':hashlib.sha256(source).hexdigest(),'seed':2027,'batch_size':2,'batches':2,'training_steps':0,'formal_rows':0}
raw=(json.dumps(spec,indent=2)+'\n').encode();(archive/'spec.json').write_bytes(raw)
sftp.mkdir(root)
for name,data in [('spec.json',raw),('verify_forward.py',source)]:
    with sftp.open(root+'/'+name,'wx') as stream:stream.write(data)
runner="""import json,os,subprocess
from pathlib import Path
root=Path(ROOT)
spec=json.loads((root/'spec.json').read_bytes())
runtime=Path(spec['runtime'])
env=json.loads((runtime/'env_spec.json').read_bytes())['env']
command=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',str(runtime/'venv/bin/python'),'-u',str(root/'verify_forward.py'),'--runtime',str(runtime),'--fixtures',spec['fixtures'],'--output',str(root/'results')]
result=subprocess.run(command,cwd=str(runtime/'PV-Ground'),env=dict(os.environ,**env))
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
""".replace('ROOT',repr(root))
compile(runner,'forward_controller','exec')
(archive/'controller.py').write_bytes(runner.encode());(archive/'verify_forward.py').write_bytes(source)
with sftp.open(root+'/controller.py','wx') as stream:stream.write(runner.encode())
inner='exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
_,stdout,stderr=client.exec_command('screen -dmS mcln_pvg_forward_v2 bash -c '+shlex.quote(inner),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
_,stdout,stderr=client.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=stdout.read().decode();assert stdout.channel.recv_exit_status()==0 and process.strip(),stderr.read().decode()
receipt={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'process':process.strip(),'screen':'mcln_pvg_forward_v2','estimate_seconds':[60,180],'optimizer_steps':0,'formal_rows':0}
raw=(json.dumps(receipt,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with sftp.open(root+'/launch.json','wx') as stream:stream.write(raw)
(archive/'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
sftp.close();client.close();print('PVG_FORWARD_LAUNCHED '+json.dumps(receipt),flush=True)
