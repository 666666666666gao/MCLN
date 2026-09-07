import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
root = '/root/autodl-tmp/mcln_pvground_training_interface_20260908_v1'
manifest = '/root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1/input_manifest.json'
reference = '/root/autodl-tmp/mcln_pvground_train_fixtures_20260908_v2/fixtures'
archive = repo / 'refine-logs/pvground_training_interface_20260908_v1'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(runtime+'/build_receipt.json','rb') as stream:
    build = json.loads(stream.read())
assert build['status']=='pass' and build['spec_sha256']=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
_, stdout, stderr = client.exec_command('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits', timeout=30)
gpu = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status()==0 and int(gpu)<500
files = {'export_fixtures.py':repo/'scripts/export_pvground_train_fixtures.py',
    'check_training.py':repo/'scripts/check_pvground_training_interface.py',
    'plan.md':repo/'docs/PVG_TRAINING_INTERFACE_PLAN_2026-09-08.md'}
spec = {'root':root,'runtime':runtime,'manifest':manifest,'reference_fixtures':reference,
    'env_spec_sha256':build['spec_sha256'],'files':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in files.items()},
    'seed':2027,'training_row_ids':[0,173,237,455],'batch_size':2,'optimizer_steps':2,
    'eval_forwards':2,'formal_rows':0,'new_checkpoints':0,'lr':1e-5,'backbone_lr':1e-5}
runner = '''import hashlib,json,os,subprocess
from pathlib import Path
root=Path(ROOT)
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
runtime=Path(spec['runtime'])
env=json.loads((runtime/'env_spec.json').read_bytes())['env']
export=['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'export_fixtures.py'),'--manifest',spec['manifest'],'--output',str(root/'fixtures'),'--training-labels','--reference-fixtures',spec['reference_fixtures']]
result=subprocess.run(export,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'))
(root/'export.exit').write_text(str(result.returncode)+'\\n')
if result.returncode:
    (root/'controller.exit').write_text(str(result.returncode)+'\\n')
    raise SystemExit(result.returncode)
check=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',str(runtime/'venv/bin/python'),'-u',str(root/'check_training.py'),'--runtime',str(runtime),'--fixtures',str(root/'fixtures'),'--output',str(root/'results')]
result=subprocess.run(check,cwd=str(runtime/'PV-Ground'),env=dict(os.environ,**env))
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''.replace('ROOT',repr(root))
compile(runner,'controller.py','exec')
archive.mkdir()
sftp.mkdir(root)
data = {'spec.json':(json.dumps(spec,indent=2)+'\n').encode(),'controller.py':runner.encode()}
data.update({k:v.read_bytes() for k,v in files.items()})
for name,raw in data.items():
    assert os.environ['MCLN_SSH_PASSWORD'].encode() not in raw
    (archive/name).write_bytes(raw)
    with sftp.open(root+'/'+name,'wx') as stream:stream.write(raw)
    with sftp.open(root+'/'+name,'rb') as stream:assert stream.read()==raw
screen = 'mcln_pvg_train_iface_v1'
inner = 'exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
_, stdout, stderr = client.exec_command('screen -dmS '+screen+' bash -c '+shlex.quote(inner),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
_, stdout, stderr = client.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=stdout.read().decode().strip()
assert stdout.channel.recv_exit_status()==0 and process,stderr.read().decode()
launch = {'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'process':process,'screen':screen,'estimate_seconds':[120,240],'poll_after_seconds':180,
    'spec_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
raw=(json.dumps(launch,indent=2)+'\n').encode()
(archive/'launch.json').write_bytes(raw)
with sftp.open(root+'/launch.json','wx') as stream:stream.write(raw)
(archive/'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
sftp.close()
client.close()
print('PVG_TRAINING_INTERFACE_LAUNCHED '+json.dumps(launch),flush=True)
