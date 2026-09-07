import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
root='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_v1'
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
archive=repo/'refine-logs/pvground_scanrefer_finetune_20260908_v1'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
interface_root='/root/autodl-tmp/mcln_pvground_training_interface_20260908_v1'
with sftp.open(interface_root+'/controller.exit','r') as stream:assert stream.read().decode().strip()=='0'
with sftp.open(interface_root+'/results/receipt.json','rb') as stream:interface=json.loads(stream.read())
assert interface['status']=='pass' and interface['optimizer_steps']==2
probe="import json,os,shutil,socket,subprocess; print(json.dumps({'uid':os.getuid(),'host':socket.gethostname(),'disk_free':shutil.disk_usage('/root/autodl-tmp').free,'gpu_mib':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip()}))"
_,stdout,stderr=client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(probe),timeout=30)
preflight=json.loads(stdout.read());assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
assert preflight['uid']==0 and preflight['disk_free']>3*1024**3 and int(preflight['gpu_mib'])<500
files={'train.py':repo/'scripts/run_pvground_scanrefer_finetune.py','plan.md':repo/'docs/PVG_SCANREFER_FINETUNE_PLAN_2026-09-08.md'}
spec={'root':root,'runtime':runtime,'input_manifest':'/root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1/input_manifest.json',
    'reference_fixtures':'/root/autodl-tmp/mcln_pvground_train_fixtures_20260908_v2/fixtures',
    'training_interface_receipt':interface_root+'/results/receipt.json','env_spec_sha256':interface['env_spec_sha256'],
    'checkpoint_sha256':interface['checkpoint_sha256'],'batch_size':8,'fit_passes':1,'seed':2027,
    'lr':1e-5,'lr_backbone':1e-5,'primary_mode':'bbs','formal_rows':0,
    'files':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in files.items()}}
controller='''import hashlib,json,os,subprocess
from pathlib import Path
root=Path(ROOT)
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
runtime=Path(spec['runtime'])
env=json.loads((runtime/'env_spec.json').read_bytes())['env']
command=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',str(runtime/'venv/bin/python'),'-u',str(root/'train.py'),'--spec',str(root/'spec.json')]
result=subprocess.run(command,cwd=str(runtime/'PV-Ground'),env=dict(os.environ,**env))
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''.replace('ROOT',repr(root))
compile(controller,'controller.py','exec')
archive.mkdir();sftp.mkdir(root)
data={'spec.json':(json.dumps(spec,indent=2)+'\n').encode(),'controller.py':controller.encode(),'preflight.json':(json.dumps(preflight,indent=2)+'\n').encode()}
data.update({k:v.read_bytes() for k,v in files.items()})
for name,raw in data.items():
    assert os.environ['MCLN_SSH_PASSWORD'].encode() not in raw
    (archive/name).write_bytes(raw)
    with sftp.open(root+'/'+name,'wx') as stream:stream.write(raw)
    with sftp.open(root+'/'+name,'rb') as stream:assert stream.read()==raw
screen='mcln_pvg_scan_finetune_v1'
inner='exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
_,stdout,stderr=client.exec_command('screen -dmS '+screen+' bash -c '+shlex.quote(inner),timeout=30)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
_,stdout,stderr=client.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=stdout.read().decode().strip();assert stdout.channel.recv_exit_status()==0 and process,stderr.read().decode()
launch={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'process':process,
    'screen':screen,'first_check_after_seconds':240,'fit_steps':3723,'fit_rows':29778,'holdout_rows':6887,
    'duration_estimate':'several hours; revise from measured batch8 time','formal_rows':0,
    'spec_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
raw=(json.dumps(launch,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with sftp.open(root+'/launch.json','wx') as stream:stream.write(raw)
(archive/'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
sftp.close();client.close();print('PVG_SCANREFER_FINETUNE_LAUNCHED '+json.dumps(launch),flush=True)
