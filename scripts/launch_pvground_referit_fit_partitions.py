"""Launch annotation-only fixed partition preparation while Scan training continues."""
import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_referit_fit_partitions_20260917_v1'
archive=repo/'refine-logs'/Path(root).name.replace('mcln_','',1)
old_path=repo/'refine-logs/text_position_l1_20260905_v2/input_manifest.json'
old=json.loads(old_path.read_bytes())
reference=dict(split_salt=old['split_salt'],row_ids=old['row_ids'],original_manifest_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest())
files={'prepare.py':(repo/'scripts/prepare_pvground_referit_fit_partitions.py').read_bytes(),'nr_reference.json':(json.dumps(reference,separators=(',',':'))+'\n').encode()}
scan=json.loads((repo/'refine-logs/pvground_scanrefer_finetune_20260917_rec_competition_v1/spec.json').read_bytes())
spec=json.loads((repo/'refine-logs/pvground_sr_native_batch_20260917_v1/spec.json').read_bytes())
spec=dict(runtime=spec['runtime'],selection_root=spec['selection_root'],selection_manifest_sha256=spec['selection_manifest_sha256'],
 scan_input_manifest=scan['input_manifest'],scan_input_manifest_sha256=hashlib.sha256((repo/'refine-logs/pvground_scanrefer_finetune_20260917_rec_competition_v1/input_manifest.json').read_bytes()).hexdigest(),
 script_sha256=hashlib.sha256(files['prepare.py']).hexdigest(),reference_sha256=hashlib.sha256(files['nr_reference.json']).hexdigest())
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
files['controller.py']=b'''import json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\\n')
spec=json.loads((root/'spec.json').read_bytes());runtime=Path(spec['runtime'])
env=dict(os.environ,**json.loads((runtime/'env_spec.json').read_bytes())['env'])
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'run.log').open('w') as log:
 result=subprocess.run([str(runtime/'venv/bin/python'),'-u',str(root/'prepare.py'),'--spec',str(root/'spec.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30);s=c.open_sftp()
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
 if name.endswith('.py'):compile(raw,name,'exec')
 (archive/name).write_bytes(raw)
 with s.open(root+'/'+name,'wx') as f:f.write(raw)
inner='exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')
_,out,err=c.exec_command('screen -dmS mcln_pvg_referit_partitions_v1 bash -c '+shlex.quote(inner),timeout=30);assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30);process=out.read().decode().strip();assert process and out.channel.recv_exit_status()==0
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),process=process,scope='native annotations, no text parsing, point sampling, model or optimizer')
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wx') as f:f.write(raw)
print(json.dumps(record));s.close();c.close()