"""Queue the single Sr F batch probe after the actual REC-only Scan formal gate."""
import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_sr_rec_competition_preparation_20260917_v1'
archive=repo/'refine-logs'/Path(root).name.replace('mcln_','',1)
assert json.loads((archive/'receipt.json').read_bytes())['status']=='cpu_preparation_pass'
controller=r'''import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\n')
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
formal=Path(spec['formal_root'])
formal_spec=json.loads((formal/'spec.json').read_bytes())
time.sleep(max(0,datetime.datetime.fromisoformat(formal_spec['first_check_cst']).timestamp()-time.time()))
while not (formal/'controller.exit').is_file():
 process=Path('/proc')/str(spec['formal_pid'])/'cmdline'
 assert process.is_file() and (str(formal)+'/controller.py').encode() in process.read_bytes()
 time.sleep(300)
assert (formal/'controller.exit').read_text().strip()=='0'
decision=json.loads((formal/'decision.json').read_bytes())
if decision['status']=='skipped_primary_rec_regression':
 result=dict(status='skipped_scanrefer_rec',model_forwards=0,optimizer_steps=0)
else:
 assert decision['status']=='launching_fixed_formal'
 audit=json.loads((formal/'audit.json').read_bytes())
 assert audit['integrity_pass'] and audit['formal_rows']==9508 and audit['scanrefer_mask_gate'] is False
 if not audit['advance_to_nr3d_sr3d_rec']:
  result=dict(status='skipped_scanrefer_formal_rec',model_forwards=0,optimizer_steps=0,checks=audit['checks'])
 else:
  runtime=Path(formal_spec['runtime']);env=json.loads((runtime/'env_spec.json').read_bytes())['env']
  command=[str(runtime/'venv/bin/python'),'-u',str(root/'check.py'),'--formal-root',str(formal),'--output',str(root/'actual_probe')]
  with (root/'actual_probe.log').open('w') as log:
   run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,**env))
  assert run.returncode==0,'actual Sr F probe failed; inspect original log'
  result=dict(status='sr_real_batch_probe_complete',receipt=str(root/'actual_probe/receipt.json'),training_jobs_launched=0)
(root/'decision.json').write_text(json.dumps(result,indent=2)+'\n')
(root/'controller.exit').write_text('0\n');print(json.dumps(result),flush=True)
'''
compile(controller,'controller.py','exec')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(root+'/spec.json','rb') as f:assert f.read()==(archive/'spec.json').read_bytes()
assert 'controller.py' not in s.listdir(root)
raw=controller.encode();(archive/'controller.py').write_bytes(raw)
with s.open(root+'/controller.py','wx') as f:f.write(raw)
inner='exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')+' > '+shlex.quote(root+'/run.log')+' 2>&1'
_,out,err=c.exec_command('screen -dmS mcln_pvg_sr_rec_competition_probe_v1 bash -c '+shlex.quote(inner),timeout=30);assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=out.read().decode().strip();assert process and out.channel.recv_exit_status()==0
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),process=process,poll_seconds=300,controller_sha256=hashlib.sha256(raw).hexdigest(),scope='one gated real Sr batch forward/backward; not Sr training',nr_training_job_queued=False)
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wx') as f:f.write(raw)
print(json.dumps(record));s.close();c.close()