import datetime,hashlib,json,os,subprocess,time
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
