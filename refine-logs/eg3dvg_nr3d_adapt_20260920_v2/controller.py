import fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;spec=json.loads((r/'spec.json').read_text());old=Path(spec['transfer_root'])
while not (old/'controller.exit').exists():time.sleep(180)
assert (old/'controller.exit').read_text().strip()=='0','Transfer evaluation did not complete cleanly'
assert json.loads((old/'formal/audit.json').read_text())['integrity_pass']
lock=open('/root/autodl-tmp/mcln_v99_backbone_gpu0.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
def run_stage(name,command,cwd):
 with (r/(name+'.log')).open('xb') as f:code=subprocess.call(command,cwd=str(cwd),stdout=f,stderr=subprocess.STDOUT)
 (r/(name+'.exit')).write_text(str(code)+'\n')
 if code:
  (r/'controller.exit').write_text(str(code)+'\n');raise SystemExit(code)
for stage in ['preflight','fit']:
 run_stage(stage,[sys.executable,'-u',str(r/'train.py'),'--spec',str(r/'spec.json'),'--stage',stage],spec['source'])
fit=json.loads((r/'fit/receipt.json').read_text());assert fit['optimizer_steps']==5614 and fit['rows']==44909
post=r/'evaluation';post.mkdir()
eval_spec=json.loads((old/'spec.json').read_text());eval_spec.update({'checkpoint':fit['checkpoint_path'],'checkpoint_sha256':fit['checkpoint_sha256'],'checkpoint_optimizer_steps':5614,'experiment_type':'single_epoch_adaptation','training_spec_sha256':fit['spec_sha256'],'evaluator_sha256':hashlib.sha256((r/'evaluate_adapted.py').read_bytes()).hexdigest(),'auditor_sha256':hashlib.sha256((r/'audit_adapted.py').read_bytes()).hexdigest()})
(post/'spec.json').write_text(json.dumps(eval_spec,indent=2));shutil.copyfile(str(old/'annotation_manifest.json'),str(post/'annotation_manifest.json'))
for stage in ['preflight','formal']:
 run_stage('evaluation_'+stage,[sys.executable,'-u',str(r/'evaluate_adapted.py'),'--spec',str(post/'spec.json'),'--stage',stage],eval_spec['source'])
run_stage('evaluation_audit',[sys.executable,'-u',str(r/'audit_adapted.py'),'--root',str(post)],eval_spec['source'])
(r/'controller.exit').write_text('0\n')
