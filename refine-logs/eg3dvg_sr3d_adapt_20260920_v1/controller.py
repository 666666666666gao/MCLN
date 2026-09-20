import datetime,fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent
spec=json.loads((r/'spec.json').read_text())
nr=Path(spec['nr_adaptation_root'])
sr=Path(spec['transfer_root'])

def wait_complete(root):
    while not (root/'controller.exit').exists():time.sleep(300)
    assert (root/'controller.exit').read_text().strip()=='0',str(root)

def decision(action,metrics):
    record={'action':action,'metrics':metrics,'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'spec_sha256':hashlib.sha256((r/'spec.json').read_bytes()).hexdigest(),'proceed_to_gpu_preflight':action=='run_fixed_adaptation'}
    with (r/'decision.json').open('x') as f:json.dump(record,f,indent=2)
    print('SR_ADAPTATION_DECISION '+json.dumps(record),flush=True)

wait_complete(nr)
nr_audit=json.loads((nr/'evaluation/formal/audit.json').read_text())
assert nr_audit['integrity_pass']
nr_hits=nr_audit['metrics']['bbs']
if nr_hits['rec_hits25']<spec['required_nr_hits25'] or nr_hits['rec_hits50']<spec['required_nr_hits50']:
    decision('deferred_for_nr_priority',{'nr':nr_hits})
    (r/'controller.exit').write_text('0\n')
    raise SystemExit(0)
wait_complete(sr)
sr_audit=json.loads((sr/'formal/audit.json').read_text())
assert sr_audit['integrity_pass']
sr_hits=sr_audit['metrics']['bbs']
if sr_hits['rec_hits25']>=spec['protected_sr_hits25'] and sr_hits['rec_hits50']>=spec['protected_sr_hits50']:
    decision('skipped_zero_update_already_meets_protected',{'nr':nr_hits,'sr':sr_hits})
    (r/'controller.exit').write_text('0\n')
    raise SystemExit(0)
decision('run_fixed_adaptation',{'nr':nr_hits,'sr':sr_hits})
lock=open('/root/autodl-tmp/mcln_v99_backbone_gpu0.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()

def run_stage(name,command,cwd):
    with (r/(name+'.log')).open('xb') as f:code=subprocess.call(command,cwd=str(cwd),stdout=f,stderr=subprocess.STDOUT)
    (r/(name+'.exit')).write_text(str(code)+'\n')
    if code:
        (r/'controller.exit').write_text(str(code)+'\n')
        raise SystemExit(code)

for stage in ['preflight','fit']:
    run_stage(stage,[sys.executable,'-u',str(r/'train.py'),'--spec',str(r/'spec.json'),'--stage',stage],spec['source'])
fit=json.loads((r/'fit/receipt.json').read_text())
assert fit['optimizer_steps']==9730 and fit['rows']==77836
post=r/'evaluation';post.mkdir()
eval_spec=json.loads((sr/'spec.json').read_text())
eval_spec.update({'checkpoint':fit['checkpoint_path'],'checkpoint_sha256':fit['checkpoint_sha256'],'checkpoint_optimizer_steps':9730,
                  'experiment_type':'single_epoch_adaptation','training_spec_sha256':fit['spec_sha256'],
                  'evaluator_sha256':hashlib.sha256((r/'evaluate_adapted.py').read_bytes()).hexdigest(),
                  'auditor_sha256':hashlib.sha256((r/'audit_adapted.py').read_bytes()).hexdigest()})
(post/'spec.json').write_text(json.dumps(eval_spec,indent=2))
(post/'annotation_manifest.json').write_bytes((sr/'annotation_manifest.json').read_bytes())
for stage in ['preflight','formal']:
    run_stage('evaluation_'+stage,[sys.executable,'-u',str(r/'evaluate_adapted.py'),'--spec',str(post/'spec.json'),'--stage',stage],eval_spec['source'])
run_stage('evaluation_audit',[sys.executable,'-u',str(r/'audit_adapted.py'),'--root',str(post)],eval_spec['source'])
run_stage('paired_rec',[sys.executable,'-u',str(r/'analyze_pair.py'),'--start',str(sr),'--adaptation',str(r)],spec['source'])
(r/'controller.exit').write_text('0\n')
