import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260908_v1')
spec=json.loads((root/'spec.json').read_bytes())
for name,digest in spec['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert (root/'cpu_probe.exit').read_text().strip()=='0'
assert json.loads((root/'preparation_receipt.json').read_bytes())['status']=='pass'
deadline=datetime.datetime.fromisoformat(spec['first_check_cst']).timestamp()
time.sleep(max(0,deadline-time.time()))
training=Path(spec['training_root']);audit_root=Path(spec['training_audit_root'])
while not (audit_root/'controller.exit').is_file():
    process=Path('/proc')/str(spec['training_audit_controller_pid'])/'cmdline'
    assert process.is_file() and (str(audit_root)+'/controller.py').encode() in process.read_bytes(),'endpoint audit controller disappeared without exit receipt'
    time.sleep(spec['poll_seconds'])
assert (training/'controller.exit').is_file()
codes={'training':int((training/'controller.exit').read_text()),'audit':int((audit_root/'controller.exit').read_text())}
if any(codes.values()):
    result={'status':'dependency_failed','exit_codes':codes,'formal_rows':0}
    (root/'decision.json').write_text(json.dumps(result)+'\n')
    print('PVG_FORMAL_DEPENDENCY_FAILED '+json.dumps(result),flush=True)
    raise SystemExit(1)
audited=json.loads((audit_root/'audit.json').read_bytes())
assert audited['integrity_pass']
assert audited['receipt_sha256']==hashlib.sha256((training/'receipt.json').read_bytes()).hexdigest()
if not audited['primary_rec_nonregression']:
    result={'status':'skipped_primary_rec_regression','formal_rows':0,'primary_mode':'bbs','transitions':audited['transitions']['bbs']}
    (root/'decision.json').write_text(json.dumps(result)+'\n')
    print('PVG_FORMAL_SKIPPED '+json.dumps(result),flush=True)
    raise SystemExit(0)
runtime=Path(spec['runtime'])
environment=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
command=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',str(runtime/'venv/bin/python'),'-u',str(root/'evaluate.py'),'--spec',str(root/'spec.json')]
decision={'status':'launching_fixed_formal','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'training_receipt_sha256':hashlib.sha256((training/'receipt.json').read_bytes()).hexdigest(),
          'training_audit_sha256':hashlib.sha256((audit_root/'audit.json').read_bytes()).hexdigest(),'formal_rows_expected':9508}
(root/'decision.json').write_text(json.dumps(decision)+'\n')
print('PVG_FORMAL_LAUNCH '+json.dumps(decision),flush=True)
result=subprocess.run(command,cwd=str(runtime/'PV-Ground'),env=dict(os.environ,**environment['env']))
(root/'evaluation.exit').write_text(str(result.returncode)+'\n')
if result.returncode!=0:raise SystemExit(result.returncode)
result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'audit.py'),'--root',str(root),'--out',str(root/'audit.json')])
(root/'audit.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
