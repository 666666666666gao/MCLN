import ast,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).parent
spec=json.loads((root/'preparation_spec.json').read_bytes())
def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):d.update(block)
 return d.hexdigest()
for name,digest in spec['files'].items():assert sha(root/name)==digest,name
for name in spec['files']:
 if name.endswith('.py'):compile((root/name).read_bytes(),str(root/name),'exec')
formal=Path(spec['formal_root'])
assert not (formal/'controller.exit').exists()
assert (str(formal)+'/controller.py').encode() in Path('/proc/'+str(spec['formal_pid'])+'/cmdline').read_bytes()
original=ast.parse(Path(spec['scan_driver']).read_text())
new=ast.parse((root/'train.py').read_text())
unchanged=[]
for name in ['prepare','native_loss','evaluate','loader']:
 get=lambda tree:next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(get(original),include_attributes=False)==ast.dump(get(new),include_attributes=False),name
 unchanged.append(name)
test=subprocess.run([sys.executable,str(root/'tests/test_pvground_referit_fit_dataset.py')],
 stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root),OMP_NUM_THREADS='1'))
(root/'dataset_test.log').write_bytes(test.stdout+test.stderr);assert test.returncode==0,test.stderr.decode()
datasets={}
for dataset,path in spec['training_specs'].items():
 target=Path(path);training=json.loads(target.read_bytes())
 assert sha(target)==spec['training_spec_sha256'][dataset]
 for name,digest in training['files'].items():assert sha(target.parent/name)==digest,name
 assert sha(training['checkpoint']['path'])==training['checkpoint_sha256']
 assert sha(training['partition'])==training['partition_sha256']
 assert sha(training['input_manifest'])==training['input_manifest_sha256']
 assert sha(training['full_point_manifest'])==training['full_point_manifest_sha256']
 assert sha(Path(training['runtime'])/'PV-Ground/src/grounding_evaluator.py')==training['evaluator_sha256']
 run=subprocess.run([sys.executable,str(target.parent/'train.py'),'--spec',str(target),'--preflight-only'],
  stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 assert run.returncode!=0 and b'controller.exit' in run.stderr and b'FileNotFoundError' in run.stderr
 (root/(dataset+'_gate_rejection.log')).write_bytes(run.stdout+run.stderr)
 assert not (target.parent/'imports.json').exists() and not (target.parent/'capacity.json').exists()
 datasets[dataset]=dict(fit_rows=training['fit_rows'],holdout_rows=training['holdout_rows'],steps=training['total_steps'],
  spec_sha256=sha(target),driver_sha256=sha(target.parent/'train.py'),gate_rejected_before_model=True,
  checkpoint_sha256=training['checkpoint_sha256'])
stat=os.statvfs('/root/autodl-tmp')
record=dict(status='cpu_preparation_pass',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 datasets=datasets,unchanged_scan_functions=unchanged,dataset_behavior_tests=3,model_forwards=0,optimizer_steps=0,
 formal_rows=0,training_jobs_launched=0,free_bytes=stat.f_bavail*stat.f_frsize,
 scope='full-fit driver installation and CPU contract checks; actual full loader/backward/training/formal still pending')
(root/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
