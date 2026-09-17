"""Install fixed Nr/Sr full-fit drivers and verify CPU contracts, without launching training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path(__file__).resolve().parents[1]
scan_archive = repo/'refine-logs/pvground_scanrefer_finetune_20260917_rec_competition_v1'
scan = json.loads((scan_archive/'spec.json').read_bytes())
partition_archive = repo/'refine-logs/pvground_referit_fit_partitions_20260917_v1'
part_spec = json.loads((partition_archive/'spec.json').read_bytes())
part_receipt = json.loads((partition_archive/'receipt.json').read_bytes())
partition_root = '/root/autodl-tmp/mcln_pvground_referit_fit_partitions_20260917_v1'
formal = '/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260917_rec_competition_v1'
bundle = json.loads((repo/'refine-logs/pvground_runtime_20260908_v1/source_bundle_receipt.json').read_bytes())
evaluator_sha = bundle['sources']['PV-Ground']['files']['src/grounding_evaluator.py']['sha256']
root = '/root/autodl-tmp/mcln_pvground_referit_rec_training_preparation_20260917_v1'
archive = repo/'refine-logs'/Path(root).name.replace('mcln_', '', 1)
files = {
    'train.py': (repo/'scripts/run_pvground_referit_rec_competition.py').read_bytes(),
    'pvground_referit_fit_dataset.py': (repo/'scripts/pvground_referit_fit_dataset.py').read_bytes(),
    'tests/test_pvground_referit_fit_dataset.py': (repo/'tests/test_pvground_referit_fit_dataset.py').read_bytes(),
}
for name in ['pvground_rec_competition.py', 'pvground_referit_rec_competition.py',
             'pvground_source_query.py', 'pvground_observation_query.py', 'pvground_task_observation_query.py']:
    files[name] = (repo/'models'/name).read_bytes()
plan = '''# ReferIt F continuation, fixed before Scan F endpoint

Only after the current Scan F formal9508 integrity and REC-only pass, and the
dataset's real F backward probe restores all state, may full training start.
This preparation does not queue full training. Preserve current Scan execution.

Use the same task-observation architecture and native-bbs competition as Scan F.
Initialize each dataset from its own published full pretrained parent (Nr epoch25,
Sr epoch31), not a failed endpoint or Scan checkpoint. Native butd_cls supplies
instance boxes plus predicted classes. All native losses remain; Mask is diagnostic.
Only actual referring rows receive the auxiliary competition; detection rows
contribute zero to it, with the full batch denominator retained.

Single seed2027, batch8, one fixed fit pass, LR/backbone LR1e-5, native AdamW
weight decay0.0005, clip0.1, no intermediate selection or added epochs.
Nr:36747 fit /6172 holdout,4594 updates. Sr:65558 /10328,8195 updates.
Physical-room detection exclusions and annotation order are bound by saved hashes.
Module holdout was seen by upstream pretraining; it is not formal scene generalization.

Record full initial and terminal holdout bbs/bbf boxes/scores/rows. bbs is primary;
only both bbs thresholds nonregressing versus its own initial allow formal evaluation.
No Mask gate. Formal Nr7899 and Sr17726 use the same architecture and bbs rule.
Target Nr at least4726/4059 hits (exceeds59.82/51.38), Sr at least12139/10335
(preserves protected68.4813/58.3042, exceeding68.43/57.30).
These are intended acceptance rules; the formal execution/audit must still be prepared.

Use serial GPU lock, disk capacity check, initial/terminal independent audit and
exact-row coverage before deleting superseded latest. Do not launch until complete
controller/evaluation continuation and disk provision are ready.
'''
files['plan.md'] = plan.encode()
preparation = r'''import ast,datetime,hashlib,json,os,subprocess,sys
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
'''
files['prepare.py'] = preparation.encode()

c = paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp();s.mkdir(root);s.mkdir(root+'/tests');archive.mkdir()
for name, raw in files.items():
    if name.endswith('.py'):compile(raw, name, 'exec')
    local = archive/name;local.parent.mkdir(parents=True, exist_ok=True);local.write_bytes(raw)
    with s.open(root+'/'+name, 'wx') as f:f.write(raw)
training_specs = {};training_shas = {}
parents = {
    'nr3d': ('nr', 'd2d9afaf9c293c54977f3555a46c7bb2a72d9f80163a3f602dd8426032d7fa5d', 'NR3D'),
    'sr3d': ('sr', 'a4a14b0090947177a648703ad6de094246891d89174fffa56fe454f730dbe3dc', 'SR3D'),
}
for dataset, (short, digest, upper) in parents.items():
    target = '/root/autodl-tmp/mcln_pvground_'+short+'_finetune_20260917_rec_competition_v1'
    s.mkdir(target)
    local = archive/dataset;local.mkdir()
    bound = {name: raw for name, raw in files.items() if '/' not in name and name != 'prepare.py'}
    for name, raw in bound.items():
        with s.open(target+'/'+name, 'wx') as f:f.write(raw)
    summary = part_receipt['datasets'][dataset]
    spec = {key: scan[key] for key in ['runtime','training_interface_receipt','env_spec_sha256','batch_size','fit_passes','seed',
        'lr','lr_backbone','primary_mode','model_source','source_port','source_port_sha256','source_query_read','source_query_module_sha256',
        'observation_state','observation_module_sha256','task_read','task_module_sha256','rec_competition','competition_weight','competition_module_sha256']}
    spec.update(root=target, dataset=dataset, scan_formal_root=formal,
        actual_probe_root='/root/autodl-tmp/mcln_pvground_'+short+'_rec_competition_preparation_20260917_v1/actual_probe',
        checkpoint=dict(path='/root/autodl-tmp/mcln_pvground_'+short+'_checkpoint_inspection_20260908_v1/PV-Ground_'+upper+'.pth',sha256=digest),
        checkpoint_sha256=digest,input_manifest=part_spec['selection_root']+'/manifest.json',input_manifest_sha256=part_spec['selection_manifest_sha256'],
        full_point_manifest=part_spec['scan_input_manifest'],full_point_manifest_sha256=part_spec['scan_input_manifest_sha256'],
        partition=partition_root+'/'+dataset+'_partition.json',partition_sha256=summary['partition_sha256'],
        fit_rows=summary['total_fit_rows'],holdout_rows=summary['holdout_rows'],total_steps=summary['batch8_updates_one_pass'],
        evaluator_sha256=evaluator_sha,formal_rows=0,files={name:hashlib.sha256(raw).hexdigest() for name,raw in bound.items()})
    raw = (json.dumps(spec,indent=2)+'\n').encode();(local/'spec.json').write_bytes(raw)
    with s.open(target+'/spec.json', 'wx') as f:f.write(raw)
    training_specs[dataset]=target+'/spec.json';training_shas[dataset]=hashlib.sha256(raw).hexdigest()
formal_launch=json.loads((repo/'refine-logs/pvground_scanrefer_formal_20260917_rec_competition_v1/launch.json').read_bytes())
spec=dict(formal_root=formal,formal_pid=int(formal_launch['process'].split()[0]),scan_driver=scan['root']+'/train.py',
          training_specs=training_specs,training_spec_sha256=training_shas,files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
raw=(json.dumps(spec,indent=2)+'\n').encode();(archive/'preparation_spec.json').write_bytes(raw)
with s.open(root+'/preparation_spec.json','wx') as f:f.write(raw)
runtime=scan['runtime']+'/venv/bin/python'
_,out,err=c.exec_command("CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "+shlex.quote(runtime)+' -u '+shlex.quote(root+'/prepare.py')+' > '+shlex.quote(root+'/prepare.log')+' 2>&1',timeout=60)
code=out.channel.recv_exit_status()
with s.open(root+'/prepare.exit','wx') as f:f.write(str(code)+'\n')
names=s.listdir(root)
for name in ['prepare.exit','prepare.log','receipt.json','dataset_test.log','nr3d_gate_rejection.log','sr3d_gate_rejection.log']:
    if name in names:s.get(root+'/'+name,str(archive/name))
assert code==0,(archive/'prepare.log').read_text()
print((archive/'receipt.json').read_text(),flush=True)
s.close();c.close()
