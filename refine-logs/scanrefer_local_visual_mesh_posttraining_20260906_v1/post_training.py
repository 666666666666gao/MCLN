import datetime,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path
root=Path(__file__).parent
expected=json.loads((root/'preparation.json').read_text())
def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):d.update(block)
 return d.hexdigest()
for name,digest in expected['files'].items():assert sha(root/name)==digest,name
training=Path(expected['training_directory'])
assert (training/'controller.exit').read_text().strip()=='0'
assert sha(training/'input_manifest.json')==expected['training_manifest_sha256']
train=json.loads((training/'input_manifest.json').read_text())
assert train['data_root']=='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/'
assert not (training/'independent_audit.json').exists()
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
command=[sys.executable,'-m','scripts.audit_scanrefer_local_visual_pair',str(training),str(training/'independent_audit.json')]
with (training/'independent_audit.log').open('xb') as log:
 subprocess.check_call(command,cwd=str(root),env=environment,stdout=log,stderr=subprocess.STDOUT)
audit=json.loads((training/'independent_audit.json').read_text())
receipt=json.loads((training/'receipt.json').read_text())
assert audit['integrity_pass'] and receipt['status']=='complete'
assert audit['receipt_sha256']==sha(training/'receipt.json')
assert receipt['steps_per_arm']==2482 and receipt['holdout_rows']==6887
for arm in ['control','local']:
 assert audit['checkpoints'][arm]['optimizer_steps']==2482
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
formal=Path(expected['formal_directory'])
formal.mkdir()
(formal/'scripts').mkdir()
for name in expected['formal_files']:
 (formal/name).write_bytes((root/name).read_bytes())
(formal/'data_inputs.json').write_bytes((training/'data_inputs.json').read_bytes())
manifest={'schema':'mcln-scanrefer-local-visual-official-input-v2',
 'training_directory':str(training),'training_receipt_sha256':sha(training/'receipt.json'),
 'training_audit_sha256':sha(training/'independent_audit.json'),'trained_checkpoint':receipt['checkpoints']['local'],
 'data_root':train['data_root'],'val_superpoint_files':train['superpoint_files']['val'],
 'files':{name:sha(formal/name) for name in expected['formal_files'] if name!='scripts/__init__.py'},
 'arms':['protected_v99','local_v99'],'formal_rows':9508,'optimizer_steps':0,
 'scan_rec_historical_floor_hits':[5572,4797],'scan_mask_paper_floor_percent':[58.70,50.70,44.72],
 'nr3d_sr3d_mask_gate':False,
 'decision':'One fixed endpoint after the registered mesh-data repeat;development scores do not select epochs or thresholds.'}
(formal/'input_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
controller='''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd FORMAL_ROOT
flock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock PYTHON_EXEC -u scripts/evaluate_scanrefer_local_visual_official.py --manifest FORMAL_MANIFEST
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''.replace('FORMAL_ROOT',shlex.quote(str(formal))).replace('PYTHON_EXEC',shlex.quote(sys.executable)).replace('FORMAL_MANIFEST',shlex.quote(str(formal/'input_manifest.json')))
(formal/'controller.sh').write_text(controller)
subprocess.check_call(['bash','-n',str(formal/'controller.sh')])
screen='mcln_scanrefer_mesh_official_v1'
launch_command=['screen','-dmS',screen,'bash','-lc','cd '+shlex.quote(str(formal))+' && bash controller.sh > run.log 2>&1']
subprocess.check_call(launch_command)
sessions=subprocess.check_output(['screen','-ls']).decode()
matches=[line.strip() for line in sessions.splitlines() if screen in line]
assert len(matches)==1
launch={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'screen_session':matches,'training_audit_sha256':sha(training/'independent_audit.json'),
 'manifest_sha256':sha(formal/'input_manifest.json'),'controller_sha256':sha(formal/'controller.sh'),
 'command':launch_command,'formal_rows_planned':9508,'optimizer_steps':0,'checkpoint_writes':0,
 'status':'formal evaluation launched;not a completed metric or promotion'}
(formal/'launch.json').write_text(json.dumps(launch,indent=2,sort_keys=True)+'\n')
(root/'executed.json').write_text(json.dumps(launch,indent=2,sort_keys=True)+'\n')
print(json.dumps(launch))
