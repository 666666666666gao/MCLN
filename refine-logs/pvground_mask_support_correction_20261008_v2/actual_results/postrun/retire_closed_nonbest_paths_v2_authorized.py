"""Standing authorization: retire one closed nonbest head and archived remote NPZs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).resolve().parents[1]
receipt_path=local/'postrun/cleanup_receipt.json'
assert not receipt_path.exists()
policy_path=local.parent/'pvground_support_boundary_cases_20261007/CLEANUP_STANDING_AUTHORIZATION_20261008.json'
policy=json.loads(policy_path.read_bytes());assert policy['future_repeated_approval_required'] is False
wait=json.loads((local/'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode']==0
audit=json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['execution_scope']=='LOCAL_CPU_ARTIFACT_ONLY_ACTUAL_RESULT_AUDIT'
assert not audit['blocking_findings']
assert audit['metric_best_selection']['candidate']=='content'
inspection=json.loads((local/'postrun/checkpoint_inspection.json').read_bytes())
assert inspection['status']=='PASS_CLOSED_CPU_STRICT_RECONSTRUCTION'
assert inspection['full_cpu_state_tensors']==1314
best=next(row for row in inspection['weights'] if row['arm']=='content')
nonbest=next(row for row in inspection['weights'] if row['arm']=='box_conditioned')
archive=Path(inspection['best_local_archive'])
assert archive.stat().st_size==best['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==best['sha256']
intake=json.loads((local/'complete_fit/INTAKE.json').read_bytes())
arrays=[row for row in intake['files'] if row['name'].endswith('.npz')]
assert len(arrays)==2378 and len({row['name'] for row in arrays})==2378
for row in arrays:
    path=local/'complete_fit'/row['name']
    assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
spec=json.loads((local/'pair_spec.json').read_bytes())
previous_inventory=json.loads((local.parent/'pvground_support_boundary_cases_20261007/remaining_weight_inventory_20261008.json').read_bytes())
code=r'''import datetime,hashlib,json,os,shutil,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root'])
assert root==Path('/root/autodl-tmp/pvground_mask_support_correction_20261008_v2') and root.resolve()==root
assert (root/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((root/'fit_status.json').read_bytes())['status']=='complete'
assert not (root/'weight_and_array_cleanup_receipt.json').exists()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
protected=[]
for row in b['previous_protected']:
    path=Path(row['path']);assert path.resolve()==path and path.stat().st_size==row['bytes']
    protected.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
for row in b['dependencies']:
    assert sha(row['path'])==row['sha256']
winner=Path(b['best']['path']);assert winner.resolve()==winner and root in winner.parents
assert winner.stat().st_size==b['best']['bytes'] and sha(winner)==b['best']['sha256']
protected.append(b['best'])
negative=Path(b['nonbest']['path'])
assert negative==root/'box_conditioned/terminal.pth' and negative.resolve()==negative
assert negative.stat().st_size==b['nonbest']['bytes'] and sha(negative)==b['nonbest']['sha256']
paths=[]
for item in b['arrays']:
    rel=Path(item['name'])
    assert len(rel.parts)==2 and rel.parts[0] in ('initial_formal','formal') and rel.name.startswith('batch_') and rel.suffix=='.npz'
    path=root/rel;assert path.resolve()==path and root in path.parents
    assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    paths.append(path)
assert len(paths)==2378 and len(set(paths))==2378
assert negative not in {Path(row['path']) for row in protected}
before=shutil.disk_usage(root).free
negative.unlink()
for path in paths:path.unlink()
assert not negative.exists() and all(not path.exists() for path in paths)
for row in protected:
    path=Path(row['path']);assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
weights=[];base=Path('/root/autodl-tmp')
for task in sorted(base.iterdir()):
    if not task.is_dir() or not task.name.startswith(('pvground_','mcln_pvground_','cs_pvground_')) or task.name=='mcln_pvground_runtime_20260908_v1':continue
    for folder,dirs,names in os.walk(str(task),followlinks=False):
        dirs[:]=[name for name in dirs if name not in ('.git','venv','__pycache__','data','datasets')]
        for name in names:
            if name.endswith(('.pth','.ckpt')):
                path=Path(folder)/name;assert not path.is_symlink()
                weights.append(dict(path=str(path),bytes=path.stat().st_size))
r=dict(status='CLOSED_NONBEST_WEIGHT_AND_ARCHIVED_REMOTE_ARRAYS_REMOVED',time_cst=datetime.datetime.now().astimezone().isoformat(),
    deleted_weights=[b['nonbest']],deleted_array_count=len(paths),deleted_arrays=b['arrays'],
    released_weight_bytes=b['nonbest']['bytes'],released_array_bytes=sum(row['bytes'] for row in b['arrays']),
    released_file_bytes=b['nonbest']['bytes']+sum(row['bytes'] for row in b['arrays']),
    best_checkpoint=b['best'],best_local_archive=b['best_archive'],local_arrays=b['local_arrays'],
    preserved_weights=protected,remaining_generated_project_weight_inventory=weights,
    free_bytes_before=before,free_bytes_after=shutil.disk_usage(root).free,
    best_weights_touched=0,datasets_touched=0,text_evidence_touched=0,local_archive_deleted=0,
    repeated_user_approval_required=False,policy_sha256=b['policy_sha256'],neural_forwards=0,optimizer_updates=0)
(root/'weight_and_array_cleanup_receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
'''
payload=dict(root=spec['root'],best=best,nonbest=nonbest,arrays=arrays,dependencies=inspection['required_dependencies'],
    previous_protected=previous_inventory['weights'],best_archive=str(archive),local_arrays=str(local/'complete_fit'),
    policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest())
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
stdin,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code]),timeout=180)
stdin.write(json.dumps(payload));stdin.flush();stdin.channel.shutdown_write()
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record=json.loads(raw);client.close()
assert len(record['deleted_weights'])==1 and record['deleted_array_count']==2378
receipt_path.write_bytes(raw)
print(json.dumps({key:record[key] for key in ('status','deleted_array_count','released_weight_bytes','released_array_bytes','released_file_bytes','free_bytes_before','free_bytes_after','best_weights_touched','repeated_user_approval_required')}),flush=True)
