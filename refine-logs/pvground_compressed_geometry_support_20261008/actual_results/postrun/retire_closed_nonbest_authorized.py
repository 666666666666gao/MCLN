"""Retire the closed nonbest heads and fully archived remote candidate arrays."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).resolve().parents[1]
receipt=local/'postrun/cleanup_receipt.json'
assert not receipt.exists()
wait=json.loads((local/'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode']==0
audit=json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['execution_scope']=='ACTUAL_CLOSED_TRAINED_PAIR' and not audit['blocking_findings']
inspection=json.loads((local/'postrun/checkpoint_inspection.json').read_bytes())
assert inspection['status']=='PASS_CLOSED_CPU_BOTH_TERMINALS_STRICT_RECONSTRUCTION'
retained=json.loads((local/'postrun/RETAINED_BEST.json').read_bytes())
assert retained['full_cpu_strict_restore'] and [retained['hits025'],retained['hits050']]==[5599,4859]
best=retained['best_checkpoint'];assert best['arm']=='content'
archive=Path(retained['best_local_archive'])
assert archive.stat().st_size==best['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==best['sha256']
spec=json.loads((local/'pair_spec.json').read_bytes())
old_archive=Path(retained['prior_warm_start_local_archive'])
assert hashlib.sha256(old_archive.read_bytes()).hexdigest()==spec['warm_support_terminal_sha256']
negative=next(row for row in inspection['weights'] if row['arm']=='box_conditioned')
old=dict(path=spec['warm_support_terminal'],bytes=old_archive.stat().st_size,sha256=spec['warm_support_terminal_sha256'])
intake=json.loads((local/'complete_fit/INTAKE.json').read_bytes())
arrays=[row for row in intake['files'] if row['name'].endswith('.npz')]
assert len(arrays)==len({row['name'] for row in arrays})==2378
for row in arrays:
    path=local/'complete_fit'/row['name']
    assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
policy_path=local.parent/'pvground_support_boundary_cases_20261007/CLEANUP_STANDING_AUTHORIZATION_20261008.json'
policy=json.loads(policy_path.read_bytes());assert policy['future_repeated_approval_required'] is False
previous=json.loads((local/'CLEANUP_FOLLOWUP_20261008_1645.json').read_bytes())
protected=[row for row in previous['remaining_required_weights'] if row['path']!=old['path']]+[best]
assert len(protected)==6 and len({row['path'] for row in protected})==6
assert {row['path'] for row in inspection['required_dependencies']}.issubset({row['path'] for row in protected})
code=r'''import datetime,hashlib,json,os,shutil,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root'])
assert root==Path('/root/autodl-tmp/pvground_compressed_geometry_support_20261008') and root.resolve()==root
assert (root/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((root/'fit_status.json').read_bytes())['status']=='complete'
assert not (root/'weight_and_array_cleanup_receipt.json').exists()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
protected=[]
for row in b['protected']:
    path=Path(row['path']);assert path.resolve()==path and path.stat().st_size==row['bytes']
    protected.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
assert sha(b['best']['path'])==b['best']['sha256']
for row in b['dependencies']:assert sha(row['path'])==row['sha256']
expected_paths=[root/'box_conditioned/terminal.pth',Path('/root/autodl-tmp/pvground_mask_support_correction_20261008_v2/content/terminal.pth')]
for row,path in zip(b['nonbest'],expected_paths):
    assert Path(row['path'])==path and path.resolve()==path
    assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    assert str(path) not in {item['path'] for item in protected}
paths=[]
for row in b['arrays']:
    rel=Path(row['name'])
    assert len(rel.parts)==2 and rel.parts[0] in ('initial_formal','formal')
    assert rel.name.startswith('batch_') and rel.suffix=='.npz'
    path=root/rel;assert path.resolve()==path and root in path.parents
    assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    paths.append(path)
assert len(paths)==len(set(paths))==2378
before=shutil.disk_usage(root).free
for path in expected_paths:path.unlink()
for path in paths:path.unlink()
assert all(not path.exists() for path in expected_paths+paths)
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
assert {row['path'] for row in weights}=={row['path'] for row in protected}
record=dict(status='CLOSED_NONBEST_HEADS_AND_ARCHIVED_REMOTE_ARRAYS_REMOVED',
    time_cst=datetime.datetime.now().astimezone().isoformat(),deleted_weights=b['nonbest'],
    deleted_array_count=len(paths),deleted_arrays=b['arrays'],
    released_weight_bytes=sum(row['bytes'] for row in b['nonbest']),
    released_array_bytes=sum(row['bytes'] for row in b['arrays']),
    released_file_bytes=sum(row['bytes'] for row in b['nonbest'])+sum(row['bytes'] for row in b['arrays']),
    preserved_weights=protected,remaining_generated_project_weight_inventory=weights,
    best_checkpoint=b['best'],best_local_archive=b['best_archive'],local_arrays=b['local_arrays'],
    prior_warm_start_local_archive=b['prior_warm_start_local_archive'],
    prior_warm_weight_needed_for_final_cpu_reconstruction=False,
    negative_terminal_archived=False,free_bytes_before=before,free_bytes_after=shutil.disk_usage(root).free,
    best_weights_touched=0,dependencies_touched=0,datasets_touched=0,text_evidence_touched=0,
    local_archives_deleted=0,repeated_user_approval_required=False,policy_sha256=b['policy_sha256'],
    neural_forwards=0,optimizer_updates=0)
(root/'weight_and_array_cleanup_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''
body=dict(root=spec['root'],best=best,nonbest=[negative,old],arrays=arrays,
    protected=protected,dependencies=inspection['required_dependencies'],best_archive=str(archive),
    local_arrays=str(local/'complete_fit'),prior_warm_start_local_archive=str(old_archive),
    policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest())
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
stdin,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code]),timeout=180)
stdin.write(json.dumps(body));stdin.flush();stdin.channel.shutdown_write()
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode();client.close()
record=json.loads(raw);assert len(record['deleted_weights'])==2 and record['deleted_array_count']==2378
receipt.write_bytes(raw)
print(json.dumps({key:record[key] for key in ('status','released_weight_bytes','released_array_bytes','released_file_bytes','free_bytes_before','free_bytes_after','best_weights_touched','repeated_user_approval_required')}),flush=True)
