"""Remove fixed closed nonbest weights after audited selection and CPU witness."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


root=Path('/root/autodl-tmp/pvground_reference_keep_20261006')
decision=json.loads(sys.argv[1])
assert not (root/'weight_retention.json').exists()
assert (root/'fit_controller.exit').read_text().strip()=='0'
status=json.loads((root/'fit_status.json').read_bytes())
assert status['status']=='complete' and status['protected_parents_exact']
spec=json.loads((root/'control_spec.json').read_bytes())
pattern='^'+spec['runtime']+'/venv/bin/python -B -u '+str(root)+'/controller.py --phase fit$'
assert subprocess.run(['pgrep','-af',pattern],stdout=subprocess.PIPE).returncode==1
inspection_path=root/'closed_weight_inspection.json'
assert sha(inspection_path)==decision['inspection_sha256']
inspection=json.loads(inspection_path.read_bytes())
assert inspection['status']=='CLOSED_FIXED_CHECKPOINTS_CPU_INSPECTED'
assert inspection['decision']['summary_sha256']==decision['summary_sha256']
assert inspection['winner']==decision['winner']
assert inspection['weights_deleted']==inspection['weights_created']==inspection['optimizer_updates']==0
identities=inspection['identities']
expected={'protected_geometry_parent':Path(spec['selected_terminal'])}
assert str(expected['protected_geometry_parent'])=='/root/autodl-tmp/pvground_mask_reference_20261006/fused_mask_reference/initial.pth'
for arm in ('control','keep'):
    for stage,name in (('initial_formal','initial.pth'),('formal','terminal.pth')):
        expected[arm+'/'+stage]=root/arm/name
assert set(expected)==set(identities)==set(inspection['decision']['hits'])
for key,path in expected.items():
    assert path.resolve()==path and str(path)==identities[key]['path']
    assert path.stat().st_size==identities[key]['bytes'] and sha(path)==identities[key]['sha256']
    if key!='protected_geometry_parent':
        assert root in path.parents
winner=max(expected,key=lambda key:(inspection['decision']['hits'][key][0]>=5620 and inspection['decision']['hits'][key][1]>=4764,
    inspection['decision']['hits'][key][1],inspection['decision']['hits'][key][0],key=='protected_geometry_parent'))
assert winner==decision['winner']
assert sha(spec['base_terminal'])==spec['base_terminal_sha256']
deleted=[]
for key,path in expected.items():
    if key!=winner:
        path.unlink()
        deleted.append(dict(candidate=key,**identities[key]))
assert sha(expected[winner])==identities[winner]['sha256']
assert sha(spec['base_terminal'])==spec['base_terminal_sha256']
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status='CLOSED_NONBEST_WEIGHTS_REMOVED',
    retained_best=dict(candidate=winner,hits=inspection['decision']['hits'][winner],**identities[winner]),
    deleted=deleted,released_bytes=sum(item['bytes'] for item in deleted),weights_before=5,weights_after=1,
    original_G_preserved=True,V99_paths_touched=0,official_PV_paths_touched=0,raw_evidence_deleted=False,
    negative_weight_archive_created=False,decision_summary_sha256=decision['summary_sha256'],
    fresh_audit_sha256=decision['audit_sha256'],inspection_sha256=decision['inspection_sha256'])
(root/'weight_retention.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
