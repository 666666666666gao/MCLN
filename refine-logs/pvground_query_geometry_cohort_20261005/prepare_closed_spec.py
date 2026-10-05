import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
study=root.parent/'pvground_query_supported_geometry_20261005'
wait=json.loads((study/'fit_wait.json').read_bytes())
assert wait['observer_closed'] and wait['terminal']['exitcode']==0
assert not wait['terminal']['controller_alive']
assert wait['terminal']['status']['status']=='complete'
assert set(wait['terminal']['status']['completed_runs'])=={
    'control/train','control/formal','query_supported/train','query_supported/formal'}
assert not (root/'spec.json').exists()
spec=json.loads((root.parent/'pvground_mask_branch_responsibility_20261005'/'spec.json').read_bytes())
spec.update(root='/root/autodl-tmp/pvground_query_geometry_cohort_20261005',
    purpose='Fixed parent-supported cohort geometry before/after the closed two-arm fit; not formal accuracy.',
    terminal_heads={})
for arm in ('control','query_supported'):
    receipt=json.loads((study/'complete'/arm/'receipt.json').read_bytes())
    formal=json.loads((study/'complete'/arm/'formal'/'receipt.json').read_bytes())
    restore=json.loads((study/'complete'/arm/'formal_restore.json').read_bytes())
    assert receipt['status']=='complete' and receipt['training_steps']==3723
    assert receipt['fit_rows']==29778 and receipt['fit_seen_exactly_once']
    assert receipt['parent_and_zero_R_states_exact']
    assert formal['status']=='pass' and formal['rows']==9508
    assert restore['terminal_sha256']==receipt['terminal_sha256']
    assert (study/'complete'/arm/'train.exit').read_text().strip()=='0'
    assert (study/'complete'/arm/'formal.exit').read_text().strip()=='0'
    spec['terminal_heads'][arm]=dict(
        path='/root/autodl-tmp/pvground_query_supported_geometry_20261005/'+arm+'/terminal.pth',
        sha256=receipt['terminal_sha256'],spec_sha256=receipt['spec_sha256'])
spec['source_receipts']=dict(
    fit_wait_sha256=hashlib.sha256((study/'fit_wait.json').read_bytes()).hexdigest(),
    intake_sha256=hashlib.sha256((study/'complete'/'INTAKE.json').read_bytes()).hexdigest())
(root/'spec.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='PREPARED_AFTER_ACTUAL_CLOSED_PAIR',terminal_heads=spec['terminal_heads'])))
