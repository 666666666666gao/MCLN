"""CPU header check and authorized cleanup of this closed comparison's weights."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch

root=Path(sys.argv[1]).resolve()
decision=json.loads(sys.argv[2])
assert str(root)=='/root/autodl-tmp/pvground_support_reference_20261005'
assert not (root/'weight_retention.json').exists()
status=json.loads((root/'fit_status.json').read_bytes())
assert status['status']=='complete' and status['protected_parents_exact']
assert (root/'fit_controller.exit').read_text().strip()=='0'
spec=json.loads((root/'control_spec.json').read_bytes())
sys.path.insert(0,spec['helper_root'])
sys.path.insert(0,str(root))
assert hashlib.sha256((root/'support_reference.py').read_bytes()).hexdigest()==spec['support_reference_sha256']
assert all(hashlib.sha256(Path(spec['helper_root'],name).read_bytes()).hexdigest()==digest for name,digest in spec['runner_files'].items())
pattern='^'+spec['runtime']+'/venv/bin/python -B -u '+str(root)+'/controller.py --phase fit$'
assert subprocess.run(['pgrep','-af',pattern],stdout=subprocess.PIPE).returncode==1
parent=Path(spec['geometry_terminal']).resolve()
assert str(parent)=='/root/autodl-tmp/pvground_auxiliary_target_20261005/control/terminal.pth'
assert hashlib.sha256(parent.read_bytes()).hexdigest()==spec['geometry_terminal_sha256']
parent_payload=torch.load(str(parent),map_location='cpu')
parent_state=parent_payload['state_delta']
paths=dict(protected_geometry_parent=parent,control=root/'control/terminal.pth',support_reference=root/'support_reference/terminal.pth')
identities={}
for arm in ('control','support_reference'):
    path=paths[arm]
    assert path.resolve()==root/arm/'terminal.pth'
    fit=json.loads((root/arm/'receipt.json').read_bytes())
    formal=json.loads((root/arm/'formal/receipt.json').read_bytes())
    restored=json.loads((root/arm/'formal_restore.json').read_bytes())
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert digest==fit['terminal_sha256']==restored['terminal_sha256']
    assert formal['rows']==9508 and formal['status']=='pass'
    assert [formal['metrics']['bbs'][key] for key in ('rec_hits25','rec_hits50')]==decision['hits'][arm]
    payload=torch.load(str(path),map_location='cpu')
    assert payload['step']==3723 and payload['geometry_parent_fit_updates']==11169 and payload['total_geometry_fit_updates']==14892
    assert len(payload['row_ids'])==len(set(payload['row_ids']))==29778 and payload['head_only']
    state=payload['state_delta']
    assert payload['reference_enabled']==(arm=='support_reference')
    assert payload['reference_loss_weight']==(1.0 if arm=='support_reference' else 0.0)
    from pvground_boundary_box_refiner import BoundaryBoxRefiner
    from support_reference import SupportReferenceBoxRefiner
    head=SupportReferenceBoxRefiner() if arm=='support_reference' else BoundaryBoxRefiner('distribution')
    assert len(state)==(12 if arm=='support_reference' else 10)
    assert sum(value.numel() for value in state.values())==(459180 if arm=='support_reference' else 456102)
    assert all(name.startswith('candidate_box_refiner.') for name in state)
    stripped={name[len('candidate_box_refiner.'):]:value for name,value in state.items()}
    assert set(stripped)==set(head.state_dict())
    head.load_state_dict(stripped,strict=True)
    assert all(torch.equal(value,head.state_dict()[name]) for name,value in stripped.items())
    assert all(state[name].shape==value.shape for name,value in parent_state.items())
    assert payload['boundary_mode']=='distribution' and payload['support_arm']=='whole_range' and payload['use_whole_range']
    assert payload['boundary_loss_weight']==1./7 and payload['base_terminal_sha256']==spec['base_terminal_sha256']
    assert payload['checkpoint_sha256']==spec['checkpoint_sha256'] and payload['source_port_sha256']==spec['source_port_sha256']
    assert payload['geometry_terminal_sha256']==spec['geometry_terminal_sha256'] and payload['extra_geometry_weight']==1.0
    assert payload['auxiliary_target_mode']=='native_gt'
    assert payload['support_reference_sha256']==spec['support_reference_sha256']
    assert len(payload['optimizer']['state'])==len(state) and all(int(value['step'])==3723 for value in payload['optimizer']['state'].values())
    identities[arm]=dict(path=str(path),bytes=len(raw),sha256=digest)
winner=decision['winner']
assert winner in paths and decision['hits']['protected_geometry_parent']==[5616,4511]
assert winner=='protected_geometry_parent' or decision['hits'][winner][1]>decision['hits']['protected_geometry_parent'][1]
identities['protected_geometry_parent']=dict(path=str(parent),bytes=parent.stat().st_size,sha256=spec['geometry_terminal_sha256'])
deleted=[]
for name,path in paths.items():
    if name!=winner:
        deleted.append(dict(system=name,**identities[name]))
        path.unlink()
assert paths[winner].is_file()
assert Path(spec['base_terminal']).is_file()
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status='CLOSED_NONBEST_WEIGHTS_REMOVED',
    retained_best=dict(system=winner,hits=decision['hits'][winner],**identities[winner]),deleted=deleted,
    released_bytes=sum(entry['bytes'] for entry in deleted),best_delta_has_all_geometry_states=True,strict_CPU_head_restore=True,
    obsolete_geometry_parent_needed_for_reconstruction=False,
    original_G_parent_preserved=True,V99_paths_touched=0,local_weight_archive_created=False,
    inference_or_optimizer_replayed=False,CPU_weight_headers_checked=3)
(root/'weight_retention.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
