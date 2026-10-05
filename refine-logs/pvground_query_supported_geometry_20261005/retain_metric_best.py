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
assert str(root)=='/root/autodl-tmp/pvground_query_supported_geometry_20261005'
assert not (root/'weight_retention.json').exists()
status=json.loads((root/'fit_status.json').read_bytes())
assert status['status']=='complete' and status['protected_parents_exact']
assert (root/'fit_controller.exit').read_text().strip()=='0'
spec=json.loads((root/'control_spec.json').read_bytes())
pattern='^'+spec['runtime']+'/venv/bin/python -B -u '+str(root)+'/controller.py --phase fit$'
assert subprocess.run(['pgrep','-af',pattern],stdout=subprocess.PIPE).returncode==1
parent=Path(spec['geometry_terminal']).resolve()
assert str(parent)=='/root/autodl-tmp/pvground_boundary_fit_20261004/distribution/terminal.pth'
assert hashlib.sha256(parent.read_bytes()).hexdigest()==spec['geometry_terminal_sha256']
parent_payload=torch.load(str(parent),map_location='cpu')
parent_state=parent_payload['state_delta']
paths=dict(protected_geometry_parent=parent,control=root/'control/terminal.pth',query_supported=root/'query_supported/terminal.pth')
identities={}
for arm in ('control','query_supported'):
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
    assert payload['step']==3723 and payload['geometry_parent_fit_updates']==3723 and payload['total_geometry_fit_updates']==7446
    assert len(payload['row_ids'])==len(set(payload['row_ids']))==29778 and payload['head_only']
    state=payload['state_delta']
    assert len(state)==10 and set(state)==set(parent_state) and sum(value.numel() for value in state.values())==456102
    assert all(name.startswith('candidate_box_refiner.') and value.shape==parent_state[name].shape for name,value in state.items())
    assert payload['boundary_mode']=='distribution' and payload['support_arm']=='whole_range' and payload['use_whole_range']
    assert payload['boundary_loss_weight']==1./7 and payload['base_terminal_sha256']==spec['base_terminal_sha256']
    assert payload['checkpoint_sha256']==spec['checkpoint_sha256'] and payload['source_port_sha256']==spec['source_port_sha256']
    assert len(payload['optimizer']['state'])==10 and all(int(value['step'])==3723 for value in payload['optimizer']['state'].values())
    identities[arm]=dict(path=str(path),bytes=len(raw),sha256=digest)
winner=decision['winner']
assert winner in paths and decision['hits']['protected_geometry_parent']==[5616,4506]
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
    released_bytes=sum(entry['bytes'] for entry in deleted),best_delta_has_all10_geometry_states=True,
    obsolete_geometry_parent_needed_for_reconstruction=False,
    original_G_parent_preserved=True,V99_paths_touched=0,local_weight_archive_created=False,
    inference_or_optimizer_replayed=False,CPU_weight_headers_checked=3)
(root/'weight_retention.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
