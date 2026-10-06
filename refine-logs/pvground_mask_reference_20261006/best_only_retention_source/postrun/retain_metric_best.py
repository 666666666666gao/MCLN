"""Remove only fixed closed comparison checkpoints after accepted metric selection."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve()
decision = json.loads(sys.argv[2])
assert str(root)=='/root/autodl-tmp/pvground_mask_reference_20261006'
assert not (root/'weight_retention.json').exists()
assert (root/'fit_controller.exit').read_text().strip()=='0'
status = json.loads((root/'fit_status.json').read_bytes())
assert status['status']=='complete' and status['protected_parents_exact']
base = json.loads((root/'native_reference_spec.json').read_bytes())
pattern = '^'+base['runtime']+'/venv/bin/python -B -u '+str(root)+'/controller.py --phase fit$'
assert subprocess.run(['pgrep','-af',pattern],stdout=subprocess.PIPE).returncode==1
parent = Path(base['geometry_terminal']).resolve()
assert str(parent)=='/root/autodl-tmp/pvground_auxiliary_target_20261005/control/terminal.pth'
assert sha(parent)==base['geometry_terminal_sha256']
g_parent = Path(base['base_terminal'])
assert sha(g_parent)==base['base_terminal_sha256']
paths = {'protected_geometry_parent':parent}
identities = {'protected_geometry_parent':dict(path=str(parent),bytes=parent.stat().st_size,
    sha256=base['geometry_terminal_sha256'])}
prior = torch.load(str(parent),map_location='cpu')
assert prior['step']==3723 and len(prior['state_delta'])==10
assert sha(root/'mask_reference.py')==base['mask_reference_sha256']
for name,digest in base['runner_files'].items():
    assert sha(Path(base['helper_root'])/name)==digest
sys.path.insert(0,base['helper_root'])
sys.path.insert(0,str(root))
from mask_reference import MaskReferenceBoxRefiner

for arm in ('native_reference','fused_mask_reference'):
    spec_path = root/(arm+'_spec.json')
    spec = json.loads(spec_path.read_bytes())
    assert spec['geometry_terminal_sha256']==base['geometry_terminal_sha256']
    for stage,name in (('initial_formal','initial.pth'),('formal','terminal.pth')):
        identifier = arm+'/'+stage
        path = root/arm/name
        assert path.resolve()==path and root in path.parents
        formal = json.loads((root/arm/stage/'receipt.json').read_bytes())
        assert formal['status']=='pass' and formal['rows']==formal['formal_rows']==9508
        assert [formal['metrics']['bbs'][key] for key in ('rec_hits25','rec_hits50')]==decision['hits'][identifier]
        payload = torch.load(str(path),map_location='cpu')
        step = 0 if stage=='initial_formal' else 3723
        assert payload['step']==step and payload['spec_sha256']==sha(spec_path)
        assert payload['reference_mode']==spec['reference_mode']
        assert payload['common_output_reset']==['output.weight','output.bias']
        assert payload['retained_hidden_prior_updates']==11169
        for key in ('checkpoint_sha256','base_terminal_sha256','geometry_terminal_sha256','source_port_sha256','mask_reference_sha256'):
            assert payload[key]==spec[key]
        state = payload['state_delta']
        assert len(state)==10 and sum(value.numel() for value in state.values())==456102
        assert all(key.startswith('candidate_box_refiner.') for key in state)
        head = MaskReferenceBoxRefiner(spec['reference_mode'])
        stripped = {key[len('candidate_box_refiner.'):]:value for key,value in state.items()}
        assert set(stripped)==set(head.state_dict())
        assert all(value.shape==head.state_dict()[key].shape and value.dtype==head.state_dict()[key].dtype
            for key,value in stripped.items())
        head.load_state_dict(stripped,strict=True)
        assert all(torch.equal(value,head.state_dict()[key]) for key,value in stripped.items())
        if step==0:
            assert payload['zero_update_architecture'] and not payload['optimizer']['state']
            assert payload['reset_output_total_updates']==0
            assert torch.count_nonzero(stripped['output.weight'])==torch.count_nonzero(stripped['output.bias'])==0
            assert all(torch.equal(value,prior['state_delta'][key]) for key,value in state.items()
                if not key.startswith('candidate_box_refiner.output.'))
        else:
            assert payload['head_only'] and payload['retained_hidden_total_updates']==14892
            assert payload['reset_output_total_updates']==3723 and len(payload['optimizer']['state'])==10
            assert all(int(value['step'])==3723 for value in payload['optimizer']['state'].values())
            assert len(payload['row_ids'])==len(set(payload['row_ids']))==29778
            fit = json.loads((root/arm/'receipt.json').read_bytes())
            assert sha(path)==fit['terminal_sha256']
            assert sha(root/arm/'train.jsonl')==fit['train_log_sha256']
        paths[identifier]=path
        identities[identifier]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),
            actual_step=step,zero_update_architecture=step==0)
assert set(paths)==set(decision['hits']) and decision['hits']['protected_geometry_parent']==[5616,4511]
winner = decision['winner']
assert winner in paths
best = max(paths,key=lambda key:(decision['hits'][key][1],key=='protected_geometry_parent',decision['hits'][key][0]))
assert winner==best
if winner!='protected_geometry_parent':
    restored = decision['selected_restore']
    assert restored['status']=='pass' and restored['checkpoint_sha256']==identities[winner]['sha256']
    assert restored['strict_CPU_geometry_restore'] and restored['all_full_model_states_equal_to_original_construction']
    assert restored['old_geometry_checkpoint_needed_by_direct_factory'] is False
    assert restored['optimizer']['all_keys_moments_steps_and_groups_exact']
    assert restored['actual_step']==identities[winner]['actual_step']
deleted=[]
for identifier,path in paths.items():
    if identifier!=winner:
        assert path.resolve()==Path(identities[identifier]['path'])
        path.unlink()
        deleted.append(dict(candidate=identifier,**identities[identifier]))
assert paths[winner].is_file() and sha(paths[winner])==identities[winner]['sha256']
assert sha(g_parent)==base['base_terminal_sha256']
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status='CLOSED_NONBEST_WEIGHTS_REMOVED',
    retained_best=dict(candidate=winner,hits=decision['hits'][winner],**identities[winner]),
    deleted=deleted,released_bytes=sum(entry['bytes'] for entry in deleted),
    weights_before=5,weights_after=1,original_G_preserved=True,V99_paths_touched=0,
    negative_weight_archive_created=False,raw_evidence_deleted=False,
    decision_summary_sha256=decision['summary_sha256'],fresh_audit_sha256=decision['audit_sha256'])
(root/'weight_retention.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
