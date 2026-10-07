"""Check fixed closed checkpoint states on CPU before any retention action."""
import datetime
import hashlib
import json
from pathlib import Path
import random
import sys

import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


root = Path('/root/autodl-tmp/pvground_reference_keep_20261006')
decision = json.loads(sys.argv[1])
assert (root/'fit_controller.exit').read_text().strip() == '0'
status = json.loads((root/'fit_status.json').read_bytes())
assert status['status'] == 'complete' and status['protected_parents_exact']
destination = root/'closed_weight_inspection.json'
assert not destination.exists()
base = json.loads((root/'control_spec.json').read_bytes())
parent = Path(base['selected_terminal'])
assert str(parent) == '/root/autodl-tmp/pvground_mask_reference_20261006/fused_mask_reference/initial.pth'
assert sha(parent) == base['selected_terminal_sha256']
assert sha(base['base_terminal']) == base['base_terminal_sha256']
sys.path.insert(0, base['helper_root'])
sys.path.insert(0, str(root))
sys.path.insert(0, base['model_source'])
from mask_reference import MaskReferenceBoxRefiner
from whole_model_preflight_checks import optimizer_restore_exact

parent_payload = torch.load(str(parent), map_location='cpu')
assert parent_payload['step'] == 0 and parent_payload['zero_update_architecture']
identities = {'protected_geometry_parent': dict(path=str(parent), bytes=parent.stat().st_size,
    sha256=sha(parent), actual_step=0)}
payloads = {'protected_geometry_parent': parent_payload}
for arm in ('control', 'keep'):
    spec_path = root/(arm+'_spec.json')
    spec = json.loads(spec_path.read_bytes())
    assert spec['selected_terminal_sha256'] == base['selected_terminal_sha256']
    for stage, name in (('initial_formal', 'initial.pth'), ('formal', 'terminal.pth')):
        identifier = arm+'/'+stage
        path = root/arm/name
        assert path.resolve() == path and root in path.parents
        formal = json.loads((root/arm/stage/'receipt.json').read_bytes())
        assert formal['status'] == 'pass' and formal['rows'] == formal['formal_rows'] == 9508
        assert [formal['metrics']['bbs'][key] for key in ('rec_hits25','rec_hits50')] == decision['hits'][identifier]
        payload = torch.load(str(path), map_location='cpu')
        step = 0 if stage == 'initial_formal' else 3723
        assert payload['step'] == step and payload['spec_sha256'] == sha(spec_path)
        assert payload['reference_mode'] == 'fused_mask'
        assert payload['common_output_reset'] == ['output.weight','output.bias']
        assert payload['retained_hidden_prior_updates'] == 11169
        for key in ('checkpoint_sha256','base_terminal_sha256','selected_terminal_sha256','source_port_sha256','mask_reference_sha256'):
            assert payload[key] == spec[key]
        head = MaskReferenceBoxRefiner('fused_mask')
        state = payload['state_delta']
        assert len(state) == 10 and sum(value.numel() for value in state.values()) == 456102
        assert all(key.startswith('candidate_box_refiner.') for key in state)
        stripped = {key[len('candidate_box_refiner.'):]: value for key,value in state.items()}
        head.load_state_dict(stripped, strict=True)
        assert all(torch.equal(value,head.state_dict()[key]) for key,value in stripped.items())
        optimizer = torch.optim.AdamW(tuple(head.parameters()), lr=spec['lr'], weight_decay=spec['weight_decay'])
        optimizer.load_state_dict(payload['optimizer'])
        optimizer_check = optimizer_restore_exact(optimizer,payload['optimizer'])
        if step == 0:
            assert payload['zero_update_architecture'] and not optimizer.state
            assert all(torch.equal(value,parent_payload['state_delta'][key]) for key,value in state.items())
        else:
            assert payload['head_only'] and payload['retained_hidden_total_updates'] == 14892
            assert payload['reset_output_total_updates'] == 3723
            assert payload['reference_keep_weight'] == (0.0 if arm == 'control' else 1.0)
            assert len(optimizer.state) == 10 and all(int(item['step']) == 3723 for item in optimizer.state.values())
            assert len(payload['row_ids']) == len(set(payload['row_ids'])) == 29778
            fit = json.loads((root/arm/'receipt.json').read_bytes())
            assert sha(path) == fit['terminal_sha256']
            assert sha(root/arm/'train.jsonl') == fit['train_log_sha256']
        identities[identifier] = dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),
            actual_step=step,strict_CPU_geometry_restore=True,optimizer=optimizer_check)
        payloads[identifier] = payload
assert set(identities) == set(decision['hits'])
assert decision['hits']['protected_geometry_parent'] == [5598,4848]
winner = max(identities,key=lambda key:(decision['hits'][key][0] >= 5620 and decision['hits'][key][1] >= 4764,
    decision['hits'][key][1],decision['hits'][key][0],key == 'protected_geometry_parent'))
assert winner == decision['winner']
if winner == 'protected_geometry_parent':
    prior_path = Path('/root/autodl-tmp/pvground_mask_reference_20261006/postrun_restoration/fused_mask_reference_initial_formal.json')
    prior = json.loads(prior_path.read_bytes())
    assert prior['status'] == 'pass' and prior['checkpoint_sha256'] == identities[winner]['sha256']
    assert prior['strict_CPU_geometry_restore'] and prior['all_full_model_states_equal_to_original_construction']
    selected_restore = dict(reused_unchanged_full_model_CPU_witness=True,path=str(prior_path),sha256=sha(prior_path),
        selected_checkpoint_sha256=identities[winner]['sha256'])
else:
    import os
    import numpy as np
    from pcdet.config import cfg,cfg_from_yaml_file
    from selected_mask_reference_factory import build_selected_mask_reference_model
    runtime = Path(base['runtime'])
    env = json.loads((runtime/'env_spec.json').read_bytes())
    official = env['weight_dirs']['scanrefer']
    assert sha(official['path']) == official['sha256'] == base['checkpoint_sha256']
    os.chdir(base['model_source'])
    cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
    official_payload = torch.load(official['path'],map_location='cpu')
    g_payload = torch.load(base['base_terminal'],map_location='cpu')
    def seed():
        random.seed(2027)
        np.random.seed(2027)
        torch.manual_seed(2027)
    seed()
    previous,_,_ = build_selected_mask_reference_model(cfg,official_payload,g_payload,parent_payload,
        json.loads(Path(base['input_manifest']).read_bytes())['data_root'])
    expected = previous.state_dict()
    expected.update(payloads[winner]['state_delta'])
    seed()
    selected,_,construction = build_selected_mask_reference_model(cfg,official_payload,g_payload,payloads[winner],
        json.loads(Path(base['input_manifest']).read_bytes())['data_root'])
    assert set(selected.state_dict()) == set(expected)
    assert all(torch.equal(value,expected[key]) for key,value in selected.state_dict().items())
    optimizer = torch.optim.AdamW(tuple(p for p in selected.parameters() if p.requires_grad),lr=base['lr'],weight_decay=base['weight_decay'])
    optimizer.load_state_dict(payloads[winner]['optimizer'])
    selected_restore = dict(reused_unchanged_full_model_CPU_witness=False,strict_full_model_CPU_restore=True,
        all_full_model_states_equal_to_parent_plus_selected_delta=True,construction=construction,
        optimizer=optimizer_restore_exact(optimizer,payloads[winner]['optimizer']),selected_checkpoint_sha256=identities[winner]['sha256'])
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status='CLOSED_FIXED_CHECKPOINTS_CPU_INSPECTED',
    winner=winner,identities=identities,selected_restore=selected_restore,decision=decision,script_sha256=sha(__file__),
    GPU_forward_replayed=False,optimizer_updates=0,weights_created=0,weights_deleted=0)
destination.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
