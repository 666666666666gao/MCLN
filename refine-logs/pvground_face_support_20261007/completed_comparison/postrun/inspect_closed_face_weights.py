"""Inspect this closed negative pair on CPU; keep the unchanged protected model."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


root = Path('/root/autodl-tmp/pvground_face_support_20261007')
decision = json.loads(sys.argv[1])
assert decision['winner'] == 'protected_geometry_parent'
assert (root / 'fit_controller.exit').read_text().strip() == '0'
status = json.loads((root / 'fit_status.json').read_bytes())
assert status['status'] == 'complete' and status['protected_parents_exact']
destination = root / 'closed_weight_inspection.json'
assert not destination.exists()
spec = json.loads((root / 'pair_spec.json').read_bytes())
parent = Path(spec['selected_terminal'])
assert str(parent) == '/root/autodl-tmp/pvground_mask_reference_20261006/fused_mask_reference/initial.pth'
assert sha(parent) == spec['selected_terminal_sha256']
assert sha(spec['base_terminal']) == spec['base_terminal_sha256']
sys.path.insert(0, spec['helper_root'])
sys.path.insert(0, str(root))
from face_region_box_refiner import FaceRegionBoxRefiner
from whole_model_preflight_checks import optimizer_restore_exact

protected = torch.load(str(parent), map_location='cpu')
assert protected['step'] == 0 and protected['zero_update_architecture']
assert decision['hits']['protected_geometry_parent'] == [5598, 4848]
assert torch.count_nonzero(protected['state_delta']['candidate_box_refiner.output.weight']) == 0
assert torch.count_nonzero(protected['state_delta']['candidate_box_refiner.output.bias']) == 0
identities = {'protected_geometry_parent': dict(path=str(parent), bytes=parent.stat().st_size,
    sha256=sha(parent), actual_step=0)}
fit = json.loads((root / 'receipt.json').read_bytes())
formal = json.loads((root / 'formal/receipt.json').read_bytes())
assert fit['status'] == 'complete' and fit['training_steps_per_arm'] == 3723
assert formal['status'] == 'pass' and formal['formal_rows'] == formal['rows'] == 9508
assert sha(root / 'train.jsonl') == fit['train_log_sha256']
for arm in ('face_center', 'face_region'):
    identifier = arm + '/formal'
    path = root / arm / 'terminal.pth'
    assert path.resolve() == path and root in path.parents
    assert [formal['metrics'][arm][key] for key in ('rec_hits25', 'rec_hits50')] == decision['hits'][identifier]
    assert sha(path) == fit['terminal_sha256'][arm]
    payload = torch.load(str(path), map_location='cpu')
    assert payload['step'] == 3723 and payload['sampling_mode'] == arm and payload['head_only']
    assert payload['spec_sha256'] == sha(root / 'pair_spec.json') and payload['sampler_sha256'] == spec['sampler_sha256']
    assert payload['reference_mode'] == 'fused_mask' and payload['reference_keep_weight'] == 0.0
    assert payload['extra_geometry_weight'] == 1.0 and payload['common_output_reset'] == ['output.weight', 'output.bias']
    assert payload['retained_hidden_prior_updates'] == 11169 and payload['retained_hidden_total_updates'] == 14892
    assert payload['reset_output_total_updates'] == 3723 and len(payload['row_ids']) == len(set(payload['row_ids'])) == 29778
    for key in ('checkpoint_sha256', 'base_terminal_sha256', 'selected_terminal_sha256', 'source_port_sha256'):
        assert payload[key] == spec[key]
    state = payload['state_delta']
    assert len(state) == 10 and sum(value.numel() for value in state.values()) == 456102
    assert all(key.startswith('candidate_box_refiner.') for key in state)
    stripped = {key[len('candidate_box_refiner.'):]: value for key, value in state.items()}
    head = FaceRegionBoxRefiner(arm)
    head.load_state_dict(stripped, strict=True)
    assert all(torch.equal(value, head.state_dict()[key]) for key, value in stripped.items())
    optimizer = torch.optim.AdamW(head.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    optimizer.load_state_dict(payload['optimizer'])
    restored = optimizer_restore_exact(optimizer, payload['optimizer'])
    assert len(optimizer.state) == 10 and all(int(item['step']) == 3723 for item in optimizer.state.values())
    identities[identifier] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path), actual_step=3723,
        declared_sampler_restored=head.sampling_mode, strict_CPU_geometry_restore=True, optimizer=restored)
assert set(identities) == set(decision['hits'])
winner = max(identities, key=lambda key: (decision['hits'][key][0] >= 5620 and decision['hits'][key][1] >= 4764,
    decision['hits'][key][1], decision['hits'][key][0], key == 'protected_geometry_parent'))
assert winner == decision['winner']
prior_path = Path('/root/autodl-tmp/pvground_mask_reference_20261006/postrun_restoration/fused_mask_reference_initial_formal.json')
prior = json.loads(prior_path.read_bytes())
assert prior['status'] == 'pass' and prior['checkpoint_sha256'] == identities[winner]['sha256']
assert prior['strict_CPU_geometry_restore'] and prior['all_full_model_states_equal_to_original_construction']
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), status='CLOSED_FIXED_FACE_CHECKPOINTS_CPU_INSPECTED',
    winner=winner, identities=identities, decision=decision, script_sha256=sha(__file__),
    selected_restore=dict(reused_unchanged_full_model_CPU_witness=True, path=str(prior_path), sha256=sha(prior_path),
        selected_checkpoint_sha256=identities[winner]['sha256']),
    GPU_forward_replayed=False, optimizer_updates=0, weights_created=0, weights_deleted=0)
destination.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
