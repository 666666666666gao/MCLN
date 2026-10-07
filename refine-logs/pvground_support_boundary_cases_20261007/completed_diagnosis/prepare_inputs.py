"""Prepare the191 diagnosed rows and retain their original B8 contexts."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
previous = root.parent / 'pvground_face_support_20261007'
source = root.parent / 'pvground_reference_keep_20261006/cached_reference_errors/0.25_damaged.jsonl'
history = root.parent / 'pvground_mask_reference_20261006/complete_fit/fused_mask_reference/initial_formal/rows.jsonl'
cases = [json.loads(line) for line in source.read_text().splitlines()]
rows = {row['row_id']: row for row in map(json.loads, history.read_text().splitlines())}
blocks = sorted({row['row_id'] // 8 for row in cases})
ids = [index for block in blocks for index in range(block * 8, min((block + 1) * 8, 9508))]
items = []
groups = {'overextended': 0, 'missing_gt_extent': 0}
for row in cases:
    old = rows[row['row_id']]
    assert old['bbs']['query'] == row['query']
    assert old['scan_id'] == row['scan_id'] and old['target_id'] == row['target_id']
    assert old['root_box'] == row['root_gt']
    group = 'overextended' if row['gt_volume_covered'] >= .95 and row['reference_to_gt_volume_ratio'] > 4 else 'missing_gt_extent'
    assert group != 'missing_gt_extent' or row['gt_volume_covered'] < .95
    groups[group] += 1
    items.append(dict(cached=row, point_sha256=old['point_sha256'], group=group))
assert len(items) == 191 and groups == {'overextended': 128, 'missing_gt_extent': 63}
packet = dict(status='INPUTS_PREPARED_NOT_EXECUTED', historical_source_sha256=hashlib.sha256(history.read_bytes()).hexdigest(),
    cohort_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), groups=groups, original_formal_batch_size=8,
    batch_indices=blocks, forward_row_ids=ids, forward_batches=len(blocks), diagnostic_rows=items)
(root / 'case_manifest.json').write_text(json.dumps(packet, indent=2) + '\n', encoding='utf-8')
old_spec = json.loads((previous / 'pair_spec.json').read_bytes())
keys = ['runtime', 'env_spec_sha256', 'input_manifest', 'model_source', 'source_port', 'source_port_sha256',
        'helper_root', 'runner_files', 'base_terminal', 'base_terminal_sha256', 'selected_terminal',
        'selected_terminal_sha256', 'selected_factory_sha256', 'mask_reference_sha256', 'checkpoint_sha256', 'seed']
spec = {key: old_spec[key] for key in keys}
spec.update(root='/root/autodl-tmp/pvground_support_boundary_cases_20261007', batch_size=8,
    diagnostic_rows=191, forward_batches=len(blocks), forward_rows=len(ids), optimizer_steps=0,
    weights_created=0, no_multiseed=True, required_reserve_bytes=600 * 1024**2,
    estimated_seconds=900, first_check_seconds=720, later_poll_seconds=240,
    estimate_basis='Same B8 formal forward measured0.286sec/row; target-context rows plus warm model/data load and bounded raw-slice writes; estimate15min, firstcheck12min.')
(root / 'diagnostic_spec.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
for name in ('selected_mask_reference_factory.py', 'mask_reference.py'):
    (root / name).write_bytes((previous / name).read_bytes())
print(json.dumps(dict(cases=191, groups=groups, original_batches=len(blocks), forward_rows=len(ids))))
