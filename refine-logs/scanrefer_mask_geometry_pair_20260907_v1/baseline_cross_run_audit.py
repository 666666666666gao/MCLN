"""Recount the complete pre-update baseline without changing training or selection."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

root = Path('/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1')
previous_root = Path('/root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1')
sys.path.insert(0, str(root / 'scripts'))
from audit_scanrefer_mask_geometry_pair import geometry_objective, overlap


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(root / 'input_manifest.json')
assert sha(root / 'input_manifest.json') == '15f46411069a7172a55373c5c13075b22bcb4146d39251e9fca2c37ed5867eb3'
for name, digest in manifest['files'].items():
    assert sha(root / name) == digest, name
prior_manifest = read(previous_root / 'input_manifest.json')
for field in ['artifacts', 'source_manifest_sha256', 'data_root', 'split_protocol_sha256']:
    assert manifest[field] == prior_manifest[field], field
assert manifest['train_superpoint_files'] == prior_manifest['train_superpoint_files']
rows_by_arm = read(root / 'baseline_rows.json')
metrics = read(root / 'baseline_metrics.json')
native_metrics = read(root / 'baseline_native_metrics.json')
geometry_metrics = read(root / 'baseline_geometry_metrics.json')
control, candidate = 'native_gt', 'native_gt_mask_geometry'
for record in [rows_by_arm, metrics, native_metrics, geometry_metrics]:
    assert set(record) == {control, candidate}
    assert record[control] == record[candidate]
rows = rows_by_arm[control]
previous = read(previous_root / 'baseline_rows.json')['gt_only']
assert len(rows) == len(previous) == 6887
assert [row['row_id'] for row in rows] == read(Path(manifest['split_protocol']))['row_ids']['holdout']
identity_fields = ['row_id', 'scan_id', 'physical_space', 'point_sha256']
identity_differences = [after['row_id'] for before, after in zip(previous, rows)
                        if any(before[field] != after[field] for field in identity_fields)]
decision_fields = ['rec_iou', 'native_rec_iou', 'native_query_index', 'mask_iou', 'selected_variant_position']
differences = {field: [after['row_id'] for before, after in zip(previous, rows)
                      if before[field] != after[field]] for field in decision_fields}
system = {'rows': len(rows), 'mask_miou': sum(row['mask_iou'] for row in rows) / len(rows) * 100.}
native = {'rows': len(rows)}
for field in ['rec_iou', 'native_rec_iou', 'mask_iou', 'soft_iou', 'hard_iou']:
    assert all(math.isfinite(row[field]) and 0. <= row[field] <= 1. for row in rows), field
for threshold, suffix in [(.25, '025'), (.5, '050')]:
    for field, prefix in [('rec_iou', 'rec'), ('mask_iou', 'mask')]:
        system[prefix + '_hits' + suffix] = sum(row[field] > threshold for row in rows)
    native['rec_hits' + suffix] = sum(row['native_rec_iou'] > threshold for row in rows)
assert system == metrics[control]
assert native == native_metrics[control]
assert all(0 <= row['matched_root_query'] < 256 for row in rows)
soft = np.asarray([row['soft_box'] for row in rows], dtype=np.float64)
hard = np.asarray([row['hard_box'] for row in rows], dtype=np.float64)
targets = np.asarray([row['root_box'] for row in rows], dtype=np.float64)
valid = np.asarray([row['hard_valid'] for row in rows], dtype=bool)
assert np.all(hard[~valid] == 0.)
l1, giou = geometry_objective(soft, targets)
recomputed = {'soft_iou': overlap(soft, targets), 'hard_iou': overlap(hard, targets) * valid,
              'soft_l1': l1, 'soft_giou_loss': giou}
errors = {}
for field, values in recomputed.items():
    observed = np.asarray([row[field] for row in rows])
    errors[field] = float(np.max(np.abs(values - observed)))
    np.testing.assert_allclose(values, observed, atol=2e-5, rtol=2e-6)
geometry = geometry_metrics[control]
assert geometry['rows'] == len(rows) and geometry['hard_valid'] == int(valid.sum())
for kind, mean_field in [('soft', 'soft_mean_iou'), ('hard', 'hard_mean_iou_invalid_as_zero')]:
    values = [row[kind + '_iou'] for row in rows]
    assert geometry[kind + '_hits'] == [sum(value > threshold for value in values) for threshold in [.25, .5]]
    np.testing.assert_allclose(geometry[mean_field], np.mean(recomputed[kind + '_iou']), atol=2e-5, rtol=2e-6)
np.testing.assert_allclose(geometry['mean_soft_geometry_loss'], np.mean(5. * l1 + giou), atol=2e-5, rtol=2e-6)
files = ['baseline_rows.json', 'baseline_metrics.json', 'baseline_native_metrics.json', 'baseline_geometry_metrics.json']
result = {
    'schema': 'mcln-mask-geometry-baseline-cross-run-audit-v1', 'status': 'pass',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'rows': len(rows), 'current_arms_exact_row_parity': True,
    'same_protected_artifact_source_mesh_split_manifests': True,
    'identity_difference_row_ids': identity_differences, 'cross_run_difference_row_ids': differences,
    'system_metrics': system, 'native_rec_metrics': native, 'matched_root_geometry_metrics': geometry,
    'maximum_saved_geometry_errors': errors,
    'baseline_file_sha256': {name: sha(root / name) for name in files},
    'prior_baseline_rows_sha256': sha(previous_root / 'baseline_rows.json'),
    'training_manifest_sha256': sha(root / 'input_manifest.json'),
    'audit_source_sha256': sha(Path(__file__)),
    'gpu_forwards': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0, 'formal_rows': 0,
    'baseline_only_not_quality_gain': True, 'is_new_promotion_gate': False,
    'geometry_scope': 'Current GT-matched root query, not deployed REC',
}
with (root / 'baseline_cross_run_audit.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
print(json.dumps(result), flush=True)
