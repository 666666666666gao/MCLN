"""Additional requested narrative/provenance checks; no source/model execution."""
import hashlib
import json
from pathlib import Path

import numpy as np

A = Path(__file__).absolute().parent
T = A.parent.parent
C = T / 'pvground_compressed_geometry_support_20261008'
O = T / 'pvground_mask_reference_20261006'


def load(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


path = C / 'complete_fit/formal/rows.jsonl'
rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
truth = np.asarray([r['root_box'] for r in rows], dtype=np.float64)
pred = np.asarray([r['arms']['content']['box'] for r in rows], dtype=np.float64)
lengths = np.maximum(np.minimum(pred[:, :3]+pred[:, 3:]/2, truth[:, :3]+truth[:, 3:]/2)
                     - np.maximum(pred[:, :3]-pred[:, 3:]/2, truth[:, :3]-truth[:, 3:]/2), 0)
zero = lengths.prod(1) == 0
mask_iou = np.asarray([r['arms']['content']['mask_iou'] for r in rows])
assert int(zero.sum()) == 3074
assert int((zero & (mask_iou > .5)).sum()) == 0
manifest_path = T / 'pvground_g_p2_20261002/complete/source/appearance_source_manifest.json'
input_path = manifest_path.parent / 'input_manifest.json'
manifest = load(manifest_path)
assert sha(manifest_path) == load(input_path)['source_manifest_sha256']
handler_path = T / 'pv_ground_source_20260905/src/visual_data_handlers.py'
assert sha(handler_path) == manifest['files']['src/visual_data_handlers.py']
loader_path = manifest_path.parent / 'joint_det_dataset.py'
assert sha(loader_path) == manifest['files']['src/joint_det_dataset.py']
retained_path = C / 'postrun/RETAINED_BEST.json'
retained = load(retained_path)
archive = Path(retained['best_local_archive'])
cross = load(A.parent / 'RETAINED_CROSS_FORWARD_HYPOTHESIS.json')
assert sha(archive) == retained['best_checkpoint']['sha256'] == cross['retained_checkpoint_sha256']
assert archive.stat().st_size == retained['best_checkpoint']['bytes']
current_spec = load(C / 'pair_spec.json')
old_retention_path = O / 'weight_retention.json'
old_retention = load(old_retention_path)
assert old_retention['retained_best']['sha256'] == current_spec['selected_terminal_sha256']
assert old_retention['retained_best']['candidate'] == 'fused_mask_reference/initial_formal'
assert old_retention['retained_best']['actual_step'] == 0
assert retained['spec_sha256'] == sha(C / 'pair_spec.json')
for dependency in retained['required_dependencies']:
    key = {'PV-Ground_ScanRefer.pth': 'checkpoint_sha256',
           'mcln_pvground_scanrefer_finetune_20260918_semantic_assignment_v1': 'base_terminal_sha256',
           'pvground_mask_reference_20261006': 'selected_terminal_sha256'}
    matches = [field for marker, field in key.items() if marker in dependency['path']]
    assert len(matches) == 1 and dependency['sha256'] == current_spec[matches[0]]
hashes = load(A / 'REVIEWED_FILE_HASHES.json')
for p in (manifest_path, handler_path, retained_path, archive, old_retention_path, C/'postrun/checkpoint_inspection.json'):
    hashes[str(p)] = dict(bytes=p.stat().st_size, sha256=sha(p))
(A / 'REVIEWED_FILE_HASHES.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
result = dict(status='PASS', source_current_rows_sha256=sha(path), box_zero_overlap_rows=int(zero.sum()),
    box_zero_overlap_with_mask_iou_above_half=int((zero & (mask_iou > .5)).sum()),
    box_zero_overlap_with_positive_mask_iou=int((zero & (mask_iou > 0)).sum()),
    box_zero_overlap_max_saved_mask_iou=float(mask_iou[zero].max()),
    box_zero_overlap_does_not_mean_zero_mask_intersection=True,
    original_scan_handler_source_sha256=sha(handler_path),
    original_scan_handler_matches_appearance_source_manifest=True,
    original_loader_matches_import_and_appearance_manifest=True,
    current_selected_checkpoint_archive_sha256=sha(archive), current_selected_checkpoint_archive_bytes=archive.stat().st_size,
    current_selected_checkpoint_loaded=False, core_dependency_receipts_and_specs_match=True,
    old_zero_update_snapshot_identity=old_retention['retained_best'],
    raw_scan_dataset_not_rederived=True, same_forward_current_model_result_not_established=True)
(A / 'SUPPLEMENTARY_VERIFY.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
