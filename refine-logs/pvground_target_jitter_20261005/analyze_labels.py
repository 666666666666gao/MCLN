"""Recount the existing fixed candidates under pre-jitter annotation boxes."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
cohort = root.parent / 'pvground_query_geometry_cohort_20261005'
sys.path.insert(0, str(cohort))
from cohort_metrics import ARMS, cpu_iou, faces, paired

receipt = json.loads((root / 'complete/receipt.json').read_bytes())
assert receipt['status'] == 'complete' and receipt['sampled_points_and_native_noisy_GT_exact']
assert receipt['model_forwards'] == receipt['weights_loaded'] == receipt['optimizer_steps'] == 0
assert not receipt['CUDA_initialized'] and not receipt['native_gt_protocol_changed']
rows = [json.loads(line) for line in (root / 'complete/rows.jsonl').read_text().splitlines()]
prior_rows = [json.loads(line) for line in (cohort / 'complete/rows.jsonl').read_text().splitlines()]
assert len(rows) == len(prior_rows) == 64
arrays_by_batch = []
for index in range(8):
    with np.load(cohort / ('complete/batch_%02d.npz' % index), allow_pickle=False) as batch:
        arrays_by_batch.append({key: batch[key].copy() for key in batch.files})
arrays = {key: np.concatenate([batch[key] for batch in arrays_by_batch], axis=0) for key in arrays_by_batch[0]}
for row, prior_row in zip(rows, prior_rows):
    assert row['row_id'] == prior_row['row_id']
    assert row['point_sha256'] == prior_row['point_sha256']
    assert row['noisy_root_box'] == prior_row['root_box']
    assert row['scan_id'] == prior_row['scan_id']
assert np.array_equal(arrays['row_id'], np.asarray([row['row_id'] for row in rows]))
noisy = np.asarray([row['noisy_root_box'] for row in rows], dtype=np.float64)
clean = np.asarray([row['pre_jitter_root_box'] for row in rows], dtype=np.float64)
assert np.array_equal(arrays['root_box'], noisy)
assert np.isfinite(clean).all() and (clean[:, 3:] > 0).all()
support = ((2 * arrays['query_intersection'] > arrays['query_union']) &
           (2 * arrays['fused_intersection'] > arrays['fused_union']) &
           (arrays['matched_slot'] < 0))
ious = {target: {arm: cpu_iou(arrays[arm + '_boxes'], truth) for arm in ARMS}
        for target, truth in (('native_noisy', noisy), ('pre_jitter', clean))}
assert all(np.array_equal(ious['native_noisy'][arm] > .5, arrays[arm + '_iou'] > .5) for arm in ARMS)
qual = {target: support & (ious[target]['parent'] <= .5) for target in ious}
assert int(qual['native_noisy'].sum()) == 1090
coarse = arrays['coarse_box'].astype(np.float64)


def face_target(truth):
    low = truth[:, None, :3] - truth[:, None, 3:] / 2
    high = truth[:, None, :3] + truth[:, None, 3:] / 2
    return np.concatenate(((coarse[..., :3] - low) / coarse[..., 3:] * 4 - 2,
                           (high - coarse[..., :3]) / coarse[..., 3:] * 4 - 2), axis=-1)


targets = {'native_noisy': face_target(noisy), 'pre_jitter': face_target(clean)}
outside = {target: (value < -4) | (value > 4) for target, value in targets.items()}
assert np.array_equal(outside['native_noisy'], arrays['outside'])
fixed = qual['native_noisy']
row_index = np.arange(64)
selected = arrays['bbs'].argmax(-1)
selected_group = np.zeros_like(fixed)
selected_group[row_index, selected] = True
groups = dict(fixed_noisy_qualified=fixed, clean_qualified=qual['pre_jitter'],
              fixed_noisy_qualified_inside_clean_range=fixed & ~outside['pre_jitter'].any(-1),
              fixed_noisy_qualified_outside_clean_range=fixed & outside['pre_jitter'].any(-1),
              selected_query=selected_group)
results = {}
for target, truth in (('native_noisy', noisy), ('pre_jitter', clean)):
    result = dict(qualification_candidates=int(qual[target].sum()),
        qualification_rows=int(qual[target].any(-1).sum()),
        fixed_noisy_cohort_outside_candidates=int((fixed & outside[target].any(-1)).sum()),
        fixed_noisy_cohort_outside_faces=int((fixed[..., None] & outside[target]).sum()), arms={}, paired={})
    for arm in ARMS:
        error = np.abs(faces(arrays[arm + '_boxes']) - faces(truth)[:, None]).max(-1)
        item = {}
        for name, group in groups.items():
            count = int(group.sum())
            item[name] = dict(candidates=count)
            if count:
                item[name].update(mean_iou=float(ious[target][arm][group].mean()),
                    median_iou=float(np.median(ious[target][arm][group])),
                    hits50=int(((ious[target][arm] > .5) & group).sum()),
                    median_max_face_error_m=float(np.median(error[group])))
        result['arms'][arm] = item
    for before, after in (('parent', 'query_supported'), ('control', 'query_supported')):
        result['paired'][before + '_to_' + after] = {
            name: paired(ious[target][before], ious[target][after], group)
            for name, group in groups.items() if group.any()}
    results[target] = result
shift = np.abs(faces(noisy) - faces(clean)).max(-1)
summary = dict(status='CPU_RECOUNT_COMPLETE', rows=64, candidates=16384, accuracy_result=False,
    native_inputs_exact=True, model_forwards=0, weights_loaded=0, optimizer_steps=0,
    supported_unmatched_candidates=int(support.sum()),
    qualification_removed_by_clean_target=int((qual['native_noisy'] & ~qual['pre_jitter']).sum()),
    qualification_added_by_clean_target=int((~qual['native_noisy'] & qual['pre_jitter']).sum()),
    target_jitter=dict(median_max_face_shift_m=float(np.median(shift)),
        max_face_shift_m=float(shift.max()), rows_shift_over_1cm=int((shift > .01).sum()),
        rows_shift_over_5cm=int((shift > .05).sum())), targets=results,
    limits='Augmented fit64, correlated candidates, native noisy-label matching and Mask support held fixed. Pre-jitter boxes are annotation-derived, not a new evaluation protocol. No DFL recomputation without logits; no proof of formal-accuracy cause.')
out = root / 'analysis'
out.mkdir(exist_ok=True)
(out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
