"""Use the already-selected reference model's stored rows; no model or remote I/O."""
import hashlib
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root.parent / 'pvground_mask_reference_20261006/complete_fit/fused_mask_reference/initial_formal'
rows_path = source / 'rows.jsonl'
receipt = json.loads((source / 'receipt.json').read_bytes())
assert receipt['status'] == 'pass' and receipt['formal_rows'] == 9508
assert hashlib.sha256(rows_path.read_bytes()).hexdigest() == receipt['rows_sha256']
rows = [json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines()]
assert [row['row_id'] for row in rows] == list(range(9508))
assert [sum(row['bbs']['iou'] > t for row in rows) for t in (.25, .5)] == [5598, 4848]
assert all(row['bbs']['box'] == row['bbs']['reference_box'] for row in rows)


def volume(box):
    return box[3] * box[4] * box[5]


def features(row):
    gt = row['root_box']
    reference = row['bbs']['reference_box']
    prior = row['bbs']['coarse_box']
    assert min(gt[3:]) > 0 and min(reference[3:]) > 0 and min(prior[3:]) > 0
    gt_low = [gt[k] - gt[k + 3] / 2 for k in range(3)]
    gt_high = [gt[k] + gt[k + 3] / 2 for k in range(3)]
    ref_low = [reference[k] - reference[k + 3] / 2 for k in range(3)]
    ref_high = [reference[k] + reference[k + 3] / 2 for k in range(3)]
    inside = [max(0, min(gt_high[k], ref_high[k]) - max(gt_low[k], ref_low[k])) for k in range(3)]
    return dict(row_id=row['row_id'], scan_id=row['scan_id'], target_id=row['target_id'],
        query=row['bbs']['query'], prior_iou=row['bbs']['coarse_iou'], reference_iou=row['bbs']['iou'],
        mask_iou=row['bbs']['mask_iou'], reference_valid=row['bbs']['reference_valid'],
        reference_to_gt_volume_ratio=volume(reference) / volume(gt),
        gt_volume_covered=inside[0] * inside[1] * inside[2] / volume(gt),
        max_absolute_face_error_m=max(abs(a - b) for a, b in zip(ref_low + ref_high, gt_low + gt_high)),
        reference_box=reference, prior_box=prior, root_gt=gt)


def summarize(cohort):
    details = [features(row) for row in cohort]
    assert details
    return dict(rows=len(details), mask_good50=sum(item['mask_iou'] > .5 for item in details),
        invalid_reference=sum(not item['reference_valid'] for item in details),
        gt_covered95_and_reference_volume_over4=sum(item['gt_volume_covered'] >= .95 and item['reference_to_gt_volume_ratio'] > 4 for item in details),
        gt_coverage_below95=sum(item['gt_volume_covered'] < .95 for item in details),
        median_reference_to_gt_volume_ratio=statistics.median(item['reference_to_gt_volume_ratio'] for item in details),
        median_gt_volume_covered=statistics.median(item['gt_volume_covered'] for item in details),
        median_max_face_error_m=statistics.median(item['max_absolute_face_error_m'] for item in details))


cohorts = {}
for threshold in (.25, .5):
    name = str(threshold)
    cohorts[name + '_repaired'] = [row for row in rows if row['bbs']['coarse_iou'] <= threshold < row['bbs']['iou']]
    cohorts[name + '_damaged'] = [row for row in rows if row['bbs']['iou'] <= threshold < row['bbs']['coarse_iou']]
damaged25 = {row['row_id'] for row in cohorts['0.25_damaged']}
damaged50 = {row['row_id'] for row in cohorts['0.5_damaged']}
wide_damage_only = [row for row in cohorts['0.25_damaged'] if row['row_id'] not in damaged50]
strict_damage_only = [row for row in cohorts['0.5_damaged'] if row['row_id'] not in damaged25]
summary = dict(status='CACHED_SELECTED_REFERENCE_ERRORS_ANALYZED', source_rows_sha256=receipt['rows_sha256'],
    source_model_hits=[5598, 4848], new_model_evaluations=0, new_optimizer_steps=0, remote_queries=0,
    cohorts={name: summarize(values) for name, values in cohorts.items()},
    damaged_at_both_thresholds=len(damaged25 & damaged50),
    wide_damage_only=len(wide_damage_only), strict_damage_only=len(strict_damage_only),
    no_inference_gt_rule_proposed=True,
    scope='Fixed same selected Query: coarse/native prior versus neutral Mask reference. Geometry overlap is not physical-instance identity; thresholds for error grouping are descriptive, not inference gates.')
output = root / 'cached_reference_errors'
assert not output.exists()
output.mkdir()
for name, values in cohorts.items():
    (output / (name + '.jsonl')).write_text(''.join(json.dumps(features(row)) + '\n' for row in values), encoding='utf-8')
(output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary), flush=True)
