"""Describe the completed fused retry against the raw endpoint; no model or GPU execution."""
import bisect
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

root = Path(r'C:\Users\gb\.codex\tmp\pvground_fused_support_20261002')
raw_root = root / 'complete_tail_fused_retry'
control_root = root / 'complete_tail_raw/arm'
raw_path = raw_root / 'arm/formal/rows.jsonl'
control_path = control_root / 'formal/rows.jsonl'
raw_bytes = raw_path.read_bytes()
control_bytes = control_path.read_bytes()
intake = json.loads((raw_root / 'INTAKE.json').read_bytes())
assert hashlib.sha256(raw_bytes).hexdigest() == intake['files']['arm/formal/rows.jsonl']['sha256']
raw_receipt = json.loads((raw_root / 'arm/formal/receipt.json').read_bytes())
control_receipt = json.loads((control_root / 'formal/receipt.json').read_bytes())
assert hashlib.sha256(control_bytes).hexdigest() == control_receipt['rows_sha256']
rows = [json.loads(line) for line in raw_bytes.splitlines()]
controls = [json.loads(line) for line in control_bytes.splitlines()]
assert len(rows) == len(controls) == 9508
assert len({row['row_id'] for row in rows}) == 9508
for row, control in zip(rows, controls):
    for key in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'):
        assert row[key] == control[key], (row['row_id'], key)
    assert all(math.isfinite(value) for value in row['root_box'])
    assert all(value > 0 for value in row['root_box'][3:])
volumes = [math.prod(row['root_box'][3:]) for row in rows]
cuts = statistics.quantiles(volumes, n=4, method='inclusive')
groups = [[index for index, volume in enumerate(volumes) if bisect.bisect_left(cuts, volume) == group]
          for group in range(4)]
assert all(groups) and sum(map(len, groups)) == 9508


def distribution(values):
    assert values and all(math.isfinite(value) for value in values)
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                p90=statistics.quantiles(values, n=10, method='inclusive')[8], maximum=max(values))


def faces(box):
    return [box[axis] + sign * box[axis+3] / 2 for axis in range(3) for sign in (-1, 1)]


def describe(indices, mode):
    selected = [rows[index][mode] for index in indices]
    reference = [controls[index][mode] for index in indices]
    result = dict(rows=len(indices), gt_volume_m3_min=min(volumes[index] for index in indices),
                  gt_volume_m3_max=max(volumes[index] for index in indices))
    for threshold, suffix in ((.25, '25'), (.5, '50')):
        coarse_hits = sum(item['coarse_iou'] > threshold for item in selected)
        refined_hits = sum(item['iou'] > threshold for item in selected)
        reference_hits = sum(item['iou'] > threshold for item in reference)
        fixes = sum(item['coarse_iou'] <= threshold < item['iou'] for item in selected)
        breaks = sum(item['iou'] <= threshold < item['coarse_iou'] for item in selected)
        paired_fixes = sum(old['iou'] <= threshold < new['iou'] for old, new in zip(reference, selected))
        paired_breaks = sum(new['iou'] <= threshold < old['iou'] for old, new in zip(reference, selected))
        assert refined_hits - coarse_hits == fixes - breaks
        assert refined_hits - reference_hits == paired_fixes - paired_breaks
        result['rec'+suffix] = dict(coarse=coarse_hits, refined=refined_hits,
            same_query_fixes=fixes, same_query_breaks=breaks, same_query_net=fixes-breaks,
            same_tail_raw=reference_hits,
            versus_same_tail_raw=refined_hits-reference_hits,
            paired_fixes=paired_fixes, paired_breaks=paired_breaks,
            full256_saved_GPU_flags=sum(item['oracle'+suffix][-1] for item in selected))
    center_shift = []
    size_change = []
    maximum_face_change = []
    relative_face_change = []
    absolute_iou_change = []
    for index, item in zip(indices, selected):
        coarse = item['coarse_box']
        refined = item['box']
        assert all(math.isfinite(value) for value in coarse+refined)
        center_shift.append(1000 * math.dist(coarse[:3], refined[:3]))
        size_change.append(1000 * math.dist(coarse[3:], refined[3:]))
        face_changes = [abs(a-b) for a, b in zip(faces(coarse), faces(refined))]
        maximum_face_change.append(1000 * max(face_changes))
        relative_face_change.append(max(change / rows[index]['root_box'][3+face//2]
                                        for face, change in enumerate(face_changes)))
        absolute_iou_change.append(abs(item['iou']-item['coarse_iou']))
    result['center_shift_mm'] = distribution(center_shift)
    result['size_change_l2_mm'] = distribution(size_change)
    result['maximum_axis_face_change_mm'] = distribution(maximum_face_change)
    result['maximum_face_change_relative_to_GT_axis_size'] = distribution(relative_face_change)
    result['absolute_iou_change'] = distribution(absolute_iou_change)
    result['maximum_axis_face_change_over_10mm_rows'] = sum(value > 10 for value in maximum_face_change)
    return result


overall = {mode: describe(list(range(9508)), mode) for mode in ('bbs', 'bbf')}
by_volume = {f'Q{group+1}': {mode: describe(indices, mode) for mode in ('bbs', 'bbf')}
             for group, indices in enumerate(groups)}
for mode in ('bbs', 'bbf'):
    for suffix in ('25', '50'):
        assert overall[mode]['rec'+suffix]['refined'] == raw_receipt['metrics'][mode]['rec_hits'+suffix]
        assert overall[mode]['rec'+suffix]['same_tail_raw'] == control_receipt['metrics'][mode]['rec_hits'+suffix]
        for key in ('coarse', 'refined', 'same_query_fixes', 'same_query_breaks', 'same_query_net',
                    'same_tail_raw', 'versus_same_tail_raw',
                    'paired_fixes', 'paired_breaks', 'full256_saved_GPU_flags'):
            assert sum(group[mode]['rec'+suffix][key] for group in by_volume.values()) == overall[mode]['rec'+suffix][key]
result = dict(status='pass', time_cst=datetime.datetime.now().astimezone().isoformat(),
    fused_formal_path=str(raw_path), fused_formal_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    same_tail_raw_path=str(control_path),
    same_tail_raw_sha256=hashlib.sha256(control_bytes).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    GPU_forward=False, optimizer_updates=0,
    grouping='Inclusive GT-volume quartile cuts; equal-volume rows stay together; bisect_left assigns a cut value to the lower group.',
    volume_cuts_m3=cuts, overall=overall, by_GT_volume=by_volume,
    evidence_limits=('Saved evaluator boxes/IoUs only; box thresholds already independently CPU-recounted. '
        'GT is used only for offline diagnostic grouping, not inference. Groups count expressions rather than independent objects; '
        'small volume does not imply few observed points. Same-query coarse/refined changes are an internal diagnostic, '
        'not a trained no-refiner ablation. The same-tail raw comparison retains the documented amended E0 gate and cross-process initial output differences; exact functional-start parity is not established. '
        'Full256 values are saved GPU flags, not independently reconstructed all-candidate boxes. '
        'Single seed; no robustness or physical-instance-identity claim. Only archived completed fused/raw formal results are used.'))
output = root / 'FUSED_GEOMETRY_PROFILE.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(output=str(output), cuts=cuts, overall_bbs=overall['bbs'],
    volume_bbs={key:value['bbs'] for key,value in by_volume.items()})))
