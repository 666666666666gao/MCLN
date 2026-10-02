"""Describe archived selected-query P3 corrections without replaying a model."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

root = Path(r'C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete')
intake = json.loads((root / 'INTAKE.json').read_bytes())


def quantile(values, probability):
    values = sorted(values)
    location = (len(values) - 1) * probability
    lower = int(location)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (location - lower)


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values),
                median=quantile(values, .5), p90=quantile(values, .9),
                p99=quantile(values, .99), maximum=max(values))


def edges(box):
    return [box[axis] + sign * box[axis + 3] / 2
            for axis in range(3) for sign in (-1, 1)]


def summarize(rows):
    shifts = []
    relative_shifts = []
    iou_changes = []
    center_shifts = []
    size_changes = []
    edge_error_changes = []
    for row in rows:
        sample = row['bbs']
        coarse, final, target = sample['coarse_box'], sample['box'], row['root_box']
        assert all(math.isfinite(v) for v in coarse + final + target)
        assert all(v > 0 for v in target[3:])
        coarse_edges, final_edges, target_edges = edges(coarse), edges(final), edges(target)
        shifts.append(max(abs(a - b) for a, b in zip(final_edges, coarse_edges)))
        relative_shifts.append(max(abs(a - b) / target[3 + index // 2]
                                   for index, (a, b) in enumerate(zip(final_edges, coarse_edges))))
        center_shifts.append(math.sqrt(sum((final[k] - coarse[k]) ** 2 for k in range(3))))
        size_changes.append(max(abs(final[k] - coarse[k]) for k in range(3, 6)))
        iou_changes.append(sample['iou'] - sample['coarse_iou'])
        error0 = statistics.mean(abs(a - b) / target[3 + index // 2]
                                for index, (a, b) in enumerate(zip(coarse_edges, target_edges)))
        error1 = statistics.mean(abs(a - b) / target[3 + index // 2]
                                for index, (a, b) in enumerate(zip(final_edges, target_edges)))
        edge_error_changes.append(error1 - error0)
    fixes = sum(r['bbs']['coarse_iou'] <= .5 < r['bbs']['iou'] for r in rows)
    breaks = sum(r['bbs']['iou'] <= .5 < r['bbs']['coarse_iou'] for r in rows)
    return dict(rows=len(rows), final_hits50=sum(r['bbs']['iou'] > .5 for r in rows),
                coarse_hits50=sum(r['bbs']['coarse_iou'] > .5 for r in rows),
                fixes50=fixes, breaks50=breaks, net50=fixes-breaks,
                nonpositive_selected_coarse_sizes=sum(any(v <= 0 for v in r['bbs']['coarse_box'][3:]) for r in rows),
                nonpositive_selected_final_sizes=sum(any(v <= 0 for v in r['bbs']['box'][3:]) for r in rows),
                center_shift_m=describe(center_shifts), max_size_change_m=describe(size_changes),
                max_face_shift_m=describe(shifts), max_face_shift_over_GT_axis_size=describe(relative_shifts),
                iou_delta=describe(iou_changes),
                mean_normalized_face_error_delta=describe(edge_error_changes),
                positive_iou_delta=sum(v > 0 for v in iou_changes),
                negative_iou_delta=sum(v < 0 for v in iou_changes),
                zero_iou_delta=sum(v == 0 for v in iou_changes),
                abs_iou_delta_over_001=sum(abs(v) > .01 for v in iou_changes),
                abs_iou_delta_over_005=sum(abs(v) > .05 for v in iou_changes),
                max_face_shift_under_1cm=sum(v < .01 for v in shifts))


results = {}
sources = {}
for stage, expected_n, expected_hits in [('terminal', 6887, 5614), ('formal', 9508, 4401)]:
    name = 'p3/' + stage + '/rows.jsonl'
    raw = (root / name).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == intake['files'][name]['sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == expected_n
    result = summarize(rows)
    assert result['final_hits50'] == expected_hits
    assert result['final_hits50'] - result['coarse_hits50'] == result['net50']
    result['coarse_iou_groups'] = []
    for lower, upper in [(0, .25), (.25, .45), (.45, .5), (.5, .55), (.55, 1.000001)]:
        subset = [r for r in rows if lower <= r['bbs']['coarse_iou'] < upper]
        result['coarse_iou_groups'].append(dict(lower_inclusive=lower, upper_exclusive=upper,
                                               summary=summarize(subset)))
    # Equal row-count ranks by GT volume are diagnostic only, with deterministic row-ID tie order.
    ranked = sorted(rows, key=lambda r: (math.prod(r['root_box'][3:]), r['row_id']))
    result['GT_volume_rank_groups'] = []
    for group in range(4):
        subset = ranked[len(rows) * group // 4:len(rows) * (group + 1) // 4]
        result['GT_volume_rank_groups'].append(dict(group=group+1,
            GT_volume_min=math.prod(subset[0]['root_box'][3:]),
            GT_volume_max=math.prod(subset[-1]['root_box'][3:]), summary=summarize(subset)))
    results[stage] = result
    sources[stage] = dict(path=str(root / name), bytes=len(raw), sha256=digest)

receipt = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    status='complete_archived_selected_query_analysis', sources=sources,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results=results,
    primary_mode='bbs', model_forwards=0, optimizer_steps=0,
    geometry_recording='Both stored coarse sizes and evaluated final sizes were clamped to at least 1e-6 by the original evaluation runner before recording. Statistics describe saved evaluated geometry, not unclamped network residuals; saved positive sizes do not prove raw output validity.',
    limits='Archived selected and coarse evaluated boxes only, with the original evaluation size clamp preserved. IoU deltas use stored values whose threshold parity was independently checked in CPU_RECOUNT. This is internal forward correction at the same selected Query, not a trained-without-P3 ablation or measurement of unclamped regression logits. GT sizes/volume groups are offline diagnostic labels; no inference GT is added. No complete candidate arrays, member points or raw residual logits are available. Terminal and formal contain different scenes, so group differences do not establish a cause of generalization loss. Coordinate units inherit the ScanNet evaluation protocol and are not independently calibrated here. Current same-tail pair is unchanged.')
(root / 'SELECTED_REFINEMENT_MAGNITUDE.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({stage: {key: result[key] for key in
    ('rows','coarse_hits50','final_hits50','fixes50','breaks50','net50','max_face_shift_m',
     'max_face_shift_over_GT_axis_size','iou_delta','abs_iou_delta_over_001',
     'max_face_shift_under_1cm')} for stage,result in results.items()}))
