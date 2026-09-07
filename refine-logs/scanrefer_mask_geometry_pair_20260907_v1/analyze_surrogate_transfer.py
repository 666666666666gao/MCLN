"""Paired analysis of archived predictions only; no inference or tuning."""
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    'baseline_rows.json': '2dee88c98cfa718187418f7a27f2a87cc95820191a79ffed0a66b8674507f9ee',
    'terminal_rows.json': '06cbf10a4bd5f7dab0f89fcf704591a7bc445b65719e03204c5ada9893646601',
}
data = {}
for name, expected in EXPECTED.items():
    raw = (ROOT / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, name
    data[name] = json.loads(raw)
baseline = data['baseline_rows.json']['native_gt']
control = data['terminal_rows.json']['native_gt']
candidate = data['terminal_rows.json']['native_gt_mask_geometry']
assert len(baseline) == len(control) == len(candidate) == 6887
assert len({r['row_id'] for r in baseline}) == len(baseline)
for triple in zip(baseline, control, candidate):
    for key in ['row_id', 'scan_id', 'physical_space', 'point_sha256', 'root_box']:
        assert triple[0][key] == triple[1][key] == triple[2][key], key


def effective(row, field):
    return row[field] if field != 'hard_iou' or row['hard_valid'] else 0.0


def effects(old, new, field, threshold):
    before = [effective(r, field) >= threshold for r in old]
    after = [effective(r, field) >= threshold for r in new]
    repair = sum(not a and b for a, b in zip(before, after))
    damage = sum(a and not b for a, b in zip(before, after))
    return {'old_hits': sum(before), 'new_hits': sum(after),
            'repair': repair, 'damage': damage, 'net': repair - damage}


def comparison(old, new, indices):
    old = [old[i] for i in indices]
    new = [new[i] for i in indices]
    result = {'rows': len(indices)}
    for field in ['soft_iou', 'hard_iou', 'native_rec_iou', 'rec_iou']:
        result[field] = {
            'old_mean': statistics.mean(effective(r, field) for r in old),
            'new_mean': statistics.mean(effective(r, field) for r in new),
            'thresholds': {str(t): effects(old, new, field, t) for t in [.25, .5]},
        }
    result['soft_repairs_at_050'] = {}
    for direction, predicate in [
        ('repair', lambda a, b: a['soft_iou'] < .5 <= b['soft_iou']),
        ('damage', lambda a, b: b['soft_iou'] < .5 <= a['soft_iou']),
    ]:
        pairs = [(a, b) for a, b in zip(old, new) if predicate(a, b)]
        result['soft_repairs_at_050'][direction] = {
            'rows': len(pairs),
            'hard_effects': effects([p[0] for p in pairs], [p[1] for p in pairs], 'hard_iou', .5),
            'deployed_effects': effects([p[0] for p in pairs], [p[1] for p in pairs], 'rec_iou', .5),
            'old_hard_already_passed': sum(effective(a, 'hard_iou') >= .5 for a, b in pairs),
            'row_ids': [a['row_id'] for a, b in pairs],
        }
    return result


result = {
    'schema': 'mcln-mask-geometry-saved-surrogate-transfer-v1',
    'source_hashes': EXPECTED,
    'scope': '6887 backbone-seen module holdout; GT-matched root diagnostics are not deployment REC',
    'row_points_and_root_gt_exactly_equal': True,
    'new_inference': False, 'formal_rows': 0, 'parameters_updated': 0,
    'thresholds': [.25, .5],
    'matched_query_index_is_not_semantic_identity_proof': True,
    'full_variant_set_and_point_logits_saved': False,
    'comparisons': {}, 'geometry_gap': {},
}
for label, old in [('candidate_vs_control', control), ('candidate_vs_baseline', baseline)]:
    same = [i for i in range(6887) if old[i]['matched_root_query'] == candidate[i]['matched_root_query']]
    changed = [i for i in range(6887) if old[i]['matched_root_query'] != candidate[i]['matched_root_query']]
    result['comparisons'][label] = {
        'all': comparison(old, candidate, range(6887)),
        'same_matched_query_index': comparison(old, candidate, same),
        'changed_matched_query_index': comparison(old, candidate, changed),
    }
for label, rows in [('baseline', baseline), ('control', control), ('candidate', candidate)]:
    valid = [r for r in rows if r['hard_valid']]
    absolute_gap = [abs(r['soft_iou'] - r['hard_iou']) for r in valid]
    normalized_face_gap = []
    for row in valid:
        soft, hard, gt = row['soft_box'], row['hard_box'], row['root_box']
        assert min(gt[3:]) > 0
        normalized_face_gap.append(statistics.mean(
            abs((soft[j] + side * soft[j+3] / 2) -
                (hard[j] + side * hard[j+3] / 2)) / gt[j+3]
            for j in range(3) for side in [-1, 1]))
    result['geometry_gap'][label] = {
        'valid_hard_rows': len(valid),
        'mean_abs_soft_hard_iou_gap': statistics.mean(absolute_gap),
        'median_abs_soft_hard_iou_gap': statistics.median(absolute_gap),
        'mean_normalized_face_gap': statistics.mean(normalized_face_gap),
        'median_normalized_face_gap': statistics.median(normalized_face_gap),
        'soft050_fail_hard050_pass': sum(r['soft_iou'] < .5 <= r['hard_iou'] for r in valid),
        'soft050_pass_hard050_fail': sum(r['hard_iou'] < .5 <= r['soft_iou'] for r in valid),
    }
output = ROOT / 'surrogate_transfer_analysis.json'
output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
brief = {k: {'all_rows': v['all']['rows'],
             'same_query_rows': v['same_matched_query_index']['rows'],
             'changed_query_rows': v['changed_matched_query_index']['rows'],
             'same_query_050': {f: v['same_matched_query_index'][f]['thresholds']['0.5']
                                for f in ['soft_iou', 'hard_iou', 'rec_iou']},
             'soft_repairs': {d: {k: n for k, n in x.items() if k != 'row_ids'}
                              for d, x in v['all']['soft_repairs_at_050'].items()}}
         for k, v in result['comparisons'].items()}
print(json.dumps({'comparisons': brief, 'geometry_gap': result['geometry_gap'],
                  'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))
