"""Separate overlapping-root and nearer-other-GT proxies in retained candidates."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    source = Path(r'C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\candidate_audit\full_gt_scope_v2')
    output = Path(__file__).parent/'cached_p3_error_proxies'
    assert not output.exists()
    receipt = json.loads((source/'receipt.json').read_bytes())
    intake = json.loads((source/'INTAKE.json').read_bytes())
    raw = (source/'rows.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['rows_sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == receipt['rows'] == 9508
    labels = ('root_nearest_with_nonzero_overlap', 'another_scene_GT_nearer', 'no_annotated_overlap')
    counts = {label: {key: 0 for key in (
        'strict_errors', 'has_qualified_box', 'no_qualified_box',
        'has_unmatched_qualified_box', 'has_qualified_box_and_mask_same_query',
        'has_unmatched_qualified_box_and_mask_same_query', 'selected_loose_hit',
        'first_qualified_rank_beyond32', 'root_absent_from_scene_detection_GT',
    )} for label in labels}
    matched = {'root_matched_selected_strict_errors': 0, 'unmatched_selected_strict_errors': 0}
    index = 0
    hits = 0
    chunks = {}
    for entry in intake['chunks']:
        path = source/entry['name']
        content = path.read_bytes()
        assert len(content) == entry['bytes']
        chunks[entry['name']] = hashlib.sha256(content).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            assert data['root_iou'].shape[1:] == (256,)
            for offset, identity in enumerate(data['row_id']):
                row = rows[index]
                assert int(identity) == row['row_id'] == index
                query = row['selected_query']
                iou = data['root_iou'][offset]
                scene = data['best_scene_GT_iou'][offset]
                assert float(iou[query]) == row['selected_iou']
                assert np.array_equal(data['root_joint_best_scene_overlap'][offset], iou >= scene)
                hits += int(iou[query] > .5)
                if iou[query] <= .5:
                    if iou[query] > 0 and iou[query] >= scene[query]:
                        label = labels[0]
                    elif scene[query] > iou[query]:
                        label = labels[1]
                    else:
                        assert iou[query] == scene[query] == 0
                        label = labels[2]
                    group = counts[label]
                    good = iou > .5
                    same_mask_good = data['root_mask_iou'][offset] > .5
                    unmatched = data['matched_GT_slot'][offset] < 0
                    assert float(data['root_mask_iou'][offset, query]) == row['selected_mask_iou']
                    group['strict_errors'] += 1
                    group['has_qualified_box'] += int(good.any())
                    group['no_qualified_box'] += int(not good.any())
                    group['has_unmatched_qualified_box'] += int((good & unmatched).any())
                    group['has_qualified_box_and_mask_same_query'] += int((good & same_mask_good).any())
                    group['has_unmatched_qualified_box_and_mask_same_query'] += int((good & same_mask_good & unmatched).any())
                    group['selected_loose_hit'] += int(iou[query] > .25)
                    rank = row['first_qualified_rank']['50']
                    group['first_qualified_rank_beyond32'] += int(rank is not None and rank > 32)
                    group['root_absent_from_scene_detection_GT'] += int(not row['root_in_scene_detection_GT'])
                    matched['root_matched_selected_strict_errors'] += int(row['selected_matched_slot'] == 0)
                    matched['unmatched_selected_strict_errors'] += int(row['selected_matched_slot'] < 0)
                index += 1
    assert index == 9508 and hits == receipt['rec_hits']['50'] == 4406
    assert sum(g['strict_errors'] for g in counts.values()) == 5102
    assert sum(g['has_qualified_box'] for g in counts.values()) == 3408
    assert sum(g['has_unmatched_qualified_box'] for g in counts.values()) == 3209
    assert sum(g['selected_loose_hit'] for g in counts.values()) == 1160
    assert sum(g['first_qualified_rank_beyond32'] for g in counts.values()) == 1916
    assert matched['root_matched_selected_strict_errors'] + matched['unmatched_selected_strict_errors'] == 5102
    result = {
        'status': 'complete_cached_CPU_analysis', 'model': 'completed_tail_fused_negative_result',
        'rows': index, 'selected_hits50': hits, 'strict_errors': 5102,
        'geometry_proxy_groups': counts, 'selected_matching_roles': matched,
        'interpretation_limits': [
            'Closest box overlap is a geometry proxy, not proof of physical instance identity.',
            'The scene annotation list covers detection-vocabulary objects only; root GT is independently evaluated.',
            'The all-zero-overlap case is reported separately instead of treating a zero tie as a correct-root identity.',
            'A qualified box and mask refer to the same retained query; this uses GT only for offline diagnosis.',
            'This is the old fused P3 model, not the active G candidate-consistency pair.',
        ],
        'all_candidates_retained': 256, 'optimizer_updates': 0,
        'GPU_forward_executed': False, 'rows_sha256': receipt['rows_sha256'],
        'chunk_sha256': chunks, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir()
    (output/'SUMMARY.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status','rows','strict_errors','geometry_proxy_groups','selected_matching_roles')}, indent=2))


if __name__ == '__main__':
    main()
