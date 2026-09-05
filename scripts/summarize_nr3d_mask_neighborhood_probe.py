"""Summarize paired M4 evidence; no model or data reads."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(receipt):
    pairs = [(a, b) for batch in receipt['batches'] for a, b in zip(batch['native']['rows'], batch['nearest']['rows'])]
    neighborhoods = [item for batch in receipt['batches'] for item in batch['neighborhoods']]
    assert len(pairs) == len(neighborhoods) == 16
    counts = {}
    for cohort in ['present', 'target_bearing', 'majority_positive']:
        counts[cohort] = {name: sum(item['counts'][cohort][name] for item in neighborhoods)
                          for name in ['slots', 'empty_radius', 'native_without_target_seed',
                                       'nearest_without_target_seed', 'target_center_restored', 'target_center_lost']}

    def compare(values):
        native, nearest = zip(*values)
        thresholds = {}
        for threshold in [.25, .5]:
            fixes = sum(a <= threshold < b for a, b in values)
            breaks = sum(b <= threshold < a for a, b in values)
            thresholds[str(threshold)] = {'native_hits':sum(x > threshold for x in native),
                                           'nearest_hits':sum(x > threshold for x in nearest),
                                           'fixes':fixes, 'breaks':breaks, 'net':fixes-breaks}
        return {'native_mean_iou':sum(native)/len(native), 'nearest_mean_iou':sum(nearest)/len(nearest),
                'delta_mean_iou':sum(b-a for a,b in values)/len(values), 'thresholds':thresholds}

    metrics = {}
    for branch in ['text', 'query', 'fused']:
        values = [(a['mask_branches']['selected_queries']['native_mask']['mask_iou_by_branch'][branch],
                   b['mask_branches']['selected_queries']['native_mask']['mask_iou_by_branch'][branch]) for a,b in pairs]
        metrics['native_selected_' + branch] = compare(values)
    metrics['original_matched_raw_mask'] = compare([(a['original_matched_raw_mask_iou'], b['original_matched_raw_mask_iou']) for a,b in pairs])
    metrics['full256_raw_mask_oracle'] = compare([(max(a['all_raw_query_mask_ious']), max(b['all_raw_query_mask_ious'])) for a,b in pairs])
    rows = []
    for (native, nearest), locality in zip(pairs, neighborhoods):
        rows.append({'fit_row_id':native['fit_row_id'], 'scan_id':native['scan_id'],
                     'native_matched_query':native['native_matched_query'], 'nearest_matched_query':nearest['native_matched_query'],
                     'native_selected_query':native['mask_selection']['query'],
                     'original_matched_raw_iou':[native['original_matched_raw_mask_iou'], nearest['original_matched_raw_mask_iou']],
                     'native_selected_mask_iou':[native['mask_selection']['mask_iou'], nearest['mask_selection']['mask_iou']],
                     'majority_counts':locality['counts']['majority_positive']})
    return {'rows':rows, 'counts':counts, 'metrics':metrics,
            'changed_hungarian_rows':sum(a['native_matched_query'] != b['native_matched_query'] for a,b in pairs),
            'seed_index_coordinates_equal_all':all(a['seed_index_coordinate_equal'] and b['seed_index_coordinate_equal'] for a,b in pairs),
            'native_raw_mask_max_abs_delta_from_m3':max(a['m3_native_raw_mask_max_abs_delta'] for a,b in pairs),
            'grounding_and_shared_features_exactly_equal':all(batch['grounding_and_shared_features_exactly_equal'] for batch in receipt['batches']),
            'formal_promotion':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    result = summarize(json.loads(raw))
    result['receipt_sha256'] = hashlib.sha256(raw).hexdigest()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key != 'rows'}, ensure_ascii=False))
