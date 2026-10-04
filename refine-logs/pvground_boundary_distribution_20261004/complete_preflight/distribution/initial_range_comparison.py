"""Record actual paired starts, including observed floating point/selection changes."""
def compare_initial_rows(actual, reference):
    assert len(actual) == len(reference) == 6887
    differences = {mode: dict(query_rows=[], box_rows=[], iou_rows=[], mask_rows=[],
        threshold25_rows=[], threshold50_rows=[], max_box_delta=0.) for mode in ('bbs', 'bbf')}
    for a, b in zip(actual, reference):
        for key in ('row_id', 'scan_id', 'target_id', 'root_box', 'point_sha256'):
            assert a[key] == b[key], key
        for mode, record in differences.items():
            assert a[mode]['box'] == a[mode]['coarse_box']
            assert a[mode]['iou'] == a[mode]['coarse_iou']
            for key, label in (('query', 'query'), ('box', 'box'), ('iou', 'iou'), ('mask_iou', 'mask')):
                if a[mode][key] != b[mode][key]:
                    record[label+'_rows'].append(a['row_id'])
            for threshold, label in ((.25, 'threshold25'), (.5, 'threshold50')):
                if (a[mode]['iou'] > threshold) != (b[mode]['iou'] > threshold):
                    record[label+'_rows'].append(a['row_id'])
            record['max_box_delta'] = max(record['max_box_delta'],
                max(abs(x-y) for x, y in zip(a[mode]['box'], b[mode]['box'])))
    return dict(input_identities_exact=True, zero_head_within_forward_exact=True,
        cross_process_output_exact=all(not v[key+'_rows'] for v in differences.values()
            for key in ('query', 'box', 'iou', 'mask')),
        rec_threshold_decisions_exact=all(not v[key+'_rows'] for v in differences.values()
            for key in ('threshold25', 'threshold50')), differences=differences)
