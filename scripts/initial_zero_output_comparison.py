"""Compare actual zero-output starts without claiming float-list identity."""
def compare_initial_rows(actual, reference):
    assert len(actual) == len(reference) == 6887
    changes = {mode:dict(query_row_ids=[],box_row_ids=[],iou_row_ids=[],mask_row_ids=[],max_box_delta=0.)
               for mode in ('bbs','bbf')}
    for a,r in zip(actual,reference):
        for key in ('row_id','scan_id','target_id','root_box','point_sha256'):
            assert a[key] == r[key],key
        for mode in ('bbs','bbf'):
            assert a[mode]['box'] == a[mode]['coarse_box'],(mode,'zero_box')
            assert a[mode]['iou'] == a[mode]['coarse_iou'],(mode,'zero_iou')
            for threshold in (.25,.5):
                assert (a[mode]['iou']>threshold) == (r[mode]['iou']>threshold),(a['row_id'],mode,threshold)
            for key,label in (('query','query'),('box','box'),('iou','iou'),('mask_iou','mask')):
                if a[mode][key] != r[mode][key]:
                    changes[mode][label+'_row_ids'].append(a['row_id'])
            changes[mode]['max_box_delta']=max(changes[mode]['max_box_delta'],
                max(abs(x-y) for x,y in zip(a[mode]['box'],r[mode]['box'])))
    exact=all(not changes[mode][key+'_row_ids'] for mode in changes for key in ('query','box','iou'))
    return dict(rec_inputs_exact=exact,input_identities_exact=True,rec_threshold_decisions_exact=True,
        within_forward_selected_zero_refinement_exact=True,cross_process_rec_output_exact=exact,
        cross_process_differences=changes,
        mask_difference_row_ids={mode:changes[mode]['mask_row_ids'] for mode in changes})
