"""Describe the observed startup difference without changing run evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

local = Path(__file__).parent
output = local / 'analysis_initial_comparison'
output.mkdir()
original = local / 'analyze_complete.py'
content = original.read_bytes()
(output / 'failed_analyze_complete.py').write_bytes(content)
(output / 'FAILED_ATTEMPT.json').write_text(json.dumps({
    'exit_code': 1,
    'error': 'AssertionError: the two arms did not reproduce the same G starting outputs',
    'source_sha256': hashlib.sha256(content).hexdigest(),
    'output_directory_created': (local / 'analysis').exists(),
    'source_line': 132,
    'remote_model_changed': False,
}, indent=2), encoding='utf-8')
spec = importlib.util.spec_from_file_location('pair_analysis', original)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
root = local / 'complete'
control, cm = analysis.read_evaluation(root / 'g_control/initial', 'initial', 6887)
method, pm = analysis.read_evaluation(root / 'g_consistent/initial', 'initial', 6887)
summary = {'rows': 6887, 'metrics': {'control': cm, 'consistent': pm},
           'input_and_gt_exact': True, 'modes': {}, 'GPU_forwards': 0,
           'optimizer_updates': 0, 'model_source_changes': 0}
changed = []
for mode in ('bbs', 'bbf'):
    analysis.compare_rows(control, method, mode)
    fields = {key: {'control': cm[mode][key], 'consistent': pm[mode][key]}
              for key in cm[mode] if cm[mode][key] != pm[mode][key]}
    rows = [(a, b) for a, b in zip(control, method)]
    bitmaps = {metric + str(threshold): sum(
        (a[mode][metric] > threshold) != (b[mode][metric] > threshold)
        for a, b in rows)
        for metric in ('iou', 'mask_iou') for threshold in (.25, .5)}
    summary['modes'][mode] = {
        'differing_metric_fields': fields,
        'threshold_bitmap_difference_counts': bitmaps,
        'selected_query_changes': sum(a[mode]['query'] != b[mode]['query'] for a,b in rows),
        'selected_box_changes': sum(a[mode]['box'] != b[mode]['box'] for a,b in rows),
        'max_abs_selected_box_difference': max(abs(x-y) for a,b in rows
                                              for x,y in zip(a[mode]['box'], b[mode]['box'])),
        'selected_iou_changes': sum(a[mode]['iou'] != b[mode]['iou'] for a,b in rows),
        'selected_mask_iou_changes': sum(a[mode]['mask_iou'] != b[mode]['mask_iou'] for a,b in rows),
        'max_abs_mask_iou_difference': max(abs(a[mode]['mask_iou']-b[mode]['mask_iou']) for a,b in rows),
        'oracle_vector_changes': {key:sum(a[mode][key] != b[mode][key] for a,b in rows)
                                  for key in ('oracle25', 'oracle50')},
    }
    for a,b in rows:
        if a[mode] != b[mode]:
            changed.append({'row_id':a['row_id'], 'mode':mode,
                            'control':a[mode], 'consistent':b[mode]})
summary['source_row_hashes'] = {
    arm:hashlib.sha256((root / arm / 'initial/rows.jsonl').read_bytes()).hexdigest()
    for arm in ('g_control', 'g_consistent')}
(output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
(output / 'CHANGED_ROWS.jsonl').write_text(
    ''.join(json.dumps(row)+'\n' for row in changed), encoding='utf-8')
print(json.dumps(summary['modes']))
