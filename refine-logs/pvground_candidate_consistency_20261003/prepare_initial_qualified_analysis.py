"""Preserve the failed exact check and prepare an explicitly qualified comparison."""
import hashlib
import json
from pathlib import Path

local = Path(__file__).parent
source = local / 'analyze_complete.py'
diagnostic = local / 'analysis_initial_comparison/SUMMARY.json'
startup = json.loads(diagnostic.read_bytes())
assert startup['input_and_gt_exact']
for mode in ('bbs', 'bbf'):
    assert all(value == 0 for value in startup['modes'][mode]['threshold_bitmap_difference_counts'].values())
content = source.read_text(encoding='utf-8')
old = "            assert cm == pm, 'the two arms did not reproduce the same G starting outputs'"
new = """            # The original exact-summary check failed; preserve and disclose it.
            for mode in ('bbs', 'bbf'):
                for metric in ('iou', 'mask_iou'):
                    for threshold in (.25, .5):
                        assert all((a[mode][metric] > threshold) == (b[mode][metric] > threshold)
                                   for a, b in zip(control, method)), (mode, metric, threshold)
            initial_diagnostic = args.root.parent/'analysis_initial_comparison/SUMMARY.json'
            startup_comparison = read_json(initial_diagnostic)
            startup_comparison['diagnostic_sha256'] = hashlib.sha256(initial_diagnostic.read_bytes()).hexdigest()"""
assert content.count(old) == 1
content = content.replace(old, new)
old = "        'same_fit_batch_order': True, 'training_qualification': qualification,"
new = old + "\n        'starting_output_comparison': startup_comparison,\n        'bitwise_paired_comparison': False,"
assert content.count(old) == 1
content = content.replace(old, new)
old = "        'interpretation_limits':["
new = old + "\n            'The original exact starting-summary assertion failed and is archived. Inputs/GT and initial REC/Mask threshold bitmaps match; selected boxes, some queries, continuous Mask IoU and Top-k oracle vectors differ across processes. This is a qualified same-budget empirical comparison, not bitwise-paired evidence.',"
assert content.count(old) == 1
content = content.replace(old, new)
output = local / 'analyze_complete_initial_qualified.py'
with output.open('x', encoding='utf-8', newline='\n') as handle:
    handle.write(content)
print(json.dumps({'analysis_source': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                  'original_analysis_preserved_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'diagnostic_sha256': hashlib.sha256(diagnostic.read_bytes()).hexdigest(),
                  'model_or_loss_change': False}))
