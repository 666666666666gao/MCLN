import datetime
import hashlib
import json
import math
from pathlib import Path

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
root = repo / 'refine-logs/pvground_training_interface_20260908_v1'
receipt = json.loads((root / 'receipt.json').read_bytes())
evaluation = json.loads((root / 'evaluation_interface.json').read_bytes())
fixtures = json.loads((root / 'fixture_receipt.json').read_bytes())
reference = json.loads((repo / 'refine-logs/pvground_train_fixtures_20260908_v2/receipt.json').read_bytes())
assert (root / 'controller.exit').read_text().strip() == '0'
assert receipt['status'] == evaluation['status'] == fixtures['status'] == 'pass'
assert receipt['optimizer_steps'] == receipt['train_forwards'] == receipt['eval_forwards'] == 2
assert receipt['formal_rows'] == evaluation['formal_rows'] == 0
assert receipt['frozen_parameters_unchanged'] and evaluation['checkpoint_state_unchanged']
assert not receipt['metric_claim'] and not receipt['new_checkpoint_files']
assert receipt['strict_state_tensors'] == 1234
assert receipt['trainable_tensors'] == 783 and receipt['trainable_parameters'] == 27959611
assert len(evaluation['rows']) == len(fixtures['rows']) == len(reference['rows']) == 4
for actual, old in zip(fixtures['rows'], reference['rows']):
    for key in ['training_row_id', 'scan_id', 'text', 'tensor_sha256']:
        assert actual[key] == old[key], key
assert hashlib.sha256((repo / 'scripts/check_pvground_training_interface.py').read_bytes()).hexdigest() == receipt['script_sha256']
assert hashlib.sha256((root / 'fixture_receipt.json').read_bytes()).hexdigest() == receipt['fixture_receipt_sha256']
for step in receipt['steps']:
    assert step['all_present_gradients_finite']
    assert all(math.isfinite(value) for value in step['losses'].values())
    assert len(step['missing_gradients']) == 24
    for name in ['backbone_net', 'cross_encoder', 'gumbel', 'decoder', 'prediction_heads', 'x_mask', 'x_query']:
        assert step['gradient_groups'][name]['nonzero_tensors'] > 0, name
assert receipt['steps'][0]['missing_gradients'] == receipt['steps'][1]['missing_gradients']
comparisons = 0
for mode in ['bbs', 'bbf']:
    for threshold in [.25, .5]:
        for count in [1, 5, 10]:
            key = str(('last_', threshold, count, mode))
            hits = sum(any(iou > threshold for iou in row[mode]['top10_ious'][:count]) for row in evaluation['rows'])
            assert hits == evaluation['evaluator_dets'][key]
            assert evaluation['evaluator_gts'][key] == 4
            comparisons += 1
    key = 'mask_pos' if mode == 'bbs' else 'mask_sem'
    assert abs(sum(row[mode]['mask_iou'] for row in evaluation['rows']) - evaluation['evaluator_dets'][key]) < 1e-6
    comparisons += 1
result = {'status': 'pass', 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'interface_comparisons': comparisons, 'same_input_training_rows': 4, 'optimizer_steps': 2,
          'formal_rows': 0, 'metric_claim': False, 'trainable_tensors': 783, 'gradient_tensors_per_step': 759,
          'missing_gradient_tensors_per_step': 24, 'changed_parameter_tensors': len(receipt['changed_parameters']),
          'changed_buffer_tensors': len(receipt['changed_buffers']),
          'receipt_sha256': hashlib.sha256((root / 'receipt.json').read_bytes()).hexdigest()}
(root / 'local_verification.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
(root / 'verify_local.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result), flush=True)
