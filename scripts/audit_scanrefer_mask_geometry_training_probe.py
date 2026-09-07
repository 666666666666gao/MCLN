"""Independent numeric and scope checks for the disposable geometry updates."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--gradient-directory', type=Path, required=True)
    args = parser.parse_args()
    raw = (args.directory / 'receipt.json').read_bytes()
    result = json.loads(raw)
    reference = json.loads((args.gradient_directory / 'receipt.json').read_bytes())
    assert result['status'] == 'pass' and result['rows'] == 16
    assert result['disposable_optimizer_steps_per_arm'] == 2
    assert result['checkpoint_writes'] == result['formal_rows'] == 0
    assert len(result['requires_grad_parameters']) == 84 and result['full_state_tensors'] == 1144
    max_l1_error, max_giou_error = 0., 0.
    for row, old in zip(result['observations'], reference['observations']):
        assert row['row_ids'] == old['row_ids'] and row['point_sha256'] == old['point_sha256']
        stats = row['target_stats']
        assert len(stats['query_indices']) == 4 and all(0 <= index < 256 for index in stats['query_indices'])
        boxes = np.asarray(stats['soft_boxes'], dtype=np.float64)
        roots = np.asarray(old['root_boxes'], dtype=np.float64)[:, 0]
        size, root_size = np.maximum(boxes[:, 3:], 1e-6), np.maximum(roots[:, 3:], 1e-6)
        lo, hi = boxes[:, :3] - size / 2., boxes[:, :3] + size / 2.
        root_lo, root_hi = roots[:, :3] - root_size / 2., roots[:, :3] + root_size / 2.
        intersection = np.maximum(np.minimum(hi, root_hi) - np.maximum(lo, root_lo), 0.).prod(1)
        union = size.prod(1) + root_size.prod(1) - intersection
        outer = (np.maximum(hi, root_hi) - np.minimum(lo, root_lo)).prod(1)
        giou_loss = 1. - (intersection / np.maximum(union, 1e-6) - (outer - union) / np.maximum(outer, 1e-6))
        l1 = np.abs(boxes[:, :3] - roots[:, :3]).sum(1) + .2 * np.abs(boxes[:, 3:] - roots[:, 3:]).sum(1)
        max_l1_error = max(max_l1_error, float(np.max(np.abs(l1 - stats['l1_per_row']))))
        max_giou_error = max(max_giou_error, float(np.max(np.abs(giou_loss - stats['giou_per_row']))))
        assert np.allclose(l1, stats['l1_per_row'], atol=2e-5, rtol=0.)
        assert np.allclose(giou_loss, stats['giou_per_row'], atol=2e-5, rtol=0.)
        assert np.isclose(np.mean(5. * l1 + giou_loss), row['mask_geometry_loss'], atol=2e-5, rtol=0.)
        cosine = row['gradient_dot'] / (row['native_gradient_norm'] * row['geometry_gradient_norm'])
        assert np.isclose(cosine, row['gradient_cosine']) and abs(cosine) <= 1.
    for arm, update in result['updates'].items():
        assert update['frozen_state_unchanged'] and len(update['changed_tensors']) == 82
        assert set(update['changed_tensors']).issubset(result['requires_grad_parameters'])
        assert [step['step'] for step in update['steps']] == [1, 2]
        assert update['steps'][0]['native_loss'] == result['observations'][0]['native_loss']
        assert update['steps'][0]['mask_geometry_loss'] == result['observations'][0]['mask_geometry_loss']
        assert all(np.isfinite(step['gradient_norm']) for step in update['steps'])
    record = {'status': 'pass', 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'receipt_sha256': hashlib.sha256(raw).hexdigest(), 'rows': 16,
        'max_l1_error': max_l1_error, 'max_giou_error': max_giou_error,
        'same_input_points_as_zero_update_probe': True, 'scope': 'Independent saved-coordinate loss recount; no quality acceptance.'}
    with (args.directory / 'independent_recount.json').open('x') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
