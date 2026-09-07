"""Independently recompute saved geometry/IoU/loss values and trace assertions."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    options = parser.parse_args()
    directory = options.directory
    receipt_bytes = (directory / 'receipt.json').read_bytes()
    receipt = json.loads(receipt_bytes)
    manifest_bytes = (directory / 'input_manifest.json').read_bytes()
    assert receipt['input_manifest_sha256'] == hashlib.sha256(manifest_bytes).hexdigest()
    assert receipt['status'] == 'pass' and receipt['rows'] == 16
    assert receipt['optimizer_steps'] == receipt['checkpoint_writes'] == receipt['formal_rows'] == 0
    identities = []
    valid_counts = np.zeros(7, dtype=np.int64)
    max_iou_error = 0.
    max_loss_error = 0.
    for row in receipt['observations']:
        identities.extend(row['row_ids'])
        boxes = np.asarray(row['boxes'], dtype=np.float64)
        valid = np.asarray(row['valid'], dtype=bool)
        roots = np.asarray(row['root_boxes'], dtype=np.float64)[:, :, None]
        assert boxes.shape == (4, 16, 7, 6) and valid.shape == (4, 16, 7)
        lower = boxes[..., :3] - boxes[..., 3:] / 2.
        upper = boxes[..., :3] + boxes[..., 3:] / 2.
        gt_lower = roots[..., :3] - roots[..., 3:] / 2.
        gt_upper = roots[..., :3] + roots[..., 3:] / 2.
        intersection = np.maximum(np.minimum(upper, gt_upper) - np.maximum(lower, gt_lower), 0.).prod(-1)
        union = boxes[..., 3:].prod(-1) + roots[..., 3:].prod(-1) - intersection
        iou = intersection / np.maximum(union, 1e-8)
        error = np.abs(iou - np.asarray(row['root_ious']))
        max_iou_error = max(max_iou_error, float(error.max()))
        assert np.allclose(iou, row['root_ious'], atol=2e-6, rtol=0.)
        counts = valid.sum(axis=(0, 1))
        assert counts.tolist() == row['valid_per_variant']
        valid_counts += counts
        for index, name in enumerate(receipt['variant_names']):
            loss = np.abs(boxes[:, :, index] - roots[:, :, 0]).mean(-1)[valid[:, :, index]].mean()
            stats = row['loss_gradients']['coordinate_l1/' + name]
            max_loss_error = max(max_loss_error, abs(loss - stats['loss']))
            assert np.isclose(loss, stats['loss'], atol=2e-6, rtol=0.)
            for group in ['query_mask_outputs', 'text_mask_outputs']:
                assert stats['groups'][group]['connected_tensors'] == 0
                assert stats['groups'][group]['nonzero_tensors'] == 0
                assert stats['groups'][group]['norm'] == 0.
            if 1 <= index <= 4:
                assert stats['groups']['core_parameters']['norm'] == 0.
            else:
                assert stats['groups']['native_center_output']['norm'] > 0.
                assert stats['groups']['native_size_output']['norm'] > 0.
        for name in ['existing_readout_gt_loss', 'native_gt_loss']:
            stats = row['loss_gradients'][name]['groups']
            for group in ['query_mask_outputs', 'text_mask_outputs']:
                assert stats[group]['connected_tensors'] == 4 and stats[group]['nonzero_tensors'] == 4
                assert stats[group]['norm'] > 0.
    assert len(identities) == len(set(identities)) == 16
    result = {'status': 'pass', 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest(), 'rows': 16,
        'valid_per_variant': valid_counts.tolist(), 'max_independent_iou_error': max_iou_error,
        'max_independent_l1_error': max_loss_error,
        'scope': 'Independent NumPy recomputation and saved trace consistency; not an independent rerun of native autograd.',
        'optimizer_steps': 0, 'checkpoint_writes': 0, 'formal_rows': 0}
    with (directory / 'independent_recount.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
