"""Independent box/IoU recount and shared-input check for the isolated prototype."""
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
    earlier = json.loads((args.gradient_directory / 'receipt.json').read_bytes())
    identity = {}
    for batch in earlier['observations']:
        for index, row_id in enumerate(batch['row_ids']):
            identity[row_id] = {'point_sha256': batch['point_sha256'][index],
                'query_indices': batch['query_indices'][index],
                'hard_boxes': np.asarray(batch['boxes'][index])[:, 2],
                'valid': np.asarray(batch['valid'][index])[:, 2]}
    assert len(result['rows']) == len(identity) == 16
    arrays = {'hard': [], 'soft': [], 'valid': [], 'mass': []}
    maximum_error = 0.
    for row in result['rows']:
        reference = identity[row['row_id']]
        assert row['point_sha256'] == reference['point_sha256']
        assert row['query_indices'] == reference['query_indices']
        assert np.array_equal(row['hard_boxes'], reference['hard_boxes'])
        assert np.array_equal(row['valid'], reference['valid'])
        root = np.asarray(row['root_box'], dtype=np.float64)
        for source in ['hard', 'soft']:
            boxes = np.asarray(row[source + '_boxes'], dtype=np.float64)
            lo, hi = boxes[:, :3] - boxes[:, 3:] / 2., boxes[:, :3] + boxes[:, 3:] / 2.
            gtlo, gthi = root[:3] - root[3:] / 2., root[:3] + root[3:] / 2.
            intersection = np.maximum(np.minimum(hi, gthi) - np.maximum(lo, gtlo), 0.).prod(-1)
            union = boxes[:, 3:].prod(-1) + root[3:].prod() - intersection
            ious = intersection / np.maximum(union, 1e-8)
            error = np.max(np.abs(ious - row[source + '_ious']))
            maximum_error = max(maximum_error, float(error))
            assert np.allclose(ious, row[source + '_ious'], atol=2e-6, rtol=0.)
            arrays[source].append(ious)
        arrays['valid'].append(row['valid'])
        arrays['mass'].append(row['threshold_excluded_probability_mass'])
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    valid = arrays['valid'].astype(bool)
    stats = result['summary']
    assert stats['valid_candidate_pairs'] == int(valid.sum())
    for source in ['hard', 'soft']:
        ious = arrays[source]
        assert stats[source + '_candidate_hits'] == [int(((ious > threshold) & valid).sum()) for threshold in [.25, .5]]
        assert stats[source + '_row_oracle_hits'] == [int(((ious > threshold) & valid).any(1).sum()) for threshold in [.25, .5]]
        assert np.isclose(stats[source + '_mean_iou'], ious[valid].mean(), atol=2e-6, rtol=0.)
    mass = arrays['mass'][valid]
    assert np.isclose(stats['threshold_excluded_probability_mass_mean'], mass.mean())
    assert np.isclose(stats['threshold_excluded_probability_mass_median'], np.median(mass))
    assert stats['pairs_with_excluded_mass_above_one_percent'] == int((mass > .01).sum())
    assert all(row['original_query_mask_gradient_norm'] > 0 and row['native_parameter_gradient_norm'] > 0 for row in result['gradient_records'])
    assert result['optimizer_steps'] == result['formal_rows'] == result['checkpoint_writes'] == 0
    record = {'status': 'pass', 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'receipt_sha256': hashlib.sha256(raw).hexdigest(), 'rows': 16, 'same_inputs_and_hard_boxes_as_gradient_probe': True,
        'max_independent_iou_error': maximum_error, 'summary': stats,
        'scope': 'Independent saved-coordinate recount and cross-run identity; not a second native autograd execution.'}
    with (args.directory / 'independent_recount.json').open('x') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
