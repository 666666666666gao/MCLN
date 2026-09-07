"""Independent CPU recount of root geometry, optimizer presence and paired REC."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def overlap(boxes, targets):
    a, b = np.asarray(boxes, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    sizes_a, sizes_b = np.maximum(a[:, 3:], 1e-6), np.maximum(b[:, 3:], 1e-6)
    extent = np.maximum(np.minimum(a[:, :3] + sizes_a / 2, b[:, :3] + sizes_b / 2)
                        - np.maximum(a[:, :3] - sizes_a / 2, b[:, :3] - sizes_b / 2), 0)
    intersection = extent.prod(1)
    return intersection / np.maximum(sizes_a.prod(1) + sizes_b.prod(1) - intersection, 1e-6)


def geometry_objective(boxes, roots):
    boxes, roots = np.asarray(boxes, dtype=np.float64), np.asarray(roots, dtype=np.float64)
    assert boxes.shape == roots.shape and boxes.shape[1] == 6
    assert np.isfinite(boxes).all() and np.isfinite(roots).all()
    assert np.all(boxes[:, 3:] >= 0.) and np.all(roots[:, 3:] >= 0.)
    size, root_size = np.maximum(boxes[:, 3:], 1e-6), np.maximum(roots[:, 3:], 1e-6)
    lo, hi = boxes[:, :3] - size / 2., boxes[:, :3] + size / 2.
    root_lo, root_hi = roots[:, :3] - root_size / 2., roots[:, :3] + root_size / 2.
    intersection = np.maximum(np.minimum(hi, root_hi) - np.maximum(lo, root_lo), 0.).prod(1)
    union = size.prod(1) + root_size.prod(1) - intersection
    outer = (np.maximum(hi, root_hi) - np.minimum(lo, root_lo)).prod(1)
    assert np.all(union > 0.) and np.all(outer > 0.)
    giou_loss = 1. - (intersection / union - (outer - union) / outer)
    l1 = np.abs(boxes[:, :3] - roots[:, :3]).sum(1) + .2 * np.abs(boxes[:, 3:] - roots[:, 3:]).sum(1)
    return l1, giou_loss


def effects(before, after, field, threshold):
    old = np.asarray([row[field] > threshold for row in before])
    new = np.asarray([row[field] > threshold for row in after])
    repair, damage = int((new & ~old).sum()), int((old & ~new).sum())
    return {'repair': repair, 'damage': damage, 'net': repair - damage}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    root = args.manifest.parent
    manifest = json.loads(args.manifest.read_text())
    receipt = json.loads((root / 'receipt.json').read_text())
    assert receipt['schema'] == 'mcln-scanrefer-mask-geometry-gt-pair-v1'
    assert receipt['status'] == 'complete' and receipt['formal_rows'] == 0
    assert receipt['manifest_sha256'] == sha(args.manifest)
    for name, digest in manifest['files'].items():
        assert sha(root / name) == digest, name
    for name, item in manifest['artifacts'].items():
        assert sha(item['path']) == item['sha256'], name
    source = Path(manifest['model_source'])
    assert sha(source / 'local_visual_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'local_visual_source_manifest.json').read_text())['files'].items():
        assert sha(source / name) == digest, name
    assert len(manifest['train_superpoint_files']) == 1201
    for name, digest in manifest['train_superpoint_files'].items():
        assert sha(Path(manifest['data_root']) / 'superpoints/train' / name) == digest, name
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_text())['row_ids']
    protocol = json.loads((root / 'protocol.json').read_text())
    assert protocol['row_ids'] == partitions
    assert not set(protocol['physical_spaces']['fit']).intersection(protocol['physical_spaces']['holdout'])
    assert len(partitions['fit']) == 29778 and len(partitions['holdout']) == 6887
    names = protocol['core_trainable_tensors']
    assert len(names) == 84 and all(n.startswith(('decoder.5.', 'prediction_heads.5.', 'x_query.', 'x_mask.', 'rel_encoder.')) for n in names)
    assert protocol['readout_trainable_tensors'] == []
    for file, key in [('fit_point_batches.json', 'fit_batches_sha256'),
                      ('baseline_rows.json', 'baseline_rows_sha256'), ('terminal_rows.json', 'terminal_rows_sha256')]:
        assert sha(root / file) == receipt[key], file
    batches = json.loads((root / 'fit_point_batches.json').read_text())
    assert len(batches) == receipt['steps_per_arm'] == manifest['steps_per_arm'] == 2482
    assert sorted(i for batch in batches for i in batch['row_ids']) == partitions['fit']
    presence_counts = {arm: dict.fromkeys(names, 0) for arm in ['native_gt', 'native_gt_mask_geometry']}
    max_geometry_errors = {'l1': 0., 'giou': 0.}
    for step, batch in enumerate(batches, 1):
        assert batch['step'] == step and len(batch['row_ids']) == (12 if step < 2482 else 6)
        count = len(batch['row_ids'])
        assert len(batch['point_sha256']) == count and all(len(h) == 64 for h in batch['point_sha256'])
        assert set(batch['arms']) == set(presence_counts)
        for arm, row in batch['arms'].items():
            stats = row['mask_geometry_stats']
            assert len(stats['query_indices']) == count and all(0 <= q < 256 for q in stats['query_indices'])
            l1, giou = geometry_objective(stats['soft_boxes'], batch['roots'])
            for field, values in [('l1', l1), ('giou', giou)]:
                error = float(np.max(np.abs(values - stats[field + '_per_row'])))
                max_geometry_errors[field] = max(error, max_geometry_errors[field])
                np.testing.assert_allclose(values, stats[field + '_per_row'], atol=2e-5, rtol=2e-6)
            auxiliary = float(np.mean(5. * l1 + giou))
            np.testing.assert_allclose(row['mask_geometry_loss'], auxiliary, atol=2e-5, rtol=2e-6)
            expected = row['native_loss'] + (row['mask_geometry_loss'] if arm == 'native_gt_mask_geometry' else 0.)
            np.testing.assert_allclose(row['loss'], expected, atol=2e-5, rtol=2e-6)
            assert np.isfinite(row['gradient_norm_before_clip'])
            bits = row['gradient_presence']
            assert len(bits) == len(names) and set(bits).issubset({'0', '1'})
            for name, bit in zip(names, bits):
                presence_counts[arm][name] += int(bit)
    initial = {k[7:]: v for k, v in torch.load(manifest['artifacts']['backbone']['path'], map_location='cpu')['model'].items()}
    checkpoint_checks = {}
    for arm, item in receipt['checkpoints'].items():
        path = Path(item['path'])
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
        state = torch.load(path, map_location='cpu')
        assert state['schema'] == 'mcln-scanrefer-mask-geometry-core-state-v1' and state['arm'] == arm
        assert state['manifest_sha256'] == sha(args.manifest) and state['steps'] == 2482
        assert state['pretrained_artifacts'] == manifest['artifacts']
        assert sorted(state['core_parameters']) == sorted(names) and state['core_trainable_tensors'] == names
        changed = []
        for name, value in state['core_parameters'].items():
            assert value.shape == initial[name].shape and value.dtype == initial[name].dtype
            assert torch.isfinite(value).all()
            if not torch.equal(value, initial[name]):
                changed.append(name)
        assert sorted(changed) == sorted(receipt['changed_core_tensors'][arm]) and changed
        optimizer = state['optimizer']
        assert len(optimizer['param_groups']) == 1
        group = optimizer['param_groups'][0]
        assert group['lr'] == 1e-6 and group['weight_decay'] == .0005
        assert len(group['params']) == len(set(group['params'])) == len(names)
        expected_ids = {index for index, name in zip(group['params'], names) if presence_counts[arm][name]}
        assert set(optimizer['state']) == expected_ids
        assert state['optimizer_parameter_steps'] == presence_counts[arm]
        for index, name in zip(group['params'], names):
            if index not in expected_ids:
                assert torch.equal(state['core_parameters'][name], initial[name]), name
                continue
            value = optimizer['state'][index]
            assert float(value['step']) == presence_counts[arm][name]
            for key in ['exp_avg', 'exp_avg_sq']:
                assert value[key].shape == initial[name].shape and torch.isfinite(value[key]).all()
        restored = dict(initial)
        restored.update(state['core_parameters'])
        assert len(restored) == 1144
        checkpoint_checks[arm] = {'changed_parameters': len(changed), 'optimizer_state_count': len(expected_ids),
                                 'optimizer_parameter_steps': presence_counts[arm],
                                 'restored_model_tensor_count': len(restored), 'bytes': path.stat().st_size}
    baseline = json.loads((root / 'baseline_rows.json').read_text())
    terminal = json.loads((root / 'terminal_rows.json').read_text())
    assert baseline['native_gt'] == baseline['native_gt_mask_geometry']
    for stage, rows_by_arm in [('baseline', baseline), ('terminal', terminal)]:
        native_metrics = json.loads((root / (stage + '_native_metrics.json')).read_text())
        for arm, rows in rows_by_arm.items():
            assert [row['row_id'] for row in rows] == partitions['holdout']
            for old, row in zip(baseline[arm], rows):
                for key in ['row_id', 'scan_id', 'point_sha256', 'physical_space']:
                    assert old[key] == row[key]
            assert all(0 <= row['matched_root_query'] < 256 for row in rows)
            soft = np.asarray([row['soft_box'] for row in rows])
            hard = np.asarray([row['hard_box'] for row in rows])
            roots = np.asarray([row['root_box'] for row in rows])
            valid = np.asarray([row['hard_valid'] for row in rows], dtype=bool)
            assert np.all(hard[~valid] == 0.)
            np.testing.assert_allclose(overlap(soft, roots), [row['soft_iou'] for row in rows], atol=2e-5, rtol=2e-6)
            np.testing.assert_allclose(overlap(hard, roots) * valid, [row['hard_iou'] for row in rows], atol=2e-5, rtol=2e-6)
            l1, giou = geometry_objective(soft, roots)
            np.testing.assert_allclose(l1, [row['soft_l1'] for row in rows], atol=2e-5, rtol=2e-6)
            np.testing.assert_allclose(giou, [row['soft_giou_loss'] for row in rows], atol=2e-5, rtol=2e-6)
            geometry_metric = json.loads((root / (stage + '_geometry_metrics.json')).read_text())[arm]
            assert geometry_metric['rows'] == len(rows) and geometry_metric['hard_valid'] == int(valid.sum())
            for kind in ['soft', 'hard']:
                assert geometry_metric[kind + '_hits'] == [sum(row[kind + '_iou'] > t for row in rows) for t in [.25, .5]]
            np.testing.assert_allclose(geometry_metric['mean_soft_geometry_loss'], np.mean(5. * l1 + giou), atol=2e-5, rtol=2e-6)
            metric = receipt[stage + '_metrics'][arm]
            for field, key in [('rec_iou', 'rec_hits'), ('mask_iou', 'mask_hits')]:
                for threshold, suffix in [(.25, '025'), (.5, '050')]:
                    assert sum(row[field] > threshold for row in rows) == metric[key + suffix]
            np.testing.assert_allclose(np.mean([row['mask_iou'] for row in rows]) * 100,
                                       metric['mask_miou'], atol=1e-10, rtol=0)
            for threshold, suffix in [(.25, '025'), (.5, '050')]:
                assert sum(row['native_rec_iou'] > threshold for row in rows) == native_metrics[arm]['rec_hits' + suffix]
    assert all(a['point_sha256'] == b['point_sha256'] and a['root_box'] == b['root_box']
               for a, b in zip(terminal['native_gt'], terminal['native_gt_mask_geometry']))
    for field, key in [('rec_iou', 'system_rec_effects'), ('native_rec_iou', 'native_rec_effects')]:
        for reference in ['baseline', 'native_gt']:
            before = baseline['native_gt_mask_geometry'] if reference == 'baseline' else terminal['native_gt']
            for threshold in [.25, .5]:
                assert effects(before, terminal['native_gt_mask_geometry'], field, threshold) == receipt[key][reference][str(threshold)]
    eligible = all(item['net'] >= 0 for comparison in receipt['system_rec_effects'].values() for item in comparison.values())
    assert eligible == receipt['eligible_for_fixed_terminal_formal_evaluation']
    result = {'schema': 'mcln-mask-geometry-gt-independent-audit-v1', 'status': 'pass',
              'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'receipt_sha256': sha(root / 'receipt.json'), 'manifest_sha256': sha(args.manifest),
              'fit_rows_verified': 29778, 'holdout_rows_verified': 6887, 'formal_rows': 0,
              'maximum_saved_geometry_errors': max_geometry_errors, 'checkpoints': checkpoint_checks,
              'eligible_for_fixed_terminal_formal_evaluation': eligible,
              'decision': 'fixed_formal_evaluation_next' if eligible else 'seal_fixed_configuration',
              'method_quality_pass_is_distinct_from_integrity_pass': True}
    with (root / 'independent_audit.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
