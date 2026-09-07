"""CPU verification of saved native box endpoints, GT targets and paired metrics."""
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
    assert receipt['schema'] == 'mcln-scanrefer-native-box-transfer-pair-v1'
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
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_text())['row_ids']
    protocol = json.loads((root / 'protocol.json').read_text())
    assert protocol['row_ids'] == partitions
    assert not set(protocol['physical_spaces']['fit']).intersection(protocol['physical_spaces']['holdout'])
    assert len(partitions['fit']) == 29778 and len(partitions['holdout']) == 6887
    names = protocol['core_trainable_tensors']
    assert len(names) == 16 and all(n.startswith(('prediction_heads.5.center_residual_head.',
                                                'prediction_heads.5.size_pred_head.')) for n in names)
    assert protocol['readout_trainable_tensors'] == protocol['teacher_trainable_tensors'] == []
    for file, key in [('fit_point_batches.json', 'fit_batches_sha256'),
                      ('baseline_rows.json', 'baseline_rows_sha256'), ('terminal_rows.json', 'terminal_rows_sha256')]:
        assert sha(root / file) == receipt[key], file
    batches = json.loads((root / 'fit_point_batches.json').read_text())
    assert len(batches) == receipt['steps_per_arm'] == manifest['steps_per_arm'] == 2482
    assert sorted(i for batch in batches for i in batch['row_ids']) == partitions['fit']
    eligible_counts = {'gt_only': 0, 'gt_teacher_box': 0}
    for step, batch in enumerate(batches, 1):
        assert batch['step'] == step and len(batch['row_ids']) == (12 if step < 2482 else 6)
        count = len(batch['row_ids'])
        assert len(batch['point_sha256']) == count and all(len(h) == 64 for h in batch['point_sha256'])
        teacher_ious = overlap(batch['teacher_boxes'], batch['roots'])
        assert len(batch['teacher_selected']) == count and all(0 <= i < 112 for i in batch['teacher_selected'])
        for arm, row in batch['arms'].items():
            stats = row['teacher_box_stats']
            np.testing.assert_allclose(stats['teacher_root_ious'], teacher_ious, atol=2e-5, rtol=0)
            gain = np.maximum(np.asarray(stats['teacher_root_ious']) - stats['student_root_ious'], 0)
            gain *= np.asarray(stats['teacher_root_ious']) > .25
            np.testing.assert_allclose(stats['gain_weights'], gain, atol=1e-7, rtol=0)
            assert list(gain > 0) == stats['eligible']
            assert len(stats['student_query_indices']) == count and all(0 <= q < 256 for q in stats['student_query_indices'])
            eligible_counts[arm] += int((gain > 0).sum())
            per_row = 5 * np.asarray(stats['l1_per_row']) + stats['giou_per_row']
            auxiliary = float((gain * per_row).sum() / gain.sum()) if gain.sum() else 0.
            np.testing.assert_allclose(row['teacher_box_loss'], auxiliary, atol=2e-5, rtol=2e-6)
            expected = row['native_loss'] + (row['teacher_box_loss'] if arm == 'gt_teacher_box' else 0.)
            np.testing.assert_allclose(row['loss'], expected, atol=2e-5, rtol=2e-6)
            assert np.isfinite(row['gradient_norm_before_clip'])
    initial = {k[7:]: v for k, v in torch.load(manifest['artifacts']['backbone']['path'], map_location='cpu')['model'].items()}
    checkpoint_checks = {}
    for arm, item in receipt['checkpoints'].items():
        path = Path(item['path'])
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
        state = torch.load(path, map_location='cpu')
        assert state['schema'] == 'mcln-scanrefer-native-box-head-state-v1' and state['arm'] == arm
        assert state['manifest_sha256'] == sha(args.manifest) and state['steps'] == 2482
        assert state['pretrained_artifacts'] == manifest['artifacts']
        assert sorted(state['head_parameters']) == sorted(names) and state['core_trainable_tensors'] == names
        changed = []
        for name, value in state['head_parameters'].items():
            assert value.shape == initial[name].shape and value.dtype == initial[name].dtype
            assert torch.isfinite(value).all()
            if not torch.equal(value, initial[name]):
                changed.append(name)
        assert sorted(changed) == sorted(receipt['changed_core_tensors'][arm]) and changed
        optimizer = state['optimizer']
        assert len(optimizer['state']) == 16 and len(optimizer['param_groups']) == 1
        group = optimizer['param_groups'][0]
        assert group['lr'] == 1e-6 and group['weight_decay'] == .0005
        assert len(group['params']) == 16 and set(group['params']) == set(optimizer['state'])
        for index, name in zip(group['params'], names):
            value = optimizer['state'][index]
            assert float(value['step']) == 2482
            for key in ['exp_avg', 'exp_avg_sq']:
                assert value[key].shape == initial[name].shape and torch.isfinite(value[key]).all()
        checkpoint_checks[arm] = {'changed_parameters': len(changed), 'optimizer_steps': 2482,
                                 'restored_model_tensor_count': len(initial), 'bytes': path.stat().st_size}
    baseline = json.loads((root / 'baseline_rows.json').read_text())
    terminal = json.loads((root / 'terminal_rows.json').read_text())
    assert baseline['gt_only'] == baseline['gt_teacher_box']
    for stage, rows_by_arm in [('baseline', baseline), ('terminal', terminal)]:
        native_metrics = json.loads((root / (stage + '_native_metrics.json')).read_text())
        for arm, rows in rows_by_arm.items():
            assert [row['row_id'] for row in rows] == partitions['holdout']
            for old, row in zip(baseline[arm], rows):
                for key in ['row_id', 'scan_id', 'point_sha256', 'physical_space']:
                    assert old[key] == row[key]
            metric = receipt[stage + '_metrics'][arm]
            for field, key in [('rec_iou', 'rec_hits'), ('mask_iou', 'mask_hits')]:
                for threshold, suffix in [(.25, '025'), (.5, '050')]:
                    assert sum(row[field] > threshold for row in rows) == metric[key + suffix]
            np.testing.assert_allclose(np.mean([row['mask_iou'] for row in rows]) * 100,
                                       metric['mask_miou'], atol=1e-10, rtol=0)
            for threshold, suffix in [(.25, '025'), (.5, '050')]:
                assert sum(row['native_rec_iou'] > threshold for row in rows) == native_metrics[arm]['rec_hits' + suffix]
    for field, key in [('rec_iou', 'system_rec_effects'), ('native_rec_iou', 'native_rec_effects')]:
        for reference in ['baseline', 'gt_only']:
            before = baseline['gt_teacher_box'] if reference == 'baseline' else terminal['gt_only']
            for threshold in [.25, .5]:
                assert effects(before, terminal['gt_teacher_box'], field, threshold) == receipt[key][reference][str(threshold)]
    eligible = all(item['net'] >= 0 for comparison in receipt['system_rec_effects'].values() for item in comparison.values())
    assert eligible == receipt['eligible_for_fixed_terminal_formal_evaluation']
    result = {'schema': 'mcln-native-box-transfer-independent-audit-v1', 'status': 'pass',
              'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'receipt_sha256': sha(root / 'receipt.json'), 'manifest_sha256': sha(args.manifest),
              'fit_rows_verified': 29778, 'holdout_rows_verified': 6887, 'formal_rows': 0,
              'teacher_eligible_rows': eligible_counts, 'checkpoints': checkpoint_checks,
              'eligible_for_fixed_terminal_formal_evaluation': eligible,
              'decision': 'fixed_formal_evaluation_next' if eligible else 'seal_fixed_configuration',
              'method_quality_pass_is_distinct_from_integrity_pass': True}
    with (root / 'independent_audit.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
