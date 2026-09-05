"""Audit the raw-box appearance contract once per unaugmented training scene."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    root = options.manifest.parent
    manifest = json.loads(options.manifest.read_text())
    source = Path(manifest['model_source'])

    def verify_inputs():
        assert file_sha(source / 'g0_source_manifest.json') == manifest['source_manifest_sha256']
        for name, digest in json.loads((source / 'g0_source_manifest.json').read_text())['files'].items():
            assert file_sha(source / name) == digest, name
        for name, digest in manifest['files'].items():
            assert file_sha(root / name) == digest, name
        for name, metadata in manifest['data_files'].items():
            assert file_sha(Path(name)) == metadata['sha256'], name
        assert file_sha(Path(manifest['prior_receipt'])) == manifest['prior_receipt_sha256']

    verify_inputs()
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(root / 'scripts')] + list(scripts.__path__)
    from src.joint_det_dataset import Joint3DDataset
    from scripts.nr3d_object_point_appearance import box_crop_mask
    from scripts.run_nr3d_view_pair_role import read_train_rows

    assert not torch.cuda.is_available()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    raw_rows = read_train_rows(Path('/root/autodl-tmp/DATA_ROOT'))
    first = {}
    for index, row in enumerate(raw_rows):
        if row['scan_id'] not in first:
            first[row['scan_id']] = index
    row_ids = sorted(first.values())
    assert len(raw_rows) == 32919 and len(row_ids) == 511
    prior = json.loads(Path(manifest['prior_receipt']).read_text())
    prior_rows = {row['scan_id']: row for row in prior['rows']}
    assert set(first) == set(prior_rows)

    class SceneInputs(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[index] for index in row_ids]
            super()._scene_graph_parse(annos)

    dataset = SceneInputs(dataset_dict={'nr3d': 1}, test_dataset='nr3d', split='train',
                          data_path='/root/autodl-tmp/DATA_ROOT/', use_color=True,
                          detect_intermediate=True, butd_cls=True, skip_missing_superpoints=True)
    dataset.augment = False
    assert len(dataset) == 511
    records, violations = [], []
    total_objects, total_crop_points = 0, 0
    added_points, removed_points, changed_slots = 0, 0, 0
    started = time.time()
    for index, row_id in enumerate(row_ids):
        item = dataset[index]
        scan_id = item['scan_ids']
        assert scan_id == raw_rows[row_id]['scan_id']
        points = item['point_clouds']
        scan = dataset.scans[scan_id]
        assert points.shape == (50000, 6)
        assert np.array_equal(points[:, :3], scan.orig_pc.astype(np.float32))
        assert np.array_equal(points[:, 3:6], (scan.color - dataset.mean_rgb).astype(np.float32))
        boxes = item['all_detected_boxes']
        valid = item['all_detected_bbox_label_mask'].astype(bool)
        assert hashlib.sha256(points.tobytes()).hexdigest() == prior_rows[scan_id]['point_cloud_sha256']
        assert hashlib.sha256(boxes.tobytes()).hexdigest() == prior_rows[scan_id]['input_boxes_sha256']
        counts, nonpositive, object_ids = [], [], np.flatnonzero(valid)
        assert object_ids.tolist() == prior_rows[scan_id]['object_ids']
        xyz = torch.from_numpy(points[:, :3])
        for slot, object_id in enumerate(object_ids):
            box = boxes[object_id]
            legacy = (np.abs(points[:, :3] - box[:3]) <= box[3:] * .5).all(axis=-1)
            assert int(legacy.sum()) == prior_rows[scan_id]['crop_point_counts'][slot]
            inside = box_crop_mask(xyz, torch.from_numpy(box)).numpy()
            explicit = ((points[:, :3] >= box[:3] - box[3:] * .5) &
                        (points[:, :3] <= box[:3] + box[3:] * .5)).all(axis=-1)
            assert np.array_equal(inside, explicit)
            added = int((inside & ~legacy).sum())
            removed = int((legacy & ~inside).sum())
            added_points += added
            removed_points += removed
            changed_slots += int(added + removed > 0)
            count = int(inside.sum())
            bad_axes = np.flatnonzero(box[3:] <= 0).tolist()
            counts.append(count)
            nonpositive.append(len(bad_axes))
            if count == 0 or bad_axes:
                violations.append({'scan_id': scan_id, 'fit_row_id': row_id, 'object_id': int(object_id),
                                   'crop_points': count, 'box': box.tolist(), 'nonpositive_axes': bad_axes})
        total_objects += len(object_ids)
        total_crop_points += sum(counts)
        records.append({'scan_id': scan_id, 'row_id': row_id, 'object_ids': object_ids.tolist(),
                        'point_cloud_sha256': hashlib.sha256(points.tobytes()).hexdigest(),
                        'input_boxes_sha256': hashlib.sha256(boxes.tobytes()).hexdigest(),
                        'crop_point_counts': counts, 'nonpositive_axis_counts': nonpositive})
        if (index + 1) % 64 == 0:
            print('OBJECT CROP AUDIT', json.dumps({'scenes': index + 1, 'objects': total_objects,
                  'violating_slots': len(violations), 'elapsed_seconds': time.time() - started}), flush=True)
    verify_inputs()
    receipt = {'schema': 'mcln-all-nr3d-object-crop-inputs-v2', 'status': 'complete',
               'input_contract_pass': not violations, 'training_expressions_represented': 32919,
               'unique_scenes': 511, 'sampled_rows': 511, 'valid_object_slots': total_objects,
               'summed_crop_points_with_box_overlaps': total_crop_points,
               'added_point_memberships_vs_absolute': added_points,
               'removed_point_memberships_vs_absolute': removed_points,
               'changed_object_slots_vs_absolute': changed_slots,
               'same_input_points_boxes_object_slots_and_legacy_counts': True,
               'torch_module_predicate_matches_numpy_explicit_bounds': True,
               'prior_receipt_sha256': manifest['prior_receipt_sha256'],
               'empty_crop_slots': sum(row['crop_points'] == 0 for row in violations),
               'slots_with_nonpositive_axes': sum(bool(row['nonpositive_axes']) for row in violations),
               'violations': violations, 'rows': records, 'augmentation': False,
               'input_points_equal_serialized_scene_xyz_and_native_rgb': True,
               'instance_memberships_used_for_cropping': False, 'formal_validation_rows': 0,
               'native_model_forwards': 0, 'gpu_forwards': 0, 'optimizer_steps': 0,
               'source_and_data_unchanged': True, 'manifest_sha256': file_sha(options.manifest),
               'elapsed_seconds_excluding_dataset_init': time.time() - started}
    with (root / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('OBJECT CROP AUDIT COMPLETE', json.dumps({key: value for key, value in receipt.items()
          if key not in ['rows', 'violations']}), flush=True)


if __name__ == '__main__':
    main()
