"""Certify Sr3D training crops using exact shared inputs and new-scan checks."""

import argparse
import ast
import csv
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
        for key in ['nr_crop_receipt', 'nr_crop_manifest', 'protocol_receipt']:
            assert file_sha(Path(manifest[key])) == manifest[key + '_sha256'], key

    verify_inputs()
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(root / 'scripts')] + list(scripts.__path__)
    from src.joint_det_dataset import Joint3DDataset
    from scripts.nr3d_object_point_appearance import box_crop_mask

    assert not torch.cuda.is_available()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    nr = json.loads(Path(manifest['nr_crop_receipt']).read_text())
    assert nr['input_contract_pass'] and nr['torch_module_predicate_matches_numpy_explicit_bounds']
    nr_manifest = json.loads(Path(manifest['nr_crop_manifest']).read_text())
    module_name = 'scripts/nr3d_object_point_appearance.py'
    assert nr_manifest['files'][module_name] == manifest['files'][module_name]
    nr_rows = {row['scan_id']: row for row in nr['rows']}
    protocol = json.loads(Path(manifest['protocol_receipt']).read_text())
    sr_protocol = protocol['annotations']['sr3d']['splits']['train']
    with (source / 'data/meta_data/sr3d_train_scans.txt').open() as stream:
        scan_ids = set(ast.literal_eval(stream.read()))
    with Path('/root/autodl-tmp/DATA_ROOT/refer_it_3d/sr3d.csv').open() as stream:
        raw_rows = [row for row in csv.DictReader(stream) if row['scan_id'] in scan_ids
                    and str(row['mentions_target_class']).lower() == 'true']
    first = {}
    for index, row in enumerate(raw_rows):
        if row['scan_id'] not in first:
            first[row['scan_id']] = index
    row_ids = sorted(first.values())
    assert len(raw_rows) == sr_protocol['effective_rows'] == 65846
    assert sorted(first) == sr_protocol['scene_ids'] and len(row_ids) == 1018

    class SceneInputs(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[index] for index in row_ids]
            super()._scene_graph_parse(annos)

    dataset = SceneInputs(dataset_dict={'sr3d': 1}, test_dataset='sr3d', split='train',
                          data_path='/root/autodl-tmp/DATA_ROOT/', use_color=True,
                          detect_intermediate=True, butd_cls=True, skip_missing_superpoints=True)
    dataset.augment = False
    assert len(dataset) == 1018
    records, violations = [], []
    total_objects = total_points = certified_scans = evaluated_scans = 0
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
        object_ids = np.flatnonzero(item['all_detected_bbox_label_mask'].astype(bool))
        point_sha = hashlib.sha256(points.tobytes()).hexdigest()
        box_sha = hashlib.sha256(boxes.tobytes()).hexdigest()
        if scan_id in nr_rows:
            prior = nr_rows[scan_id]
            assert point_sha == prior['point_cloud_sha256'] and box_sha == prior['input_boxes_sha256']
            assert object_ids.tolist() == prior['object_ids']
            counts, nonpositive = prior['crop_point_counts'], prior['nonpositive_axis_counts']
            assert min(counts) > 0 and not any(nonpositive)
            certified_scans += 1
            method = 'same_points_boxes_slots_and_predicate_as_completed_nr_audit'
        else:
            counts, nonpositive = [], []
            xyz = torch.from_numpy(points[:, :3])
            for object_id in object_ids:
                box = boxes[object_id]
                inside = box_crop_mask(xyz, torch.from_numpy(box)).numpy()
                explicit = ((points[:, :3] >= box[:3] - box[3:] * .5) &
                            (points[:, :3] <= box[:3] + box[3:] * .5)).all(axis=-1)
                assert np.array_equal(inside, explicit)
                count = int(inside.sum())
                axes = np.flatnonzero(box[3:] <= 0).tolist()
                counts.append(count)
                nonpositive.append(len(axes))
                if count == 0 or axes:
                    violations.append({'scan_id': scan_id, 'row_id': row_id, 'object_id': int(object_id),
                                       'crop_points': count, 'box': box.tolist(), 'nonpositive_axes': axes})
            evaluated_scans += 1
            method = 'module_torch_crop_checked_against_numpy_explicit_bounds'
        total_objects += len(object_ids)
        total_points += sum(counts)
        records.append({'scan_id': scan_id, 'row_id': row_id, 'object_ids': object_ids.tolist(),
                        'point_cloud_sha256': point_sha, 'input_boxes_sha256': box_sha,
                        'crop_point_counts': counts, 'nonpositive_axis_counts': nonpositive,
                        'certification_method': method})
        if (index + 1) % 128 == 0:
            print('SR OBJECT CROP AUDIT', json.dumps({'scans': index + 1, 'objects': total_objects,
                  'new_scan_evaluations': evaluated_scans, 'violating_slots': len(violations),
                  'elapsed_seconds': time.time() - started}), flush=True)
    assert certified_scans == 468 and evaluated_scans == 550
    verify_inputs()
    receipt = {'schema': 'mcln-sr3d-object-crop-inputs-v1', 'status': 'complete',
               'input_contract_pass': not violations, 'training_expressions_represented': 65846,
               'unique_scans': 1018, 'physical_spaces': 490, 'sampled_rows': 1018,
               'exact_shared_input_certifications': certified_scans, 'new_scan_crop_evaluations': evaluated_scans,
               'valid_object_slots': total_objects, 'summed_crop_points_with_box_overlaps': total_points,
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
    print('SR OBJECT CROP AUDIT COMPLETE', json.dumps({key: value for key, value in receipt.items()
          if key not in ['rows', 'violations']}), flush=True)


if __name__ == '__main__':
    main()
