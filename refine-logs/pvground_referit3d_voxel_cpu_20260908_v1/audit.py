"""CPU audit of fixed real ReferIt3D samples through the author voxel processor."""
import argparse
import copy
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True, type=Path)
    args = parser.parse_args()
    started = time.time()
    spec = json.loads(args.spec.read_bytes())
    root = args.spec.parent
    runtime = Path(spec['runtime'])
    selection_root = Path(spec['selection_root'])
    input_manifest = selection_root / 'manifest.json'
    assert sha(input_manifest) == spec['selection_manifest_sha256']
    manifest = json.loads(input_manifest.read_bytes())
    native = Path(manifest['model_source'])
    source_manifest = native / 'appearance_source_manifest.json'
    assert sha(source_manifest) == manifest['source_manifest_sha256']
    for name, digest in json.loads(source_manifest.read_bytes())['files'].items():
        assert sha(native / name) == digest, name
    verified = {}
    for name in ['fixed_selection.py', 'preflight_rows.json', 'annotation_receipt.json']:
        path = selection_root / name
        expected = manifest['input_files'][str(path)]
        assert sha(path) == expected, name
        verified[str(path)] = expected
    point_archive = Path(manifest['data_root']) / 'train_v3scans.pkl'
    assert sha(point_archive) == manifest['input_files'][str(point_archive.resolve())]
    verified[str(point_archive.resolve())] = sha(point_archive)
    annotation = json.loads((selection_root / 'annotation_receipt.json').read_bytes())
    for path, info in annotation['annotations_and_split_files'].items():
        current = Path(path) if '/DATA_ROOT/' in path else native / 'data/meta_data' / Path(path).name
        assert sha(current) == info['sha256'], str(current)
        verified[str(current)] = info['sha256']
    for name, digest in manifest['superpoints'].items():
        path = Path(manifest['data_root']) / 'superpoints/train' / name
        assert sha(path) == digest, name
        verified[str(path)] = digest
    bundle = json.loads((runtime / 'source_bundle_receipt.json').read_bytes())['sources']['PV-Ground']['files']
    for name in ['prepare_data.py', 'wandb_config.yaml']:
        path = runtime / 'PV-Ground' / name
        assert sha(path) == bundle[name]['sha256'], name
        verified[str(path)] = sha(path)

    import numpy as np
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    os.chdir(str(native))
    sys.path[:0] = [str(native), str(selection_root)]
    from src.joint_det_dataset import Joint3DDataset
    from fixed_selection import build_probe_dataset
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve() == native / 'src/joint_det_dataset.py'
    sys.path.insert(0, str(runtime / 'PV-Ground'))
    from prepare_data import DataProcessor
    from pcdet.config import cfg, cfg_from_yaml_file
    assert Path(sys.modules['prepare_data'].__file__).resolve() == runtime / 'PV-Ground/prepare_data.py'
    cfg_from_yaml_file(str(runtime / 'PV-Ground/wandb_config.yaml'), cfg)
    assert [item.NAME for item in cfg.DATA_PROCESSOR] == ['transform_points_to_voxels']
    voxel_cfg = cfg.DATA_PROCESSOR[0]
    assert voxel_cfg.MAX_POINTS_PER_VOXEL == 5
    assert voxel_cfg.MAX_NUMBER_OF_VOXELS == {'train': 50000, 'test': 50000}
    minimum = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE[:3], dtype=np.float32)
    maximum = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE[3:], dtype=np.float32)
    size = np.asarray(voxel_cfg.VOXEL_SIZE, dtype=np.float32)
    grid = np.rint((maximum - minimum) / size).astype(np.int32)
    processors = {mode: DataProcessor(cfg.DATA_PROCESSOR,
        np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), mode == 'train', 6)
        for mode in ['train', 'test']}
    selections = json.loads((selection_root / 'preflight_rows.json').read_bytes())
    dataset_args = SimpleNamespace(data_root=manifest['data_root'], use_color=True,
        use_height=False, use_multiview=False, detect_intermediate=True, butd=False,
        butd_gt=False, butd_cls=True, augment_det=False, skip_missing_superpoints=True)

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    def linear(coordinates):
        return (coordinates[:, 0].astype(np.int64) * grid[1] + coordinates[:, 1]) * grid[0] + coordinates[:, 2]

    def check_voxels(points, value):
        assert np.array_equal(value['points'], points)
        # Match the actual Point2VoxelCPU3d float32 coordinate arithmetic.
        xyz = np.floor((points[:, :3] - minimum) / size).astype(np.int32)
        valid = ((xyz >= 0) & (xyz < grid)).all(1)
        valid_indices = np.flatnonzero(valid)
        ids = linear(xyz[valid][:, ::-1])
        order = np.argsort(ids, kind='stable')
        unique, starts, counts = np.unique(ids[order], return_index=True, return_counts=True)
        positions = np.arange(len(order)) - np.repeat(starts, counts)
        kept_indices = valid_indices[order[positions < 5]]
        actual_ids = linear(value['voxel_coords'])
        actual_order = np.argsort(actual_ids)
        assert np.array_equal(actual_ids[actual_order], unique)
        assert np.array_equal(value['voxel_num_points'][actual_order], np.minimum(counts, 5))
        stored = value['voxels'][actual_order]
        occupied = np.arange(5)[None, :] < np.minimum(counts, 5)[:, None]
        assert np.array_equal(stored[occupied], points[kept_indices])
        assert np.count_nonzero(stored[~occupied]) == 0
        assert len(unique) <= 50000
        return valid, kept_indices, len(unique)

    rows = []
    batches = 0
    for dset in ['nr3d', 'sr3d']:
        seed(spec['seed'])
        print('DATASET_LOADING ' + dset, flush=True)
        dataset = build_probe_dataset(Joint3DDataset, dataset_args, annotation, selections[dset], dset)
        original = copy.deepcopy(dataset.annos)
        for augmented in [False, True]:
            dataset.augment = augmented
            point_rows, input_rows = [], []
            for index, anno in enumerate(original):
                dataset.annos = copy.deepcopy(original)
                sample_seed = spec['seed'] + index
                seed(sample_seed)
                sample = dataset[index]
                points = sample['point_clouds']
                assert points.shape == (50000, 6) and points.dtype == np.float32 and np.isfinite(points).all()
                assert sample['superpoint'].shape == (50000,)
                assert np.array_equal(sample['all_detected_boxes'], sample['all_bboxes'])
                assert np.array_equal(sample['all_detected_bbox_label_mask'], sample['all_bbox_label_mask'])
                value = processors['train'].forward({'points': points.copy(), 'use_lead_xyz': True})
                valid, kept, voxel_count = check_voxels(points, value)
                test_value = processors['test'].forward({'points': points.copy(), 'use_lead_xyz': True})
                assert all(np.array_equal(value[key], test_value[key]) for key in
                    ['points', 'voxels', 'voxel_coords', 'voxel_num_points'])
                inputs = {key: sample[key] for key in ['point_clouds', 'utterances', 'superpoint',
                    'all_detected_boxes', 'all_detected_bbox_label_mask', 'all_detected_class_ids']}
                point_rows.append(value)
                input_rows.append(inputs)
                row = dict(dataset=dset, annotation_dataset=anno['dataset'], scan_id=anno['scan_id'],
                    selection_index=index, augmented=augmented, seed=sample_seed, point_count=len(points),
                    in_range_points=int(valid.sum()), out_of_range_points=int((~valid).sum()),
                    out_of_range_by_axis=((np.floor((points[:, :3] - minimum) / size) < 0) |
                        (np.floor((points[:, :3] - minimum) / size) >= grid)).sum(0).tolist(),
                    voxel_count=voxel_count, stored_point_slots=len(kept),
                    discarded_by_five_point_cap=int(valid.sum())-len(kept),
                    object_slots=int(sample['all_detected_bbox_label_mask'].sum()),
                    point_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
                    object_box_sha256=hashlib.sha256(sample['all_detected_boxes'].tobytes()).hexdigest(),
                    superpoint_sha256=hashlib.sha256(sample['superpoint'].numpy().tobytes()).hexdigest())
                if anno['dataset'] != 'scannet':
                    target = sample['point_instance_label'] == 0
                    row['root_target_diagnostic'] = dict(input_points=int(target.sum()),
                        in_range_points=int((target & valid).sum()), stored_point_slots=int(target[kept].sum()))
                rows.append(row)
                del sample, test_value
                if len(input_rows) == spec['batch_size']:
                    collated = processors['train'].collate_batch(point_rows)
                    native_batch = torch.utils.data._utils.collate.default_collate(input_rows)
                    count = len(input_rows)
                    assert collated['batch_size'] == count
                    assert np.array_equal(collated['points'][:, 0], np.repeat(np.arange(count), 50000))
                    assert np.array_equal(collated['points'][:, 1:].reshape(count, 50000, 6),
                        native_batch['point_clouds'].numpy())
                    for batch_index, (item, one_input) in enumerate(zip(point_rows, input_rows)):
                        mask = collated['voxel_coords'][:, 0] == batch_index
                        assert np.array_equal(collated['voxel_coords'][mask, 1:], item['voxel_coords'])
                        for key in ['all_detected_boxes', 'all_detected_bbox_label_mask', 'all_detected_class_ids', 'superpoint']:
                            assert np.array_equal(native_batch[key][batch_index].numpy(), np.asarray(one_input[key]))
                    batches += 1
                    point_rows, input_rows = [], []
                    del collated, native_batch
            assert not input_rows
            print('REAL_VOXEL_PASS ' + json.dumps(dict(dataset=dset, augmented=augmented, rows=16)), flush=True)
        del dataset
        gc.collect()
    assert len(rows) == 64 and batches == 16
    assert not torch.cuda.is_initialized()
    write_json(root / 'rows.json', rows)
    receipt = dict(status='pass', time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        elapsed_seconds=time.time()-started, unique_selected_rows=32, sampled_rows=len(rows), batches=batches,
        voxel_processor_calls=2*len(rows), source_files_and_inputs=verified,
        point_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE, voxel_size=voxel_cfg.VOXEL_SIZE,
        max_points_per_voxel=5, max_voxels=50000, reference_quantization='float32',
        coordinate_order='batch, z, y, x', raw_point_order_and_rgb_preserved=True,
        stored_voxel_points_match_independent_first_five_reference=True, train_test_processor_equal=True,
        native_object_slots_and_superpoints_preserved=True,
        torch_cuda_initialized=torch.cuda.is_initialized(), torch_version=torch.__version__,
        model_forwards=0, optimizer_steps=0, formal_rows=0, weights_loaded=0,
        script_sha256=sha(__file__), spec_sha256=sha(args.spec), rows_sha256=sha(root / 'rows.json'))
    write_json(root / 'receipt.json', receipt)
    print('REAL_VOXEL_AUDIT_COMPLETE ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
