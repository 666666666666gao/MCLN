"""Capture native pre-jitter annotation boxes without changing native data."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    root = Path(spec['root'])
    prior = Path(spec['cohort_root'])
    old_spec = json.loads((prior / 'spec.json').read_bytes())
    old_imports = json.loads((prior / 'imports.json').read_bytes())
    manifest = json.loads(Path(old_spec['input_manifest']).read_bytes())
    source = Path(manifest['model_source'])
    assert sha(source / 'appearance_source_manifest.json') == manifest['source_manifest_sha256']
    for name, digest in json.loads((source / 'appearance_source_manifest.json').read_bytes())['files'].items():
        assert sha(source / name) == digest, name
    assert sha(manifest['split_protocol']) == manifest['split_protocol_sha256']
    partitions = json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
    assert len(partitions['fit']) == 29778 and len(partitions['holdout']) == 6887
    assert not set(partitions['fit']).intersection(partitions['holdout'])
    assert old_spec['seed'] == 2027 and old_spec['batch_size'] == 8
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    imported = Path(sys.modules['src.joint_det_dataset'].__file__).resolve()
    assert str(imported) == old_imports['files']['src.joint_det_dataset']
    assert sha(imported) == old_imports['sha256']['src.joint_det_dataset']

    def reset_rng():
        random.seed(2027)
        np.random.seed(2027)
        torch.manual_seed(2027)

    reset_rng()

    class FitDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            actual = {'fit': [], 'holdout': []}
            for index, row in enumerate(annos):
                row['_local_training_id'] = index
                code = (manifest['split_salt'] + '\0' + row['scan_id'].split('_')[0]).encode()
                fold = int(hashlib.sha256(code).hexdigest()[:8], 16) % 5
                actual['holdout' if fold == 0 else 'fit'].append(index)
            assert actual == partitions
            super()._scene_graph_parse(annos)

        def _get_target_boxes(self, anno, scan):
            # Pure annotation lookup on the augmented full scan; no random call.
            target = anno['target_id'][0] if isinstance(anno['target_id'], list) else anno['target_id']
            corners = scan.get_object_bbox(target).reshape(6).astype(np.float64)
            self.pre_jitter_root = np.concatenate(((corners[:3] + corners[3:]) * .5,
                                                   corners[3:] - corners[:3])).astype(np.float32)
            return super()._get_target_boxes(anno, scan)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            result['local_training_id'] = self.annos[index]['_local_training_id']
            result['pre_jitter_root_box'] = self.pre_jitter_root.copy()
            assert np.isin(result['gt_masks'], [0, 1]).all()
            result['gt_masks'] = result['gt_masks'].astype(np.bool_)
            return result

    verify_scanrefer_superpoints(manifest['data_root'], 'train', manifest['superpoint_files']['train'])
    dataset = FitDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
        data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False,
        detect_intermediate=True, butd=True, butd_cls=False, butd_gt=False,
        augment_det=False, skip_missing_superpoints=True)
    assert len(dataset) == 36665
    dataset.augment = False
    reference = json.loads((Path(old_spec['reference_fixtures']) / 'receipt.json').read_bytes())
    reset_rng()
    for row in reference['rows']:
        sample = dataset[row['training_row_id']]
        checks = dict(point_clouds=sample['point_clouds'], det_boxes=sample['all_detected_boxes'],
            det_bbox_label_mask=sample['all_detected_bbox_label_mask'],
            det_class_ids=sample['all_detected_class_ids'], superpoint=sample['superpoint'].numpy())
        assert sample['utterances'] == row['text']
        for name, array in checks.items():
            assert hashlib.sha256(array.tobytes()).hexdigest() == row['tensor_sha256'][name], name
        assert np.array_equal(sample['pre_jitter_root_box'],
                              np.concatenate([sample['center_label'][0, :3], sample['size_gts'][0]]))
    dataset.augment = True
    dataset.augment_det = True
    reset_rng()
    loader = DataLoader(Subset(dataset, partitions['fit']), batch_size=8, shuffle=True, num_workers=2,
        generator=torch.Generator().manual_seed(2027), pin_memory=True, drop_last=False)
    prior_rows = [json.loads(line) for line in (prior / 'rows.jsonl').read_text().splitlines()]
    assert len(prior_rows) == 64
    records = []
    for index, batch in enumerate(loader):
        assert index < 8
        ids = batch['local_training_id'].tolist()
        assert ids == [row['row_id'] for row in prior_rows[index * 8:(index + 1) * 8]]
        for bid, row_id in enumerate(ids):
            previous = prior_rows[index * 8 + bid]
            digest = hashlib.sha256(batch['point_clouds'][bid].numpy().tobytes()).hexdigest()
            assert digest == previous['point_sha256']
            assert batch['scan_ids'][bid] == previous['scan_id']
            valid = batch['box_label_mask'][bid].bool().nonzero().flatten().tolist()
            assert valid == previous['valid_native_GT_slots'] == [0]
            noisy = torch.cat([batch['center_label'][bid, 0, :3], batch['size_gts'][bid, 0]])
            assert noisy.tolist() == previous['root_box']
            clean = batch['pre_jitter_root_box'][bid]
            assert torch.isfinite(clean).all() and (clean[3:] > 0).all()
            records.append(dict(row_id=row_id, scan_id=previous['scan_id'], point_sha256=digest,
                valid_native_GT_slots=valid, noisy_root_box=noisy.tolist(),
                pre_jitter_root_box=clean.tolist()))
        print(json.dumps(dict(completed_rows=len(records), elapsed_seconds=time.perf_counter() - start)), flush=True)
        if len(records) == 64:
            break
    assert len(records) == 64 and 'models' not in sys.modules
    assert not torch.cuda.is_initialized()
    with (root / 'rows.jsonl').open('x') as stream:
        for row in records:
            stream.write(json.dumps(row) + '\n')
    result = dict(status='complete', finished_cst=datetime.datetime.now().astimezone().isoformat(),
        rows=64, sampled_points_and_native_noisy_GT_exact=True, root_only=True,
        elapsed_seconds=time.perf_counter() - start, optimizer_steps=0, model_forwards=0,
        weights_loaded=0, weights_created=0, CUDA_initialized=False, accuracy_result=False,
        dataset_source=str(imported), dataset_source_sha256=sha(imported),
        native_gt_protocol_changed=False, clean_target='Annotation object members in the full augmented scan before independent native box jitter')
    (root / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
