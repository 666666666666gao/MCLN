"""Check native augmented slot identity and a canonical appearance binding seam.

Uses real training annotations/scenes and native geometry functions. Synthetic
slot markers check indexing only; no pretrained features or model gains.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    manifest_path = p.parse_args().manifest.resolve()
    root = manifest_path.parent
    plan = json.loads(manifest_path.read_text())
    source = Path(plan['model_source'])
    assert sha(source / 'appearance_source_manifest.json') == plan['source_manifest_sha256']
    for name, digest in json.loads((source / 'appearance_source_manifest.json').read_text())['files'].items():
        assert sha(source / name) == digest, name
    for name, digest in plan['input_files'].items():
        assert sha(name) == digest, name
    os.chdir(str(source))
    sys.path[:0] = [str(root), str(source)]
    import numpy as np
    import torch
    from src.joint_det_dataset import Joint3DDataset, VIEW_DEP_RELS, unpickle_data, read_label_mapping
    from binding import scatter_object_appearance
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    ds = object.__new__(Joint3DDataset)
    ds.split, ds.augment = 'train', True
    ds.use_color, ds.use_height, ds.use_multiview = True, False, False
    ds.mean_rgb = np.array([109.8, 97.2, 83.8]) / 256
    ds.label_map = read_label_mapping('data/meta_data/scannetv2-labels.combined.tsv', label_from='raw_category', label_to='id')
    ds.scans = list(unpickle_data(str(Path(plan['data_root']) / 'train_v3scans.pkl')))[0]
    metadata = {row['scan_id']: row for row in json.loads(Path(plan['scene_slots']).read_text())}
    selected = json.loads(Path(plan['preflight_rows']).read_text())
    cases = [dict(row, source_group=d) for d in ['nr3d', 'sr3d'] for row in selected[d]]
    # Cover every actually observed gap in joint detection slots as well.
    detection = {row['scan_id']: row for row in ds.load_scannet_annos()}
    gap_ids = sorted(k for k, v in metadata.items()
                     if 'scannet' in v['datasets'] and v['slot_ids'] != list(range(len(v['slot_ids']))))
    assert len(gap_ids) == 10
    cases.extend(dict(detection[k], source_group='slot_gap_detection') for k in gap_ids)
    assert len(cases) == 42
    rows = []
    start = time.time()
    for index, anno in enumerate(cases):
        scan = ds.scans[anno['scan_id']]
        info = metadata[anno['scan_id']]
        canonical = np.concatenate([np.asarray(scan.orig_pc, dtype=np.float32),
                                    np.asarray(scan.color - ds.mean_rgb, dtype=np.float32)], axis=1)
        assert hashlib.sha256(canonical.tobytes()).hexdigest() == info['native_point_sha256']
        original_rgb_sha = hashlib.sha256(scan.color.tobytes()).hexdigest()
        allow = (anno['dataset'] == 'scannet'
                 or (anno['dataset'] == 'nr3d' and ds._augment_nr3d(anno['utterance']))
                 or (anno['dataset'].startswith('sr3d') and ds._find_rel(anno['utterance']) not in VIEW_DEP_RELS))
        for seed in plan['seeds']:
            np.random.seed(seed)
            scan.pc = np.copy(scan.orig_pc)
            points, aug, _ = ds._get_pc(anno, scan)
            _, boxes, valid = ds._get_scene_objects(scan)
            slots = np.flatnonzero(valid)
            assert slots.tolist() == info['slot_ids']
            if not allow:
                assert abs(aug['theta_z']) <= 5 and not aug.get('yz_flip', False) and not aug.get('xz_flip', False)
            assert not np.array_equal(points.astype(np.float32), canonical)
            assert not np.array_equal(boxes[slots].astype(np.float32), np.asarray(info['boxes'], dtype=np.float32))
            marker = np.repeat((slots + 1).astype(np.float32)[:, None], 1280, axis=1)
            visual, available = scatter_object_appearance(marker, info['available_384'], slots, valid)
            good = np.asarray(info['available_384'], dtype=np.bool_)
            assert np.array_equal(visual[slots[good], 0], (slots[good] + 1).astype(np.float32))
            assert np.count_nonzero(visual[~available]) == 0
            assert int(available.sum()) == int(good.sum())
            assert hashlib.sha256(scan.color.tobytes()).hexdigest() == original_rgb_sha
            fresh = np.concatenate([np.asarray(scan.orig_pc, dtype=np.float32),
                                    np.asarray(scan.color - ds.mean_rgb, dtype=np.float32)], axis=1)
            assert np.array_equal(fresh, canonical)
            rows.append(dict(case_index=index, source_group=anno['source_group'], dataset=anno['dataset'],
                scan_id=anno['scan_id'], seed=seed, large_rotation_allowed=allow,
                theta_z=aug['theta_z'], yz_flip=aug.get('yz_flip', False), xz_flip=aug.get('xz_flip', False),
                slots=len(slots), available_slots=int(good.sum()), slot_binding_pass=True,
                geometry_changed=True, canonical_observation_unchanged=True))
        print('AUGMENTATION SLOT CHECK', index + 1, len(cases), flush=True)
    with (root / 'rows.json').open('x') as stream:
        json.dump(rows, stream, sort_keys=True, allow_nan=False)
    for name, digest in plan['input_files'].items():
        assert sha(name) == digest, name
    receipt = dict(status='pass', cases=len(cases), checks=len(rows), gap_scenes=len(gap_ids),
        rows_sha256=sha(root / 'rows.json'), manifest_sha256=sha(manifest_path),
        view_restricted_checks=sum(not x['large_rotation_allowed'] for x in rows),
        canonical_observation_unchanged=True, original_slot_binding_pass=True,
        appearance_inputs='synthetic slot identity markers only', actual_appearance_invariance_tested=False,
        native_functions=['_get_pc', '_get_scene_objects', 'load_scannet_annos'],
        full_getitem_tested=False, model_forwards=0, optimizer_steps=0, formal_rows=0,
        elapsed_seconds_excluding_scene_loading=time.time() - start)
    with (root / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
    print('AUGMENTATION SLOT CHECK COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
