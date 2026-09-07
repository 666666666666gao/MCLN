"""CPU full native sample/appearance attachment check; markers are not embeddings."""
import argparse
import copy
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
    source_manifest = source / 'appearance_source_manifest.json'
    assert sha(source_manifest) == plan['source_manifest_sha256']
    for name, digest in json.loads(source_manifest.read_text())['files'].items():
        assert sha(source / name) == digest, name
    for name, digest in plan['input_files'].items():
        assert sha(name) == digest, name
    annotation = json.loads((root / 'annotation_receipt.json').read_text())
    for path, info in annotation['annotations_and_split_files'].items():
        current = Path(path) if '/DATA_ROOT/' in path else source / 'data/meta_data' / Path(path).name
        assert sha(current) == info['sha256'], str(current)
    os.chdir(str(source))
    sys.path[:0] = [str(root), str(source)]
    import numpy as np
    import torch
    from src.joint_det_dataset import Joint3DDataset
    from binding import attach_object_appearance
    # Reuse only the fixed, performance-independent annotation selection helper.
    from fixed_selection import build_probe_dataset
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    metadata = {row['scan_id']: row for row in json.loads(Path(plan['scene_slots']).read_text())}
    selections = json.loads((root / 'preflight_rows.json').read_text())
    assert sha(root / 'preflight_rows.json') == annotation['preflight_rows_sha256']
    args = SimpleNamespace(data_root=plan['data_root'], use_color=True,
        use_height=False, use_multiview=False, detect_intermediate=True,
        butd=False, butd_gt=False, butd_cls=True, augment_det=False,
        skip_missing_superpoints=True)

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    def equal(left, right):
        if isinstance(left, np.ndarray):
            return left.dtype == right.dtype and np.array_equal(left, right)
        if torch.is_tensor(left):
            return left.dtype == right.dtype and torch.equal(left, right)
        return left == right

    rows = []
    started = time.time()
    for dset in ['nr3d', 'sr3d']:
        seed(0)
        dataset = build_probe_dataset(Joint3DDataset, args, annotation, selections[dset], dset)
        original = copy.deepcopy(dataset.annos)
        for augmented in [False, True]:
            dataset.augment = augmented
            for value in plan['seeds']:
                batch = []
                for index, anno in enumerate(original):
                    scan = dataset.scans[anno['scan_id']]
                    info = metadata[anno['scan_id']]
                    canonical = np.concatenate([scan.orig_pc.astype(np.float32),
                        (scan.color - dataset.mean_rgb).astype(np.float32)], axis=1)
                    assert hashlib.sha256(canonical.tobytes()).hexdigest() == info['native_point_sha256']
                    sp = Path(plan['data_root']) / 'superpoints/train' / (anno['scan_id'] + '_superpoint.pth')
                    assert sha(sp) == plan['superpoints'][sp.name]
                    dataset.annos = copy.deepcopy(original)
                    seed(value * 100 + index)
                    native = dataset[index]
                    native_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
                    dataset.annos = copy.deepcopy(original)
                    seed(value * 100 + index)
                    slots = np.asarray(info['slot_ids'])
                    marker = np.repeat((slots + 1).astype(np.float32)[:, None], 1280, axis=1)
                    result = attach_object_appearance(dataset[index], marker, info['available_384'], slots)
                    assert set(result) - set(native) == {'det_visual_features', 'det_visual_available'}
                    assert all(equal(native[k], result[k]) for k in native), (dset, index)
                    assert random.getstate() == native_rng[0]
                    after = np.random.get_state()
                    assert after[0] == native_rng[1][0] and np.array_equal(after[1], native_rng[1][1])
                    assert after[2:] == native_rng[1][2:]
                    assert torch.equal(torch.get_rng_state(), native_rng[2])
                    valid = result['det_visual_available']
                    assert np.array_equal(result['det_visual_features'][valid, 0],
                                          (np.flatnonzero(valid) + 1).astype(np.float32))
                    assert np.count_nonzero(result['det_visual_features'][~valid]) == 0
                    batch.append(result)
                    rows.append(dict(dataset=dset, annotation_dataset=anno['dataset'],
                        scan_id=anno['scan_id'], augmented=augmented, seed=value,
                        original_fields=len(native), available_slots=int(valid.sum()),
                        original_fields_equal=True, rng_equal=True,
                        point_sha256=hashlib.sha256(native['point_clouds'].tobytes()).hexdigest()))
                collated = torch.utils.data._utils.collate.default_collate(batch)
                assert collated['det_visual_features'].shape == (16, 132, 1280)
                assert collated['det_visual_available'].dtype == torch.bool
                print('FULL GETITEM PASS', dset, augmented, value, len(batch), flush=True)
        del dataset, batch, collated
        gc.collect()
    for name, digest in plan['input_files'].items():
        assert sha(name) == digest, name
    with (root / 'rows.json').open('x') as stream:
        json.dump(rows, stream, sort_keys=True, allow_nan=False)
    receipt = dict(status='pass', checks=len(rows), native_getitem_calls=2 * len(rows),
        rows_sha256=sha(root / 'rows.json'), manifest_sha256=sha(manifest_path),
        full_getitem_and_collation_tested=True, original_fields_and_rng_unchanged=True,
        appearance_inputs='synthetic slot identity markers only',
        model_forwards=0, optimizer_steps=0, formal_rows=0, elapsed_seconds=time.time()-started)
    with (root / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
    print('FULL GETITEM COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
