"""Frozen validation-scene visual preprocessing, with no annotation-based cropping."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    path = parser.parse_args().manifest
    spec = json.loads(path.read_text())
    out = path.parent
    runtime = Path(spec['runtime_root'])
    runtime_spec = json.loads((runtime / 'runtime_spec_v2.json').read_text())
    assert hashlib.sha256(json.dumps(runtime_spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['runtime_spec_sha256']
    for name, info in runtime_spec['files'].items():
        if name.startswith('vendor/') or name == 'assets/model.pt':
            assert sha(runtime / name) == info['sha256'], name
    assert sha(spec['scene_list']) == spec['scene_list_sha256']
    source = Path(spec['model_source'])
    assert sha(source / 'native_source_manifest.json') == spec['source_manifest_sha256']
    source_manifest = json.loads((source / 'native_source_manifest.json').read_text())
    for name, digest in source_manifest['files'].items():
        assert sha(source / name) == digest, name
    sys.path.insert(0, str(source))
    os.chdir(str(source))
    import numpy as np
    import torch
    from openshape import G14
    from src.joint_det_dataset import unpickle_data
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = G14(torch.load(str(runtime / 'assets/model.pt'), map_location='cpu')).cuda().eval().requires_grad_(False)
    data_root = Path(spec['data_root'])
    scene_ids = sorted(set(Path(spec['scene_list']).read_text().split()))
    assert len(scene_ids) == spec['scene_count'] == 141
    # Serialized object annotations are never accessed. Only original scene
    # XYZ/RGB and the established predicted detector boxes enter this cache.
    scene_file = data_root / 'val_v3scans.pkl'
    source_data_sha = sha(scene_file)
    scenes = list(unpickle_data(str(scene_file)))[0]
    print('CACHE_INPUT_READY', len(scene_ids), source_data_sha, flush=True)
    records = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        xyz = np.asarray(scene.orig_pc, dtype=np.float32)
        rgb = np.asarray(scene.color, dtype=np.float32)
        assert xyz.shape == rgb.shape == (50000, 3)
        assert np.isfinite(xyz).all() and np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1
        detector_path = data_root / 'group_free_pred_bboxes/group_free_pred_bboxes_val' / (scene_id + '.npy')
        detector = np.load(str(detector_path), allow_pickle=True).item()
        bounds = np.asarray(detector['box'])
        boxes = np.concatenate([(bounds[:, :3] + bounds[:, 3:]) * .5, bounds[:, 3:] - bounds[:, :3]], axis=1).astype(np.float32)
        assert len(boxes) == len(detector['class'])
        features = np.zeros((len(boxes), 1280), dtype=np.float32)
        counts = np.zeros(len(boxes), dtype=np.int32)
        valid = np.zeros(len(boxes), dtype=np.bool_)
        for slot, box in enumerate(boxes):
            inside = ((xyz >= box[:3] - box[3:] * .5) & (xyz <= box[:3] + box[3:] * .5)).all(axis=1)
            counts[slot] = int(inside.sum())
            if counts[slot] < 384:
                continue
            pc = np.concatenate([xyz[inside], rgb[inside]], axis=1)
            seed = int(hashlib.sha256((scene_id + ':' + str(slot)).encode()).hexdigest()[:8], 16)
            if len(pc) > 10000:
                pc = pc[np.random.RandomState(seed).choice(len(pc), 10000, replace=False)]
            pc[:, :3] -= pc[:, :3].mean(axis=0)
            radius = np.linalg.norm(pc[:, :3], axis=1).max()
            assert radius > 0
            pc[:, :3] /= radius
            torch.manual_seed(seed)
            with torch.no_grad():
                value = model(torch.from_numpy(pc.T.copy())[None].cuda())[0].cpu().numpy()
            assert value.shape == (1280,) and np.isfinite(value).all() and np.linalg.norm(value) > 0
            features[slot] = value
            valid[slot] = True
        name = scene_id + '.npz'
        with (out / 'features' / name).open('xb') as stream:
            np.savez_compressed(stream, features=features, available=valid, boxes=boxes, point_counts=counts)
        native_rgb = np.asarray(scene.color - np.array([109.8, 97.2, 83.8]) / 256, dtype=np.float32)
        record = dict(scene_id=scene_id, file='features/' + name, file_sha256=sha(out / 'features' / name),
                      object_slots=len(boxes), available_slots=int(valid.sum()),
                      detector_sha256=sha(detector_path),
                      native_point_sha256=hashlib.sha256(np.concatenate([xyz, native_rgb], axis=1).tobytes()).hexdigest(),
                      point_counts=counts.tolist())
        records.append(record)
        with (out / 'scenes.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print('CACHE_SCENE', len(records), scene_id, len(boxes), int(valid.sum()), flush=True)
    torch.cuda.synchronize()
    receipt = dict(status='complete', scene_count=len(records),
                   object_slots=sum(v['object_slots'] for v in records),
                   available_slots=sum(v['available_slots'] for v in records),
                   manifest_sha256=sha(path), source_data_sha256=source_data_sha,
                   scenes_sha256=sha(out / 'scenes.jsonl'), elapsed_seconds=time.time() - started,
                   gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                   gt_used_for_crop=False, language_used=False, optimizer_steps=0,
                   mcln_forwards=0, formal_rows=0, scene_list_split='ScanRefer val',
                   input_contract='Unaugmented native 50000 points, predicted GroupFree boxes, scene-slot deterministic FPS and sampling. Missing <384 point crops have zero features and available=false.')
    with (out / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print('CACHE_COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
