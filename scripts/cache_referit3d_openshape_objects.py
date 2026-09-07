"""Frozen canonical appearance for native ReferIt3D/joint-det instance slots.

Same G14 crop recipe as ScanRefer; protocol instance boxes replace GroupFree
boxes. This input distinction is explicit and is not a no-GT-box setting.
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
    path = p.parse_args().manifest.resolve()
    out = path.parent
    spec = json.loads(path.read_text())
    assert sha(__file__) == spec['script_sha256']
    # Qualification is verified again in the actual encoder process.
    qualification = spec['qualification']
    for name, digest in qualification['files'].items():
        assert sha(name) == digest, name
    decision = json.loads(Path(qualification['decision']).read_text())
    assert decision['status'] == 'formal_evaluated_and_audited'
    assert decision['promotion']['advance_to_nr3d_sr3d_rec']
    assert decision['formal_rows'] == 9508
    runtime = Path(spec['runtime_root'])
    runtime_spec = json.loads((runtime / 'runtime_spec_v2.json').read_text())
    assert hashlib.sha256(json.dumps(runtime_spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['runtime_spec_sha256']
    for name, info in runtime_spec['files'].items():
        if name.startswith('vendor/') or name == 'assets/model.pt':
            assert sha(runtime / name) == info['sha256'], name
    source = Path(spec['model_source'])
    assert sha(source / 'appearance_source_manifest.json') == spec['source_manifest_sha256']
    for name, digest in json.loads((source / 'appearance_source_manifest.json').read_text())['files'].items():
        assert sha(source / name) == digest, name
    assert sha(spec['scene_slots']) == spec['scene_slots_sha256']
    metadata = json.loads(Path(spec['scene_slots']).read_text())
    assert len(metadata) == len({row['scan_id'] for row in metadata}) == 1200
    scene_file = Path(spec['data_root']) / 'train_v3scans.pkl'
    assert sha(scene_file) == spec['scene_pickle_sha256']
    sys.path.insert(0, str(source))
    os.chdir(str(source))
    import numpy as np
    import torch
    from openshape import G14
    from src.joint_det_dataset import unpickle_data
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = G14(torch.load(str(runtime / 'assets/model.pt'), map_location='cpu')).cuda().eval().requires_grad_(False)
    scenes = list(unpickle_data(str(scene_file)))[0]
    records = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for info in metadata:
        scene_id = info['scan_id']
        scene = scenes[scene_id]
        xyz = np.asarray(scene.orig_pc, dtype=np.float32)
        rgb = np.asarray(scene.color, dtype=np.float32)
        assert xyz.shape == rgb.shape == (50000, 3)
        assert np.isfinite(xyz).all() and np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1
        native_rgb = np.asarray(scene.color - np.array([109.8, 97.2, 83.8]) / 256, dtype=np.float32)
        point_sha = hashlib.sha256(np.concatenate([xyz, native_rgb], axis=1).tobytes()).hexdigest()
        assert point_sha == info['native_point_sha256']
        boxes = np.asarray(info['boxes'], dtype=np.float32)
        slots = np.asarray(info['slot_ids'], dtype=np.int64)
        features = np.zeros((len(slots), 1280), dtype=np.float32)
        counts = np.zeros(len(slots), dtype=np.int32)
        available = np.zeros(len(slots), dtype=np.bool_)
        for compact_index, (slot, box) in enumerate(zip(slots, boxes)):
            inside = ((xyz >= box[:3] - box[3:] * .5) & (xyz <= box[:3] + box[3:] * .5)).all(axis=1)
            counts[compact_index] = int(inside.sum())
            assert int(counts[compact_index]) == info['crop_point_counts'][compact_index]
            if counts[compact_index] < 384:
                continue
            pc = np.concatenate([xyz[inside], rgb[inside]], axis=1)
            # Seed by the original instance slot, never the compact row number.
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
            features[compact_index] = value
            available[compact_index] = True
        assert available.tolist() == info['available_384']
        name = 'features/' + scene_id + '.npz'
        with (out / name).open('xb') as stream:
            np.savez_compressed(stream, features=features, available=available, boxes=boxes,
                point_counts=counts, slot_ids=slots, predicted_class_ids=np.asarray(info['predicted_class_ids'], dtype=np.int64))
        record = dict(scene_id=scene_id, datasets=info['datasets'], file=name, file_sha256=sha(out/name),
            object_slots=len(slots), available_slots=int(available.sum()), slot_ids=slots.tolist(),
            padded_slot_count=info['padded_slot_count'], native_point_sha256=point_sha,
            padded_boxes_sha256=info['padded_boxes_sha256'], point_counts=counts.tolist())
        records.append(record)
        with (out / 'scenes.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print('REFERIT APPEARANCE CACHE', len(records), 1200, scene_id, int(available.sum()), flush=True)
    torch.cuda.synchronize()
    assert sha(spec['scene_slots']) == spec['scene_slots_sha256']
    assert sha(scene_file) == spec['scene_pickle_sha256']
    receipt = dict(status='complete', scene_count=len(records),
        object_slots=sum(row['object_slots'] for row in records),
        available_slots=sum(row['available_slots'] for row in records),
        scenes_sha256=sha(out/'scenes.jsonl'), manifest_sha256=sha(path),
        scene_slots_sha256=spec['scene_slots_sha256'], qualification=qualification,
        elapsed_seconds=time.time()-started, gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        uses_protocol_instance_boxes=True, predicted_classes=True, target_ids_used_for_cropping=False,
        instance_masks_used_for_cropping=False, language_used_for_encoding=False,
        mcln_forwards=0, optimizer_steps=0, formal_rows=0,
        input_contract='Canonical original XYZ/RGB, protocol instance boxes, same frozen G14 recipe; original slot IDs; missing <384 crops remain unavailable.')
    with (out/'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
    print('REFERIT APPEARANCE CACHE COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
