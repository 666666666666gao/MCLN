"""Complete canonical appearance metadata for native joint detection scenes.

CPU-only preparation. Keep protocol instance slots and predicted classes; do
not use target labels or instance masks to clean the AABB point crops.
"""
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
    manifest_path = parser.parse_args().manifest.resolve()
    root = manifest_path.parent
    plan = json.loads(manifest_path.read_text())
    source = Path(plan['model_source'])
    source_manifest = source / 'appearance_source_manifest.json'
    assert sha(source_manifest) == plan['source_manifest_sha256']
    for name, digest in json.loads(source_manifest.read_text())['files'].items():
        assert sha(source / name) == digest, name
    for name, digest in plan['input_files'].items():
        assert sha(name) == digest, name
    original = json.loads(Path(plan['language_slots']).read_text())
    coverage = json.loads(Path(plan['coverage_receipt']).read_text())
    records = {row['scan_id']: row for row in original}
    assert len(records) == len(original) == 1061
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from src.joint_det_dataset import Joint3DDataset, unpickle_data, read_label_mapping
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    dataset = object.__new__(Joint3DDataset)
    dataset.split, dataset.augment = 'train', False
    dataset.label_map = read_label_mapping(
        'data/meta_data/scannetv2-labels.combined.tsv',
        label_from='raw_category', label_to='id')
    dataset.scans = list(unpickle_data(str(Path(plan['data_root']) / 'train_v3scans.pkl')))[0]
    annos = dataset.load_scannet_annos()
    detection = {row['scan_id'] for row in annos}
    assert len(annos) == len(detection) == coverage['detection_rows'] == 1199
    missing = sorted(detection - set(records))
    assert missing == coverage['missing_detection_scenes'] and len(missing) == 139
    predictions = json.loads((source / 'data/cls_results.json').read_text())
    mean_rgb = np.array([109.8, 97.2, 83.8]) / 256
    start = time.time()
    for index, scan_id in enumerate(missing):
        scan = dataset.scans[scan_id]
        scan.pc = np.copy(scan.orig_pc)
        _, boxes, valid = dataset._get_scene_objects(scan)
        boxes = boxes.astype(np.float32)
        slots = np.flatnonzero(valid)
        classes = np.asarray(predictions[scan_id])
        classes = classes[classes > -1]
        assert len(classes) == len(slots)
        xyz = np.asarray(scan.orig_pc, dtype=np.float32)
        native = np.concatenate([xyz, np.asarray(scan.color - mean_rgb, dtype=np.float32)], axis=1)
        assert native.shape == (50000, 6) and np.isfinite(native).all()
        counts = []
        for slot in slots:
            box = boxes[slot]
            inside = ((xyz >= box[:3] - box[3:] * .5)
                      & (xyz <= box[:3] + box[3:] * .5)).all(axis=1)
            counts.append(int(inside.sum()))
        records[scan_id] = dict(
            scan_id=scan_id, datasets=[], slot_ids=slots.tolist(),
            padded_slot_count=len(valid), boxes=boxes[slots].tolist(),
            predicted_class_ids=classes.tolist(), crop_point_counts=counts,
            available_384=[count >= 384 for count in counts],
            native_point_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
            padded_boxes_sha256=hashlib.sha256(boxes.tobytes()).hexdigest())
        if index == 0 or (index + 1) % 32 == 0:
            print('JOINT DETECTION SLOT EXPORT', json.dumps(dict(
                completed=index + 1, total=len(missing), elapsed_seconds=time.time() - start)), flush=True)
    for scan_id in detection:
        records[scan_id]['datasets'].append('scannet')
    assert len(records) == coverage['union_scenes'] == 1200
    # Existing language records must be byte-value identical except the new
    # membership tag. Deep-copy via re-reading the unchanged original file.
    for row in json.loads(Path(plan['language_slots']).read_text()):
        actual = dict(records[row['scan_id']])
        actual['datasets'] = [name for name in actual['datasets'] if name != 'scannet']
        assert actual == row, row['scan_id']
    result = sorted(records.values(), key=lambda row: row['scan_id'])
    with (root / 'scene_slots.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
    counts_by_source = {}
    for name in ['nr3d', 'sr3d', 'scannet']:
        subset = [row for row in result if name in row['datasets']]
        counts_by_source[name] = dict(
            scenes=len(subset), slots=sum(len(row['slot_ids']) for row in subset),
            available_384=sum(sum(row['available_384']) for row in subset),
            noncontiguous_scenes=sum(row['slot_ids'] != list(range(len(row['slot_ids']))) for row in subset))
    for name, digest in plan['input_files'].items():
        assert sha(name) == digest, name
    receipt = dict(
        status='complete', schema='mcln-referit-joint-detection-slot-extension-v1',
        unique_scenes=len(result), added_scenes=len(missing), by_source=counts_by_source,
        original_language_records_preserved=True, native_detection_rows=len(annos),
        language_only_scenes=coverage['language_only_scenes'],
        scene_slots_sha256=sha(root / 'scene_slots.json'), manifest_sha256=sha(manifest_path),
        elapsed_seconds_excluding_scene_loading=time.time() - start,
        protocol_instance_boxes=True, predicted_classes=True,
        target_ids_used_for_cropping=False, instance_masks_used_for_cropping=False,
        appearance_embeddings_computed=False, model_forwards=0, optimizer_steps=0,
        formal_rows=0, augmentation=False)
    with (root / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
    print('JOINT DETECTION SLOT EXPORT COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
