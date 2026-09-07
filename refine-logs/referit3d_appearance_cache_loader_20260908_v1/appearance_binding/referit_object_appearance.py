"""Canonical frozen appearance attached to native ReferIt3D instance slots."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .object_appearance_inputs import attach_object_appearance


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


class ReferItObjectAppearanceDataset(Dataset):
    """Validate canonical observations once, then retain native augmentation."""

    def __init__(self, native, cache_root, expected_receipt_sha256):
        assert native.butd_cls and not native.butd and not native.butd_gt
        self.native = native
        root = Path(cache_root)
        assert file_sha(root / 'receipt.json') == expected_receipt_sha256
        receipt = json.loads((root / 'receipt.json').read_text())
        assert receipt['status'] == 'complete' and receipt['uses_protocol_instance_boxes']
        assert not receipt['target_ids_used_for_cropping'] and not receipt['instance_masks_used_for_cropping']
        assert file_sha(root / 'scenes.jsonl') == receipt['scenes_sha256']
        rows = [json.loads(line) for line in (root / 'scenes.jsonl').read_text().splitlines()]
        records = {row['scene_id']: row for row in rows}
        assert len(records) == len(rows) == receipt['scene_count']
        self.features = {}
        canonical = copy.copy(native)
        canonical.augment = False
        for scene_id in sorted({row['scan_id'] for row in native.annos}):
            record = records[scene_id]
            scan = copy.copy(native.scans[scene_id])
            scan.pc = np.copy(scan.orig_pc)
            points = np.concatenate([scan.orig_pc.astype(np.float32),
                (scan.color - native.mean_rgb).astype(np.float32)], axis=1)
            assert hashlib.sha256(points.tobytes()).hexdigest() == record['native_point_sha256'], scene_id
            _, boxes, valid = canonical._get_scene_objects(scan)
            boxes = boxes.astype(np.float32)
            assert hashlib.sha256(boxes.tobytes()).hexdigest() == record['padded_boxes_sha256'], scene_id
            path = root / record['file']
            assert file_sha(path) == record['file_sha256'], scene_id
            with np.load(str(path), allow_pickle=False) as stored:
                feature = {key: stored[key] for key in
                           ['features', 'available', 'slot_ids', 'boxes', 'predicted_class_ids']}
            slots = feature['slot_ids']
            assert np.array_equal(slots, np.flatnonzero(valid))
            assert slots.tolist() == record['slot_ids']
            assert np.array_equal(feature['boxes'], boxes[slots])
            classes = np.asarray(native.cls_results[scene_id])
            assert np.array_equal(feature['predicted_class_ids'], classes[classes > -1])
            assert feature['features'].shape == (len(slots), 1280)
            assert feature['available'].shape == (len(slots),) and feature['available'].dtype == np.bool_
            assert np.isfinite(feature['features']).all()
            assert np.count_nonzero(feature['features'][~feature['available']]) == 0
            self.features[scene_id] = feature

    def __len__(self):
        return len(self.native)

    def __getitem__(self, index):
        scene_id = self.native.annos[index]['scan_id']
        feature = self.features[scene_id]
        sample = self.native[index]
        assert np.array_equal(sample['all_detected_class_ids'][feature['slot_ids']],
                              feature['predicted_class_ids'])
        return attach_object_appearance(sample, feature['features'],
                                        feature['available'], feature['slot_ids'])
