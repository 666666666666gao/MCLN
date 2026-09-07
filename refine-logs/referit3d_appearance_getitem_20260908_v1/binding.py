"""Bind compact frozen object features to native, potentially sparse slots."""
import numpy as np


def scatter_object_appearance(features, available, slot_ids, valid_mask):
    """Keep native slot identity; unavailable crops and padding contribute zero."""
    features = np.asarray(features, dtype=np.float32)
    available = np.asarray(available, dtype=np.bool_)
    slot_ids = np.asarray(slot_ids, dtype=np.int64)
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    assert features.shape == (len(slot_ids), 1280)
    assert available.shape == slot_ids.shape
    assert np.array_equal(np.flatnonzero(valid_mask), slot_ids)
    assert np.isfinite(features).all()
    output = np.zeros((len(valid_mask), 1280), dtype=np.float32)
    output_available = np.zeros(len(valid_mask), dtype=np.bool_)
    output[slot_ids[available]] = features[available]
    output_available[slot_ids] = available
    return output, output_available


def attach_object_appearance(sample, features, available, slot_ids):
    """Append canonical appearance to an already constructed native sample."""
    visual, valid = scatter_object_appearance(
        features, available, slot_ids, sample['all_detected_bbox_label_mask'])
    result = dict(sample)
    result['det_visual_features'] = visual
    result['det_visual_available'] = valid
    return result
