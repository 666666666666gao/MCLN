"""Diagnostics and optimizer accounting for the fixed Mask geometry pair."""
import torch

from models.rec_mask_geometry import (
    _find_superpoint_map, normalize_mcln_mask_logits, mask_logits_to_point_aabbs)

CORE_PREFIXES = ('decoder.5.', 'prediction_heads.5.', 'x_query.', 'x_mask.', 'rel_encoder.')


def optimizer_presence_counts(bitmasks, names):
    counts = dict.fromkeys(names, 0)
    assert len(counts) == len(names)
    for bits in bitmasks:
        assert len(bits) == len(names) and set(bits).issubset({'0', '1'})
        for name, bit in zip(names, bits):
            counts[name] += int(bit)
    return counts


@torch.no_grad()
def hard_root_geometry(outputs, inputs, query_indices):
    superpoints, _ = _find_superpoint_map(outputs, inputs)
    boxes, validity = [], []
    for index, query in enumerate(query_indices):
        _, _, fused, _ = normalize_mcln_mask_logits(outputs, index, query.reshape(1))
        box, valid, _ = mask_logits_to_point_aabbs(
            inputs['point_clouds'][index, :, :3], superpoints[index].long().reshape(-1),
            fused, logit_threshold=0., quantiles=(.005,), min_points=5, max_point_fraction=.5)
        boxes.append(box[0, 0])
        validity.append(valid[0, 0])
    return torch.stack(boxes), torch.stack(validity)
