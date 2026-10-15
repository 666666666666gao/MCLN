"""Original-matching geometry supervision of an explicitly configured endpoint."""
import json
from pathlib import Path

import torch


SUPERVISION_ENDPOINT = json.loads(
    Path(__file__).with_name('matched_endpoint_config.json').read_text()
)['endpoint']
assert SUPERVISION_ENDPOINT in ('coarse', 'final')


def matched_endpoint_geometry_loss(predictions, targets, indices, criterion,
                                   num_decoder_layers, endpoint):
    center_key, size_key = {
        'coarse': ('native_coarse_center', 'native_coarse_size'),
        'final': ('last_center', 'last_pred_size'),
    }[endpoint]
    boxes = torch.cat((predictions[center_key], predictions[size_key]), dim=-1)
    matched_count = sum(len(target_indices) for _, target_indices in indices)
    assert matched_count > 0
    num_boxes = torch.as_tensor([matched_count], dtype=torch.float, device=boxes.device)
    losses = criterion.loss_boxes({'pred_boxes': boxes}, targets, indices, num_boxes, None)
    extra = (10 * losses['loss_bbox'] + 2 * losses['loss_giou']) / (num_decoder_layers + 1)
    return extra, dict(endpoint=endpoint, matched_pairs=matched_count)
