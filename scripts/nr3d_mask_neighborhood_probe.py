"""Two-nearest-seed intervention and measured superpoint locality."""

import torch
from torch import nn


class NearestTwoGroup(nn.Module):
    """Return the two nearest seed features and their indices, with no radius."""

    def forward(self, xyz, centers, features):
        distances = torch.cdist(centers, xyz, compute_mode='donot_use_mm_for_euclid_dist')
        indices = distances.topk(2, dim=-1, largest=False, sorted=True).indices
        gathered = features.gather(2, indices.flatten(1).unsqueeze(1).expand(-1, features.shape[1], -1))
        return gathered.reshape(features.shape[0], features.shape[1], centers.shape[1], 2), indices


def describe_neighborhoods(seed_xyz, centers, native_indices, nearest_indices,
                          superpoints, target_mask, seed_indices, radius):
    """Preserve actual IDs/distances and exclude absent integer SP slots."""
    distance2 = ((centers[:, None, :] - seed_xyz[None, :, :]) ** 2).sum(-1)
    in_radius = distance2 < radius ** 2
    empty = ~in_radius.any(-1)
    assert (native_indices[empty] == 0).all()
    assert in_radius.gather(1, native_indices)[~empty].all()
    total = torch.bincount(superpoints, minlength=len(centers))
    positive = torch.bincount(superpoints[target_mask], minlength=len(centers))
    seed_target = target_mask[seed_indices]
    native_target = seed_target[native_indices].any(-1)
    nearest_target = seed_target[nearest_indices].any(-1)
    native_distance = distance2.gather(1, native_indices).sqrt()
    nearest_distance = distance2.gather(1, nearest_indices).sqrt()
    cohorts = {'present': total > 0, 'target_bearing': positive > 0, 'majority_positive': positive * 2 > total}
    counts = {}
    for name, selected in cohorts.items():
        empty_selected = empty & selected
        counts[name] = {'slots': int(selected.sum()), 'empty_radius': int(empty_selected.sum()),
                        'native_without_target_seed': int((selected & ~native_target).sum()),
                        'nearest_without_target_seed': int((selected & ~nearest_target).sum()),
                        'target_center_restored': int((selected & ~native_target & nearest_target).sum()),
                        'target_center_lost': int((selected & native_target & ~nearest_target).sum()),
                        'empty_native_seed0_distances': native_distance[empty_selected, 0].tolist(),
                        'empty_nearest_distances': nearest_distance[empty_selected, 0].tolist()}
    return {'counts': counts, 'centroids': centers.tolist(), 'seed_xyz': seed_xyz.tolist(),
            'seed_input_indices': seed_indices.tolist(), 'seed_is_target': seed_target.tolist(),
            'point_counts': total.tolist(), 'target_point_counts': positive.tolist(),
            'native_indices': native_indices.tolist(), 'nearest_indices': nearest_indices.tolist(),
            'native_distances': native_distance.tolist(), 'nearest_distances': nearest_distance.tolist(),
            'radius_population': in_radius.sum(-1).tolist(), 'radius': radius,
            'seed_membership_is_receptive_field_coverage': False}
