"""Small helpers for observing actual Mask gradients and seed neighborhoods."""

import torch


def query_gradient_support(gradient):
    if gradient is None:
        return {'connected': False, 'nonzero_query_ids': []}
    assert torch.isfinite(gradient).all()
    norms = gradient.detach().double().norm(dim=-1)
    return {'connected': True, 'norms': norms.tolist(),
            'nonzero_query_ids': (norms > 0).nonzero().reshape(-1).tolist()}


def gradient_connection(gradient):
    if gradient is None:
        return {'connected': False}
    assert torch.isfinite(gradient).all()
    return {'connected': True, 'norm': float(gradient.detach().double().norm()),
            'nonzero_elements': int((gradient != 0).sum()), 'elements': gradient.numel()}


def paired_query_gradients(mask_gradient, grounding_gradient):
    assert mask_gradient.shape == grounding_gradient.shape
    mask = mask_gradient.detach().double()
    grounding = grounding_gradient.detach().double()
    mask_norm = mask.norm(dim=-1)
    grounding_norm = grounding.norm(dim=-1)
    paired = (mask_norm > 0) & (grounding_norm > 0)
    cosine = (mask[paired] * grounding[paired]).sum(-1) / (mask_norm[paired] * grounding_norm[paired])
    return {'mask_norms': mask_norm.tolist(), 'grounding_norms': grounding_norm.tolist(),
            'paired_query_ids': paired.nonzero().reshape(-1).tolist(),
            'cosines': cosine.tolist(), 'negative_cosine_count': int((cosine < 0).sum())}


def superpoint_neighborhood_counts(superpoints, target_mask, seed_indices, neighbor_indices, slots):
    counts = torch.bincount(superpoints, minlength=slots)
    positive = torch.bincount(superpoints[target_mask], minlength=slots)
    target_seed_centers = target_mask[seed_indices]
    neighbor_target = target_seed_centers[neighbor_indices.long()]
    target_neighbors = neighbor_target.sum(-1)
    touched = positive > 0
    majority = positive * 2 > counts
    return {'target_points': int(target_mask.sum()), 'target_seed_centers': int(target_seed_centers.sum()),
            'slots': slots, 'occupied_slots': int((counts > 0).sum()),
            'target_touching_slots': int(touched.sum()), 'majority_positive_slots': int(majority.sum()),
            'target_touching_slots_without_target_seed_center': int((touched & (target_neighbors == 0)).sum()),
            'majority_slots_without_target_seed_center': int((majority & (target_neighbors == 0)).sum()),
            'majority_target_neighbor_counts': target_neighbors[majority].tolist(),
            'neighbor_centers_measure_receptive_field_coverage': False}
