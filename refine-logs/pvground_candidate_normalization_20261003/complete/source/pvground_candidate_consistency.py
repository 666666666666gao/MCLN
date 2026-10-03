"""Training-only final contrastive target replacement for G-qualified queries.

The same semantic correspondence is expanded and the entire expanded final
contrastive term uses its expanded correspondence count. Native Hungarian
regression, Mask responsibilities, CE replacement and inference are unchanged.
"""
import torch

from pvground_semantic_assignment import qualified_unmatched


def candidate_consistency_correction(predictions, batch, indices, set_criterion):
    assert all(name == 'scanrefer' for name in batch['language_dataset'])
    selected = qualified_unmatched(predictions, batch, indices)
    expanded = []
    targets = []
    map_names = ('positive_map', 'modify_positive_map', 'pron_positive_map',
                 'other_entity_map', 'rel_positive_map')
    for bid, (queries, matched_targets) in enumerate(indices):
        valid = batch['box_label_mask'][bid].bool()
        assert bool(valid[0])
        additional = selected[bid].nonzero(as_tuple=False).flatten().to(queries.device)
        expanded.append((torch.cat((queries, additional)),
                         torch.cat((matched_targets, torch.zeros_like(additional)))))
        targets.append({name: batch[name][bid][valid] for name in map_names})
    outputs = {name: predictions[name] for name in ('proj_tokens', 'tokenized')}
    outputs['proj_queries'] = predictions['last_proj_queries']
    num_boxes = sum(len(queries) for queries, _ in indices)
    old = set_criterion.loss_sem_align(outputs, targets, indices, num_boxes, None)['loss_sem_align']
    actual = predictions['last__loss_sem_align']
    assert torch.allclose(old, actual, rtol=1e-6, atol=1e-6)
    expanded_count = sum(len(queries) for queries, _ in expanded)
    new = set_criterion.loss_sem_align(outputs, targets, expanded, expanded_count, None)['loss_sem_align']
    # Use the same weight and seven-head averaging as the native PV criterion.
    correction = (new - old) * (.5 / 7)
    assert bool(torch.isfinite(correction))
    return correction, dict(contrastive_reassigned_queries=int(selected.sum()),
                           contrastive_reassigned_samples=int(selected.any(-1).sum()),
                           contrastive_native=float(old), contrastive_replaced=float(new),
                           contrastive_reconstruction_error=float((old - actual).abs()),
                           contrastive_native_denominator=num_boxes,
                           contrastive_expanded_denominator=expanded_count,
                           semantic_target_only=False,
                           expanded_count_normalization=True), selected


def verify_consistency_replacement(predictions, batch, indices, set_criterion, correction, selected):
    """Witness against native loss and protect the geometry/matching boundary."""
    old = predictions['last__loss_sem_align'] * (.5 / 7)
    corrected = old + correction
    repeated, _, repeated_selected = candidate_consistency_correction(predictions, batch, indices, set_criterion)
    assert torch.equal(selected, repeated_selected)
    assert torch.allclose(correction, repeated, rtol=1e-6, atol=1e-6)
    projection = predictions['last_proj_queries']
    gradient = torch.autograd.grad(corrected, projection, retain_graph=True)[0]
    assert bool(torch.isfinite(gradient).all())
    geometry = torch.autograd.grad(correction,
        (predictions['last_center'], predictions['last_pred_size']),
        retain_graph=True, allow_unused=True)
    assert all(value is None for value in geometry)
    return dict(replacement_finite=True, qualified_queries=int(selected.sum()),
                contrastive_projection_gradient_norm=float(gradient.norm()),
                direct_geometry_gradient_absent=True, native_regression_matches_unchanged=True)
