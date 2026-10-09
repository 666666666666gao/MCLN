"""Counterfactual heads on one frozen parent; deployment uses the PV forward."""
import torch


def apply_corrected_query_masks(model, parent, corrected_masks, raw_points):
    predictions = dict(parent)
    predictions['sp_last_pred_masks'] = corrected_masks
    # Geometry stays frozen and its hard reference is detached. Only the native
    # Query/fused Mask objective trains the corrected logits kept in this dict.
    with torch.no_grad():
        center, size = model.candidate_box_refiner(parent['support_geometry_query'], raw_points,
            parent['native_coarse_center'], parent['native_coarse_size'], predictions)
    predictions['last_center'], predictions['last_pred_size'] = center, size
    # R's output is identically zero. The single parent semantic-head invocation
    # is therefore also the deployed score for either corrected-support arm.
    assert torch.count_nonzero(model.boundary_evidence_readback.output.weight) == 0
    assert torch.count_nonzero(model.boundary_evidence_readback.output.bias) == 0
    predictions['last_sem_cls_scores'] = parent['last_sem_cls_scores'].detach()
    return predictions


def support_pair(model, heads, parent, raw_points):
    assert model.candidate_support_corrector is None
    assert not any(parameter.requires_grad for parameter in model.parameters())
    content_masks, geometry = heads['content'](parent['support_query_features'],
        parent['support_super_features'], raw_points, parent['native_coarse_center'],
        parent['native_coarse_size'], parent)
    prior_masks = []
    for bid in range(len(raw_points)):
        prior_masks.append(heads['selected_query'].correct_one(parent['support_query_features'][bid],
            parent['support_super_features'][bid], parent['native_coarse_center'][bid],
            parent['native_coarse_size'][bid], parent['last_pred_masks'][bid][0],
            parent['sp_last_pred_masks'][bid], parent['adaptive_weights'][bid], geometry[bid]))
    outputs = dict(content=apply_corrected_query_masks(model, parent, content_masks, raw_points),
        selected_query=apply_corrected_query_masks(model, parent, prior_masks, raw_points))
    for value in outputs.values():
        assert torch.equal(value['last_sem_cls_scores'], parent['last_sem_cls_scores'])
        assert all(torch.equal(a, b) for a, b in zip(value['last_pred_masks'], parent['last_pred_masks']))
        assert all(torch.equal(a, b) for a, b in zip(value['adaptive_weights'], parent['adaptive_weights']))
    return outputs
