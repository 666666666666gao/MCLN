"""Real-model readback checks, not yet executed or a runtime PASS."""
import torch
from torch import nn

from native_root_bbs import native_root_bbs


def observed_readback_forward(model, inputs):
    """The deployed native semantic subhead must run once after both adapters."""
    assert model.boundary_evidence_readback is not None
    semantic_head = model.prediction_heads[-1].sem_cls_scores_head
    assert not semantic_head.training
    for module in semantic_head.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.Dropout)):
            assert not module.training
    events = []
    modules = [('refiner', model.candidate_box_refiner),
               ('readback', model.boundary_evidence_readback),
               ('native_semantic_head', semantic_head)]
    handles = [module.register_forward_pre_hook(
        lambda module, args, label=label: events.append(label)) for label, module in modules]
    predictions = model(inputs)
    for handle in handles:
        handle.remove()
    assert events == ['refiner', 'readback', 'native_semantic_head'], events
    return predictions, dict(order=events, final_semantic_head_calls=1)


def fixed_geometry_and_masks(reference, predictions):
    """Compare RNG-paired native forwards; a changed semantic path cannot move them."""
    tensor_keys = ('last_center', 'last_pred_size', 'p3_coarse_center', 'p3_coarse_size',
                   'boundary_logits', 'whole_mask_range_evidence', 'last_proj_queries')
    list_keys = ('last_pred_masks', 'sp_last_pred_masks', 'adaptive_weights')
    for key in tensor_keys:
        assert torch.equal(reference[key].detach(), predictions[key].detach()), key
    for key in list_keys:
        assert len(reference[key]) == len(predictions[key])
        assert all(torch.equal(a.detach(), b.detach())
                   for a, b in zip(reference[key], predictions[key])), key
    return dict(exact_tensor_keys=list(tensor_keys), exact_list_keys=list(list_keys))


def native_bbs_witness(semantic_logits, batch):
    """Check actual per-row native reductions, all256 scores/ranks and gradients."""
    assert semantic_logits.shape[1] == 256 and semantic_logits.requires_grad
    score = native_root_bbs(semantic_logits, batch)
    probability = semantic_logits.softmax(-1)
    reference = []
    for bid in range(len(probability)):
        current = (probability[bid] * (batch['positive_map'][bid, 0] > 0)[None]).sum(-1)
        for name in ('modify_positive_map', 'pron_positive_map', 'rel_positive_map'):
            current = current + (probability[bid] * batch[name][bid, 0, None]).sum(-1)
        current = current - (probability[bid] * batch['other_entity_map'][bid, 0, None]).sum(-1)
        reference.append(current)
    reference = torch.stack(reference)
    assert torch.equal(score, reference)
    assert torch.equal(score.argsort(-1, descending=True), reference.argsort(-1, descending=True))
    observed_grad, = torch.autograd.grad(score.sum(), semantic_logits, retain_graph=True)
    native_grad, = torch.autograd.grad(reference.sum(), semantic_logits, retain_graph=True)
    assert torch.equal(observed_grad, native_grad)
    assert torch.isfinite(observed_grad).all()
    return dict(candidates=256, score_exact=True, rank_exact=True, semantic_logit_gradient_exact=True,
                semantic_logit_gradient_norm=float(observed_grad.norm()))


def readback_semantic_route(predictions, correction, readback, previous_updates):
    """Isolate native ScanRefer CE+G; frozen earlier heads have no readback route."""
    assert previous_updates in (0, 1)
    loss = predictions['last__loss_ce'] * (.5 / 7) + correction
    assert torch.isfinite(loss) and loss.requires_grad
    named = tuple(readback.named_parameters())
    gradients = torch.autograd.grad(loss, tuple(value for _, value in named), retain_graph=True)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    norms = {name: float(gradient.norm()) for (name, _), gradient in zip(named, gradients)}
    assert norms['output.weight'] > 0
    if previous_updates == 1:
        for prefix in ('query.', 'text.', 'evidence.', 'language_read.', 'face_read.'):
            assert sum(value for name, value in norms.items() if name.startswith(prefix)) > 0, prefix
    assert not predictions['last_center'].requires_grad
    assert not predictions['last_pred_size'].requires_grad
    assert not predictions['whole_mask_range_evidence'].requires_grad
    assert all(not value.requires_grad for key in ('last_pred_masks', 'sp_last_pred_masks', 'adaptive_weights')
               for value in predictions[key])
    return dict(previous_updates=previous_updates, isolated_semantic_loss=float(loss),
                parameter_gradient_norms=norms, final_box_and_masks_frozen=True)
