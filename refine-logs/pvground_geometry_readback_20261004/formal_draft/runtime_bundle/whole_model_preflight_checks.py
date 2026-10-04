"""Isolate the newly added whole-range loss route in the actual PV graph."""
import torch


def whole_range_loss_routes(predictions, use_whole_range):
    loss = predictions['loss_bbox'] + predictions['loss_giou']
    evidence = predictions['whole_mask_range_evidence']
    assert evidence.shape[-2:] == (256, 109) and torch.isfinite(evidence).all()
    gradient, = torch.autograd.grad(loss, (evidence,), retain_graph=True, allow_unused=True)
    if not use_whole_range:
        assert gradient is None
        return dict(evidence_shape=list(evidence.shape), range_gradient=0.,
                    global_only_mask_gradients={'text': 0., 'query': 0., 'alpha': 0.},
                    global_range_disabled=True)
    assert gradient is not None and torch.isfinite(gradient).all()
    masks = tuple(predictions['last_pred_masks'] + predictions['sp_last_pred_masks'] +
                  predictions['adaptive_weights'])
    gradients = torch.autograd.grad(evidence, masks, grad_outputs=gradient.detach(),
                                    retain_graph=True, allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    batch = len(predictions['last_pred_masks'])
    assert len(gradients) == batch * 3
    norms = {label: sum(float(g.norm()) for g in gradients[index * batch:(index + 1) * batch]
                       if g is not None)
             for index, label in enumerate(('text', 'query', 'alpha'))}
    return dict(evidence_shape=list(evidence.shape), range_gradient=float(gradient.norm()),
                global_only_mask_gradients=norms, global_range_disabled=False)


def optimizer_restore_exact(optimizer, restored):
    actual = optimizer.state_dict()
    assert actual['param_groups'] == restored['param_groups']
    assert set(actual['state']) == set(restored['state'])
    for key, reference in restored['state'].items():
        current = actual['state'][key]
        assert set(current) == set(reference)
        for name in reference:
            expected, observed = reference[name], current[name]
            if torch.is_tensor(expected):
                assert torch.equal(observed.detach().cpu(), expected.detach().cpu()), (key, name)
            else:
                assert observed == expected, (key, name)
    return dict(moment_and_step_states=len(actual['state']), param_groups=len(actual['param_groups']),
                all_keys_moments_steps_and_groups_exact=True)
