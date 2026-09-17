"""Behavioral tests for native-label replacement, not an extra ranking loss."""
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('assignment', Path(__file__).parents[1] / 'models/pvground_semantic_assignment.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    torch.manual_seed(2027)
    p = dict(last_sem_cls_scores=torch.randn(2, 5, 4, requires_grad=True),
        last_center=torch.zeros(2, 5, 3, requires_grad=True),
        last_pred_size=torch.full((2, 5, 3), 2., requires_grad=True))
    b = dict(language_dataset=['scanrefer'] * 2, box_label_mask=torch.ones(2, 2),
        center_label=torch.zeros(2, 2, 3), size_gts=torch.full((2, 2, 3), 2.))
    b['positive_map'] = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.]]] * 2)
    for key in ['modify_positive_map', 'pron_positive_map', 'rel_positive_map']:
        b[key] = torch.zeros(2, 2, 4)
    # Match a non-root first: neither target ordering nor query id is identity.
    idx = [(torch.tensor([3, 1]), torch.tensor([1, 0])),
           (torch.tensor([2, 4]), torch.tensor([0, 1]))]
    return p, b, idx


def native_ce(p, b, idx, replace=False):
    logp = p['last_sem_cls_scores'].log_softmax(-1)
    labels = torch.zeros_like(logp); labels[..., -1] = 1
    w = torch.full_like(logp[..., -1], .1)
    for bid, (queries, targets) in enumerate(idx):
        for q, t in zip(queries.tolist(), targets.tolist()):
            labels[bid, q] = (.6*b['positive_map'][bid, t] + .2*b['modify_positive_map'][bid, t]
                + .2*b['pron_positive_map'][bid, t] + .1*b['rel_positive_map'][bid, t])
            w[bid, q] = 1
    if replace:
        mask = module.qualified_unmatched(p, b, idx)
        root = .6*b['positive_map'][:, 0] + .2*b['modify_positive_map'][:, 0] + .2*b['pron_positive_map'][:, 0] + .1*b['rel_positive_map'][:, 0]
        labels[mask] = root[:, None].expand_as(labels)[mask]
    return (((labels*torch.log(labels+1e-6)-labels*logp).sum(-1))*w).sum()/4


def main():
    p, b, idx = fixture()
    p['last__loss_ce'] = native_ce(p, b, idx)
    corr, stats = module.semantic_assignment_correction(p, b, idx, .1)
    assert stats['reassigned_queries'] == 6 and stats['matched_queries'] == 4
    expected = native_ce(p, b, idx, replace=True) * (.5/7)
    actual = p['last__loss_ce']*(.5/7)+corr
    assert torch.allclose(actual, expected)
    module.verify_native_replacement(p, b, idx, .1, corr)
    grad = torch.autograd.grad(actual, p['last_sem_cls_scores'], retain_graph=True)[0]
    mask = module.qualified_unmatched(p, b, idx)
    assert bool((grad[..., 0][mask] < 0).all()) and bool((grad[..., -1][mask] > 0).all())
    # Label replacement is independent of score ranking and is root-relative.
    p, b, idx = fixture()
    p['last_center'] = torch.full((2, 5, 3), 10., requires_grad=True)
    corr, stats = module.semantic_assignment_correction(p, b, idx, .1)
    assert stats['reassigned_queries'] == 0 and float(corr) == 0
    corr.backward(); assert bool((p['last_sem_cls_scores'].grad == 0).all())
    assert p['last_center'].grad is None and p['last_pred_size'].grad is None
    # A half-volume box is exactly IoU .5, and is not a strict-qualified positive.
    p, b, idx = fixture()
    with torch.no_grad(): p['last_pred_size'][:, :, 2] = 1.
    assert not bool(module.qualified_unmatched(p, b, idx).any())
    # Preserve modifiers, pronouns and relations with the native, non-unit mass.
    p, b, idx = fixture()
    b['modify_positive_map'][:, 0, 1] = 1
    b['pron_positive_map'][:, 0, 0] = 1
    b['rel_positive_map'][:, 0, 2] = 1
    p['last__loss_ce'] = native_ce(p, b, idx)
    corr, _ = module.semantic_assignment_correction(p, b, idx, .1)
    module.verify_native_replacement(p, b, idx, .1, corr)
    assert torch.allclose(module.root_token_target(b).sum(-1), torch.full((2,), 1.1))
    print('SEMANTIC_ASSIGNMENT_CPU_BEHAVIOR_PASS 4 cases', flush=True)


if __name__ == '__main__':
    main()
