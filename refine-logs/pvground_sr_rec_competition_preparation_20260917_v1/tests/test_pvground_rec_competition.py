"""CPU behavioral checks: matched identity, geometry labels, and deployment score."""
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('competition', Path(__file__).parents[1] / 'models/pvground_rec_competition.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    logits = torch.tensor([[[0., 2.], [2., 0.], [0., 1.]]], requires_grad=True)
    centers = torch.tensor([[[0., 0., 0.], [5., 0., 0.], [0., 0., 0.]]], requires_grad=True)
    sizes = torch.tensor([[[2., 2., 2.], [2., 2., 2.], [1., 1., 1.]]], requires_grad=True)
    prediction = dict(last_sem_cls_scores=logits, last_center=centers, last_pred_size=sizes)
    batch = dict(positive_map=torch.tensor([[[1., 0.]]]),
                 box_label_mask=torch.ones(1, 1), center_label=torch.zeros(1, 1, 3),
                 size_gts=torch.full((1, 1, 3), 2.))
    for name in ['modify_positive_map', 'pron_positive_map', 'rel_positive_map', 'other_entity_map']:
        batch[name] = torch.zeros(1, 1, 2)
    return prediction, batch, [(torch.tensor([0]), torch.tensor([0]))]


def main():
    p, b, match = fixture()
    value, counts = module.competition_loss(p, b, match)
    assert counts == dict(eligible25=1, active25=1, eligible50=1, active50=1)
    value.backward()
    assert p['last_sem_cls_scores'].grad[0, 0, 0] < 0
    assert p['last_sem_cls_scores'].grad[0, 1, 0] > 0
    assert p['last_center'].grad is None and p['last_pred_size'].grad is None
    # An unmatched candidate with identical good geometry must not be rewarded.
    p, b, match = fixture()
    p['last_pred_size'] = torch.full((1, 3, 3), 2., requires_grad=True)
    value, _ = module.competition_loss(p, b, match)
    value.backward()
    assert torch.equal(p['last_sem_cls_scores'].grad[0, 2], torch.zeros(2))
    # No qualifying matched root: leave regression to the native loss.
    p, b, match = fixture()
    p['last_center'] = torch.full((1, 3, 3), 10., requires_grad=True)
    value, counts = module.competition_loss(p, b, match)
    assert float(value) == 0 and counts['eligible25'] == counts['eligible50'] == 0
    value.backward()
    assert torch.equal(p['last_sem_cls_scores'].grad, torch.zeros_like(p['last_sem_cls_scores']))
    # All candidates qualify: no negative comparison exists.
    p, b, match = fixture()
    p['last_center'] = torch.zeros(1, 3, 3)
    p['last_pred_size'] = torch.full((1, 3, 3), 2.)
    value, counts = module.competition_loss(p, b, match)
    assert float(value) == 0 and counts['eligible25'] == counts['eligible50'] == 0
    # Signed deployment score retains relation and other-entity terms exactly.
    p, b, _ = fixture()
    b['rel_positive_map'][0, 0, 1] = .3
    b['other_entity_map'][0, 0, 1] = .7
    probs = p['last_sem_cls_scores'].softmax(-1)
    expected = probs[..., 0] + probs[..., 1] * .3 - probs[..., 1] * .7
    assert torch.equal(module.bbs_scores(p['last_sem_cls_scores'], b), expected)
    print('REC_COMPETITION_CPU_BEHAVIOR_PASS 5 cases', flush=True)


if __name__ == '__main__':
    main()
