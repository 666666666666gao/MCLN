import torch

from models.cs_native_root_quality import root_matched_quality_loss
from models.source_choice_adapter import compute_default_source_scores


def test_root_quality_excludes_good_unmatched_and_other_matched_queries():
    scores = torch.tensor([[0.2, 0.9, 0.3, 0.4]], requires_grad=True)
    boxes = torch.tensor([[
        [0, 0, 0, 2, 2, 2],
        [0, 0, 0, 2, 2, 2],
        [5, 5, 5, 2, 2, 2],
        [6, 6, 6, 2, 2, 2],
    ]], dtype=torch.float)
    gt = torch.tensor([[
        [0, 0, 0, 2, 2, 2],
        [6, 6, 6, 2, 2, 2],
    ]], dtype=torch.float)
    matches = [(torch.tensor([0, 3]), torch.tensor([0, 1]))]

    loss = root_matched_quality_loss(
        scores, boxes, gt, matches, ['scanrefer'], 0.1,
    )
    loss.backward()
    assert loss.item() > 0
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 2] > 0
    assert scores.grad[0, 1] == 0
    assert scores.grad[0, 3] == 0


def test_root_quality_skips_scannet_and_root_without_qualified_box():
    scores = torch.zeros(2, 2, requires_grad=True)
    boxes = torch.tensor([
        [[5, 5, 5, 2, 2, 2], [6, 6, 6, 2, 2, 2]],
        [[0, 0, 0, 2, 2, 2], [5, 5, 5, 2, 2, 2]],
    ], dtype=torch.float)
    gt = torch.tensor([
        [[0, 0, 0, 2, 2, 2]],
        [[0, 0, 0, 2, 2, 2]],
    ], dtype=torch.float)
    matches = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([0]), torch.tensor([0])),
    ]
    loss = root_matched_quality_loss(
        scores, boxes, gt, matches, ['scanrefer', 'scannet'], 0.1,
    )
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_root_quality_does_not_train_scannet_rows_in_mixed_batch():
    scores = torch.zeros(2, 2, requires_grad=True)
    boxes = torch.tensor([
        [[0, 0, 0, 2, 2, 2], [5, 5, 5, 2, 2, 2]],
        [[0, 0, 0, 2, 2, 2], [5, 5, 5, 2, 2, 2]],
    ], dtype=torch.float)
    gt = torch.tensor([
        [[0, 0, 0, 2, 2, 2]],
        [[0, 0, 0, 2, 2, 2]],
    ], dtype=torch.float)
    matches = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([0]), torch.tensor([0])),
    ]
    loss = root_matched_quality_loss(
        scores, boxes, gt, matches, ['scanrefer', 'scannet'], 0.1,
    )
    loss.backward()
    assert scores.grad[0].abs().sum().item() > 0
    assert torch.equal(scores.grad[1], torch.zeros_like(scores.grad[1]))


def test_root_quality_reaches_native_semantic_logits():
    semantic_logits = torch.tensor([[
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [2.0, 0.0, 0.0],
    ]], requires_grad=True)
    positive = torch.tensor([[[1.0, 0.0, 0.0]]])
    zeros = torch.zeros_like(positive)
    outputs = {
        'last_sem_cls_scores': semantic_logits,
        'positive_map': positive,
        'modify_positive_map': zeros,
        'pron_positive_map': zeros,
        'rel_positive_map': zeros,
        'other_entity_map': zeros,
    }
    scores = compute_default_source_scores(outputs, outputs)
    boxes = torch.tensor([[
        [0, 0, 0, 2, 2, 2],
        [5, 5, 5, 2, 2, 2],
        [0, 0, 0, 2, 2, 2],
    ]], dtype=torch.float)
    gt = boxes[:, :1]
    matches = [(torch.tensor([0]), torch.tensor([0]))]

    loss = root_matched_quality_loss(
        scores, boxes, gt, matches, ['scanrefer'], 0.1,
    )
    loss.backward()
    assert semantic_logits.grad[0, 0].abs().sum() > 0
    assert semantic_logits.grad[0, 1].abs().sum() > 0
    assert torch.equal(semantic_logits.grad[0, 2], torch.zeros(3))
