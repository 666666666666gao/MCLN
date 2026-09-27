import torch

from scripts.cs_native_root_quality import root_matched_quality_loss


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
