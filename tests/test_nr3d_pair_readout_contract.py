import torch

from scripts.nr3d_pair_readout_contract import covered_ranking_loss, decide, split_rows


def test_bad_boxes_are_never_positive_and_invalid_slots_cannot_receive_gradient():
    logits = torch.tensor([[2., 0., 100.], [1., 3., 5.]], requires_grad=True)
    ious = torch.tensor([[.8, .1, .99], [.25, .2, .1]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    loss, count = covered_ranking_loss(logits, ious, valid)
    assert count == 1
    assert torch.allclose(loss, torch.log1p(torch.exp(torch.tensor(-2.))))
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0
    assert logits.grad[0, 2] == 0 and torch.count_nonzero(logits.grad[1]) == 0


def test_no_coverage_including_empty_legal_set_gives_finite_zero_loss():
    logits = torch.tensor([[2., -float("inf")], [-float("inf"), -float("inf")]], requires_grad=True)
    valid = torch.tensor([[True, False], [False, False]])
    loss, count = covered_ranking_loss(logits, torch.zeros(2, 2), valid)
    assert count == 0 and loss == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_quality_weights_only_distribute_mass_among_qualifying_boxes():
    logits = torch.tensor([[0., 0., 0.]], requires_grad=True)
    loss, _ = covered_ranking_loss(logits, torch.tensor([[.8, .4, .2]]),
                                   torch.ones(1, 3, dtype=torch.bool))
    loss.backward()
    assert torch.allclose(logits.grad, torch.tensor([[-1/3, 0., 1/3]]), atol=1e-6)


def test_scene_split_never_separates_expressions_of_one_scene():
    rows = [{"scan_id": "scene{:04d}_00".format(i // 3)} for i in range(60)]
    parts = split_rows(rows)
    scenes = [{rows[i]["scan_id"] for i in ids} for ids in parts.values()]
    assert not scenes[0] & scenes[1]
    assert sorted(parts["fit"] + parts["holdout"]) == list(range(60))


def test_weak_control_victory_cannot_promote_a_head_that_breaks_protected_hits():
    rows = [{"raw_token_count": 15, "distractor_count": 3,
             "protected_mask_iou": .8, "scores": {
                 "global": {"box_iou": 0., "mask_iou": .2},
                 "pair": {"box_iou": .4, "mask_iou": .4},
                 "protected": {"box_iou": .8, "mask_iou": .8}}}]
    result = decide(rows)
    assert result["mechanism_screen_pass"]
    assert not result["practical_screen_pass"]
    assert not result["eligible_for_decoder_experiment"]
