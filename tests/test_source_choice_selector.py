import torch

from models.source_choice_selector import (
    SourceChoiceSelector,
    compute_precision_gain_source_targets,
    compute_source_choice_loss,
)


def test_precision_gain_target_keeps_default_for_small_same_bucket_gap():
    source_ious = torch.tensor([
        [0.31, 0.32],
        [0.24, 0.31],
        [0.52, 0.49],
    ])

    target = compute_precision_gain_source_targets(
        source_ious,
        source_names=["default", "mask_text"],
        default_source="default",
        min_iou_gap=0.05,
        thresholds=(0.25, 0.5),
    )

    assert target.tolist() == [0, 1, 0]


def test_selector_forward_selects_deployable_scores_without_gt():
    torch.manual_seed(0)
    selector = SourceChoiceSelector(
        d_model=8,
        hidden_dim=16,
        source_names=["default", "mask_text"],
        text_dim=8,
    )
    candidate_feats = torch.randn(2, 4, 8)
    candidate_boxes = torch.rand(2, 4, 6)
    source_scores = {
        "default": torch.tensor([[0.5, 0.1, 0.2, 0.0], [0.1, 0.8, 0.2, 0.3]]),
        "mask_text": torch.tensor([[0.2, 0.7, 0.1, 0.0], [0.9, 0.1, 0.2, 0.3]]),
    }
    text_feats = torch.randn(2, 5, 8)
    text_mask = torch.tensor([[False, False, False, True, True], [False, False, True, True, True]])

    out = selector(
        candidate_feats=candidate_feats,
        candidate_boxes=candidate_boxes,
        source_scores=source_scores,
        text_feats=text_feats,
        text_mask=text_mask,
    )

    assert out["selector_choice_scores"].shape == (2, 2)
    assert out["selected_source_scores"].shape == (2, 4)
    assert out["selected_source_id"].shape == (2,)
    assert out["selector_choice_source_names"] == ["default", "mask_text"]
    for batch_idx, selected_id in enumerate(out["selected_source_id"].tolist()):
        name = out["selector_choice_source_names"][selected_id]
        assert torch.equal(
            out["selected_source_scores"][batch_idx],
            source_scores[name][batch_idx],
        )


def test_source_choice_loss_uses_gt_iou_targets():
    choice_scores = torch.tensor([[-1.0, 2.0], [-1.0, 2.0]], requires_grad=True)
    source_ious = torch.tensor([[0.30, 0.31], [0.20, 0.31]])

    loss, stats = compute_source_choice_loss(
        choice_scores,
        source_ious,
        source_names=["default", "mask_text"],
        default_source="default",
        target_mode="precision_gain_default_sourcewise_focal_bce",
        min_iou_gap=0.05,
    )

    assert loss.item() > 0
    assert stats["source_choice_target_non_default_ratio"].item() == 0.5
    assert stats["source_choice_selected_non_default_ratio"].item() == 1.0
    assert stats["source_choice_target_acc"].item() == 0.5
    assert "source_choice_false_override_ratio" in stats
    assert stats["source_choice_false_override_ratio"].item() == 0.5
    assert stats["source_choice_default_acc025"].item() == 0.5
    assert stats["source_choice_default_acc050"].item() == 0.0
    assert stats["source_choice_oracle_acc025"].item() == 1.0
    assert stats["source_choice_oracle_acc050"].item() == 0.0
    assert stats["source_choice_selector_fix025"].item() == 0.5
    assert stats["source_choice_selector_break025"].item() == 0.0
    assert stats["source_choice_selector_fix050"].item() == 0.0
    assert stats["source_choice_selector_break050"].item() == 0.0
    loss.backward()
    assert choice_scores.grad is not None
