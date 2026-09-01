import torch

from models.relation_counterfactual_auxiliary import (
    build_relation_predicate_masks,
    compute_relation_counterfactual_auxiliary_loss,
    resolve_train_only_relation_anchors,
)


def _box(x, size=1.0):
    return [float(x), 0.0, 0.0, size, size, size]


def _xyz_box(x, y, z, size=1.0):
    return [float(x), float(y), float(z), size, size, size]


def test_conservative_resolver_keeps_all_non_target_anchor_class_instances():
    scene_boxes = torch.tensor([[
        _box(5.0), _box(0.0), _box(1.0), _box(8.0)
    ]])
    scene_classes = torch.tensor([[1, 2, 2, 3]])
    scene_valid = torch.ones(1, 4, dtype=torch.bool)
    gt_boxes = torch.tensor([[_box(5.0), _box(0.0)]])
    gt_valid = torch.tensor([[True, False]])

    default = resolve_train_only_relation_anchors(
        pseudo_anchor_boxes=torch.tensor([[_box(0.0)]]),
        gt_boxes=gt_boxes,
        gt_valid=gt_valid,
        scene_boxes=scene_boxes,
        scene_class_ids=scene_classes,
        scene_valid=scene_valid,
        target_ids=torch.tensor([0]),
        sample_datasets=["nr3d"],
    )
    assert not bool(default["reliable_mask"].item())

    conservative = resolve_train_only_relation_anchors(
        pseudo_anchor_boxes=torch.tensor([[_box(0.0)]]),
        gt_boxes=gt_boxes,
        gt_valid=gt_valid,
        scene_boxes=scene_boxes,
        scene_class_ids=scene_classes,
        scene_valid=scene_valid,
        target_ids=torch.tensor([0]),
        sample_datasets=["nr3d"],
        conservative_anchor_set=True,
    )
    assert bool(conservative["reliable_mask"].item())
    assert conservative["anchor_valid_mask"].tolist() == [
        [False, True, True, False]
    ]
    assert conservative["anchor_candidate_count_mean"].item() == 2.0
    assert conservative["conservative_anchor_set_ratio"].item() == 1.0
    assert conservative["conservative_row_mask"].tolist() == [True]


def test_relation_negative_must_disagree_with_every_plausible_anchor():
    anchors = torch.tensor([[_box(0.0), _box(1.0)]])
    valid = torch.tensor([[True, True]])
    candidates = torch.tensor([[
        _box(-2.0),  # left of both: robustly wrong
        _box(0.5),   # right of one and left of the other: ambiguous
        _box(3.0),   # right of both: consistent
    ]])
    result = build_relation_predicate_masks(
        candidate_boxes=candidates,
        target_boxes=torch.tensor([_box(5.0)]),
        anchor_boxes=anchors,
        anchor_valid=valid,
        relation_labels=["on the right of"],
        predicate_margin=0.08,
        conservative_rows=torch.tensor([True]),
    )
    assert result["reference_valid_mask"].tolist() == [True]
    assert result["reference_anchor_count"].tolist() == [2.0]
    assert result["ambiguous_reference_mask"].tolist() == [True]
    assert result["inconsistent_mask"].tolist() == [[True, False, False]]


def test_candidate_is_not_compared_against_itself_as_the_only_anchor():
    # Conservative resolver tensors retain the scene axis even when only one
    # anchor hypothesis is valid. The second slot is deliberately invalid.
    anchor = torch.tensor([[_box(0.0), _box(100.0)]])
    result = build_relation_predicate_masks(
        candidate_boxes=torch.tensor([[_box(0.0)]]),
        target_boxes=torch.tensor([_box(5.0)]),
        anchor_boxes=anchor,
        anchor_valid=torch.tensor([[True, False]]),
        relation_labels=["on the right of"],
        conservative_rows=torch.tensor([True]),
    )
    assert result["reference_valid_mask"].tolist() == [True]
    assert result["inconsistent_mask"].tolist() == [[False]]
    assert result["anchor_self_exclusion_ratio"].item() == 1.0


def test_anchor_set_loss_only_updates_selected_deployed_scores():
    scores = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    result = compute_relation_counterfactual_auxiliary_loss(
        deployed_scores=scores,
        candidate_boxes=torch.tensor([[
            _box(5.0), _box(-2.0), _box(3.0)
        ]]),
        candidate_valid=torch.tensor([[True, True, True]]),
        box_ious=torch.tensor([[0.8, 0.0, 0.1]]),
        target_boxes=torch.tensor([_box(5.0)]),
        anchor_boxes=torch.tensor([[_box(0.0), _box(1.0)]]),
        anchor_valid=torch.tensor([[True, True]]),
        conservative_rows=torch.tensor([True]),
        anchor_reliable=torch.tensor([True]),
        relation_labels=["on the right of"],
        target_affinity=torch.ones(1, 3),
        attribute_affinity=torch.zeros(1, 3),
        attribute_present=torch.tensor([False]),
        anchor_text_present=torch.tensor([True]),
        relation_text_present=torch.tensor([True]),
        sample_mask=torch.tensor([True]),
        parent_top_k=3,
        max_negatives=1,
    )
    assert result["loss"].item() > 0.0
    assert result["hard_negative_row_ratio"].item() == 1.0
    assert result["nonzero_loss_batch_ratio"].item() == 1.0
    assert result["violating_selected_count_mean"].item() == 1.0
    assert result["selected_score_gradient_l1"].item() == 2.0
    result["loss"].backward()
    assert scores.grad[0, 0].item() < 0.0
    assert scores.grad[0, 1].item() > 0.0
    assert scores.grad[0, 2].item() == 0.0


def test_view_dependent_ambiguous_anchors_fail_closed():
    result = build_relation_predicate_masks(
        candidate_boxes=torch.tensor([[_box(-2.0), _box(7.0)]]),
        target_boxes=torch.tensor([_box(5.0)]),
        anchor_boxes=torch.tensor([[_box(0.0), _box(10.0)]]),
        anchor_valid=torch.tensor([[True, True]]),
        conservative_rows=torch.tensor([True]),
        relation_labels=["on the right of"],
    )
    assert result["reference_valid_mask"].tolist() == [True]
    assert result["reference_anchor_count"].tolist() == [2.0]
    assert result["ambiguous_reference_mask"].tolist() == [True]
    assert result["inconsistent_mask"].tolist() == [[False, False]]


def test_view_invariant_directions_certify_only_the_required_side():
    contracts = (
        ("above", 2, 1.0),
        ("below", 2, -1.0),
    )
    target = torch.tensor([_xyz_box(0.0, 0.0, 0.0)])
    for relation, axis, expected_sign in contracts:
        correct_anchor = [0.0, 0.0, 0.0]
        wrong_anchor = [0.0, 0.0, 0.0]
        correct_anchor[axis] = -5.0 * expected_sign
        wrong_anchor[axis] = 5.0 * expected_sign
        result = build_relation_predicate_masks(
            candidate_boxes=target.unsqueeze(1),
            target_boxes=target,
            anchor_boxes=torch.tensor([[
                _xyz_box(*correct_anchor),
                _xyz_box(*wrong_anchor),
            ]]),
            anchor_valid=torch.tensor([[True, True]]),
            conservative_rows=torch.tensor([True]),
            relation_labels=[relation],
        )
        assert result["reference_valid_mask"].tolist() == [True]
        assert result["reference_anchor_count"].tolist() == [1.0]
        assert result["ambiguous_reference_mask"].tolist() == [False]


def test_view_dependent_relations_are_rotation_invariant_demonstrations():
    relations = (
        "on the left of", "on the right of", "behind", "in front of"
    )
    anchor = torch.tensor([[_xyz_box(0.0, 0.0, 0.0)]])
    horizontal = build_relation_predicate_masks(
        candidate_boxes=torch.tensor([[
            _xyz_box(-2.0, 0.0, 0.0),
            _xyz_box(3.0, 0.0, 0.0),
        ]]),
        target_boxes=torch.tensor([_xyz_box(5.0, 0.0, 0.0)]),
        anchor_boxes=anchor,
        anchor_valid=torch.tensor([[True]]),
        conservative_rows=torch.tensor([True]),
        relation_labels=[relations[0]],
    )
    rotated = build_relation_predicate_masks(
        candidate_boxes=torch.tensor([[
            _xyz_box(0.0, -2.0, 0.0),
            _xyz_box(0.0, 3.0, 0.0),
        ]]),
        target_boxes=torch.tensor([_xyz_box(0.0, 5.0, 0.0)]),
        anchor_boxes=anchor,
        anchor_valid=torch.tensor([[True]]),
        conservative_rows=torch.tensor([True]),
        relation_labels=[relations[0]],
    )
    assert horizontal["inconsistent_mask"].tolist() == [[True, False]]
    assert torch.equal(
        horizontal["inconsistent_mask"], rotated["inconsistent_mask"]
    )
    for relation in relations[1:]:
        result = build_relation_predicate_masks(
            candidate_boxes=torch.tensor([[
                _xyz_box(-2.0, 0.0, 0.0),
                _xyz_box(3.0, 0.0, 0.0),
            ]]),
            target_boxes=torch.tensor([_xyz_box(5.0, 0.0, 0.0)]),
            anchor_boxes=anchor,
            anchor_valid=torch.tensor([[True]]),
            conservative_rows=torch.tensor([True]),
            relation_labels=[relation],
        )
        assert torch.equal(
            result["inconsistent_mask"], horizontal["inconsistent_mask"]
        )


def test_resolver_rejects_multiple_parsed_pseudo_anchors():
    try:
        resolve_train_only_relation_anchors(
            pseudo_anchor_boxes=torch.tensor([[
                _box(0.0), _box(1.0)
            ]]),
            gt_boxes=torch.tensor([[_box(5.0), _box(0.0)]]),
            gt_valid=torch.tensor([[True, False]]),
            scene_boxes=torch.tensor([[_box(5.0), _box(0.0)]]),
            scene_class_ids=torch.tensor([[1, 2]]),
            scene_valid=torch.tensor([[True, True]]),
            target_ids=torch.tensor([0]),
            sample_datasets=["nr3d"],
        )
    except ValueError as error:
        assert "exactly one parsed anchor" in str(error)
    else:
        raise AssertionError("multiple parsed pseudo anchors were accepted")


def test_uncertifiable_metric_and_contact_relations_fail_closed():
    cases = (
        ("near", _xyz_box(0.1, 0.0, 0.0)),
        ("far from", _xyz_box(5.0, 0.0, 0.0)),
        ("on", _xyz_box(0.0, 0.0, 1.0)),
    )
    anchor = torch.tensor([[_xyz_box(0.0, 0.0, 0.0)]])
    for relation, target_box in cases:
        target = torch.tensor([target_box])
        conservative = build_relation_predicate_masks(
            candidate_boxes=target.unsqueeze(1),
            target_boxes=target,
            anchor_boxes=anchor,
            anchor_valid=torch.tensor([[True]]),
            conservative_rows=torch.tensor([True]),
            relation_labels=[relation],
        )
        legacy = build_relation_predicate_masks(
            candidate_boxes=target.unsqueeze(1),
            target_boxes=target,
            anchor_boxes=anchor,
            anchor_valid=torch.tensor([[True]]),
            relation_labels=[relation],
        )
        assert conservative["reference_valid_mask"].tolist() == [False]
        assert legacy["reference_valid_mask"].tolist() == [True]


def test_sr3d_exact_anchor_stays_legacy_single_anchor_when_opted_in():
    exact_anchor = _box(0.0)
    resolved = resolve_train_only_relation_anchors(
        pseudo_anchor_boxes=torch.tensor([[exact_anchor]]),
        gt_boxes=torch.tensor([[_box(5.0), exact_anchor]]),
        gt_valid=torch.tensor([[True, True]]),
        scene_boxes=torch.tensor([[
            _box(5.0), exact_anchor, _box(10.0)
        ]]),
        scene_class_ids=torch.tensor([[1, 2, 2]]),
        scene_valid=torch.tensor([[True, True, True]]),
        target_ids=torch.tensor([0]),
        sample_datasets=["sr3d"],
        conservative_anchor_set=True,
    )
    assert resolved["conservative_row_mask"].tolist() == [False]
    assert resolved["anchor_valid_mask"].tolist() == [[True, False, False]]
    predicate = build_relation_predicate_masks(
        candidate_boxes=torch.tensor([[exact_anchor]]),
        target_boxes=torch.tensor([_box(5.0)]),
        anchor_boxes=resolved["anchor_boxes"],
        anchor_valid=resolved["anchor_valid_mask"],
        conservative_rows=resolved["conservative_row_mask"],
        relation_labels=["on the right of"],
    )
    assert predicate["inconsistent_mask"].tolist() == [[True]]
    assert predicate["anchor_self_exclusion_ratio"].item() == 0.0


def test_no_relation_negative_has_zero_loss_and_zero_gradient():
    scores = torch.tensor([[0.0, 1.0]], requires_grad=True)
    result = compute_relation_counterfactual_auxiliary_loss(
        deployed_scores=scores,
        candidate_boxes=torch.tensor([[_box(5.0), _box(-2.0)]]),
        candidate_valid=torch.tensor([[True, True]]),
        box_ious=torch.tensor([[0.8, 0.0]]),
        target_boxes=torch.tensor([_box(5.0)]),
        anchor_boxes=torch.tensor([[_box(0.0)]]),
        anchor_valid=torch.tensor([[True]]),
        anchor_reliable=torch.tensor([True]),
        relation_labels=["between"],
        target_affinity=torch.ones(1, 2),
        attribute_affinity=torch.zeros(1, 2),
        attribute_present=torch.tensor([False]),
        anchor_text_present=torch.tensor([True]),
        relation_text_present=torch.tensor([True]),
        sample_mask=torch.tensor([True]),
        parent_top_k=2,
        max_negatives=1,
    )
    assert result["loss"].item() == 0.0
    assert result["nonzero_loss_batch_ratio"].item() == 0.0
    assert result["violating_selected_count_mean"].item() == 0.0
    assert result["selected_score_gradient_l1"].item() == 0.0
    result["loss"].backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))
