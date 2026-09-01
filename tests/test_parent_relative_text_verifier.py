import torch

from models.losses import _expand_parent_relative_target_rows
from models.parent_relative_text_verifier import (
    ParentRelativeTextVerifier,
    _empirical_binary_loss,
    apply_discrete_parent_relative_selection,
    build_counterfactual_parent_views,
    build_parent_relative_detector_valid,
    build_parent_relative_text_verifier_batch,
    compute_parent_relative_text_verifier_loss,
    prepare_counterfactual_parent_score_axis,
)


def test_counterfactual_parent_score_axis_is_a_leaf_for_frozen_parent_scores():
    frozen_parent_scores = torch.tensor([[0.8, 0.2]], requires_grad=False)

    audit_axis = prepare_counterfactual_parent_score_axis(
        frozen_parent_scores,
        module_training=True,
        counterfactual_training=True,
    )
    assert audit_axis is not frozen_parent_scores
    assert audit_axis.is_leaf
    assert audit_axis.requires_grad
    assert torch.equal(audit_axis, frozen_parent_scores)

    legacy_axis = prepare_counterfactual_parent_score_axis(
        frozen_parent_scores,
        module_training=False,
        counterfactual_training=True,
    )
    assert legacy_axis is frozen_parent_scores

    default_off_axis = prepare_counterfactual_parent_score_axis(
        frozen_parent_scores,
        module_training=True,
        counterfactual_training=False,
    )
    assert default_off_axis is frozen_parent_scores


def _fixture(relation_required=False, anchor_mass=0.8,
             overlap_sources=False, candidate_valid=None,
             counterfactual_training=False):
    torch.manual_seed(7)
    batch_size, query_count = 2, 6
    query_dim, base_dim = 8, 12
    parent_scores = torch.tensor([
        [0.90, 0.82, 0.78, 0.40, 0.20, 0.10],
        [0.70, 0.68, 0.45, 0.30, 0.20, 0.10],
    ])
    contrastive_scores = torch.tensor([
        [0.20, 0.10, 0.95, 0.50, 0.30, 0.00],
        [0.10, 0.80, 0.20, 0.70, 0.30, 0.00],
    ])
    if overlap_sources:
        contrastive_scores = parent_scores.clone()
    if candidate_valid is None:
        candidate_valid = torch.ones_like(parent_scores, dtype=torch.bool)
    full_state = {
        "features": torch.randn(batch_size, query_count, base_dim),
        "boxes": torch.cat((
            torch.randn(batch_size, query_count, 3),
            torch.rand(batch_size, query_count, 3) + 0.5,
        ), dim=-1),
        "default_scores": parent_scores.clone(),
        "contrastive_scores": contrastive_scores,
        "num_queries": query_count,
        "schema_version": "fixture",
        "feature_names": ["f{}".format(i) for i in range(base_dim)],
    }
    query_features = torch.randn(
        batch_size, query_count, query_dim, requires_grad=True
    )
    slot_mask = torch.zeros(batch_size, 2, dtype=torch.bool)
    if relation_required:
        slot_mask[:, 0] = True
    slots = {
        "global_slot": torch.randn(batch_size, query_dim),
        "target_slot": torch.randn(batch_size, query_dim),
        "attr_slot": torch.randn(batch_size, query_dim),
        "rel_slots": torch.randn(batch_size, 2, query_dim),
        "anchor_slots": torch.randn(batch_size, 2, query_dim),
        "slot_mask": slot_mask,
        "parse_confidence": torch.ones(batch_size),
        "coverage_stats": {
            "has_target": torch.ones(batch_size, dtype=torch.bool),
        },
    }
    sacr = {
        "structured_scores": torch.randn(batch_size, query_count),
        "target_attr_scores": torch.randn(batch_size, query_count),
        "relation_anchor_scores": torch.randn(batch_size, query_count),
        "relation_geometry_signatures": torch.randn(
            batch_size, query_count, 11
        ),
        "relation_candidate_mask": torch.ones(
            batch_size, query_count, dtype=torch.bool
        ),
        "structured_valid_mask": torch.ones(
            batch_size, dtype=torch.bool
        ),
        "anchor_top1_mass": torch.full((batch_size,), anchor_mass),
        "anchor_entropy": torch.full((batch_size,), 0.2),
        "relation_active_ratio_per_sample": slot_mask.float().mean(dim=1),
    }
    batch = build_parent_relative_text_verifier_batch(
        full_state=full_state,
        query_features=query_features,
        parent_scores=parent_scores,
        candidate_valid=candidate_valid,
        slot_dict=slots,
        sacr_outputs=sacr,
        topk_per_source=2,
        max_candidates=4,
    )
    module = ParentRelativeTextVerifier(
        query_dim=query_dim,
        base_feature_dim=base_dim,
        slot_dim=query_dim,
        hidden_dim=16,
        num_heads=4,
        dropout=0.0,
        max_parent_score_gap=0.25,
        promotion_margin=1e-4,
        min_parse_confidence=0.5,
        min_anchor_mass=0.5,
        detach_inputs=True,
        counterfactual_training=counterfactual_training,
    )
    return module, batch, query_features, parent_scores


def _force_positive_predictions(module):
    with torch.no_grad():
        module.action_head.weight.zero_()
        module.action_head.bias.fill_(4.0)
        module.repair_head.weight.zero_()
        module.repair_head.bias.copy_(torch.tensor([4.0, -4.0]))
        module.break_head.weight.zero_()
        module.break_head.bias.fill_(-4.0)


def test_candidate_union_contains_parent_and_text_candidate():
    _, batch, _, _ = _fixture()
    assert torch.equal(batch["parent_query_index"], torch.tensor([0, 0]))
    assert torch.equal(batch["query_indices"][:, 0], torch.tensor([0, 0]))
    assert 2 in batch["query_indices"][0].tolist()
    assert 1 in batch["query_indices"][1].tolist()
    assert torch.equal(
        batch["parent_position"].sum(dim=1), torch.ones(2, dtype=torch.long)
    )


def test_candidate_union_has_no_default_rank_fillers_when_sources_overlap():
    _, batch, _, _ = _fixture(overlap_sources=True)
    assert torch.equal(
        batch["valid_mask"].sum(dim=1), torch.tensor([2, 2])
    )
    assert not batch["valid_mask"][:, 2:].any()


def test_counterfactual_parent_views_are_distinct_and_actual_feasible():
    module, batch, _, _ = _fixture()
    actual_output = module(batch)

    counterfactual = build_counterfactual_parent_views(
        batch, actual_output
    )

    assert counterfactual is not None
    assert torch.equal(
        counterfactual["counterfactual_source_rows"],
        torch.tensor([0, 0, 1]),
    )
    assert torch.equal(
        counterfactual["counterfactual_view_kind"],
        torch.tensor([0, 1, 0]),
    )
    parent_queries = counterfactual["parent_query_index"].tolist()
    assert parent_queries == [2, 1, 1]
    assert len(set(parent_queries[:2])) == 2
    for view_idx, source_row in enumerate(
            counterfactual["counterfactual_source_rows"].tolist()):
        compact_parent = int(
            counterfactual["parent_position"][view_idx].long().argmax()
        )
        assert actual_output["feasible_mask"][
            source_row, compact_parent
        ]
    actual_parent_position = int(batch["parent_position"][0].long().argmax())
    assert counterfactual["valid_mask"][0, actual_parent_position]
    assert counterfactual["valid_mask"][1, actual_parent_position]


def test_leave_one_out_view_makes_actual_parent_a_feasible_repair():
    module, batch, _, _ = _fixture(counterfactual_training=True)
    actual_output = module(batch)
    counterfactual = build_counterfactual_parent_views(
        batch, actual_output
    )

    loo_rows = (
        counterfactual["counterfactual_view_kind"] == 1
    ).nonzero(as_tuple=False).reshape(-1)
    assert loo_rows.numel() > 0
    counterfactual_output = module(counterfactual)
    for view_idx in loo_rows.tolist():
        source_row = int(
            counterfactual["counterfactual_source_rows"][view_idx].item()
        )
        actual_parent_query = int(
            batch["parent_query_index"][source_row].item()
        )
        actual_parent_position = (
            counterfactual["query_indices"][view_idx]
            == actual_parent_query
        ).nonzero(as_tuple=False).reshape(-1)
        assert actual_parent_position.numel() == 1
        actual_parent_position = int(actual_parent_position.item())
        loo_parent_position = int(
            counterfactual["parent_position"][view_idx].long().argmax().item()
        )
        assert counterfactual["default_scores"][
            view_idx, loo_parent_position
        ] == counterfactual["default_scores"][
            view_idx, actual_parent_position
        ]
        assert counterfactual_output["feasible_mask"][
            view_idx, actual_parent_position
        ]

    candidate_ious = torch.zeros_like(counterfactual["default_scores"])
    first_loo = int(loo_rows[0].item())
    first_source = int(
        counterfactual["counterfactual_source_rows"][first_loo].item()
    )
    actual_parent_query = int(batch["parent_query_index"][first_source].item())
    actual_parent_position = int((
        counterfactual["query_indices"][first_loo] == actual_parent_query
    ).nonzero(as_tuple=False).item())
    candidate_ious[first_loo, actual_parent_position] = 0.80
    result = compute_parent_relative_text_verifier_loss(
        counterfactual_output,
        candidate_ious,
        counterfactual_training=True,
    )
    assert result["stats"]["fix_pair_count"].item() > 0.0


def test_leave_one_out_view_ignores_high_scores_in_invalid_padding():
    module, batch, _, _ = _fixture(
        overlap_sources=True, counterfactual_training=True
    )
    batch = dict(batch)
    batch["default_scores"] = batch["default_scores"].clone()
    batch["default_scores"][~batch["valid_mask"]] = 10.0
    actual_output = module(batch)

    counterfactual = build_counterfactual_parent_views(
        batch, actual_output
    )
    counterfactual_output = module(counterfactual)

    assert counterfactual is not None
    assert torch.equal(
        counterfactual["counterfactual_view_kind"],
        torch.tensor([1, 1]),
    )
    for view_idx, source_row in enumerate(
            counterfactual["counterfactual_source_rows"].tolist()):
        actual_parent_query = int(
            batch["parent_query_index"][source_row].item()
        )
        actual_parent_position = int((
            (counterfactual["query_indices"][view_idx]
             == actual_parent_query)
            & counterfactual["valid_mask"][view_idx]
        ).nonzero(as_tuple=False).item())
        loo_parent_position = int(
            counterfactual["parent_position"][view_idx].long().argmax().item()
        )
        expected_parent_score = batch["default_scores"][
            source_row,
            batch["parent_position"][source_row],
        ].item()
        assert counterfactual["default_scores"][
            view_idx, loo_parent_position
        ].item() == expected_parent_score
        assert counterfactual["default_scores"][
            view_idx, actual_parent_position
        ].item() == expected_parent_score
        assert counterfactual_output["feasible_mask"][
            view_idx, actual_parent_position
        ]


def test_counterfactual_parent_views_abstain_without_actual_feasible_candidate():
    module, batch, _, _ = _fixture()
    module.max_parent_score_gap = 0.0
    actual_output = module(batch)

    counterfactual = build_counterfactual_parent_views(
        batch, actual_output
    )

    assert counterfactual is None


def test_text_parent_view_uses_true_top1_without_feasible_fallback():
    module, batch, _, _ = _fixture()
    module.max_parent_score_gap = 0.10
    actual_output = module(batch)

    counterfactual = build_counterfactual_parent_views(
        batch, actual_output
    )

    assert counterfactual is not None
    assert not (
        (counterfactual["counterfactual_source_rows"] == 0)
        & (counterfactual["counterfactual_view_kind"] == 0)
    ).any()
    assert torch.equal(
        counterfactual["counterfactual_view_kind"],
        torch.tensor([1, 0]),
    )
    assert torch.equal(
        counterfactual["parent_query_index"], torch.tensor([1, 1])
    )


def test_actual_text_top1_does_not_create_a_fake_text_view():
    module, batch, _, _ = _fixture(overlap_sources=True)
    actual_output = module(batch)

    counterfactual = build_counterfactual_parent_views(
        batch, actual_output
    )

    assert counterfactual is not None
    assert torch.equal(
        counterfactual["counterfactual_view_kind"],
        torch.tensor([1, 1]),
    )


def test_counterfactual_head_is_absent_when_feature_is_disabled():
    legacy_module, _, _, _ = _fixture(counterfactual_training=False)
    counterfactual_module, _, _, _ = _fixture(
        counterfactual_training=True
    )

    assert legacy_module.transition_utility_head is None
    assert not any(
        key.startswith("transition_utility_head.")
        for key in legacy_module.state_dict()
    )
    assert counterfactual_module.transition_utility_head is not None


def test_legacy_loss_keeps_all_valid_non_parent_auxiliary_supervision():
    module, batch, _, _ = _fixture(
        relation_required=True, anchor_mass=0.2,
        counterfactual_training=False,
    )
    output = module(batch)
    assert not output["deterministic_reliable_rows"].any()
    candidate_ious = torch.tensor([
        [0.10, 0.60, 0.20, 0.05],
        [0.60, 0.40, 0.15, 0.05],
    ])

    result = compute_parent_relative_text_verifier_loss(
        output, candidate_ious, counterfactual_training=False
    )

    assert result["stats"]["repair_loss"].item() > 0.0
    assert result["stats"]["break_loss"].item() > 0.0
    assert "transition_utility_loss" not in result["stats"]
    assert "fix_pair_count" not in result["stats"]


def test_counterfactual_target_rows_follow_expanded_source_identity():
    source_rows = torch.tensor([0, 0, 1], dtype=torch.long)
    end_points = {
        "center_label": torch.tensor([
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
        ]),
        "size_gts": torch.tensor([
            [[0.5, 0.6, 0.7]],
            [[0.8, 0.9, 1.0]],
        ]),
        "box_label_mask": torch.tensor([[1], [1]]),
        "unrelated": torch.tensor([9.0]),
    }

    expanded = _expand_parent_relative_target_rows(
        end_points, source_rows, actual_batch_size=2
    )

    assert torch.equal(
        expanded["center_label"][:, 0, 0],
        torch.tensor([1.0, 1.0, 4.0]),
    )
    assert torch.equal(
        expanded["size_gts"][:, 0, 0],
        torch.tensor([0.5, 0.5, 0.8]),
    )
    assert expanded["box_label_mask"].shape == (3, 1)
    assert expanded["unrelated"] is end_points["unrelated"]


def test_formal_detector_filter_matches_shared_evaluator_contract():
    candidate_boxes = torch.tensor([[[
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    ], [
        4.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    ]]])
    inputs = {
        "det_boxes": torch.tensor([[[
            4.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ]]]),
        "det_bbox_label_mask": torch.tensor([[True]]),
    }

    valid = build_parent_relative_detector_valid(candidate_boxes, inputs)

    assert torch.equal(valid, torch.tensor([[False, True]]))


def test_raw_top1_and_text_top1_outside_detector_axis_are_never_candidates():
    candidate_valid = torch.tensor([
        [False, True, False, True, False, False],
        [False, True, True, False, False, False],
    ])
    module, batch, _, _ = _fixture(candidate_valid=candidate_valid)
    _force_positive_predictions(module)
    output = module(batch)

    assert torch.equal(batch["parent_query_index"], torch.tensor([1, 1]))
    for row in range(candidate_valid.shape[0]):
        selected = batch["query_indices"][row, batch["valid_mask"][row]]
        assert candidate_valid[row, selected].all()
    assert candidate_valid[
        torch.arange(candidate_valid.shape[0]),
        output["selected_query_indices"],
    ].all()


def test_no_detector_valid_candidate_is_finite_differentiable_fallback():
    candidate_valid = torch.ones(2, 6, dtype=torch.bool)
    candidate_valid[0] = False
    module, batch, _, _ = _fixture(candidate_valid=candidate_valid)
    _force_positive_predictions(module)
    output = module(batch)

    assert not batch["deployable_rows"][0]
    assert not batch["input_valid_rows"][0]
    assert not batch["detector_valid_mask"][0].any()
    assert not output["switch_mask"][0]
    result = compute_parent_relative_text_verifier_loss(
        output, torch.zeros_like(batch["default_scores"])
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert module.action_head.bias.grad is not None
    assert torch.isfinite(module.action_head.bias.grad).all()


def test_candidate_union_rejects_capacity_that_can_truncate_topk_union():
    _, batch, query_features, parent_scores = _fixture()
    full_state = {
        key: value
        for key, value in batch.items()
        if key in (
            "features", "boxes", "default_scores", "contrastive_scores",
            "num_queries", "schema_version", "feature_names",
        )
    }
    full_state["num_queries"] = 4
    full_state["features"] = batch["features"]
    full_state["boxes"] = batch["boxes"]
    full_state["default_scores"] = batch["default_scores"]
    full_state["contrastive_scores"] = batch["contrastive_scores"]
    slots = {
        "global_slot": batch["language_context"][:, 0],
        "target_slot": batch["language_context"][:, 1],
        "attr_slot": batch["language_context"][:, 2],
        "rel_slots": batch["language_context"][:, 3:4],
        "anchor_slots": batch["language_context"][:, 4:5],
        "slot_mask": torch.ones(2, 1, dtype=torch.bool),
        "parse_confidence": batch["parse_confidence"],
        "coverage_stats": {"has_target": batch["has_target"]},
    }
    with torch.no_grad():
        try:
            build_parent_relative_text_verifier_batch(
                full_state,
                query_features[:, :4],
                parent_scores[:, :4],
                torch.ones_like(parent_scores[:, :4], dtype=torch.bool),
                slots,
                sacr_outputs=None,
                topk_per_source=2,
                max_candidates=3,
            )
        except ValueError as error:
            assert "complete Parent/Text Top-K union" in str(error)
        else:
            raise AssertionError("truncating candidate capacity was accepted")


def test_initialization_is_exact_parent_fallback():
    module, batch, _, parent_scores = _fixture()
    output = module(batch)
    assert not output["switch_mask"].any()
    assert torch.equal(
        output["selected_query_indices"], batch["parent_query_index"]
    )
    refined = apply_discrete_parent_relative_selection(
        parent_scores,
        output["selected_query_indices"],
        output["parent_query_indices"],
        output["switch_mask"],
    )
    assert torch.equal(refined, parent_scores)


def test_positive_candidate_switch_changes_only_selected_score():
    module, batch, _, parent_scores = _fixture()
    _force_positive_predictions(module)
    output = module(batch)
    assert output["switch_mask"].all()
    refined = apply_discrete_parent_relative_selection(
        parent_scores,
        output["selected_query_indices"],
        output["parent_query_indices"],
        output["switch_mask"],
    )
    rows = torch.arange(parent_scores.shape[0])
    assert torch.equal(
        refined[rows, output["parent_query_indices"]],
        parent_scores[rows, output["parent_query_indices"]],
    )
    assert torch.equal(refined.argmax(dim=1), output["selected_query_indices"])
    changed = refined != parent_scores
    assert torch.equal(changed.sum(dim=1), torch.ones(2, dtype=torch.long))


def test_relation_without_reliable_anchor_abstains():
    module, batch, _, _ = _fixture(
        relation_required=True, anchor_mass=0.2
    )
    _force_positive_predictions(module)
    output = module(batch)
    assert not output["deterministic_reliable_rows"].any()
    assert not output["switch_mask"].any()


def test_infeasible_score_gap_cannot_switch():
    module, batch, _, _ = _fixture()
    module.max_parent_score_gap = 0.01
    _force_positive_predictions(module)
    output = module(batch)
    assert not output["feasible_mask"].any()
    assert not output["switch_mask"].any()


def test_absolute_break_risk_vetoes_relative_repair_win():
    module, batch, _, _ = _fixture()
    with torch.no_grad():
        module.action_head.weight.zero_()
        module.action_head.bias.fill_(4.0)
        module.repair_head.weight.zero_()
        module.repair_head.bias.copy_(torch.tensor([4.0, -4.0]))
        module.break_head.weight.zero_()
        # sigmoid(4) > sigmoid(3), so v1 called this safe despite an
        # absolute predicted break risk above 0.5.
        module.break_head.bias.fill_(3.0)

    output = module(batch)

    assert output["predicted_repair_mask"].any()
    assert not output["predicted_no_break_mask"].any()
    assert not output["eligible_mask"].any()
    assert not output["switch_mask"].any()


def test_action_equal_to_fixed_fallback_abstains():
    module, batch, _, _ = _fixture()
    _force_positive_predictions(module)
    with torch.no_grad():
        module.action_head.bias.zero_()

    output = module(batch)

    assert output["eligible_mask"].any()
    assert not output["switch_mask"].any()


def test_repair_probability_is_auxiliary_not_a_deployment_veto():
    module, batch, _, _ = _fixture()
    _force_positive_predictions(module)
    with torch.no_grad():
        module.repair_head.bias.zero_()

    output = module(batch)

    assert not output["predicted_repair_mask"].any()
    assert output["eligible_mask"].any()
    assert output["switch_mask"].all()


def test_probability_gated_binary_loss_preserves_empirical_prior():
    logits = torch.full((1, 4), 2.0)
    targets = torch.tensor([[True, False, False, False]])
    valid = torch.ones_like(targets)

    actual = _empirical_binary_loss(logits, targets, valid)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="mean"
    )

    assert torch.allclose(actual, expected)


def test_loss_is_finite_and_detaches_backbone_inputs():
    module, batch, query_features, _ = _fixture()
    output = module(batch)
    candidate_ious = torch.tensor([
        [0.10, 0.60, 0.20, 0.05],
        [0.10, 0.55, 0.15, 0.05],
    ])
    result = compute_parent_relative_text_verifier_loss(
        output, candidate_ious
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert query_features.grad is None
    assert module.action_head.bias.grad is not None
    assert torch.isfinite(module.action_head.bias.grad).all()
    assert result["stats"]["positive_row_ratio"].item() > 0.0
    assert result["stats"]["fallback_ratio"].item() == 1.0


def test_transition_utility_distinguishes_fix_break_and_neutral_pairs():
    module, batch, _, _ = _fixture(counterfactual_training=True)
    output = module(batch)
    output["transition_utility"].retain_grad()
    candidate_ious = torch.tensor([
        [0.80, 0.10, 0.70, 0.05],
        [0.10, 0.80, 0.15, 0.05],
    ])

    result = compute_parent_relative_text_verifier_loss(
        output, candidate_ious, counterfactual_training=True
    )
    result["loss"].backward()

    assert result["stats"]["utility_positive_pair_ratio"].item() > 0.0
    assert result["stats"]["utility_negative_pair_ratio"].item() > 0.0
    assert result["stats"]["utility_neutral_pair_ratio"].item() > 0.0
    assert result["stats"]["fix_pair_count"].item() > 0.0
    assert result["stats"]["break_pair_count"].item() > 0.0
    assert result["stats"]["neutral_pair_count"].item() > 0.0
    assert result["stats"]["nonfinite_count"].item() == 0.0
    assert output["transition_utility"].grad is not None
    assert output["transition_utility"].grad.abs().sum().item() > 0.0


def test_actual_and_counterfactual_score_axes_receive_finite_gradients():
    module, batch, _, _ = _fixture(counterfactual_training=True)
    module.detach_inputs = False
    batch = dict(batch)
    batch["default_scores"] = (
        batch["default_scores"].detach().requires_grad_(True)
    )
    batch["default_scores"].retain_grad()
    actual_output = module(batch)
    counterfactual_batch = build_counterfactual_parent_views(
        batch, actual_output
    )
    assert counterfactual_batch is not None
    counterfactual_batch["default_scores"].retain_grad()
    counterfactual_output = module(counterfactual_batch)
    actual_ious = torch.tensor([
        [0.10, 0.60, 0.20, 0.05],
        [0.60, 0.40, 0.15, 0.05],
    ])
    counterfactual_ious = actual_ious.index_select(
        0, counterfactual_batch["counterfactual_source_rows"]
    )
    actual_loss = compute_parent_relative_text_verifier_loss(
        actual_output,
        actual_ious,
        counterfactual_training=True,
    )["loss"]
    counterfactual_loss = compute_parent_relative_text_verifier_loss(
        counterfactual_output,
        counterfactual_ious,
        counterfactual_training=True,
    )["loss"]

    (0.5 * (actual_loss + counterfactual_loss)).backward()

    for score_axis in (
            batch["default_scores"],
            counterfactual_batch["default_scores"]):
        assert score_axis.grad is not None
        assert torch.isfinite(score_axis.grad).all()
        assert score_axis.grad.abs().sum().item() > 0.0


def test_no_non_parent_candidate_falls_back_with_finite_loss():
    module, batch, _, _ = _fixture()
    candidate_count = batch["valid_mask"].shape[1]
    trimmed = {}
    for key, value in batch.items():
        if (
                isinstance(value, torch.Tensor)
                and value.dim() >= 2
                and value.shape[:2] == (2, candidate_count)):
            trimmed[key] = value[:, :1]
        else:
            trimmed[key] = value
    output = module(trimmed)
    assert not output["non_parent_mask"].any()
    assert not output["switch_mask"].any()
    result = compute_parent_relative_text_verifier_loss(
        output, torch.zeros(2, 1)
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert module.action_head.bias.grad is not None
    assert torch.isfinite(module.action_head.bias.grad).all()


def test_padded_and_nonfinite_rows_abstain_with_finite_loss():
    module, batch, _, _ = _fixture(overlap_sources=True)
    _force_positive_predictions(module)
    corrupt = dict(batch)
    corrupt["query_features"] = batch["query_features"].clone()
    corrupt["query_features"][0, 0, 0] = float("nan")
    output = module(corrupt)
    assert not output["input_valid_rows"][0]
    assert not output["switch_mask"][0]
    assert not output["eligible_mask"][~batch["valid_mask"]].any()
    candidate_ious = torch.zeros_like(batch["default_scores"])
    candidate_ious[1, 0] = float("nan")
    result = compute_parent_relative_text_verifier_loss(
        output, candidate_ious
    )
    assert torch.isfinite(result["loss"])
    assert all(
        bool(torch.isfinite(value).item())
        for value in result["stats"].values()
    )


def test_nonfinite_parent_row_is_preserved_and_cannot_switch():
    module, batch, _, parent_scores = _fixture()
    _force_positive_predictions(module)
    invalid_scores = parent_scores.clone()
    invalid_scores[0, 0] = float("nan")
    invalid_batch = dict(batch)
    invalid_batch["input_valid_rows"] = torch.tensor([False, True])
    output = module(invalid_batch)
    output["switch_mask"][0] = False
    output["selected_query_indices"][0] = output[
        "parent_query_indices"
    ][0]
    refined = apply_discrete_parent_relative_selection(
        invalid_scores,
        output["selected_query_indices"],
        output["parent_query_indices"],
        output["switch_mask"],
    )
    assert torch.isnan(refined[0, 0])
    assert torch.equal(refined[0, 1:], invalid_scores[0, 1:])


def test_loss_excludes_detection_only_rows_and_handles_empty_batch():
    module, batch, _, _ = _fixture()
    output = module(batch)
    candidate_ious = torch.tensor([
        [0.10, 0.60, 0.20, 0.05],
        [0.10, 0.55, 0.15, 0.05],
    ])
    partial = compute_parent_relative_text_verifier_loss(
        output,
        candidate_ious,
        sample_mask=torch.tensor([True, False]),
    )
    assert torch.isfinite(partial["loss"])
    assert partial["stats"]["positive_row_ratio"].item() == 1.0

    empty = compute_parent_relative_text_verifier_loss(
        output,
        candidate_ious,
        sample_mask=torch.tensor([False, False]),
    )
    assert torch.isfinite(empty["loss"])
    assert empty["loss"].item() == 0.0
    assert all(torch.isfinite(value) for value in empty["stats"].values())
    empty["loss"].backward()
    assert module.action_head.bias.grad is not None
    assert torch.isfinite(module.action_head.bias.grad).all()
