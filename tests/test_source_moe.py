"""Tests for the source-level mixture-of-experts fusion module."""

import pytest
import torch
import torch.nn as nn

from models.source_moe import (
    AdaptiveSourceMixer,
    CalibratedPairwiseRiskSetActionHead,
    DenseQualityFallbackSetActionHead,
    FallbackTokenSetActionHead,
    QueryFallbackGate,
    RichFallbackTokenSetActionHead,
    SelectedCandidateAbstentionHead,
    CounterfactualSelectedRiskHead,
    SourceMoE,
    _balanced_deployment_boundary_loss,
    _counterfactual_benefit_hazard_loss,
    _counterfactual_complementary_logodds_loss,
    _counterfactual_hazard_residual_loss,
    _counterfactual_selected_risk_loss,
    _positive_candidate_top1_margin_loss,
    _empirical_setwise_action_risk_loss,
    _prior_restored_balanced_benefit_loss,
    _risk_aware_dense_quality_action_loss,
    _rowwise_boundary_calibration_loss,
    box_tier_constrained_mask_quality,
    build_risk_separated_action_target,
    build_setwise_action_target,
    compute_query_box_ious,
    compute_source_moe_fallback_gate_loss,
    compute_source_moe_ranking_loss,
    compute_load_balance_loss,
    dense_quality_expected_score,
    rank_normalize,
    standardize_source_scores,
    straight_through_rank_normalize,
    threshold_anchor_ranking_loss,
    threshold_transition_targets,
    transition_logits_expected_utility,
)


SOURCES = ("default", "contrastive_text", "mask_text")


def _v23_source_moe_pair():
    common = dict(
        source_names=SOURCES,
        shared_source="default",
        d_model=8,
        hidden_dim=16,
        text_dim=8,
        top_k=1,
        use_fallback_gate=True,
        gate_hidden_dim=8,
        gate_candidate_top_k=3,
        gate_context_layers=0,
        gate_context_heads=2,
        gate_context_dropout=0.0,
    )
    v19 = SourceMoE(
        **common,
        gate_action_mode="cascade_opportunity_verified_correction",
    )
    v23 = SourceMoE(
        **common,
        gate_action_mode="cascade_v23_dense_quality_correction",
    )
    missing, unexpected = v23.load_state_dict(v19.state_dict(), strict=False)
    assert not unexpected
    assert len(missing) == 39
    assert all(name.startswith((
        "adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
    )) for name in missing)
    generator = torch.Generator().manual_seed(2301)
    batch_size, num_queries, feature_dim = 2, 5, 8
    inputs = {
        "candidate_feats": torch.randn(
            batch_size, num_queries, feature_dim, generator=generator
        ),
        "candidate_boxes": torch.rand(
            batch_size, num_queries, 6, generator=generator
        ) + 0.1,
        "source_scores": {
            name: torch.randn(
                batch_size, num_queries, generator=generator
            )
            for name in SOURCES
        },
        "valid_mask": torch.ones(
            batch_size, num_queries, dtype=torch.bool
        ),
        "source_validity": torch.ones(
            batch_size, num_queries, len(SOURCES), dtype=torch.bool
        ),
        "text_feats": torch.randn(
            batch_size, 4, feature_dim, generator=generator
        ),
        "text_mask": torch.zeros(batch_size, 4, dtype=torch.bool),
        "gate_rich_candidate_feats": torch.randn(
            batch_size, num_queries, 2 * feature_dim + 24,
            generator=generator,
        ),
    }
    return v19, v23, inputs


def _v25_source_moe_pair():
    v19, _, inputs = _v23_source_moe_pair()
    v25 = SourceMoE(
        source_names=SOURCES,
        shared_source="default",
        d_model=8,
        hidden_dim=16,
        text_dim=8,
        top_k=1,
        use_fallback_gate=True,
        gate_hidden_dim=8,
        gate_candidate_top_k=3,
        gate_context_layers=0,
        gate_context_heads=2,
        gate_context_dropout=0.0,
        gate_action_mode="cascade_v25_pairwise_calibrated_correction",
    )
    missing, unexpected = v25.load_state_dict(
        v19.state_dict(), strict=False
    )
    assert not unexpected
    assert len(missing) == 69
    assert all(name.startswith((
        "adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
    )) for name in missing)
    return v19, v25, inputs


def _v26_source_moe_pair():
    v19, _, inputs = _v23_source_moe_pair()
    v26 = SourceMoE(
        source_names=SOURCES,
        shared_source="default",
        d_model=8,
        hidden_dim=16,
        text_dim=8,
        top_k=1,
        use_fallback_gate=True,
        gate_hidden_dim=8,
        gate_candidate_top_k=3,
        gate_context_layers=0,
        gate_context_heads=2,
        gate_context_dropout=0.0,
        gate_action_mode="cascade_v26_prior_restored_pairwise_correction",
    )
    missing, unexpected = v26.load_state_dict(
        v19.state_dict(), strict=False
    )
    assert not unexpected
    assert len(missing) == 69
    assert all(name.startswith((
        "adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
    )) for name in missing)
    return v19, v26, inputs


def _v28_source_moe_pair():
    v19, _, inputs = _v23_source_moe_pair()
    v28 = SourceMoE(
        source_names=SOURCES,
        shared_source="default",
        d_model=8,
        hidden_dim=16,
        text_dim=8,
        top_k=1,
        use_fallback_gate=True,
        gate_hidden_dim=8,
        gate_candidate_top_k=3,
        gate_context_layers=0,
        gate_context_heads=2,
        gate_context_dropout=0.0,
        gate_action_mode="cascade_v28_selected_abstention_correction",
    )
    missing, unexpected = v28.load_state_dict(
        v19.state_dict(), strict=False
    )
    assert not unexpected
    assert len(missing) == 75
    assert all(name.startswith((
        "adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
        "fallback_gate.cascade_selected_abstention_head.",
    )) for name in missing)
    return v19, v28, inputs


def _v29_source_moe_pair():
    v19, _, inputs = _v23_source_moe_pair()
    v29 = SourceMoE(
        source_names=SOURCES,
        shared_source="default",
        d_model=8,
        hidden_dim=16,
        text_dim=8,
        top_k=1,
        use_fallback_gate=True,
        gate_hidden_dim=8,
        gate_candidate_top_k=3,
        gate_context_layers=0,
        gate_context_heads=2,
        gate_context_dropout=0.0,
        gate_action_mode="cascade_v29_counterfactual_selected_correction",
    )
    missing, unexpected = v29.load_state_dict(
        v19.state_dict(), strict=False
    )
    assert not unexpected
    assert len(missing) == 75
    assert all(name.startswith((
        "adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
        "fallback_gate.cascade_counterfactual_selected_risk_head.",
    )) for name in missing)
    return v19, v29, inputs


def _v37_source_moe_pair():
    v19, _, inputs = _v23_source_moe_pair()
    v37 = SourceMoE(
        source_names=SOURCES,
        shared_source="default",
        d_model=8,
        hidden_dim=16,
        text_dim=8,
        top_k=1,
        use_fallback_gate=True,
        gate_hidden_dim=8,
        gate_candidate_top_k=3,
        gate_context_layers=0,
        gate_context_heads=2,
        gate_context_dropout=0.0,
        gate_action_mode=(
            "cascade_v37_counterfactual_benefit_hazard_correction"
        ),
    )
    missing, unexpected = v37.load_state_dict(
        v19.state_dict(), strict=False
    )
    assert not unexpected
    assert len(missing) == 75
    assert all(name.startswith((
        "adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
        "fallback_gate.cascade_counterfactual_benefit_hazard_head.",
    )) for name in missing)
    return v19, v37, inputs


def _v38_source_moe_pair():
    v19, _, inputs = _v23_source_moe_pair()
    v38 = SourceMoE(
        source_names=SOURCES,
        shared_source="default",
        d_model=8,
        hidden_dim=16,
        text_dim=8,
        top_k=1,
        use_fallback_gate=True,
        gate_hidden_dim=8,
        gate_candidate_top_k=3,
        gate_context_layers=0,
        gate_context_heads=2,
        gate_context_dropout=0.0,
        gate_action_mode="cascade_v38_complementary_logodds_correction",
    )
    missing, unexpected = v38.load_state_dict(
        v19.state_dict(), strict=False
    )
    assert not unexpected
    assert len(missing) == 75
    assert all(name.startswith((
        "adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
        "fallback_gate.cascade_counterfactual_logodds_head.",
    )) for name in missing)
    return v19, v38, inputs


def _v39_source_moe_pair():
    v19, _, inputs = _v23_source_moe_pair()
    v39 = SourceMoE(
        source_names=SOURCES,
        shared_source="default",
        d_model=8,
        hidden_dim=16,
        text_dim=8,
        top_k=1,
        use_fallback_gate=True,
        gate_hidden_dim=8,
        gate_candidate_top_k=3,
        gate_context_layers=0,
        gate_context_heads=2,
        gate_context_dropout=0.0,
        gate_action_mode="cascade_v39_hazard_residual_correction",
    )
    missing, unexpected = v39.load_state_dict(
        v19.state_dict(), strict=False
    )
    assert not unexpected
    assert len(missing) == 75
    assert all(name.startswith((
        "adaptive_source_mixer.",
        "fallback_gate.cascade_dense_quality_set_head.",
        "fallback_gate.cascade_pairwise_calibrated_set_head.",
        "fallback_gate.cascade_counterfactual_hazard_residual_head.",
    )) for name in missing)
    return v19, v39, inputs


def _batch(batch_size=2, num_queries=8, feat_dim=64, text_len=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "candidate_feats": torch.randn(
            batch_size, num_queries, feat_dim, generator=generator
        ),
        "candidate_boxes": torch.rand(
            batch_size, num_queries, 6, generator=generator
        ),
        "source_scores": {
            name: torch.randn(batch_size, num_queries, generator=generator)
            for name in SOURCES
        },
        "text_feats": torch.randn(
            batch_size, text_len, feat_dim, generator=generator
        ),
        "text_mask": torch.zeros(batch_size, text_len, dtype=torch.bool),
    }


def _module(**kwargs):
    kwargs.setdefault("source_names", SOURCES)
    kwargs.setdefault("d_model", 64)
    kwargs.setdefault("text_dim", 64)
    return SourceMoE(**kwargs)


def test_rank_normalize_maps_to_unit_interval_by_descending_rank():
    scores = torch.tensor([[3.0, 1.0, 2.0]])
    normalized = rank_normalize(scores)
    assert torch.allclose(normalized, torch.tensor([[1.0, 0.0, 0.5]]))


def test_rank_normalize_is_scale_invariant():
    scores = torch.tensor([[0.001, 0.002, 0.003]])
    scaled = scores * 10_000.0
    assert torch.allclose(rank_normalize(scores), rank_normalize(scaled))


def test_transition_logits_expected_utility_matches_metric_values():
    logits = torch.tensor([[[[0.0, 0.0, 20.0], [0.0, 0.0, 20.0]]]])
    weights = torch.tensor([2.0, 1.0])
    utility = transition_logits_expected_utility(logits, weights, 2.0)
    assert utility.item() == pytest.approx(1.0, abs=1e-5)

    neutral = torch.zeros_like(logits)
    neutral_utility = transition_logits_expected_utility(
        neutral, weights, 2.0
    )
    assert neutral_utility.item() == pytest.approx(-1.0 / 3.0, abs=1e-6)


def test_source_score_standardization_uses_only_valid_queries_per_scene():
    scores = torch.tensor([
        [1.0, 2.0, 3.0, 1000.0],
        [5.0, 5.0, 5.0, float("nan")],
    ])
    valid_mask = torch.tensor([
        [True, True, True, False],
        [True, True, True, False],
    ])

    standardized = standardize_source_scores(scores, valid_mask)

    assert torch.allclose(
        standardized[:, :3].mean(dim=1), torch.zeros(2), atol=1e-6
    )
    assert standardized[0, :3].square().mean() == pytest.approx(
        1.0, abs=3e-5
    )
    assert torch.equal(standardized[1], torch.zeros(4))
    assert torch.equal(standardized[:, 3], torch.zeros(2))


def test_initializes_exactly_to_shared_source():
    """The zero-init routed scale must make the module a no-op at step 0.

    This is the architectural guarantee that adding the MoE cannot regress the
    protected baseline before any training happens.
    """
    module = _module()
    batch = _batch()
    out = module(**batch)
    expected = rank_normalize(batch["source_scores"]["default"])
    assert torch.allclose(out["selected_source_scores"], expected, atol=1e-6)


def test_initial_gate_breaks_topk_ties_without_changing_baseline_output():
    torch.manual_seed(19)
    module = _module(top_k=1)
    batch = _batch(batch_size=2, num_queries=128, seed=29)

    out = module(**batch)

    usage = out["moe_expert_mask"].sum(dim=(0, 1))
    assert torch.all(usage > 0)
    expected = rank_normalize(batch["source_scores"]["default"])
    assert torch.equal(out["selected_source_scores"], expected)


def test_degenerates_to_shared_source_when_routed_scale_is_zeroed():
    module = _module()
    with torch.no_grad():
        for param in module.router.parameters():
            param.normal_()
        module.routed_scale.zero_()
    batch = _batch(seed=3)
    out = module(**batch)
    expected = rank_normalize(batch["source_scores"]["default"])
    assert torch.allclose(out["selected_source_scores"], expected, atol=1e-6)


def test_output_shape_matches_candidate_axis():
    module = _module()
    batch = _batch(batch_size=3, num_queries=16)
    out = module(**batch)
    assert out["selected_source_scores"].shape == (3, 16)


def test_routing_is_per_query_not_per_sample():
    """Routing decisions must be able to differ across queries in one sample."""
    module = _module(top_k=1)
    with torch.no_grad():
        for param in module.router.parameters():
            param.zero_()
        module.router[0].weight[0, 0] = 1.0
        module.router[0].weight[1, 0] = -1.0
        module.router[-1].weight[0, 0] = 1.0
        module.router[-1].weight[1, 1] = 1.0
    batch = _batch(num_queries=32, seed=5)
    batch["candidate_feats"][0, :, 0] = torch.where(
        torch.arange(32) % 2 == 0,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )
    out = module(**batch)
    per_sample_choice = out["moe_expert_mask"][0].argmax(dim=-1)
    assert per_sample_choice.unique().numel() > 1


def test_top_k_activates_exactly_k_routed_experts_per_query():
    module = _module(top_k=1)
    batch = _batch()
    out = module(**batch)
    assert torch.all(out["moe_expert_mask"].sum(dim=-1) == 1)


def test_gradient_flows_to_candidate_features():
    """Distinguishes this module from the non-differentiable argmax selector."""
    module = _module()
    with torch.no_grad():
        module.routed_scale.fill_(1.0)
        for param in module.router.parameters():
            param.normal_()
    batch = _batch(seed=7)
    batch["candidate_feats"].requires_grad_(True)
    out = module(**batch)
    out["selected_source_scores"].sum().backward()
    grad = batch["candidate_feats"].grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_gradient_reaches_router_through_straight_through_estimator():
    module = _module()
    with torch.no_grad():
        module.routed_scale.fill_(1.0)
    batch = _batch(seed=11)
    out = module(**batch)
    out["selected_source_scores"].sum().backward()
    router_grad = module.router[-1].weight.grad
    assert router_grad is not None
    assert router_grad.abs().sum() > 0


def test_balance_loss_is_minimized_by_uniform_load():
    uniform_probs = torch.full((1, 8, 4), 0.25)
    uniform_mask = torch.zeros(1, 8, 4)
    uniform_mask[:, 0::4, 0] = 1.0
    uniform_mask[:, 1::4, 1] = 1.0
    uniform_mask[:, 2::4, 2] = 1.0
    uniform_mask[:, 3::4, 3] = 1.0

    collapsed_probs = torch.zeros(1, 8, 4)
    collapsed_probs[..., 0] = 1.0
    collapsed_mask = torch.zeros(1, 8, 4)
    collapsed_mask[..., 0] = 1.0

    uniform = compute_load_balance_loss(uniform_probs, uniform_mask)
    collapsed = compute_load_balance_loss(collapsed_probs, collapsed_mask)
    assert uniform < collapsed
    assert pytest.approx(1.0, abs=1e-6) == float(uniform)
    assert pytest.approx(4.0, abs=1e-6) == float(collapsed)


def test_top2_balance_loss_is_normalized_to_one_at_uniform_load():
    probabilities = torch.full((1, 8, 4), 0.25)
    dispatch = torch.zeros_like(probabilities)
    dispatch[:, 0::2, 0:2] = 1.0
    dispatch[:, 1::2, 2:4] = 1.0
    assert float(compute_load_balance_loss(
        probabilities, dispatch
    )) == pytest.approx(1.0, abs=1e-6)


def test_balance_loss_penalizes_router_collapse_end_to_end():
    module = _module(top_k=1)
    with torch.no_grad():
        module.router[-1].bias.copy_(torch.tensor([10.0, -10.0]))
    batch = _batch(seed=13)
    out = module(**batch)
    usage = out["moe_expert_usage_contrastive_text"]
    assert float(usage) == pytest.approx(1.0, abs=1e-6)
    assert float(out["moe_balance_loss"]) > 1.5


def test_invalid_queries_are_excluded_from_output():
    module = _module()
    batch = _batch(num_queries=8)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[:, 4:] = False
    out = module(valid_mask=valid, **batch)
    scores = out["selected_source_scores"]
    assert torch.all(scores[:, 4:] <= -1e3)
    assert scores.argmax(dim=1).max() < 4


def test_missing_source_raises():
    module = _module()
    batch = _batch()
    del batch["source_scores"]["mask_text"]
    with pytest.raises(KeyError):
        module(**batch)


def test_wrong_source_shape_raises():
    module = _module()
    batch = _batch()
    batch["source_scores"]["mask_text"] = torch.randn(2, 99)
    with pytest.raises(ValueError):
        module(**batch)


def test_shared_source_must_be_listed():
    with pytest.raises(ValueError):
        SourceMoE(source_names=SOURCES, shared_source="not_a_source")


def test_duplicate_sources_rejected():
    with pytest.raises(ValueError):
        SourceMoE(source_names=("default", "default"))


def test_single_source_degenerates_without_router():
    module = SourceMoE(source_names=("default",), d_model=64, text_dim=64)
    batch = _batch()
    batch["source_scores"] = {"default": batch["source_scores"]["default"]}
    out = module(**batch)
    expected = rank_normalize(batch["source_scores"]["default"])
    assert torch.allclose(out["selected_source_scores"], expected, atol=1e-6)
    assert float(out["moe_balance_loss"]) == 0.0


def test_non_finite_source_scores_are_sanitized():
    module = _module()
    batch = _batch()
    batch["source_scores"]["default"][0, 0] = float("nan")
    batch["source_scores"]["contrastive_text"][0, 1] = float("inf")
    out = module(**batch)
    assert torch.isfinite(out["selected_source_scores"]).all()


def test_straight_through_rank_has_exact_forward_values_and_score_gradients():
    scores = torch.tensor([[3.0, 1.0, 2.0]], requires_grad=True)
    normalized = straight_through_rank_normalize(scores)
    assert torch.equal(normalized.detach(), rank_normalize(scores.detach()))
    normalized[:, 0].sum().backward()
    assert scores.grad is not None
    assert float(scores.grad.abs().sum()) > 0.0


def test_query_box_iou_targets_identify_best_candidate():
    boxes = torch.tensor([[[
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0
    ], [
        2.0, 0.0, 0.0, 1.0, 1.0, 1.0
    ]]])
    targets = boxes[:, 1:2].clone()
    ious = compute_query_box_ious(
        boxes, targets, torch.ones(1, 1, dtype=torch.bool)
    )
    assert torch.allclose(ious, torch.tensor([[0.0, 1.0]]))


def test_mask_quality_cannot_override_a_higher_box_threshold_tier():
    quality = box_tier_constrained_mask_quality(
        box_ious=torch.tensor([[0.51, 0.10]]),
        mask_ious=torch.tensor([[0.01, 1.00]]),
    )
    assert quality[0, 0] > quality[0, 1]


def test_anchor_loss_is_zero_when_correct_shared_query_has_safe_margin():
    loss = threshold_anchor_ranking_loss(
        scores=torch.tensor([[1.0, 0.8, 0.7]]),
        box_ious=torch.tensor([[0.6, 0.1, 0.2]]),
        anchor_indices=torch.tensor([0]),
    )
    assert float(loss) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "scores,ious",
    [
        ([[0.8, 0.9]], [[0.6, 0.1]]),
        ([[0.8, 0.7]], [[0.1, 0.6]]),
    ],
)
def test_anchor_loss_penalizes_breaks_and_missing_fixes(scores, ious):
    scores = torch.tensor(scores, requires_grad=True)
    loss = threshold_anchor_ranking_loss(
        scores=scores,
        box_ious=torch.tensor(ious),
        anchor_indices=torch.tensor([0]),
        margin=0.05,
    )
    assert float(loss) == pytest.approx(0.15, abs=1e-6)
    loss.backward()
    assert scores.grad is not None
    assert float(scores.grad.abs().sum()) > 0.0


def test_anchor_fix_must_outrank_every_incorrect_query():
    scores = torch.tensor([[0.8, 0.9, 1.2]], requires_grad=True)
    loss = threshold_anchor_ranking_loss(
        scores=scores,
        box_ious=torch.tensor([[0.1, 0.6, 0.2]]),
        anchor_indices=torch.tensor([0]),
        margin=0.05,
    )
    assert float(loss) == pytest.approx(0.35, abs=1e-6)
    loss.backward()
    assert scores.grad is not None
    assert float(scores.grad[0, 0]) == pytest.approx(0.0)
    assert float(scores.grad[0, 1]) < 0.0
    assert float(scores.grad[0, 2]) > 0.0


def test_ranking_loss_includes_weighted_anchor_protection():
    boxes = torch.tensor([[
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [3.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    ]])
    scores = torch.tensor([[0.8, 0.9]], requires_grad=True)
    ranking = compute_source_moe_ranking_loss(
        scores=scores,
        candidate_boxes=boxes,
        gt_boxes=boxes[:, :1],
        gt_valid=torch.ones(1, 1, dtype=torch.bool),
        anchor_indices=torch.tensor([0]),
        anchor_loss_weight=2.0,
        anchor_margin=0.05,
    )
    assert float(ranking["anchor_loss"]) == pytest.approx(0.15, abs=1e-6)
    expected = ranking["box_loss"] + 2.0 * ranking["anchor_loss"]
    assert torch.allclose(ranking["loss"], expected)


def test_real_ranking_loss_trains_residual_scale_reranker_and_shared_source():
    module = _module(hidden_dim=64, query_heads=4, top_k=1)
    batch = _batch(batch_size=2, num_queries=6, seed=23)
    for scores in batch["source_scores"].values():
        scores.requires_grad_(True)
    boxes = torch.tensor([[[
        float(query_idx * 2), 0.0, 0.0, 1.0, 1.0, 1.0
    ] for query_idx in range(6)]]).expand(2, -1, -1).clone()
    batch["candidate_boxes"] = boxes
    outputs = module(**batch)
    targets = torch.tensor([
        [[2.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
        [[4.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
    ])
    ranking = compute_source_moe_ranking_loss(
        outputs["selected_source_scores"],
        boxes,
        targets,
        torch.ones(2, 1, dtype=torch.bool),
    )
    ranking["loss"].backward()

    assert module.routed_scale.grad is not None
    assert float(module.routed_scale.grad.abs().sum()) > 0.0
    assert module.query_reranker.score.weight.grad is not None
    assert float(module.query_reranker.score.weight.grad.abs().sum()) > 0.0
    shared_grad = batch["source_scores"]["default"].grad
    assert shared_grad is not None
    assert float(shared_grad.abs().sum()) > 0.0


def test_threshold_transition_targets_are_relative_to_default_query():
    quality = torch.tensor([
        [0.60, 0.10, 0.40],
        [0.10, 0.60, 0.40],
    ])
    targets = threshold_transition_targets(
        quality, torch.tensor([0, 0])
    )
    assert targets.shape == (2, 3, 2)
    assert torch.equal(
        targets[0],
        torch.tensor([[1, 1], [0, 0], [1, 0]]),
    )
    assert torch.equal(
        targets[1],
        torch.tensor([[1, 1], [2, 2], [2, 1]]),
    )


def test_zero_initialized_fallback_gate_is_exact_shared_identity():
    module = _module(
        use_fallback_gate=True,
        gate_candidate_top_k=4,
    )
    batch = _batch(batch_size=3, num_queries=12, seed=37)
    with torch.no_grad():
        module.routed_scale.fill_(0.8)
        module.query_reranker.score.weight.normal_()
    out = module(**batch)
    expected = rank_normalize(batch["source_scores"]["default"])
    assert torch.equal(out["selected_source_scores"], expected)
    assert not bool(out["moe_gate_switch"].any().item())
    assert not torch.equal(out["moe_candidate_scores"], expected)
    assert torch.all(out["moe_gate_candidate_mask"].sum(dim=1) <= 4)


def test_shared_query_is_single_source_of_truth_for_gate_ties():
    module = _module(use_fallback_gate=True)
    batch = _batch(batch_size=2, num_queries=8, seed=38)
    batch["source_scores"]["default"].zero_()

    out = module(**batch)

    assert torch.equal(out["moe_shared_query"], out["moe_gate_default_query"])
    assert torch.equal(
        out["moe_shared_query"],
        out["selected_source_scores"].argmax(dim=1),
    )


def test_enriched_fallback_gate_is_exact_identity_at_initialization():
    evidence_dim = 11
    module = _module(
        use_fallback_gate=True,
        gate_candidate_top_k=4,
        gate_use_evidence_features=True,
        gate_evidence_dim=evidence_dim,
    )
    batch = _batch(batch_size=3, num_queries=12, seed=39)
    batch["gate_candidate_feats"] = torch.randn(3, 12, evidence_dim)

    out = module(**batch)

    expected = rank_normalize(batch["source_scores"]["default"])
    assert torch.equal(out["selected_source_scores"], expected)
    assert not bool(out["moe_gate_switch"].any().item())
    assert module.fallback_gate.query_dim == (
        module.router_input_dim + evidence_dim + len(SOURCES)
    )


def test_contextual_fallback_gate_is_exact_identity_at_initialization():
    module = _module(
        use_fallback_gate=True,
        gate_candidate_top_k=4,
        gate_context_layers=1,
        gate_context_heads=4,
        gate_context_dropout=0.0,
    )
    batch = _batch(batch_size=3, num_queries=12, seed=43)

    out = module(**batch)

    expected = rank_normalize(batch["source_scores"]["default"])
    assert torch.equal(out["selected_source_scores"], expected)
    assert not bool(out["moe_gate_switch"].any().item())
    assert out["moe_gate_context_scale"].item() == pytest.approx(0.0)
    assert torch.equal(
        out["moe_gate_candidate_mask"].sum(dim=1),
        torch.full((3,), 4),
    )
    assert not bool(out["moe_gate_candidate_mask"][
        torch.arange(3), out["moe_gate_default_query"]
    ].any().item())
    assert any(
        "fallback_gate.context_encoder" in key
        for key in module.state_dict()
    )


def test_context_gate_single_query_has_no_alternative_and_falls_back():
    gate = QueryFallbackGate(
        query_dim=8,
        hidden_dim=16,
        candidate_top_k=8,
        context_layers=1,
        context_heads=4,
        context_dropout=0.0,
    )
    shared_scores = torch.tensor([[0.25], [0.75]])

    out = gate(
        query_features=torch.randn(2, 1, 8),
        candidate_scores=torch.tensor([[1.0], [2.0]]),
        shared_scores=shared_scores,
        valid_mask=torch.ones(2, 1, dtype=torch.bool),
    )

    assert not bool(out["moe_gate_candidate_mask"].any().item())
    assert not bool(out["moe_gate_switch"].any().item())
    assert torch.equal(
        out["moe_gate_selected_query"],
        torch.zeros(2, dtype=torch.long),
    )
    assert torch.equal(out["selected_source_scores"], shared_scores)


def test_expected_utility_gate_can_use_supervised_quality_heads():
    gate = QueryFallbackGate(
        query_dim=2,
        hidden_dim=12,
        candidate_top_k=2,
        mask_utility_weight=0.0,
        context_layers=0,
        action_mode="expected_utility",
    )
    gate.encoder = nn.Identity()
    with torch.no_grad():
        gate.box_head.weight.zero_()
        gate.box_head.bias.zero_()
        # Pair features are [query, default, delta, six score features].
        # The third score feature is candidate_score - default_score.
        for threshold_index in range(2):
            fix_index = threshold_index * 3 + 2
            gate.box_head.weight[fix_index, 8] = 5.0

    out = gate(
        query_features=torch.zeros(1, 3, 2),
        candidate_scores=torch.tensor([[0.0, 2.0, 1.0]]),
        shared_scores=torch.tensor([[3.0, 2.0, 1.0]]),
        valid_mask=torch.ones(1, 3, dtype=torch.bool),
    )

    assert torch.allclose(
        out["moe_gate_decision_margin"],
        torch.zeros_like(out["moe_gate_decision_margin"]),
    )
    assert out["moe_gate_expected_utility"][0, 1] > 0.0
    assert bool(out["moe_gate_switch"].item())
    assert out["moe_gate_selected_query"].item() == 1


def test_direct_utility_gate_starts_at_fallback_and_uses_its_scalar_head():
    gate = QueryFallbackGate(
        query_dim=2,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=0,
        action_mode="direct_utility",
    )
    inputs = {
        "query_features": torch.zeros(1, 3, 2),
        "candidate_scores": torch.tensor([[0.0, 2.0, 1.0]]),
        "shared_scores": torch.tensor([[3.0, 2.0, 1.0]]),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    initial = gate(**inputs)
    assert torch.equal(
        initial["moe_gate_direct_utility"],
        torch.zeros_like(initial["moe_gate_direct_utility"]),
    )
    assert not bool(initial["moe_gate_switch"].item())

    with torch.no_grad():
        gate.utility_head.bias.fill_(1.0)
    promoted = gate(**inputs)
    assert torch.equal(
        promoted["moe_gate_action_margin"],
        promoted["moe_gate_direct_utility"],
    )
    assert bool(promoted["moe_gate_switch"].item())
    assert promoted["moe_gate_selected_query"].item() == 1


def test_hierarchical_utility_separates_candidate_rank_from_row_switch():
    gate = QueryFallbackGate(
        query_dim=2,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=0,
        action_mode="hierarchical_utility",
    )
    inputs = {
        "query_features": torch.zeros(1, 3, 2),
        "candidate_scores": torch.tensor([[0.0, 2.0, 1.0]]),
        "shared_scores": torch.tensor([[3.0, 2.0, 1.0]]),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    initial = gate(**inputs)
    assert torch.equal(
        initial["moe_gate_row_switch_margin"],
        torch.zeros_like(initial["moe_gate_row_switch_margin"]),
    )
    assert not bool(initial["moe_gate_switch"].item())

    with torch.no_grad():
        gate.utility_head.bias.fill_(10.0)
    vetoed = gate(**inputs)
    assert bool((vetoed["moe_gate_action_margin"] > 0.0).all().item())
    assert not bool(vetoed["moe_gate_switch"].item())

    with torch.no_grad():
        gate.row_switch_head[-1].bias.fill_(1.0)
    promoted = gate(**inputs)
    assert bool(promoted["moe_gate_switch"].item())
    assert promoted["moe_gate_selected_query"].item() == 1


def test_pairwise_verifier_starts_at_fallback_and_verifies_selected_query():
    gate = QueryFallbackGate(
        query_dim=2,
        hidden_dim=12,
        candidate_top_k=1,
        context_layers=1,
        context_heads=4,
        context_dropout=0.0,
        action_mode="pairwise_verifier",
    )
    inputs = {
        "query_features": torch.tensor([[[0.0, 0.0],
                                           [1.0, 0.0],
                                           [0.0, 1.0]]]),
        "candidate_scores": torch.tensor([[0.0, 2.0, 1.0]]),
        "shared_scores": torch.tensor([[3.0, 2.0, 1.0]]),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    initial = gate(**inputs)
    assert initial["moe_gate_selected_query"].item() == 0
    assert torch.equal(
        initial["moe_gate_row_switch_margin"],
        torch.zeros_like(initial["moe_gate_row_switch_margin"]),
    )

    with torch.no_grad():
        gate.pairwise_switch_head[-1].bias.fill_(1.0)
    promoted = gate(**inputs)

    assert bool(promoted["moe_gate_switch"].item())
    assert promoted["moe_gate_selected_query"].item() == 1
    assert gate.pairwise_switch_head[0].in_features == 49


def test_topn_pairwise_verifier_scores_each_candidate_before_selection():
    class CandidateFeatureMargin(nn.Module):
        def forward(self, features):
            hidden_dim = (features.shape[-1] - 1) // 4
            return features[..., hidden_dim:hidden_dim + 1]

    gate = QueryFallbackGate(
        query_dim=2,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=0,
        action_mode="topn_pairwise_verifier",
    )
    gate.encoder = nn.Identity()
    gate.pairwise_switch_head = CandidateFeatureMargin()
    inputs = {
        "query_features": torch.tensor([[[0.0, 0.0],
                                         [1.0, 0.0],
                                         [2.0, 0.0]]]),
        "candidate_scores": torch.tensor([[0.0, 3.0, 2.0]]),
        "shared_scores": torch.tensor([[4.0, 3.0, 2.0]]),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    out = gate(**inputs)

    assert out["moe_gate_action_margin"].shape == (1, 3)
    assert out["moe_gate_row_switch_margin"].shape == (1, 3)
    assert bool(out["moe_gate_switch"].item())
    assert out["moe_gate_selected_query"].item() == 2


def test_dual_evidence_verifier_requires_benefit_and_safety_agreement():
    gate = QueryFallbackGate(
        query_dim=2,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=0,
        action_mode="topn_dual_evidence_verifier",
    )
    inputs = {
        "query_features": torch.tensor([[[0.0, 0.0],
                                         [1.0, 0.0],
                                         [2.0, 0.0]]]),
        "candidate_scores": torch.tensor([[0.0, 3.0, 2.0]]),
        "shared_scores": torch.tensor([[4.0, 3.0, 2.0]]),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    initial = gate(**inputs)
    assert not bool(initial["moe_gate_switch"].item())
    assert torch.equal(
        initial["moe_gate_row_switch_margin"],
        torch.minimum(
            initial["moe_gate_row_benefit_margin"],
            initial["moe_gate_row_safety_margin"],
        ),
    )

    with torch.no_grad():
        gate.pairwise_switch_head[-1].bias.fill_(1.0)
    benefit_only = gate(**inputs)
    assert not bool(benefit_only["moe_gate_switch"].item())

    with torch.no_grad():
        gate.safety_switch_head[-1].bias.fill_(1.0)
    agreed = gate(**inputs)
    assert bool(agreed["moe_gate_switch"].item())
    assert gate.safety_switch_head[0].in_features == 29

    with torch.no_grad():
        gate.safety_switch_head[-1].bias.fill_(-1.0)
    vetoed = gate(**inputs)
    assert not bool(vetoed["moe_gate_switch"].item())


def test_absolute_quality_delta_starts_at_fallback_and_reranks_topn():
    gate = QueryFallbackGate(
        query_dim=2,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=0,
        action_mode="topn_absolute_quality_delta",
    )
    gate.encoder = nn.Identity()
    inputs = {
        "query_features": torch.tensor([[[0.0, 0.0],
                                         [1.0, 0.0],
                                         [2.0, 0.0]]]),
        "candidate_scores": torch.tensor([[0.0, 3.0, 2.0]]),
        "shared_scores": torch.tensor([[4.0, 3.0, 2.0]]),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    initial = gate(**inputs)
    assert not bool(initial["moe_gate_switch"].item())
    assert torch.equal(
        initial["moe_gate_absolute_quality_margin"],
        torch.zeros_like(initial["moe_gate_absolute_quality_margin"]),
    )
    assert gate.absolute_quality_head.in_features == 12
    assert gate.absolute_quality_head.out_features == 6

    with torch.no_grad():
        gate.absolute_quality_head.weight[:, 0].fill_(4.0)
    reranked = gate(**inputs)

    assert bool(reranked["moe_gate_switch"].item())
    assert reranked["moe_gate_selected_query"].item() == 2
    assert reranked["moe_gate_absolute_box_logits"].shape == (1, 3, 2)
    assert reranked["moe_gate_absolute_mask_logits"].shape == (1, 3, 2)


def test_absolute_quality_forward_loss_trains_dense_quality_head():
    gate = QueryFallbackGate(
        query_dim=4,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=1,
        context_heads=4,
        context_dropout=0.0,
        action_mode="topn_absolute_quality_delta",
    )
    outputs = gate(
        query_features=torch.randn(2, 3, 4),
        candidate_scores=torch.tensor([[0.0, 2.0, 1.0],
                                       [0.0, 2.0, 1.0]]),
        shared_scores=torch.tensor([[3.0, 2.0, 1.0],
                                    [3.0, 2.0, 1.0]]),
        valid_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=outputs["moe_gate_box_logits"],
        decision_logits=outputs["moe_gate_decision_logits"],
        box_ious=torch.tensor([[0.10, 0.70, 0.20],
                               [0.70, 0.30, 0.10]]),
        default_indices=outputs["moe_gate_default_query"],
        candidate_mask=outputs["moe_gate_candidate_mask"],
        mask_logits=outputs["moe_gate_mask_logits"],
        mask_ious=torch.tensor([[0.20, 0.80, 0.10],
                                [0.80, 0.20, 0.10]]),
        action_margin=outputs["moe_gate_action_margin"],
        absolute_box_logits=outputs["moe_gate_absolute_box_logits"],
        absolute_box_iou=outputs["moe_gate_absolute_box_iou"],
        absolute_mask_logits=outputs["moe_gate_absolute_mask_logits"],
        absolute_mask_iou=outputs["moe_gate_absolute_mask_iou"],
        setwise_temperature=0.25,
        objective="topn_absolute_quality_calibrated",
    )

    assert torch.isfinite(result["loss"])
    assert result["absolute_quality_loss"].item() > 0.0
    target = result["selection_target_distribution"]
    assert target[0, 0].item() == 0.0
    assert torch.equal(target[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    result["loss"].backward()
    assert gate.absolute_quality_head.weight.grad is not None
    assert float(
        gate.absolute_quality_head.weight.grad.abs().sum()
    ) > 0.0


def _cascade_gate(utility_direction=1.0):
    gate = QueryFallbackGate(
        query_dim=2,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=0,
        action_mode="cascade_absolute_quality_correction",
    )
    gate.encoder = nn.Identity()
    with torch.no_grad():
        gate.utility_head.weight.zero_()
        gate.utility_head.weight[0, 0] = float(utility_direction)
        gate.utility_head.bias.zero_()
        gate.pairwise_switch_head[-1].weight.zero_()
        gate.pairwise_switch_head[-1].bias.fill_(1.0)
    gate.eval()
    return gate


def _cascade_inputs():
    return {
        "query_features": torch.tensor([[[0.0, 0.0],
                                         [1.0, 0.0],
                                         [2.0, 0.0]]]),
        "candidate_scores": torch.tensor([[0.0, 3.0, 2.0]]),
        "shared_scores": torch.tensor([[4.0, 3.0, 2.0]]),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }


def test_cascade_zero_initialization_exactly_preserves_v12_pairwise_output():
    pairwise = QueryFallbackGate(
        query_dim=2,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=0,
        action_mode="pairwise_verifier",
    )
    pairwise.encoder = nn.Identity()
    with torch.no_grad():
        pairwise.utility_head.weight.zero_()
        pairwise.utility_head.weight[0, 0] = 1.0
        pairwise.utility_head.bias.zero_()
        pairwise.pairwise_switch_head[-1].weight.zero_()
        pairwise.pairwise_switch_head[-1].bias.fill_(1.0)
    pairwise.eval()

    cascade = _cascade_gate()
    missing, unexpected = cascade.load_state_dict(
        pairwise.state_dict(), strict=False
    )
    assert not unexpected
    assert all(
        name.startswith((
            "absolute_quality_head.",
            "cascade_quality_adapter.",
            "cascade_correction_head.",
        ))
        for name in missing
    )

    v12 = pairwise(**_cascade_inputs())
    initial = cascade(**_cascade_inputs())

    assert v12["moe_gate_selected_query"].item() == 2
    assert torch.equal(
        initial["moe_gate_action_anchor_query"],
        v12["moe_gate_selected_query"],
    )
    assert torch.equal(
        initial["moe_gate_selected_query"],
        v12["moe_gate_selected_query"],
    )
    assert torch.equal(
        initial["selected_source_scores"],
        v12["selected_source_scores"],
    )
    assert not bool(initial["moe_gate_correction_switch"].item())


@pytest.mark.parametrize("action_mode", [
    "cascade_absolute_quality_correction",
    "cascade_opportunity_quality_correction",
    "cascade_opportunity_verified_correction",
    "cascade_joint_risk_correction",
    "cascade_v19_fallback_set_correction",
])
def test_cascade_module_construction_preserves_v12_global_rng_stream(
        action_mode):
    common = {
        "query_dim": 8,
        "hidden_dim": 12,
        "candidate_top_k": 2,
        "context_layers": 1,
        "context_heads": 4,
        "context_dropout": 0.1,
    }
    torch.manual_seed(817)
    QueryFallbackGate(action_mode="pairwise_verifier", **common)
    expected = torch.rand(16)

    torch.manual_seed(817)
    QueryFallbackGate(
        action_mode=action_mode, **common
    )
    actual = torch.rand(16)

    assert torch.equal(actual, expected)


def test_cascade_can_correct_harmful_v12_switch_back_to_shared_default():
    gate = _cascade_gate(utility_direction=1.0)
    initial = gate(**_cascade_inputs())
    assert initial["moe_gate_action_anchor_query"].item() == 2

    with torch.no_grad():
        gate.cascade_correction_head[-1].weight.zero_()
        gate.cascade_correction_head[-1].bias.fill_(1.0)
    corrected = gate(**_cascade_inputs())

    assert bool(corrected["moe_gate_cascade_base_switch"].item())
    assert bool(corrected["moe_gate_correction_switch"].item())
    assert corrected["moe_gate_selected_query"].item() == 0
    assert not bool(corrected["moe_gate_switch"].item())
    assert bool(corrected["moe_gate_candidate_mask"][0, 0].item())
    assert not bool(corrected["moe_gate_candidate_mask"][0, 2].item())
    assert torch.equal(
        corrected["selected_source_scores"],
        _cascade_inputs()["shared_scores"],
    )


def test_cascade_can_promote_better_alternative_from_dynamic_anchor():
    class CandidateCoordinateMargin(nn.Module):
        def forward(self, features):
            return features[..., 12:13]

    gate = _cascade_gate(utility_direction=-1.0)
    gate.cascade_quality_adapter = nn.Identity()
    gate.cascade_correction_head = CandidateCoordinateMargin()

    promoted = gate(**_cascade_inputs())

    assert promoted["moe_gate_action_anchor_query"].item() == 1
    assert bool(promoted["moe_gate_correction_switch"].item())
    assert promoted["moe_gate_selected_query"].item() == 2


def test_cascade_loss_builds_targets_relative_to_dynamic_v12_anchor():
    gate = _cascade_gate(utility_direction=1.0)
    outputs = gate(**_cascade_inputs())
    anchor = outputs["moe_gate_action_anchor_query"]
    assert anchor.item() == 2

    result = compute_source_moe_fallback_gate_loss(
        box_logits=outputs["moe_gate_box_logits"],
        decision_logits=outputs["moe_gate_decision_logits"],
        box_ious=torch.tensor([[0.60, 0.20, 0.10]]),
        default_indices=anchor,
        candidate_mask=outputs["moe_gate_candidate_mask"],
        action_margin=outputs["moe_gate_action_margin"],
        row_switch_margin=outputs["moe_gate_row_switch_margin"],
        absolute_box_logits=outputs["moe_gate_absolute_box_logits"],
        absolute_box_iou=outputs["moe_gate_absolute_box_iou"],
        setwise_temperature=0.25,
        objective="cascade_absolute_quality_calibrated",
    )

    assert torch.equal(result["box_targets"][0, 0], torch.tensor([2, 2]))
    assert result["decision_utility"][0, 0].item() > 0.0
    assert bool(result["active_mask"][0, 0].item())
    assert not bool(result["active_mask"][0, 2].item())
    assert torch.isfinite(result["loss"])


def _opportunity_cascade_gate(
        utility_direction=1.0,
        action_mode="cascade_opportunity_quality_correction",
        rich_feature_dim=None):
    gate = QueryFallbackGate(
        query_dim=2,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=0,
        action_mode=action_mode,
        rich_feature_dim=rich_feature_dim,
    )
    gate.encoder = nn.Identity()
    with torch.no_grad():
        gate.utility_head.weight.zero_()
        gate.utility_head.weight[0, 0] = float(utility_direction)
        gate.utility_head.bias.zero_()
        gate.pairwise_switch_head[-1].weight.zero_()
        gate.pairwise_switch_head[-1].bias.fill_(1.0)
    gate.eval()
    return gate


@pytest.mark.parametrize("action_mode", [
    "cascade_opportunity_quality_correction",
    "cascade_opportunity_verified_correction",
    "cascade_joint_risk_correction",
])
def test_opportunity_cascade_zero_initialization_preserves_v12_output(
        action_mode):
    pairwise = QueryFallbackGate(
        query_dim=2,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=0,
        action_mode="pairwise_verifier",
    )
    pairwise.encoder = nn.Identity()
    with torch.no_grad():
        pairwise.utility_head.weight.zero_()
        pairwise.utility_head.weight[0, 0] = 1.0
        pairwise.utility_head.bias.zero_()
        pairwise.pairwise_switch_head[-1].weight.zero_()
        pairwise.pairwise_switch_head[-1].bias.fill_(1.0)
    pairwise.eval()

    opportunity = _opportunity_cascade_gate(action_mode=action_mode)
    missing, unexpected = opportunity.load_state_dict(
        pairwise.state_dict(), strict=False
    )
    assert not unexpected
    assert all(
        name.startswith((
            "absolute_quality_head.",
            "cascade_quality_adapter.",
            "cascade_correction_head.",
            "cascade_opportunity_head.",
            "cascade_candidate_safety_head.",
            "cascade_joint_action_head.",
        ))
        for name in missing
    )

    v12 = pairwise(**_cascade_inputs())
    initial = opportunity(**_cascade_inputs())
    assert torch.equal(
        initial["moe_gate_action_anchor_query"],
        v12["moe_gate_selected_query"],
    )
    assert torch.equal(
        initial["moe_gate_selected_query"],
        v12["moe_gate_selected_query"],
    )
    assert torch.equal(
        initial["selected_source_scores"],
        v12["selected_source_scores"],
    )
    assert torch.equal(
        initial["moe_gate_opportunity_margin"],
        torch.zeros_like(initial["moe_gate_opportunity_margin"]),
    )
    assert not bool(initial["moe_gate_correction_switch"].item())
    if action_mode in (
            "cascade_opportunity_verified_correction",
            "cascade_joint_risk_correction"):
        assert torch.equal(
            initial["moe_gate_row_safety_margin"],
            torch.zeros_like(initial["moe_gate_row_safety_margin"]),
        )
    if action_mode == "cascade_joint_risk_correction":
        assert torch.equal(
            initial["moe_gate_joint_action_margin"],
            torch.zeros_like(initial["moe_gate_joint_action_margin"]),
        )


def test_opportunity_cascade_separates_row_gate_from_query_ranking():
    class CandidateCoordinateMargin(nn.Module):
        def forward(self, features):
            return features[..., 12:13]

    class ConstantOpportunity(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = float(value)

        def forward(self, features):
            return features.new_full((features.shape[0], 1), self.value)

    gate = _opportunity_cascade_gate(utility_direction=-1.0)
    gate.cascade_quality_adapter = nn.Identity()
    gate.cascade_correction_head = nn.Sequential(
        nn.Identity(), CandidateCoordinateMargin()
    )
    gate.cascade_opportunity_head = ConstantOpportunity(-1.0)

    vetoed = gate(**_cascade_inputs())
    assert vetoed["moe_gate_action_anchor_query"].item() == 1
    assert vetoed["moe_gate_selected_query"].item() == 1
    assert not bool(vetoed["moe_gate_correction_switch"].item())

    gate.cascade_opportunity_head = ConstantOpportunity(1.0)
    corrected = gate(**_cascade_inputs())
    assert bool(corrected["moe_gate_correction_switch"].item())
    assert corrected["moe_gate_selected_query"].item() == 2


def test_verified_opportunity_requires_selected_candidate_safety():
    class CandidateCoordinateMargin(nn.Module):
        def forward(self, features):
            return features[..., 12:13]

    class ConstantMargin(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = float(value)

        def forward(self, features):
            shape = features.shape[:-1] + (1,)
            return features.new_full(shape, self.value)

    gate = _opportunity_cascade_gate(
        utility_direction=-1.0,
        action_mode="cascade_opportunity_verified_correction",
    )
    gate.cascade_quality_adapter = nn.Identity()
    gate.cascade_correction_head = nn.Sequential(
        nn.Identity(), CandidateCoordinateMargin()
    )
    gate.cascade_opportunity_head = ConstantMargin(1.0)
    gate.cascade_candidate_safety_head = ConstantMargin(-1.0)

    vetoed = gate(**_cascade_inputs())
    assert vetoed["moe_gate_action_anchor_query"].item() == 1
    assert vetoed["moe_gate_selected_query"].item() == 1
    assert not bool(vetoed["moe_gate_correction_switch"].item())

    gate.cascade_candidate_safety_head = ConstantMargin(1.0)
    verified = gate(**_cascade_inputs())
    assert bool(verified["moe_gate_correction_switch"].item())
    assert verified["moe_gate_selected_query"].item() == 2


def test_joint_cascade_zero_init_migrates_v19_to_dynamic_anchor():
    v19 = _opportunity_cascade_gate(
        utility_direction=-1.0,
        action_mode="cascade_opportunity_verified_correction",
    )
    with torch.no_grad():
        v19.cascade_correction_head[-1].bias.fill_(1.0)
        v19.cascade_opportunity_head[-1].bias.fill_(1.0)
        v19.cascade_candidate_safety_head[-1].bias.fill_(1.0)
    v19_output = v19(**_cascade_inputs())
    assert bool(v19_output["moe_gate_correction_switch"].item())

    joint = _opportunity_cascade_gate(
        utility_direction=-1.0,
        action_mode="cascade_joint_risk_correction",
    )
    missing, unexpected = joint.load_state_dict(v19.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(name.startswith("cascade_joint_action_head.") for name in missing)

    initial = joint(**_cascade_inputs())
    assert torch.equal(
        initial["moe_gate_action_anchor_query"],
        v19_output["moe_gate_action_anchor_query"],
    )
    assert torch.equal(
        initial["moe_gate_selected_query"],
        initial["moe_gate_action_anchor_query"],
    )
    assert torch.equal(
        initial["moe_gate_joint_action_margin"],
        torch.zeros_like(initial["moe_gate_joint_action_margin"]),
    )
    assert not bool(initial["moe_gate_correction_switch"].item())


def _v19_fallback_set_pair():
    v19 = _opportunity_cascade_gate(
        utility_direction=-1.0,
        action_mode="cascade_opportunity_verified_correction",
    )
    with torch.no_grad():
        v19.cascade_correction_head[-1].bias.fill_(1.0)
        v19.cascade_opportunity_head[-1].bias.fill_(1.0)
        v19.cascade_candidate_safety_head[-1].bias.fill_(1.0)
    v19_output = v19(**_cascade_inputs())
    assert bool(v19_output["moe_gate_correction_switch"].item())

    v21 = _opportunity_cascade_gate(
        utility_direction=-1.0,
        action_mode="cascade_v19_fallback_set_correction",
    )
    missing, unexpected = v21.load_state_dict(v19.state_dict(), strict=False)
    assert not unexpected
    assert len(missing) == 15
    assert all(
        name.startswith("cascade_fallback_set_action_head.")
        for name in missing
    )
    return v19, v19_output, v21


def test_v21_zero_init_exactly_preserves_deployed_v19_output():
    _, v19_output, v21 = _v19_fallback_set_pair()

    initial = v21(**_cascade_inputs())

    assert torch.equal(
        initial["moe_gate_v19_fallback_query"],
        v19_output["moe_gate_selected_query"],
    )
    assert torch.equal(
        initial["moe_gate_selected_query"],
        v19_output["moe_gate_selected_query"],
    )
    assert torch.equal(
        initial["selected_source_scores"],
        v19_output["selected_source_scores"],
    )
    assert torch.equal(
        initial["moe_gate_joint_action_margin"],
        torch.zeros_like(initial["moe_gate_joint_action_margin"]),
    )
    assert torch.equal(
        initial["moe_gate_v19_correction_switch"],
        v19_output["moe_gate_correction_switch"],
    )
    assert not bool(initial["moe_gate_correction_switch"].item())


def test_v21_can_undo_harmful_v19_correction_to_v12_anchor():
    class SelectQuery(nn.Module):
        def __init__(self, query_index):
            super().__init__()
            self.query_index = int(query_index)

        def forward(self, action_features, fallback_indices, candidate_mask):
            margin = action_features.new_zeros(candidate_mask.shape)
            margin[:, self.query_index] = 1.0
            return {
                "margin": margin,
                "fallback_logit": margin.new_zeros(margin.shape[0]),
            }

    _, v19_output, v21 = _v19_fallback_set_pair()
    v12_anchor = v19_output["moe_gate_action_anchor_query"].item()
    assert v19_output["moe_gate_selected_query"].item() != v12_anchor
    v21.cascade_fallback_set_action_head = SelectQuery(v12_anchor)

    corrected = v21(**_cascade_inputs())
    query_ious = torch.tensor([0.10, 0.70, 0.20])

    assert bool(corrected["moe_gate_candidate_mask"][0, v12_anchor].item())
    assert bool(corrected["moe_gate_correction_switch"].item())
    assert corrected["moe_gate_selected_query"].item() == v12_anchor
    assert query_ious[v12_anchor] > query_ious[
        v19_output["moe_gate_selected_query"].item()
    ]


def test_fallback_token_set_head_is_query_permutation_equivariant():
    torch.manual_seed(911)
    head = FallbackTokenSetActionHead(
        input_dim=7,
        hidden_dim=8,
        max_candidates=3,
        num_heads=2,
        dropout=0.0,
    )
    with torch.no_grad():
        head.score.weight.normal_()
    head.eval()
    features = torch.randn(2, 5, 7)
    fallback = torch.tensor([0, 3])
    candidate_mask = torch.tensor([
        [False, True, True, False, True],
        [True, True, False, False, True],
    ])
    original = head(features, fallback, candidate_mask)

    permutation = torch.tensor([2, 4, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    permuted = head(
        features[:, permutation],
        inverse[fallback],
        candidate_mask[:, permutation],
    )

    assert torch.allclose(
        permuted["margin"], original["margin"][:, permutation], atol=1e-6
    )
    assert torch.allclose(
        permuted["fallback_logit"], original["fallback_logit"], atol=1e-6
    )


def test_v21_loss_target_is_relative_to_deployed_v19_fallback():
    _, _, v21 = _v19_fallback_set_pair()
    outputs = v21(**_cascade_inputs())
    fallback = outputs["moe_gate_supervision_fallback_query"]
    anchor = outputs["moe_gate_action_anchor_query"]
    fallback_index = fallback.item()
    anchor_index = anchor.item()
    assert fallback_index == 0
    assert anchor_index == 1

    result = compute_source_moe_fallback_gate_loss(
        box_logits=outputs["moe_gate_box_logits"],
        decision_logits=outputs["moe_gate_decision_logits"],
        box_ious=torch.tensor([[0.10, 0.70, 0.20]]),
        default_indices=fallback,
        candidate_mask=outputs["moe_gate_candidate_mask"],
        action_margin=outputs["moe_gate_action_margin"],
        joint_action_margin=outputs["moe_gate_joint_action_margin"],
        setwise_temperature=0.25,
        objective="cascade_v19_fallback_set_risk_calibrated",
    )

    assert not bool(result["active_mask"][0, fallback_index].item())
    assert bool(result["active_mask"][0, anchor_index].item())
    assert torch.equal(
        result["box_targets"][0, anchor_index], torch.tensor([2, 2])
    )
    assert result["decision_utility"][0, anchor_index].item() > 0.0
    assert torch.equal(
        result["joint_target_distribution"][0],
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
    )


def test_v21_deployment_boundary_has_positive_and_fallback_gradients():
    candidate_margin = torch.zeros(2, 3, requires_grad=True)
    result = _balanced_deployment_boundary_loss(
        candidate_margin=candidate_margin,
        decision_utility=torch.tensor([
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 0.0],
        ]),
        active_mask=torch.tensor([
            [False, True, True],
            [False, True, True],
        ]),
        sample_mask=torch.ones(2, dtype=torch.bool),
        temperature=0.25,
    )

    assert torch.equal(result["positive_rows"], torch.tensor([True, False]))
    assert torch.equal(result["fallback_rows"], torch.tensor([False, True]))
    assert result["positive_loss"].item() > 0.0
    assert result["fallback_loss"].item() > 0.0
    result["loss"].backward()
    assert candidate_margin.grad[0, 1].item() < 0.0
    assert candidate_margin.grad[1, 1].item() > 0.0
    assert candidate_margin.grad[1, 2].item() > 0.0


def _v22_rich_set_pair(rich_dim=7):
    v19, v19_output, _ = _v19_fallback_set_pair()
    v22 = _opportunity_cascade_gate(
        utility_direction=-1.0,
        action_mode="cascade_v19_rich_set_correction",
        rich_feature_dim=rich_dim,
    )
    missing, unexpected = v22.load_state_dict(v19.state_dict(), strict=False)
    assert not unexpected
    assert len(missing) == 17
    assert all(
        name.startswith("cascade_rich_fallback_set_action_head.")
        for name in missing
    )
    inputs = _cascade_inputs()
    inputs["rich_candidate_features"] = torch.randn(1, 3, rich_dim)
    return v19_output, v22, inputs


def test_v22_zero_init_exactly_preserves_deployed_v19_output():
    v19_output, v22, inputs = _v22_rich_set_pair()

    initial = v22(**inputs)

    assert torch.equal(
        initial["moe_gate_v19_fallback_query"],
        v19_output["moe_gate_selected_query"],
    )
    assert torch.equal(
        initial["moe_gate_selected_query"],
        v19_output["moe_gate_selected_query"],
    )
    assert torch.equal(
        initial["selected_source_scores"],
        v19_output["selected_source_scores"],
    )
    assert torch.equal(
        initial["moe_gate_joint_action_margin"],
        torch.zeros_like(initial["moe_gate_joint_action_margin"]),
    )


@pytest.mark.parametrize("failure", ["missing", "shape", "nonfinite"])
def test_v22_rich_evidence_fails_closed(failure):
    _, v22, inputs = _v22_rich_set_pair()
    if failure == "missing":
        inputs.pop("rich_candidate_features")
    elif failure == "shape":
        inputs["rich_candidate_features"] = torch.zeros(1, 3, 6)
    else:
        inputs["rich_candidate_features"][0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="rich_candidate_features"):
        v22(**inputs)


def test_rich_fallback_set_head_is_query_permutation_equivariant():
    torch.manual_seed(919)
    head = RichFallbackTokenSetActionHead(
        action_dim=7,
        rich_dim=5,
        hidden_dim=8,
        max_candidates=3,
        num_heads=2,
        dropout=0.0,
    )
    with torch.no_grad():
        head.set_head.score.weight.normal_()
    head.eval()
    action_features = torch.randn(2, 5, 7)
    rich_features = torch.randn(2, 5, 5)
    fallback = torch.tensor([0, 3])
    candidate_mask = torch.tensor([
        [False, True, True, False, True],
        [True, True, False, False, True],
    ])
    original = head(
        action_features, rich_features, fallback, candidate_mask
    )

    permutation = torch.tensor([2, 4, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    permuted = head(
        action_features[:, permutation],
        rich_features[:, permutation],
        inverse[fallback],
        candidate_mask[:, permutation],
    )

    assert torch.allclose(
        permuted["margin"], original["margin"][:, permutation], atol=1e-6
    )
    assert torch.allclose(
        permuted["fallback_logit"], original["fallback_logit"], atol=1e-6
    )


def test_v22_rich_branch_receives_nonzero_gradient():
    _, v22, inputs = _v22_rich_set_pair()
    with torch.no_grad():
        v22.cascade_rich_fallback_set_action_head.set_head.score.weight.normal_()
    v22.train()

    outputs = v22(**inputs)
    active_margin = outputs["moe_gate_joint_action_margin"][
        outputs["moe_gate_candidate_mask"]
    ]
    active_margin.square().sum().backward()

    rich_norm = v22.cascade_rich_fallback_set_action_head.rich_norm
    assert rich_norm.weight.grad is not None
    assert rich_norm.weight.grad.abs().sum().item() > 0.0


def test_v23_zero_init_exactly_preserves_complete_v19_source_moe_output():
    v19, v23, inputs = _v23_source_moe_pair()
    v19.eval()
    v23.eval()

    v19_inputs = dict(inputs)
    v19_inputs.pop("source_validity")
    expected = v19(**v19_inputs)
    actual = v23(**inputs)

    for key in (
            "moe_candidate_scores", "selected_source_scores",
            "moe_gate_selected_query"):
        assert torch.equal(actual[key], expected[key])
    assert torch.equal(
        actual["moe_gate_joint_action_margin"],
        torch.zeros_like(actual["moe_gate_joint_action_margin"]),
    )
    assert torch.equal(
        actual["moe_query_routed_scale"],
        torch.tanh(v23.routed_scale).expand_as(
            actual["moe_query_routed_scale"]
        ),
    )


def test_v25_zero_init_exactly_preserves_complete_v19_source_moe_output():
    v19, v25, inputs = _v25_source_moe_pair()
    v19.eval()
    v25.eval()

    v19_inputs = dict(inputs)
    v19_inputs.pop("source_validity")
    expected = v19(**v19_inputs)
    actual = v25(**inputs)

    for key in (
            "moe_candidate_scores", "selected_source_scores",
            "moe_gate_selected_query"):
        assert torch.equal(actual[key], expected[key])
    for key in (
            "moe_gate_joint_action_margin",
            "moe_gate_pairwise_calibrated_margin",
            "moe_gate_row_benefit_margin"):
        assert torch.equal(actual[key], torch.zeros_like(actual[key]))
    assert v25.fallback_gate.cascade_pairwise_calibrated_set_head\
        .utility_head.bias.requires_grad


def test_v26_zero_init_exactly_preserves_complete_v19_source_moe_output():
    v19, v26, inputs = _v26_source_moe_pair()
    v19.eval()
    v26.eval()

    v19_inputs = dict(inputs)
    v19_inputs.pop("source_validity")
    expected = v19(**v19_inputs)
    actual = v26(**inputs)

    for key in (
            "moe_candidate_scores", "selected_source_scores",
            "moe_gate_selected_query"):
        assert torch.equal(actual[key], expected[key])
    for key in (
            "moe_gate_joint_action_margin",
            "moe_gate_pairwise_utility_margin",
            "moe_gate_row_benefit_margin"):
        assert torch.equal(actual[key], torch.zeros_like(actual[key]))


def test_v28_zero_init_exactly_preserves_complete_v19_source_moe_output():
    v19, v28, inputs = _v28_source_moe_pair()
    v19.eval()
    v28.eval()

    v19_inputs = dict(inputs)
    v19_inputs.pop("source_validity")
    expected = v19(**v19_inputs)
    actual = v28(**inputs)

    for key in (
            "moe_candidate_scores", "selected_source_scores",
            "moe_gate_selected_query"):
        assert torch.equal(actual[key], expected[key])
    for key in (
            "moe_gate_joint_action_margin",
            "moe_gate_pairwise_utility_margin",
            "moe_gate_candidate_selection_margin",
            "moe_gate_selected_abstention_margin"):
        assert torch.equal(actual[key], torch.zeros_like(actual[key]))


def test_v28_selected_abstention_is_query_permutation_equivariant():
    torch.manual_seed(2801)
    head = SelectedCandidateAbstentionHead(hidden_dim=8, dropout=0.0)
    with torch.no_grad():
        head.row_head[-1].weight.normal_()
        head.row_head[-1].bias.fill_(0.2)
    head.eval()
    pair_features = torch.randn(2, 5, 8)
    selection_margin = torch.tensor([
        [0.0, 0.7, -0.3, 0.2, 0.5],
        [0.6, -0.2, 0.1, 0.8, 0.0],
    ])
    candidate_mask = torch.tensor([
        [False, True, True, False, True],
        [True, True, False, True, False],
    ])
    result = head(pair_features, selection_margin, candidate_mask)
    permutation = torch.tensor([2, 4, 0, 3, 1])
    changed = head(
        pair_features[:, permutation],
        selection_margin[:, permutation],
        candidate_mask[:, permutation],
    )

    assert torch.allclose(
        changed["margin"], result["margin"][:, permutation], atol=1e-6
    )
    assert torch.allclose(
        changed["row_margin"], result["row_margin"], atol=1e-6
    )
    deployed_max = result["margin"].masked_fill(
        ~candidate_mask, -1e4
    ).max(dim=1).values
    assert torch.allclose(deployed_max, result["row_margin"], atol=1e-6)


def test_v29_zero_init_exactly_preserves_complete_v19_source_moe_output():
    v19, v29, inputs = _v29_source_moe_pair()
    v19.eval()
    v29.eval()

    v19_inputs = dict(inputs)
    v19_inputs.pop("source_validity")
    expected = v19(**v19_inputs)
    actual = v29(**inputs)

    for key in (
            "moe_candidate_scores", "selected_source_scores",
            "moe_gate_selected_query"):
        assert torch.equal(actual[key], expected[key])
    for key in (
            "moe_gate_joint_action_margin",
            "moe_gate_pairwise_utility_margin",
            "moe_gate_candidate_selection_margin",
            "moe_gate_selected_abstention_margin",
            "moe_gate_counterfactual_risk_margin"):
        assert torch.equal(actual[key], torch.zeros_like(actual[key]))


def test_v37_zero_init_exactly_preserves_complete_v19_source_moe_output():
    v19, v37, inputs = _v37_source_moe_pair()
    v19.eval()
    v37.eval()

    v19_inputs = dict(inputs)
    v19_inputs.pop("source_validity")
    expected = v19(**v19_inputs)
    actual = v37(**inputs)

    for key in (
            "moe_candidate_scores", "selected_source_scores",
            "moe_gate_selected_query"):
        assert torch.equal(actual[key], expected[key])
    for key in (
            "moe_gate_joint_action_margin",
            "moe_gate_pairwise_utility_margin",
            "moe_gate_candidate_selection_margin",
            "moe_gate_selected_abstention_margin",
            "moe_gate_counterfactual_risk_margin",
            "moe_gate_counterfactual_benefit_margin",
            "moe_gate_counterfactual_hazard_margin"):
        assert torch.equal(actual[key], torch.zeros_like(actual[key]))


def test_v38_zero_init_exactly_preserves_complete_v19_source_moe_output():
    v19, v38, inputs = _v38_source_moe_pair()
    v19.eval()
    v38.eval()

    v19_inputs = dict(inputs)
    v19_inputs.pop("source_validity")
    expected = v19(**v19_inputs)
    actual = v38(**inputs)

    for key in (
            "moe_candidate_scores", "selected_source_scores",
            "moe_gate_selected_query"):
        assert torch.equal(actual[key], expected[key])
    for key in (
            "moe_gate_joint_action_margin",
            "moe_gate_pairwise_utility_margin",
            "moe_gate_candidate_selection_margin",
            "moe_gate_selected_abstention_margin",
            "moe_gate_counterfactual_risk_margin",
            "moe_gate_counterfactual_benefit_margin",
            "moe_gate_counterfactual_hazard_margin"):
        assert torch.equal(actual[key], torch.zeros_like(actual[key]))


def test_v29_counterfactual_risk_is_query_permutation_equivariant():
    torch.manual_seed(2901)
    head = CounterfactualSelectedRiskHead(hidden_dim=8, dropout=0.0)
    with torch.no_grad():
        head.risk_head[-1].weight.normal_()
        head.risk_head[-1].bias.fill_(0.2)
    head.eval()
    pair_features = torch.randn(2, 5, 8)
    selection_margin = torch.tensor([
        [0.0, 0.7, -0.3, 0.2, 0.5],
        [0.6, -0.2, 0.1, 0.8, 0.0],
    ])
    candidate_mask = torch.tensor([
        [False, True, True, False, True],
        [True, True, False, True, False],
    ])
    result = head(pair_features, selection_margin, candidate_mask)
    permutation = torch.tensor([2, 4, 0, 3, 1])
    changed = head(
        pair_features[:, permutation],
        selection_margin[:, permutation],
        candidate_mask[:, permutation],
    )

    assert torch.allclose(
        changed["margin"], result["margin"][:, permutation], atol=1e-6
    )
    assert torch.allclose(
        changed["candidate_risk"],
        result["candidate_risk"][:, permutation],
        atol=1e-6,
    )
    assert torch.allclose(
        changed["row_margin"], result["row_margin"], atol=1e-6
    )
    deployed_max = result["margin"].masked_fill(
        ~candidate_mask, -1e4
    ).max(dim=1).values
    assert torch.allclose(deployed_max, result["row_margin"], atol=1e-6)


def test_v26_deploys_benefit_margin_and_keeps_utility_auxiliary():
    _, v26, inputs = _v26_source_moe_pair()
    v26.eval()
    head = v26.fallback_gate.cascade_pairwise_calibrated_set_head
    with torch.no_grad():
        head.utility_head.bias.fill_(3.0)
        head.benefit_head.bias.fill_(-2.0)
    fallback = v26(**inputs)

    assert torch.equal(
        fallback["moe_gate_joint_action_margin"],
        fallback["moe_gate_row_benefit_margin"],
    )
    active = fallback["moe_gate_candidate_mask"]
    assert torch.equal(
        fallback["moe_gate_pairwise_utility_margin"][active],
        torch.full_like(
            fallback["moe_gate_pairwise_utility_margin"][active], 3.0
        ),
    )
    assert not bool(fallback["moe_gate_correction_switch"].any().item())

    with torch.no_grad():
        head.utility_head.bias.fill_(-3.0)
        head.benefit_head.bias.fill_(2.0)
    corrected = v26(**inputs)
    assert bool(corrected["moe_gate_correction_switch"].all().item())
    assert torch.equal(
        corrected["moe_gate_joint_action_margin"],
        corrected["moe_gate_row_benefit_margin"],
    )
    assert torch.equal(
        corrected["moe_gate_pairwise_utility_margin"][active],
        torch.full_like(
            corrected["moe_gate_pairwise_utility_margin"][active], -3.0
        ),
    )


def test_v23_requires_explicit_source_validity_and_masks_absent_source():
    _, v23, inputs = _v23_source_moe_pair()
    with pytest.raises(ValueError, match="explicit source_validity"):
        v23(**{key: value for key, value in inputs.items()
               if key != "source_validity"})

    absent = dict(inputs)
    absent["source_scores"] = dict(inputs["source_scores"])
    absent["source_scores"].pop("mask_text")
    absent["source_validity"] = inputs["source_validity"].clone()
    absent["source_validity"][..., 2] = False
    output = v23(**absent)

    assert torch.isfinite(output["moe_candidate_scores"]).all()
    assert torch.equal(
        output["moe_router_probs"][..., 1],
        torch.zeros_like(output["moe_router_probs"][..., 1]),
    )
    assert torch.equal(
        output["moe_expert_mask"][..., 1],
        torch.zeros_like(output["moe_expert_mask"][..., 1]),
    )


def test_adaptive_source_mixer_is_source_permutation_equivariant():
    torch.manual_seed(2302)
    original = AdaptiveSourceMixer(
        context_dim=6, rich_dim=7, hidden_dim=8,
        source_count=3, shared_index=0,
    )
    permuted = AdaptiveSourceMixer(
        context_dim=6, rich_dim=7, hidden_dim=8,
        source_count=3, shared_index=1,
    )
    permuted.load_state_dict(original.state_dict())
    with torch.no_grad():
        original.source_router[-1].weight.normal_()
        original.mix_residual[-1].weight.normal_()
        permuted.load_state_dict(original.state_dict())
    context = torch.randn(2, 4, 6)
    rich = torch.randn(2, 4, 7)
    ranks = torch.rand(2, 4, 3)
    validity = torch.ones(2, 4, 3, dtype=torch.bool)
    base_logits = torch.randn(2, 4, 2)
    result = original(
        context, rich, ranks, validity, base_logits,
        torch.tensor([0.4]), top_k=1,
    )

    source_permutation = torch.tensor([2, 0, 1])
    routed_permutation = torch.tensor([1, 0])
    changed = permuted(
        context,
        rich,
        ranks[..., source_permutation],
        validity[..., source_permutation],
        base_logits[..., routed_permutation],
        torch.tensor([0.4]),
        top_k=1,
    )

    assert torch.allclose(
        changed["fused_score"], result["fused_score"], atol=1e-6
    )
    assert torch.allclose(changed["features"], result["features"], atol=1e-6)


def test_dense_quality_score_is_coordinatewise_monotonic():
    box_logits = torch.tensor([[0.0, 0.0]])
    box_iou = torch.tensor([0.4])
    mask_logits = torch.tensor([[0.0, 0.0]])
    mask_iou = torch.tensor([0.3])
    baseline = dense_quality_expected_score(
        box_logits, box_iou, mask_logits, mask_iou,
        mask_utility_weight=0.25,
    )

    for changed in (
            (box_logits + torch.tensor([[1.0, 0.0]]), box_iou,
             mask_logits, mask_iou),
            (box_logits, box_iou + 0.1, mask_logits, mask_iou),
            (box_logits, box_iou,
             mask_logits + torch.tensor([[0.0, 1.0]]), mask_iou),
            (box_logits, box_iou, mask_logits, mask_iou + 0.1)):
        assert dense_quality_expected_score(
            *changed, mask_utility_weight=0.25
        ).item() > baseline.item()


def test_dense_quality_set_head_is_query_permutation_equivariant():
    torch.manual_seed(2303)
    head = DenseQualityFallbackSetActionHead(
        action_dim=7, rich_dim=5, adaptive_dim=4, hidden_dim=8,
        max_candidates=3, num_heads=2, dropout=0.0,
    )
    with torch.no_grad():
        head.quality_head.weight.normal_()
    head.eval()
    action = torch.randn(2, 5, 7)
    rich = torch.randn(2, 5, 5)
    adaptive = torch.randn(2, 5, 4)
    fallback = torch.tensor([0, 3])
    candidate_mask = torch.tensor([
        [False, True, True, False, True],
        [True, True, False, False, True],
    ])
    result = head(action, rich, adaptive, fallback, candidate_mask)
    permutation = torch.tensor([2, 4, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    changed = head(
        action[:, permutation], rich[:, permutation],
        adaptive[:, permutation], inverse[fallback],
        candidate_mask[:, permutation],
    )

    for key in (
            "margin", "box_threshold_logits", "box_iou",
            "mask_threshold_logits", "mask_iou", "quality", "uncertainty"):
        assert torch.allclose(
            changed[key], result[key][:, permutation], atol=1e-6
        )


def test_v27_uncertainty_risk_is_identity_at_init_and_trains_quality_head():
    _, v27, inputs = _v23_source_moe_pair()
    v27.fallback_gate.uncertainty_weight = 0.5
    v27.eval()

    initial = v27(**inputs)
    candidate_mask = initial["moe_gate_candidate_mask"]
    fallback = initial["moe_gate_supervision_fallback_query"]
    quality_mask = candidate_mask.clone()
    quality_mask[torch.arange(fallback.shape[0]), fallback] = True
    for row in range(quality_mask.shape[0]):
        uncertainty = initial["moe_gate_quality_uncertainty"][
            row, quality_mask[row]
        ]
        assert torch.equal(uncertainty, uncertainty[:1].expand_as(uncertainty))
    assert torch.equal(
        initial["moe_gate_risk_quality_margin"],
        torch.zeros_like(initial["moe_gate_risk_quality_margin"]),
    )
    assert torch.equal(
        initial["moe_gate_joint_action_margin"],
        initial["moe_gate_risk_quality_margin"],
    )
    assert initial["moe_gate_quality_uncertainty_mean"].item() == pytest.approx(
        0.5
    )

    quality_head = v27.fallback_gate.cascade_dense_quality_set_head.quality_head
    with torch.no_grad():
        quality_head.weight.normal_(std=0.1)
        quality_head.bias.normal_(std=0.1)
    output = v27(**inputs)
    assert torch.allclose(
        output["moe_gate_joint_action_margin"],
        output["moe_gate_risk_quality_margin"],
        atol=1e-7,
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=output["moe_gate_box_logits"],
        decision_logits=output["moe_gate_decision_logits"],
        box_ious=torch.tensor([
            [0.10, 0.70, 0.20, 0.55, 0.30],
            [0.65, 0.20, 0.60, 0.10, 0.80],
        ]),
        default_indices=output["moe_gate_supervision_fallback_query"],
        candidate_mask=output["moe_gate_candidate_mask"],
        mask_logits=output["moe_gate_mask_logits"],
        mask_ious=torch.tensor([
            [0.20, 0.65, 0.10, 0.50, 0.25],
            [0.60, 0.15, 0.55, 0.05, 0.75],
        ]),
        action_margin=output["moe_gate_action_margin"],
        joint_action_margin=output["moe_gate_joint_action_margin"],
        absolute_box_logits=output["moe_gate_absolute_box_logits"],
        absolute_box_iou=output["moe_gate_absolute_box_iou"],
        absolute_mask_logits=output["moe_gate_absolute_mask_logits"],
        absolute_mask_iou=output["moe_gate_absolute_mask_iou"],
        objective="cascade_v27_uncertainty_quality_risk",
        setwise_temperature=0.25,
    )
    result["loss"].backward()

    assert result["quality_action_loss"].item() > 0.0
    assert quality_head.weight.grad is not None
    assert torch.isfinite(quality_head.weight.grad).all()
    assert quality_head.weight.grad.abs().sum().item() > 0.0


def test_v27_cost_aware_boundary_separates_ranking_from_switching():
    margin = torch.zeros(2, 2, requires_grad=True)
    result = _risk_aware_dense_quality_action_loss(
        candidate_margin=margin,
        deployment_utility=torch.tensor([
            [0.0, -0.5],
            [0.0, 1.0],
        ]),
        box_ious=torch.tensor([
            [0.60, 0.40],
            [0.10, 0.60],
        ]),
        mask_ious=None,
        fallback_indices=torch.zeros(2, dtype=torch.long),
        active_mask=torch.tensor([
            [False, True],
            [False, True],
        ]),
        sample_mask=torch.ones(2, dtype=torch.bool),
        mask_utility_weight=0.25,
        false_override_weight=2.0,
        temperature=0.25,
    )

    result["loss"].backward()
    assert result["target_positive_ratio"].item() == pytest.approx(0.5)
    assert margin.grad[0, 1].item() > 0.0
    assert margin.grad[1, 1].item() < 0.0


def test_v25_pairwise_head_is_query_permutation_equivariant():
    torch.manual_seed(2501)
    head = CalibratedPairwiseRiskSetActionHead(
        action_dim=7, rich_dim=5, adaptive_dim=4, evidence_dim=6,
        hidden_dim=8, max_candidates=3, num_heads=2, dropout=0.0,
    )
    with torch.no_grad():
        head.utility_head.weight.normal_()
        head.utility_head.bias.fill_(-0.3)
        head.benefit_head.weight.normal_()
        head.benefit_head.bias.fill_(-0.7)
    head.eval()
    action = torch.randn(2, 5, 7)
    rich = torch.randn(2, 5, 5)
    adaptive = torch.randn(2, 5, 4)
    evidence = torch.rand(2, 5, 6)
    fallback = torch.tensor([0, 3])
    candidate_mask = torch.tensor([
        [False, True, True, False, True],
        [True, True, False, False, True],
    ])
    result = head(
        action, rich, adaptive, evidence, fallback, candidate_mask
    )
    permutation = torch.tensor([2, 4, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    changed = head(
        action[:, permutation], rich[:, permutation],
        adaptive[:, permutation], evidence[:, permutation],
        inverse[fallback], candidate_mask[:, permutation],
    )

    for key in ("margin", "benefit_margin"):
        assert torch.allclose(
            changed[key], result[key][:, permutation], atol=1e-6
        )
    assert torch.equal(
        changed["fallback_logit"], result["fallback_logit"]
    )


def test_v25_loss_regresses_deployed_utility_and_auxiliary_benefit():
    utility_margin = torch.zeros(2, 3, requires_grad=True)
    benefit_margin = torch.zeros(2, 3, requires_grad=True)
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(2, 3, 2, 3),
        decision_logits=torch.zeros(2, 3, 3),
        box_ious=torch.tensor([
            [0.10, 0.70, 0.20],
            [0.70, 0.20, 0.60],
        ]),
        default_indices=torch.zeros(2, dtype=torch.long),
        candidate_mask=torch.tensor([
            [False, True, True],
            [False, True, True],
        ]),
        action_margin=utility_margin,
        row_benefit_margin=benefit_margin,
        joint_action_margin=utility_margin,
        absolute_box_logits=torch.zeros(2, 3, 2),
        absolute_box_iou=torch.full((2, 3), 0.5),
        objective="cascade_v25_pairwise_calibrated_risk",
        setwise_temperature=0.25,
    )

    assert result["utility_regression_loss"].item() > 0.0
    assert result["benefit_loss"].item() > 0.0
    result["loss"].backward()
    assert utility_margin.grad[0, 1].item() < 0.0
    assert utility_margin.grad[1, 1].item() > 0.0
    assert benefit_margin.grad[0, 1].item() < 0.0
    assert benefit_margin.grad[1, 1].item() > 0.0


def test_v26_prior_restored_loss_separates_deployment_and_utility_gradients():
    utility_margin = torch.zeros(2, 3, requires_grad=True)
    benefit_margin = torch.zeros(2, 3, requires_grad=True)
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(2, 3, 2, 3),
        decision_logits=torch.zeros(2, 3, 3),
        box_ious=torch.tensor([
            [0.10, 0.70, 0.20],
            [0.70, 0.20, 0.60],
        ]),
        default_indices=torch.zeros(2, dtype=torch.long),
        candidate_mask=torch.tensor([
            [False, True, True],
            [False, True, True],
        ]),
        action_margin=benefit_margin,
        row_benefit_margin=benefit_margin,
        joint_action_margin=benefit_margin,
        pairwise_utility_margin=utility_margin,
        absolute_box_logits=torch.zeros(2, 3, 2),
        absolute_box_iou=torch.full((2, 3), 0.5),
        objective="cascade_v26_prior_restored_pairwise_risk",
        setwise_temperature=0.25,
        boundary_loss_weight=1.0,
    )

    assert result["utility_regression_loss"].item() > 0.0
    assert result["benefit_loss"].item() > 0.0
    assert result["pairwise_rank_loss"].item() > 0.0
    assert result["boundary_calibration_loss"].item() > 0.0
    assert result["joint_action_loss"].item() > 0.0
    assert result["benefit_prior_shift"].item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item()
    )
    assert result["benefit_positive_prior"].item() == pytest.approx(1.0 / 3.0)
    result["loss"].backward()
    assert utility_margin.grad[0, 1].item() < 0.0
    assert utility_margin.grad[1, 1].item() > 0.0
    assert benefit_margin.grad[0, 1].item() < 0.0
    assert benefit_margin.grad[1, 1].item() > 0.0
    assert result["stats"][
        "source_moe_gate_benefit_prior_shift"
    ].item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_v28_selected_abstention_tracks_the_deployed_candidate_utility():
    utility_margin = torch.zeros(2, 3, requires_grad=True)
    benefit_margin = torch.zeros(2, 3, requires_grad=True)
    selection_margin = torch.tensor([
        [0.0, 0.4, 0.1],
        [0.0, 0.4, 0.1],
    ], requires_grad=True)
    row_margin = torch.zeros(2, requires_grad=True)
    joint_margin = (
        selection_margin
        - selection_margin[:, 1:].max(dim=1).values.unsqueeze(1)
        + row_margin.unsqueeze(1)
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(2, 3, 2, 3),
        decision_logits=torch.zeros(2, 3, 3),
        box_ious=torch.tensor([
            [0.70, 0.20, 0.10],
            [0.10, 0.70, 0.20],
        ]),
        default_indices=torch.zeros(2, dtype=torch.long),
        candidate_mask=torch.tensor([
            [False, True, True],
            [False, True, True],
        ]),
        action_margin=joint_margin,
        row_benefit_margin=benefit_margin,
        joint_action_margin=joint_margin,
        pairwise_utility_margin=utility_margin,
        candidate_selection_margin=selection_margin,
        selected_abstention_margin=row_margin,
        absolute_box_logits=torch.zeros(2, 3, 2),
        absolute_box_iou=torch.full((2, 3), 0.5),
        objective="cascade_v28_selected_abstention_risk",
        setwise_temperature=0.25,
    )

    assert result["abstention_regression_loss"].item() > 0.0
    assert result["abstention_benefit_loss"].item() > 0.0
    assert result["abstention_selection_rank_loss"].item() > 0.0
    assert result["stats"][
        "source_moe_gate_abstention_target_positive_ratio"
    ].item() == pytest.approx(0.5)
    assert result["stats"][
        "source_moe_gate_policy_selected_positive_count"
    ].item() == pytest.approx(1.0)
    assert result["stats"][
        "source_moe_gate_policy_opportunity_capture_ratio"
    ].item() == pytest.approx(1.0)
    assert result["stats"][
        "source_moe_gate_abstention_conditional_recall_ratio"
    ].item() == pytest.approx(0.0)
    result["loss"].backward()
    assert row_margin.grad[0].item() > 0.0
    assert row_margin.grad[1].item() < 0.0
    assert utility_margin.grad is not None
    assert benefit_margin.grad is not None
    assert selection_margin.grad is not None
    assert torch.isfinite(selection_margin.grad).all()


def test_v29_counterfactual_risk_calibrates_raw_zero_deployment_boundary():
    candidate_risk = torch.zeros(2, 3, requires_grad=True)
    selection_margin = torch.zeros(2, 3)
    decision_utility = torch.tensor([
        [1.0, -1.0, -2.0],
        [-1.0, -2.0, -3.0],
    ])
    active = torch.ones(2, 3, dtype=torch.bool)
    sample = torch.ones(2, dtype=torch.bool)

    result = _counterfactual_selected_risk_loss(
        candidate_risk,
        selection_margin,
        decision_utility,
        active,
        sample,
        false_positive_weight=2.0,
        temperature=0.25,
    )

    assert result["deployment_boundary_loss"].item() > 0.0
    assert result["classification_loss"].item() == pytest.approx(
        result["deployment_boundary_loss"].item()
    )
    result["loss"].backward()
    # Deployment uses raw risk > 0, so useful/unsafe candidates must receive
    # gradients across that same zero boundary.
    assert candidate_risk.grad[0, 0].item() < 0.0
    assert candidate_risk.grad[0, 1].item() > 0.0
    assert candidate_risk.grad[1, 0].item() > 0.0
    assert torch.isfinite(candidate_risk.grad).all()


def test_v34_counterfactual_risk_supervises_every_positive_candidate_per_row():
    candidate_risk = torch.zeros(2, 3, requires_grad=True)
    selection_margin = torch.zeros(2, 3)
    decision_utility = torch.tensor([
        [1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ])
    active = torch.tensor([
        [True, True, False],
        [True, False, False],
    ])
    sample = torch.ones(2, dtype=torch.bool)

    result = _counterfactual_selected_risk_loss(
        candidate_risk,
        selection_margin,
        decision_utility,
        active,
        sample,
        false_positive_weight=2.0,
        temperature=0.25,
    )

    classification_grad = torch.autograd.grad(
        result["classification_loss"], candidate_risk, retain_graph=True
    )[0]
    regression_grad = torch.autograd.grad(
        result["regression_loss"], candidate_risk
    )[0]
    for gradient in (classification_grad, regression_grad):
        assert gradient[0, 0].item() < 0.0
        assert gradient[0, 1].item() < 0.0
        assert gradient[1, 0].item() < 0.0
        # Row averaging gives a row with two positive candidates the same
        # total gradient as a row with one positive candidate.
        assert gradient[0, :2].sum().item() == pytest.approx(
            gradient[1, 0].item()
        )
        assert gradient[0, 2].item() == pytest.approx(0.0)
        assert gradient[1, 1:].abs().sum().item() == pytest.approx(0.0)


def test_v35_counterfactual_raw_risk_does_not_double_charge_break_cost():
    decision_utility = torch.tensor([[1.0], [-2.0]])
    selection_margin = torch.zeros(2, 1)
    active = torch.ones(2, 1, dtype=torch.bool)
    sample = torch.ones(2, dtype=torch.bool)
    results = []
    gradients = []
    for false_positive_weight in (1.0, 4.0):
        candidate_risk = torch.zeros(2, 1, requires_grad=True)
        result = _counterfactual_selected_risk_loss(
            candidate_risk,
            selection_margin,
            decision_utility,
            active,
            sample,
            false_positive_weight=false_positive_weight,
            temperature=0.25,
        )
        results.append(result)
        gradients.append(torch.autograd.grad(result["loss"], candidate_risk)[0])

    # Break cost is already encoded as utility -2. Reapplying it to raw BCE
    # or regression would move the neutral decision away from risk == 0.
    assert results[0]["classification_loss"].item() == pytest.approx(
        results[1]["classification_loss"].item()
    )
    assert results[0]["regression_loss"].item() == pytest.approx(
        results[1]["regression_loss"].item()
    )
    assert torch.allclose(gradients[0], gradients[1], rtol=0.0, atol=1e-7)
    assert gradients[0][0, 0].item() < 0.0
    assert gradients[0][1, 0].item() > 0.0
    assert gradients[0].sum().item() == pytest.approx(0.0, abs=1e-7)


def test_v36_counterfactual_boundary_gap_is_symmetric_around_raw_zero():
    candidate_risk = torch.zeros(2, 1, requires_grad=True)
    result = _counterfactual_selected_risk_loss(
        candidate_risk,
        torch.zeros(2, 1),
        torch.tensor([[1.0], [-2.0]]),
        torch.ones(2, 1, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        false_positive_weight=2.0,
        temperature=0.25,
    )

    classification_grad = torch.autograd.grad(
        result["classification_loss"], candidate_risk
    )[0]
    assert result["classification_loss"].item() > torch.log(
        torch.tensor(2.0)
    ).item()
    assert classification_grad[0, 0].item() < 0.0
    assert classification_grad[1, 0].item() > 0.0
    assert classification_grad.sum().item() == pytest.approx(0.0, abs=1e-7)


def test_v37_benefit_hazard_loss_separates_gain_from_break_evidence():
    candidate_benefit = torch.zeros(2, 2, requires_grad=True)
    candidate_hazard = torch.zeros(2, 2, requires_grad=True)
    result = _counterfactual_benefit_hazard_loss(
        candidate_benefit,
        candidate_hazard,
        torch.tensor([[0.4, 0.1], [0.3, 0.2]]),
        torch.tensor([[1.0, -2.0], [-3.0, -1.0]]),
        torch.ones(2, 2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        focal_gamma=2.0,
    )

    benefit_grad, hazard_grad = torch.autograd.grad(
        result["loss"], (candidate_benefit, candidate_hazard)
    )
    # The useful candidate learns gain without manufacturing negative hazard.
    assert benefit_grad[0, 0].item() < 0.0
    assert hazard_grad[0, 0].item() == pytest.approx(0.0)
    # The deployed hard negatives learn break evidence without negative gain.
    assert benefit_grad[0, 1].item() == pytest.approx(0.0)
    assert hazard_grad[0, 1].item() < 0.0
    assert benefit_grad[1, 0].item() == pytest.approx(0.0)
    assert hazard_grad[1, 0].item() < 0.0
    assert torch.isfinite(benefit_grad).all()
    assert torch.isfinite(hazard_grad).all()


def test_v37_focal_boundary_targets_misclassified_candidate_evidence():
    benefit = torch.tensor([[2.0], [2.0]], requires_grad=True)
    hazard = torch.tensor([[0.0], [0.0]], requires_grad=True)
    result = _counterfactual_benefit_hazard_loss(
        benefit,
        hazard,
        torch.zeros(2, 1),
        torch.tensor([[1.0], [-2.0]]),
        torch.ones(2, 1, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        focal_gamma=2.0,
    )

    classification_grad = torch.autograd.grad(
        result["classification_loss"], benefit
    )[0]
    # The already-correct positive is nearly ignored, while the harmful
    # false switch receives the hard-example focal correction.
    assert classification_grad[0, 0].item() < 0.0
    assert classification_grad[1, 0].item() > 0.0
    assert abs(classification_grad[1, 0].item()) > (
        10.0 * abs(classification_grad[0, 0].item())
    )


def test_v37_decomposed_head_is_permutation_equivariant_and_zero_identity():
    torch.manual_seed(3701)
    head = CounterfactualSelectedRiskHead(
        hidden_dim=8, dropout=0.0, decomposed=True
    )
    pair_features = torch.randn(2, 4, 8)
    selection_margin = torch.tensor([
        [0.1, 0.7, -0.2, 0.3],
        [0.6, -0.1, 0.2, 0.0],
    ])
    candidate_mask = torch.tensor([
        [True, True, False, True],
        [True, False, True, True],
    ])
    result = head(pair_features, selection_margin, candidate_mask)
    assert torch.equal(
        result["candidate_risk"], torch.zeros_like(selection_margin)
    )
    assert torch.equal(
        result["row_margin"], torch.zeros_like(result["row_margin"])
    )

    with torch.no_grad():
        head.risk_head[-1].weight.normal_()
        head.risk_head[-1].bias.copy_(torch.tensor([0.2, -0.1]))
    permutation = torch.tensor([2, 0, 3, 1])
    result = head(pair_features, selection_margin, candidate_mask)
    changed = head(
        pair_features[:, permutation],
        selection_margin[:, permutation],
        candidate_mask[:, permutation],
    )
    for key in (
            "candidate_benefit", "candidate_hazard", "candidate_risk",
            "margin"):
        assert torch.allclose(
            changed[key], result[key][:, permutation], atol=1e-6
        )
    expected_risk = (
        torch.relu(result["candidate_benefit"])
        - torch.relu(result["candidate_hazard"])
    )
    expected_risk = expected_risk.masked_fill(~candidate_mask, 0.0)
    assert torch.allclose(result["candidate_risk"], expected_risk, atol=1e-6)
    deployed_max = result["margin"].masked_fill(
        ~candidate_mask, -1e4
    ).max(dim=1).values
    assert torch.allclose(deployed_max, result["row_margin"], atol=1e-6)


def test_v38_complementary_logodds_have_balanced_class_gradients():
    benefit = torch.zeros(2, 1, requires_grad=True)
    hazard = torch.zeros(2, 1, requires_grad=True)
    result = _counterfactual_complementary_logodds_loss(
        benefit,
        hazard,
        torch.zeros(2, 1),
        torch.tensor([[3.0], [-6.0]]),
        torch.ones(2, 1, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        focal_gamma=2.0,
    )

    benefit_grad, hazard_grad = torch.autograd.grad(
        result["loss"], (benefit, hazard)
    )
    assert benefit_grad[0, 0].item() < 0.0
    assert hazard_grad[0, 0].item() > 0.0
    assert benefit_grad[1, 0].item() > 0.0
    assert hazard_grad[1, 0].item() < 0.0
    # Utility magnitude is deliberately normalized: break cost changes the
    # label confidence, not the deployment scale or per-class gradient sum.
    assert benefit_grad[0, 0].item() == pytest.approx(
        -benefit_grad[1, 0].item(), abs=1e-7
    )
    assert hazard_grad[0, 0].item() == pytest.approx(
        -hazard_grad[1, 0].item(), abs=1e-7
    )


def test_v38_head_deploys_raw_complementary_logodds_difference():
    head = CounterfactualSelectedRiskHead(
        hidden_dim=8,
        dropout=0.0,
        decomposed=True,
        complementary_log_odds=True,
    )
    with torch.no_grad():
        head.risk_head[-1].weight.zero_()
        head.risk_head[-1].bias.copy_(torch.tensor([-0.2, 0.3]))
    mask = torch.tensor([[True, True, False]])
    result = head(
        torch.randn(1, 3, 8),
        torch.tensor([[0.5, 0.2, -0.1]]),
        mask,
    )

    assert result["candidate_benefit"][0, 0].item() == pytest.approx(-0.2)
    assert result["candidate_hazard"][0, 0].item() == pytest.approx(0.3)
    assert result["candidate_risk"][0, 0].item() == pytest.approx(-0.5)
    assert result["row_margin"].item() == pytest.approx(-0.5)


def test_v39_hazard_residual_has_opposed_gain_and_hazard_gradients():
    gain = torch.zeros(2, 1, requires_grad=True)
    hazard = torch.zeros(2, 1, requires_grad=True)
    result = _counterfactual_hazard_residual_loss(
        gain,
        hazard,
        torch.zeros(2, 1),
        torch.tensor([[2.0], [-2.0]]),
        torch.ones(2, 1, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        focal_gamma=2.0,
    )
    gain_grad, hazard_grad = torch.autograd.grad(
        result["loss"], (gain, hazard)
    )
    assert gain_grad[0, 0].item() < 0.0
    assert hazard_grad[0, 0].item() > 0.0
    assert gain_grad[1, 0].item() > 0.0
    assert hazard_grad[1, 0].item() < 0.0
    assert torch.isfinite(gain_grad).all()
    assert torch.isfinite(hazard_grad).all()


def test_v39_head_deploys_gain_minus_relu_hazard_and_preserves_v19_identity():
    v19, v39, inputs = _v39_source_moe_pair()
    v39.eval()
    expected = v19(**inputs)
    actual = v39(**inputs)
    assert torch.equal(
        actual["selected_source_scores"], expected["selected_source_scores"]
    )
    with torch.no_grad():
        head = v39.fallback_gate.cascade_counterfactual_hazard_residual_head
        head.risk_head[-1].weight.zero_()
        head.risk_head[-1].bias.copy_(torch.tensor([0.7, 0.3]))
    pair_features = torch.randn(1, 3, 8)
    selection_margin = torch.tensor([[0.3, 0.2, -0.1]])
    candidate_mask = torch.tensor([[True, True, False]])
    result = v39.fallback_gate.cascade_counterfactual_hazard_residual_head(
        pair_features, selection_margin, candidate_mask
    )
    assert result["candidate_risk"][0, 0].item() == pytest.approx(0.4)
    assert result["candidate_risk"][0, 1].item() == pytest.approx(0.4)
    assert result["row_margin"].item() == pytest.approx(0.4)
    permutation = torch.tensor([2, 0, 1])
    changed = v39.fallback_gate.cascade_counterfactual_hazard_residual_head(
        pair_features[:, permutation],
        selection_margin[:, permutation],
        candidate_mask[:, permutation],
    )
    for key in (
            "candidate_benefit", "candidate_hazard", "candidate_risk",
            "margin"):
        assert torch.allclose(
            changed[key], result[key][:, permutation], atol=1e-6
        )


def test_v30_positive_candidate_top1_margin_trains_hard_policy_choice():
    selection_margin = torch.tensor([
        [0.0, 0.1, 0.3],
        [0.0, 0.4, 0.1],
    ], requires_grad=True)
    decision_utility = torch.tensor([
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
    ])
    active = torch.ones(2, 3, dtype=torch.bool)
    sample = torch.ones(2, dtype=torch.bool)

    loss, rows, violation = _positive_candidate_top1_margin_loss(
        selection_margin,
        decision_utility,
        active,
        sample,
        margin=0.05,
    )

    assert loss.item() > 0.0
    assert rows.tolist() == [True, True]
    assert violation[0].item() > 0.0
    assert violation[1].item() < 0.0
    loss.backward()
    assert selection_margin.grad[0, 1].item() < 0.0
    assert selection_margin.grad[0, 2].item() > 0.0
    assert selection_margin.grad[1, 1].item() == pytest.approx(0.0)
    assert selection_margin.grad[1, 2].item() == pytest.approx(0.0)


def test_v33_counterfactual_objective_excludes_shifted_selected_row_bce():
    utility_margin = torch.zeros(2, 3, requires_grad=True)
    benefit_margin = torch.zeros(2, 3, requires_grad=True)
    selection_margin = torch.tensor([
        [0.0, 0.4, 0.1],
        [0.0, 0.4, 0.1],
    ], requires_grad=True)
    candidate_risk = torch.zeros(2, 3, requires_grad=True)
    selected_risk = candidate_risk[:, 1]
    joint_margin = (
        selection_margin
        - selection_margin[:, 1:].max(dim=1).values.unsqueeze(1)
        + selected_risk.unsqueeze(1)
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(2, 3, 2, 3),
        decision_logits=torch.zeros(2, 3, 3),
        box_ious=torch.tensor([
            [0.70, 0.20, 0.10],
            [0.10, 0.70, 0.20],
        ]),
        default_indices=torch.zeros(2, dtype=torch.long),
        candidate_mask=torch.tensor([
            [False, True, True],
            [False, True, True],
        ]),
        action_margin=joint_margin,
        row_benefit_margin=benefit_margin,
        joint_action_margin=joint_margin,
        pairwise_utility_margin=utility_margin,
        candidate_selection_margin=selection_margin,
        selected_abstention_margin=selected_risk,
        counterfactual_risk_margin=candidate_risk,
        absolute_box_logits=torch.zeros(2, 3, 2),
        absolute_box_iou=torch.full((2, 3), 0.5),
        objective="cascade_v29_counterfactual_selected_risk",
        setwise_temperature=0.25,
    )

    assert result["abstention_benefit_loss"].item() > 0.0
    expected = (
        result["abstention_regression_loss"]
        + result["abstention_selection_regression_loss"]
        + result["abstention_selection_rank_loss"]
        + result["benefit_loss"]
        + result["positive_mass_loss"]
        + result["positive_top1_loss"]
        + result["counterfactual_risk_loss"]
        + result["absolute_quality_loss"]
        + result["dense_quality_rank_loss"]
    )
    assert result["decision_loss"].item() == pytest.approx(expected.item())


def test_v37_objective_wires_decomposed_candidate_evidence_end_to_end():
    utility_margin = torch.zeros(2, 3, requires_grad=True)
    benefit_margin = torch.zeros(2, 3, requires_grad=True)
    selection_margin = torch.tensor([
        [0.0, 0.4, 0.1],
        [0.0, 0.4, 0.1],
    ], requires_grad=True)
    candidate_benefit = torch.zeros(2, 3, requires_grad=True)
    candidate_hazard = torch.zeros(2, 3, requires_grad=True)
    selected_risk = (
        torch.relu(candidate_benefit) - torch.relu(candidate_hazard)
    )[:, 1]
    joint_margin = (
        selection_margin
        - selection_margin[:, 1:].max(dim=1).values.unsqueeze(1)
        + selected_risk.unsqueeze(1)
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(2, 3, 2, 3),
        decision_logits=torch.zeros(2, 3, 3),
        box_ious=torch.tensor([
            [0.70, 0.20, 0.10],
            [0.10, 0.70, 0.20],
        ]),
        default_indices=torch.zeros(2, dtype=torch.long),
        candidate_mask=torch.tensor([
            [False, True, True],
            [False, True, True],
        ]),
        action_margin=joint_margin,
        row_benefit_margin=benefit_margin,
        joint_action_margin=joint_margin,
        pairwise_utility_margin=utility_margin,
        candidate_selection_margin=selection_margin,
        selected_abstention_margin=selected_risk,
        counterfactual_benefit_margin=candidate_benefit,
        counterfactual_hazard_margin=candidate_hazard,
        absolute_box_logits=torch.zeros(2, 3, 2),
        absolute_box_iou=torch.full((2, 3), 0.5),
        objective="cascade_v37_counterfactual_benefit_hazard_risk",
        setwise_temperature=0.25,
        focal_gamma=2.0,
    )

    assert result["counterfactual_risk_loss"].item() > 0.0
    assert result["counterfactual_benefit_regression_loss"].item() > 0.0
    assert result["counterfactual_hazard_regression_loss"].item() > 0.0
    result["loss"].backward()
    assert candidate_hazard.grad[0, 1].item() < 0.0
    assert candidate_benefit.grad[1, 1].item() < 0.0
    assert torch.isfinite(candidate_benefit.grad).all()
    assert torch.isfinite(candidate_hazard.grad).all()


def test_v38_objective_wires_complementary_logodds_end_to_end():
    utility_margin = torch.zeros(2, 3, requires_grad=True)
    benefit_margin = torch.zeros(2, 3, requires_grad=True)
    selection_margin = torch.tensor([
        [0.0, 0.4, 0.1],
        [0.0, 0.4, 0.1],
    ], requires_grad=True)
    candidate_benefit = torch.zeros(2, 3, requires_grad=True)
    candidate_hazard = torch.zeros(2, 3, requires_grad=True)
    selected_risk = (candidate_benefit - candidate_hazard)[:, 1]
    joint_margin = (
        selection_margin
        - selection_margin[:, 1:].max(dim=1).values.unsqueeze(1)
        + selected_risk.unsqueeze(1)
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(2, 3, 2, 3),
        decision_logits=torch.zeros(2, 3, 3),
        box_ious=torch.tensor([
            [0.70, 0.20, 0.10],
            [0.10, 0.70, 0.20],
        ]),
        default_indices=torch.zeros(2, dtype=torch.long),
        candidate_mask=torch.tensor([
            [False, True, True],
            [False, True, True],
        ]),
        action_margin=joint_margin,
        row_benefit_margin=benefit_margin,
        joint_action_margin=joint_margin,
        pairwise_utility_margin=utility_margin,
        candidate_selection_margin=selection_margin,
        selected_abstention_margin=selected_risk,
        counterfactual_benefit_margin=candidate_benefit,
        counterfactual_hazard_margin=candidate_hazard,
        absolute_box_logits=torch.zeros(2, 3, 2),
        absolute_box_iou=torch.full((2, 3), 0.5),
        objective="cascade_v38_complementary_logodds_risk",
        setwise_temperature=0.25,
        focal_gamma=2.0,
    )

    assert result["counterfactual_risk_loss"].item() > 0.0
    assert result[
        "counterfactual_benefit_classification_loss"
    ].item() > 0.0
    assert result[
        "counterfactual_hazard_classification_loss"
    ].item() > 0.0
    result["loss"].backward()
    assert candidate_benefit.grad[0, 1].item() > 0.0
    assert candidate_hazard.grad[0, 1].item() < 0.0
    assert candidate_benefit.grad[1, 1].item() < 0.0
    assert candidate_hazard.grad[1, 1].item() > 0.0


def test_v39_objective_wires_hazard_residual_end_to_end():
    utility_margin = torch.zeros(2, 3, requires_grad=True)
    benefit_margin = torch.zeros(2, 3, requires_grad=True)
    selection_margin = torch.tensor([
        [0.0, 0.4, 0.1],
        [0.0, 0.4, 0.1],
    ], requires_grad=True)
    candidate_benefit = torch.zeros(2, 3, requires_grad=True)
    candidate_hazard = torch.zeros(2, 3, requires_grad=True)
    selected_risk = (
        candidate_benefit - torch.relu(candidate_hazard)
    )[:, 1]
    joint_margin = (
        selection_margin
        - selection_margin[:, 1:].max(dim=1).values.unsqueeze(1)
        + selected_risk.unsqueeze(1)
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(2, 3, 2, 3),
        decision_logits=torch.zeros(2, 3, 3),
        box_ious=torch.tensor([
            [0.70, 0.20, 0.10],
            [0.10, 0.70, 0.20],
        ]),
        default_indices=torch.zeros(2, dtype=torch.long),
        candidate_mask=torch.tensor([
            [False, True, True],
            [False, True, True],
        ]),
        action_margin=joint_margin,
        row_benefit_margin=benefit_margin,
        joint_action_margin=joint_margin,
        pairwise_utility_margin=utility_margin,
        candidate_selection_margin=selection_margin,
        selected_abstention_margin=selected_risk,
        counterfactual_benefit_margin=candidate_benefit,
        counterfactual_hazard_margin=candidate_hazard,
        absolute_box_logits=torch.zeros(2, 3, 2),
        absolute_box_iou=torch.full((2, 3), 0.5),
        objective="cascade_v39_hazard_residual_risk",
        setwise_temperature=0.25,
        focal_gamma=2.0,
    )
    assert result["counterfactual_risk_loss"].item() > 0.0
    assert result[
        "counterfactual_benefit_classification_loss"
    ].item() > 0.0
    assert result[
        "counterfactual_hazard_classification_loss"
    ].item() > 0.0
    result["loss"].backward()
    assert candidate_benefit.grad[0, 1].item() > 0.0
    assert candidate_hazard.grad[0, 1].item() < 0.0
    assert candidate_benefit.grad[1, 1].item() < 0.0
    assert candidate_hazard.grad[1, 1].item() > 0.0


def test_v26_prior_restored_benefit_has_balanced_per_class_gradients():
    logits = torch.zeros(1, 5, requires_grad=True)
    targets = torch.tensor([[True, False, False, False, False]])
    active = torch.ones_like(targets)
    loss, prior_shift, positive_prior = (
        _prior_restored_balanced_benefit_loss(
            logits, targets, active, false_positive_weight=2.0
        )
    )

    assert prior_shift.item() == pytest.approx(torch.log(torch.tensor(2.5)).item())
    assert positive_prior.item() == pytest.approx(2.0 / 7.0)
    loss.backward()
    assert logits.grad[0, 0].item() < 0.0
    assert bool((logits.grad[0, 1:] > 0.0).all().item())
    assert torch.isfinite(logits.grad).all()


def test_rowwise_boundary_calibration_pushes_positive_rows_across_zero():
    margin = torch.zeros(3, 4, requires_grad=True)
    utility = torch.tensor([
        [1.0, -1.0, -1.0, -1.0],
        [-1.0, -2.0, -3.0, -4.0],
        [0.0, 1.0, -1.0, -1.0],
    ])
    active = torch.ones_like(utility, dtype=torch.bool)
    sample = torch.ones(3, dtype=torch.bool)
    result = _rowwise_boundary_calibration_loss(
        margin, utility, active, sample,
        temperature=0.25, false_positive_weight=2.0,
    )

    assert result["loss"].item() > 0.0
    assert result["positive_ratio"].item() == pytest.approx(2.0 / 3.0)
    result["loss"].backward()
    assert torch.isfinite(margin.grad).all()
    assert bool((margin.grad[0] < 0.0).all().item())
    assert bool((margin.grad[1] > 0.0).all().item())


def test_v26_pairwise_rank_is_shift_invariant_with_fixed_zero_benefit_boundary():
    common = dict(
        box_logits=torch.zeros(2, 3, 2, 3),
        decision_logits=torch.zeros(2, 3, 3),
        box_ious=torch.tensor([
            [0.10, 0.70, 0.20],
            [0.70, 0.20, 0.60],
        ]),
        default_indices=torch.zeros(2, dtype=torch.long),
        candidate_mask=torch.tensor([
            [False, True, True],
            [False, True, True],
        ]),
        pairwise_utility_margin=torch.zeros(2, 3),
        absolute_box_logits=torch.zeros(2, 3, 2),
        absolute_box_iou=torch.full((2, 3), 0.5),
        objective="cascade_v26_prior_restored_pairwise_risk",
        setwise_temperature=0.25,
    )
    benefit = torch.tensor([
        [0.0, 0.4, -0.2],
        [0.0, -0.3, 0.1],
    ])
    baseline = compute_source_moe_fallback_gate_loss(
        **common,
        action_margin=benefit,
        row_benefit_margin=benefit,
        joint_action_margin=benefit,
    )
    shifted = benefit + torch.tensor([[0.0], [1.5]])
    changed = compute_source_moe_fallback_gate_loss(
        **common,
        action_margin=shifted,
        row_benefit_margin=shifted,
        joint_action_margin=shifted,
    )

    assert changed["pairwise_rank_loss"].item() == pytest.approx(
        baseline["pairwise_rank_loss"].item(), abs=1e-6
    )
    assert changed["benefit_loss"].item() != pytest.approx(
        baseline["benefit_loss"].item()
    )


def test_v23_dense_objective_reaches_set_encoder_quality_router_and_mix():
    _, v23, inputs = _v23_source_moe_pair()
    for parameter in v23.parameters():
        parameter.requires_grad = False
    for name, parameter in v23.named_parameters():
        if name.startswith((
                "adaptive_source_mixer.",
                "fallback_gate.cascade_dense_quality_set_head.",
        )):
            parameter.requires_grad = True
    with torch.no_grad():
        v23.fallback_gate.cascade_dense_quality_set_head.quality_head.weight.normal_(
            std=0.1
        )
    output = v23(**inputs)
    generator = torch.Generator().manual_seed(2304)
    box_ious = torch.rand(2, 5, generator=generator)
    mask_ious = torch.rand(2, 5, generator=generator)
    result = compute_source_moe_fallback_gate_loss(
        box_logits=output["moe_gate_box_logits"],
        decision_logits=output["moe_gate_decision_logits"],
        box_ious=box_ious,
        default_indices=output["moe_gate_supervision_fallback_query"],
        candidate_mask=output["moe_gate_candidate_mask"],
        mask_logits=output["moe_gate_mask_logits"],
        mask_ious=mask_ious,
        action_margin=output["moe_gate_action_margin"],
        joint_action_margin=output["moe_gate_joint_action_margin"],
        absolute_box_logits=output["moe_gate_absolute_box_logits"],
        absolute_box_iou=output["moe_gate_absolute_box_iou"],
        absolute_mask_logits=output["moe_gate_absolute_mask_logits"],
        absolute_mask_iou=output["moe_gate_absolute_mask_iou"],
        objective="cascade_v23_dense_quality_risk",
        setwise_temperature=0.25,
    )
    result["loss"].backward()

    for prefix in (
            "fallback_gate.cascade_dense_quality_set_head.input_projection",
            "fallback_gate.cascade_dense_quality_set_head.quality_head",
            "adaptive_source_mixer.source_router.2",
            "adaptive_source_mixer.mix_residual.2"):
        gradients = [
            parameter.grad
            for name, parameter in v23.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0
    assert result["absolute_quality_loss"].item() > 0.0
    assert result["dense_quality_rank_loss"].item() >= 0.0


def _empirical_shared_margin_gradient(fallback_rows, false_positive_weight):
    shared_margin = torch.zeros((), requires_grad=True)
    row_count = 1 + int(fallback_rows)
    candidate_margin = shared_margin.expand(row_count, 1)
    decision_utility = torch.cat((
        torch.ones(1, 1),
        -torch.ones(fallback_rows, 1),
    ), dim=0)
    loss, _, _, _ = _empirical_setwise_action_risk_loss(
        candidate_margin=candidate_margin,
        decision_utility=decision_utility,
        active_mask=torch.ones(row_count, 1, dtype=torch.bool),
        sample_mask=torch.ones(row_count, dtype=torch.bool),
        temperature=0.25,
        gamma=0.0,
        false_positive_weight=false_positive_weight,
    )
    loss.backward()
    return shared_margin.grad.item()


def test_v22_empirical_risk_preserves_observed_fallback_row_prior():
    balanced_gradient = _empirical_shared_margin_gradient(
        fallback_rows=1, false_positive_weight=2.0
    )
    fallback_heavy_gradient = _empirical_shared_margin_gradient(
        fallback_rows=3, false_positive_weight=2.0
    )

    assert fallback_heavy_gradient > balanced_gradient > 0.0


def test_v22_false_switch_cost_strengthens_fallback_gradient():
    unit_cost_gradient = _empirical_shared_margin_gradient(
        fallback_rows=2, false_positive_weight=1.0
    )
    high_cost_gradient = _empirical_shared_margin_gradient(
        fallback_rows=2, false_positive_weight=3.0
    )

    assert high_cost_gradient > unit_cost_gradient


def test_joint_cascade_can_choose_safe_candidate_below_rank_winner():
    class CandidateCoordinateMargin(nn.Module):
        def forward(self, features):
            return features[..., 12:13]

    class CandidateSafetyMargin(nn.Module):
        def forward(self, features):
            return 0.25 - features[..., 12:13]

    class ConstantOpportunity(nn.Module):
        def forward(self, features):
            return features.new_ones(features.shape[0], 1)

    class SelectSafetyEvidence(nn.Module):
        def forward(self, features):
            return features[..., -2:-1]

    gate = _opportunity_cascade_gate(
        utility_direction=-1.0,
        action_mode="cascade_joint_risk_correction",
    )
    gate.cascade_quality_adapter = nn.Identity()
    gate.cascade_correction_head = nn.Sequential(
        nn.Identity(), CandidateCoordinateMargin()
    )
    gate.cascade_opportunity_head = ConstantOpportunity()
    gate.cascade_candidate_safety_head = CandidateSafetyMargin()
    gate.cascade_joint_action_head = SelectSafetyEvidence()
    gate.decision_margin = 0.5

    output = gate(**_cascade_inputs())

    rank_winner = output["moe_gate_action_margin"].masked_fill(
        ~output["moe_gate_candidate_mask"], -1e4
    ).argmax(dim=1)
    joint_winner = output["moe_gate_joint_action_margin"].masked_fill(
        ~output["moe_gate_candidate_mask"], -1e4
    ).argmax(dim=1)
    assert rank_winner.item() == 2
    assert joint_winner.item() == 0
    assert output["moe_gate_selected_query"].item() == 0
    assert bool(output["moe_gate_correction_switch"].item())


def _opportunity_loss_inputs(negative_rows=1):
    batch_size = 1 + int(negative_rows)
    box_ious = torch.tensor(
        [[0.10, 0.70, 0.20]]
        + [[0.70, 0.20, 0.60]] * int(negative_rows)
    )
    action_margin = torch.zeros(
        batch_size, 3, requires_grad=True
    )
    opportunity_margin = torch.zeros(batch_size, requires_grad=True)
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(batch_size, 3, 2, 3),
        decision_logits=torch.zeros(batch_size, 3, 3),
        box_ious=box_ious,
        default_indices=torch.zeros(batch_size, dtype=torch.long),
        candidate_mask=torch.tensor(
            [[False, True, True]] * batch_size
        ),
        action_margin=action_margin,
        row_switch_margin=opportunity_margin,
        absolute_box_logits=torch.zeros(batch_size, 3, 2),
        absolute_box_iou=torch.full((batch_size, 3), 0.5),
        setwise_temperature=0.25,
        objective="cascade_opportunity_balanced_calibrated",
    )
    return result, action_margin, opportunity_margin


def test_opportunity_objective_balances_rows_and_conditions_query_rank():
    result, action_margin, opportunity_margin = _opportunity_loss_inputs()
    assert torch.equal(
        result["row_switch_targets"], torch.tensor([1, 0])
    )
    assert torch.equal(
        result["selection_target_distribution"][0],
        torch.tensor([0.0, 1.0, 0.0]),
    )
    assert torch.equal(
        result["selection_target_distribution"][1],
        torch.zeros(3),
    )
    duplicated, _, _ = _opportunity_loss_inputs(negative_rows=4)
    assert duplicated["row_switch_loss"].item() == pytest.approx(
        result["row_switch_loss"].item()
    )

    result["loss"].backward()
    assert opportunity_margin.grad[0].item() < 0.0
    assert opportunity_margin.grad[1].item() > 0.0
    assert float(action_margin.grad[0].abs().sum()) > 0.0
    assert torch.equal(action_margin.grad[1], torch.zeros(3))


def test_verified_opportunity_trains_candidate_safety_without_prior_removal():
    batch_size = 2
    action_margin = torch.zeros(batch_size, 3, requires_grad=True)
    opportunity_margin = torch.zeros(batch_size, requires_grad=True)
    candidate_safety_margin = torch.zeros(
        batch_size, 3, requires_grad=True
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(batch_size, 3, 2, 3),
        decision_logits=torch.zeros(batch_size, 3, 3),
        box_ious=torch.tensor([
            [0.10, 0.70, 0.20],
            [0.70, 0.20, 0.60],
        ]),
        default_indices=torch.zeros(batch_size, dtype=torch.long),
        candidate_mask=torch.tensor([
            [False, True, True],
            [False, True, True],
        ]),
        action_margin=action_margin,
        row_switch_margin=opportunity_margin,
        row_safety_margin=candidate_safety_margin,
        absolute_box_logits=torch.zeros(batch_size, 3, 2),
        absolute_box_iou=torch.full((batch_size, 3), 0.5),
        setwise_temperature=0.25,
        objective="cascade_opportunity_verified_calibrated",
    )

    assert torch.equal(result["row_switch_targets"], torch.tensor([1, 0]))
    assert torch.isfinite(result["safety_loss"])
    assert result["stats"][
        "source_moe_gate_safety_positive_ratio"
    ].item() == 0.0
    result["loss"].backward()
    assert candidate_safety_margin.grad[0, 1].item() < 0.0
    assert candidate_safety_margin.grad[0, 2].item() > 0.0
    assert candidate_safety_margin.grad[1, 1].item() > 0.0
    assert candidate_safety_margin.grad[1, 2].item() > 0.0
    assert opportunity_margin.grad[0].item() < 0.0
    assert opportunity_margin.grad[1].item() > 0.0
    assert float(action_margin.grad[0].abs().sum()) > 0.0
    assert torch.equal(action_margin.grad[1], torch.zeros(3))


def test_joint_risk_loss_calibrates_fallback_and_candidate_gradients():
    batch_size = 2
    action_margin = torch.zeros(batch_size, 3, requires_grad=True)
    opportunity_margin = torch.zeros(batch_size, requires_grad=True)
    candidate_safety_margin = torch.zeros(
        batch_size, 3, requires_grad=True
    )
    joint_action_margin = torch.zeros(
        batch_size, 3, requires_grad=True
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(batch_size, 3, 2, 3),
        decision_logits=torch.zeros(batch_size, 3, 3),
        box_ious=torch.tensor([
            [0.10, 0.70, 0.20],
            [0.70, 0.20, 0.60],
        ]),
        default_indices=torch.zeros(batch_size, dtype=torch.long),
        candidate_mask=torch.tensor([
            [False, True, True],
            [False, True, True],
        ]),
        action_margin=action_margin,
        row_switch_margin=opportunity_margin,
        row_safety_margin=candidate_safety_margin,
        joint_action_margin=joint_action_margin,
        absolute_box_logits=torch.zeros(batch_size, 3, 2),
        absolute_box_iou=torch.full((batch_size, 3), 0.5),
        setwise_temperature=0.25,
        objective="cascade_joint_risk_calibrated",
    )

    target = result["joint_target_distribution"]
    assert torch.equal(target[0], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert torch.equal(target[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert result["stats"][
        "source_moe_gate_joint_action_positive_prior"
    ].item() == pytest.approx(0.5)
    assert result["stats"][
        "source_moe_gate_joint_action_prior_log_odds"
    ].item() == pytest.approx(-torch.log(torch.tensor(2.0)).item())
    assert torch.isfinite(result["joint_action_loss"])

    result["loss"].backward()
    assert joint_action_margin.grad[0, 1].item() < 0.0
    assert joint_action_margin.grad[0, 2].item() > 0.0
    assert joint_action_margin.grad[1, 1].item() > 0.0
    assert joint_action_margin.grad[1, 2].item() > 0.0


def test_pairwise_verifier_forward_loss_trains_rank_and_switch_heads():
    gate = QueryFallbackGate(
        query_dim=4,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=1,
        context_heads=4,
        context_dropout=0.0,
        action_mode="pairwise_verifier",
    )
    outputs = gate(
        query_features=torch.randn(2, 3, 4),
        candidate_scores=torch.tensor([[0.0, 2.0, 1.0],
                                       [0.0, 2.0, 1.0]]),
        shared_scores=torch.tensor([[3.0, 2.0, 1.0],
                                    [3.0, 2.0, 1.0]]),
        valid_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=outputs["moe_gate_box_logits"],
        decision_logits=outputs["moe_gate_decision_logits"],
        box_ious=torch.tensor([[0.10, 0.70, 0.20],
                               [0.70, 0.30, 0.10]]),
        default_indices=outputs["moe_gate_default_query"],
        candidate_mask=outputs["moe_gate_candidate_mask"],
        action_margin=outputs["moe_gate_action_margin"],
        row_switch_margin=outputs["moe_gate_row_switch_margin"],
        setwise_temperature=0.25,
        objective="pairwise_risk_calibrated",
    )

    result["loss"].backward()
    assert gate.utility_head.weight.grad is not None
    assert float(gate.utility_head.weight.grad.abs().sum()) > 0.0
    assert gate.pairwise_switch_head[-1].weight.grad is not None
    assert float(
        gate.pairwise_switch_head[-1].weight.grad.abs().sum()
    ) > 0.0


def test_topn_pairwise_forward_loss_trains_all_candidate_verifier_margins():
    gate = QueryFallbackGate(
        query_dim=4,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=1,
        context_heads=4,
        context_dropout=0.0,
        action_mode="topn_pairwise_verifier",
    )
    outputs = gate(
        query_features=torch.randn(2, 3, 4),
        candidate_scores=torch.tensor([[0.0, 2.0, 1.0],
                                       [0.0, 2.0, 1.0]]),
        shared_scores=torch.tensor([[3.0, 2.0, 1.0],
                                    [3.0, 2.0, 1.0]]),
        valid_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=outputs["moe_gate_box_logits"],
        decision_logits=outputs["moe_gate_decision_logits"],
        box_ious=torch.tensor([[0.10, 0.70, 0.20],
                               [0.70, 0.30, 0.10]]),
        default_indices=outputs["moe_gate_default_query"],
        candidate_mask=outputs["moe_gate_candidate_mask"],
        action_margin=outputs["moe_gate_action_margin"],
        row_switch_margin=outputs["moe_gate_row_switch_margin"],
        setwise_temperature=0.25,
        objective="pairwise_risk_calibrated",
    )

    assert result["row_switch_targets"].shape == (2, 3)
    assert result["row_utility_target"].shape == (2, 3)
    result["loss"].backward()
    assert gate.utility_head.weight.grad is not None
    assert float(gate.utility_head.weight.grad.abs().sum()) > 0.0
    assert gate.pairwise_switch_head[-1].weight.grad is not None
    assert float(
        gate.pairwise_switch_head[-1].weight.grad.abs().sum()
    ) > 0.0


def test_dual_evidence_forward_loss_trains_benefit_and_safety_heads():
    gate = QueryFallbackGate(
        query_dim=4,
        hidden_dim=12,
        candidate_top_k=2,
        context_layers=1,
        context_heads=4,
        context_dropout=0.0,
        action_mode="topn_dual_evidence_verifier",
    )
    outputs = gate(
        query_features=torch.randn(2, 3, 4),
        candidate_scores=torch.tensor([[0.0, 2.0, 1.0],
                                       [0.0, 2.0, 1.0]]),
        shared_scores=torch.tensor([[3.0, 2.0, 1.0],
                                    [3.0, 2.0, 1.0]]),
        valid_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=outputs["moe_gate_box_logits"],
        decision_logits=outputs["moe_gate_decision_logits"],
        box_ious=torch.tensor([[0.10, 0.70, 0.20],
                               [0.70, 0.30, 0.10]]),
        default_indices=outputs["moe_gate_default_query"],
        candidate_mask=outputs["moe_gate_candidate_mask"],
        mask_logits=outputs["moe_gate_mask_logits"],
        mask_ious=torch.tensor([[0.10, 0.70, 0.20],
                                [0.70, 0.30, 0.10]]),
        action_margin=outputs["moe_gate_action_margin"],
        row_switch_margin=outputs["moe_gate_row_switch_margin"],
        row_benefit_margin=outputs["moe_gate_row_benefit_margin"],
        row_safety_margin=outputs["moe_gate_row_safety_margin"],
        setwise_temperature=0.25,
        objective="topn_dual_risk_calibrated",
    )

    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert gate.pairwise_switch_head[-1].weight.grad is not None
    assert float(
        gate.pairwise_switch_head[-1].weight.grad.abs().sum()
    ) > 0.0
    assert gate.safety_switch_head[-1].weight.grad is not None
    assert float(
        gate.safety_switch_head[-1].weight.grad.abs().sum()
    ) > 0.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"candidate_top_k": 0}, "candidate_top_k"),
        ({"context_layers": -1}, "context_layers"),
        ({"context_heads": 0}, "context_heads"),
        ({"context_dropout": 1.0}, "context_dropout"),
        (
            {
                "hidden_dim": 15,
                "context_layers": 1,
                "context_heads": 4,
            },
            "divide",
        ),
    ],
)
def test_context_gate_rejects_invalid_configuration(kwargs, message):
    defaults = {
        "query_dim": 8,
        "hidden_dim": 16,
        "candidate_top_k": 2,
        "context_layers": 1,
        "context_heads": 4,
        "context_dropout": 0.0,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        QueryFallbackGate(**defaults)


def test_context_layers_require_enabled_fallback_gate():
    with pytest.raises(ValueError, match="require the fallback gate"):
        _module(use_fallback_gate=False, gate_context_layers=1)


def test_disabled_context_gate_adds_no_checkpoint_parameters():
    module = _module(
        use_fallback_gate=True,
        gate_context_layers=0,
    )

    assert module.fallback_gate.context_encoder is None
    assert not any("context_encoder" in key for key in module.state_dict())


def test_context_gate_ignores_queries_outside_the_candidate_set():
    torch.manual_seed(44)
    gate = QueryFallbackGate(
        query_dim=8,
        hidden_dim=16,
        candidate_top_k=2,
        context_layers=1,
        context_heads=4,
        context_dropout=0.0,
    ).eval()
    with torch.no_grad():
        gate.context_encoder.residual_scale.fill_(1.0)
        for head in (gate.box_head, gate.mask_head, gate.decision_head):
            head.weight.normal_()
    query_features = torch.randn(1, 5, 8)
    candidate_scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    shared_scores = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0]])
    valid_mask = torch.ones(1, 5, dtype=torch.bool)

    before = gate(
        query_features, candidate_scores, shared_scores, valid_mask
    )
    changed = query_features.clone()
    changed[:, 4] += 1000.0
    after = gate(changed, candidate_scores, shared_scores, valid_mask)

    candidate_mask = before["moe_gate_candidate_mask"]
    assert not bool(candidate_mask[0, 4].item())
    assert torch.equal(
        before["moe_gate_decision_logits"][candidate_mask],
        after["moe_gate_decision_logits"][candidate_mask],
    )


def test_context_gate_is_query_permutation_equivariant():
    torch.manual_seed(45)
    gate = QueryFallbackGate(
        query_dim=8,
        hidden_dim=16,
        candidate_top_k=3,
        context_layers=1,
        context_heads=4,
        context_dropout=0.0,
    ).eval()
    with torch.no_grad():
        gate.context_encoder.residual_scale.fill_(0.7)
        for head in (gate.box_head, gate.mask_head, gate.decision_head):
            head.weight.normal_()
    query_features = torch.randn(2, 6, 8)
    candidate_scores = torch.tensor([
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 3.0, 5.0, 2.0, 6.0, 4.0],
    ])
    shared_scores = torch.tensor([
        [7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
        [2.0, 4.0, 6.0, 3.0, 7.0, 5.0],
    ])
    valid_mask = torch.ones(2, 6, dtype=torch.bool)
    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    inverse = torch.argsort(permutation)

    expected = gate(
        query_features, candidate_scores, shared_scores, valid_mask
    )
    actual = gate(
        query_features[:, permutation],
        candidate_scores[:, permutation],
        shared_scores[:, permutation],
        valid_mask[:, permutation],
    )

    assert torch.allclose(
        actual["moe_gate_decision_logits"][:, inverse],
        expected["moe_gate_decision_logits"],
        atol=1e-6,
    )
    assert torch.equal(
        actual["moe_gate_candidate_mask"][:, inverse],
        expected["moe_gate_candidate_mask"],
    )


def test_context_gate_attention_receives_gradients_when_enabled():
    torch.manual_seed(46)
    gate = QueryFallbackGate(
        query_dim=8,
        hidden_dim=16,
        candidate_top_k=3,
        context_layers=1,
        context_heads=4,
        context_dropout=0.0,
    )
    with torch.no_grad():
        gate.context_encoder.residual_scale.fill_(0.5)
        gate.decision_head.weight.normal_()
    query_features = torch.randn(2, 6, 8)
    candidate_scores = torch.tensor([
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 3.0, 5.0, 2.0, 6.0, 4.0],
    ])
    shared_scores = candidate_scores + 1.0
    valid_mask = torch.ones(2, 6, dtype=torch.bool)

    out = gate(
        query_features, candidate_scores, shared_scores, valid_mask
    )
    loss = out["moe_gate_decision_logits"][
        out["moe_gate_candidate_mask"]
    ].square().mean()
    loss.backward()

    attention_weight = (
        gate.context_encoder.attention[0].in_proj_weight
    )
    assert attention_weight.grad is not None
    assert torch.isfinite(attention_weight.grad).all()
    assert float(attention_weight.grad.abs().sum()) > 0.0
    assert gate.context_encoder.residual_scale.grad is not None
    assert float(
        gate.context_encoder.residual_scale.grad.abs().sum()
    ) > 0.0


def test_enriched_gate_receives_standardized_raw_sources_in_source_order():
    evidence_dim = 5
    module = _module(
        use_fallback_gate=True,
        gate_use_evidence_features=True,
        gate_evidence_dim=evidence_dim,
    )
    batch = _batch(batch_size=2, num_queries=6, seed=40)
    batch["gate_candidate_feats"] = torch.randn(2, 6, evidence_dim)
    batch["valid_mask"] = torch.tensor([
        [True, True, True, True, True, False],
        [True, True, True, True, False, False],
    ])
    captured = []
    handle = module.fallback_gate.encoder[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach())
    )
    try:
        module(**batch)
    finally:
        handle.remove()

    gate_query_dim = module.fallback_gate.query_dim
    query_features = captured[0][..., :gate_query_dim]
    actual_sources = query_features[..., -len(SOURCES):]
    expected_sources = torch.stack([
        standardize_source_scores(
            batch["source_scores"][name], batch["valid_mask"]
        )
        for name in SOURCES
    ], dim=-1)
    assert torch.allclose(actual_sources, expected_sources, atol=1e-6)


@pytest.mark.parametrize("mode", ["missing", "wrong_width", "non_finite"])
def test_enriched_gate_rejects_invalid_decoder_evidence(mode):
    module = _module(
        use_fallback_gate=True,
        gate_use_evidence_features=True,
        gate_evidence_dim=7,
    )
    batch = _batch(batch_size=2, num_queries=6, seed=42)
    if mode == "wrong_width":
        batch["gate_candidate_feats"] = torch.randn(2, 6, 8)
    elif mode == "non_finite":
        batch["gate_candidate_feats"] = torch.randn(2, 6, 7)
        batch["gate_candidate_feats"][0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="gate_candidate_feats"):
        module(**batch)


def test_evidence_features_require_a_fallback_gate_and_dimension():
    with pytest.raises(ValueError, match="require the fallback gate"):
        _module(gate_use_evidence_features=True, gate_evidence_dim=8)
    with pytest.raises(ValueError, match="gate_evidence_dim"):
        _module(
            use_fallback_gate=True,
            gate_use_evidence_features=True,
        )


def test_fallback_gate_can_promote_a_positive_non_default_candidate():
    module = _module(
        use_fallback_gate=True,
        gate_candidate_top_k=8,
        gate_mask_utility_weight=0.0,
    )
    batch = _batch(batch_size=2, num_queries=8, seed=41)
    batch["source_scores"]["default"] = torch.tensor([
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ])
    with torch.no_grad():
        module.fallback_gate.decision_head.bias[0] = -10.0
        module.fallback_gate.decision_head.bias[1] = -10.0
        module.fallback_gate.decision_head.bias[2] = 10.0
    out = module(**batch)
    selected = out["selected_source_scores"].argmax(dim=1)
    assert bool(out["moe_gate_switch"].all().item())
    assert torch.all(selected != 0)
    assert torch.equal(selected, out["moe_gate_selected_query"])


def test_fallback_gate_loss_is_finite_and_trains_both_quality_heads():
    box_logits = torch.zeros(2, 4, 2, 3, requires_grad=True)
    mask_logits = torch.zeros(2, 4, 2, 3, requires_grad=True)
    decision_logits = torch.zeros(2, 4, 3, requires_grad=True)
    box_ious = torch.tensor([
        [0.60, 0.10, 0.70, 0.40],
        [0.10, 0.60, 0.40, 0.05],
    ])
    mask_ious = torch.tensor([
        [0.60, 0.10, 0.20, 0.70],
        [0.10, 0.60, 0.40, 0.05],
    ])
    candidate_mask = torch.ones(2, 4, dtype=torch.bool)
    result = compute_source_moe_fallback_gate_loss(
        box_logits=box_logits,
        decision_logits=decision_logits,
        box_ious=box_ious,
        default_indices=torch.tensor([0, 0]),
        candidate_mask=candidate_mask,
        mask_logits=mask_logits,
        mask_ious=mask_ious,
    )
    assert torch.isfinite(result["loss"])
    assert float(result["loss"]) > 0.0
    assert not bool(result["active_mask"][:, 0].any().item())
    result["loss"].backward()
    assert box_logits.grad is not None
    assert mask_logits.grad is not None
    assert decision_logits.grad is not None
    assert float(box_logits.grad.abs().sum()) > 0.0
    assert float(mask_logits.grad.abs().sum()) > 0.0
    assert float(decision_logits.grad.abs().sum()) > 0.0
    assert torch.equal(
        result["decision_targets"],
        torch.tensor([
            [1, 0, 0, 0],
            [1, 2, 2, 1],
        ]),
    )
    assert torch.equal(
        result["selection_targets"], torch.tensor([0, 2])
    )


def test_balanced_focal_objective_is_the_unchanged_default():
    kwargs = {
        "box_logits": torch.randn(2, 4, 2, 3),
        "decision_logits": torch.randn(2, 4, 3),
        "box_ious": torch.tensor([
            [0.60, 0.10, 0.70, 0.40],
            [0.10, 0.60, 0.40, 0.05],
        ]),
        "default_indices": torch.tensor([0, 0]),
        "candidate_mask": torch.ones(2, 4, dtype=torch.bool),
    }

    implicit = compute_source_moe_fallback_gate_loss(**kwargs)
    explicit = compute_source_moe_fallback_gate_loss(
        **kwargs, objective="balanced_focal"
    )

    for key in (
            "loss", "box_loss", "decision_loss", "decision_class_loss",
            "selection_loss"):
        assert torch.equal(implicit[key], explicit[key])
    assert torch.equal(implicit["decision_targets"], explicit["decision_targets"])
    assert torch.equal(implicit["selection_targets"], explicit["selection_targets"])


def test_calibrated_utility_objective_has_finite_gradients():
    box_logits = torch.zeros(2, 4, 2, 3, requires_grad=True)
    mask_logits = torch.zeros(2, 4, 2, 3, requires_grad=True)
    decision_logits = torch.zeros(2, 4, 3, requires_grad=True)
    result = compute_source_moe_fallback_gate_loss(
        box_logits=box_logits,
        decision_logits=decision_logits,
        box_ious=torch.tensor([
            [0.60, 0.10, 0.70, 0.40],
            [0.10, 0.60, 0.40, 0.05],
        ]),
        default_indices=torch.tensor([0, 0]),
        candidate_mask=torch.ones(2, 4, dtype=torch.bool),
        mask_logits=mask_logits,
        mask_ious=torch.tensor([
            [0.60, 0.10, 0.20, 0.70],
            [0.10, 0.60, 0.40, 0.05],
        ]),
        objective="calibrated_utility",
    )

    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["utility_regression_loss"])
    result["loss"].backward()
    for logits in (box_logits, mask_logits, decision_logits):
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert float(logits.grad.abs().sum()) > 0.0


def test_balanced_calibrated_utility_upweights_rare_switch_rows():
    box_logits = torch.zeros(4, 2, 2, 3)
    common = {
        "box_logits": box_logits,
        "box_ious": torch.tensor([
            [0.60, 0.10],
            [0.60, 0.10],
            [0.60, 0.10],
            [0.10, 0.60],
        ]),
        "default_indices": torch.zeros(4, dtype=torch.long),
        "candidate_mask": torch.ones(4, 2, dtype=torch.bool),
        "setwise_temperature": 0.25,
    }

    initial_logits = torch.zeros(4, 2, 3)
    initial_logits[:, 1, 2] = -2.0
    calibrated_logits = initial_logits.clone().requires_grad_()
    balanced_logits = initial_logits.clone().requires_grad_()
    calibrated = compute_source_moe_fallback_gate_loss(
        **common,
        decision_logits=calibrated_logits,
        objective="calibrated_utility",
    )
    balanced = compute_source_moe_fallback_gate_loss(
        **common,
        decision_logits=balanced_logits,
        objective="balanced_calibrated_utility",
    )

    assert balanced["selection_loss"] > calibrated["selection_loss"]
    assert torch.equal(
        balanced["utility_regression_loss"],
        calibrated["utility_regression_loss"],
    )
    assert torch.isfinite(balanced["loss"])
    calibrated["loss"].backward()
    balanced["loss"].backward()
    assert balanced_logits.grad[3, 1, 2] < calibrated_logits.grad[3, 1, 2]


def test_fallback_gate_reports_switch_confusion_and_query_match():
    decision_logits = torch.zeros(3, 3, 3)
    decision_logits[0, 1, 2] = 2.0
    decision_logits[0, 2, 2] = -2.0
    decision_logits[1, 1, 2] = 2.0
    decision_logits[1, 2, 2] = -2.0
    decision_logits[2, 1, 2] = 2.0
    decision_logits[2, 2, 2] = -2.0

    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(3, 3, 2, 3),
        decision_logits=decision_logits,
        box_ious=torch.tensor([
            [0.10, 0.60, 0.10],
            [0.60, 0.10, 0.60],
            [0.10, 0.10, 0.60],
        ]),
        default_indices=torch.zeros(3, dtype=torch.long),
        candidate_mask=torch.ones(3, 3, dtype=torch.bool),
    )

    stats = result["stats"]
    assert stats["source_moe_gate_oracle_switch_ratio"].item() == pytest.approx(
        2.0 / 3.0
    )
    assert (
        stats["source_moe_gate_oracle_switch_recall_ratio"].item() == 0.5
    )
    assert (
        stats["source_moe_gate_predicted_switch_precision_ratio"].item()
        == pytest.approx(1.0 / 3.0)
    )
    assert stats["source_moe_gate_false_switch_ratio"].item() == pytest.approx(
        2.0 / 3.0
    )
    assert stats["source_moe_gate_oracle_query_match_ratio"].item() == 0.5
    assert stats["source_moe_gate_supervised_sample_count"].item() == 3
    assert stats["source_moe_gate_oracle_switch_count"].item() == 2
    assert stats["source_moe_gate_predicted_switch_count"].item() == 3
    assert stats["source_moe_gate_beneficial_switch_count"].item() == 1
    assert stats["source_moe_gate_harmful_switch_count"].item() == 2
    assert stats["source_moe_gate_oracle_query_match_count"].item() == 1


def test_setwise_action_target_splits_tied_positive_utility():
    utility = torch.tensor([[0.0, 0.75, 0.75, -1.0]])
    active = torch.tensor([[False, True, True, True]])

    target = build_setwise_action_target(
        utility, active, temperature=0.25
    )

    assert target.shape == (1, 5)
    assert target[0, 2].item() == pytest.approx(target[0, 3].item())
    assert target[0, 2].item() > target[0, 0].item()
    assert target[0, 1].item() < 1e-6
    assert target[0, 4].item() < 1e-6
    assert target.sum().item() == pytest.approx(1.0)


def test_setwise_action_target_is_exact_fallback_without_positive_utility():
    target = build_setwise_action_target(
        torch.tensor([[0.0, -0.5, 0.0]]),
        torch.tensor([[False, True, True]]),
        temperature=0.25,
    )

    assert torch.equal(target, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_risk_separated_action_target_removes_fallback_from_positive_rows():
    target = build_risk_separated_action_target(
        torch.tensor([
            [0.0, 0.50, 0.25, -1.0],
            [0.0, -0.50, 0.0, -0.25],
        ]),
        torch.tensor([
            [False, True, True, True],
            [False, True, True, True],
        ]),
        temperature=0.25,
    )

    assert target.shape == (2, 5)
    assert target[0, 0].item() == 0.0
    assert target[0, 1].item() == 0.0
    assert target[0, 2].item() > target[0, 3].item() > 0.0
    assert target[0, 4].item() == 0.0
    assert target[0].sum().item() == pytest.approx(1.0)
    assert torch.equal(target[1], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))


def test_setwise_fallback_loss_has_finite_gradients_and_soft_ties():
    box_logits = torch.zeros(1, 4, 2, 3, requires_grad=True)
    decision_logits = torch.zeros(1, 4, 3, requires_grad=True)
    result = compute_source_moe_fallback_gate_loss(
        box_logits=box_logits,
        decision_logits=decision_logits,
        box_ious=torch.tensor([[0.10, 0.60, 0.70, 0.05]]),
        default_indices=torch.tensor([0]),
        candidate_mask=torch.ones(1, 4, dtype=torch.bool),
        setwise_temperature=0.25,
        objective="calibrated_utility",
    )

    target = result["selection_target_distribution"]
    assert target.shape == (1, 5)
    assert target[0, 2].item() == pytest.approx(target[0, 3].item())
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert torch.isfinite(box_logits.grad).all()
    assert torch.isfinite(decision_logits.grad).all()
    assert float(decision_logits.grad.abs().sum()) > 0.0


def test_calibrated_utility_penalizes_overestimation_more_than_underestimation():
    common = {
        "box_logits": torch.zeros(1, 2, 2, 3),
        "box_ious": torch.tensor([[0.60, 0.70]]),
        "default_indices": torch.tensor([0]),
        "candidate_mask": torch.ones(1, 2, dtype=torch.bool),
        "false_override_weight": 3.0,
        "objective": "calibrated_utility",
    }
    underestimated = compute_source_moe_fallback_gate_loss(
        **common,
        decision_logits=torch.tensor([[[0.0, 0.0, 0.0],
                                       [0.0, 0.0, -0.5]]]),
    )
    overestimated = compute_source_moe_fallback_gate_loss(
        **common,
        decision_logits=torch.tensor([[[0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.5]]]),
    )

    assert underestimated["decision_utility"][0, 1].item() == pytest.approx(0.0)
    assert overestimated["utility_regression_loss"] > (
        underestimated["utility_regression_loss"]
    )


def test_calibrated_utility_regresses_the_deployed_expected_utility_margin():
    box_logits = torch.zeros(1, 2, 2, 3, requires_grad=True)
    mask_logits = torch.zeros(1, 2, 2, 3, requires_grad=True)
    threshold_weights = torch.tensor([2.0, 1.0])
    action_margin = transition_logits_expected_utility(
        box_logits, threshold_weights, break_cost=2.0
    ) + 0.25 * transition_logits_expected_utility(
        mask_logits, threshold_weights, break_cost=2.0
    )
    result = compute_source_moe_fallback_gate_loss(
        box_logits=box_logits,
        mask_logits=mask_logits,
        decision_logits=torch.zeros(1, 2, 3, requires_grad=True),
        box_ious=torch.tensor([[0.10, 0.60]]),
        mask_ious=torch.tensor([[0.10, 0.60]]),
        default_indices=torch.tensor([0]),
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
        objective="calibrated_utility",
        action_margin=action_margin,
    )

    result["utility_regression_loss"].backward()
    assert box_logits.grad is not None
    assert mask_logits.grad is not None
    assert float(box_logits.grad.abs().sum()) > 0.0
    assert float(mask_logits.grad.abs().sum()) > 0.0


def test_hierarchical_loss_trains_candidate_rank_and_row_switch_separately():
    candidate_margin = torch.zeros(3, 3, requires_grad=True)
    row_switch_margin = torch.zeros(3, requires_grad=True)
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(3, 3, 2, 3, requires_grad=True),
        decision_logits=torch.zeros(3, 3, 3, requires_grad=True),
        box_ious=torch.tensor([
            [0.10, 0.60, 0.70],
            [0.60, 0.10, 0.20],
            [0.10, 0.20, 0.70],
        ]),
        default_indices=torch.zeros(3, dtype=torch.long),
        candidate_mask=torch.ones(3, 3, dtype=torch.bool),
        objective="balanced_calibrated_utility",
        setwise_temperature=0.25,
        action_margin=candidate_margin,
        row_switch_margin=row_switch_margin,
    )

    assert result["selection_target_distribution"].shape == (3, 3)
    assert result["selection_target_distribution"][0].sum().item() == (
        pytest.approx(1.0)
    )
    assert result["selection_target_distribution"][1].sum().item() == (
        pytest.approx(0.0)
    )
    assert torch.isfinite(result["row_switch_loss"])
    result["loss"].backward()
    assert candidate_margin.grad is not None
    assert row_switch_margin.grad is not None
    assert float(candidate_margin.grad.abs().sum()) > 0.0
    assert float(row_switch_margin.grad.abs().sum()) > 0.0


def test_hierarchical_risk_objective_preserves_row_prior_without_rebalancing():
    common = {
        "box_logits": torch.zeros(4, 2, 2, 3),
        "decision_logits": torch.zeros(4, 2, 3),
        "box_ious": torch.tensor([
            [0.60, 0.10],
            [0.60, 0.10],
            [0.60, 0.10],
            [0.10, 0.60],
        ]),
        "default_indices": torch.zeros(4, dtype=torch.long),
        "candidate_mask": torch.ones(4, 2, dtype=torch.bool),
        "action_margin": torch.zeros(4, 2, requires_grad=True),
        "setwise_temperature": 0.25,
        "false_override_weight": 2.0,
    }
    balanced_margin = torch.zeros(4, requires_grad=True)
    risk_margin = torch.zeros(4, requires_grad=True)

    balanced = compute_source_moe_fallback_gate_loss(
        **common,
        row_switch_margin=balanced_margin,
        objective="balanced_calibrated_utility",
    )
    risk = compute_source_moe_fallback_gate_loss(
        **common,
        row_switch_margin=risk_margin,
        objective="hierarchical_risk_calibrated",
    )
    balanced["row_switch_loss"].backward()
    risk["row_switch_loss"].backward()

    assert abs(risk_margin.grad[3]) < abs(balanced_margin.grad[3])
    assert abs(risk_margin.grad[0]) > abs(balanced_margin.grad[0])
    assert torch.isfinite(risk["loss"])


def test_hierarchical_risk_objective_requires_row_switch_head():
    with pytest.raises(ValueError, match="requires row_switch_margin"):
        compute_source_moe_fallback_gate_loss(
            box_logits=torch.zeros(1, 2, 2, 3),
            decision_logits=torch.zeros(1, 2, 3),
            box_ious=torch.tensor([[0.60, 0.10]]),
            default_indices=torch.tensor([0]),
            candidate_mask=torch.ones(1, 2, dtype=torch.bool),
            objective="hierarchical_risk_calibrated",
        )


def test_pairwise_risk_targets_the_candidate_selected_by_deployment():
    common = {
        "box_logits": torch.zeros(1, 3, 2, 3),
        "decision_logits": torch.zeros(1, 3, 3),
        "box_ious": torch.tensor([[0.10, 0.70, 0.10]]),
        "default_indices": torch.tensor([0]),
        "candidate_mask": torch.ones(1, 3, dtype=torch.bool),
        "action_margin": torch.tensor([[0.0, 0.0, 1.0]]),
        "row_switch_margin": torch.zeros(1),
        "setwise_temperature": 0.25,
    }
    hierarchical = compute_source_moe_fallback_gate_loss(
        **common,
        objective="hierarchical_risk_calibrated",
    )
    pairwise = compute_source_moe_fallback_gate_loss(
        **common,
        objective="pairwise_risk_calibrated",
    )

    assert hierarchical["row_switch_targets"].item() == 1
    assert pairwise["row_switch_targets"].item() == 0
    assert hierarchical["row_utility_target"].item() > 0.0
    assert pairwise["row_utility_target"].item() == pytest.approx(0.0)


def test_pairwise_risk_trains_candidate_rank_on_fallback_rows():
    action_margin = torch.zeros(1, 3, requires_grad=True)
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(1, 3, 2, 3),
        decision_logits=torch.zeros(1, 3, 3),
        box_ious=torch.tensor([[0.70, 0.10, 0.30]]),
        default_indices=torch.tensor([0]),
        candidate_mask=torch.ones(1, 3, dtype=torch.bool),
        action_margin=action_margin,
        row_switch_margin=torch.zeros(1, requires_grad=True),
        setwise_temperature=0.25,
        objective="pairwise_risk_calibrated",
    )

    assert result["row_switch_targets"].item() == 0
    assert result["selection_target_distribution"][0, 2] > (
        result["selection_target_distribution"][0, 1]
    )
    result["selection_loss"].backward()
    assert action_margin.grad is not None
    assert float(action_margin.grad.abs().sum()) > 0.0


def test_topn_pairwise_risk_regresses_each_candidate_margin():
    action_margin = torch.zeros(1, 3, requires_grad=True)
    verifier_margin = torch.zeros(1, 3, requires_grad=True)
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(1, 3, 2, 3),
        decision_logits=torch.zeros(1, 3, 3),
        box_ious=torch.tensor([[0.10, 0.70, 0.10]]),
        default_indices=torch.tensor([0]),
        candidate_mask=torch.ones(1, 3, dtype=torch.bool),
        action_margin=action_margin,
        row_switch_margin=verifier_margin,
        setwise_temperature=0.25,
        objective="pairwise_risk_calibrated",
    )

    assert torch.equal(
        result["row_switch_targets"],
        torch.tensor([[False, True, False]]).long(),
    )
    assert result["verifier_target_distribution"].shape == (1, 4)
    assert result["verifier_target_distribution"][0, 2] > (
        result["verifier_target_distribution"][0, 0]
    )
    assert result["stats"][
        "source_moe_gate_row_target_switch_count"
    ].item() == pytest.approx(1.0)
    result["loss"].backward()
    assert action_margin.grad is not None
    assert verifier_margin.grad is not None
    assert float(action_margin.grad.abs().sum()) > 0.0
    assert float(verifier_margin.grad.abs().sum()) > 0.0


def test_topn_risk_separates_fallback_and_neutral_verifier_targets():
    common = {
        "box_logits": torch.zeros(2, 4, 2, 3),
        "decision_logits": torch.zeros(2, 4, 3),
        "box_ious": torch.tensor([
            [0.30, 0.70, 0.30, 0.10],
            [0.60, 0.10, 0.30, 0.60],
        ]),
        "default_indices": torch.zeros(2, dtype=torch.long),
        "candidate_mask": torch.ones(2, 4, dtype=torch.bool),
        "action_margin": torch.zeros(2, 4),
        "row_switch_margin": torch.zeros(2, 4),
        "setwise_temperature": 0.25,
    }

    legacy = compute_source_moe_fallback_gate_loss(
        **common, objective="pairwise_risk_calibrated"
    )
    risk = compute_source_moe_fallback_gate_loss(
        **common, objective="topn_risk_calibrated"
    )

    legacy_target = legacy["verifier_target_distribution"]
    risk_target = risk["verifier_target_distribution"]
    assert legacy_target[0, 0].item() > 0.0
    assert risk_target[0, 0].item() == 0.0
    assert risk_target[0, 2].item() == pytest.approx(1.0)
    assert torch.equal(
        risk_target[1], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    )

    utility = risk["row_utility_target"]
    verifier_utility = risk["verifier_utility_target"]
    assert utility[0, 1].item() > 0.0
    assert verifier_utility[0, 1].item() == pytest.approx(
        utility[0, 1].item()
    )
    assert utility[0, 2].item() == pytest.approx(0.0)
    assert verifier_utility[0, 2].item() == pytest.approx(-0.25)
    assert utility[0, 3].item() < 0.0
    assert verifier_utility[0, 3].item() == pytest.approx(
        utility[0, 3].item()
    )

    assert torch.equal(
        legacy["verifier_utility_target"], legacy["row_utility_target"]
    )
    assert torch.equal(
        legacy["selection_target_distribution"],
        risk["selection_target_distribution"],
    )


@pytest.mark.parametrize(
    "row_switch_margin,setwise_temperature",
    [
        (torch.zeros(1), 0.25),
        (torch.zeros(1, 3), 0.0),
    ],
)
def test_topn_risk_requires_candidate_margins_and_positive_temperature(
        row_switch_margin, setwise_temperature):
    with pytest.raises(ValueError, match="topn_risk_calibrated requires"):
        compute_source_moe_fallback_gate_loss(
            box_logits=torch.zeros(1, 3, 2, 3),
            decision_logits=torch.zeros(1, 3, 3),
            box_ious=torch.tensor([[0.30, 0.70, 0.10]]),
            default_indices=torch.tensor([0]),
            candidate_mask=torch.ones(1, 3, dtype=torch.bool),
            action_margin=torch.zeros(1, 3),
            row_switch_margin=row_switch_margin,
            setwise_temperature=setwise_temperature,
            objective="topn_risk_calibrated",
        )


def test_dual_risk_safety_gradients_reject_neutral_and_negative_candidates():
    benefit_margin = torch.zeros(1, 4, requires_grad=True)
    safety_margin = torch.zeros(1, 4, requires_grad=True)
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(1, 4, 2, 3),
        decision_logits=torch.zeros(1, 4, 3),
        box_ious=torch.tensor([[0.30, 0.70, 0.30, 0.10]]),
        default_indices=torch.tensor([0]),
        candidate_mask=torch.ones(1, 4, dtype=torch.bool),
        action_margin=torch.zeros(1, 4),
        row_switch_margin=torch.minimum(benefit_margin, safety_margin),
        row_benefit_margin=benefit_margin,
        row_safety_margin=safety_margin,
        setwise_temperature=0.25,
        objective="topn_dual_risk_calibrated",
    )

    result["safety_loss"].backward()
    assert safety_margin.grad[0, 0].item() == 0.0
    assert safety_margin.grad[0, 1].item() < 0.0
    assert safety_margin.grad[0, 2].item() > 0.0
    assert safety_margin.grad[0, 3].item() > 0.0


def test_dual_risk_requires_benefit_and_safety_margins():
    with pytest.raises(
            ValueError, match=r"requires \[B,Q\] benefit and safety"):
        compute_source_moe_fallback_gate_loss(
            box_logits=torch.zeros(1, 3, 2, 3),
            decision_logits=torch.zeros(1, 3, 3),
            box_ious=torch.tensor([[0.30, 0.70, 0.10]]),
            default_indices=torch.tensor([0]),
            candidate_mask=torch.ones(1, 3, dtype=torch.bool),
            action_margin=torch.zeros(1, 3),
            row_switch_margin=torch.zeros(1, 3),
            setwise_temperature=0.25,
            objective="topn_dual_risk_calibrated",
        )


def test_absolute_quality_objective_requires_dense_predictions():
    with pytest.raises(ValueError, match="requires absolute quality"):
        compute_source_moe_fallback_gate_loss(
            box_logits=torch.zeros(1, 3, 2, 3),
            decision_logits=torch.zeros(1, 3, 3),
            box_ious=torch.tensor([[0.30, 0.70, 0.10]]),
            default_indices=torch.tensor([0]),
            candidate_mask=torch.ones(1, 3, dtype=torch.bool),
            action_margin=torch.zeros(1, 3),
            setwise_temperature=0.25,
            objective="topn_absolute_quality_calibrated",
        )


def test_fallback_gate_rejects_misaligned_row_switch_margin():
    with pytest.raises(ValueError, match="row_switch_margin must be finite"):
        compute_source_moe_fallback_gate_loss(
            box_logits=torch.zeros(2, 3, 2, 3),
            decision_logits=torch.zeros(2, 3, 3),
            box_ious=torch.zeros(2, 3),
            default_indices=torch.zeros(2, dtype=torch.long),
            candidate_mask=torch.ones(2, 3, dtype=torch.bool),
            action_margin=torch.zeros(2, 3),
            row_switch_margin=torch.zeros(2, 2),
        )


def test_pairwise_risk_objective_requires_pairwise_margin():
    with pytest.raises(ValueError, match="pairwise_risk_calibrated requires"):
        compute_source_moe_fallback_gate_loss(
            box_logits=torch.zeros(1, 2, 2, 3),
            decision_logits=torch.zeros(1, 2, 3),
            box_ious=torch.tensor([[0.60, 0.10]]),
            default_indices=torch.tensor([0]),
            candidate_mask=torch.ones(1, 2, dtype=torch.bool),
            action_margin=torch.zeros(1, 2),
            objective="pairwise_risk_calibrated",
        )


def test_hierarchical_row_veto_prevents_candidate_max_false_switch():
    result = compute_source_moe_fallback_gate_loss(
        box_logits=torch.zeros(1, 3, 2, 3),
        decision_logits=torch.zeros(1, 3, 3),
        box_ious=torch.tensor([[0.60, 0.10, 0.20]]),
        default_indices=torch.tensor([0]),
        candidate_mask=torch.ones(1, 3, dtype=torch.bool),
        action_margin=torch.tensor([[0.0, 100.0, 50.0]]),
        row_switch_margin=torch.tensor([-1.0]),
        objective="balanced_calibrated_utility",
    )

    assert result["stats"][
        "source_moe_gate_predicted_switch_count"
    ].item() == 0.0
    assert result["stats"][
        "source_moe_gate_harmful_switch_count"
    ].item() == 0.0
