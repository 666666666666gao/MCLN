import copy

import math

import pytest
import torch
from torch.nn import functional as F
from types import MethodType
from types import SimpleNamespace

from main_utils import BaseTrainTester
from models.losses import (
    build_joint_query_mask_candidate_mask,
    compute_hungarian_loss,
    compute_joint_query_mask_candidate_loss,
    dice_loss,
    lovasz_hinge_loss,
    sigmoid_focal_loss,
)
from models.joint_query_quality import (
    JOINT_QUERY_GATE_EVIDENCE_DIM,
    JOINT_QUERY_GATE_EVIDENCE_NAMES,
    SOURCE_DISTRIBUTION_RELIABILITY_DIM,
    JointQualityAdaptiveSourceMixer,
    JointQueryQualityReranker,
    QuerySuperpointMaskRefiner,
    _straight_through_rank_normalize,
    build_joint_query_gate_evidence,
    build_source_distribution_reliability_features,
    compute_joint_query_quality_loss,
    compute_joint_query_source_mix_alignment_loss,
    joint_query_target_quality,
    ordinal_threshold_logits,
    smooth_metric_aligned_query_utility,
    summarize_joint_query_residual,
)
from models.mask_fusion import (
    QUERY_MASK_SOURCE_EVIDENCE_DIM,
    apply_query_mask_calibration,
    apply_query_superpoint_mask_residual,
    fuse_query_mask_logits,
)


def _inputs(batch=2, queries=5, features=8):
    torch.manual_seed(7)
    values = torch.randn(batch, queries, features)
    baseline = torch.tensor([
        [0.9, 0.2, 0.7, 0.1, 0.4],
        [0.1, 0.8, 0.3, 0.6, 0.2],
    ])[:batch, :queries]
    valid = torch.ones(batch, queries, dtype=torch.bool)
    return values, baseline, valid


def _gate_outputs(batch=2, queries=5):
    torch.manual_seed(71)
    candidate_mask = torch.tensor([
        [False, True, True, False, False],
        [True, False, False, True, False],
    ])[:batch, :queries]
    default = torch.tensor([0, 1])[:batch]
    selected = torch.tensor([2, 3])[:batch].clamp(max=queries - 1)
    anchor = torch.tensor([4, 0])[:batch].clamp(max=queries - 1)
    return {
        "moe_gate_candidate_mask": candidate_mask,
        "moe_gate_default_query": default.clamp(max=queries - 1),
        "moe_gate_selected_query": selected,
        "moe_gate_action_anchor_query": anchor,
        "moe_candidate_scores": torch.randn(batch, queries),
        "moe_gate_expected_utility": torch.randn(batch, queries),
        "moe_gate_direct_utility": torch.randn(batch, queries),
        "moe_gate_action_margin": torch.randn(batch, queries),
        "moe_gate_box_logits": torch.randn(batch, queries, 2, 3),
        "moe_gate_mask_logits": torch.randn(batch, queries, 2, 3),
        "moe_gate_decision_logits": torch.randn(batch, queries, 3),
    }


def test_v46_gate_evidence_contract_uses_only_deployed_gate_outputs():
    valid = torch.ones(2, 5, dtype=torch.bool)
    outputs = _gate_outputs()
    evidence = build_joint_query_gate_evidence(outputs, valid)

    assert len(JOINT_QUERY_GATE_EVIDENCE_NAMES) == 24
    assert JOINT_QUERY_GATE_EVIDENCE_DIM == 24
    assert evidence.shape == (2, 5, 24)
    assert torch.isfinite(evidence).all()
    assert bool(((evidence >= 0.0) & (evidence <= 1.0)).all().item())
    torch.testing.assert_close(
        evidence[..., 9:15].reshape(2, 5, 2, 3).sum(-1),
        torch.ones(2, 5, 2),
    )
    torch.testing.assert_close(
        evidence[..., 15:21].reshape(2, 5, 2, 3).sum(-1),
        torch.ones(2, 5, 2),
    )
    torch.testing.assert_close(
        evidence[..., 21:24].sum(-1), torch.ones(2, 5)
    )


def test_v46_gate_evidence_is_query_permutation_equivariant():
    valid = torch.ones(2, 5, dtype=torch.bool)
    outputs = _gate_outputs()
    direct = build_joint_query_gate_evidence(outputs, valid)
    permutation = torch.tensor([3, 1, 4, 0, 2])
    inverse = permutation.argsort()
    permuted_outputs = {}
    index_keys = {
        "moe_gate_default_query",
        "moe_gate_selected_query",
        "moe_gate_action_anchor_query",
    }
    for key, value in outputs.items():
        if key in index_keys:
            permuted_outputs[key] = inverse[value]
        else:
            permuted_outputs[key] = value[:, permutation]
    permuted = build_joint_query_gate_evidence(
        permuted_outputs, valid[:, permutation]
    )

    torch.testing.assert_close(direct, permuted[:, inverse])


def test_v46_gate_evidence_rejects_missing_or_invalid_contracts():
    valid = torch.ones(2, 5, dtype=torch.bool)
    outputs = _gate_outputs()
    outputs.pop("moe_gate_box_logits")
    with pytest.raises(ValueError, match="outputs are missing"):
        build_joint_query_gate_evidence(outputs, valid)

    outputs = _gate_outputs()
    outputs["moe_gate_action_margin"] = torch.full((2, 5), float("nan"))
    with pytest.raises(ValueError, match="action_margin"):
        build_joint_query_gate_evidence(outputs, valid)


def test_v46_gate_evidence_preserves_identity_and_detaches_gate_inputs():
    features, baseline, valid = _inputs()
    gate_evidence = torch.rand(
        *baseline.shape, JOINT_QUERY_GATE_EVIDENCE_DIM, requires_grad=True
    )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_gate_evidence=True, detach_inputs=True,
    )

    identity = model(features, baseline, valid, gate_evidence=gate_evidence)
    assert torch.equal(identity["scores"], baseline)
    assert torch.count_nonzero(identity["residual"]) == 0

    with torch.no_grad():
        model.residual_head.weight.normal_(mean=0.0, std=0.1)
    outputs = model(features, baseline, valid, gate_evidence=gate_evidence)
    outputs["scores"].sum().backward()
    assert gate_evidence.grad is None
    assert model.input_projection[1].weight.grad is not None

    with pytest.raises(ValueError, match="gate_evidence"):
        model(features, baseline, valid)
    with pytest.raises(ValueError, match="requires use_gate_evidence"):
        JointQueryQualityReranker(8)(
            features, baseline, valid, gate_evidence=gate_evidence
        )


def test_zero_initialization_is_exact_baseline_identity():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    ).eval()
    outputs = model(features, baseline, valid)
    assert torch.equal(outputs["residual"], torch.zeros_like(baseline))
    assert torch.equal(outputs["selected_indices"], baseline.argmax(1))
    assert torch.equal(outputs["scores"], baseline)


def test_parent_score_preserving_margin_is_safe_at_step_zero():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        candidate_promotion_margin=0.05,
    ).eval()

    outputs = model(features, baseline, valid)
    row = torch.arange(baseline.shape[0])
    parent = baseline.argmax(dim=1)
    parent_mask = torch.zeros_like(valid)
    parent_mask[row, parent] = True
    expected_residual = torch.where(
        parent_mask, torch.zeros_like(baseline),
        torch.full_like(baseline, -0.05),
    )

    assert torch.equal(outputs["learned_residual"], torch.zeros_like(baseline))
    torch.testing.assert_close(outputs["residual"], expected_residual)
    torch.testing.assert_close(
        outputs["scores"], baseline + expected_residual
    )
    assert torch.equal(outputs["scores"][row, parent], baseline[row, parent])
    assert torch.equal(outputs["selected_indices"], parent)


def test_parent_score_preserving_candidate_must_clear_promotion_margin():
    torch.manual_seed(5201)
    features = torch.randn(1, 3, 8)
    baseline = torch.tensor([[1.0, 0.9, 0.2]])
    valid = torch.ones_like(baseline, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=1.0, preserve_parent_score=True,
        candidate_promotion_margin=0.05,
    ).eval()
    with torch.no_grad():
        model.residual_head.weight.zero_()
        model.residual_head.bias.fill_(torch.atanh(torch.tensor(0.2)))

    outputs = model(features, baseline, valid)

    torch.testing.assert_close(
        outputs["learned_residual"], torch.full_like(baseline, 0.2)
    )
    torch.testing.assert_close(outputs["scores"][0, 0], baseline[0, 0])
    torch.testing.assert_close(outputs["scores"][0, 1], torch.tensor(1.05))
    assert outputs["selected_indices"].item() == 1
    outputs["scores"].sum().backward()
    assert torch.isfinite(model.residual_head.bias.grad).all()
    assert model.residual_head.bias.grad.abs().sum().item() > 0.0


def test_parent_score_contract_rejects_unsafe_configuration():
    with pytest.raises(ValueError, match="requires preserved parent score"):
        JointQueryQualityReranker(
            8, candidate_promotion_margin=0.05
        )
    with pytest.raises(ValueError, match="must be below max_delta"):
        JointQueryQualityReranker(
            8, max_delta=0.05, preserve_parent_score=True,
            candidate_promotion_margin=0.05,
        )



def test_v52_parent_transition_step_zero_is_safe_and_topk_bounded():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        candidate_promotion_margin=0.05,
        use_parent_transition_advantage=True,
        parent_transition_break_cost=4.0,
        parent_transition_candidate_top_k=2,
    ).eval()

    outputs = model(features, baseline, valid)
    row = torch.arange(baseline.shape[0])
    parent = baseline.argmax(dim=1)
    candidate_mask = outputs["parent_transition_candidate_mask"]
    expected_nonparent = candidate_mask.clone()
    expected_nonparent[row, parent] = False

    assert candidate_mask.sum(dim=1).tolist() == [2, 2]
    assert torch.equal(outputs["selected_indices"], parent)
    assert torch.equal(outputs["scores"][row, parent], baseline[row, parent])
    torch.testing.assert_close(
        outputs["residual"][expected_nonparent],
        torch.full_like(outputs["residual"][expected_nonparent], -0.05),
    )
    assert torch.count_nonzero(outputs["residual"][~candidate_mask]) == 0
    assert torch.count_nonzero(outputs["parent_transition_logits"]) == 0
    assert torch.count_nonzero(outputs["parent_transition_advantage"]) == 0


def test_v52_parent_transition_loss_is_balanced_and_trainable():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        candidate_promotion_margin=0.05,
        use_parent_transition_advantage=True,
        parent_transition_break_cost=4.0,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    box = torch.tensor([
        [0.80, 0.10, 0.70, 0.20, 0.60],
        [0.10, 0.10, 0.30, 0.70, 0.20],
    ])
    mask = torch.zeros_like(box)

    supervision = compute_joint_query_quality_loss(
        outputs, box, mask,
        listwise_loss_weight=0.0,
        transition_loss_weight=1.0,
        transition_break_cost=4.0,
        transition_neutral_weight=0.25,
        quality_loss_weight=0.0,
        anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0,
    )

    assert supervision["transition_loss"].item() > 0.0
    torch.testing.assert_close(
        supervision["loss"], supervision["transition_loss"]
    )
    assert supervision["stats"]["transition_break_target_ratio"] > 0
    assert supervision["stats"]["transition_fix_target_ratio"] > 0
    supervision["loss"].backward()
    gradient = model.parent_transition_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0
    assert model.residual_head.weight.grad is None
    assert model.quality_head.weight.grad is not None
    assert torch.count_nonzero(model.quality_head.weight.grad) == 0


def test_v52_parent_transition_rejects_unsafe_contracts():
    with pytest.raises(ValueError, match="requires preserved parent score"):
        JointQueryQualityReranker(
            8, use_parent_transition_advantage=True,
        )
    with pytest.raises(ValueError, match="restriction requires"):
        JointQueryQualityReranker(
            8, parent_transition_candidate_top_k=8,
        )
    with pytest.raises(ValueError, match="break cost must be positive"):
        JointQueryQualityReranker(
            8, preserve_parent_score=True,
            use_parent_transition_advantage=True,
            parent_transition_break_cost=0.0,
        )


def test_v53_factorized_hit_step_zero_is_safe_nested_and_topk_bounded():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        candidate_promotion_margin=0.05,
        use_factorized_hit_advantage=True,
        parent_transition_break_cost=4.0,
        parent_transition_candidate_top_k=2,
    ).eval()

    outputs = model(features, baseline, valid)
    row = torch.arange(baseline.shape[0])
    parent = baseline.argmax(dim=1)
    candidate_mask = outputs["parent_transition_candidate_mask"]
    probabilities = outputs["factorized_hit_probabilities"]

    assert candidate_mask.sum(dim=1).tolist() == [2, 2]
    assert torch.equal(outputs["selected_indices"], parent)
    assert torch.equal(outputs["scores"][row, parent], baseline[row, parent])
    assert bool((probabilities[..., 1] <= probabilities[..., 0]).all())
    assert torch.count_nonzero(outputs["factorized_hit_logits"]) > 0
    assert bool((outputs["parent_transition_advantage"] <= 0.0).all())
    assert torch.count_nonzero(outputs["residual"][~candidate_mask]) == 0


def test_v53_factorized_hit_loss_is_dense_balanced_and_trainable():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        candidate_promotion_margin=0.05,
        use_factorized_hit_advantage=True,
        parent_transition_break_cost=4.0,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    box = torch.tensor([
        [0.80, 0.10, 0.70, 0.20, 0.60],
        [0.10, 0.10, 0.30, 0.70, 0.20],
    ])
    mask = torch.zeros_like(box)

    supervision = compute_joint_query_quality_loss(
        outputs, box, mask,
        listwise_loss_weight=0.0,
        transition_loss_weight=0.0,
        factorized_hit_loss_weight=1.0,
        quality_loss_weight=0.0,
        anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0,
    )

    assert supervision["factorized_hit_loss"].item() > 0.0
    torch.testing.assert_close(
        supervision["loss"], supervision["factorized_hit_loss"]
    )
    assert supervision["stats"]["factorized_hit_025_target_ratio"] > 0
    assert supervision["stats"]["factorized_hit_050_target_ratio"] > 0
    supervision["loss"].backward()
    gradient = model.factorized_hit_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0
    assert model.residual_head.weight.grad is None
    assert model.quality_head.weight.grad is not None
    assert torch.count_nonzero(model.quality_head.weight.grad) == 0


def test_v56_factorized_parent_pair_loss_learns_fix_and_break_order():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_factorized_hit_advantage=True,
        use_factorized_nested_dominance=True,
        factorized_hit_break_cost=1.0,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    # Row 0 parent q0 is a hit: q1/q3/q4 are break candidates.
    # Row 1 parent q1 is a miss: q2/q3 are fix candidates.
    box = torch.tensor([
        [0.80, 0.10, 0.70, 0.20, 0.10],
        [0.10, 0.10, 0.70, 0.80, 0.20],
    ])
    mask = torch.zeros_like(box)

    supervision = compute_joint_query_quality_loss(
        outputs, box, mask,
        listwise_loss_weight=0.0,
        transition_loss_weight=0.0,
        factorized_hit_loss_weight=0.0,
        factorized_pair_loss_weight=1.0,
        quality_loss_weight=0.0,
        anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0,
        anchor_margin=0.05,
        anchor_margin_050=0.10,
    )

    assert supervision["factorized_pair_loss"].item() > 0.0
    torch.testing.assert_close(
        supervision["loss"], supervision["factorized_pair_loss"]
    )
    assert supervision["stats"]["factorized_hard_anchor_025_protect_ratio"] > 0
    assert supervision["stats"]["factorized_hard_anchor_050_repair_ratio"] > 0
    supervision["loss"].backward()
    gradient = model.factorized_hit_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_v62_decomposed_transition_step_zero_is_parent_identity():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_decomposed_transition_advantage=True,
        parent_transition_break_cost=2.0,
        parent_transition_candidate_top_k=5,
    ).eval()
    outputs = model(features, baseline, valid)
    assert torch.equal(outputs["scores"], baseline)
    assert torch.equal(
        outputs["selected_indices"], baseline.argmax(dim=1)
    )
    assert torch.count_nonzero(outputs["residual"]) == 0
    assert outputs["decomposed_transition_logits"].shape == (2, 5, 2, 2)
    logits = outputs["decomposed_transition_logits"]
    torch.testing.assert_close(logits[..., 0], torch.zeros((2, 5, 2)))
    torch.testing.assert_close(
        logits[..., 1], torch.full((2, 5, 2), math.log(2.0))
    )
    # The direction prior is exactly the configured cost threshold.
    torch.testing.assert_close(
        outputs["decomposed_fix_break_utility"],
        torch.zeros((2, 5, 2)),
    )


def test_v62_decomposed_transition_loss_is_group_balanced_and_trainable():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        preserve_parent_score=True,
        use_decomposed_transition_advantage=True,
        parent_transition_break_cost=4.0,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    box = torch.tensor([
        [0.80, 0.10, 0.70, 0.20, 0.10],
        [0.10, 0.10, 0.70, 0.80, 0.20],
    ])
    supervision = compute_joint_query_quality_loss(
        outputs, box, torch.zeros_like(box),
        listwise_loss_weight=0.0,
        transition_loss_weight=1.0,
        transition_break_cost=4.0,
        quality_loss_weight=0.0,
        anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0,
    )
    assert supervision["transition_loss"].item() > 0.0
    torch.testing.assert_close(
        supervision["loss"], supervision["transition_loss"]
    )
    stats = supervision["stats"]
    assert stats["decomposed_change_025_target_ratio"] > 0
    assert stats["decomposed_break_025_target_ratio"] > 0
    assert stats["decomposed_fix_025_target_ratio"] > 0
    supervision["loss"].backward()
    gradient = model.decomposed_transition_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0
    assert model.parent_transition_head is None
    assert model.factorized_hit_head is None
    assert model.residual_head.weight.grad is None


def test_v62c_decomposed_counterfactual_cost_receipts_are_detached():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        preserve_parent_score=True,
        use_decomposed_transition_advantage=True,
        parent_transition_break_cost=4.0,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    costs = outputs["decomposed_counterfactual_costs"]
    selected = outputs["decomposed_counterfactual_selected_indices"]
    assert costs.shape == (7,)
    assert selected.shape == (2, 7)
    assert not costs.requires_grad
    assert not selected.requires_grad
    box = torch.tensor([
        [0.80, 0.10, 0.70, 0.20, 0.10],
        [0.10, 0.10, 0.70, 0.80, 0.20],
    ])
    supervision = compute_joint_query_quality_loss(
        outputs, box, torch.zeros_like(box),
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0,
    )
    for suffix in ("1p25", "1p50", "2p00", "2p50", "3p00", "3p50", "4p00"):
        for metric in ("switch_ratio", "fix025", "break025",
                       "fix050", "break050"):
            value = supervision["stats"][
                "decomposed_cf_cost{}_{}".format(suffix, metric)
            ]
            assert torch.isfinite(value)


def test_v62_decomposed_transition_rejects_unsafe_contracts():
    with pytest.raises(ValueError, match="requires preserved parent"):
        JointQueryQualityReranker(
            8, use_decomposed_transition_advantage=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        JointQueryQualityReranker(
            8, preserve_parent_score=True,
            use_decomposed_transition_advantage=True,
            use_factorized_hit_advantage=True,
        )


def test_v61_factorized_counterfactual_cost_receipts_are_detached():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_factorized_hit_advantage=True,
        use_factorized_nested_dominance=True,
        factorized_hit_break_cost=1.0,
        parent_transition_candidate_top_k=5,
    )
    with torch.no_grad():
        model.factorized_hit_head.weight.normal_(mean=0.0, std=0.2)
        model.factorized_hit_head.bias.copy_(torch.tensor((0.1, -0.2)))
    outputs = model(features, baseline, valid)
    costs = outputs["factorized_counterfactual_costs"]
    selected = outputs["factorized_counterfactual_selected_indices"]
    torch.testing.assert_close(
        costs, costs.new_tensor((1.0, 1.25, 1.5, 2.0, 3.0, 4.0))
    )
    assert selected.shape == (features.shape[0], costs.numel())
    assert not costs.requires_grad
    assert not selected.requires_grad
    torch.testing.assert_close(selected[:, 0], outputs["selected_indices"])

    box = torch.tensor([
        [0.80, 0.10, 0.70, 0.20, 0.10],
        [0.10, 0.10, 0.70, 0.80, 0.20],
    ])
    supervision = compute_joint_query_quality_loss(
        outputs, box, torch.zeros_like(box),
        listwise_loss_weight=0.0, quality_loss_weight=0.0,
        anchor_loss_weight=0.0, factorized_hit_loss_weight=0.0,
        factorized_pair_loss_weight=0.0, source_mix_loss_weight=0.0,
    )
    for cost_suffix in ("1p00", "1p25", "1p50", "2p00", "3p00", "4p00"):
        for metric in ("switch_ratio", "fix025", "break025",
                       "fix050", "break050"):
            value = supervision["stats"][
                "factorized_cf_cost{}_{}".format(cost_suffix, metric)
            ]
            assert torch.isfinite(value)
            assert 0.0 <= value.item() <= 1.0


def test_v55_nested_dominance_uses_weakest_threshold_utility():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_factorized_hit_advantage=True,
        use_factorized_nested_dominance=True,
        factorized_hit_break_cost=1.0,
        parent_transition_candidate_top_k=5,
    ).eval()
    with torch.no_grad():
        model.factorized_hit_head.weight.normal_(mean=0.0, std=0.2)
        model.factorized_hit_head.bias.copy_(torch.tensor((0.1, -0.2)))

    outputs = model(features, baseline, valid)
    expected = outputs["factorized_fix_break_utility"].min(dim=-1).values

    torch.testing.assert_close(
        outputs["parent_transition_advantage"], expected
    )
    assert torch.count_nonzero(expected) > 0


def test_v53_factorized_hit_rejects_unsafe_contracts():
    with pytest.raises(ValueError, match="requires preserved parent score"):
        JointQueryQualityReranker(
            8, use_factorized_hit_advantage=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        JointQueryQualityReranker(
            8, preserve_parent_score=True,
            use_parent_transition_advantage=True,
            use_factorized_hit_advantage=True,
        )
    with pytest.raises(ValueError, match="requires factorized hit advantage"):
        JointQueryQualityReranker(
            8, use_factorized_nested_dominance=True,
        )


def test_v42_zero_initialization_is_exact_box_and_mask_identity():
    features, baseline, valid = _inputs()
    base_mask_weights = torch.tensor([
        [0.2, 0.4, 0.6, 0.8, 0.5],
        [0.7, 0.3, 0.1, 0.9, 0.5],
    ])
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    ).eval()

    outputs = model(features, baseline, valid, base_mask_weights)

    assert torch.equal(outputs["scores"], baseline)
    assert torch.count_nonzero(outputs["residual"]) == 0
    assert torch.equal(outputs["mask_fusion_weights"], base_mask_weights)
    assert torch.count_nonzero(outputs["mask_alpha_residual"]) == 0
    assert torch.count_nonzero(outputs["mask_logit_bias"]) == 0


def test_v43_source_evidence_preserves_exact_step_zero_identity():
    features, baseline, valid = _inputs()
    base_mask_weights = torch.rand_like(baseline)
    source_evidence = torch.rand(
        *baseline.shape, QUERY_MASK_SOURCE_EVIDENCE_DIM
    )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True, use_source_mask_evidence=True,
    ).eval()

    outputs = model(
        features, baseline, valid, base_mask_weights, source_evidence
    )

    assert torch.equal(outputs["scores"], baseline)
    assert torch.equal(outputs["mask_fusion_weights"], base_mask_weights)
    assert torch.count_nonzero(outputs["residual"]) == 0
    assert torch.count_nonzero(outputs["mask_alpha_residual"]) == 0
    assert torch.count_nonzero(outputs["mask_logit_bias"]) == 0


def test_v43_source_evidence_is_required_validated_and_detached():
    features, baseline, valid = _inputs()
    base_mask_weights = torch.rand_like(baseline)
    source_evidence = torch.rand(
        *baseline.shape, QUERY_MASK_SOURCE_EVIDENCE_DIM,
        requires_grad=True,
    )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True, use_source_mask_evidence=True,
        detach_inputs=True,
    )
    with torch.no_grad():
        model.mask_calibration_head.weight.normal_(mean=0.0, std=0.1)

    outputs = model(
        features, baseline, valid, base_mask_weights, source_evidence
    )
    outputs["mask_logit_bias"].sum().backward()

    assert source_evidence.grad is None
    assert model.input_projection[1].weight.grad is not None
    with pytest.raises(ValueError, match="source_mask_evidence"):
        model(features, baseline, valid, base_mask_weights)
    with pytest.raises(ValueError, match="source_mask_evidence"):
        model(
            features, baseline, valid, base_mask_weights,
            source_evidence[..., :-1],
        )
    with pytest.raises(ValueError, match="requires enabled mask calibration"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4,
            use_source_mask_evidence=True,
        )


def test_v42_mask_calibration_head_receives_both_channel_gradients():
    features, baseline, valid = _inputs()
    base_mask_weights = torch.full_like(baseline, 0.4)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    )

    outputs = model(features, baseline, valid, base_mask_weights)
    loss = (
        outputs["mask_fusion_weights"].sum()
        + outputs["mask_logit_bias"].sum()
    )
    loss.backward()

    gradient = model.mask_calibration_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert bool((gradient.abs().sum(dim=1) > 0.0).all().item())


def test_v42_reaches_pure_source_and_unit_threshold_bias_without_saturation():
    features, baseline, valid = _inputs()
    base_mask_weights = torch.full_like(baseline, 0.5)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    ).eval()
    half_logit = torch.atanh(torch.tensor(0.5))
    with torch.no_grad():
        model.mask_calibration_head.bias.fill_(half_logit)

    outputs = model(features, baseline, valid, base_mask_weights)

    torch.testing.assert_close(
        outputs["mask_alpha_residual"], torch.full_like(baseline, 0.5)
    )
    torch.testing.assert_close(
        outputs["mask_fusion_weights"], torch.ones_like(baseline)
    )
    torch.testing.assert_close(
        outputs["mask_logit_bias"], torch.ones_like(baseline)
    )


def test_invalid_queries_are_never_selected():
    features, baseline, valid = _inputs()
    valid[:, 0] = False
    baseline[:, 0] = 1000.0
    outputs = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    ).eval()(features, baseline, valid)
    assert not bool((outputs["selected_indices"] == 0).any().item())
    assert torch.equal(outputs["residual"][:, 0], torch.zeros(2))


def test_query_permutation_equivariance():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    ).eval()
    permutation = torch.tensor([3, 1, 4, 0, 2])
    direct = model(features, baseline, valid)
    permuted = model(
        features[:, permutation], baseline[:, permutation], valid[:, permutation]
    )
    inverse = permutation.argsort()
    for key in (
            "scores", "residual", "box_iou", "mask_iou", "quality",
            "centered_quality", "baseline_rank", "baseline_standardized"):
        assert torch.allclose(direct[key], permuted[key][:, inverse], atol=1e-6)
    for key in ("box_logits", "mask_logits"):
        assert torch.allclose(
            direct[key], permuted[key][:, inverse], atol=1e-6
        )


def test_v42_calibration_is_permutation_equivariant_and_masks_invalid_queries():
    features, baseline, valid = _inputs()
    valid[:, 4] = False
    base_mask_weights = torch.tensor([
        [0.2, 0.4, 0.6, 0.8, 0.5],
        [0.7, 0.3, 0.1, 0.9, 0.5],
    ])
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    ).eval()
    with torch.no_grad():
        model.mask_calibration_head.weight.normal_(mean=0.0, std=0.1)
        model.mask_calibration_head.bias.copy_(torch.tensor([0.1, -0.2]))
    permutation = torch.tensor([3, 1, 4, 0, 2])

    direct = model(features, baseline, valid, base_mask_weights)
    permuted = model(
        features[:, permutation], baseline[:, permutation],
        valid[:, permutation], base_mask_weights[:, permutation],
    )
    inverse = permutation.argsort()
    for key in (
            "scores", "mask_fusion_weights", "mask_alpha_residual",
            "mask_logit_bias"):
        torch.testing.assert_close(
            direct[key], permuted[key][:, inverse], atol=1e-6, rtol=1e-6
        )
    assert torch.count_nonzero(direct["mask_alpha_residual"][:, 4]) == 0
    assert torch.count_nonzero(direct["mask_logit_bias"][:, 4]) == 0
    assert not bool((direct["selected_indices"] == 4).any().item())


def test_v43_source_evidence_is_query_permutation_equivariant():
    features, baseline, valid = _inputs()
    base_mask_weights = torch.rand_like(baseline)
    source_evidence = torch.rand(
        *baseline.shape, QUERY_MASK_SOURCE_EVIDENCE_DIM
    )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True, use_source_mask_evidence=True,
    ).eval()
    with torch.no_grad():
        model.quality_head.weight.normal_(mean=0.0, std=0.1)
        model.residual_head.weight.normal_(mean=0.0, std=0.1)
        model.mask_calibration_head.weight.normal_(mean=0.0, std=0.1)
    permutation = torch.tensor([3, 1, 4, 0, 2])

    direct = model(
        features, baseline, valid, base_mask_weights, source_evidence
    )
    permuted = model(
        features[:, permutation], baseline[:, permutation],
        valid[:, permutation], base_mask_weights[:, permutation],
        source_evidence[:, permutation],
    )
    inverse = permutation.argsort()
    for key in (
            "scores", "residual", "mask_fusion_weights",
            "mask_alpha_residual", "mask_logit_bias"):
        torch.testing.assert_close(
            direct[key], permuted[key][:, inverse], atol=1e-6, rtol=1e-6
        )


def test_box_tier_strictly_dominates_mask_quality():
    box = torch.tensor([[0.24, 0.26, 0.49, 0.51]])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    quality = joint_query_target_quality(box, mask, mask_weight=0.25)
    assert quality[0, 1] > quality[0, 0]
    assert quality[0, 3] > quality[0, 2]


def test_box_tier_rejects_mask_weight_that_can_cross_a_tier():
    box = torch.tensor([[0.25, 0.251]])
    mask = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="below 0.8"):
        joint_query_target_quality(box, mask, mask_weight=0.8)


def test_ordinal_threshold_probabilities_are_nested():
    raw = torch.tensor([
        [[-2.0, 4.0], [3.0, -1.0], [0.0, 0.0]],
    ])
    probability = ordinal_threshold_logits(raw).sigmoid()
    assert torch.all(probability[..., 1] <= probability[..., 0])
    torch.testing.assert_close(
        probability[..., 1],
        raw[..., 0].sigmoid() * raw[..., 1].sigmoid(),
    )


def test_listwise_ranking_directly_trains_shared_quality_head():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    )
    outputs = model(features, baseline, valid)
    box = torch.tensor([
        [0.1, 0.8, 0.4, 0.2, 0.6],
        [0.7, 0.1, 0.2, 0.9, 0.3],
    ])
    mask = torch.tensor([
        [0.2, 0.7, 0.5, 0.1, 0.8],
        [0.6, 0.2, 0.4, 0.7, 0.3],
    ])
    loss = compute_joint_query_quality_loss(
        outputs, box, mask,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
    )["loss"]
    loss.backward()
    assert model.quality_head.weight.grad.abs().sum() > 0
    assert model.residual_head.weight.grad.abs().sum() > 0


def test_multitask_loss_has_finite_nonzero_head_gradients():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    )
    outputs = model(features, baseline, valid)
    box = torch.tensor([
        [0.1, 0.8, 0.4, 0.2, 0.6],
        [0.7, 0.1, 0.2, 0.9, 0.3],
    ])
    mask = torch.tensor([
        [0.2, 0.7, 0.5, 0.1, 0.8],
        [0.6, 0.2, 0.4, 0.7, 0.3],
    ])
    losses = compute_joint_query_quality_loss(outputs, box, mask)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert model.residual_head.weight.grad.abs().sum() > 0
    assert model.quality_head.weight.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_reranker_detaches_backbone_inputs_by_default():
    features, baseline, valid = _inputs()
    features.requires_grad_(True)
    baseline.requires_grad_(True)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    )
    outputs = model(features, baseline, valid)
    outputs["scores"].sum().backward()
    assert features.grad is None
    assert baseline.grad is None


def test_residual_summary_ignores_invalid_queries_and_tracks_variation():
    residual = torch.tensor([
        [0.0, 2.0, 100.0],
        [-1.0, 1.0, 0.0],
    ])
    valid = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])
    stats = summarize_joint_query_residual(residual, valid)
    torch.testing.assert_close(
        stats["residual_abs_mean"], torch.tensor(0.8)
    )
    torch.testing.assert_close(
        stats["residual_abs_max"], torch.tensor(2.0)
    )
    expected_std = torch.tensor((1.0 + (2.0 / 3.0) ** 0.5) / 2.0)
    torch.testing.assert_close(stats["residual_query_std"], expected_std)


def test_loss_rejects_out_of_range_targets():
    features, baseline, valid = _inputs()
    outputs = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    )(features, baseline, valid)
    box = torch.zeros(2, 5)
    box[0, 0] = 1.1
    with pytest.raises(ValueError, match="IoU targets"):
        compute_joint_query_quality_loss(outputs, box, torch.zeros_like(box))


def test_empty_supervision_returns_differentiable_zero():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    )
    outputs = model(features, baseline, valid)
    loss = compute_joint_query_quality_loss(
        outputs,
        torch.zeros(2, 5),
        torch.zeros(2, 5),
        sample_mask=torch.zeros(2, dtype=torch.bool),
    )["loss"]
    assert loss.item() == 0.0
    loss.backward()


class _ZeroSetCriterion:
    def __call__(self, outputs, targets):
        del targets
        zero = outputs["pred_boxes"].sum() * 0.0
        return {
            key: zero
            for key in (
                "loss_ce", "loss_bbox", "loss_giou", "loss_mask",
                "loss_dice", "sp_loss_mask", "sp_loss_dice",
                "corresponding_loss_mask", "corresponding_loss_dice",
                "adaptive_weight_loss_mask", "adaptive_weight_loss_dice",
            )
        }, None


def _joint_loss_end_points(model):
    torch.manual_seed(19)
    batch_size, queries, targets, points, superpoint_count = 2, 4, 2, 6, 3
    features = torch.randn(batch_size, queries, 8)
    baseline = torch.tensor([
        [0.8, 0.5, 0.3, 0.1],
        [0.2, 0.7, 0.4, 0.1],
    ])
    valid = torch.ones(batch_size, queries, dtype=torch.bool)
    base_mask_weights = torch.full_like(baseline, 0.4)
    if model.use_adaptive_source_mixing:
        source_scores = torch.stack((
            baseline,
            torch.flip(baseline, dims=(1,)),
            torch.roll(baseline, shifts=1, dims=1),
        ), dim=-1)
        joint = model(
            features, baseline, valid,
            base_mask_weights=(
                base_mask_weights if model.use_mask_calibration else None
            ),
            source_score_stack=source_scores,
            source_validity=torch.ones_like(
                source_scores, dtype=torch.bool
            ),
        )
    elif model.use_mask_calibration:
        joint = model(features, baseline, valid, base_mask_weights)
    else:
        joint = model(features, baseline, valid)
    centers = torch.tensor([
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0],
         [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
        [[1.0, 0.0, 0.0], [0.7, 0.0, 0.0],
         [1.8, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
    sizes = torch.ones(batch_size, queries, 3)
    gt_centers = torch.tensor([
        [[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]],
        [[1.0, 0.0, 0.0], [4.0, 4.0, 4.0]],
    ])
    gt_sizes = torch.ones(batch_size, targets, 3)
    gt_masks = torch.zeros(batch_size, targets, points)
    gt_masks[0, 0, :4] = 1.0
    gt_masks[1, 0, 2:] = 1.0
    superpoints = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
    ], dtype=torch.long)
    text_masks = [
        torch.randn(queries, superpoint_count)
        for _ in range(batch_size)
    ]
    query_masks = [
        torch.randn(queries, superpoint_count)
        for _ in range(batch_size)
    ]
    token_map = torch.zeros(batch_size, targets, 5)
    end_points = {
        "center_label": gt_centers,
        "size_gts": gt_sizes,
        "sem_cls_label": torch.zeros(
            batch_size, targets, dtype=torch.long
        ),
        "gt_masks": gt_masks,
        "positive_map": token_map,
        "modify_positive_map": token_map.clone(),
        "pron_positive_map": token_map.clone(),
        "other_entity_map": token_map.clone(),
        "rel_positive_map": token_map.clone(),
        "box_label_mask": torch.tensor([[1, 0], [1, 0]]),
        "auxi_entity_positive_map": torch.zeros(batch_size, 1, 5),
        "auxi_box": torch.zeros(batch_size, 6),
        "proposal_center": centers,
        "proposal_pred_size": sizes,
        "proposal_sem_cls_scores": torch.zeros(batch_size, queries, 2),
        "last_center": centers,
        "last_pred_size": sizes,
        "last_sem_cls_scores": torch.zeros(batch_size, queries, 2),
        "last_pred_masks": text_masks,
        "sp_last_pred_masks": query_masks,
        "adaptive_weights": [
            torch.tensor(0.4) for _ in range(batch_size)
        ],
        "superpoints": superpoints,
        "super_xyz_list": [torch.zeros(superpoint_count, 3)] * batch_size,
        "language_dataset": ["scanrefer"] * batch_size,
        "sample_dataset": ["scanrefer"] * batch_size,
        "selected_source_scores": joint["scores"],
        "source_choice_source_scores": {"contrastive_text": baseline},
        "moe_shared_source": "contrastive_text",
        "moe_shared_query": baseline.argmax(dim=1),
        "moe_valid_mask": valid,
    }
    if model.use_mask_calibration:
        calibrated = apply_query_mask_calibration(
            text_masks, query_masks,
            joint["mask_fusion_weights"], joint["mask_logit_bias"],
        )
        end_points["last_pred_masks"] = calibrated[0]
        end_points["sp_last_pred_masks"] = calibrated[1]
        end_points["adaptive_weights"] = calibrated[2]
    for key, value in joint.items():
        end_points["joint_query_quality_{}".format(key)] = value
    return end_points


def test_joint_only_fast_loss_matches_full_loss_and_gradients():
    torch.manual_seed(23)
    fast_model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0
    )
    full_model = copy.deepcopy(fast_model)
    common = {
        "num_decoder_layers": 1,
        "joint_query_quality_loss_weight": 1.0,
        "joint_query_quality_mask_weight": 0.25,
        "joint_query_quality_temperature": 0.25,
        "joint_query_quality_aux_loss_weight": 1.0,
        "joint_query_quality_anchor_loss_weight": 0.5,
        "joint_query_quality_anchor_margin": 0.05,
        "source_moe_balance_loss_weight": 0.0,
        "source_moe_rank_loss_weight": 0.0,
        "source_moe_gate_loss_weight": 0.0,
    }
    fast_loss, fast_end_points = compute_hungarian_loss(
        _joint_loss_end_points(fast_model),
        set_criterion=None,
        joint_query_quality_train_only=True,
        **common
    )
    full_loss, full_end_points = compute_hungarian_loss(
        _joint_loss_end_points(full_model),
        set_criterion=_ZeroSetCriterion(),
        joint_query_quality_train_only=False,
        **common
    )
    torch.testing.assert_close(fast_loss, full_loss)
    for key in (
            "joint_query_quality_loss",
            "joint_query_quality_listwise_loss",
            "joint_query_quality_aux_loss",
            "joint_query_quality_anchor_loss"):
        torch.testing.assert_close(fast_end_points[key], full_end_points[key])

    fast_loss.backward()
    full_loss.backward()
    for (fast_name, fast_parameter), (full_name, full_parameter) in zip(
            fast_model.named_parameters(), full_model.named_parameters()):
        assert fast_name == full_name
        if fast_parameter.grad is None or full_parameter.grad is None:
            assert fast_parameter.grad is None
            assert full_parameter.grad is None
        else:
            torch.testing.assert_close(
                fast_parameter.grad, full_parameter.grad
            )


def test_source_mix_alignment_is_active_in_fast_training_path():
    torch.manual_seed(24)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_adaptive_source_mixing=True,
        source_count=3, shared_source_index=0,
    )
    loss, end_points = compute_hungarian_loss(
        _joint_loss_end_points(model),
        num_decoder_layers=1,
        set_criterion=None,
        joint_query_quality_train_only=True,
        joint_query_quality_loss_weight=1.0,
        joint_query_quality_source_mix_loss_weight=0.25,
        joint_query_quality_source_mix_alignment_temperature=0.25,
        source_moe_balance_loss_weight=0.0,
        source_moe_rank_loss_weight=0.0,
        source_moe_gate_loss_weight=0.0,
    )

    assert end_points[
        "joint_query_quality_source_mix_alignment_loss"
    ].item() > 0.0
    loss.backward()
    gradient = model.adaptive_source_mixer.source_router[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


class _MaskCalibrationCriterion:
    def forward_query_mask_fusion(self, outputs, targets):
        del targets
        loss = outputs["pred_boxes"].sum() * 0.0
        for text, query, alpha in zip(
                outputs["pred_masks"], outputs["sp_pred_masks"],
                outputs["adaptive_weights"]):
            fused = (
                alpha.unsqueeze(-1) * text
                + (1.0 - alpha.unsqueeze(-1)) * query
            )
            loss = loss + fused.square().mean()
        return {
            "adaptive_weight_loss_mask": loss,
            "adaptive_weight_loss_dice": loss * 0.0,
        }, None


def test_v42_fast_mask_loss_backpropagates_to_calibration_head():
    torch.manual_seed(29)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    )
    loss, end_points = compute_hungarian_loss(
        _joint_loss_end_points(model),
        num_decoder_layers=1,
        set_criterion=_MaskCalibrationCriterion(),
        joint_query_quality_train_only=True,
        joint_query_quality_loss_weight=1.0,
        joint_query_quality_mask_weight=0.25,
        joint_query_quality_temperature=0.25,
        joint_query_quality_aux_loss_weight=1.0,
        joint_query_quality_anchor_loss_weight=0.5,
        joint_query_quality_anchor_margin=0.05,
        source_moe_balance_loss_weight=0.0,
        source_moe_rank_loss_weight=0.0,
        source_moe_gate_loss_weight=0.0,
    )
    assert end_points["adaptive_weight_loss_mask"].item() > 0.0

    loss.backward()

    gradient = model.mask_calibration_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert bool((gradient.abs().sum(dim=1) > 0.0).all().item())


def test_candidate_mask_selection_unions_deployed_and_box_oracle_queries():
    scores = torch.tensor([
        [0.9, 0.8, 0.1, 4.0],
        [0.1, 0.9, 0.8, 0.7],
    ])
    box_ious = torch.tensor([
        [0.1, 0.2, 0.95, 1.0],
        [0.9, 0.1, 0.2, 0.3],
    ])
    valid = torch.tensor([
        [True, True, True, False],
        [True, True, True, True],
    ])
    sample_mask = torch.tensor([True, False])

    selected = build_joint_query_mask_candidate_mask(
        scores, box_ious, valid, sample_mask, top_k=1
    )

    assert torch.equal(
        selected,
        torch.tensor([
            [True, False, True, False],
            [False, False, False, False],
        ]),
    )


def test_lovasz_hinge_has_exact_single_pixel_margin_and_gradient():
    logits = torch.tensor([[0.0]], requires_grad=True)
    targets = torch.ones_like(logits)

    loss = lovasz_hinge_loss(logits, targets, num_masks=1)

    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-1.0]]))


def test_lovasz_hinge_is_zero_past_margin_and_permutation_invariant():
    logits = torch.tensor([
        [2.0, -3.0, 4.0, -2.0],
        [0.3, -0.4, -1.2, 1.5],
    ])
    targets = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    permutation = torch.tensor([2, 0, 3, 1])

    perfect = lovasz_hinge_loss(
        logits[:1], targets[:1], num_masks=1
    )
    original = lovasz_hinge_loss(logits, targets, num_masks=2)
    permuted = lovasz_hinge_loss(
        logits.index_select(1, permutation),
        targets.index_select(1, permutation),
        num_masks=2,
    )

    assert perfect.item() == 0.0
    torch.testing.assert_close(original, permuted, rtol=0.0, atol=0.0)


def test_candidate_mask_loss_updates_only_selected_alpha_and_bias_queries():
    text = [torch.tensor([
        [3.0, -2.0, -1.0, 2.0],
        [-1.0, 2.0, 3.0, -2.0],
        [-2.0, 3.0, -2.0, 3.0],
    ])]
    query = [torch.tensor([
        [-2.0, 3.0, 2.0, -1.0],
        [2.0, -1.0, -2.0, 3.0],
        [3.0, -2.0, 3.0, -2.0],
    ])]
    alpha = torch.tensor([[0.4, 0.5, 0.6]], requires_grad=True)
    bias = torch.zeros(1, 3, requires_grad=True)
    calibrated = apply_query_mask_calibration(
        text, query, alpha, bias
    )
    gt_masks = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    superpoints = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    candidate_mask = torch.tensor([[True, False, True]])

    losses = compute_joint_query_mask_candidate_loss(
        calibrated[0], calibrated[1], calibrated[2],
        gt_masks, superpoints, candidate_mask,
    )
    loss = 10.0 * losses["mask_loss"] + 2.0 * losses["dice_loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert bool((alpha.grad[0, [0, 2]].abs() > 0.0).all().item())
    assert alpha.grad[0, 1].item() == 0.0
    assert bool((bias.grad[0, [0, 2]].abs() > 0.0).all().item())
    assert bias.grad[0, 1].item() == 0.0


def test_candidate_gather_before_fusion_matches_dense_reference_exactly():
    text = [torch.tensor([
        [3.0, -2.0, -1.0, 2.0],
        [-1.0, 2.0, 3.0, -2.0],
        [-2.0, 3.0, -2.0, 3.0],
    ])]
    query = [torch.tensor([
        [-2.0, 3.0, 2.0, -1.0],
        [2.0, -1.0, -2.0, 3.0],
        [3.0, -2.0, 3.0, -2.0],
    ])]
    fast_alpha = torch.tensor([[0.4, 0.5, 0.6]], requires_grad=True)
    fast_bias = torch.zeros(1, 3, requires_grad=True)
    dense_alpha = fast_alpha.detach().clone().requires_grad_(True)
    dense_bias = fast_bias.detach().clone().requires_grad_(True)
    gt_masks = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    superpoints = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    candidate_mask = torch.tensor([[True, False, True]])

    fast_calibrated = apply_query_mask_calibration(
        text, query, fast_alpha, fast_bias
    )
    fast_losses = compute_joint_query_mask_candidate_loss(
        fast_calibrated[0], fast_calibrated[1], fast_calibrated[2],
        gt_masks, superpoints, candidate_mask,
    )
    fast_total = (
        10.0 * fast_losses["mask_loss"]
        + 2.0 * fast_losses["dice_loss"]
    )

    dense_calibrated = apply_query_mask_calibration(
        text, query, dense_alpha, dense_bias
    )
    dense_fused = fuse_query_mask_logits(
        dense_calibrated[0][0], dense_calibrated[1][0],
        dense_calibrated[2][0],
    )[candidate_mask[0]]
    dense_target = gt_masks[0, 0].unsqueeze(0).expand_as(dense_fused)
    dense_total = (
        10.0 * sigmoid_focal_loss(dense_fused, dense_target, 2)
        + 2.0 * dice_loss(dense_fused, dense_target, 2)
    )

    torch.testing.assert_close(fast_total, dense_total, rtol=0.0, atol=0.0)
    fast_total.backward()
    dense_total.backward()
    torch.testing.assert_close(
        fast_alpha.grad, dense_alpha.grad, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        fast_bias.grad, dense_bias.grad, rtol=0.0, atol=0.0
    )


def test_v44_candidate_mask_loss_is_active_in_fast_training_path():
    torch.manual_seed(31)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    )
    loss, end_points = compute_hungarian_loss(
        _joint_loss_end_points(model),
        num_decoder_layers=1,
        set_criterion=_MaskCalibrationCriterion(),
        joint_query_quality_train_only=True,
        joint_query_quality_loss_weight=1.0,
        joint_query_quality_candidate_mask_loss_weight=0.25,
        joint_query_quality_candidate_mask_top_k=2,
        source_moe_balance_loss_weight=0.0,
        source_moe_rank_loss_weight=0.0,
        source_moe_gate_loss_weight=0.0,
    )

    assert end_points["joint_query_quality_candidate_mask_loss"].item() > 0.0
    assert end_points["joint_query_quality_candidate_dice_loss"].item() > 0.0
    ratio = end_points[
        "joint_query_quality_candidate_mask_query_ratio"
    ].item()
    assert 0.0 < ratio <= 1.0

    loss.backward()
    gradient = model.mask_calibration_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_v45_candidate_lovasz_loss_is_active_in_fast_training_path():
    torch.manual_seed(37)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
    )
    loss, end_points = compute_hungarian_loss(
        _joint_loss_end_points(model),
        num_decoder_layers=1,
        set_criterion=_MaskCalibrationCriterion(),
        joint_query_quality_train_only=True,
        joint_query_quality_loss_weight=1.0,
        joint_query_quality_candidate_lovasz_loss_weight=0.1,
        joint_query_quality_candidate_mask_top_k=2,
        source_moe_balance_loss_weight=0.0,
        source_moe_rank_loss_weight=0.0,
        source_moe_gate_loss_weight=0.0,
    )

    assert end_points[
        "joint_query_quality_candidate_lovasz_loss"
    ].item() > 0.0
    loss.backward()
    gradient = model.mask_calibration_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_v48_spatial_mask_refiner_is_exact_identity_and_detaches_inputs():
    torch.manual_seed(47)
    query = torch.randn(2, 4, 8, requires_grad=True)
    superpoints = [
        torch.randn(8, 7, requires_grad=True),
        torch.randn(8, 5, requires_grad=True),
    ]
    valid = torch.tensor([
        [True, True, True, False],
        [True, False, True, True],
    ])
    refiner = QuerySuperpointMaskRefiner(
        d_model=8, hidden_dim=4, max_delta=2.0, detach_inputs=True
    )

    residuals = refiner(query, superpoints, valid)

    assert [tuple(row.shape) for row in residuals] == [(4, 7), (4, 5)]
    assert all(torch.count_nonzero(row).item() == 0 for row in residuals)
    loss = sum(row.sum() for row in residuals)
    loss.backward()
    gradient = refiner.query_projection[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0
    assert query.grad is None
    assert all(row.grad is None for row in superpoints)


def test_v48_spatial_mask_refiner_is_query_and_superpoint_equivariant():
    torch.manual_seed(48)
    query = torch.randn(1, 4, 8)
    superpoints = [torch.randn(8, 6)]
    refiner = QuerySuperpointMaskRefiner(
        d_model=8, hidden_dim=4, max_delta=2.0
    )
    with torch.no_grad():
        refiner.query_projection[-1].weight.normal_(std=0.1)
    direct = refiner(query, superpoints)[0]
    query_order = torch.tensor([2, 0, 3, 1])
    point_order = torch.tensor([5, 1, 3, 0, 4, 2])
    permuted = refiner(
        query[:, query_order],
        [superpoints[0][:, point_order]],
    )[0]

    torch.testing.assert_close(
        direct[query_order][:, point_order], permuted
    )


def test_v48_spatial_residual_corrects_final_fused_logits_exactly():
    torch.manual_seed(49)
    text = [torch.randn(3, 5)]
    query = [torch.randn(3, 5)]
    alpha = torch.tensor([0.35, 0.6, 0.8])
    residual = [torch.randn(3, 5)]

    refined_text, refined_query = apply_query_superpoint_mask_residual(
        text, query, residual
    )
    refined = fuse_query_mask_logits(
        refined_text[0], refined_query[0], alpha
    )
    expected = fuse_query_mask_logits(text[0], query[0], alpha) + residual[0]

    torch.testing.assert_close(refined, expected, rtol=0.0, atol=1e-6)


def test_v48_candidate_mask_loss_reaches_spatial_refiner():
    torch.manual_seed(50)
    features, baseline, valid = _inputs(batch=1, queries=3, features=8)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
        use_spatial_mask_refiner=True,
        spatial_mask_d_model=8,
        spatial_mask_hidden_dim=4,
    )
    outputs = model(
        features,
        baseline,
        valid,
        base_mask_weights=torch.full((1, 3), 0.5),
        spatial_query_features=torch.randn(1, 3, 8),
        spatial_superpoint_features=[torch.randn(8, 6)],
    )
    text = [torch.randn(3, 6)]
    query = [torch.randn(3, 6)]
    calibrated = apply_query_mask_calibration(
        text,
        query,
        outputs["mask_fusion_weights"],
        outputs["mask_logit_bias"],
    )
    refined = apply_query_superpoint_mask_residual(
        calibrated[0], calibrated[1], outputs["mask_spatial_residuals"]
    )
    losses = compute_joint_query_mask_candidate_loss(
        refined[0],
        refined[1],
        calibrated[2],
        torch.tensor([[[1, 0, 1, 0, 1, 0]]]),
        torch.tensor([[0, 1, 2, 3, 4, 5]]),
        torch.ones(1, 3, dtype=torch.bool),
        compute_lovasz=True,
    )
    total = losses["mask_loss"] + losses["dice_loss"] + losses[
        "lovasz_loss"
    ]
    total.backward()

    gradient = model.spatial_mask_refiner.query_projection[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


class _V48ForwardCrossEncoder(torch.nn.Module):
    def forward(self, vis_feats, pos_feats, padding_mask, text_feats,
                text_padding_mask, end_points, detected_feats,
                detected_mask, spatial_point_xyz):
        del pos_feats, padding_mask, end_points, detected_feats
        del detected_mask, spatial_point_xyz
        return vis_feats, text_feats


class _V48ForwardPosition(torch.nn.Module):
    def forward(self, xyz):
        return xyz.new_zeros(xyz.shape[0], 288, xyz.shape[1])


class _V48ForwardSuperGrouper(torch.nn.Module):
    def forward(self, xyz, super_xyz, features):
        del xyz
        count = super_xyz.shape[1]
        grouped = features[:, :, :count].unsqueeze(-1)
        indices = torch.arange(
            count, device=features.device, dtype=torch.long
        ).reshape(1, count, 1)
        return grouped, indices


class _V48ForwardDecoder(torch.nn.Module):
    def forward(self, query, points, text, query_pos, query_mask,
                text_padding_mask, detected_feats=None, detected_mask=None):
        del points, text, query_pos, query_mask, text_padding_mask
        del detected_feats, detected_mask
        return query


class _V48ForwardBoxHead(torch.nn.Module):
    def forward(self, features, base_xyz, end_points, prefix):
        batch, _, queries = features.shape
        center = base_xyz
        size = features.new_ones(batch, queries, 3)
        end_points[prefix + "center"] = center
        end_points[prefix + "pred_size"] = size
        end_points[prefix + "sem_cls_scores"] = features.new_zeros(
            batch, queries, 2
        )
        return center, size


class _V48ForwardSWA(torch.nn.Module):
    def forward(self, source, query, attn_mask=None, pe=None):
        del attn_mask, pe
        batch_query = query.transpose(0, 1)
        weights = source.new_ones(
            batch_query.shape[0], batch_query.shape[1], source.shape[0]
        )
        return batch_query, weights, weights


class _V48ForwardSelector(torch.nn.Module):
    def forward(self, candidate_feats, candidate_boxes, source_scores,
                valid_mask, text_feats, text_mask):
        del candidate_feats, candidate_boxes, source_scores
        del text_feats, text_mask
        scores = torch.linspace(
            1.0, 0.0, valid_mask.shape[1], device=valid_mask.device
        ).unsqueeze(0).expand(valid_mask.shape[0], -1)
        return {"selected_source_scores": scores}


def _v48_forward_model(monkeypatch):
    from models.mcln import MCLN

    model = object.__new__(MCLN)
    torch.nn.Module.__init__(model)
    model.num_queries = 256
    model.num_decoder_layers = 1
    model.self_position_embedding = "none"
    model.contrastive_align_loss = True
    model.butd = False
    model.source_moe_gate_use_evidence_features = False
    model.source_moe_gate_use_rich_features = True
    model.source_moe_shared_source = "default"
    model.use_joint_query_quality_reranker = True
    model.joint_query_quality_use_mask_calibration = True
    model.joint_query_quality_use_source_mask_evidence = True
    model.joint_query_quality_use_gate_evidence = False
    model.joint_query_quality_use_spatial_mask_refiner = True
    model.source_choice_selector_sources = (
        "default", "default_rank_blend_contrastive010"
    )
    model.cross_encoder = _V48ForwardCrossEncoder()
    model.pos_embed = _V48ForwardPosition()
    model.contrastive_align_projection_text = torch.nn.Linear(288, 64)
    model.contrastive_align_projection_image = torch.nn.Linear(288, 64)
    model.x_mask = torch.nn.Identity()
    model.super_grouper = _V48ForwardSuperGrouper()
    model.rel_encoder = torch.nn.Linear(3, 288, bias=False)
    torch.nn.init.zeros_(model.rel_encoder.weight)
    model.decoder_query_proj = torch.nn.Identity()
    model.decoder = torch.nn.ModuleList([_V48ForwardDecoder()])
    model.proposal_head = _V48ForwardBoxHead()
    model.prediction_heads = torch.nn.ModuleList([_V48ForwardBoxHead()])
    model.x_query = torch.nn.Identity()
    model.swa_layers = torch.nn.ModuleList([
        _V48ForwardSWA(), _V48ForwardSWA(), _V48ForwardSWA()
    ])
    model.swa_ffn_layers = torch.nn.ModuleList([
        torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()
    ])
    model.query_mask_fusion_calibrator = None
    model.source_choice_selector = _V48ForwardSelector()
    model.source_moe = None
    model.joint_query_quality_reranker = JointQueryQualityReranker(
        152, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
        use_source_mask_evidence=True,
        use_spatial_mask_refiner=True,
        spatial_mask_d_model=288,
        spatial_mask_hidden_dim=4,
    )

    def run_backbones(self, inputs):
        batch, points = inputs["point_clouds"].shape[:2]
        channel = torch.linspace(
            -1.0, 1.0, 288, device=inputs["point_clouds"].device
        ).reshape(1, 288, 1)
        point = torch.linspace(
            -0.5, 0.5, points, device=inputs["point_clouds"].device
        ).reshape(1, 1, points)
        point_features = (
            channel + point * channel.square()
        ).expand(batch, -1, -1).contiguous()
        return {
            "fp2_xyz": inputs["point_clouds"][..., :3],
            "fp2_features": point_features,
            "text_feats": inputs["point_clouds"].new_ones(batch, 4, 288),
            "text_attention_mask": torch.zeros(
                batch, 4, dtype=torch.bool,
                device=inputs["point_clouds"].device,
            ),
        }

    def generate_queries(self, xyz, features, end_points):
        end_points["query_points_xyz"] = xyz[:, :self.num_queries]
        end_points["query_points_feature"] = features[:, :, :self.num_queries]
        return end_points

    def prediction_head(self, query, superpoint_feats):
        scores = query.mean(dim=-1, keepdim=True)
        masks = torch.einsum("bqd,bsd->bqs", query, superpoint_feats)
        return scores, masks, torch.zeros_like(masks, dtype=torch.bool)

    model._run_backbones = MethodType(run_backbones, model)
    model._generate_queries = MethodType(generate_queries, model)
    model.prediction_head = MethodType(prediction_head, model)

    def source_batch(end_points, inputs, source_names,
                     include_rich_candidate_feats):
        del inputs
        assert include_rich_candidate_feats is True
        scores = end_points["last_center"].new_zeros(1, 256)
        routed_scores = torch.linspace(
            0.0, 1.0, 256, device=scores.device
        ).unsqueeze(0)
        named_scores = {
            name: scores if name == "default" else routed_scores
            for name in source_names
        }
        return {
            "candidate_feats": end_points["source_choice_candidate_feats"],
            "candidate_boxes": torch.cat((
                end_points["last_center"], end_points["last_pred_size"]
            ), dim=-1),
            "source_scores": named_scores,
            "source_validity": torch.ones(
                1, 256, len(source_names), dtype=torch.bool,
                device=scores.device,
            ),
            "valid_mask": torch.ones_like(scores, dtype=torch.bool),
            "text_feats": end_points["text_feats"],
            "text_mask": end_points["text_attention_mask"],
            "rich_candidate_feats": scores.unsqueeze(-1).expand(-1, -1, 152),
        }

    monkeypatch.setattr(
        "models.mcln.build_mcln_source_choice_batch", source_batch
    )
    return model.eval()


def test_v49_real_mcln_forward_is_identity_and_routes_all_sources(monkeypatch):
    torch.manual_seed(52)
    model = _v48_forward_model(monkeypatch)
    model.joint_query_quality_use_adaptive_source_mixing = True
    model.joint_query_quality_reranker = JointQueryQualityReranker(
        152, hidden_dim=16, num_heads=4, dropout=0.0,
        use_mask_calibration=True,
        use_source_mask_evidence=True,
        use_spatial_mask_refiner=True,
        spatial_mask_d_model=288,
        spatial_mask_hidden_dim=4,
        use_adaptive_source_mixing=True,
        source_count=2,
        shared_source_index=0,
    )
    points = 256
    outputs = model({
        "point_clouds": torch.randn(1, points, 3),
        "superpoint": torch.arange(points).remainder(8).unsqueeze(0),
        "text": ["target"],
    })

    assert torch.equal(
        outputs["selected_source_scores"],
        outputs["joint_query_quality_parent_scores"],
    )
    assert outputs["joint_query_quality_source_mix_weights"].shape == (
        1, 256, 2
    )
    assert outputs[
        "joint_query_quality_source_mix_weight_default"
    ].item() > 0.0
    assert outputs[
        "joint_query_quality_source_mix_weight_default_rank_blend_contrastive010"
    ].item() > 0.0
    assert outputs[
        "joint_query_quality_source_mix_residual_abs_mean"
    ].item() == 0.0
    (-outputs["selected_source_scores"][:, 127].mean()).backward()
    gradient = model.joint_query_quality_reranker.adaptive_source_mixer\
        .strength_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_v48_real_mcln_forward_writes_masks_and_backpropagates_candidate_loss(
        monkeypatch):
    torch.manual_seed(51)
    model = _v48_forward_model(monkeypatch)
    points = 256
    superpoints = torch.arange(points).remainder(8).unsqueeze(0)
    inputs = {
        "point_clouds": torch.randn(1, points, 3),
        "superpoint": superpoints,
        "text": ["target"],
    }

    identity = model(inputs)
    assert identity["moe_shared_source"] == "default"
    assert torch.equal(
        identity["moe_shared_query"], torch.zeros(1, dtype=torch.long)
    )
    assert torch.equal(
        identity["moe_valid_mask"], torch.ones(1, 256, dtype=torch.bool)
    )
    assert torch.count_nonzero(
        identity["joint_query_quality_mask_spatial_residuals"][0]
    ).item() == 0
    identity_masks = [row.detach().clone() for row in identity["last_pred_masks"]]

    with torch.no_grad():
        model.joint_query_quality_reranker.spatial_mask_refiner\
            .query_projection[-1].weight.normal_(std=0.1)
    refined = model(inputs)
    residual = refined["joint_query_quality_mask_spatial_residuals"][0]
    assert residual.abs().sum().item() > 0.0
    assert not torch.equal(refined["last_pred_masks"][0], identity_masks[0])
    assert refined["joint_query_quality_mask_spatial_residual_abs_mean"].item() > 0.0
    assert refined[
        "joint_query_quality_mask_spatial_superpoint_std_mean"
    ].item() > 0.0
    assert refined[
        "joint_query_quality_mask_spatial_query_std_mean"
    ].item() > 0.0

    gt_mask = torch.tensor(
        [[[1, 0] * (points // 2)]], dtype=torch.float32
    )
    token_map = torch.zeros(1, 1, 5)
    refined.update({
        "center_label": torch.zeros(1, 1, 3),
        "size_gts": torch.ones(1, 1, 3),
        "sem_cls_label": torch.zeros(1, 1, dtype=torch.long),
        "gt_masks": gt_mask,
        "positive_map": token_map,
        "modify_positive_map": token_map.clone(),
        "pron_positive_map": token_map.clone(),
        "other_entity_map": token_map.clone(),
        "rel_positive_map": token_map.clone(),
        "box_label_mask": torch.ones(1, 1, dtype=torch.long),
        "auxi_entity_positive_map": torch.zeros(1, 1, 5),
        "auxi_box": torch.zeros(1, 6),
        "language_dataset": ["scanrefer"],
        "sample_dataset": ["scanrefer"],
    })
    total, refined = compute_hungarian_loss(
        refined,
        num_decoder_layers=1,
        set_criterion=_MaskCalibrationCriterion(),
        joint_query_quality_train_only=True,
        joint_query_quality_loss_weight=1.0,
        joint_query_quality_mask_weight=0.25,
        joint_query_quality_temperature=0.25,
        joint_query_quality_aux_loss_weight=1.0,
        joint_query_quality_anchor_loss_weight=0.5,
        joint_query_quality_anchor_margin=0.05,
        joint_query_quality_candidate_mask_loss_weight=0.25,
        joint_query_quality_candidate_lovasz_loss_weight=0.1,
        joint_query_quality_candidate_mask_top_k=16,
        source_moe_balance_loss_weight=0.0,
        source_moe_rank_loss_weight=0.0,
        source_moe_gate_loss_weight=0.0,
    )
    assert refined["joint_query_quality_candidate_mask_loss"].item() > 0.0
    assert refined["joint_query_quality_candidate_dice_loss"].item() > 0.0
    assert refined["joint_query_quality_candidate_lovasz_loss"].item() > 0.0
    total.backward()
    gradient = model.joint_query_quality_reranker.spatial_mask_refiner\
        .query_projection[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_plain_selector_forward_does_not_publish_joint_training_aliases(
        monkeypatch):
    model = _v48_forward_model(monkeypatch)
    model.joint_query_quality_reranker = None
    model.use_joint_query_quality_reranker = False
    model.joint_query_quality_use_mask_calibration = False
    model.joint_query_quality_use_source_mask_evidence = False
    model.joint_query_quality_use_spatial_mask_refiner = False
    points = 256
    outputs = model({
        "point_clouds": torch.randn(1, points, 3),
        "superpoint": torch.arange(points).remainder(8).unsqueeze(0),
        "text": ["target"],
    })

    assert "selected_source_scores" in outputs
    assert "moe_shared_source" not in outputs
    assert "moe_shared_query" not in outputs
    assert "moe_valid_mask" not in outputs


def test_joint_quality_source_mixer_is_identity_then_trains_router():
    torch.manual_seed(4901)
    reranker = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_adaptive_source_mixing=True,
        source_count=3, shared_source_index=0,
    )
    features, baseline, valid = _inputs(features=8)
    source_scores = torch.randn(2, 5, 3)
    source_validity = torch.ones(2, 5, 3, dtype=torch.bool)
    source_validity[0, 2, 2] = False
    optimizer = torch.optim.SGD(
        reranker.adaptive_source_mixer.parameters(), lr=0.5
    )

    initial = reranker(
        features, baseline, valid,
        source_score_stack=source_scores,
        source_validity=source_validity,
    )
    assert torch.equal(initial["scores"], baseline)
    assert torch.count_nonzero(
        initial["source_mix_residual_logit"]
    ).item() == 0
    assert initial["source_mix_weights"][0, 2, 2].item() == 0.0
    (-initial["scores"][:, 2].mean()).backward()
    strength_gradient = reranker.adaptive_source_mixer.strength_head[-1]\
        .weight.grad
    assert strength_gradient is not None
    assert strength_gradient.abs().sum().item() > 0.0
    optimizer.step()
    optimizer.zero_grad()

    updated = reranker(
        features, baseline, valid,
        source_score_stack=source_scores,
        source_validity=source_validity,
    )
    assert updated["source_mix_residual_logit"].abs().sum().item() > 0.0
    (-updated["scores"][:, 2].mean()).backward()
    router_gradient = reranker.adaptive_source_mixer.source_router[-1]\
        .weight.grad
    assert router_gradient is not None
    assert router_gradient.abs().sum().item() > 0.0


def test_joint_quality_source_mixer_is_routed_source_permutation_equivariant():
    torch.manual_seed(4902)
    mixer = JointQualityAdaptiveSourceMixer(
        hidden_dim=8, source_count=3, shared_index=0,
    )
    with torch.no_grad():
        mixer.source_router[-1].weight.normal_()
        mixer.strength_head[-1].weight.normal_()
    hidden = torch.randn(2, 5, 8)
    quality = torch.rand(2, 5, 6)
    parent = torch.randn(2, 5)
    scores = torch.randn(2, 5, 3)
    validity = torch.ones(2, 5, 3, dtype=torch.bool)
    valid = torch.ones(2, 5, dtype=torch.bool)

    original = mixer(hidden, quality, parent, scores, validity, valid)
    permutation = torch.tensor([0, 2, 1])
    changed = mixer(
        hidden, quality, parent,
        scores[..., permutation], validity[..., permutation], valid,
    )

    assert torch.allclose(
        changed["source_mix_residual_logit"],
        original["source_mix_residual_logit"], atol=1e-6,
    )
    assert torch.allclose(
        changed["source_mix_weights"],
        original["source_mix_weights"][..., permutation], atol=1e-6,
    )


def test_source_distribution_reliability_is_finite_and_fails_closed():
    scores = torch.tensor([[[3.0, 1.0, 0.0],
                            [1.0, 2.0, 0.0],
                            [0.0, 0.0, 0.0]]])
    validity = torch.ones_like(scores, dtype=torch.bool)
    validity[..., 2] = False

    features = build_source_distribution_reliability_features(
        scores, validity, shared_index=0
    )

    assert features.shape == (1, 3, 3, SOURCE_DISTRIBUTION_RELIABILITY_DIM)
    assert torch.isfinite(features).all()
    assert torch.count_nonzero(features[..., 2, :]).item() == 0
    assert torch.count_nonzero(features[..., 0, 4:]).item() == 0
    assert bool((features[..., :2, 1:][validity[..., :2].unsqueeze(-1)
                .expand(-1, -1, -1, 5)] >= 0.0).all().item())


def test_distribution_reliability_mixer_preserves_identity_and_permutation():
    torch.manual_seed(4906)
    mixer = JointQualityAdaptiveSourceMixer(
        hidden_dim=8, source_count=3, shared_index=0,
        use_distribution_reliability=True,
    )
    hidden = torch.randn(2, 5, 8)
    quality = torch.rand(2, 5, 6)
    parent = torch.randn(2, 5)
    scores = torch.randn(2, 5, 3)
    validity = torch.ones(2, 5, 3, dtype=torch.bool)
    valid = torch.ones(2, 5, dtype=torch.bool)

    identity = mixer(hidden, quality, parent, scores, validity, valid)
    assert torch.count_nonzero(
        identity["source_mix_residual_logit"]
    ).item() == 0
    assert identity["source_mix_distribution_reliability"].shape == (
        2, 5, 3, SOURCE_DISTRIBUTION_RELIABILITY_DIM
    )

    with torch.no_grad():
        mixer.source_router[-1].weight.normal_()
        mixer.strength_head[-1].weight.normal_()
    original = mixer(hidden, quality, parent, scores, validity, valid)
    permutation = torch.tensor([0, 2, 1])
    changed = mixer(
        hidden, quality, parent,
        scores[..., permutation], validity[..., permutation], valid,
    )
    torch.testing.assert_close(
        changed["source_mix_residual_logit"],
        original["source_mix_residual_logit"], atol=1e-6, rtol=0.0,
    )
    torch.testing.assert_close(
        changed["source_mix_weights"],
        original["source_mix_weights"][..., permutation],
        atol=1e-6, rtol=0.0,
    )
    torch.testing.assert_close(
        changed["source_mix_distribution_reliability"],
        original["source_mix_distribution_reliability"][..., permutation, :],
        atol=1e-6, rtol=0.0,
    )
    original["source_mix_weights"][..., 1].sum().backward()
    reliability_gradient = mixer.source_encoder[0].weight.grad[
        :, -SOURCE_DISTRIBUTION_RELIABILITY_DIM:
    ]
    assert torch.isfinite(reliability_gradient).all()
    assert reliability_gradient.abs().sum().item() > 0.0


def test_distribution_reliability_is_optional_and_v50_shape_is_unchanged():
    baseline = JointQualityAdaptiveSourceMixer(
        hidden_dim=8, source_count=4, shared_index=0
    )
    enhanced = JointQualityAdaptiveSourceMixer(
        hidden_dim=8, source_count=4, shared_index=0,
        use_distribution_reliability=True,
    )
    assert baseline.source_encoder[0].in_features == 8 + 9
    assert enhanced.source_encoder[0].in_features == (
        8 + 9 + SOURCE_DISTRIBUTION_RELIABILITY_DIM
    )
    with pytest.raises(ValueError, match="requires adaptive source mixing"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4,
            use_source_distribution_reliability=True,
        )


def test_source_mix_alignment_trains_router_at_step_zero_identity():
    torch.manual_seed(4903)
    reranker = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_adaptive_source_mixing=True,
        source_count=3, shared_source_index=0,
    )
    features, baseline, valid = _inputs(features=8)
    source_scores = torch.stack((
        baseline,
        torch.flip(baseline, dims=(1,)),
        torch.roll(baseline, shifts=1, dims=1),
    ), dim=-1)
    source_validity = torch.ones_like(source_scores, dtype=torch.bool)
    outputs = reranker(
        features, baseline, valid,
        source_score_stack=source_scores,
        source_validity=source_validity,
    )
    box_ious = torch.tensor([
        [0.9, 0.1, 0.7, 0.2, 0.4],
        [0.2, 0.8, 0.3, 0.6, 0.1],
    ])
    mask_ious = torch.tensor([
        [0.8, 0.2, 0.6, 0.1, 0.5],
        [0.1, 0.7, 0.4, 0.5, 0.2],
    ])

    supervision = compute_joint_query_quality_loss(
        outputs, box_ious, mask_ious,
        source_mix_loss_weight=0.25,
        source_mix_alignment_temperature=0.25,
        source_mix_query_focus_weight=0.75,
    )

    assert torch.equal(outputs["scores"], baseline)
    assert supervision["source_mix_alignment_loss"].item() > 0.0
    supervision["loss"].backward()
    gradient = reranker.adaptive_source_mixer.source_router[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_source_mix_alignment_reaches_trainable_source_at_step_zero():
    torch.manual_seed(4905)
    reranker = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        use_adaptive_source_mixing=True,
        source_count=4, shared_source_index=0,
        detach_inputs=False,
    )
    features, baseline, valid = _inputs(features=8)
    structured_scores = torch.randn(2, 5, requires_grad=True)
    source_scores = torch.stack((
        baseline.detach(),
        torch.flip(baseline.detach(), dims=(1,)),
        torch.roll(baseline.detach(), shifts=1, dims=1),
        structured_scores,
    ), dim=-1)
    source_validity = torch.ones_like(source_scores, dtype=torch.bool)
    outputs = reranker(
        features, baseline, valid,
        source_score_stack=source_scores,
        source_validity=source_validity,
    )
    box_ious = torch.tensor([
        [0.9, 0.1, 0.7, 0.2, 0.4],
        [0.2, 0.8, 0.3, 0.6, 0.1],
    ])
    mask_ious = torch.tensor([
        [0.8, 0.2, 0.6, 0.1, 0.5],
        [0.1, 0.7, 0.4, 0.5, 0.2],
    ])
    supervision = compute_joint_query_quality_loss(
        outputs, box_ious, mask_ious,
        source_mix_loss_weight=0.25,
        source_mix_alignment_temperature=0.25,
        source_mix_query_focus_weight=0.75,
    )

    assert torch.equal(outputs["scores"], baseline)
    supervision["loss"].backward()
    assert structured_scores.grad is not None
    assert torch.isfinite(structured_scores.grad).all()
    assert structured_scores.grad.abs().sum().item() > 0.0


def test_source_mix_alignment_is_routed_source_permutation_invariant():
    torch.manual_seed(4904)
    mixer = JointQualityAdaptiveSourceMixer(
        hidden_dim=8, source_count=3, shared_index=0,
    )
    with torch.no_grad():
        mixer.source_router[-1].weight.normal_()
    hidden = torch.randn(2, 5, 8)
    quality_evidence = torch.rand(2, 5, 6)
    parent = torch.randn(2, 5)
    source_scores = torch.randn(2, 5, 3)
    source_validity = torch.ones(2, 5, 3, dtype=torch.bool)
    valid = torch.ones(2, 5, dtype=torch.bool)
    target = torch.rand(2, 5)
    original = mixer(
        hidden, quality_evidence, parent, source_scores,
        source_validity, valid,
    )
    permutation = torch.tensor([0, 2, 1])
    changed = mixer(
        hidden, quality_evidence, parent,
        source_scores[..., permutation],
        source_validity[..., permutation], valid,
    )

    original_loss = compute_joint_query_source_mix_alignment_loss(
        dict(original, valid_mask=valid), target,
        query_relevance=torch.softmax(target / 0.25, dim=1),
        query_focus_weight=0.75,
    )
    changed_loss = compute_joint_query_source_mix_alignment_loss(
        dict(changed, valid_mask=valid), target,
        query_relevance=torch.softmax(target / 0.25, dim=1),
        query_focus_weight=0.75,
    )

    torch.testing.assert_close(
        changed_loss["loss"], original_loss["loss"], atol=1e-7, rtol=0.0
    )
    torch.testing.assert_close(
        changed_loss["target_top1_acc"],
        original_loss["target_top1_acc"], atol=0.0, rtol=0.0
    )


def test_source_mix_alignment_focus_prioritizes_high_quality_queries():
    outputs = {
        "source_mix_weights": torch.tensor([[
            [0.1, 0.9],
            [0.1, 0.9],
        ]]),
        "source_mix_ranks": torch.tensor([[
            [1.0, 0.0],
            [1.0, 0.0],
        ]]),
        "source_mix_validity": torch.ones(1, 2, 2, dtype=torch.bool),
        "valid_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    target_quality = torch.tensor([[1.0, 0.0]])
    relevance = torch.tensor([[0.99, 0.01]])

    uniform = compute_joint_query_source_mix_alignment_loss(
        outputs, target_quality, query_focus_weight=0.0,
    )
    focused = compute_joint_query_source_mix_alignment_loss(
        outputs, target_quality, query_relevance=relevance,
        query_focus_weight=0.75,
    )

    assert focused["loss"] > uniform["loss"]


def test_source_mix_alignment_zero_focus_exactly_matches_v49_objective():
    outputs = {
        "source_mix_weights": torch.tensor([[
            [0.7, 0.3],
            [0.4, 0.6],
            [0.2, 0.8],
        ]]),
        "source_mix_ranks": torch.tensor([[
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ]]),
        "source_mix_validity": torch.ones(1, 3, 2, dtype=torch.bool),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    target_quality = torch.tensor([[0.9, 0.4, 0.1]])
    target_rank = torch.tensor([[1.0, 0.5, 0.0]])
    target_logits = -(
        outputs["source_mix_ranks"] - target_rank.unsqueeze(-1)
    ).abs() / 0.25
    target_weights = torch.softmax(target_logits, dim=-1)
    target_weights = target_weights / target_weights.sum(
        dim=-1, keepdim=True
    ).clamp(min=1e-6)
    expected = -(
        target_weights
        * outputs["source_mix_weights"].clamp(min=1e-8).log()
    ).sum(dim=-1).mean()

    observed = compute_joint_query_source_mix_alignment_loss(
        outputs,
        target_quality,
        temperature=0.25,
        query_focus_weight=0.0,
    )["loss"]

    torch.testing.assert_close(observed, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan")])
def test_source_mix_alignment_rejects_invalid_query_focus(weight):
    outputs = {
        "source_mix_weights": torch.full((1, 2, 2), 0.5),
        "source_mix_ranks": torch.tensor([[
            [1.0, 0.0],
            [0.0, 1.0],
        ]]),
        "source_mix_validity": torch.ones(1, 2, 2, dtype=torch.bool),
        "valid_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="query_focus_weight"):
        compute_joint_query_source_mix_alignment_loss(
            outputs, torch.tensor([[1.0, 0.0]]),
            query_relevance=torch.tensor([[0.9, 0.1]]),
            query_focus_weight=weight,
        )


def test_v41_v42_v43_v46_v48_and_v49_parameter_contracts_are_exact():
    v41 = JointQueryQualityReranker(152, hidden_dim=128, num_heads=4)
    v42 = JointQueryQualityReranker(
        152, hidden_dim=128, num_heads=4, use_mask_calibration=True
    )
    v43 = JointQueryQualityReranker(
        152, hidden_dim=128, num_heads=4, use_mask_calibration=True,
        use_source_mask_evidence=True,
    )
    v46 = JointQueryQualityReranker(
        152, hidden_dim=128, num_heads=4, use_mask_calibration=True,
        use_source_mask_evidence=True, use_gate_evidence=True,
    )
    v48 = JointQueryQualityReranker(
        152, hidden_dim=128, num_heads=4, use_mask_calibration=True,
        use_source_mask_evidence=True,
        use_spatial_mask_refiner=True,
    )
    v49 = JointQueryQualityReranker(
        152, hidden_dim=128, num_heads=4, use_mask_calibration=True,
        use_source_mask_evidence=True,
        use_spatial_mask_refiner=True,
        use_adaptive_source_mixing=True,
        source_count=3, shared_source_index=0,
    )

    assert len(v41.state_dict()) == 20
    assert sum(parameter.numel() for parameter in v41.parameters()) == 153531
    assert len(v42.state_dict()) == 22
    assert sum(parameter.numel() for parameter in v42.parameters()) == 153919
    assert len(v43.state_dict()) == 22
    assert sum(parameter.numel() for parameter in v43.parameters()) == 155219
    assert len(v46.state_dict()) == 22
    assert sum(parameter.numel() for parameter in v46.parameters()) == 158339
    assert len(v48.state_dict()) == 34
    assert sum(parameter.numel() for parameter in v48.parameters()) == 176979
    assert len(v49.state_dict()) == 45
    assert sum(parameter.numel() for parameter in v49.parameters()) == 229460


class _TrainModeToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(3, 3), torch.nn.Dropout(0.5)
        )
        self.joint_query_quality_reranker = JointQueryQualityReranker(
            3, hidden_dim=8, num_heads=2, dropout=0.1
        )


def _joint_only_args(**overrides):
    values = {
        "source_choice_selector_train_only": False,
        "source_moe_train_only": False,
        "source_moe_gate_train_only": False,
        "source_moe_gate_new_heads_only": False,
        "query_mask_fusion_train_only": False,
        "joint_query_quality_train_only": True,
        "use_joint_query_quality_reranker": True,
        "joint_query_quality_lr": 3e-4,
        "lr": 2e-5,
        "lr_backbone": 2e-4,
        "text_encoder_lr": 3e-6,
        "weight_decay": 5e-4,
        "frozen": False,
        "small_lr": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_joint_only_optimizer_contains_exactly_the_new_module():
    model = _TrainModeToy()
    optimizer = BaseTrainTester.get_optimizer(_joint_only_args(), model)
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {
        id(parameter)
        for parameter in model.joint_query_quality_reranker.parameters()
    }
    assert optimized == expected
    assert all(
        parameter.requires_grad
        for parameter in model.joint_query_quality_reranker.parameters()
    )
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())



def test_parent_transition_optimizer_excludes_bypassed_output_heads():
    model = _TrainModeToy()
    model.joint_query_quality_reranker = JointQueryQualityReranker(
        3, hidden_dim=8, num_heads=2, dropout=0.1,
        preserve_parent_score=True,
        use_parent_transition_advantage=True,
        parent_transition_candidate_top_k=2,
    )
    args = _joint_only_args(
        joint_query_quality_use_parent_transition_advantage=True
    )
    optimizer = BaseTrainTester.get_optimizer(args, model)
    optimized_names = {
        name for name, parameter in model.named_parameters()
        if any(
            id(parameter) == id(optimized)
            for group in optimizer.param_groups
            for optimized in group["params"]
        )
    }
    assert any(
        name.startswith(
            "joint_query_quality_reranker.parent_transition_head."
        )
        for name in optimized_names
    )
    assert not any(
        name.startswith((
            "joint_query_quality_reranker.quality_head.",
            "joint_query_quality_reranker.residual_head.",
        ))
        for name in optimized_names
    )
    assert not model.joint_query_quality_reranker.quality_head.weight.requires_grad
    assert not model.joint_query_quality_reranker.residual_head.weight.requires_grad

def test_joint_only_train_mode_keeps_frozen_backbone_in_eval():
    model = _TrainModeToy().train()
    BaseTrainTester._set_source_moe_train_mode(model, _joint_only_args())
    assert model.training is False
    assert model.backbone.training is False
    assert model.joint_query_quality_reranker.training is True


def test_sacr_joint_only_optimizer_and_train_mode_include_structured_modules():
    model = _TrainModeToy().train()
    model.structured_slot_builder = torch.nn.Sequential(
        torch.nn.Linear(3, 3), torch.nn.Dropout(0.5)
    )
    model.sacr_head = torch.nn.Sequential(
        torch.nn.Linear(3, 3), torch.nn.Dropout(0.5)
    )
    model.sacr_residual_scale = torch.nn.Parameter(torch.tensor([0.1]))
    args = _joint_only_args(use_sacr_source=True)
    optimizer = BaseTrainTester.get_optimizer(args, model)
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith((
            "joint_query_quality_reranker.",
            "structured_slot_builder.",
            "sacr_head.",
            "sacr_residual_scale",
        ))
    }
    assert optimized == expected

    BaseTrainTester._set_source_moe_train_mode(model, args)
    assert model.training is False
    assert model.backbone.training is False
    assert model.joint_query_quality_reranker.training is True
    assert model.structured_slot_builder.training is True
    assert model.sacr_head.training is True



def _manual_joint_outputs(scores, baseline_indices, valid, source_ranks=None):
    outputs = {
        "scores": scores,
        "baseline_indices": baseline_indices,
        "selected_indices": scores.argmax(dim=1),
        "box_logits": torch.zeros(
            *scores.shape, 2, dtype=scores.dtype, device=scores.device
        ),
        "box_iou": torch.full_like(scores, 0.5),
        "mask_logits": torch.zeros(
            *scores.shape, 2, dtype=scores.dtype, device=scores.device
        ),
        "mask_iou": torch.full_like(scores, 0.5),
        "valid_mask": valid,
    }
    if source_ranks is not None:
        outputs["source_mix_ranks"] = source_ranks
        outputs["source_mix_validity"] = torch.ones_like(
            source_ranks, dtype=torch.bool
        )
        outputs["source_mix_weights"] = torch.full_like(
            source_ranks, 1.0 / source_ranks.shape[-1]
        )
    return outputs


def test_v51_smooth_metric_utility_rewards_both_box_thresholds():
    box = torch.tensor([[0.24, 0.26, 0.49, 0.51]])
    mask = torch.full_like(box, 0.5)
    utility = smooth_metric_aligned_query_utility(
        box, mask, temperature=0.05, mask_weight=0.25
    )
    assert bool((utility[:, 1:] > utility[:, :-1]).all().item())
    assert (utility[0, 3] - utility[0, 2]) > (
        utility[0, 1] - utility[0, 0]
    )


def test_v51_direct_residual_scale_bounds_free_residual_path():
    torch.manual_seed(5101)
    full = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        direct_residual_scale=1.0,
    )
    bounded = copy.deepcopy(full)
    bounded.direct_residual_scale = 0.25
    with torch.no_grad():
        full.residual_head.weight.normal_()
        full.residual_head.bias.fill_(0.3)
        bounded.load_state_dict(full.state_dict())
    features, baseline, valid = _inputs()
    full_out = full(features, baseline, valid)
    bounded_out = bounded(features, baseline, valid)
    torch.testing.assert_close(
        full_out["raw_direct_residual_logit"],
        bounded_out["raw_direct_residual_logit"],
    )
    torch.testing.assert_close(
        bounded_out["direct_residual_logit"],
        0.25 * full_out["direct_residual_logit"],
    )


def test_v51_bidirectional_anchor_repairs_parent_misses():
    scores = torch.tensor([[0.60, 0.40, 0.20]], requires_grad=True)
    valid = torch.ones_like(scores, dtype=torch.bool)
    outputs = _manual_joint_outputs(
        scores, torch.tensor([0]), valid
    )
    box = torch.tensor([[0.10, 0.70, 0.20]])
    mask = torch.zeros_like(box)
    legacy = compute_joint_query_quality_loss(
        outputs, box, mask, quality_loss_weight=0.0,
        anchor_loss_weight=1.0, bidirectional_anchor=False,
    )
    repaired = compute_joint_query_quality_loss(
        outputs, box, mask, quality_loss_weight=0.0,
        anchor_loss_weight=1.0, bidirectional_anchor=True,
        anchor_margin=0.05, anchor_margin_050=0.10,
    )
    assert legacy["repair_anchor_loss"].item() == 0.0
    torch.testing.assert_close(
        repaired["repair_anchor_loss"], torch.tensor(0.275)
    )
    assert repaired["protect_anchor_loss"].item() == 0.0
    repaired["loss"].backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum().item() > 0.0


def test_v51_bidirectional_anchor_preserves_correct_parent():
    scores = torch.tensor([[0.80, 0.90, 0.20]], requires_grad=True)
    valid = torch.ones_like(scores, dtype=torch.bool)
    outputs = _manual_joint_outputs(
        scores, torch.tensor([0]), valid
    )
    box = torch.tensor([[0.80, 0.10, 0.20]])
    mask = torch.zeros_like(box)
    supervision = compute_joint_query_quality_loss(
        outputs, box, mask, quality_loss_weight=0.0,
        anchor_loss_weight=1.0, bidirectional_anchor=True,
        anchor_margin=0.05, anchor_margin_050=0.10,
    )
    torch.testing.assert_close(
        supervision["protect_anchor_loss"], torch.tensor(0.175)
    )
    assert supervision["repair_anchor_loss"].item() == 0.0


def test_v51_candidate_union_and_pairwise_gain_are_trainable():
    scores = torch.tensor(
        [[0.90, 0.80, 0.10, 0.00]], requires_grad=True
    )
    valid = torch.ones_like(scores, dtype=torch.bool)
    source_ranks = torch.tensor([[
        [0.1, 0.0],
        [1.0, 0.2],
        [0.2, 1.0],
        [0.0, 0.1],
    ]])
    outputs = _manual_joint_outputs(
        scores, torch.tensor([0]), valid, source_ranks=source_ranks
    )
    box = torch.tensor([[0.10, 0.40, 0.80, 0.20]])
    mask = torch.tensor([[0.10, 0.30, 0.70, 0.20]])
    supervision = compute_joint_query_quality_loss(
        outputs, box, mask,
        use_metric_aligned_utility=True,
        metric_utility_temperature=0.05,
        pairwise_loss_weight=1.0,
        deploy_candidate_top_k=1,
        source_candidate_top_k=1,
        oracle_candidate_top_k=1,
        quality_loss_weight=0.0,
        anchor_loss_weight=0.0,
    )
    torch.testing.assert_close(
        supervision["stats"]["candidate_query_ratio"], torch.tensor(0.75)
    )
    assert supervision["pairwise_loss"].item() > 0.0
    supervision["loss"].backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum().item() > 0.0


def test_v51_source_candidate_union_is_source_permutation_invariant():
    scores = torch.tensor([[0.90, 0.80, 0.10, 0.00]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    source_ranks = torch.tensor([[
        [0.1, 0.0],
        [1.0, 0.2],
        [0.2, 1.0],
        [0.0, 0.1],
    ]])
    box = torch.tensor([[0.10, 0.40, 0.80, 0.20]])
    mask = torch.tensor([[0.10, 0.30, 0.70, 0.20]])
    kwargs = {
        "use_metric_aligned_utility": True,
        "pairwise_loss_weight": 1.0,
        "deploy_candidate_top_k": 1,
        "source_candidate_top_k": 1,
        "oracle_candidate_top_k": 1,
        "quality_loss_weight": 0.0,
        "anchor_loss_weight": 0.0,
    }
    first = compute_joint_query_quality_loss(
        _manual_joint_outputs(
            scores, torch.tensor([0]), valid,
            source_ranks=source_ranks,
        ),
        box, mask, **kwargs
    )
    second = compute_joint_query_quality_loss(
        _manual_joint_outputs(
            scores, torch.tensor([0]), valid,
            source_ranks=source_ranks.flip(-1),
        ),
        box, mask, **kwargs
    )
    torch.testing.assert_close(first["loss"], second["loss"])
    torch.testing.assert_close(
        first["stats"]["candidate_query_ratio"],
        second["stats"]["candidate_query_ratio"],
    )



def test_candidate_mask_loss_supports_variable_superpoint_counts():
    text = [
        torch.randn(3, 4, requires_grad=True),
        torch.randn(3, 3, requires_grad=True),
    ]
    query = [
        torch.randn(3, 4, requires_grad=True),
        torch.randn(3, 3, requires_grad=True),
    ]
    weights = [
        torch.full((3,), 0.5),
        torch.full((3,), 0.5),
    ]
    gt_masks = torch.tensor([
        [[1.0, 0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0, 1.0]],
    ])
    superpoints = torch.tensor([
        [0, 1, 2, 3],
        [0, 1, 2, 2],
    ], dtype=torch.long)
    candidate_mask = torch.tensor([
        [True, False, True],
        [False, True, True],
    ])
    losses = compute_joint_query_mask_candidate_loss(
        text, query, weights, gt_masks, superpoints, candidate_mask,
        compute_lovasz=True,
    )
    total = sum(losses.values())
    assert torch.isfinite(total)
    assert all(value.item() > 0.0 for value in losses.values())
    total.backward()
    for tensor in text + query:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()



def test_straight_through_rank_has_finite_gradient_for_constant_scores():
    scores = torch.zeros(2, 5, requires_grad=True)
    valid = torch.tensor([
        [True, True, True, True, True],
        [True, True, False, False, False],
    ])
    ranks = _straight_through_rank_normalize(scores, valid)
    weights = torch.arange(5, dtype=scores.dtype).unsqueeze(0)
    (ranks * weights).sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_v63_setwise_tier_step_zero_is_parent_identity():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        parent_transition_candidate_top_k=5,
    ).eval()
    outputs = model(features, baseline, valid)
    assert torch.equal(outputs["scores"], baseline)
    assert torch.equal(
        outputs["selected_indices"], baseline.argmax(dim=1)
    )
    assert torch.count_nonzero(outputs["residual"]) == 0
    torch.testing.assert_close(
        outputs["setwise_tier_advantage"],
        torch.zeros(2, 5, 2),
    )
    assert outputs["setwise_tier_branch_scores"].shape == (2, 5, 2)


def test_v63_setwise_tier_loss_balances_repair_and_stay_rows():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    # Row zero has a tier-2 repair for a tier-1 parent. Row one already has
    # a tier-2 parent and must stay despite several plausible alternatives.
    box = torch.tensor([
        [0.40, 0.10, 0.70, 0.30, 0.20],
        [0.10, 0.80, 0.70, 0.55, 0.20],
    ])
    mask = torch.tensor([
        [0.40, 0.10, 0.75, 0.30, 0.20],
        [0.10, 0.80, 0.70, 0.55, 0.20],
    ])
    supervision = compute_joint_query_quality_loss(
        outputs, box, mask,
        listwise_loss_weight=0.0,
        transition_loss_weight=1.0,
        quality_loss_weight=0.0,
        anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0,
    )
    stats = supervision["stats"]
    torch.testing.assert_close(
        stats["setwise_tier_repair_row_ratio"],
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        stats["setwise_tier_stay_row_ratio"],
        torch.tensor(0.5),
    )
    assert supervision["transition_loss"].item() > 0.0
    torch.testing.assert_close(
        supervision["loss"], supervision["transition_loss"]
    )
    supervision["loss"].backward()
    gradient = model.setwise_tier_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0
    assert model.decomposed_transition_head is None
    assert model.factorized_hit_head is None
    assert model.residual_head.weight.grad is None


def test_v64_setwise_rank_loss_is_opt_in_and_has_finite_gradient():
    torch.manual_seed(640)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47, 0.46]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    # Parent query 0 is tier 1; query 2 is the sole best tier-2 repair.
    # Under V67, safe-neutral query 1 is not a negative. Make true @0.25
    # break query 3 outrank repair query 2 to retain a deterministic
    # negative rank gap for this legacy V64 opt-in test.
    branch_scores = outputs["setwise_tier_branch_scores"]
    custom = branch_scores.detach().new_full(branch_scores.shape, -0.03)
    custom[:, 0, :] = 0.0
    custom[:, 2, :] = -0.02
    custom[:, 3, :] = -0.01
    outputs = dict(outputs)
    outputs["setwise_tier_branch_scores"] = (
        branch_scores + custom - branch_scores.detach()
    )
    box = torch.tensor([[0.40, 0.30, 0.70, 0.20, 0.10]])
    mask = box.clone()
    common = dict(
        listwise_loss_weight=0.0,
        transition_loss_weight=1.0,
        quality_loss_weight=0.0,
        anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0,
    )
    legacy = compute_joint_query_quality_loss(
        outputs, box, mask, setwise_rank_loss_weight=0.0, **common
    )
    ranked = compute_joint_query_quality_loss(
        outputs, box, mask, setwise_rank_loss_weight=2.0, **common
    )
    rank_loss = ranked["stats"]["setwise_tier_rank_loss"]
    assert rank_loss.item() > 0.0
    torch.testing.assert_close(
        ranked["transition_loss"],
        legacy["transition_loss"] + 2.0 * rank_loss,
    )
    assert ranked["stats"][
        "setwise_tier_branch_025_rank_margin"
    ].item() < 0.0
    assert ranked["stats"][
        "setwise_tier_branch_025_rank_recall"
    ].item() == 0.0
    ranked["loss"].backward()
    gradient = model.setwise_tier_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_v65_setwise_repair_boundary_loss_is_opt_in_and_finite():
    torch.manual_seed(650)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47, 0.46]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    # Make the best repair (query 2) rank above every non-repair while it
    # remains below the deployment boundary. Offsets are detached constants,
    # so gradients still flow into the setwise head.
    branch_scores = outputs["setwise_tier_branch_scores"]
    target_scores = branch_scores.detach().new_full(
        branch_scores.shape, -0.03)
    target_scores[:, 0, :] = 0.0
    target_scores[:, 2, :] = -0.01
    outputs = dict(outputs)
    outputs["setwise_tier_branch_scores"] = (
        branch_scores + target_scores - branch_scores.detach()
    )
    box = torch.tensor([[0.40, 0.30, 0.70, 0.20, 0.10]])
    common = dict(
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=2.0, quality_loss_weight=0.0,
        anchor_loss_weight=0.0, source_mix_loss_weight=0.0,
    )
    legacy = compute_joint_query_quality_loss(
        outputs, box, box, setwise_repair_boundary_loss_weight=0.0,
        **common)
    weighted = compute_joint_query_quality_loss(
        outputs, box, box, setwise_repair_boundary_loss_weight=2.0,
        **common)
    boundary_loss = weighted["stats"][
        "setwise_tier_repair_boundary_loss"]
    assert boundary_loss.item() > 0.0
    assert weighted["stats"][
        "setwise_tier_repair_boundary_eligible_ratio"
    ].item() == 1.0
    torch.testing.assert_close(
        weighted["transition_loss"],
        legacy["transition_loss"] + 2.0 * boundary_loss,
    )
    weighted["loss"].backward()
    gradient = model.setwise_tier_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_v66_setwise_negative_tail_loss_is_opt_in_and_finite():
    torch.manual_seed(660)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47, 0.46]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    box = torch.tensor([[0.40, 0.30, 0.70, 0.20, 0.10]])
    common = dict(
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=2.0, quality_loss_weight=0.0,
        anchor_loss_weight=0.0, source_mix_loss_weight=0.0,
    )
    legacy = compute_joint_query_quality_loss(
        outputs, box, box, setwise_negative_tail_loss_weight=0.0,
        **common)
    weighted = compute_joint_query_quality_loss(
        outputs, box, box, setwise_negative_tail_loss_weight=1.0,
        **common)
    tail_loss = weighted["stats"]["setwise_tier_negative_tail_loss"]
    assert tail_loss.item() > 0.0
    torch.testing.assert_close(
        weighted["transition_loss"],
        legacy["transition_loss"] + tail_loss,
    )
    weighted["loss"].backward()
    gradient = model.setwise_tier_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_v64_setwise_rank_loss_rejects_negative_weight():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    with pytest.raises(ValueError, match="setwise_rank_loss_weight"):
        compute_joint_query_quality_loss(
            outputs, torch.rand_like(baseline), torch.rand_like(baseline),
            listwise_loss_weight=0.0,
            transition_loss_weight=1.0,
            setwise_rank_loss_weight=-1.0,
            quality_loss_weight=0.0,
            anchor_loss_weight=0.0,
            source_mix_loss_weight=0.0,
        )


def test_v66_setwise_negative_tail_loss_rejects_negative_weight():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    with pytest.raises(
            ValueError, match="setwise_negative_tail_loss_weight"):
        compute_joint_query_quality_loss(
            outputs, torch.rand_like(baseline), torch.rand_like(baseline),
            transition_loss_weight=1.0,
            setwise_negative_tail_loss_weight=-1.0,
        )


def test_v65_setwise_repair_boundary_loss_rejects_negative_weight():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    with pytest.raises(
            ValueError, match="setwise_repair_boundary_loss_weight"):
        compute_joint_query_quality_loss(
            outputs, torch.rand_like(baseline), torch.rand_like(baseline),
            transition_loss_weight=1.0,
            setwise_repair_boundary_loss_weight=-1.0,
        )


def test_v63_setwise_tier_rejects_unsafe_contracts():
    with pytest.raises(ValueError, match="requires preserved parent"):
        JointQueryQualityReranker(
            8, use_setwise_tier_advantage=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        JointQueryQualityReranker(
            8, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decomposed_transition_advantage=True,
        )


def test_v63c_setwise_tier_advantage_is_candidate_centered():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        parent_transition_candidate_top_k=5,
    ).eval()
    with torch.no_grad():
        model.setwise_tier_head[-1].weight.normal_(mean=0.0, std=0.2)
    outputs = model(features, baseline, valid)
    candidate = outputs["parent_transition_candidate_mask"].clone()
    row = torch.arange(features.shape[0])
    candidate[row, outputs["baseline_indices"]] = False
    advantage = outputs["setwise_tier_advantage"]
    candidate_count = candidate.sum(dim=1, keepdim=True).clamp(min=1)
    candidate_mean = (
        advantage * candidate.unsqueeze(-1)
    ).sum(dim=1) / candidate_count
    torch.testing.assert_close(
        candidate_mean, torch.zeros_like(candidate_mean),
        atol=1e-6, rtol=1e-6,
    )
    assert model.setwise_tier_head[-1].bias is None



def test_v69_decoupled_setwise_step_zero_is_parent_identity():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        parent_transition_candidate_top_k=5,
    ).eval()
    outputs = model(features, baseline, valid)
    assert torch.equal(outputs["scores"], baseline)
    assert torch.equal(outputs["selected_indices"], baseline.argmax(dim=1))
    assert torch.count_nonzero(outputs["residual"]) == 0
    assert model.setwise_tier_head is None
    assert model.setwise_promotion_head is not None
    assert model.setwise_safety_head is not None
    torch.testing.assert_close(
        outputs["setwise_decoupled_promotion_safety"], torch.tensor(1.0))


def test_v69_promotion_rank_does_not_update_safety_output_head():
    torch.manual_seed(690)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47, 0.46]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    branch_scores = outputs["setwise_tier_branch_scores"]
    custom = branch_scores.detach().new_full(branch_scores.shape, -0.03)
    custom[:, 0, :] = 0.0
    custom[:, 2, 0] = -0.02
    custom[:, 3, 0] = -0.01
    custom[:, 2, 1] = 0.03
    outputs = dict(outputs)
    outputs["setwise_tier_branch_scores"] = (
        branch_scores + custom - branch_scores.detach()
    )
    box = torch.tensor([[0.40, 0.30, 0.70, 0.35, 0.30]])
    common = dict(
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0,
    )
    base_loss = compute_joint_query_quality_loss(
        outputs, box, box, setwise_rank_loss_weight=0.0, **common
    )["loss"]
    ranked_loss = compute_joint_query_quality_loss(
        outputs, box, box, setwise_rank_loss_weight=2.0, **common
    )["loss"]
    # Isolate only V64's candidate-internal rank term.  The base safety
    # objective is still allowed to teach safe repairs to cross the boundary.
    (ranked_loss - base_loss).backward()
    promotion_grad = model.setwise_promotion_head[-1].weight.grad
    safety_grad = model.setwise_safety_head[-1].weight.grad
    assert promotion_grad is not None
    assert promotion_grad.abs().sum().item() > 0.0
    assert safety_grad is not None
    torch.testing.assert_close(safety_grad, torch.zeros_like(safety_grad))


def test_v69_mask025_hazard_is_vetoed_only_by_safety_output_head():
    torch.manual_seed(691)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47, 0.46]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    branch_scores = outputs["setwise_tier_branch_scores"]
    custom = branch_scores.detach().new_full(branch_scores.shape, -0.04)
    custom[:, 0, :] = 0.0
    custom[:, 2, :] = 0.03
    outputs = dict(outputs)
    outputs["setwise_tier_branch_scores"] = (
        branch_scores + custom - branch_scores.detach()
    )
    box = torch.tensor([[0.40, 0.35, 0.70, 0.30, 0.20]])
    mask = torch.tensor([[0.40, 0.35, 0.10, 0.30, 0.20]])
    supervision = compute_joint_query_quality_loss(
        outputs, box, mask, listwise_loss_weight=0.0,
        transition_loss_weight=1.0, setwise_rank_loss_weight=0.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0,
    )
    assert supervision["stats"]["setwise_safety_hazard_query_ratio"].item() > 0
    supervision["loss"].backward()
    promotion_grad = model.setwise_promotion_head[-1].weight.grad
    safety_grad = model.setwise_safety_head[-1].weight.grad
    assert promotion_grad is not None
    assert safety_grad is not None
    torch.testing.assert_close(promotion_grad, torch.zeros_like(promotion_grad))
    assert safety_grad.abs().sum().item() > 0.0


def test_v69_decoupled_setwise_requires_setwise_mode():
    with pytest.raises(ValueError, match="require setwise tier advantage"):
        JointQueryQualityReranker(
            8, preserve_parent_score=True,
            use_decoupled_setwise_heads=True,
        )



def test_v70_dense_safety_is_opt_in_row_balanced_and_head_isolated():
    torch.manual_seed(700)
    features = torch.randn(2, 5, 8)
    baseline = torch.tensor([
        [0.50, 0.49, 0.48, 0.47, 0.46],
        [0.50, 0.49, 0.48, 0.47, 0.46],
    ])
    valid = torch.ones(2, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    branch_scores = outputs["setwise_tier_branch_scores"]
    custom = branch_scores.detach().new_full(branch_scores.shape, -0.04)
    custom[:, 0, :] = 0.0
    # Row zero has one mask hazard; row one has three.  All four use the same
    # safety margin, so row-balanced dense loss must equal one softplus value.
    custom[0, 2, 1] = 0.03
    custom[1, 1:4, 1] = 0.03
    outputs = dict(outputs)
    outputs["setwise_tier_branch_scores"] = (
        branch_scores + custom - branch_scores.detach()
    )
    box = torch.tensor([
        [0.40, 0.35, 0.70, 0.30, 0.30],
        [0.40, 0.70, 0.65, 0.60, 0.30],
    ])
    mask = torch.tensor([
        [0.40, 0.35, 0.10, 0.30, 0.30],
        [0.40, 0.10, 0.10, 0.10, 0.30],
    ])
    common = dict(
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0, quality_loss_weight=0.0,
        anchor_loss_weight=0.0, source_mix_loss_weight=0.0,
    )
    base = compute_joint_query_quality_loss(
        outputs, box, mask, setwise_dense_safety_loss_weight=0.0,
        **common)
    dense = compute_joint_query_quality_loss(
        outputs, box, mask, setwise_dense_safety_loss_weight=2.0,
        **common)
    dense_loss = dense["stats"]["setwise_dense_safety_loss"]
    expected = F.softplus(torch.tensor(0.02 + 0.03))
    torch.testing.assert_close(dense_loss, expected)
    torch.testing.assert_close(
        dense["transition_loss"], base["transition_loss"] + 2.0 * dense_loss)
    assert dense["stats"][
        "setwise_dense_safety_violation_ratio"].item() == 1.0
    (dense["loss"] - base["loss"]).backward()
    promotion_grad = model.setwise_promotion_head[-1].weight.grad
    safety_grad = model.setwise_safety_head[-1].weight.grad
    assert promotion_grad is not None
    assert safety_grad is not None
    torch.testing.assert_close(promotion_grad, torch.zeros_like(promotion_grad))
    assert safety_grad.abs().sum().item() > 0.0


def test_v70_dense_safety_rejects_legacy_shared_setwise_mode():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    with pytest.raises(ValueError, match="requires decoupled setwise heads"):
        compute_joint_query_quality_loss(
            outputs, torch.rand_like(baseline), torch.rand_like(baseline),
            listwise_loss_weight=0.0, transition_loss_weight=1.0,
            setwise_dense_safety_loss_weight=1.0,
            quality_loss_weight=0.0, anchor_loss_weight=0.0,
            source_mix_loss_weight=0.0,
        )


def test_v71_balanced_safety_supervises_safe_and_hazard_classes_equally():
    torch.manual_seed(710)
    features = torch.randn(2, 5, 8)
    baseline = torch.tensor([
        [0.50, 0.49, 0.48, 0.47, 0.46],
        [0.50, 0.49, 0.48, 0.47, 0.46],
    ])
    valid = torch.ones(2, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    branch_scores = outputs["setwise_tier_branch_scores"]
    custom = branch_scores.detach().new_full(branch_scores.shape, -0.04)
    custom[:, 0, :] = 0.0
    # Every safe candidate has +0.04 safety margin. Row zero has one hazard
    # and row one has three; all hazards have -0.03. Candidate, row, and
    # class balancing therefore reduce to one softplus value per class.
    custom[:, 1:, 1] = 0.04
    custom[0, 2, 1] = -0.03
    custom[1, 1:4, 1] = -0.03
    outputs = dict(outputs)
    outputs["setwise_tier_branch_scores"] = (
        branch_scores + custom - branch_scores.detach()
    )
    box = torch.tensor([
        [0.40, 0.35, 0.70, 0.30, 0.30],
        [0.40, 0.70, 0.65, 0.60, 0.30],
    ])
    mask = torch.tensor([
        [0.40, 0.35, 0.10, 0.30, 0.30],
        [0.40, 0.10, 0.10, 0.10, 0.30],
    ])
    common = dict(
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0, quality_loss_weight=0.0,
        anchor_loss_weight=0.0, source_mix_loss_weight=0.0,
    )
    base = compute_joint_query_quality_loss(
        outputs, box, mask, setwise_balanced_safety_loss_weight=0.0,
        **common)
    balanced = compute_joint_query_quality_loss(
        outputs, box, mask, setwise_balanced_safety_loss_weight=2.0,
        **common)
    balanced_loss = balanced["stats"]["setwise_balanced_safety_loss"]
    expected_safe = F.softplus(torch.tensor(0.02 - 0.04))
    expected_hazard = F.softplus(torch.tensor(0.02 - 0.03))
    expected = 0.5 * (expected_safe + expected_hazard)
    torch.testing.assert_close(balanced_loss, expected)
    torch.testing.assert_close(
        balanced["transition_loss"],
        base["transition_loss"] + 2.0 * balanced_loss,
    )
    assert balanced["stats"][
        "setwise_balanced_safety_safe_violation_ratio"].item() == 0.0
    assert balanced["stats"][
        "setwise_balanced_safety_hazard_violation_ratio"].item() == 0.0
    (balanced["loss"] - base["loss"]).backward()
    promotion_grad = model.setwise_promotion_head[-1].weight.grad
    safety_grad = model.setwise_safety_head[-1].weight.grad
    assert promotion_grad is not None
    assert safety_grad is not None
    torch.testing.assert_close(promotion_grad, torch.zeros_like(promotion_grad))
    assert safety_grad.abs().sum().item() > 0.0


def test_v71_balanced_safety_rejects_dense_safety_double_counting():
    features, baseline, valid = _inputs()
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    with pytest.raises(ValueError, match="mutually exclusive"):
        compute_joint_query_quality_loss(
            outputs, torch.rand_like(baseline), torch.rand_like(baseline),
            listwise_loss_weight=0.0, transition_loss_weight=1.0,
            setwise_dense_safety_loss_weight=1.0,
            setwise_balanced_safety_loss_weight=1.0,
            quality_loss_weight=0.0, anchor_loss_weight=0.0,
            source_mix_loss_weight=0.0,
        )


def test_v72_factorized_safety_is_identity_and_uses_three_vetoes():
    features, baseline, valid = _inputs()
    with pytest.raises(
            ValueError,
            match="factorized setwise safety requires decoupled"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            use_setwise_tier_advantage=True,
            use_factorized_setwise_safety=True,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        parent_transition_candidate_top_k=5,
    ).eval()
    outputs = model(features, baseline, valid)
    assert torch.equal(outputs["scores"], baseline)
    assert torch.equal(
        outputs["selected_indices"], baseline.argmax(dim=1)
    )
    assert model.setwise_safety_head[-1].out_features == 3
    assert torch.count_nonzero(model.setwise_safety_head[-1].weight) == 0
    assert outputs["setwise_safety_criterion_scores"].shape == (2, 5, 3)
    torch.testing.assert_close(
        outputs["setwise_tier_branch_scores"][..., 1],
        outputs["setwise_safety_criterion_scores"].min(dim=-1).values,
    )
    torch.testing.assert_close(
        outputs["setwise_factorized_safety"], torch.tensor(1.0)
    )


def test_v72_factorized_safety_balances_each_criterion_and_gradients():
    torch.manual_seed(720)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47, 0.46]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    criterion_scores = outputs["setwise_safety_criterion_scores"]
    target = criterion_scores.detach().new_tensor([[[
        0.00, 0.00, 0.00,
    ], [
        -0.03, -0.04, -0.05,
    ], [
        0.04, 0.03, 0.02,
    ], [
        0.05, -0.02, -0.03,
    ], [
        0.06, 0.04, 0.03,
    ]]])
    outputs = dict(outputs)
    outputs["setwise_safety_criterion_scores"] = (
        criterion_scores + target - criterion_scores.detach()
    )
    box = torch.tensor([[0.60, 0.20, 0.55, 0.40, 0.70]])
    mask = torch.tensor([[0.40, 0.10, 0.30, 0.20, 0.50]])
    common = dict(
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0,
    )
    base = compute_joint_query_quality_loss(
        outputs, box, mask,
        setwise_factorized_safety_loss_weight=0.0,
        **common,
    )
    factorized = compute_joint_query_quality_loss(
        outputs, box, mask,
        setwise_factorized_safety_loss_weight=2.0,
        **common,
    )
    criterion_hazards = torch.tensor([[
        [True, True, True],
        [False, False, False],
        [False, True, True],
        [False, False, False],
    ]])
    candidate_target = target[:, 1:, :]
    expected_criteria = []
    for index in range(3):
        hazard = criterion_hazards[..., index]
        safe = ~hazard
        safe_loss = F.softplus(
            0.02 - candidate_target[..., index][safe]
        ).mean()
        hazard_loss = F.softplus(
            0.02 + candidate_target[..., index][hazard]
        ).mean()
        expected_criteria.append(0.5 * (safe_loss + hazard_loss))
    expected = torch.stack(expected_criteria).mean()
    torch.testing.assert_close(
        factorized["stats"]["setwise_factorized_safety_loss"], expected
    )
    torch.testing.assert_close(
        factorized["transition_loss"],
        base["transition_loss"] + 2.0 * expected,
    )
    (factorized["loss"] - base["loss"]).backward()
    promotion_grad = model.setwise_promotion_head[-1].weight.grad
    safety_grad = model.setwise_safety_head[-1].weight.grad
    assert promotion_grad is not None
    torch.testing.assert_close(
        promotion_grad, torch.zeros_like(promotion_grad)
    )
    assert safety_grad is not None
    assert safety_grad.shape[0] == 3
    assert bool((safety_grad.abs().sum(dim=1) > 0.0).all().item())


def test_v72_factorized_safety_loss_rejects_wrong_head_and_double_counting():
    features, baseline, valid = _inputs()
    scalar_model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        parent_transition_candidate_top_k=5,
    )
    scalar_outputs = scalar_model(features, baseline, valid)
    common = dict(
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0,
    )
    with pytest.raises(ValueError, match="requires factorized setwise"):
        compute_joint_query_quality_loss(
            scalar_outputs, torch.rand_like(baseline),
            torch.rand_like(baseline),
            setwise_factorized_safety_loss_weight=1.0,
            **common,
        )
    factorized_model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        parent_transition_candidate_top_k=5,
    )
    factorized_outputs = factorized_model(features, baseline, valid)
    with pytest.raises(ValueError, match="mutually exclusive"):
        compute_joint_query_quality_loss(
            factorized_outputs, torch.rand_like(baseline),
            torch.rand_like(baseline),
            setwise_balanced_safety_loss_weight=1.0,
            setwise_factorized_safety_loss_weight=1.0,
            **common,
        )


def test_v73_factorized_risk_bound_is_identity_and_uses_six_vetoes():
    features, baseline, valid = _inputs()
    with pytest.raises(
            ValueError, match="risk bound requires factorized safety"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            max_delta=0.25, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decoupled_setwise_heads=True,
            use_factorized_setwise_risk_bound=True,
            parent_transition_candidate_top_k=5,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        parent_transition_candidate_top_k=5,
    ).eval()
    outputs = model(features, baseline, valid)
    assert torch.equal(outputs["scores"], baseline)
    assert torch.equal(
        outputs["selected_indices"], baseline.argmax(dim=1)
    )
    assert model.setwise_safety_head[-1].out_features == 6
    assert torch.count_nonzero(model.setwise_safety_head[-1].weight) == 0
    assert outputs["setwise_safety_bound_scores"].shape == (2, 5, 3, 2)
    torch.testing.assert_close(
        outputs["setwise_safety_criterion_scores"],
        outputs["setwise_safety_bound_scores"].min(dim=-1).values,
    )
    torch.testing.assert_close(
        outputs["setwise_tier_branch_scores"][..., 1],
        outputs["setwise_safety_bound_scores"].amin(dim=(-1, -2)),
    )
    torch.testing.assert_close(
        outputs["setwise_factorized_risk_bound"], torch.tensor(1.0)
    )


def test_v73_risk_bound_loss_trains_point_and_cost_weighted_guard():
    torch.manual_seed(730)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47, 0.46]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    bound_scores = outputs["setwise_safety_bound_scores"]
    target = bound_scores.detach().new_tensor([[[[
        0.00, 0.00,
    ], [
        0.00, 0.00,
    ], [
        0.00, 0.00,
    ]], [[
        -0.03, -0.05,
    ], [
        -0.04, -0.06,
    ], [
        -0.05, -0.07,
    ]], [[
        0.04, 0.02,
    ], [
        0.03, -0.01,
    ], [
        0.02, -0.02,
    ]], [[
        0.05, 0.01,
    ], [
        -0.02, -0.04,
    ], [
        -0.03, -0.05,
    ]], [[
        0.06, 0.03,
    ], [
        0.04, 0.01,
    ], [
        0.03, 0.00,
    ]]]])
    outputs = dict(outputs)
    outputs["setwise_safety_bound_scores"] = (
        bound_scores + target - bound_scores.detach()
    )
    outputs["setwise_safety_criterion_scores"] = outputs[
        "setwise_safety_bound_scores"
    ].min(dim=-1).values
    box = torch.tensor([[0.60, 0.20, 0.55, 0.40, 0.70]])
    mask = torch.tensor([[0.40, 0.10, 0.30, 0.20, 0.50]])
    common = dict(
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    base = compute_joint_query_quality_loss(
        outputs, box, mask,
        setwise_factorized_risk_bound_loss_weight=0.0,
        **common,
    )
    bounded = compute_joint_query_quality_loss(
        outputs, box, mask,
        setwise_factorized_risk_bound_loss_weight=2.0,
        **common,
    )
    criterion_hazards = torch.tensor([[[
        True, True, True,
    ], [
        False, False, False,
    ], [
        False, True, True,
    ], [
        False, False, False,
    ]]])
    candidate_target = target[:, 1:, :, :]
    point_terms = []
    guard_terms = []
    for index in range(3):
        hazard = criterion_hazards[..., index]
        safe = ~hazard
        point_safe = F.softplus(
            0.02 - candidate_target[..., index, 0][safe]
        ).mean()
        point_hazard = F.softplus(
            0.02 + candidate_target[..., index, 0][hazard]
        ).mean()
        guard_safe = F.softplus(
            0.02 - candidate_target[..., index, 1][safe]
        ).mean()
        guard_hazard = F.softplus(
            0.02 + candidate_target[..., index, 1][hazard]
        ).mean()
        point_terms.append(0.5 * (point_safe + point_hazard))
        guard_terms.append((guard_safe + 4.0 * guard_hazard) / 5.0)
    point_expected = torch.stack(point_terms).mean()
    guard_expected = torch.stack(guard_terms).mean()
    expected = 0.5 * (point_expected + guard_expected)
    torch.testing.assert_close(
        bounded["stats"]["setwise_factorized_risk_bound_point_loss"],
        point_expected,
    )
    torch.testing.assert_close(
        bounded["stats"]["setwise_factorized_risk_bound_guard_loss"],
        guard_expected,
    )
    torch.testing.assert_close(
        bounded["transition_loss"],
        base["transition_loss"] + 2.0 * expected,
    )
    (bounded["loss"] - base["loss"]).backward()
    promotion_grad = model.setwise_promotion_head[-1].weight.grad
    safety_grad = model.setwise_safety_head[-1].weight.grad
    assert promotion_grad is not None
    torch.testing.assert_close(
        promotion_grad, torch.zeros_like(promotion_grad)
    )
    assert safety_grad is not None
    assert safety_grad.shape[0] == 6
    assert bool((safety_grad.abs().sum(dim=1) > 0.0).all().item())


def test_v73_risk_bound_loss_rejects_wrong_head_and_double_counting():
    features, baseline, valid = _inputs()
    factorized_model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        parent_transition_candidate_top_k=5,
    )
    factorized_outputs = factorized_model(features, baseline, valid)
    common = dict(
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0,
    )
    with pytest.raises(ValueError, match="requires factorized risk-bound"):
        compute_joint_query_quality_loss(
            factorized_outputs, torch.rand_like(baseline),
            torch.rand_like(baseline),
            setwise_factorized_risk_bound_loss_weight=1.0,
            **common,
        )
    bounded_model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        parent_transition_candidate_top_k=5,
    )
    bounded_outputs = bounded_model(features, baseline, valid)
    with pytest.raises(ValueError, match="mutually exclusive"):
        compute_joint_query_quality_loss(
            bounded_outputs, torch.rand_like(baseline),
            torch.rand_like(baseline),
            setwise_factorized_safety_loss_weight=1.0,
            setwise_factorized_risk_bound_loss_weight=1.0,
            **common,
        )


def test_v74_safety_veto_gate_is_identity_and_keeps_safety_absolute():
    features, baseline, valid = _inputs()
    with pytest.raises(
            ValueError, match="veto gate requires decoupled"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            use_setwise_tier_advantage=True,
            use_setwise_safety_veto_gate=True,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        parent_transition_candidate_top_k=5,
    ).eval()
    outputs = model(features, baseline, valid)
    assert torch.equal(outputs["scores"], baseline)
    assert torch.equal(
        outputs["selected_indices"], baseline.argmax(dim=1)
    )
    candidate = outputs["parent_transition_candidate_mask"].clone()
    row = torch.arange(candidate.shape[0])
    candidate[row, outputs["baseline_indices"]] = False
    reachable = candidate & outputs["setwise_tier_reachable_mask"]
    assert bool(reachable.any().item())
    torch.testing.assert_close(
        outputs["setwise_safety_criterion_scores"][reachable],
        torch.zeros_like(
            outputs["setwise_safety_criterion_scores"][reachable]
        ),
    )
    torch.testing.assert_close(
        outputs["setwise_safety_bound_scores"][reachable],
        torch.zeros_like(outputs["setwise_safety_bound_scores"][reachable]),
    )
    torch.testing.assert_close(
        outputs["setwise_tier_branch_scores"][..., 1][reachable],
        torch.zeros_like(
            outputs["setwise_tier_branch_scores"][..., 1][reachable]
        ),
    )
    torch.testing.assert_close(
        outputs["setwise_factorized_risk_bound"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        outputs["setwise_safety_veto_gate"], torch.tensor(1.0)
    )


def test_v74_safety_veto_only_clamps_positive_promotion_magnitude():
    class _CandidateValues(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer("values", torch.tensor(values).float())

        def forward(self, pair_features):
            values = self.values.to(
                device=pair_features.device, dtype=pair_features.dtype
            )
            return values.unsqueeze(0).expand(
                pair_features.shape[0], -1, -1
            )

    features, baseline, valid = _inputs(batch=1)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        parent_transition_candidate_top_k=5,
    ).eval()
    model.setwise_promotion_head = _CandidateValues([
        [0.0], [0.40], [0.10], [-0.10], [-0.40],
    ])
    safe_bounds = [[[0.20, 0.20]] * 3 for _ in range(5)]
    model.setwise_safety_head = _CandidateValues([
        [value for criterion in row for value in criterion]
        for row in safe_bounds
    ])
    safe_outputs = model(features, baseline, valid)
    promotion = safe_outputs["setwise_tier_advantage"][..., 0]
    torch.testing.assert_close(
        safe_outputs["parent_transition_advantage"], promotion
    )

    unsafe_bounds = copy.deepcopy(safe_bounds)
    unsafe_bounds[1][2][1] = -0.20
    model.setwise_safety_head = _CandidateValues([
        [value for criterion in row for value in criterion]
        for row in unsafe_bounds
    ])
    unsafe_outputs = model(features, baseline, valid)
    assert promotion[0, 1] > 0.0
    torch.testing.assert_close(
        unsafe_outputs["parent_transition_advantage"][0, 1],
        promotion.new_zeros(()),
    )
    torch.testing.assert_close(
        unsafe_outputs["parent_transition_advantage"][0, 2:],
        promotion[0, 2:],
    )


def test_v75_cost_calibration_is_identity_and_offsets_only_guard():
    features, baseline, valid = _inputs()
    with pytest.raises(
            ValueError, match="requires factorized risk bound"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            max_delta=0.25, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decoupled_setwise_heads=True,
            use_factorized_setwise_safety=True,
            use_cost_calibrated_setwise_risk_bound=True,
            parent_transition_candidate_top_k=5,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_cost_calibrated_setwise_risk_bound=True,
        parent_transition_break_cost=4.0,
        parent_transition_candidate_top_k=5,
    ).eval()
    outputs = model(features, baseline, valid)
    assert torch.equal(outputs["scores"], baseline)
    candidate = outputs["parent_transition_candidate_mask"].clone()
    row = torch.arange(candidate.shape[0])
    candidate[row, outputs["baseline_indices"]] = False
    reachable = candidate & outputs["setwise_tier_reachable_mask"]
    bounds = outputs["setwise_safety_bound_scores"][reachable]
    torch.testing.assert_close(bounds[..., 0], torch.zeros_like(bounds[..., 0]))
    torch.testing.assert_close(
        bounds[..., 1],
        torch.full_like(bounds[..., 1], math.log(4.0)),
    )
    torch.testing.assert_close(
        outputs["setwise_safety_criterion_scores"][reachable],
        torch.zeros_like(
            outputs["setwise_safety_criterion_scores"][reachable]
        ),
    )
    torch.testing.assert_close(
        outputs["setwise_cost_calibrated_risk_bound"], torch.tensor(1.0)
    )


def test_v75_risk_loss_removes_guard_cost_prior_before_training():
    torch.manual_seed(750)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47, 0.46]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_cost_calibrated_setwise_risk_bound=True,
        parent_transition_break_cost=4.0,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    box = torch.tensor([[0.20, 0.10, 0.55, 0.40, 0.70]])
    mask = torch.tensor([[0.40, 0.10, 0.30, 0.20, 0.50]])
    result = compute_joint_query_quality_loss(
        outputs, box, mask,
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=1.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    expected = F.softplus(torch.tensor(0.02))
    torch.testing.assert_close(
        result["stats"]["setwise_factorized_risk_bound_point_loss"],
        expected,
    )
    torch.testing.assert_close(
        result["stats"]["setwise_factorized_risk_bound_guard_loss"],
        expected,
    )


def test_v76_slack_quantile_bound_contract_and_step_zero_identity():
    features, baseline, valid = _inputs()
    common = dict(
        input_dim=8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        parent_transition_candidate_top_k=5,
    )
    with pytest.raises(
            ValueError, match="requires factorized risk bound"):
        JointQueryQualityReranker(
            **common, use_setwise_safety_slack_quantile_bound=True
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        JointQueryQualityReranker(
            **common, use_factorized_setwise_risk_bound=True,
            use_cost_calibrated_setwise_risk_bound=True,
            use_setwise_safety_slack_quantile_bound=True,
        )
    model = JointQueryQualityReranker(
        **common, use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        parent_transition_break_cost=4.0,
    ).eval()
    outputs = model(features, baseline, valid)
    assert torch.equal(outputs["scores"], baseline)
    assert torch.equal(
        outputs["selected_indices"], baseline.argmax(dim=1)
    )
    candidate = outputs["parent_transition_candidate_mask"].clone()
    row = torch.arange(candidate.shape[0])
    candidate[row, outputs["baseline_indices"]] = False
    reachable = candidate & outputs["setwise_tier_reachable_mask"]
    torch.testing.assert_close(
        outputs["setwise_safety_bound_scores"][reachable],
        torch.zeros_like(outputs["setwise_safety_bound_scores"][reachable]),
    )
    torch.testing.assert_close(
        outputs["setwise_safety_slack_quantile_bound"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        outputs["setwise_cost_calibrated_risk_bound"], torch.tensor(0.0)
    )


def test_v76_slack_quantile_loss_matches_parent_relative_boundaries():
    torch.manual_seed(760)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47, 0.46]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        parent_transition_break_cost=4.0,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    box = torch.tensor([[0.60, 0.20, 0.55, 0.40, 0.70]])
    mask = torch.tensor([[0.40, 0.10, 0.30, 0.20, 0.50]])
    result = compute_joint_query_quality_loss(
        outputs, box, mask,
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=1.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    stats = result["stats"]
    torch.testing.assert_close(
        stats["setwise_factorized_risk_bound_point_loss"],
        torch.tensor((0.95 + 0.325 + 0.50) / 3.0),
    )
    torch.testing.assert_close(
        stats["setwise_factorized_risk_bound_guard_loss"],
        torch.tensor((1.10 + 0.925 + 1.10) / 3.0),
    )
    torch.testing.assert_close(
        stats["setwise_safety_slack_box025_quantile_coverage"],
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        stats["setwise_safety_slack_box050_quantile_coverage"],
        torch.tensor(0.50),
    )
    result["loss"].backward()
    final = model.setwise_safety_head[-1]
    assert final.weight.grad is not None
    assert bool(torch.isfinite(final.weight.grad).all().item())
    assert int(torch.count_nonzero(final.weight.grad).item()) > 0


def test_v77_slack_pairwise_order_cancels_row_bias_and_adds_exact_gaps():
    torch.manual_seed(770)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47, 0.46]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires slack quantile bound"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            max_delta=0.25, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decoupled_setwise_heads=True,
            use_factorized_setwise_safety=True,
            use_factorized_setwise_risk_bound=True,
            use_setwise_safety_slack_pairwise_order=True,
            parent_transition_candidate_top_k=5,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        parent_transition_break_cost=4.0,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    assert torch.equal(outputs["scores"], baseline)
    torch.testing.assert_close(
        outputs["setwise_safety_slack_pairwise_order"], torch.tensor(1.0)
    )
    box = torch.tensor([[0.60, 0.20, 0.55, 0.40, 0.70]])
    mask = torch.tensor([[0.40, 0.10, 0.30, 0.20, 0.50]])
    result = compute_joint_query_quality_loss(
        outputs, box, mask,
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=1.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    absolute_point = (0.95 + 0.325 + 0.50) / 3.0
    absolute_guard = (1.10 + 0.925 + 1.10) / 3.0
    pair_gap = (1.40 + 0.65 + 1.00) / 3.0
    stats = result["stats"]
    torch.testing.assert_close(
        stats["setwise_factorized_risk_bound_point_loss"],
        torch.tensor(absolute_point + pair_gap),
    )
    torch.testing.assert_close(
        stats["setwise_factorized_risk_bound_guard_loss"],
        torch.tensor(absolute_guard + pair_gap),
    )
    torch.testing.assert_close(
        stats["setwise_safety_slack_box025_point_pair_mae"],
        torch.tensor(1.40),
    )
    torch.testing.assert_close(
        stats["setwise_safety_slack_box050_guard_pair_mae"],
        torch.tensor(0.65),
    )
    torch.testing.assert_close(
        stats["setwise_safety_slack_mask025_point_pair_order_accuracy"],
        torch.tensor(0.0),
    )
    result["loss"].backward()
    final = model.setwise_safety_head[-1]
    assert final.weight.grad is not None
    assert bool(torch.isfinite(final.weight.grad).all().item())
    assert int(torch.count_nonzero(final.weight.grad).item()) > 0


def test_v78_proposal_conditioned_safety_calibrates_only_the_proposal():
    torch.manual_seed(780)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47, 0.46]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires safety-slack pairwise order"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            max_delta=0.25, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decoupled_setwise_heads=True,
            use_factorized_setwise_safety=True,
            use_factorized_setwise_risk_bound=True,
            use_setwise_safety_veto_gate=True,
            use_setwise_safety_slack_quantile_bound=True,
            use_proposal_conditioned_safety=True,
            parent_transition_candidate_top_k=5,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_proposal_conditioned_safety=True,
        parent_transition_break_cost=4.0,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    assert torch.equal(outputs["scores"], baseline)
    assert outputs["setwise_proposal_indices"].tolist() == [1]
    assert outputs["setwise_proposal_mask"].sum().item() == 1
    assert not bool(outputs["setwise_proposal_promotable_mask"].any().item())
    torch.testing.assert_close(
        outputs["setwise_proposal_conditioned_safety"], torch.tensor(1.0)
    )
    box = torch.tensor([[0.60, 0.20, 0.55, 0.40, 0.70]])
    mask = torch.tensor([[0.40, 0.10, 0.30, 0.20, 0.50]])
    result = compute_joint_query_quality_loss(
        outputs, box, mask,
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=1.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    pair_gap = (1.40 + 0.65 + 1.00) / 3.0
    proposal_point = (0.20 + 0.60 + 0.60) / 3.0
    proposal_guard = 4.0 * proposal_point
    stats = result["stats"]
    torch.testing.assert_close(
        stats["setwise_factorized_risk_bound_point_loss"],
        torch.tensor(proposal_point + pair_gap),
    )
    torch.testing.assert_close(
        stats["setwise_factorized_risk_bound_guard_loss"],
        torch.tensor(proposal_guard + pair_gap),
    )
    torch.testing.assert_close(
        stats["setwise_proposal_hazard_ratio"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        stats["setwise_proposal_promotable_ratio"], torch.tensor(0.0)
    )
    result["loss"].backward()
    final = model.setwise_safety_head[-1]
    assert final.weight.grad is not None
    assert int(torch.count_nonzero(final.weight.grad).item()) > 0


def test_v78_strict_propose_then_verify_has_no_safe_fallback():
    class FixedPromotion(torch.nn.Module):
        def forward(self, pair_features):
            values = pair_features.new_tensor((0.0, 2.0, 1.5, 0.0))
            return values.view(1, 4, 1).expand(
                pair_features.shape[0], -1, -1
            )

    class FixedSafety(torch.nn.Module):
        def __init__(self, proposal_safe):
            super().__init__()
            self.proposal_safe = proposal_safe

        def forward(self, pair_features):
            values = pair_features.new_ones((1, 4, 6))
            values[:, 1] = 1.0 if self.proposal_safe else -1.0
            return values.expand(pair_features.shape[0], -1, -1)

    torch.manual_seed(781)
    features = torch.randn(1, 4, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_proposal_conditioned_safety=True,
        parent_transition_candidate_top_k=4,
    )
    model.setwise_promotion_head = FixedPromotion()
    model.setwise_safety_head = FixedSafety(proposal_safe=False)
    rejected = model(features, baseline, valid)
    assert rejected["setwise_proposal_indices"].tolist() == [1]
    assert rejected["selected_indices"].tolist() == [0]
    assert rejected["residual"][0, 2].item() == 0.0

    model.setwise_safety_head = FixedSafety(proposal_safe=True)
    accepted = model(features, baseline, valid)
    assert accepted["setwise_proposal_indices"].tolist() == [1]
    assert accepted["selected_indices"].tolist() == [1]
    assert accepted["residual"][0, 2].item() == 0.0


def test_v79_parent_reference_cancels_row_bias_and_preserves_candidate_gaps():
    class AffineSafety(torch.nn.Module):
        def __init__(self, row_bias):
            super().__init__()
            self.row_bias = row_bias

        def forward(self, pair_features):
            candidate = pair_features[..., 0:1]
            offsets = candidate.new_tensor((1., 2., 3., 4., 5., 6.))
            return self.row_bias + candidate + offsets

    torch.manual_seed(790)
    features = torch.randn(2, 5, 8)
    baseline = torch.tensor([
        [0.50, 0.49, 0.48, 0.47, 0.46],
        [0.49, 0.50, 0.48, 0.47, 0.46],
    ])
    valid = torch.ones(2, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires safety-slack pairwise order"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            max_delta=0.25, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decoupled_setwise_heads=True,
            use_factorized_setwise_safety=True,
            use_factorized_setwise_risk_bound=True,
            use_setwise_safety_veto_gate=True,
            use_setwise_safety_slack_quantile_bound=True,
            use_parent_referenced_safety=True,
            parent_transition_candidate_top_k=5,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            max_delta=0.25, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decoupled_setwise_heads=True,
            use_factorized_setwise_safety=True,
            use_factorized_setwise_risk_bound=True,
            use_setwise_safety_veto_gate=True,
            use_setwise_safety_slack_quantile_bound=True,
            use_setwise_safety_slack_pairwise_order=True,
            use_proposal_conditioned_safety=True,
            use_parent_referenced_safety=True,
            parent_transition_candidate_top_k=5,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_parent_referenced_safety=True,
        parent_transition_candidate_top_k=5,
    )
    model.setwise_safety_head = AffineSafety(row_bias=100.0)
    biased = model(features, baseline, valid)
    model.setwise_safety_head = AffineSafety(row_bias=-37.0)
    shifted = model(features, baseline, valid)
    torch.testing.assert_close(
        biased["setwise_safety_bound_scores"],
        shifted["setwise_safety_bound_scores"],
    )
    anchors = biased["baseline_indices"]
    row = torch.arange(2)
    torch.testing.assert_close(
        biased["setwise_safety_bound_scores"][row, anchors],
        torch.zeros(2, 3, 2),
    )
    torch.testing.assert_close(
        biased["setwise_parent_referenced_safety"], torch.tensor(1.0)
    )
    assert torch.equal(biased["scores"], baseline)


def test_v80_coupled_witness_requires_one_candidate_to_satisfy_both_heads():
    torch.manual_seed(800)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.50, 0.50, 0.50, 0.50]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires parent-referenced safety"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            max_delta=0.25, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decoupled_setwise_heads=True,
            use_factorized_setwise_safety=True,
            use_factorized_setwise_risk_bound=True,
            use_setwise_safety_veto_gate=True,
            use_setwise_safety_slack_quantile_bound=True,
            use_setwise_safety_slack_pairwise_order=True,
            use_coupled_safe_repair_witness=True,
            parent_transition_candidate_top_k=5,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_parent_referenced_safety=True,
        use_coupled_safe_repair_witness=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    box = torch.tensor([[0.20, 0.10, 0.55, 0.40, 0.70]])
    mask = torch.tensor([[0.40, 0.10, 0.30, 0.20, 0.50]])
    result = compute_joint_query_quality_loss(
        outputs, box, mask,
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=0.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    stats = result["stats"]
    torch.testing.assert_close(
        stats["setwise_coupled_safe_repair_witness"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        stats["setwise_coupled_safe_repair_witness_margin"],
        torch.tensor(0.0),
    )
    assert stats["setwise_coupled_safe_repair_witness_loss"].item() > 0.0
    result["loss"].backward()
    promotion = model.setwise_promotion_head[-1]
    safety = model.setwise_safety_head[-1]
    assert promotion.weight.grad is not None
    assert safety.weight.grad is not None
    assert int(torch.count_nonzero(promotion.weight.grad).item()) > 0
    assert int(torch.count_nonzero(safety.weight.grad).item()) > 0


def test_v81_bidirectional_boundary_penalizes_hardest_nonrepair_joint_score():
    torch.manual_seed(810)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.50, 0.50, 0.50, 0.50]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires coupled safe-repair witness"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            max_delta=0.25, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decoupled_setwise_heads=True,
            use_factorized_setwise_safety=True,
            use_factorized_setwise_risk_bound=True,
            use_setwise_safety_veto_gate=True,
            use_setwise_safety_slack_quantile_bound=True,
            use_setwise_safety_slack_pairwise_order=True,
            use_parent_referenced_safety=True,
            use_bidirectional_coupled_boundary=True,
            parent_transition_candidate_top_k=5,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_parent_referenced_safety=True,
        use_coupled_safe_repair_witness=True,
        use_bidirectional_coupled_boundary=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    box = torch.tensor([[0.20, 0.10, 0.55, 0.40, 0.70]])
    mask = torch.tensor([[0.40, 0.10, 0.30, 0.20, 0.50]])
    result = compute_joint_query_quality_loss(
        outputs, box, mask,
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=0.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    stats = result["stats"]
    torch.testing.assert_close(
        stats["setwise_bidirectional_coupled_boundary"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        stats["setwise_bidirectional_coupled_negative_margin"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        stats["setwise_bidirectional_coupled_negative_violation_ratio"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        stats["setwise_bidirectional_coupled_separation_margin"],
        torch.tensor(0.0),
    )
    assert stats["setwise_bidirectional_coupled_boundary_loss"].item() > 0.0
    result["loss"].backward()
    assert int(torch.count_nonzero(
        model.setwise_promotion_head[-1].weight.grad
    ).item()) > 0
    assert int(torch.count_nonzero(
        model.setwise_safety_head[-1].weight.grad
    ).item()) > 0


def test_v82_centered_separation_uses_pair_gap_and_zero_midpoint():
    torch.manual_seed(820)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.50, 0.50, 0.50, 0.50]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires bidirectional"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            max_delta=0.25, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decoupled_setwise_heads=True,
            use_factorized_setwise_safety=True,
            use_factorized_setwise_risk_bound=True,
            use_setwise_safety_veto_gate=True,
            use_setwise_safety_slack_quantile_bound=True,
            use_setwise_safety_slack_pairwise_order=True,
            use_parent_referenced_safety=True,
            use_coupled_safe_repair_witness=True,
            use_centered_coupled_separation=True,
            parent_transition_candidate_top_k=5,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_parent_referenced_safety=True,
        use_coupled_safe_repair_witness=True,
        use_bidirectional_coupled_boundary=True,
        use_centered_coupled_separation=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    result = compute_joint_query_quality_loss(
        outputs,
        torch.tensor([[0.20, 0.10, 0.55, 0.40, 0.70]]),
        torch.tensor([[0.40, 0.10, 0.30, 0.20, 0.50]]),
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=0.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    stats = result["stats"]
    torch.testing.assert_close(
        stats["setwise_centered_coupled_separation"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        stats["setwise_centered_coupled_midpoint_abs"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        stats["setwise_centered_coupled_margin_recall"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        stats["setwise_coupled_safe_repair_witness_loss"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        stats["setwise_bidirectional_coupled_boundary_loss"],
        torch.tensor(0.0),
    )
    assert stats["setwise_centered_coupled_separation_loss"].item() > 0.0
    result["loss"].backward()
    assert int(torch.count_nonzero(
        model.setwise_promotion_head[-1].weight.grad
    ).item()) > 0
    assert int(torch.count_nonzero(
        model.setwise_safety_head[-1].weight.grad
    ).item()) > 0


def test_v83_hazard_conditioning_ignores_safe_neutral_candidates():
    torch.manual_seed(830)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.50, 0.50, 0.50, 0.50]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires centered"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            max_delta=0.25, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decoupled_setwise_heads=True,
            use_factorized_setwise_safety=True,
            use_factorized_setwise_risk_bound=True,
            use_setwise_safety_veto_gate=True,
            use_setwise_safety_slack_quantile_bound=True,
            use_setwise_safety_slack_pairwise_order=True,
            use_parent_referenced_safety=True,
            use_coupled_safe_repair_witness=True,
            use_bidirectional_coupled_boundary=True,
            use_hazard_conditioned_coupled_separation=True,
            parent_transition_candidate_top_k=5,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_parent_referenced_safety=True,
        use_coupled_safe_repair_witness=True,
        use_bidirectional_coupled_boundary=True,
        use_centered_coupled_separation=True,
        use_hazard_conditioned_coupled_separation=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    # Parent q0 is wrong. q2/q4 are safe repairs, while q1/q3 are merely safe
    # non-repairs. There is no exact regression hazard in this row, so V83
    # must use only the unpaired positive witness rather than inventing a
    # negative from q1/q3.
    result = compute_joint_query_quality_loss(
        outputs,
        torch.tensor([[0.20, 0.10, 0.55, 0.20, 0.70]]),
        torch.tensor([[0.40, 0.30, 0.30, 0.30, 0.50]]),
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=0.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    stats = result["stats"]
    torch.testing.assert_close(
        stats["setwise_hazard_conditioned_coupled_separation"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        stats["setwise_hazard_conditioned_coupled_pair_row_ratio"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        stats[
            "setwise_hazard_conditioned_coupled_unpaired_positive_row_ratio"
        ],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        stats["setwise_hazard_conditioned_coupled_negative_ratio"],
        torch.tensor(0.0),
    )
    assert stats["setwise_centered_coupled_separation_loss"].item() > 0.0


def test_v83_hazard_conditioning_pairs_repair_with_exact_regression():
    torch.manual_seed(831)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.50, 0.50, 0.50, 0.50]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_parent_referenced_safety=True,
        use_coupled_safe_repair_witness=True,
        use_bidirectional_coupled_boundary=True,
        use_centered_coupled_separation=True,
        use_hazard_conditioned_coupled_separation=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    # Parent q0 is correct at Box@.25 and Mask@.25. q2 improves tier;
    # q1 breaks both protected criteria; q3 is safe neutral. Only q1 is a V83
    # negative, so the paired-row and hazard-candidate ratios are exact.
    result = compute_joint_query_quality_loss(
        outputs,
        torch.tensor([[0.40, 0.10, 0.70, 0.40, 0.55]]),
        torch.tensor([[0.40, 0.10, 0.50, 0.30, 0.45]]),
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=0.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    stats = result["stats"]
    torch.testing.assert_close(
        stats["setwise_hazard_conditioned_coupled_pair_row_ratio"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        stats[
            "setwise_hazard_conditioned_coupled_unpaired_positive_row_ratio"
        ],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        stats["setwise_hazard_conditioned_coupled_negative_ratio"],
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        stats["setwise_centered_coupled_midpoint_abs"], torch.tensor(0.0)
    )
    assert stats["setwise_centered_coupled_separation_loss"].item() > 0.0


def test_v84_folds_box_guards_but_keeps_mask025_hard_veto():
    class FixedPromotion(torch.nn.Module):
        def forward(self, pair_features):
            values = pair_features.new_tensor((0.0, 2.0, -1.0, 0.0))
            return values.view(1, 4, 1).expand(
                pair_features.shape[0], -1, -1
            )

    class FixedBounds(torch.nn.Module):
        def __init__(self, mask_safe):
            super().__init__()
            self.mask_safe = mask_safe

        def forward(self, pair_features):
            # q1 has deliberately negative Box@.25/.50 guard values. V84
            # folds these monotonic box criteria into promotion and must not
            # let them veto a promoted candidate. Mask@.25 remains an
            # independent point/guard hard veto.
            values = pair_features.new_ones((1, 4, 6))
            values[:, 0] = 0.0
            values[:, 1, 1] = -1.0
            values[:, 1, 3] = -1.0
            values[:, 1, 4:] = 1.0 if self.mask_safe else -1.0
            return values.expand(pair_features.shape[0], -1, -1)

    torch.manual_seed(840)
    features = torch.randn(1, 4, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires hazard-conditioned"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            max_delta=0.25, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decoupled_setwise_heads=True,
            use_factorized_setwise_safety=True,
            use_factorized_setwise_risk_bound=True,
            use_setwise_safety_veto_gate=True,
            use_setwise_safety_slack_quantile_bound=True,
            use_setwise_safety_slack_pairwise_order=True,
            use_parent_referenced_safety=True,
            use_coupled_safe_repair_witness=True,
            use_bidirectional_coupled_boundary=True,
            use_centered_coupled_separation=True,
            use_monotonic_box_safety_folding=True,
            parent_transition_candidate_top_k=4,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_parent_referenced_safety=True,
        use_coupled_safe_repair_witness=True,
        use_bidirectional_coupled_boundary=True,
        use_centered_coupled_separation=True,
        use_hazard_conditioned_coupled_separation=True,
        use_monotonic_box_safety_folding=True,
        parent_transition_candidate_top_k=4,
    )
    model.setwise_promotion_head = FixedPromotion()
    model.setwise_safety_head = FixedBounds(mask_safe=True)
    accepted = model(features, baseline, valid)
    assert accepted["selected_indices"].tolist() == [1]
    assert accepted["setwise_safety_bound_scores"][0, 1, 0, 1] < 0.0
    assert accepted["setwise_safety_bound_scores"][0, 1, 1, 1] < 0.0
    assert accepted["setwise_tier_branch_scores"][0, 1, 1] > 0.0
    torch.testing.assert_close(
        accepted["setwise_monotonic_box_safety_folding"], torch.tensor(1.0)
    )

    model.setwise_safety_head = FixedBounds(mask_safe=False)
    rejected = model(features, baseline, valid)
    assert rejected["selected_indices"].tolist() == [0]
    assert rejected["setwise_tier_branch_scores"][0, 1, 1] < 0.0


def test_v85_branchwise_witness_trains_both_branches_of_one_repair():
    torch.manual_seed(850)
    features = torch.randn(1, 5, 8)
    baseline = torch.tensor([[0.50, 0.50, 0.50, 0.50, 0.50]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires monotonic"):
        JointQueryQualityReranker(
            8, hidden_dim=16, num_heads=4, dropout=0.0,
            max_delta=0.25, preserve_parent_score=True,
            use_setwise_tier_advantage=True,
            use_decoupled_setwise_heads=True,
            use_factorized_setwise_safety=True,
            use_factorized_setwise_risk_bound=True,
            use_setwise_safety_veto_gate=True,
            use_setwise_safety_slack_quantile_bound=True,
            use_setwise_safety_slack_pairwise_order=True,
            use_parent_referenced_safety=True,
            use_coupled_safe_repair_witness=True,
            use_bidirectional_coupled_boundary=True,
            use_centered_coupled_separation=True,
            use_hazard_conditioned_coupled_separation=True,
            use_same_candidate_branchwise_witness=True,
            parent_transition_candidate_top_k=5,
        )
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_parent_referenced_safety=True,
        use_coupled_safe_repair_witness=True,
        use_bidirectional_coupled_boundary=True,
        use_centered_coupled_separation=True,
        use_hazard_conditioned_coupled_separation=True,
        use_monotonic_box_safety_folding=True,
        use_same_candidate_branchwise_witness=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    result = compute_joint_query_quality_loss(
        outputs,
        torch.tensor([[0.20, 0.10, 0.55, 0.20, 0.70]]),
        torch.tensor([[0.40, 0.30, 0.30, 0.30, 0.50]]),
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=0.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    stats = result["stats"]
    torch.testing.assert_close(
        stats["setwise_same_candidate_branchwise_witness"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        stats["setwise_same_candidate_branchwise_promotion_margin"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        stats["setwise_same_candidate_branchwise_mask_safety_margin"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        stats["setwise_same_candidate_branchwise_recall"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        stats["setwise_centered_coupled_separation_loss"],
        torch.tensor(0.0),
    )
    assert stats["setwise_same_candidate_branchwise_witness_loss"].item() > 0
    result["loss"].backward()
    promotion = model.setwise_promotion_head[-1]
    safety = model.setwise_safety_head[-1]
    assert int(torch.count_nonzero(promotion.weight.grad).item()) > 0
    assert int(torch.count_nonzero(safety.weight.grad).item()) > 0


def test_v86_parent_certificate_vetoes_box_break_but_accepts_certified_repair():
    class FixedPromotion(torch.nn.Module):
        def forward(self, pair_features):
            values = pair_features.new_tensor((0.0, 2.0, 1.5, 0.0))
            return values.view(1, 4, 1).expand(
                pair_features.shape[0], -1, -1
            )

    class FixedBounds(torch.nn.Module):
        def forward(self, pair_features):
            # q1 has a Box@.50 point certificate below the parent boundary,
            # but all conservative guards and Mask@.25 are positive. q2 is
            # certified by both Box point slacks and the Mask point+guard.
            values = pair_features.new_ones((1, 4, 6))
            values[:, 0] = 0.0
            values[:, 1, 2] = -1.0
            return values.expand(pair_features.shape[0], -1, -1)

    torch.manual_seed(860)
    features = torch.randn(1, 4, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    common = dict(
        hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_parent_referenced_safety=True,
        use_coupled_safe_repair_witness=True,
        use_bidirectional_coupled_boundary=True,
        use_centered_coupled_separation=True,
        use_hazard_conditioned_coupled_separation=True,
        use_monotonic_box_safety_folding=True,
        use_same_candidate_branchwise_witness=True,
        parent_transition_candidate_top_k=4,
    )
    with pytest.raises(ValueError, match="requires same-candidate"):
        JointQueryQualityReranker(
            8,
            use_parent_non_degradation_certificate=True,
            use_same_candidate_branchwise_witness=False,
            **{key: value for key, value in common.items()
               if key != "use_same_candidate_branchwise_witness"},
        )
    model = JointQueryQualityReranker(
        8, use_parent_non_degradation_certificate=True, **common
    )
    model.setwise_promotion_head = FixedPromotion()
    model.setwise_safety_head = FixedBounds()
    outputs = model(features, baseline, valid)
    assert outputs["selected_indices"].tolist() == [2]
    assert outputs["setwise_safety_bound_scores"][0, 1, 1, 0] < 0.0
    assert outputs["setwise_tier_branch_scores"][0, 1, 1] < 0.0
    assert outputs["setwise_tier_branch_scores"][0, 2, 1] > 0.0
    torch.testing.assert_close(
        outputs["setwise_parent_non_degradation_certificate"],
        torch.tensor(1.0),
    )


def test_v86_parent_certificate_trains_all_same_candidate_certificates():
    torch.manual_seed(861)
    features = torch.randn(1, 5, 8)
    baseline = torch.full((1, 5), 0.5)
    valid = torch.ones(1, 5, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_parent_referenced_safety=True,
        use_coupled_safe_repair_witness=True,
        use_bidirectional_coupled_boundary=True,
        use_centered_coupled_separation=True,
        use_hazard_conditioned_coupled_separation=True,
        use_monotonic_box_safety_folding=True,
        use_same_candidate_branchwise_witness=True,
        use_parent_non_degradation_certificate=True,
        parent_transition_candidate_top_k=5,
    )
    outputs = model(features, baseline, valid)
    result = compute_joint_query_quality_loss(
        outputs,
        torch.tensor([[0.20, 0.10, 0.55, 0.20, 0.70]]),
        torch.tensor([[0.40, 0.30, 0.30, 0.30, 0.50]]),
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=0.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    stats = result["stats"]
    torch.testing.assert_close(
        stats["setwise_parent_non_degradation_certificate"],
        torch.tensor(1.0),
    )
    for key in (
        "setwise_parent_non_degradation_box025_margin",
        "setwise_parent_non_degradation_box050_margin",
        "setwise_parent_non_degradation_mask025_margin",
        "setwise_parent_non_degradation_recall",
    ):
        torch.testing.assert_close(stats[key], torch.tensor(0.0))
    result["loss"].backward()
    promotion = model.setwise_promotion_head[-1].weight.grad
    safety = model.setwise_safety_head[-1].weight.grad.view(6, -1)
    assert int(torch.count_nonzero(promotion).item()) > 0
    # Box point certificates 0/2 and the active Mask point/guard bottleneck
    # receive the witness gradient; Box guard channels 1/3 do not.
    assert int(torch.count_nonzero(safety[[0, 2]]).item()) > 0
    assert int(torch.count_nonzero(safety[[4, 5]]).item()) > 0
    torch.testing.assert_close(safety[[1, 3]], torch.zeros_like(safety[[1, 3]]))


def test_v87_hazard_gradient_is_attributed_only_to_responsible_certificate():
    torch.manual_seed(870)
    features = torch.randn(1, 5, 8)
    baseline = torch.full((1, 5), 0.5)
    valid = torch.ones(1, 5, dtype=torch.bool)
    common = dict(
        hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_parent_referenced_safety=True,
        use_coupled_safe_repair_witness=True,
        use_bidirectional_coupled_boundary=True,
        use_centered_coupled_separation=True,
        use_hazard_conditioned_coupled_separation=True,
        use_monotonic_box_safety_folding=True,
        use_same_candidate_branchwise_witness=True,
        use_parent_non_degradation_certificate=True,
        parent_transition_candidate_top_k=5,
    )
    with pytest.raises(ValueError, match="requires parent"):
        JointQueryQualityReranker(
            8,
            use_criterion_responsible_hazard_attribution=True,
            use_parent_non_degradation_certificate=False,
            **{key: value for key, value in common.items()
               if key != "use_parent_non_degradation_certificate"},
        )
    model = JointQueryQualityReranker(
        8, use_criterion_responsible_hazard_attribution=True, **common
    )
    outputs = model(features, baseline, valid)
    # Parent q0 is Box@.50-correct but Mask@.25-incorrect. q1 is the only
    # Box@.50 hazard; there are no repair rows, so only safety negatives act.
    result = compute_joint_query_quality_loss(
        outputs,
        torch.tensor([[0.70, 0.40, 0.70, 0.60, 0.55]]),
        torch.full((1, 5), 0.10),
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=0.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    stats = result["stats"]
    torch.testing.assert_close(
        stats["setwise_criterion_responsible_hazard_attribution"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        stats["setwise_criterion_responsible_box050_margin"],
        torch.tensor(0.0),
    )
    result["loss"].backward()
    promotion = model.setwise_promotion_head[-1].weight.grad
    safety = model.setwise_safety_head[-1].weight.grad.view(6, -1)
    # Promotion receives only the independent stay-row fallback term; the
    # Box@.50 hazard itself is attributed solely to channel 2.
    assert int(torch.count_nonzero(promotion).item()) > 0
    assert int(torch.count_nonzero(safety[2]).item()) > 0
    torch.testing.assert_close(
        safety[[0, 1, 3, 4, 5]],
        torch.zeros_like(safety[[0, 1, 3, 4, 5]]),
    )


def _v88_common_kwargs():
    return dict(
        hidden_dim=16, num_heads=4, dropout=0.0,
        max_delta=0.25, preserve_parent_score=True,
        use_setwise_tier_advantage=True,
        use_decoupled_setwise_heads=True,
        use_factorized_setwise_safety=True,
        use_factorized_setwise_risk_bound=True,
        use_setwise_safety_veto_gate=True,
        use_setwise_safety_slack_quantile_bound=True,
        use_setwise_safety_slack_pairwise_order=True,
        use_parent_referenced_safety=True,
        use_coupled_safe_repair_witness=True,
        use_bidirectional_coupled_boundary=True,
        use_centered_coupled_separation=True,
        use_hazard_conditioned_coupled_separation=True,
        use_monotonic_box_safety_folding=True,
        use_same_candidate_branchwise_witness=True,
        use_parent_non_degradation_certificate=True,
        use_independent_joint_hazard_certificate=True,
        parent_transition_candidate_top_k=4,
    )


def test_v88_independent_joint_hazard_contract_and_zero_identity():
    common = _v88_common_kwargs()
    with pytest.raises(ValueError, match="requires parent"):
        JointQueryQualityReranker(
            8,
            use_parent_non_degradation_certificate=False,
            **{key: value for key, value in common.items() if key not in (
                "use_parent_non_degradation_certificate",
            )},
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        JointQueryQualityReranker(
            8, use_criterion_responsible_hazard_attribution=True, **common
        )
    torch.manual_seed(880)
    features = torch.randn(1, 4, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47]])
    outputs = JointQueryQualityReranker(8, **common)(
        features, baseline, torch.ones(1, 4, dtype=torch.bool)
    )
    assert torch.equal(outputs["scores"], baseline)
    assert outputs["selected_indices"].tolist() == [0]
    torch.testing.assert_close(
        outputs["setwise_independent_joint_hazard_scores"],
        torch.zeros_like(
            outputs["setwise_independent_joint_hazard_scores"]
        ),
    )


def test_v88_independent_joint_hazard_is_a_fifth_deployment_veto():
    class FixedPromotion(torch.nn.Module):
        def forward(self, pair_features):
            values = pair_features.new_tensor((0.0, 2.0, 1.5, 0.0))
            return values.view(1, 4, 1).expand(
                pair_features.shape[0], -1, -1
            )

    class FixedBounds(torch.nn.Module):
        def forward(self, pair_features):
            values = pair_features.new_ones((1, 4, 6))
            values[:, 0] = 0.0
            return values.expand(pair_features.shape[0], -1, -1)

    class FixedJointHazard(torch.nn.Module):
        def __init__(self, first_safe):
            super().__init__()
            self.first_safe = first_safe

        def forward(self, pair_features):
            values = pair_features.new_tensor((
                0.0, 1.0 if self.first_safe else -1.0, 1.0, 1.0,
            ))
            return values.view(1, 4, 1).expand(
                pair_features.shape[0], -1, -1
            )

    torch.manual_seed(881)
    features = torch.randn(1, 4, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    model = JointQueryQualityReranker(8, **_v88_common_kwargs())
    model.setwise_promotion_head = FixedPromotion()
    model.setwise_safety_head = FixedBounds()
    model.independent_joint_hazard_head = FixedJointHazard(first_safe=False)
    vetoed = model(features, baseline, valid)
    assert vetoed["selected_indices"].tolist() == [2]
    assert vetoed["setwise_tier_branch_scores"][0, 1, 1] < 0.0
    model.independent_joint_hazard_head = FixedJointHazard(first_safe=True)
    accepted = model(features, baseline, valid)
    assert accepted["selected_indices"].tolist() == [1]
    assert accepted["setwise_tier_branch_scores"][0, 1, 1] > 0.0


def test_v88_joint_hazard_score_gradient_is_isolated_from_shared_features():
    torch.manual_seed(882)
    features = torch.randn(1, 4, 8, requires_grad=True)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47]])
    model = JointQueryQualityReranker(
        8, detach_inputs=False, **_v88_common_kwargs()
    )
    outputs = model(features, baseline, torch.ones(1, 4, dtype=torch.bool))
    outputs["setwise_independent_joint_hazard_scores"].sum().backward()
    assert features.grad is None
    final_gradient = model.independent_joint_hazard_head[-1].weight.grad
    assert final_gradient is not None
    assert int(torch.count_nonzero(final_gradient).item()) > 0
    for name, parameter in model.named_parameters():
        if not name.startswith("independent_joint_hazard_head."):
            assert parameter.grad is None, name


def test_v88_joint_hazard_regresses_continuous_minimum_safety_slack():
    torch.manual_seed(883)
    features = torch.randn(1, 4, 8)
    baseline = torch.full((1, 4), 0.5)
    valid = torch.ones(1, 4, dtype=torch.bool)
    model = JointQueryQualityReranker(8, **_v88_common_kwargs())
    outputs = model(features, baseline, valid)
    result = compute_joint_query_quality_loss(
        outputs,
        torch.tensor([[0.60, 0.70, 0.40, 0.70]]),
        torch.tensor([[0.40, 0.50, 0.50, 0.10]]),
        listwise_loss_weight=0.0, transition_loss_weight=1.0,
        setwise_rank_loss_weight=0.0,
        setwise_dense_safety_loss_weight=0.0,
        setwise_balanced_safety_loss_weight=0.0,
        setwise_factorized_safety_loss_weight=0.0,
        setwise_factorized_risk_bound_loss_weight=0.0,
        quality_loss_weight=0.0, anchor_loss_weight=0.0,
        source_mix_loss_weight=0.0, transition_break_cost=4.0,
    )
    stats = result["stats"]
    torch.testing.assert_close(
        stats["setwise_independent_joint_hazard_certificate"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        stats["setwise_independent_joint_hazard_target_negative_ratio"],
        torch.tensor(2.0 / 3.0),
    )
    torch.testing.assert_close(
        stats["setwise_independent_joint_hazard_loss"],
        torch.tensor((0.4 + 0.8 + 2.4) / 3.0),
    )
    torch.testing.assert_close(
        stats["setwise_independent_joint_hazard_quantile_coverage"],
        torch.tensor(2.0 / 3.0),
    )


def test_v89_frozen_raw_hazard_features_ignore_shared_hidden_drift():
    common = _v88_common_kwargs()
    with pytest.raises(ValueError, match="require independent"):
        JointQueryQualityReranker(
            8,
            use_frozen_raw_joint_hazard_features=True,
            use_independent_joint_hazard_certificate=False,
            **{key: value for key, value in common.items() if key not in (
                "use_independent_joint_hazard_certificate",
            )},
        )
    torch.manual_seed(890)
    features = torch.randn(1, 4, 8)
    baseline = torch.tensor([[0.50, 0.49, 0.48, 0.47]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    model = JointQueryQualityReranker(
        8, use_frozen_raw_joint_hazard_features=True, **common
    ).eval()
    torch.nn.init.normal_(model.independent_joint_hazard_head[-1].weight)
    before = model(features, baseline, valid)[
        "setwise_independent_joint_hazard_scores"
    ]
    with torch.no_grad():
        for parameter in model.input_projection.parameters():
            parameter.fill_(100.0)
        for parameter in model.attention.parameters():
            parameter.fill_(-50.0)
    after = model(features, baseline, valid)[
        "setwise_independent_joint_hazard_scores"
    ]
    torch.testing.assert_close(after, before)
    torch.testing.assert_close(
        model(features, baseline, valid)[
            "setwise_frozen_raw_joint_hazard_features"
        ],
        torch.tensor(1.0),
    )
