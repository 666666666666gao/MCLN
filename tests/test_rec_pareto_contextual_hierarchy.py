import pytest
import torch

from models.rec_pareto_contextual_hierarchy import (
    AsymmetricRiskContextualHierarchyCommittee,
    ParetoContextualHierarchicalReranker,
    V99_AGGREGATE_MARGIN,
    apply_asymmetric_risk_contextual_policy,
    apply_pareto_contextual_policy,
)
from scripts.run_v97_contextual_listwise_hierarchical import (
    ContextualHierarchicalQueryVariantReranker,
)
from scripts.run_v109_meshsp_nested_policy_oof import policy_accept_mask


def _inputs(proposal_logits):
    base = torch.full((1, 112), -float("inf"), dtype=torch.float32)
    base[0, :2] = torch.tensor([1.0, 0.0])
    query_logits = torch.full((1, 16, 2), -20.0, dtype=torch.float32)
    variant_logits = torch.full((1, 16, 7, 2), -20.0, dtype=torch.float32)
    query_logits[0, 0] = torch.tensor([8.0, 8.0])
    variant_logits[0, 0, 0] = torch.tensor([0.0, 0.0])
    variant_logits[0, 0, 1] = torch.tensor(proposal_logits)
    query_valid = torch.zeros(1, 16, dtype=torch.bool)
    variant_valid = torch.zeros(1, 16, 7, dtype=torch.bool)
    query_valid[0, 0] = True
    variant_valid[0, 0, :2] = True
    return base, query_logits, variant_logits, query_valid, variant_valid


def test_pareto_policy_promotes_when_both_heads_improve():
    values = _inputs([3.0, 3.0])
    result = apply_pareto_contextual_policy(
        *values, aggregate_margin=V99_AGGREGATE_MARGIN
    )
    assert result["selected_indices"].tolist() == [1]
    assert result["pareto_pass"].tolist() == [True]


def test_pareto_policy_vetoes_when_second_threshold_degrades():
    values = _inputs([3.0, -12.0])
    result = apply_pareto_contextual_policy(
        *values, aggregate_margin=V99_AGGREGATE_MARGIN
    )
    assert result["selected_indices"].tolist() == [0]
    assert result["pareto_pass"].tolist() == [False]


def test_pareto_policy_rejects_nonpositive_margin():
    with pytest.raises(ValueError, match="positive"):
        apply_pareto_contextual_policy(*_inputs([3.0, 3.0]), aggregate_margin=0.0)


def test_v109_runtime_gate_matches_frozen_oof_policy():
    generator = torch.Generator().manual_seed(109)
    batch_size = 17
    base_scores = torch.randn(
        batch_size, 112, generator=generator, dtype=torch.float32
    )
    query_logits = torch.randn(
        batch_size, 16, 2, generator=generator, dtype=torch.float32
    )
    variant_logits = torch.randn(
        batch_size, 16, 7, 2, generator=generator, dtype=torch.float32
    )
    query_valid = torch.ones(batch_size, 16, dtype=torch.bool)
    variant_valid = torch.ones(batch_size, 16, 7, dtype=torch.bool)
    policy = {
        "aggregate_margin": 0.15,
        "min_head_gain025": 0.02,
        "min_head_gain050": 0.0,
    }

    result = apply_pareto_contextual_policy(
        base_scores, query_logits, variant_logits,
        query_valid, variant_valid,
        aggregate_margin=policy["aggregate_margin"],
        min_head_gain025=policy["min_head_gain025"],
        min_head_gain050=policy["min_head_gain050"],
    )
    expected = policy_accept_mask(
        result["proposal_indices"], result["baseline_indices"],
        result["aggregate_gain"], result["head_gain"], policy,
    )

    assert torch.equal(result["switch_mask"], expected)


def test_canonical_deployment_model_matches_v97_state_contract():
    deployed = ParetoContextualHierarchicalReranker()
    experimental = ContextualHierarchicalQueryVariantReranker(128, 0.1)
    assert {
        name: (value.dtype, tuple(value.shape))
        for name, value in deployed.state_dict().items()
    } == {
        name: (value.dtype, tuple(value.shape))
        for name, value in experimental.state_dict().items()
    }


def test_v113_zero_risk_penalties_reduce_to_anchor_policy():
    base, query, variant, query_valid, variant_valid = _inputs([3.0, 3.0])
    result = apply_asymmetric_risk_contextual_policy(
        base,
        query.unsqueeze(0).repeat(3, 1, 1, 1),
        variant.unsqueeze(0).repeat(3, 1, 1, 1, 1),
        query_valid,
        variant_valid,
        aggregate_lcb_margin=V99_AGGREGATE_MARGIN,
        risk_lambda025=0.0,
        risk_lambda050=0.0,
    )

    assert result["selected_indices"].tolist() == [1]
    assert result["head_risk"].tolist() == [[0.0, 0.0]]
    assert result["anchor_agreement"].tolist() == [1.0]


def test_v113_asymmetric_risk_can_veto_an_uncertain_anchor_proposal():
    base, query, variant, query_valid, variant_valid = _inputs([0.25, 0.25])
    member_variants = variant.unsqueeze(0).repeat(3, 1, 1, 1, 1)
    member_variants[1:, 0, 0, 1] = torch.tensor([-8.0, 3.0])
    result = apply_asymmetric_risk_contextual_policy(
        base,
        query.unsqueeze(0).repeat(3, 1, 1, 1),
        member_variants,
        query_valid,
        variant_valid,
        aggregate_lcb_margin=0.12,
        min_head_lcb025=0.02,
        min_head_lcb050=0.0,
        risk_lambda025=0.5,
        risk_lambda050=0.25,
    )

    assert result["proposal_indices"].tolist() == [1]
    assert result["head_risk"][0, 0] > result["head_risk"][0, 1]
    assert result["selected_indices"].tolist() == [0]
    assert result["pareto_pass"].tolist() == [False]


def test_v113_committee_has_three_v99_compatible_members():
    committee = AsymmetricRiskContextualHierarchyCommittee()
    assert len(committee.members) == 3
    expected = {
        name: (value.dtype, tuple(value.shape))
        for name, value in ParetoContextualHierarchicalReranker(
        ).state_dict().items()
    }
    for member in committee.members:
        assert {
            name: (value.dtype, tuple(value.shape))
            for name, value in member.state_dict().items()
        } == expected
