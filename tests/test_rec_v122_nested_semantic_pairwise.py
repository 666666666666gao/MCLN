import torch

from models.rec_hierarchical_reranker import QUERY_COUNT, VARIANT_COUNT
from models.rec_semantic_candidate_critic import (
    V121_AUX_DIM,
    V121_QUERY_EMBED_DIM,
    V121_TEXT_DIM,
    V121_VARIANT_EMBED_DIM,
)
from models.rec_semantic_pairwise_utility import (
    SemanticPairwiseUtilityCritic,
)
from scripts.run_v122_meshsp_nested_semantic_pairwise_oof import (
    build_hard_pair_training_population,
    pairwise_utility_accept_mask,
)


def _model_inputs(count=8):
    return {
        "proposal_query_embedding": torch.randn(
            count, V121_QUERY_EMBED_DIM, requires_grad=True
        ),
        "proposal_variant_embedding": torch.randn(
            count, V121_VARIANT_EMBED_DIM, requires_grad=True
        ),
        "baseline_query_embedding": torch.randn(
            count, V121_QUERY_EMBED_DIM, requires_grad=True
        ),
        "baseline_variant_embedding": torch.randn(
            count, V121_VARIANT_EMBED_DIM, requires_grad=True
        ),
        "target_text": torch.randn(
            count, V121_TEXT_DIM, requires_grad=True
        ),
        "proposal_aux": torch.randn(
            count, V121_AUX_DIM, requires_grad=True
        ),
        "baseline_aux": torch.randn(
            count, V121_AUX_DIM, requires_grad=True
        ),
        "model_gain": torch.randn(count, 2, requires_grad=True),
    }


def test_v122_pairwise_critic_shape_is_finite_and_trainable():
    model = SemanticPairwiseUtilityCritic()
    inputs = _model_inputs()
    logits = model(**inputs)
    assert tuple(logits.shape) == (8, 2)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    for value in inputs.values():
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_v122_hard_population_contains_direct_fix_and_break_events():
    count = 2
    components = {
        "query_embedding": torch.randn(
            count, QUERY_COUNT, V121_QUERY_EMBED_DIM
        ),
        "variant_embedding": torch.randn(
            count, QUERY_COUNT, VARIANT_COUNT,
            V121_VARIANT_EMBED_DIM,
        ),
        "target_text": torch.randn(count, V121_TEXT_DIM),
        "candidate_aux": torch.randn(
            count, QUERY_COUNT, VARIANT_COUNT, V121_AUX_DIM
        ),
        "candidate_valid": torch.ones(
            count, QUERY_COUNT, VARIANT_COUNT, dtype=torch.bool
        ),
    }
    probabilities = torch.full(
        (count, QUERY_COUNT, VARIANT_COUNT, 2), 0.5
    )
    probabilities[:, 0, 1] = 0.8
    prediction = {
        "semantic_components": components,
        "model_hit_probability": probabilities,
        "baselines": torch.zeros(count, dtype=torch.long),
    }
    fix_row = torch.full((QUERY_COUNT, VARIANT_COUNT), 0.10)
    fix_row[0, 1] = 0.60
    break_row = torch.full((QUERY_COUNT, VARIANT_COUNT), 0.60)
    break_row[0, 1] = 0.10
    inputs, targets, event_mask, receipt = (
        build_hard_pair_training_population(
            prediction,
            [
                {"candidate_ious": fix_row},
                {"candidate_ious": break_row},
            ],
        )
    )
    assert receipt["eligible_pair_count"] == 2
    assert receipt["event_pair_count"] == 2
    assert receipt["event_counts"] == {
        "025": {"fix": 1, "break": 1, "events": 2},
        "050": {"fix": 1, "break": 1, "events": 2},
    }
    assert torch.equal(targets, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    assert event_mask.all()
    assert inputs["model_gain"].shape == (2, 2)
    assert torch.allclose(inputs["model_gain"], torch.full((2, 2), 0.3))


def test_v122_gate_uses_only_the_fixed_acc025_logit_sign():
    base = torch.tensor([True, True, True, False])
    logits = torch.tensor([
        [0.1, -9.0],
        [0.0, -9.0],
        [-0.1, 9.0],
        [1.0, 1.0],
    ])
    assert torch.equal(
        pairwise_utility_accept_mask(base, logits),
        torch.tensor([True, True, False, False]),
    )
