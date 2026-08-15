import torch

from models.rec_hierarchical_reranker import QUERY_COUNT, VARIANT_COUNT
from models.rec_semantic_candidate_critic import (
    V121_AUX_DIM,
    V121_QUERY_EMBED_DIM,
    V121_TEXT_DIM,
    V121_VARIANT_EMBED_DIM,
)
from models.rec_semantic_antisymmetric_utility import (
    SemanticAntisymmetricUtilityRanker,
)
from scripts.run_v128_meshsp_nested_semantic_protected_rescue_oof import (
    all_hard_candidate_inputs,
    build_listwise_training_population,
    listwise_abstain_loss,
    select_all_hard_candidates,
    select_v115_protected_rescue,
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


def test_v128_antisymmetric_ranker_shape_is_finite_and_trainable():
    model = SemanticAntisymmetricUtilityRanker()
    inputs = _model_inputs()
    logits = model(**inputs)
    assert tuple(logits.shape) == (8, 2)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    for value in inputs.values():
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_v128_ranker_is_strictly_antisymmetric_under_pair_swap():
    model = SemanticAntisymmetricUtilityRanker().eval()
    inputs = _model_inputs(count=11)
    direct = model(**inputs)
    swapped = {
        "proposal_query_embedding": inputs["baseline_query_embedding"],
        "proposal_variant_embedding": inputs["baseline_variant_embedding"],
        "baseline_query_embedding": inputs["proposal_query_embedding"],
        "baseline_variant_embedding": inputs["proposal_variant_embedding"],
        "target_text": inputs["target_text"],
        "proposal_aux": inputs["baseline_aux"],
        "baseline_aux": inputs["proposal_aux"],
        "model_gain": -inputs["model_gain"],
    }
    reverse = model(**swapped)
    assert torch.allclose(direct, -reverse, atol=1e-6, rtol=1e-6)
    assert torch.nn.functional.softplus(
        model.model_gain_raw_scale
    ).gt(0.0).all()


def test_v128_listwise_population_uses_fix_actions_or_baseline_target():
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
    inputs, pair_rows, pair_flat, pair_positive, receipt = (
        build_listwise_training_population(
            prediction,
            [
                {"candidate_ious": fix_row},
                {"candidate_ious": break_row},
            ],
        )
    )
    assert receipt["eligible_pair_count"] == 2
    assert receipt["rows_with_candidates"] == 2
    assert receipt["candidate_positive_actions"] == {
        "025": 1, "050": 1,
    }
    assert receipt["candidate_positive_rows"] == {"025": 1, "050": 1}
    assert receipt["baseline_target_rows"] == {"025": 1, "050": 1}
    assert torch.equal(pair_rows, torch.tensor([0, 1]))
    assert torch.equal(pair_flat, torch.tensor([1, 1]))
    assert torch.equal(
        pair_positive,
        torch.tensor([[True, True], [False, False]]),
    )
    assert inputs["model_gain"].shape == (2, 2)
    assert torch.allclose(inputs["model_gain"], torch.full((2, 2), 0.3))


def test_v128_listwise_loss_normalizes_candidates_with_baseline_action():
    logits = torch.tensor(
        [[[2.0, 2.0]], [[1.0, 1.0]]], requires_grad=True
    )
    positive = torch.tensor(
        [[[True, True]], [[False, False]]]
    )
    loss, baseline_target = listwise_abstain_loss(logits, positive)
    expected = (
        torch.nn.functional.softplus(torch.tensor(-2.0))
        + torch.nn.functional.softplus(torch.tensor(1.0))
    ) / 2.0
    assert torch.allclose(loss, expected)
    assert torch.equal(
        baseline_target,
        torch.tensor([[False, False], [True, True]]),
    )
    loss.backward()
    assert logits.grad[0].lt(0.0).all()
    assert logits.grad[1].gt(0.0).all()


class _AuxUtility(torch.nn.Module):
    def forward(self, proposal_aux, **unused):
        return torch.stack((proposal_aux[:, 0], proposal_aux[:, 1]), dim=1)


def test_v128_selector_protects_v115_and_adds_rescue_only_on_abstention():
    count = 3
    candidate_aux = torch.zeros(
        count, QUERY_COUNT, VARIANT_COUNT, V121_AUX_DIM
    )
    candidate_aux[0, 0, 1, :2] = torch.tensor([0.1, 0.2])
    candidate_aux[0, 0, 2, :2] = torch.tensor([0.9, -0.4])
    candidate_aux[1, 0, 1, :2] = torch.tensor([-0.2, 0.7])
    candidate_aux[2, 0, 1, :2] = torch.tensor([0.7, -0.1])
    components = {
        "query_embedding": torch.randn(
            count, QUERY_COUNT, V121_QUERY_EMBED_DIM
        ),
        "variant_embedding": torch.randn(
            count, QUERY_COUNT, VARIANT_COUNT,
            V121_VARIANT_EMBED_DIM,
        ),
        "target_text": torch.randn(count, V121_TEXT_DIM),
        "candidate_aux": candidate_aux,
        "candidate_valid": torch.ones(
            count, QUERY_COUNT, VARIANT_COUNT, dtype=torch.bool
        ),
    }
    probabilities = torch.full(
        (count, QUERY_COUNT, VARIANT_COUNT, 2), 0.5
    )
    probabilities[0, 0, 1] = torch.tensor([0.7, 0.7])
    probabilities[0, 0, 2] = torch.tensor([0.8, 0.7])
    probabilities[1, 0, 1] = torch.tensor([0.7, 0.7])
    probabilities[2, 0, 1] = torch.tensor([0.7, 0.7])
    prediction = {
        "semantic_components": components,
        "model_hit_probability": probabilities,
        "baselines": torch.zeros(count, dtype=torch.long),
        "proposals": torch.ones(count, dtype=torch.long),
        "base_accepted": torch.tensor([True, False, False]),
        "head_gain": torch.full((count, 2), 0.2),
        "aggregate_gain": torch.full((count,), 0.6),
    }
    population = all_hard_candidate_inputs(prediction)
    assert int(population["pair_rows"].numel()) == 4
    selected = select_all_hard_candidates(
        _AuxUtility(), prediction, "cpu"
    )
    assert selected["pair_count"] == 4
    assert torch.equal(selected["eligible_counts"], torch.tensor([2, 1, 1]))
    assert torch.equal(selected["proposals"], torch.tensor([2, 1, 1]))
    assert torch.equal(
        selected["accepted"], torch.tensor([True, False, True])
    )
    assert torch.equal(selected["selected"], torch.tensor([2, 0, 1]))
    assert torch.equal(
        selected["candidate_available"], torch.tensor([True, True, True])
    )
    assert selected["head_gain"].gt(0.0).all()
    protected = select_v115_protected_rescue(
        _AuxUtility(), prediction, "cpu"
    )
    assert torch.equal(protected["proposals"], torch.tensor([1, 1, 1]))
    assert torch.equal(protected["selected"], torch.tensor([1, 0, 1]))
    assert torch.equal(
        protected["accepted"], torch.tensor([True, False, True])
    )
    assert torch.equal(
        protected["rescue_accepted"], torch.tensor([False, False, True])
    )
    assert torch.equal(
        protected["rescue_raw_accepted"], torch.tensor([True, False, True])
    )
