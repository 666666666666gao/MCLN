import copy

import torch

from models.rec_anchored_spatial_adapter import (
    AnchoredLanguageSpatialResidualAdapter,
)
from models.rec_hierarchical_reranker import QUERY_COUNT, VARIANT_COUNT
from models.rec_semantic_candidate_critic import (
    SemanticCandidateHitCritic,
    V121_AUX_DIM,
    V121_QUERY_EMBED_DIM,
    V121_TEXT_DIM,
    V121_VARIANT_EMBED_DIM,
)
from scripts.run_v121_meshsp_nested_semantic_critic_oof import (
    candidate_hit_targets,
    flatten_semantic_candidates,
    predict_semantic_components,
    semantic_accept_mask,
)
from scripts.run_v97_contextual_listwise_hierarchical import (
    ContextualHierarchicalQueryVariantReranker,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    fit_hierarchical_normalization,
)
from test_train_scanrefer_rec_hierarchical_reranker import _materialized


def _components(count=2):
    return {
        "query_embedding": torch.randn(
            count, QUERY_COUNT, V121_QUERY_EMBED_DIM
        ),
        "variant_embedding": torch.randn(
            count, QUERY_COUNT, VARIANT_COUNT, V121_VARIANT_EMBED_DIM
        ),
        "target_text": torch.randn(count, V121_TEXT_DIM),
        "candidate_aux": torch.randn(
            count, QUERY_COUNT, VARIANT_COUNT, V121_AUX_DIM
        ),
        "candidate_valid": torch.ones(
            count, QUERY_COUNT, VARIANT_COUNT, dtype=torch.bool
        ),
    }


def test_v121_semantic_critic_shape_is_finite_and_trainable():
    model = SemanticCandidateHitCritic()
    count = 8
    inputs = {
        "query_embedding": torch.randn(
            count, V121_QUERY_EMBED_DIM, requires_grad=True
        ),
        "variant_embedding": torch.randn(
            count, V121_VARIANT_EMBED_DIM, requires_grad=True
        ),
        "target_text": torch.randn(
            count, V121_TEXT_DIM, requires_grad=True
        ),
        "candidate_aux": torch.randn(
            count, V121_AUX_DIM, requires_grad=True
        ),
    }
    logits = model(**inputs)
    assert tuple(logits.shape) == (count, 2)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    for value in inputs.values():
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_v121_flatten_uses_only_valid_candidates():
    components = _components()
    components["candidate_valid"][0, 0, 0] = False
    flattened = flatten_semantic_candidates(components)
    expected = 2 * QUERY_COUNT * VARIANT_COUNT - 1
    assert flattened["valid_mask"].sum().item() == expected
    assert flattened["query_embedding"].shape == (
        expected, V121_QUERY_EMBED_DIM
    )
    assert flattened["variant_embedding"].shape == (
        expected, V121_VARIANT_EMBED_DIM
    )
    assert flattened["target_text"].shape == (expected, V121_TEXT_DIM)
    assert flattened["candidate_aux"].shape == (expected, V121_AUX_DIM)


def test_v121_candidate_targets_are_strict_threshold_hits():
    valid = torch.ones(
        2, QUERY_COUNT, VARIANT_COUNT, dtype=torch.bool
    )
    valid[0, 0, 0] = False
    first = torch.full((QUERY_COUNT, VARIANT_COUNT), 0.10)
    second = torch.full((QUERY_COUNT, VARIANT_COUNT), 0.10)
    first[0, 0] = 0.60
    first[0, 1] = 0.30
    second[0, 0] = 0.50
    targets = candidate_hit_targets([
        {"candidate_ious": first},
        {"candidate_ious": second},
    ], valid)
    assert tuple(targets.shape) == (
        int(valid.sum().item()), 2
    )
    # The invalid 0.60 candidate is absent; 0.30 hits only .25.
    assert torch.equal(targets[0], torch.tensor([1.0, 0.0]))
    second_start = int(valid[0].sum().item())
    # IoU exactly 0.50 is a .25 hit but not a strict .50 hit.
    assert torch.equal(
        targets[second_start], torch.tensor([1.0, 0.0])
    )


def test_v121_gate_is_only_nonnegative_semantic_gain025():
    base = torch.tensor([True, True, True, False])
    gain = torch.tensor([
        [0.1, -9.0],
        [0.0, -9.0],
        [-0.1, 9.0],
        [1.0, 1.0],
    ])
    assert torch.equal(
        semantic_accept_mask(base, gain),
        torch.tensor([True, True, False, False]),
    )


def test_v121_real_v115_outputs_form_deployable_semantic_components():
    _, _, _, records = _materialized()
    statistics = fit_hierarchical_normalization(records)
    anchor = ContextualHierarchicalQueryVariantReranker(
        hidden_dim=128, dropout=0.1
    ).eval()
    model = AnchoredLanguageSpatialResidualAdapter(anchor).eval()
    first = predict_semantic_components(
        model, records, statistics, "cpu"
    )
    changed = copy.deepcopy(records)
    for record in changed:
        record["candidate_ious"].uniform_(0.0, 1.0)
    second = predict_semantic_components(
        model, changed, statistics, "cpu"
    )
    assert first["proposals"].shape == (2,)
    assert first["semantic_components"]["variant_embedding"].shape == (
        2, QUERY_COUNT, VARIANT_COUNT, V121_VARIANT_EMBED_DIM
    )
    # IoU is a training target only and cannot change inference components.
    for key, value in first["semantic_components"].items():
        assert torch.equal(value, second["semantic_components"][key])
