import copy

import torch

from models.rec_hierarchical_reranker import (
    QUERY_AUX_BINARY_DIM,
    QUERY_AUX_CONTINUOUS_DIM,
    QUERY_COUNT,
    QUERY_FEATURE_DIM,
    VARIANT_AUX_BINARY_DIM,
    VARIANT_AUX_CONTINUOUS_DIM,
    VARIANT_COUNT,
    VARIANT_FEATURE_DIM,
)
from scripts.run_v97_contextual_listwise_hierarchical import (
    ContextualHierarchicalQueryVariantReranker,
)


def inputs():
    torch.manual_seed(4)
    batch = 2
    query_valid = torch.zeros(batch, QUERY_COUNT, dtype=torch.bool)
    query_valid[:, :4] = True
    variant_valid = torch.zeros(
        batch, QUERY_COUNT, VARIANT_COUNT, dtype=torch.bool
    )
    variant_valid[:, :4, :3] = True
    return {
        "query_features": torch.randn(batch, QUERY_COUNT, QUERY_FEATURE_DIM),
        "variant_features": torch.randn(
            batch, QUERY_COUNT, VARIANT_COUNT, VARIANT_FEATURE_DIM
        ),
        "query_aux_continuous": torch.randn(
            batch, QUERY_COUNT, QUERY_AUX_CONTINUOUS_DIM
        ),
        "query_aux_binary": torch.randint(
            0, 2, (batch, QUERY_COUNT, QUERY_AUX_BINARY_DIM),
            dtype=torch.bool,
        ),
        "variant_aux_continuous": torch.randn(
            batch, QUERY_COUNT, VARIANT_COUNT, VARIANT_AUX_CONTINUOUS_DIM
        ),
        "variant_aux_binary": torch.randint(
            0, 2,
            (batch, QUERY_COUNT, VARIANT_COUNT, VARIANT_AUX_BINARY_DIM),
            dtype=torch.bool,
        ),
        "query_valid": query_valid,
        "variant_valid": variant_valid,
    }


def test_context_is_query_permutation_equivariant():
    model = ContextualHierarchicalQueryVariantReranker(128, 0.1).eval()
    values = inputs()
    order = torch.tensor([2, 0, 3, 1] + list(range(4, QUERY_COUNT)))
    inverse = torch.argsort(order)
    permuted = {}
    for name, value in values.items():
        permuted[name] = value[:, order]
    with torch.no_grad():
        reference = model(**values)
        changed = model(**permuted)
    torch.testing.assert_close(
        reference["query_logits"], changed["query_logits"][:, inverse]
    )
    torch.testing.assert_close(
        reference["variant_logits"], changed["variant_logits"][:, inverse]
    )


def test_invalid_padding_cannot_influence_valid_outputs():
    model = ContextualHierarchicalQueryVariantReranker(128, 0.1).eval()
    values = inputs()
    changed = {name: value.clone() for name, value in values.items()}
    changed["query_features"][:, 4:] = 1e6
    changed["variant_features"][:, 4:] = -1e6
    changed["query_aux_continuous"][:, 4:] = 1e6
    changed["variant_aux_continuous"][:, 4:] = -1e6
    with torch.no_grad():
        before = model(**values)
        after = model(**changed)
    torch.testing.assert_close(
        before["query_logits"][:, :4], after["query_logits"][:, :4]
    )
    torch.testing.assert_close(
        before["variant_logits"][:, :4], after["variant_logits"][:, :4]
    )


def test_valid_query_context_changes_other_valid_query_and_gradients_are_finite():
    model = ContextualHierarchicalQueryVariantReranker(128, 0.0).eval()
    values = inputs()
    changed = {name: value.clone() for name, value in values.items()}
    changed["query_features"][:, 0] += 10.0
    before = model(**values)
    after = model(**changed)
    assert not torch.equal(
        before["query_logits"][:, 1], after["query_logits"][:, 1]
    )
    loss = before["query_logits"][:, :4].sum() + before[
        "variant_logits"
    ][:, :4, :3].sum()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
