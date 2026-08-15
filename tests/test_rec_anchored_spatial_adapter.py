import torch

from models.rec_anchored_spatial_adapter import (
    AnchoredLanguageSpatialResidualAdapter,
)
from scripts.run_v97_contextual_listwise_hierarchical import (
    ContextualHierarchicalQueryVariantReranker,
)


def _actual_v99_anchor():
    return ContextualHierarchicalQueryVariantReranker(
        hidden_dim=128, dropout=0.1
    ).eval()


def _inputs():
    generator = torch.Generator().manual_seed(11)
    query_features = torch.randn(2, 16, 152, generator=generator)
    variant_features = torch.randn(2, 16, 7, 25, generator=generator)
    query_aux_continuous = torch.randn(2, 16, 4, generator=generator)
    query_aux_binary = torch.zeros(2, 16, 2, dtype=torch.bool)
    variant_aux_continuous = torch.randn(2, 16, 7, 2, generator=generator)
    variant_aux_binary = torch.zeros(2, 16, 7, 2, dtype=torch.bool)
    query_valid = torch.ones(2, 16, dtype=torch.bool)
    query_valid[1, 10:] = False
    variant_valid = query_valid.unsqueeze(-1).expand(-1, -1, 7).clone()
    query_features[~query_valid] = 0.0
    variant_features[~variant_valid] = 0.0
    query_aux_continuous[~query_valid] = 0.0
    variant_aux_continuous[~variant_valid] = 0.0
    return {
        "query_features": query_features,
        "variant_features": variant_features,
        "query_aux_continuous": query_aux_continuous,
        "query_aux_binary": query_aux_binary,
        "variant_aux_continuous": variant_aux_continuous,
        "variant_aux_binary": variant_aux_binary,
        "query_valid": query_valid,
        "variant_valid": variant_valid,
    }


def test_v115_initial_logits_exactly_equal_the_frozen_anchor():
    anchor = _actual_v99_anchor()
    adapter = AnchoredLanguageSpatialResidualAdapter(anchor).train()
    inputs = _inputs()
    with torch.no_grad():
        expected = anchor(**inputs)
        actual = adapter(**inputs)
    assert not adapter.anchor.training
    assert torch.equal(actual["query_logits"], expected["query_logits"])
    assert torch.equal(actual["variant_logits"], expected["variant_logits"])
    assert all(not parameter.requires_grad
               for parameter in adapter.anchor.parameters())


def test_v115_residual_is_bounded_and_padding_stays_zero():
    anchor = _actual_v99_anchor()
    adapter = AnchoredLanguageSpatialResidualAdapter(anchor).eval()
    inputs = _inputs()
    with torch.no_grad():
        adapter.query_delta_head[-1].bias.copy_(torch.tensor([10.0, -10.0]))
        adapter.variant_delta_head[-1].bias.copy_(torch.tensor([-10.0, 10.0]))
        expected = anchor(**inputs)
        actual = adapter(**inputs)
    query_difference = actual["query_logits"] - expected["query_logits"]
    variant_difference = actual["variant_logits"] - expected["variant_logits"]
    assert query_difference.abs().max() <= 0.250001
    assert variant_difference.abs().max() <= 0.250001
    assert query_difference[inputs["query_valid"]].abs().max() > 0.24
    assert variant_difference[inputs["variant_valid"]].abs().max() > 0.24
    assert actual["query_logits"][~inputs["query_valid"]].eq(0.0).all()
    assert actual["variant_logits"][~inputs["variant_valid"]].eq(0.0).all()


def test_v115_backward_never_updates_the_anchor():
    anchor = _actual_v99_anchor()
    adapter = AnchoredLanguageSpatialResidualAdapter(anchor).train()
    output = adapter(**_inputs())
    loss = output["query_logits"].sum() + output["variant_logits"].sum()
    loss.backward()
    assert all(parameter.grad is None for parameter in anchor.parameters())
    assert any(parameter.grad is not None for name, parameter
               in adapter.named_parameters() if not name.startswith("anchor."))
