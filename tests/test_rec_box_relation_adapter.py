import pytest
import torch

from models.rec_box_relation_adapter import (
    AnchoredTextConditionedBoxRelationAdapter,
    V129_EDGE_DIM,
    build_directed_box_relation_features,
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
    query_features[:, :, 128:131] = torch.rand(
        2, 16, 3, generator=generator
    )
    query_features[:, :, 131:134] = 0.05 + 0.45 * torch.rand(
        2, 16, 3, generator=generator
    )
    target_text = torch.randn(2, 1, 64, generator=generator)
    query_features[:, :, 64:128] = target_text
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


def _permute_queries(inputs, permutation):
    result = {}
    for name, value in inputs.items():
        if name.startswith("query_"):
            result[name] = value[:, permutation]
        elif name.startswith("variant_"):
            result[name] = value[:, permutation]
        else:
            result[name] = value
    return result


def test_v129_initial_logits_exactly_equal_the_frozen_anchor():
    anchor = _actual_v99_anchor()
    adapter = AnchoredTextConditionedBoxRelationAdapter(anchor).train()
    inputs = _inputs()
    with torch.no_grad():
        expected = anchor(**inputs)
        actual = adapter(**inputs)
    assert not adapter.anchor.training
    assert torch.equal(actual["query_logits"], expected["query_logits"])
    assert torch.equal(actual["variant_logits"], expected["variant_logits"])
    assert all(not parameter.requires_grad
               for parameter in adapter.anchor.parameters())


def test_v129_residual_is_bounded_and_padding_stays_zero():
    anchor = _actual_v99_anchor()
    adapter = AnchoredTextConditionedBoxRelationAdapter(anchor).eval()
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


def test_v129_directed_relation_schema_is_finite_and_masks_padding():
    inputs = _inputs()
    relation, pair_valid = build_directed_box_relation_features(
        inputs["query_features"], inputs["query_valid"]
    )
    assert relation.shape == (2, 16, 16, V129_EDGE_DIM)
    assert pair_valid.shape == (2, 16, 16)
    assert torch.isfinite(relation).all()
    assert relation[~pair_valid].eq(0.0).all()
    assert torch.allclose(
        relation[:, :, :, 0:3],
        -relation[:, :, :, 0:3].transpose(1, 2),
    )


def test_v129_is_query_permutation_equivariant():
    adapter = AnchoredTextConditionedBoxRelationAdapter(
        _actual_v99_anchor()
    ).eval()
    inputs = _inputs()
    permutation = torch.tensor([7, 0, 12, 5, 2, 15, 4, 10,
                                1, 14, 8, 3, 9, 6, 13, 11])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        expected = adapter(**inputs)
        actual = adapter(**_permute_queries(inputs, permutation))
    assert torch.allclose(
        actual["query_logits"][:, inverse], expected["query_logits"],
        atol=1e-6, rtol=1e-6,
    )
    assert torch.allclose(
        actual["variant_logits"][:, inverse], expected["variant_logits"],
        atol=1e-6, rtol=1e-6,
    )


def test_v129_text_and_box_relations_change_attention_without_anchor_gradients():
    anchor = _actual_v99_anchor()
    adapter = AnchoredTextConditionedBoxRelationAdapter(anchor).eval()
    inputs = _inputs()
    changed = {name: value.clone() for name, value in inputs.items()}
    changed["query_features"][:, :, 64:128].mul_(-1.0)
    changed["query_features"][:, 0, 128:134] += 0.2
    with torch.no_grad():
        original = adapter(**inputs)
        modified = adapter(**changed)
    assert not torch.equal(
        original["relation_attention"], modified["relation_attention"]
    )

    adapter.train()
    output = adapter(**inputs)
    loss = output["query_logits"].sum() + output["variant_logits"].sum()
    loss.backward()
    assert all(parameter.grad is None for parameter in anchor.parameters())
    trainable_gradients = [
        parameter.grad for name, parameter in adapter.named_parameters()
        if not name.startswith("anchor.") and parameter.grad is not None
    ]
    assert trainable_gradients
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke only")
def test_v129_gpu_relation_path_optimizes_after_zero_delta_initialization():
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    anchor = _actual_v99_anchor().to(device)
    adapter = AnchoredTextConditionedBoxRelationAdapter(anchor).to(device)
    inputs = {name: value.to(device) for name, value in _inputs().items()}
    with torch.no_grad():
        anchor_output = anchor(**inputs)
        query_target = anchor_output["query_logits"] + 0.15 * torch.sign(
            torch.randn_like(anchor_output["query_logits"])
        )
        variant_target = anchor_output["variant_logits"] + 0.15 * torch.sign(
            torch.randn_like(anchor_output["variant_logits"])
        )

    def objective(output):
        query_error = (
            output["query_logits"] - query_target
        )[inputs["query_valid"]]
        variant_error = (
            output["variant_logits"] - variant_target
        )[inputs["variant_valid"]]
        return query_error.square().mean() + variant_error.square().mean()

    adapter.eval()
    with torch.no_grad():
        initial_loss = float(objective(adapter(**inputs)).item())
    adapter.train()
    trainable = [parameter for parameter in adapter.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-3)
    relation_gradient_seen = False
    for _ in range(24):
        output = adapter(**inputs)
        loss = objective(output)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in trainable
                     if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        relation_gradient_seen |= any(
            parameter.grad is not None
            and bool(parameter.grad.abs().sum().gt(0).item())
            for name, parameter in adapter.named_parameters()
            if name.startswith("edge_encoder.")
        )
        optimizer.step()
    adapter.eval()
    with torch.no_grad():
        final_loss = float(objective(adapter(**inputs)).item())
    assert relation_gradient_seen
    assert final_loss < 0.75 * initial_loss
