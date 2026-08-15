import pytest
import torch
import torch.nn.functional as F

from models.rec_dual_semantic_spatial_adapter import (
    AnchoredDualSemanticSpatialResidualAdapter,
)
from models.rec_hyperspherical_semantic_adapter import (
    V130_SEMANTIC_INPUT_DIM,
    recover_hyperspherical_semantic_features,
)
from scripts.run_v97_contextual_listwise_hierarchical import (
    ContextualHierarchicalQueryVariantReranker,
)


def _actual_v99_anchor():
    return ContextualHierarchicalQueryVariantReranker(
        hidden_dim=128, dropout=0.1
    ).eval()


def _statistics():
    mean = torch.linspace(-0.3, 0.3, 152, dtype=torch.float32)
    std = torch.linspace(0.4, 1.4, 152, dtype=torch.float32)
    return {
        "groups": {
            "query_features": {
                "mean": mean,
                "std": std,
            }
        }
    }


def _inputs():
    generator = torch.Generator().manual_seed(11)
    raw_query_features = torch.randn(2, 16, 152, generator=generator)
    raw_query_features[:, :, :64] = F.normalize(
        raw_query_features[:, :, :64], p=2, dim=-1
    )
    target_text = F.normalize(
        torch.randn(2, 1, 64, generator=generator), p=2, dim=-1
    )
    raw_query_features[:, :, 64:128] = target_text
    variant_features = torch.randn(2, 16, 7, 25, generator=generator)
    query_aux_continuous = torch.randn(2, 16, 4, generator=generator)
    query_aux_binary = torch.zeros(2, 16, 2, dtype=torch.bool)
    variant_aux_continuous = torch.randn(2, 16, 7, 2, generator=generator)
    variant_aux_binary = torch.zeros(2, 16, 7, 2, dtype=torch.bool)
    query_valid = torch.ones(2, 16, dtype=torch.bool)
    query_valid[1, 10:] = False
    variant_valid = query_valid.unsqueeze(-1).expand(-1, -1, 7).clone()
    group = _statistics()["groups"]["query_features"]
    query_features = (
        raw_query_features - group["mean"]
    ) / group["std"]
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


def _adapter():
    return AnchoredDualSemanticSpatialResidualAdapter(
        _actual_v99_anchor(), _statistics()
    )


def _permute_queries(inputs, permutation):
    result = {}
    for name, value in inputs.items():
        if name.startswith("query_") or name.startswith("variant_"):
            result[name] = value[:, permutation]
        else:
            result[name] = value
    return result


def test_v131_initial_logits_exactly_equal_the_frozen_anchor():
    adapter = _adapter().train()
    inputs = _inputs()
    with torch.no_grad():
        expected = adapter.anchor(**inputs)
        actual = adapter(**inputs)
    assert not adapter.anchor.training
    assert torch.equal(actual["query_logits"], expected["query_logits"])
    assert torch.equal(actual["variant_logits"], expected["variant_logits"])
    assert all(not parameter.requires_grad
               for parameter in adapter.anchor.parameters())


def test_v131_residual_is_bounded_and_padding_stays_zero():
    adapter = _adapter().eval()
    inputs = _inputs()
    with torch.no_grad():
        adapter.query_delta_head[-1].bias.copy_(torch.tensor([10.0, -10.0]))
        adapter.variant_delta_head[-1].bias.copy_(torch.tensor([-10.0, 10.0]))
        expected = adapter.anchor(**inputs)
        actual = adapter(**inputs)
    query_difference = actual["query_logits"] - expected["query_logits"]
    variant_difference = actual["variant_logits"] - expected["variant_logits"]
    assert query_difference.abs().max() <= 0.250001
    assert variant_difference.abs().max() <= 0.250001
    assert query_difference[inputs["query_valid"]].abs().max() > 0.24
    assert variant_difference[inputs["variant_valid"]].abs().max() > 0.24
    assert actual["query_logits"][~inputs["query_valid"]].eq(0.0).all()
    assert actual["variant_logits"][~inputs["variant_valid"]].eq(0.0).all()


def test_v131_recovers_unit_sphere_product_difference_and_cosine():
    inputs = _inputs()
    group = _statistics()["groups"]["query_features"]
    recovered = recover_hyperspherical_semantic_features(
        inputs["query_features"], inputs["query_valid"],
        group["mean"], group["std"],
    )
    assert recovered["features"].shape == (2, 16, V130_SEMANTIC_INPUT_DIM)
    assert recovered["features"][~inputs["query_valid"]].eq(0.0).all()
    assert torch.isfinite(recovered["features"]).all()
    valid = inputs["query_valid"]
    assert torch.allclose(
        recovered["query_projection"][valid].norm(dim=-1),
        torch.ones_like(
            recovered["query_projection"][valid].norm(dim=-1)
        ),
        atol=1e-5, rtol=1e-5,
    )
    assert torch.allclose(
        recovered["cosine"],
        (
            recovered["query_projection"] * recovered["target_text"]
        ).sum(dim=-1, keepdim=True),
    )


def test_v131_is_query_permutation_equivariant():
    adapter = _adapter().eval()
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


def test_v131_both_paths_change_context_without_anchor_gradients():
    adapter = _adapter().eval()
    inputs = _inputs()
    changed = {name: value.clone() for name, value in inputs.items()}
    group = _statistics()["groups"]["query_features"]
    raw = (
        changed["query_features"] * group["std"] + group["mean"]
    )
    raw[:, 0, :64].mul_(-1.0)
    raw[:, 0, 128:131].add_(0.5)
    changed["query_features"] = (raw - group["mean"]) / group["std"]
    changed["query_features"][~changed["query_valid"]] = 0.0
    with torch.no_grad():
        original = adapter(**inputs)
        modified = adapter(**changed)
    assert not torch.equal(
        original["semantic_context"], modified["semantic_context"]
    )
    assert not torch.equal(
        original["spatial_context"], modified["spatial_context"]
    )
    assert original["dual_gates"].shape == (2, 16, 2)

    adapter.train()
    output = adapter(**inputs)
    loss = output["query_logits"].sum() + output["variant_logits"].sum()
    loss.backward()
    assert all(parameter.grad is None for parameter in adapter.anchor.parameters())
    gradients = [parameter.grad for name, parameter in adapter.named_parameters()
                 if not name.startswith("anchor.") and parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke only")
def test_v131_gpu_dual_paths_optimize_after_zero_delta_initialization():
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    adapter = _adapter().to(device)
    inputs = {name: value.to(device) for name, value in _inputs().items()}
    with torch.no_grad():
        anchor_output = adapter.anchor(**inputs)
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
    semantic_gradient_seen = False
    spatial_gradient_seen = False
    for _ in range(24):
        loss = objective(adapter(**inputs))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in trainable
                     if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        semantic_gradient_seen |= any(
            parameter.grad is not None
            and bool(parameter.grad.abs().sum().gt(0).item())
            for name, parameter in adapter.named_parameters()
            if name.startswith("semantic_encoder.")
        )
        spatial_gradient_seen |= any(
            parameter.grad is not None
            and bool(parameter.grad.abs().sum().gt(0).item())
            for name, parameter in adapter.named_parameters()
            if name.startswith("spatial_context.")
        )
        optimizer.step()
    adapter.eval()
    with torch.no_grad():
        final_loss = float(objective(adapter(**inputs)).item())
    assert semantic_gradient_seen
    assert spatial_gradient_seen
    assert final_loss < 0.75 * initial_loss
