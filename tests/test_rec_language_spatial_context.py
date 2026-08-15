import torch

from models.rec_language_spatial_context import (
    LanguageConditionedSpatialHierarchicalReranker,
)


def _inputs():
    generator = torch.Generator().manual_seed(7)
    query_features = torch.randn(2, 16, 152, generator=generator)
    variant_features = torch.randn(2, 16, 7, 25, generator=generator)
    query_aux_continuous = torch.randn(2, 16, 4, generator=generator)
    query_aux_binary = torch.zeros(2, 16, 2, dtype=torch.bool)
    variant_aux_continuous = torch.randn(2, 16, 7, 2, generator=generator)
    variant_aux_binary = torch.zeros(2, 16, 7, 2, dtype=torch.bool)
    query_valid = torch.ones(2, 16, dtype=torch.bool)
    query_valid[1, 11:] = False
    variant_valid = query_valid.unsqueeze(-1).expand(-1, -1, 7).clone()
    variant_valid[0, 3, 5:] = False
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


def test_v114_output_contract_and_padding():
    model = LanguageConditionedSpatialHierarchicalReranker().eval()
    inputs = _inputs()
    with torch.no_grad():
        output = model(**inputs)
    assert output["query_logits"].shape == (2, 16, 2)
    assert output["variant_logits"].shape == (2, 16, 7, 2)
    assert torch.isfinite(output["query_logits"]).all()
    assert torch.isfinite(output["variant_logits"]).all()
    assert output["query_logits"][~inputs["query_valid"]].eq(0.0).all()
    assert output["variant_logits"][~inputs["variant_valid"]].eq(0.0).all()


def test_v114_uses_pairwise_centers_and_target_language():
    model = LanguageConditionedSpatialHierarchicalReranker().eval()
    inputs = _inputs()
    with torch.no_grad():
        reference = model(**inputs)["query_logits"]
        moved = {name: value.clone() for name, value in inputs.items()}
        moved["query_features"][:, 0, 128:131] += torch.tensor(
            [2.0, -1.0, 0.5]
        )
        moved_output = model(**moved)["query_logits"]
        reworded = {name: value.clone() for name, value in inputs.items()}
        reworded["query_features"][:, :, 64:128] += 0.75
        reworded_output = model(**reworded)["query_logits"]
    assert not torch.equal(reference, moved_output)
    assert not torch.equal(reference, reworded_output)


def test_v114_rejects_an_unfrozen_width():
    try:
        LanguageConditionedSpatialHierarchicalReranker(hidden_dim=64)
    except ValueError as error:
        assert "hidden_dim=128" in str(error)
    else:
        raise AssertionError("V114 accepted an unfrozen hidden dimension")
