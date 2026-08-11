import pytest
import torch

import models.rec_geometry_reranker as rec_geometry_reranker
from models.rec_geometry_reranker import (
    FLAT_PARENT_PRIOR_VERSION,
    REC_GEOMETRY_MODEL_SCHEMA_VERSION,
    blend_rec_geometry_scores,
    build_deployed_parent_state,
    build_flat_parent_prior,
    build_rec_geometry_model_inputs,
    stable_flat_descending_indices,
    stable_query_descending_order,
)


def _parent_state(compact_scores=None, query_indices=None,
                  candidate_valid=None, num_queries=6):
    if compact_scores is None:
        compact_scores = torch.tensor([[2.0, 1.0]])
    if query_indices is None:
        query_indices = torch.tensor([[1, 3]])
    if candidate_valid is None:
        candidate_valid = torch.ones_like(query_indices, dtype=torch.bool)
    return build_deployed_parent_state(
        compact_scores, query_indices, candidate_valid, num_queries
    )


def test_schema_versions_are_explicit_and_stable():
    assert REC_GEOMETRY_MODEL_SCHEMA_VERSION == "rec-geometry-flat-v1"
    assert FLAT_PARENT_PRIOR_VERSION == \
        "score-desc-query-index-asc-regressed-first-v2"


def test_geometry_inputs_use_query_major_variant_minor_feature_order():
    batch_size, candidates, variants = 2, 3, 2
    base_dim, geometry_dim = 4, 5
    base = torch.arange(
        batch_size * candidates * base_dim, dtype=torch.float32
    ).reshape(batch_size, candidates, base_dim)
    geometry = 100.0 + torch.arange(
        batch_size * candidates * variants * geometry_dim,
        dtype=torch.float32,
    ).reshape(batch_size, candidates, variants, geometry_dim)
    parent_scores = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    parent_top1 = torch.tensor([
        [False, True, False],
        [True, False, False],
    ])
    valid = torch.ones(batch_size, candidates, variants, dtype=torch.bool)
    base_names = ["base_{}".format(idx) for idx in range(base_dim)]
    geometry_names = [
        "geometry_{}".format(idx) for idx in range(geometry_dim)
    ]

    flat = build_rec_geometry_model_inputs(
        base,
        geometry,
        parent_scores,
        parent_top1,
        valid,
        base_names,
        geometry_names,
    )

    assert flat["schema_version"] == REC_GEOMETRY_MODEL_SCHEMA_VERSION
    assert flat["features"].shape == (2, 6, 11)
    assert flat["valid_mask"].shape == (2, 6)
    assert flat["feature_names"] == (
        tuple(base_names)
        + tuple(geometry_names)
        + ("parent_score", "parent_is_deployed_top1")
    )
    assert flat["query_positions"].tolist() == [
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
    ]
    assert flat["variant_indices"].tolist() == [
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1],
    ]
    for batch_idx in range(batch_size):
        for query_idx in range(candidates):
            for variant_idx in range(variants):
                flat_idx = query_idx * variants + variant_idx
                expected = torch.cat([
                    base[batch_idx, query_idx],
                    geometry[batch_idx, query_idx, variant_idx],
                    parent_scores[batch_idx, query_idx].reshape(1),
                    parent_top1[batch_idx, query_idx]
                    .to(base.dtype).reshape(1),
                ])
                assert torch.equal(
                    flat["features"][batch_idx, flat_idx], expected
                )


def test_geometry_inputs_naturally_build_the_179_dimension_contract():
    flat = build_rec_geometry_model_inputs(
        torch.zeros(1, 1, 152),
        torch.zeros(1, 1, 7, 25),
        torch.zeros(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 1, 7, dtype=torch.bool),
        ["base_{}".format(idx) for idx in range(152)],
        ["geometry_{}".format(idx) for idx in range(25)],
    )

    assert 152 + 25 + 2 == 179
    assert flat["features"].shape == (1, 7, 179)
    assert len(flat["feature_names"]) == 179


def test_geometry_inputs_zero_invalid_rows_and_ignore_invalid_nonfinite_data():
    base = torch.tensor([[[1.0, 2.0], [float("nan"), float("inf")]]])
    geometry = torch.tensor([[[
        [3.0, 4.0],
        [float("nan"), float("inf")],
    ], [
        [float("nan"), float("inf")],
        [float("nan"), float("inf")],
    ]]])
    parent_scores = torch.tensor([[0.5, float("nan")]])
    parent_top1 = torch.tensor([[True, False]])
    valid = torch.tensor([[[True, False], [False, False]]])

    flat = build_rec_geometry_model_inputs(
        base,
        geometry,
        parent_scores,
        parent_top1,
        valid,
        ["base_0", "base_1"],
        ["geometry_0", "geometry_1"],
    )

    assert flat["valid_mask"].tolist() == [[True, False, False, False]]
    assert torch.equal(flat["features"][0, 0], torch.tensor([
        1.0, 2.0, 3.0, 4.0, 0.5, 1.0,
    ]))
    assert torch.equal(
        flat["features"][~flat["valid_mask"]],
        torch.zeros(3, 6),
    )
    assert torch.isfinite(flat["features"]).all()


@pytest.mark.parametrize("bad_field", [
    "base_shape",
    "geometry_shape",
    "parent_shape",
    "top1_dtype",
    "valid_dtype",
    "base_names",
    "geometry_names",
    "duplicate_names",
    "nonfinite_base",
    "nonfinite_geometry",
    "nonfinite_parent",
    "empty_valid_row",
])
def test_geometry_inputs_reject_malformed_or_nonfinite_valid_data(bad_field):
    base = torch.zeros(1, 2, 2)
    geometry = torch.zeros(1, 2, 2, 2)
    parent_scores = torch.zeros(1, 2)
    parent_top1 = torch.tensor([[True, False]])
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    base_names = ["base_0", "base_1"]
    geometry_names = ["geometry_0", "geometry_1"]

    if bad_field == "base_shape":
        base = base.unsqueeze(0)
    elif bad_field == "geometry_shape":
        geometry = geometry[:, :, 0]
    elif bad_field == "parent_shape":
        parent_scores = parent_scores[:, :1]
    elif bad_field == "top1_dtype":
        parent_top1 = parent_top1.long()
    elif bad_field == "valid_dtype":
        valid = valid.long()
    elif bad_field == "base_names":
        base_names = base_names[:1]
    elif bad_field == "geometry_names":
        geometry_names = geometry_names + ["extra"]
    elif bad_field == "duplicate_names":
        geometry_names[0] = base_names[0]
    elif bad_field == "nonfinite_base":
        base[0, 0, 0] = float("nan")
    elif bad_field == "nonfinite_geometry":
        geometry[0, 0, 0, 0] = float("inf")
    elif bad_field == "nonfinite_parent":
        parent_scores[0, 0] = float("nan")
    elif bad_field == "empty_valid_row":
        valid.zero_()

    with pytest.raises((TypeError, ValueError)):
        build_rec_geometry_model_inputs(
            base,
            geometry,
            parent_scores,
            parent_top1,
            valid,
            base_names,
            geometry_names,
        )


def test_parent_state_scatters_before_top1_and_matches_query_axis_ties():
    compact = torch.tensor([[1.0, 1.0, 0.2]])
    query_indices = torch.tensor([[5, 1, 7]])
    valid = torch.tensor([[True, True, True]])

    parent = build_deployed_parent_state(
        compact, query_indices, valid, num_queries=8
    )

    expected_scores = torch.tensor([[
        -float("inf"), 1.0, -float("inf"), -float("inf"),
        -float("inf"), 1.0, -float("inf"), 0.2,
    ]])
    expected_order = torch.tensor([[1, 5, 7, 0, 2, 3, 4, 6]])
    assert torch.equal(parent["compact_scores"], compact)
    assert torch.equal(parent["query_scores"], expected_scores)
    assert torch.equal(parent["query_order"], expected_order)
    assert torch.equal(parent["top1_query_index"], expected_order[:, 0])
    assert parent["top1_query_index"].item() == 1
    assert parent["parent_top1_mask"].tolist() == [[False, True, False]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_real_shape_parent_prior_is_backend_stable_with_ties_and_inf_padding():
    tied_scores = torch.tensor([[0.94, 0.94]], dtype=torch.float32)
    remaining_scores = torch.linspace(0.93, 0.80, 14).unsqueeze(0)
    compact_scores = torch.cat([tied_scores, remaining_scores], dim=1)
    query_indices = torch.tensor([[
        13, 2, 0, 1, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12, 14, 15,
    ]])
    candidate_valid = torch.ones(1, 16, dtype=torch.bool)
    geometry_valid = torch.ones(1, 16, 7, dtype=torch.bool)

    cpu_parent = build_deployed_parent_state(
        compact_scores, query_indices, candidate_valid, num_queries=256
    )
    cpu_prior = build_flat_parent_prior(
        cpu_parent, geometry_valid, regressed_variant_index=0
    )
    moved_parent = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in cpu_parent.items()
    }
    moved_prior = build_flat_parent_prior(
        moved_parent, geometry_valid.cuda(), regressed_variant_index=0
    )
    cuda_parent = build_deployed_parent_state(
        compact_scores.cuda(),
        query_indices.cuda(),
        candidate_valid.cuda(),
        num_queries=256,
    )
    cuda_prior = build_flat_parent_prior(
        cuda_parent, geometry_valid.cuda(), regressed_variant_index=0
    )

    expected_finite_order = torch.tensor([
        2, 13, 0, 1, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12, 14, 15,
    ])
    assert cpu_parent["top1_query_index"].item() == 2
    assert cuda_parent["top1_query_index"].item() == 2
    assert torch.equal(
        cpu_parent["query_order"][0, :16], expected_finite_order
    )
    assert torch.equal(
        cuda_parent["query_order"][0, :16].cpu(), expected_finite_order
    )
    assert torch.equal(
        cuda_parent["query_order"].cpu(), cpu_parent["query_order"]
    )
    assert torch.isneginf(cpu_parent["query_scores"][0, 16:]).sum() == 240
    assert torch.equal(moved_prior.cpu(), cpu_prior)
    assert torch.equal(cuda_prior.cpu(), cpu_prior)


def test_flat_parent_prior_ignores_negative_infinity_padding_order():
    parent = build_deployed_parent_state(
        torch.tensor([[1.0, 1.0, 0.5]]),
        torch.tensor([[5, 1, 7]]),
        torch.ones(1, 3, dtype=torch.bool),
        num_queries=8,
    )
    geometry_valid = torch.ones(1, 3, 2, dtype=torch.bool)
    expected = build_flat_parent_prior(
        parent, geometry_valid, regressed_variant_index=0
    )
    reordered = dict(parent)
    reordered["query_order"] = parent["query_order"].clone()
    reordered["query_order"][:, 3:] = reordered[
        "query_order"
    ][:, 3:].flip(1)
    padding_scores = torch.gather(
        reordered["query_scores"], 1, reordered["query_order"][:, 3:]
    )

    assert torch.isneginf(padding_scores).all()
    assert torch.equal(
        build_flat_parent_prior(
            reordered, geometry_valid, regressed_variant_index=0
        ),
        expected,
    )


@pytest.mark.parametrize("bad_scores", [
    torch.zeros(2),
    torch.zeros(0, 2),
    torch.zeros(1, 0),
    torch.zeros(1, 2, dtype=torch.long),
    torch.tensor([[float("nan"), 1.0]]),
    torch.tensor([[float("inf"), 1.0]]),
    torch.tensor([[-float("inf"), -float("inf")]]),
])
def test_stable_query_order_rejects_malformed_scores(bad_scores):
    with pytest.raises((TypeError, ValueError)):
        stable_query_descending_order(bad_scores)


def test_flat_parent_prior_rejects_extra_finite_query_axis_score():
    parent = build_deployed_parent_state(
        torch.tensor([[1.0, 0.5]]),
        torch.tensor([[1, 3]]),
        torch.ones(1, 2, dtype=torch.bool),
        num_queries=5,
    )
    parent["query_scores"] = parent["query_scores"].clone()
    parent["query_scores"][0, 4] = 0.75
    parent["query_order"] = stable_query_descending_order(
        parent["query_scores"]
    )

    with pytest.raises(ValueError, match="valid parent query indices"):
        build_flat_parent_prior(
            parent,
            torch.ones(1, 2, 2, dtype=torch.bool),
            regressed_variant_index=0,
        )


def test_flat_parent_prior_rejects_query_indices_detached_from_compact_scores():
    parent = build_deployed_parent_state(
        torch.tensor([[2.0, 1.0]]),
        torch.tensor([[1, 3]]),
        torch.ones(1, 2, dtype=torch.bool),
        num_queries=4,
    )
    parent["query_indices"] = torch.tensor([[3, 1]])

    with pytest.raises(ValueError, match="reconstruct"):
        build_flat_parent_prior(
            parent,
            torch.ones(1, 2, 2, dtype=torch.bool),
            regressed_variant_index=0,
        )


@pytest.mark.parametrize("mismatch", ["compact_scores", "query_scores"])
def test_flat_parent_prior_rejects_compact_query_score_mismatch(mismatch):
    parent = build_deployed_parent_state(
        torch.tensor([[2.0, 1.0]]),
        torch.tensor([[1, 3]]),
        torch.ones(1, 2, dtype=torch.bool),
        num_queries=4,
    )
    if mismatch == "compact_scores":
        parent["compact_scores"] = parent["compact_scores"].flip(1)
    else:
        parent["query_scores"] = parent["query_scores"].clone()
        parent["query_scores"][0, [1, 3]] = parent[
            "query_scores"
        ][0, [3, 1]]
        parent["query_order"] = stable_query_descending_order(
            parent["query_scores"]
        )

    with pytest.raises(ValueError, match="reconstruct"):
        build_flat_parent_prior(
            parent,
            torch.ones(1, 2, 2, dtype=torch.bool),
            regressed_variant_index=0,
        )


@pytest.mark.parametrize("bad_compact", [
    "missing",
    "dtype",
    "shape",
    "nonfinite_valid",
])
def test_flat_parent_prior_rejects_malformed_compact_scores(bad_compact):
    parent = build_deployed_parent_state(
        torch.tensor([[2.0, 1.0]]),
        torch.tensor([[1, 3]]),
        torch.ones(1, 2, dtype=torch.bool),
        num_queries=4,
    )
    if bad_compact == "missing":
        del parent["compact_scores"]
    elif bad_compact == "dtype":
        parent["compact_scores"] = parent["compact_scores"].long()
    elif bad_compact == "shape":
        parent["compact_scores"] = parent["compact_scores"][:, :1]
    else:
        parent["compact_scores"] = parent["compact_scores"].clone()
        parent["compact_scores"][0, 0] = float("nan")

    with pytest.raises((TypeError, ValueError)):
        build_flat_parent_prior(
            parent,
            torch.ones(1, 2, 2, dtype=torch.bool),
            regressed_variant_index=0,
        )


def test_flat_parent_prior_rejects_compact_query_score_dtype_mismatch():
    parent = build_deployed_parent_state(
        torch.tensor([[2.0, 1.0]]),
        torch.tensor([[1, 3]]),
        torch.ones(1, 2, dtype=torch.bool),
        num_queries=4,
    )
    parent["compact_scores"] = parent["compact_scores"].double()

    with pytest.raises(TypeError, match="dtype"):
        build_flat_parent_prior(
            parent,
            torch.ones(1, 2, 2, dtype=torch.bool),
            regressed_variant_index=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flat_parent_prior_rejects_compact_score_device_mismatch():
    cpu_parent = build_deployed_parent_state(
        torch.tensor([[2.0, 1.0]]),
        torch.tensor([[1, 3]]),
        torch.ones(1, 2, dtype=torch.bool),
        num_queries=4,
    )
    parent = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in cpu_parent.items()
    }
    parent["compact_scores"] = cpu_parent["compact_scores"]

    with pytest.raises(ValueError, match="device"):
        build_flat_parent_prior(
            parent,
            torch.ones(1, 2, 2, dtype=torch.bool).cuda(),
            regressed_variant_index=0,
        )


@pytest.mark.parametrize("bad_order", [
    "duplicate_padding_index",
    "out_of_range_padding_index",
    "finite_scores_increase",
    "finite_tie_reversed",
    "padding_before_finite",
])
def test_flat_parent_prior_rejects_malformed_query_order(bad_order):
    parent = build_deployed_parent_state(
        torch.tensor([[1.0, 1.0, 0.5]]),
        torch.tensor([[5, 1, 7]]),
        torch.ones(1, 3, dtype=torch.bool),
        num_queries=8,
    )
    parent["query_order"] = parent["query_order"].clone()
    if bad_order == "duplicate_padding_index":
        parent["query_order"][0, -1] = parent["query_order"][0, -2]
    elif bad_order == "out_of_range_padding_index":
        parent["query_order"][0, -1] = 8
    elif bad_order == "finite_scores_increase":
        parent["query_order"][0, 0], parent["query_order"][0, 2] = (
            parent["query_order"][0, 2].clone(),
            parent["query_order"][0, 0].clone(),
        )
    elif bad_order == "finite_tie_reversed":
        parent["query_order"][0, 0], parent["query_order"][0, 1] = (
            parent["query_order"][0, 1].clone(),
            parent["query_order"][0, 0].clone(),
        )
    elif bad_order == "padding_before_finite":
        parent["query_order"][0, 2], parent["query_order"][0, 3] = (
            parent["query_order"][0, 3].clone(),
            parent["query_order"][0, 2].clone(),
        )

    with pytest.raises(ValueError):
        build_flat_parent_prior(
            parent,
            torch.ones(1, 3, 2, dtype=torch.bool),
            regressed_variant_index=0,
        )


@pytest.mark.parametrize("bad_field", [
    "scores_shape",
    "indices_shape",
    "valid_shape",
    "scores_dtype",
    "indices_dtype",
    "valid_dtype",
    "num_queries",
    "index_range",
    "duplicate_valid_query",
    "nonfinite_valid_score",
    "empty_row",
])
def test_parent_state_rejects_malformed_inputs(bad_field):
    compact = torch.tensor([[0.5, 0.4]])
    query_indices = torch.tensor([[0, 1]])
    valid = torch.tensor([[True, True]])
    num_queries = 2

    if bad_field == "scores_shape":
        compact = compact.unsqueeze(-1)
    elif bad_field == "indices_shape":
        query_indices = query_indices[:, :1]
    elif bad_field == "valid_shape":
        valid = valid[:, :1]
    elif bad_field == "scores_dtype":
        compact = compact.long()
    elif bad_field == "indices_dtype":
        query_indices = query_indices.float()
    elif bad_field == "valid_dtype":
        valid = valid.long()
    elif bad_field == "num_queries":
        num_queries = 0
    elif bad_field == "index_range":
        query_indices[0, 1] = num_queries
    elif bad_field == "duplicate_valid_query":
        query_indices[0, 1] = query_indices[0, 0]
    elif bad_field == "nonfinite_valid_score":
        compact[0, 0] = float("nan")
    elif bad_field == "empty_row":
        valid[0] = False

    with pytest.raises((TypeError, ValueError)):
        build_deployed_parent_state(
            compact, query_indices, valid, num_queries
        )


def test_parent_state_allows_nonfinite_padding_and_masks_it_on_query_axis():
    parent = build_deployed_parent_state(
        torch.tensor([[0.5, float("nan")]]),
        torch.tensor([[2, 0]]),
        torch.tensor([[True, False]]),
        num_queries=4,
    )

    assert parent["top1_query_index"].item() == 2
    assert parent["parent_top1_mask"].tolist() == [[True, False]]
    assert torch.isneginf(parent["query_scores"][0, [0, 1, 3]]).all()


def test_flat_parent_prior_ranks_queries_before_expanding_variants():
    parent = _parent_state(
        compact_scores=torch.tensor([[1.0, 1.0]]),
        query_indices=torch.tensor([[5, 1]]),
        num_queries=6,
    )
    geometry_valid = torch.ones(1, 2, 3, dtype=torch.bool)

    prior = build_flat_parent_prior(
        parent, geometry_valid, regressed_variant_index=2
    )
    order = stable_flat_descending_indices(
        prior, geometry_valid.reshape(1, -1)
    )

    # Query 1 precedes tied query 5 on the deployed Q-axis. Within each
    # query, variant 2 is first and variants 0/1 retain index order.
    assert order == ((5, 3, 4, 2, 0, 1),)
    assert torch.isfinite(prior).all()
    assert len(set(prior[0].tolist())) == prior.shape[1]


def test_flat_parent_prior_masks_invalid_geometry_with_negative_infinity():
    parent = _parent_state()
    geometry_valid = torch.tensor([[[True, False], [True, True]]])

    prior = build_flat_parent_prior(
        parent, geometry_valid, regressed_variant_index=0
    )

    flat_valid = geometry_valid.reshape(1, -1)
    assert torch.isfinite(prior[flat_valid]).all()
    assert torch.isneginf(prior[~flat_valid]).all()
    assert len(set(prior[flat_valid].tolist())) == int(flat_valid.sum())


def test_flat_parent_prior_is_float32_and_unique_for_bfloat16_parent_scores():
    candidates, variants = 100, 7
    parent = build_deployed_parent_state(
        torch.arange(candidates, dtype=torch.bfloat16).unsqueeze(0),
        torch.arange(candidates, dtype=torch.long).unsqueeze(0),
        torch.ones(1, candidates, dtype=torch.bool),
        num_queries=candidates,
    )
    geometry_valid = torch.ones(
        1, candidates, variants, dtype=torch.bool
    )

    prior = build_flat_parent_prior(
        parent, geometry_valid, regressed_variant_index=0
    )

    flat_valid = geometry_valid.reshape(1, -1)
    unique_valid_priors = int(torch.unique(prior[flat_valid]).numel())
    assert prior.dtype == torch.float32, (
        "bfloat16 retained only {} of {} priority codes".format(
            unique_valid_priors, candidates * variants
        )
    )
    assert unique_valid_priors == candidates * variants


@pytest.mark.parametrize("bad_field", [
    "geometry_shape",
    "geometry_dtype",
    "variant_index_low",
    "variant_index_high",
    "variant_index_type",
    "valid_geometry_on_padding",
    "invalid_regressed_fallback",
    "duplicate_parent_query",
    "parent_batch_mismatch",
])
def test_flat_parent_prior_rejects_malformed_contracts(bad_field):
    parent = build_deployed_parent_state(
        torch.tensor([[2.0, 1.0]]),
        torch.tensor([[1, 3]]),
        torch.tensor([[True, True]]),
        num_queries=4,
    )
    geometry_valid = torch.ones(1, 2, 2, dtype=torch.bool)
    regressed_variant_index = 0

    if bad_field == "geometry_shape":
        geometry_valid = geometry_valid.reshape(1, -1)
    elif bad_field == "geometry_dtype":
        geometry_valid = geometry_valid.long()
    elif bad_field == "variant_index_low":
        regressed_variant_index = -1
    elif bad_field == "variant_index_high":
        regressed_variant_index = 2
    elif bad_field == "variant_index_type":
        regressed_variant_index = 0.0
    elif bad_field == "valid_geometry_on_padding":
        parent = build_deployed_parent_state(
            torch.tensor([[2.0, float("nan")]]),
            torch.tensor([[1, 0]]),
            torch.tensor([[True, False]]),
            num_queries=4,
        )
    elif bad_field == "invalid_regressed_fallback":
        geometry_valid[0, 1, 0] = False
    elif bad_field == "duplicate_parent_query":
        parent["query_indices"] = torch.tensor([[1, 1]])
    elif bad_field == "parent_batch_mismatch":
        parent["query_indices"] = parent["query_indices"].expand(
            2, -1
        ).clone()
        parent["candidate_valid"] = parent["candidate_valid"].expand(
            2, -1
        ).clone()
        geometry_valid = geometry_valid.expand(2, -1, -1).clone()

    with pytest.raises((TypeError, ValueError)):
        build_flat_parent_prior(
            parent, geometry_valid, regressed_variant_index
        )


def test_zero_geometry_weight_bypasses_flat_prior_exactly(monkeypatch):
    parent = _parent_state()

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("flat parent prior must not be built")

    monkeypatch.setattr(
        rec_geometry_reranker, "build_flat_parent_prior", fail_if_called
    )
    selected = blend_rec_geometry_scores(
        parent,
        learned_logits=torch.zeros(1, 4),
        geometry_valid=torch.ones(1, 2, 2, dtype=torch.bool),
        geometry_weight=0.0,
        regressed_variant_index=0,
    )

    assert selected["use_parent_query_axis"] is True
    assert selected["query_scores"] is parent["query_scores"]
    assert torch.equal(selected["query_scores"], parent["query_scores"])
    assert "flat_scores" not in selected


def test_nonzero_geometry_weight_rank_blends_and_masks_invalid_candidates():
    parent = _parent_state()
    geometry_valid = torch.tensor([[[True, False], [True, True]]])
    learned_logits = torch.tensor([[0.0, float("nan"), 2.0, 1.0]])

    selected = blend_rec_geometry_scores(
        parent,
        learned_logits,
        geometry_valid,
        geometry_weight=0.5,
        regressed_variant_index=0,
    )

    expected = torch.tensor([[0.5, -float("inf"), 0.75, 0.25]])
    assert selected["use_parent_query_axis"] is False
    assert torch.equal(
        selected["flat_valid_mask"], geometry_valid.reshape(1, -1)
    )
    assert torch.allclose(selected["flat_scores"], expected)
    assert torch.isfinite(
        selected["flat_scores"][selected["flat_valid_mask"]]
    ).all()
    assert torch.isneginf(
        selected["flat_scores"][~selected["flat_valid_mask"]]
    ).all()


@pytest.mark.parametrize("weight", [0.5, 1.0])
def test_float16_learned_logits_blend_to_float32_on_cpu(weight):
    parent = _parent_state()
    geometry_valid = torch.tensor([[[True, False], [True, True]]])
    learned_logits = torch.tensor(
        [[3.0, float("nan"), 2.0, 1.0]], dtype=torch.float16
    )

    selected = blend_rec_geometry_scores(
        parent,
        learned_logits,
        geometry_valid,
        geometry_weight=weight,
        regressed_variant_index=0,
    )

    flat_scores = selected["flat_scores"]
    flat_valid = selected["flat_valid_mask"]
    assert flat_scores.dtype == torch.float32
    assert torch.isfinite(flat_scores[flat_valid]).all()
    assert torch.isneginf(flat_scores[~flat_valid]).all()
    assert stable_flat_descending_indices(
        flat_scores, flat_valid
    ) == stable_flat_descending_indices(learned_logits, flat_valid)


def test_wide_bfloat16_learned_ranks_stay_unique_in_float32():
    candidates, variants = 100, 7
    parent = build_deployed_parent_state(
        torch.arange(candidates, dtype=torch.bfloat16).unsqueeze(0),
        torch.arange(candidates, dtype=torch.long).unsqueeze(0),
        torch.ones(1, candidates, dtype=torch.bool),
        num_queries=candidates,
    )
    geometry_valid = torch.ones(
        1, candidates, variants, dtype=torch.bool
    )
    learned_logits = torch.logspace(
        -30, 30, candidates * variants, dtype=torch.float32
    ).to(torch.bfloat16).unsqueeze(0)
    assert int(torch.unique(learned_logits).numel()) == candidates * variants

    selected = blend_rec_geometry_scores(
        parent,
        learned_logits,
        geometry_valid,
        geometry_weight=1.0,
        regressed_variant_index=0,
    )

    flat_scores = selected["flat_scores"]
    flat_valid = selected["flat_valid_mask"]
    assert flat_scores.dtype == torch.float32
    assert int(torch.unique(flat_scores[flat_valid]).numel()) == (
        candidates * variants
    )
    assert stable_flat_descending_indices(
        flat_scores, flat_valid
    ) == (tuple(range(candidates * variants - 1, -1, -1)),)


def test_tied_learned_and_tied_final_scores_choose_lower_flat_index():
    parent = _parent_state(
        compact_scores=torch.tensor([[1.0]]),
        query_indices=torch.tensor([[1]]),
        num_queries=2,
    )
    geometry_valid = torch.ones(1, 1, 2, dtype=torch.bool)

    learned_tie = blend_rec_geometry_scores(
        parent,
        torch.tensor([[4.0, 4.0]]),
        geometry_valid,
        geometry_weight=1.0,
        regressed_variant_index=0,
    )
    final_tie = blend_rec_geometry_scores(
        parent,
        torch.tensor([[0.0, 1.0]]),
        geometry_valid,
        geometry_weight=0.5,
        regressed_variant_index=0,
    )

    assert stable_flat_descending_indices(
        learned_tie["flat_scores"], learned_tie["flat_valid_mask"]
    ) == ((0, 1),)
    assert torch.equal(
        final_tie["flat_scores"], torch.tensor([[0.5, 0.5]])
    )
    assert stable_flat_descending_indices(
        final_tie["flat_scores"], final_tie["flat_valid_mask"]
    ) == ((0, 1),)


@pytest.mark.parametrize("weight", [
    -0.01, 1.01, float("nan"), float("inf"), "not-a-weight",
])
def test_geometry_blend_rejects_invalid_weights(weight):
    with pytest.raises((TypeError, ValueError)):
        blend_rec_geometry_scores(
            _parent_state(),
            torch.zeros(1, 4),
            torch.ones(1, 2, 2, dtype=torch.bool),
            geometry_weight=weight,
            regressed_variant_index=0,
        )


@pytest.mark.parametrize("bad_field", [
    "logit_shape",
    "logit_dtype",
    "nonfinite_valid_logit",
])
def test_nonzero_geometry_blend_rejects_malformed_logits(bad_field):
    logits = torch.zeros(1, 4)
    if bad_field == "logit_shape":
        logits = torch.zeros(1, 2, 2)
    elif bad_field == "logit_dtype":
        logits = logits.long()
    elif bad_field == "nonfinite_valid_logit":
        logits[0, 0] = float("nan")

    with pytest.raises((TypeError, ValueError)):
        blend_rec_geometry_scores(
            _parent_state(),
            logits,
            torch.ones(1, 2, 2, dtype=torch.bool),
            geometry_weight=0.5,
            regressed_variant_index=0,
        )


def test_stable_flat_order_is_lexicographic_and_excludes_invalid_entries():
    scores = torch.tensor([
        [0.5, 2.0, 2.0, 100.0],
        [1.0, 1.0, -1.0, float("nan")],
    ])
    valid = torch.tensor([
        [True, True, True, False],
        [True, True, True, False],
    ])

    order = stable_flat_descending_indices(scores, valid)

    assert order == ((1, 2, 0), (0, 1, 2))


@pytest.mark.parametrize("scores,valid", [
    (torch.zeros(1, 2, 1), torch.ones(1, 2, dtype=torch.bool)),
    (torch.zeros(1, 2), torch.ones(1, 3, dtype=torch.bool)),
    (torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.long)),
    (torch.zeros(1, 2, dtype=torch.long),
     torch.ones(1, 2, dtype=torch.bool)),
    (torch.tensor([[float("nan"), 0.0]]),
     torch.ones(1, 2, dtype=torch.bool)),
    (torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool)),
    (torch.empty(0, 2), torch.empty(0, 2, dtype=torch.bool)),
])
def test_stable_flat_order_rejects_malformed_nonfinite_or_empty_rows(
        scores, valid):
    with pytest.raises((TypeError, ValueError)):
        stable_flat_descending_indices(scores, valid)
