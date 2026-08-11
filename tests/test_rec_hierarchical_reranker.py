import inspect
import math

import pytest
import torch

from models.rec_hierarchical_reranker import (
    HIERARCHICAL_HIDDEN_DIMS,
    HIERARCHICAL_FALSE_POSITIVE_COSTS,
    HIERARCHICAL_BOOTSTRAP_REPLICATES,
    HIERARCHICAL_FOLD_COUNT,
    HIERARCHICAL_MARGIN_PERCENTILES,
    HIERARCHICAL_SEED,
    HIERARCHICAL_WEIGHT_DECAYS,
    HIERARCHICAL_THRESHOLD_WEIGHTS,
    HIERARCHICAL_THRESHOLDS,
    QUERY_AUX_BINARY_DIM,
    QUERY_AUX_CONTINUOUS_DIM,
    QUERY_COUNT,
    QUERY_FEATURE_DIM,
    VARIANT_AUX_BINARY_DIM,
    VARIANT_AUX_CONTINUOUS_DIM,
    VARIANT_COUNT,
    VARIANT_FEATURE_DIM,
    HierarchicalQueryVariantReranker,
    apply_hierarchical_policy,
    build_hierarchical_scene_folds,
    build_hierarchical_targets,
    canonical_hierarchical_scene_fold_sha256,
    choose_hierarchical_configuration,
    compute_hierarchical_loss,
    hierarchical_scene_clustered_hit_delta_bootstrap,
    monotone_hit_probabilities,
    select_hierarchical_proposal,
)


MODEL_INPUT_NAMES = (
    "query_features",
    "variant_features",
    "query_aux_continuous",
    "query_aux_binary",
    "variant_aux_continuous",
    "variant_aux_binary",
    "query_valid",
    "variant_valid",
)


def _valid_inputs(batch_size=2):
    generator = torch.Generator().manual_seed(17)
    inputs = {
        "query_features": torch.randn(
            batch_size, QUERY_COUNT, QUERY_FEATURE_DIM,
            generator=generator,
        ),
        "variant_features": torch.randn(
            batch_size, QUERY_COUNT, VARIANT_COUNT, VARIANT_FEATURE_DIM,
            generator=generator,
        ),
        "query_aux_continuous": torch.randn(
            batch_size, QUERY_COUNT, QUERY_AUX_CONTINUOUS_DIM,
            generator=generator,
        ),
        "query_aux_binary": torch.zeros(
            batch_size, QUERY_COUNT, QUERY_AUX_BINARY_DIM,
            dtype=torch.bool,
        ),
        "variant_aux_continuous": torch.randn(
            batch_size, QUERY_COUNT, VARIANT_COUNT,
            VARIANT_AUX_CONTINUOUS_DIM,
            generator=generator,
        ),
        "variant_aux_binary": torch.zeros(
            batch_size, QUERY_COUNT, VARIANT_COUNT,
            VARIANT_AUX_BINARY_DIM,
            dtype=torch.bool,
        ),
        "query_valid": torch.ones(
            batch_size, QUERY_COUNT, dtype=torch.bool,
        ),
        "variant_valid": torch.ones(
            batch_size, QUERY_COUNT, VARIANT_COUNT, dtype=torch.bool,
        ),
    }
    inputs["query_aux_binary"][:, 0, 0] = True
    inputs["variant_aux_binary"][:, 0, 0, 0] = True
    return inputs


def _forward(model, values):
    return model(**{name: values[name] for name in MODEL_INPUT_NAMES})


def test_hierarchical_constants_are_explicit_and_stable():
    assert QUERY_COUNT == 16
    assert VARIANT_COUNT == 7
    assert QUERY_FEATURE_DIM == 152
    assert VARIANT_FEATURE_DIM == 25
    assert QUERY_AUX_CONTINUOUS_DIM == 4
    assert QUERY_AUX_BINARY_DIM == 2
    assert VARIANT_AUX_CONTINUOUS_DIM == 2
    assert VARIANT_AUX_BINARY_DIM == 2
    assert HIERARCHICAL_THRESHOLDS == (0.25, 0.50)
    assert HIERARCHICAL_THRESHOLD_WEIGHTS == (2.0, 1.0)
    assert HIERARCHICAL_HIDDEN_DIMS == (64, 128)


def test_model_emits_the_fixed_hierarchical_tensor_contract():
    model = HierarchicalQueryVariantReranker(
        hidden_dim=64, dropout=0.1,
    ).eval()

    outputs = _forward(model, _valid_inputs())

    assert set(outputs) == {
        "query_logits",
        "variant_logits",
        "query_embedding",
        "variant_embedding",
    }
    assert outputs["query_logits"].shape == (2, 16, 2)
    assert outputs["variant_logits"].shape == (2, 16, 7, 2)
    assert outputs["query_embedding"].shape == (2, 16, 64)
    assert outputs["variant_embedding"].shape == (2, 16, 7, 64)
    assert all(value.dtype == torch.float32 for value in outputs.values())
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_forward_signature_excludes_training_labels_and_ground_truth():
    parameters = set(inspect.signature(
        HierarchicalQueryVariantReranker.forward
    ).parameters)
    forbidden = {"candidate_ious", "center_label", "size_gts", "gt_masks"}

    assert forbidden.isdisjoint(parameters)

    model = HierarchicalQueryVariantReranker(
        hidden_dim=64, dropout=0.1,
    ).eval()
    inputs = _valid_inputs()
    reference = _forward(model, inputs)
    unrelated = dict(inputs)
    unrelated.update({
        "candidate_ious": torch.rand(2, 16, 7),
        "center_label": object(),
        "size_gts": object(),
        "gt_masks": object(),
    })
    repeated = _forward(model, unrelated)

    for name in reference:
        assert torch.equal(reference[name], repeated[name])
    with pytest.raises(TypeError):
        model(**unrelated)


@pytest.mark.parametrize(
    "field,index",
    [
        ("query_features", (slice(None), slice(None, -1), slice(None))),
        (
            "variant_features",
            (slice(None), slice(None), slice(None, -1), slice(None)),
        ),
        (
            "query_aux_continuous",
            (slice(None), slice(None), slice(None, -1)),
        ),
        (
            "query_aux_binary",
            (slice(None), slice(None), slice(None, -1)),
        ),
        (
            "variant_aux_continuous",
            (slice(None), slice(None), slice(None), slice(None, -1)),
        ),
        (
            "variant_aux_binary",
            (slice(None), slice(None), slice(None), slice(None, -1)),
        ),
        ("query_valid", (slice(None), slice(None, -1))),
        (
            "variant_valid",
            (slice(None), slice(None), slice(None, -1)),
        ),
    ],
)
def test_model_rejects_wrong_contract_shapes(field, index):
    values = _valid_inputs()
    values[field] = values[field][index]
    model = HierarchicalQueryVariantReranker(hidden_dim=64, dropout=0.1)

    with pytest.raises(ValueError, match=field):
        _forward(model, values)


@pytest.mark.parametrize(
    "field",
    [
        "query_features",
        "variant_features",
        "query_aux_continuous",
        "variant_aux_continuous",
    ],
)
@pytest.mark.parametrize("bad_dtype", [torch.float64, torch.int64])
def test_model_requires_float32_feature_tensors(field, bad_dtype):
    values = _valid_inputs()
    values[field] = values[field].to(bad_dtype)
    model = HierarchicalQueryVariantReranker(hidden_dim=64, dropout=0.1)

    with pytest.raises(TypeError, match=field):
        _forward(model, values)


@pytest.mark.parametrize(
    "field",
    [
        "query_aux_binary",
        "variant_aux_binary",
        "query_valid",
        "variant_valid",
    ],
)
def test_model_requires_bool_binary_and_mask_tensors(field):
    values = _valid_inputs()
    values[field] = values[field].to(torch.float32)
    model = HierarchicalQueryVariantReranker(hidden_dim=64, dropout=0.1)

    with pytest.raises(TypeError, match=field):
        _forward(model, values)


@pytest.mark.parametrize(
    "field,index",
    [
        ("query_features", (0, 0, 0)),
        ("variant_features", (0, 0, 0, 0)),
        ("query_aux_continuous", (0, 0, 0)),
        ("variant_aux_continuous", (0, 0, 0, 0)),
    ],
)
def test_model_rejects_nonfinite_values_at_valid_positions(field, index):
    values = _valid_inputs()
    values[field][index] = float("nan")
    model = HierarchicalQueryVariantReranker(hidden_dim=64, dropout=0.1)

    with pytest.raises(ValueError, match=field):
        _forward(model, values)


def test_model_rejects_variant_valid_under_an_invalid_query():
    values = _valid_inputs()
    values["query_valid"][0, 0] = False
    model = HierarchicalQueryVariantReranker(hidden_dim=64, dropout=0.1)

    with pytest.raises(ValueError, match="variant_valid"):
        _forward(model, values)


def test_model_rejects_valid_query_without_any_valid_variant():
    values = _valid_inputs()
    values["variant_valid"][0, 0] = False
    model = HierarchicalQueryVariantReranker(hidden_dim=64, dropout=0.1)

    with pytest.raises(ValueError, match="query_valid"):
        _forward(model, values)


def test_model_rejects_a_row_without_a_valid_query():
    values = _valid_inputs()
    values["query_valid"][0] = False
    values["variant_valid"][0] = False
    model = HierarchicalQueryVariantReranker(hidden_dim=64, dropout=0.1)

    with pytest.raises(ValueError, match="valid query"):
        _forward(model, values)


def test_invalid_padding_is_zero_and_cannot_change_valid_outputs():
    values = _valid_inputs(batch_size=1)
    values["query_valid"][0, -1] = False
    values["variant_valid"][0, -1] = False
    values["variant_valid"][0, 0, -1] = False
    model = HierarchicalQueryVariantReranker(
        hidden_dim=64, dropout=0.1,
    ).eval()

    reference = _forward(model, values)
    padded = {name: value.clone() for name, value in values.items()}
    padded["query_features"][0, -1] = float("nan")
    padded["query_aux_continuous"][0, -1] = float("inf")
    padded["variant_features"][0, -1] = float("nan")
    padded["variant_aux_continuous"][0, -1] = float("inf")
    padded["variant_features"][0, 0, -1] = float("nan")
    padded["variant_aux_continuous"][0, 0, -1] = float("inf")
    repeated = _forward(model, padded)

    for name in reference:
        assert torch.equal(reference[name], repeated[name])
    assert torch.equal(
        repeated["query_embedding"][~values["query_valid"]],
        torch.zeros(1, 64),
    )
    assert torch.equal(
        repeated["query_logits"][~values["query_valid"]],
        torch.zeros(1, 2),
    )
    assert torch.equal(
        repeated["variant_embedding"][~values["variant_valid"]],
        torch.zeros(int((~values["variant_valid"]).sum().item()), 64),
    )
    assert torch.equal(
        repeated["variant_logits"][~values["variant_valid"]],
        torch.zeros(int((~values["variant_valid"]).sum().item()), 2),
    )


def test_monotone_probabilities_are_bounded_without_clipping():
    logits = torch.tensor([
        [[-100.0, 100.0], [0.0, 0.0], [100.0, -100.0]],
    ])

    probabilities = monotone_hit_probabilities(logits)

    assert probabilities.shape == (1, 3, 2)
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)
    assert torch.all(probabilities[..., 1] <= probabilities[..., 0])
    expected25 = logits[..., 0].sigmoid()
    expected50 = expected25 * logits[..., 1].sigmoid()
    assert torch.equal(probabilities[..., 0], expected25)
    assert torch.equal(probabilities[..., 1], expected50)


@pytest.mark.parametrize(
    "bad_logits",
    [
        torch.zeros(2, 16),
        torch.zeros(2, 16, 3),
        torch.zeros(2, 16, 2, dtype=torch.float64),
        torch.zeros(2, 16, 2, dtype=torch.int64),
        torch.tensor([[[float("nan"), 0.0]]]),
    ],
)
def test_monotone_probabilities_reject_malformed_logits(bad_logits):
    with pytest.raises((TypeError, ValueError)):
        monotone_hit_probabilities(bad_logits)


def _hierarchy_inputs(batch_size=1):
    return {
        "query_logits": torch.zeros(batch_size, QUERY_COUNT, 2),
        "variant_logits": torch.zeros(
            batch_size, QUERY_COUNT, VARIANT_COUNT, 2,
        ),
        "query_valid": torch.ones(
            batch_size, QUERY_COUNT, dtype=torch.bool,
        ),
        "variant_valid": torch.ones(
            batch_size, QUERY_COUNT, VARIANT_COUNT, dtype=torch.bool,
        ),
    }


def test_hierarchical_proposal_selects_query_before_variant():
    values = _hierarchy_inputs()
    values["query_logits"][0, 3] = torch.tensor([5.0, 5.0])
    values["query_logits"][0, 9] = torch.tensor([4.0, 4.0])
    values["variant_logits"][0, 9, 6] = torch.tensor([100.0, 100.0])
    values["variant_logits"][0, 3, 2] = torch.tensor([3.0, 2.0])

    selected = select_hierarchical_proposal(**values)

    assert set(selected) == {
        "query_indices",
        "variant_indices",
        "flat_indices",
        "query_utility",
        "variant_utility",
    }
    assert selected["query_indices"].tolist() == [3]
    assert selected["variant_indices"].tolist() == [2]
    assert selected["flat_indices"].tolist() == [3 * 7 + 2]
    assert selected["query_utility"].shape == (1, 16)
    assert selected["variant_utility"].shape == (1, 16, 7)
    assert selected["variant_utility"][0, 9, 6] > selected[
        "variant_utility"
    ][0, 3, 2]


def test_hierarchical_proposal_uses_lowest_indices_for_exact_ties():
    values = _hierarchy_inputs(batch_size=2)
    values["query_valid"][0, :4] = False
    values["variant_valid"][0, :4] = False
    values["variant_valid"][0, 4, :3] = False
    values["query_valid"][1, :11] = False
    values["variant_valid"][1, :11] = False
    values["variant_valid"][1, 11, :5] = False

    selected = select_hierarchical_proposal(**values)

    assert selected["query_indices"].tolist() == [4, 11]
    assert selected["variant_indices"].tolist() == [3, 5]
    assert selected["flat_indices"].tolist() == [31, 82]


def test_hierarchical_proposal_masks_larger_invalid_logits():
    values = _hierarchy_inputs()
    values["query_valid"][0, 0] = False
    values["variant_valid"][0, 0] = False
    values["query_logits"][0, 0] = 100.0
    values["query_logits"][0, 1] = 2.0
    values["variant_valid"][0, 1, 0] = False
    values["variant_logits"][0, 1, 0] = 100.0
    values["variant_logits"][0, 1, 1] = 2.0

    selected = select_hierarchical_proposal(**values)

    assert selected["query_indices"].item() == 1
    assert selected["variant_indices"].item() == 1


@pytest.mark.parametrize(
    "field,mutation",
    [
        ("query_logits", lambda value: value[:, :-1]),
        ("query_logits", lambda value: value[..., :1]),
        ("variant_logits", lambda value: value[:, :, :-1]),
        ("variant_logits", lambda value: value[..., :1]),
        ("query_valid", lambda value: value[:, :-1]),
        ("variant_valid", lambda value: value[:, :, :-1]),
        ("query_logits", lambda value: value.to(torch.float64)),
        ("variant_logits", lambda value: value.to(torch.int64)),
        ("query_valid", lambda value: value.to(torch.float32)),
        ("variant_valid", lambda value: value.to(torch.int64)),
    ],
)
def test_hierarchical_proposal_rejects_malformed_inputs(field, mutation):
    values = _hierarchy_inputs()
    values[field] = mutation(values[field])

    with pytest.raises((TypeError, ValueError), match=field):
        select_hierarchical_proposal(**values)


def test_hierarchical_proposal_rejects_inconsistent_masks():
    values = _hierarchy_inputs()
    values["query_valid"][0, 0] = False

    with pytest.raises(ValueError, match="variant_valid"):
        select_hierarchical_proposal(**values)


@pytest.mark.parametrize("field", ["query_logits", "variant_logits"])
def test_hierarchical_proposal_rejects_nonfinite_logits(field):
    values = _hierarchy_inputs()
    values[field].reshape(-1)[0] = float("nan")

    with pytest.raises(ValueError, match=field):
        select_hierarchical_proposal(**values)


def _sparse_label_inputs():
    candidate_ious = torch.zeros(
        2, QUERY_COUNT, VARIANT_COUNT, dtype=torch.float32,
        requires_grad=True,
    )
    variant_valid = torch.zeros(
        2, QUERY_COUNT, VARIANT_COUNT, dtype=torch.bool,
    )
    variant_valid[0, 0, 0] = True
    variant_valid[0, 1, 0] = True
    variant_valid[1, 0, 0] = True
    with torch.no_grad():
        candidate_ious[0, 0, 0] = 0.30
        candidate_ious[0, 1, 0] = 0.00
        candidate_ious[1, 0, 0] = 0.80
    return candidate_ious, variant_valid


def test_hierarchical_targets_are_detached_strict_and_query_reduced():
    ious = torch.full(
        (1, QUERY_COUNT, VARIANT_COUNT),
        float("nan"),
        dtype=torch.float32,
        requires_grad=True,
    )
    valid = torch.zeros(1, QUERY_COUNT, VARIANT_COUNT, dtype=torch.bool)
    valid[0, 0, :4] = True
    valid[0, 1, 0] = True
    above025 = torch.nextafter(
        torch.tensor(0.25), torch.tensor(float("inf"))
    )
    above050 = torch.nextafter(
        torch.tensor(0.50), torch.tensor(float("inf"))
    )
    with torch.no_grad():
        ious[0, 0, :4] = torch.tensor([
            0.25, above025, 0.50, above050,
        ])
        ious[0, 1, 0] = 0.0

    targets = build_hierarchical_targets(ious, valid)

    assert set(targets) == {
        "query_targets", "variant_targets", "query_valid"
    }
    assert targets["variant_targets"].shape == (1, 16, 7, 2)
    assert targets["query_targets"].shape == (1, 16, 2)
    assert targets["query_valid"].shape == (1, 16)
    assert targets["variant_targets"].dtype == torch.bool
    assert targets["query_targets"].dtype == torch.bool
    assert targets["query_valid"].dtype == torch.bool
    assert not targets["variant_targets"].requires_grad
    assert not targets["query_targets"].requires_grad
    assert targets["variant_targets"][0, 0, :4].tolist() == [
        [False, False],
        [True, False],
        [True, False],
        [True, True],
    ]
    assert targets["query_targets"][0, 0].tolist() == [True, True]
    assert targets["query_targets"][0, 1].tolist() == [False, False]
    assert targets["query_valid"][0, :2].tolist() == [True, True]
    assert not targets["query_valid"][0, 2:].any()
    assert not targets["variant_targets"][~valid].any()


@pytest.mark.parametrize(
    "mutation",
    [
        "iou_dtype",
        "valid_dtype",
        "iou_shape",
        "valid_shape",
        "nonfinite_valid",
        "negative_valid",
        "large_valid",
        "empty_row",
    ],
)
def test_hierarchical_targets_reject_malformed_rows(mutation):
    ious, valid = _sparse_label_inputs()
    ious = ious.detach()
    if mutation == "iou_dtype":
        ious = ious.double()
    elif mutation == "valid_dtype":
        valid = valid.long()
    elif mutation == "iou_shape":
        ious = ious[:, :, :-1]
    elif mutation == "valid_shape":
        valid = valid[:, :-1]
    elif mutation == "nonfinite_valid":
        ious[0, 0, 0] = float("nan")
    elif mutation == "negative_valid":
        ious[0, 0, 0] = -0.01
    elif mutation == "large_valid":
        ious[0, 0, 0] = 1.01
    elif mutation == "empty_row":
        valid[0].zero_()

    with pytest.raises((TypeError, ValueError)):
        build_hierarchical_targets(ious, valid)


def test_hierarchical_loss_is_row_balanced_and_reports_exact_counts():
    ious, variant_valid = _sparse_label_inputs()
    targets = build_hierarchical_targets(ious, variant_valid)
    query_logits = torch.zeros(
        2, QUERY_COUNT, 2, requires_grad=True,
    )
    variant_logits = torch.zeros(
        2, QUERY_COUNT, VARIANT_COUNT, 2, requires_grad=True,
    )

    loss, stats = compute_hierarchical_loss(
        query_logits=query_logits,
        variant_logits=variant_logits,
        query_targets=targets["query_targets"],
        variant_targets=targets["variant_targets"],
        query_valid=targets["query_valid"],
        variant_valid=variant_valid,
        false_positive_cost=4.0,
    )

    loss025 = 1.75 * math.log(2.0)
    loss050 = (
        4.0 * -math.log(0.75) + -math.log(0.25)
    ) / 2.0
    expected_head = (2.0 * loss025 + loss050) / 3.0
    assert torch.allclose(loss, torch.tensor(2.0 * expected_head))
    assert {key: int(value.item()) for key, value in stats.items()} == {
        "query_positive025": 2,
        "query_negative025": 1,
        "query_positive050": 1,
        "query_negative050": 2,
        "variant_positive025": 2,
        "variant_negative025": 1,
        "variant_positive050": 1,
        "variant_negative050": 2,
    }
    loss.backward()
    assert torch.isfinite(query_logits.grad).all()
    assert torch.isfinite(variant_logits.grad).all()


def test_duplicating_variants_does_not_change_query_or_row_loss_weight():
    ious, variant_valid = _sparse_label_inputs()
    base_targets = build_hierarchical_targets(ious, variant_valid)
    query_logits = torch.zeros(2, QUERY_COUNT, 2)
    variant_logits = torch.zeros(2, QUERY_COUNT, VARIANT_COUNT, 2)
    base_loss, _base_stats = compute_hierarchical_loss(
        query_logits,
        variant_logits,
        base_targets["query_targets"],
        base_targets["variant_targets"],
        base_targets["query_valid"],
        variant_valid,
        false_positive_cost=4.0,
    )

    duplicated_ious = ious.detach().clone()
    duplicated_valid = variant_valid.clone()
    duplicated_valid[0, 0, 1:3] = True
    duplicated_ious[0, 0, 1:3] = duplicated_ious[0, 0, 0]
    duplicated_targets = build_hierarchical_targets(
        duplicated_ious, duplicated_valid
    )
    duplicated_loss, stats = compute_hierarchical_loss(
        query_logits,
        variant_logits,
        duplicated_targets["query_targets"],
        duplicated_targets["variant_targets"],
        duplicated_targets["query_valid"],
        duplicated_valid,
        false_positive_cost=4.0,
    )

    assert torch.equal(base_loss, duplicated_loss)
    assert int(stats["variant_positive025"].item()) == 4
    assert int(stats["variant_negative050"].item()) == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "query_logit_shape",
        "variant_logit_shape",
        "query_logit_dtype",
        "variant_logit_dtype",
        "query_target_shape",
        "variant_target_shape",
        "query_target_dtype",
        "variant_target_dtype",
        "query_valid_dtype",
        "variant_valid_dtype",
        "nonfinite_query_logit",
        "nonfinite_variant_logit",
        "invalid_query_target",
        "invalid_variant_target",
        "inconsistent_query_target",
        "bad_false_positive_cost",
    ],
)
def test_hierarchical_loss_rejects_malformed_inputs(mutation):
    ious, variant_valid = _sparse_label_inputs()
    targets = build_hierarchical_targets(ious, variant_valid)
    values = {
        "query_logits": torch.zeros(2, QUERY_COUNT, 2),
        "variant_logits": torch.zeros(
            2, QUERY_COUNT, VARIANT_COUNT, 2,
        ),
        "query_targets": targets["query_targets"].clone(),
        "variant_targets": targets["variant_targets"].clone(),
        "query_valid": targets["query_valid"].clone(),
        "variant_valid": variant_valid.clone(),
        "false_positive_cost": 4.0,
    }
    if mutation == "query_logit_shape":
        values["query_logits"] = values["query_logits"][:, :-1]
    elif mutation == "variant_logit_shape":
        values["variant_logits"] = values["variant_logits"][:, :, :-1]
    elif mutation == "query_logit_dtype":
        values["query_logits"] = values["query_logits"].double()
    elif mutation == "variant_logit_dtype":
        values["variant_logits"] = values["variant_logits"].double()
    elif mutation == "query_target_shape":
        values["query_targets"] = values["query_targets"][:, :-1]
    elif mutation == "variant_target_shape":
        values["variant_targets"] = values["variant_targets"][:, :, :-1]
    elif mutation == "query_target_dtype":
        values["query_targets"] = values["query_targets"].long()
    elif mutation == "variant_target_dtype":
        values["variant_targets"] = values["variant_targets"].long()
    elif mutation == "query_valid_dtype":
        values["query_valid"] = values["query_valid"].long()
    elif mutation == "variant_valid_dtype":
        values["variant_valid"] = values["variant_valid"].long()
    elif mutation == "nonfinite_query_logit":
        values["query_logits"][0, 0, 0] = float("nan")
    elif mutation == "nonfinite_variant_logit":
        values["variant_logits"][0, 0, 0, 0] = float("nan")
    elif mutation == "invalid_query_target":
        values["query_targets"][0, 2, 0] = True
    elif mutation == "invalid_variant_target":
        values["variant_targets"][0, 0, 1, 0] = True
    elif mutation == "inconsistent_query_target":
        values["query_targets"][0, 0, 0] = False
    elif mutation == "bad_false_positive_cost":
        values["false_positive_cost"] = 3.0

    with pytest.raises((TypeError, ValueError)):
        compute_hierarchical_loss(**values)


def test_false_positive_grid_is_fixed():
    assert HIERARCHICAL_FALSE_POSITIVE_COSTS == (2.0, 4.0)


def _policy_inputs(batch_size=5):
    scores = torch.full(
        (batch_size, QUERY_COUNT * VARIANT_COUNT),
        -float("inf"),
        dtype=torch.float32,
    )
    flat_valid = torch.zeros_like(scores, dtype=torch.bool)
    scores[:, 2] = 0.9
    scores[:, 5] = 0.8
    flat_valid[:, 2] = True
    flat_valid[:, 5] = True
    return {
        "base_scores": scores,
        "proposed_flat_indices": torch.tensor(
            [2, 6, 5, 5, 5], dtype=torch.long,
        )[:batch_size],
        "predicted_gain": torch.tensor(
            [1.0, 1.0, 0.0, 0.4, 0.5], dtype=torch.float32,
        )[:batch_size],
        "variant_valid": flat_valid.reshape(
            batch_size, QUERY_COUNT, VARIANT_COUNT
        ),
        "margin": 0.5,
    }


def test_hierarchical_policy_abstains_by_default_and_promotes_one_score():
    values = _policy_inputs()

    selected = apply_hierarchical_policy(**values)

    assert set(selected) == {
        "scores", "selected_indices", "switch_mask", "baseline_indices"
    }
    assert selected["baseline_indices"].tolist() == [2, 2, 2, 2, 2]
    assert selected["selected_indices"].tolist() == [2, 2, 2, 2, 5]
    assert selected["switch_mask"].tolist() == [
        False, False, False, False, True,
    ]
    assert torch.equal(selected["scores"][:4], values["base_scores"][:4])
    assert selected["scores"][4, 5] == torch.nextafter(
        torch.tensor(0.9), torch.tensor(float("inf"))
    )
    unchanged = torch.ones(112, dtype=torch.bool)
    unchanged[5] = False
    assert torch.equal(
        selected["scores"][4, unchanged],
        values["base_scores"][4, unchanged],
    )


def test_hierarchical_policy_infinite_margin_is_no_switch():
    values = _policy_inputs(batch_size=1)
    values["proposed_flat_indices"][0] = 5
    values["predicted_gain"][0] = 100.0
    values["margin"] = float("inf")

    selected = apply_hierarchical_policy(**values)

    assert not selected["switch_mask"].item()
    assert torch.equal(selected["scores"], values["base_scores"])


@pytest.mark.parametrize(
    "mutation",
    [
        "score_shape",
        "score_dtype",
        "index_shape",
        "index_dtype",
        "gain_shape",
        "gain_dtype",
        "valid_shape",
        "valid_dtype",
        "nan_score",
        "positive_infinite_score",
        "empty_score_row",
        "valid_score_mismatch",
        "index_below_range",
        "index_above_range",
        "nonfinite_gain",
        "negative_margin",
    ],
)
def test_hierarchical_policy_rejects_malformed_inputs(mutation):
    values = _policy_inputs(batch_size=1)
    if mutation == "score_shape":
        values["base_scores"] = values["base_scores"][:, :-1]
    elif mutation == "score_dtype":
        values["base_scores"] = values["base_scores"].double()
    elif mutation == "index_shape":
        values["proposed_flat_indices"] = values[
            "proposed_flat_indices"
        ].unsqueeze(1)
    elif mutation == "index_dtype":
        values["proposed_flat_indices"] = values[
            "proposed_flat_indices"
        ].int()
    elif mutation == "gain_shape":
        values["predicted_gain"] = values["predicted_gain"].unsqueeze(1)
    elif mutation == "gain_dtype":
        values["predicted_gain"] = values["predicted_gain"].double()
    elif mutation == "valid_shape":
        values["variant_valid"] = values["variant_valid"][:, :, :-1]
    elif mutation == "valid_dtype":
        values["variant_valid"] = values["variant_valid"].long()
    elif mutation == "nan_score":
        values["base_scores"][0, 2] = float("nan")
    elif mutation == "positive_infinite_score":
        values["base_scores"][0, 2] = float("inf")
    elif mutation == "empty_score_row":
        values["base_scores"].fill_(-float("inf"))
        values["variant_valid"].zero_()
    elif mutation == "valid_score_mismatch":
        values["variant_valid"].reshape(1, -1)[0, 5] = False
    elif mutation == "index_below_range":
        values["proposed_flat_indices"][0] = -1
    elif mutation == "index_above_range":
        values["proposed_flat_indices"][0] = 112
    elif mutation == "nonfinite_gain":
        values["predicted_gain"][0] = float("nan")
    elif mutation == "negative_margin":
        values["margin"] = -0.1

    with pytest.raises((TypeError, ValueError)):
        apply_hierarchical_policy(**values)


def test_hierarchical_scene_folds_are_stable_and_scene_disjoint():
    scan_ids = []
    for index in range(17):
        scan_ids.extend([
            "scene{:02d}".format(index)
        ] * (index % 4 + 1))

    mapping = build_hierarchical_scene_folds(scan_ids)
    reordered = build_hierarchical_scene_folds(list(reversed(scan_ids)))

    assert HIERARCHICAL_FOLD_COUNT == 5
    assert HIERARCHICAL_SEED == 0
    assert mapping == reordered
    assert set(mapping.values()) == set(range(5))
    assert mapping == {
        "scene00": 4,
        "scene01": 0,
        "scene02": 0,
        "scene03": 2,
        "scene04": 3,
        "scene05": 2,
        "scene06": 0,
        "scene07": 1,
        "scene08": 2,
        "scene09": 1,
        "scene10": 4,
        "scene11": 3,
        "scene12": 1,
        "scene13": 0,
        "scene14": 1,
        "scene15": 4,
        "scene16": 3,
    }
    assert canonical_hierarchical_scene_fold_sha256(mapping) == (
        "1f9e633a2d7cc00ee73ab73e2544e986"
        "ada9b8702ef4e1415d0424bda7d0f064"
    )


def _hit_bits_for_hierarchical_scene_deltas(scene_deltas):
    scan_ids = []
    baseline = []
    proposed = []
    for scene_index, delta in enumerate(scene_deltas):
        for row_index in range(4):
            scan_ids.append("cluster{:02d}".format(scene_index))
            if delta >= 0:
                baseline.append(0)
                proposed.append(int(row_index < delta))
            else:
                baseline.append(int(row_index < -delta))
                proposed.append(0)
    return scan_ids, baseline, proposed


def test_hierarchical_scene_bootstrap_matches_fixed_audited_protocol():
    scan_ids, baseline, proposed = \
        _hit_bits_for_hierarchical_scene_deltas([3, 2, 1, 0, -1, -2])

    first = hierarchical_scene_clustered_hit_delta_bootstrap(
        scan_ids, baseline, proposed
    )
    second = hierarchical_scene_clustered_hit_delta_bootstrap(
        list(reversed(scan_ids)),
        list(reversed(baseline)),
        list(reversed(proposed)),
    )

    assert HIERARCHICAL_BOOTSTRAP_REPLICATES == 10000
    assert first == second
    assert first == {
        "confidence": 0.95,
        "delta_hits": 3,
        "lower_bound_95": -4,
        "replicates": 10000,
        "scene_count": 6,
        "seed": 0,
    }


def _hierarchical_selection_candidate(
        percentile, margin, deltas025, deltas050=None, hidden_dim=64,
        weight_decay=1e-3, false_positive_cost=4.0):
    if deltas050 is None:
        deltas050 = [0] * len(deltas025)
    scan_ids = []
    baseline025 = []
    proposed025 = []
    baseline050 = []
    proposed050 = []
    for scene_index, (delta025, delta050) in enumerate(zip(
            deltas025, deltas050)):
        assert -2 <= delta025 <= 2
        assert -1 <= delta050 <= 2
        for row_index in range(4):
            scan_ids.append("cluster{:02d}".format(scene_index))
            baseline025.append(int(row_index < 2))
            baseline050.append(int(row_index < 1))
            proposed025.append(int(row_index < 2 + delta025))
            proposed050.append(int(row_index < 1 + delta050))
    switch_bits = [
        int(before025 != after025 or before050 != after050)
        for before025, after025, before050, after050 in zip(
            baseline025, proposed025, baseline050, proposed050
        )
    ]
    return {
        "hidden_dim": hidden_dim,
        "weight_decay": weight_decay,
        "false_positive_cost": false_positive_cost,
        "margin_percentile": percentile,
        "margin": margin,
        "scan_ids": scan_ids,
        "baseline_hits025": baseline025,
        "proposed_hits025": proposed025,
        "baseline_hits050": baseline050,
        "proposed_hits050": proposed050,
        "switch_bits": switch_bits,
    }


def test_hierarchical_selector_uses_fixed_grid_and_nonnegative_bounds():
    assert HIERARCHICAL_WEIGHT_DECAYS == (1e-4, 1e-3)
    assert HIERARCHICAL_MARGIN_PERCENTILES == (
        50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 97.5, 99.0,
    )
    lower_bound_zero = _hierarchical_selection_candidate(
        90.0, 0.5, [1, 0, 0, 0, 0, 0]
    )

    choice = choose_hierarchical_configuration([lower_bound_zero])

    assert choice["eligible"] is True
    assert choice["selected"] == "hierarchical"
    assert choice["delta_hits025"] == 1
    assert choice["delta_hits050"] == 0
    assert choice["bootstrap025"]["lower_bound_95"] == 0
    assert choice["bootstrap050"]["lower_bound_95"] == 0
    assert all(
        record["hits025"] >= 0 and record["hits050"] >= 0
        for record in choice["fold_deltas"].values()
    )


def test_hierarchical_selector_rejects_fold_or_cluster_regressions():
    fold_regression = _hierarchical_selection_candidate(
        90.0, 0.5, [2, -1, 2, 2, 2, 2]
    )
    negative_lower_bound = _hierarchical_selection_candidate(
        95.0, 0.6, [0, 0, 0, 2, -1, 1]
    )

    choice = choose_hierarchical_configuration([
        fold_regression, negative_lower_bound,
    ])

    assert choice["eligible"] is False
    assert choice["selected"] == "baseline"
    assert choice["reason"] == "no-eligible-configuration"
    assert choice["eligible_candidate_count"] == 0
    for record in choice["candidate_diagnostics"]:
        assert record["failed_predicates"]
        assert record["eligible"] is False


def test_hierarchical_selector_uses_declared_tie_break_order():
    common = [1, 1, 1, 1, 1, 1]
    candidates = [
        _hierarchical_selection_candidate(
            95.0, 0.6, common, hidden_dim=128,
            weight_decay=1e-3, false_positive_cost=4.0,
        ),
        _hierarchical_selection_candidate(
            95.0, 0.6, common, hidden_dim=64,
            weight_decay=1e-4, false_positive_cost=4.0,
        ),
        _hierarchical_selection_candidate(
            95.0, 0.6, common, hidden_dim=64,
            weight_decay=1e-3, false_positive_cost=2.0,
        ),
        _hierarchical_selection_candidate(
            95.0, 0.6, common, hidden_dim=64,
            weight_decay=1e-3, false_positive_cost=4.0,
        ),
    ]

    choice = choose_hierarchical_configuration(candidates)

    assert choice["hidden_dim"] == 64
    assert choice["weight_decay"] == 1e-3
    assert choice["false_positive_cost"] == 4.0


def test_hierarchical_selector_prefers_larger_margin_then_fewer_switches():
    common = [1, 1, 1, 1, 1, 1]
    smaller = _hierarchical_selection_candidate(90.0, 0.5, common)
    larger = _hierarchical_selection_candidate(95.0, 0.6, common)

    choice = choose_hierarchical_configuration([smaller, larger])

    assert choice["margin_percentile"] == 95.0
    assert choice["margin"] == 0.6


def test_hierarchical_selector_prefers_fewer_switches_after_margin_tie():
    common = [1, 1, 1, 1, 1, 1]
    more_switches = _hierarchical_selection_candidate(95.0, 0.6, common)
    fewer_switches = _hierarchical_selection_candidate(95.0, 0.6, common)
    for row_index in range(0, len(more_switches["switch_bits"]), 4):
        assert more_switches["switch_bits"][row_index] == 0
        more_switches["switch_bits"][row_index] = 1

    choice = choose_hierarchical_configuration([
        more_switches, fewer_switches,
    ])

    assert choice["switches"] == sum(fewer_switches["switch_bits"])
    assert choice["switches"] < sum(more_switches["switch_bits"])


def test_hierarchical_no_switch_sentinel_is_diagnostic_only():
    sentinel = _hierarchical_selection_candidate(
        None, float("inf"), [0, 0, 0, 0, 0, 0]
    )

    choice = choose_hierarchical_configuration([sentinel])

    assert choice["eligible"] is False
    assert choice["selected"] == "baseline"
    assert choice["candidate_diagnostics"][0]["no_switch"] is True
    assert choice["candidate_diagnostics"][0]["switches"] == 0


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("hidden_dim", 32),
        ("weight_decay", 1e-2),
        ("false_positive_cost", 8.0),
        ("margin_percentile", 55.0),
        ("margin", -0.1),
        ("switch_bits", [2] * 24),
        ("proposed_hits025", [0] * 23),
    ],
)
def test_hierarchical_selector_rejects_noncontract_candidate(
        field, bad_value):
    candidate = _hierarchical_selection_candidate(
        95.0, 0.6, [1, 1, 1, 1, 1, 1]
    )
    candidate[field] = bad_value

    with pytest.raises((TypeError, ValueError)):
        choose_hierarchical_configuration([candidate])


def test_hierarchical_selector_requires_identical_oof_baseline_rows():
    first = _hierarchical_selection_candidate(
        90.0, 0.5, [1, 1, 1, 1, 1, 1]
    )
    second = _hierarchical_selection_candidate(
        95.0, 0.6, [1, 1, 1, 1, 1, 1]
    )
    second["baseline_hits025"][2] = 1

    with pytest.raises(ValueError, match="same OOF baseline"):
        choose_hierarchical_configuration([first, second])
