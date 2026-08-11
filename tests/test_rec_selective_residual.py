import inspect
import math

import pytest
import torch

from models.rec_selective_residual import (
    PAIR_FEATURE_DIM,
    RESIDUAL_BREAK_COSTS,
    RESIDUAL_CLASS_NAMES,
    RESIDUAL_HEAD_WEIGHTS,
    RESIDUAL_HIDDEN_DIMS,
    RESIDUAL_MARGIN_PERCENTILES,
    RESIDUAL_THRESHOLDS,
    RESIDUAL_WEIGHT_DECAYS,
    SelectiveResidualModel,
    apply_selective_policy,
    build_residual_scene_folds,
    build_selective_pair_features,
    build_selective_pair_targets,
    canonical_scene_fold_sha256,
    choose_selective_configuration,
    compute_selective_residual_loss,
    expected_selective_gain,
    scene_clustered_hit_delta_bootstrap,
)


def _pair_inputs(batch_size=2, candidate_count=112):
    feature_values = torch.arange(
        batch_size * candidate_count * 179, dtype=torch.float32
    ).reshape(batch_size, candidate_count, 179)
    feature_values = feature_values / float(feature_values.numel())
    valid = torch.ones(batch_size, candidate_count, dtype=torch.bool)
    valid[0, -1] = False
    baseline = torch.tensor([3, 4], dtype=torch.long)[:batch_size]
    parent_rank = torch.arange(
        candidate_count, dtype=torch.float32
    ).unsqueeze(0).expand(batch_size, -1).clone()
    geometry_rank = parent_rank.flip(1).clone()
    threshold_logits = torch.stack(
        (parent_rank / 10.0, geometry_rank / 10.0), dim=-1
    )
    iou_estimate = parent_rank / float(candidate_count)
    query_positions = torch.arange(
        candidate_count, dtype=torch.long
    ).div(7, rounding_mode="floor").unsqueeze(0).expand(
        batch_size, -1
    ).clone()
    return {
        "normalized_features": feature_values,
        "valid_mask": valid,
        "baseline_indices": baseline,
        "parent_rank": parent_rank,
        "geometry_rank": geometry_rank,
        "threshold_logits": threshold_logits,
        "iou_estimate": iou_estimate,
        "query_positions": query_positions,
    }


def test_pair_feature_contract_has_exact_shape_values_and_mask():
    inputs = _pair_inputs()

    pair = build_selective_pair_features(**inputs)

    assert PAIR_FEATURE_DIM == 185
    assert RESIDUAL_THRESHOLDS == (0.25, 0.50)
    assert RESIDUAL_HEAD_WEIGHTS == (2.0, 1.0)
    assert RESIDUAL_CLASS_NAMES == ("break", "neutral", "fix")
    assert set(pair) == {"features", "valid_mask", "baseline_indices"}
    assert pair["features"].shape == (2, 112, 185)
    assert pair["valid_mask"].shape == (2, 112)
    assert pair["features"].dtype == torch.float32
    assert pair["valid_mask"].dtype == torch.bool
    assert pair["baseline_indices"].dtype == torch.long
    assert not pair["valid_mask"][0, inputs["baseline_indices"][0]]
    assert not pair["valid_mask"][1, inputs["baseline_indices"][1]]

    row = 0
    candidate = 5
    baseline = int(inputs["baseline_indices"][row].item())
    expected = torch.cat([
        inputs["normalized_features"][row, candidate]
        - inputs["normalized_features"][row, baseline],
        (inputs["parent_rank"][row, candidate]
         - inputs["parent_rank"][row, baseline]).reshape(1),
        (inputs["geometry_rank"][row, candidate]
         - inputs["geometry_rank"][row, baseline]).reshape(1),
        inputs["threshold_logits"][row, candidate].sigmoid()
        - inputs["threshold_logits"][row, baseline].sigmoid(),
        (inputs["iou_estimate"][row, candidate]
         - inputs["iou_estimate"][row, baseline]).reshape(1),
        inputs["query_positions"][row, candidate].eq(
            inputs["query_positions"][row, baseline]
        ).to(torch.float32).reshape(1),
    ])
    assert torch.equal(pair["features"][row, candidate], expected)
    assert torch.equal(
        pair["features"][~pair["valid_mask"]],
        torch.zeros_like(pair["features"][~pair["valid_mask"]]),
    )


def test_pair_features_ignore_invalid_nonfinite_values():
    inputs = _pair_inputs()
    inputs["normalized_features"][0, -1] = float("nan")
    inputs["parent_rank"][0, -1] = float("inf")
    inputs["geometry_rank"][0, -1] = -float("inf")
    inputs["threshold_logits"][0, -1] = float("nan")
    inputs["iou_estimate"][0, -1] = float("nan")

    pair = build_selective_pair_features(**inputs)

    assert torch.equal(pair["features"][0, -1], torch.zeros(185))
    assert torch.isfinite(pair["features"]).all()


def test_pair_feature_builder_has_no_ground_truth_input_or_dependency():
    inputs = _pair_inputs()
    unrelated = {
        "center_label": object(),
        "size_gts": object(),
        "candidate_ious": object(),
        "target_mask": object(),
    }
    before = build_selective_pair_features(**inputs)

    unrelated.update({key: object() for key in unrelated})
    after = build_selective_pair_features(**inputs)

    forbidden = set(unrelated)
    parameters = set(inspect.signature(build_selective_pair_features).parameters)
    assert forbidden.isdisjoint(parameters)
    assert torch.equal(before["features"], after["features"])
    assert torch.equal(before["valid_mask"], after["valid_mask"])
    with pytest.raises(TypeError):
        build_selective_pair_features(
            **dict(inputs, center_label=torch.zeros(1))
        )


@pytest.mark.parametrize(
    "field,transform",
    [
        ("normalized_features", lambda value: value.double()),
        ("normalized_features", lambda value: value[:, :, :-1]),
        ("valid_mask", lambda value: value.long()),
        ("valid_mask", lambda value: value[:, :-1]),
        ("baseline_indices", lambda value: value.int()),
        ("parent_rank", lambda value: value.double()),
        ("parent_rank", lambda value: value[:, :-1]),
        ("geometry_rank", lambda value: value[:, :-1]),
        ("threshold_logits", lambda value: value[:, :, :1]),
        ("iou_estimate", lambda value: value[:, :-1]),
        ("query_positions", lambda value: value.int()),
        ("query_positions", lambda value: value[:, :-1]),
    ],
)
def test_pair_features_reject_bad_dtype_or_shape(field, transform):
    inputs = _pair_inputs()
    inputs[field] = transform(inputs[field])

    with pytest.raises((TypeError, ValueError)):
        build_selective_pair_features(**inputs)


@pytest.mark.parametrize(
    "mutation",
    [
        "nonfinite_valid_feature",
        "nonfinite_valid_rank",
        "baseline_out_of_range",
        "baseline_invalid",
        "empty_valid_row",
    ],
)
def test_pair_features_reject_invalid_values_or_baselines(mutation):
    inputs = _pair_inputs()
    if mutation == "nonfinite_valid_feature":
        inputs["normalized_features"][0, 0, 0] = float("nan")
    elif mutation == "nonfinite_valid_rank":
        inputs["geometry_rank"][0, 0] = float("inf")
    elif mutation == "baseline_out_of_range":
        inputs["baseline_indices"][0] = 112
    elif mutation == "baseline_invalid":
        inputs["valid_mask"][0, inputs["baseline_indices"][0]] = False
    elif mutation == "empty_valid_row":
        inputs["valid_mask"][0].zero_()

    with pytest.raises(ValueError):
        build_selective_pair_features(**inputs)


def test_pair_targets_use_detached_strict_signed_tiers():
    below_or_equal_025 = torch.tensor(0.25, dtype=torch.float32)
    above_025 = torch.nextafter(
        below_or_equal_025, torch.tensor(float("inf"))
    )
    above_050 = torch.nextafter(
        torch.tensor(0.50), torch.tensor(float("inf"))
    )
    candidate_ious = torch.tensor([
        [0.25, 0.0, 0.25, 0.0],
        [0.0, 0.50, 0.75, 1.0],
    ], dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        candidate_ious[0, 1] = above_025
        candidate_ious[1, 0] = above_050
    valid = torch.tensor([
        [True, True, True, False],
        [True, True, True, False],
    ])
    baseline = torch.tensor([0, 0], dtype=torch.long)

    targets = build_selective_pair_targets(
        candidate_ious, valid, baseline, thresholds=(0.25, 0.50)
    )

    assert targets.shape == (2, 4, 2)
    assert targets.dtype == torch.long
    assert not targets.requires_grad
    assert set(targets.unique().tolist()) <= {0, 1, 2}
    assert targets[0, 1].tolist() == [2, 1]
    assert targets[0, 2].tolist() == [1, 1]
    assert targets[1, 1].tolist() == [1, 0]
    assert targets[1, 2].tolist() == [1, 1]
    assert targets[:, 0].eq(1).all()
    assert targets[:, 3].eq(1).all()


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_iou_dtype",
        "wrong_valid_dtype",
        "wrong_index_dtype",
        "bad_shape",
        "invalid_baseline",
        "nan_iou",
        "negative_iou",
        "large_iou",
        "bad_thresholds",
    ],
)
def test_pair_targets_reject_malformed_inputs(mutation):
    ious = torch.zeros(2, 3, dtype=torch.float32)
    valid = torch.ones(2, 3, dtype=torch.bool)
    baseline = torch.tensor([0, 1], dtype=torch.long)
    thresholds = (0.25, 0.50)
    if mutation == "wrong_iou_dtype":
        ious = ious.double()
    elif mutation == "wrong_valid_dtype":
        valid = valid.long()
    elif mutation == "wrong_index_dtype":
        baseline = baseline.int()
    elif mutation == "bad_shape":
        valid = valid[:, :-1]
    elif mutation == "invalid_baseline":
        valid[0, 0] = False
    elif mutation == "nan_iou":
        ious[0, 0] = float("nan")
    elif mutation == "negative_iou":
        ious[0, 0] = -0.01
    elif mutation == "large_iou":
        ious[0, 0] = 1.01
    elif mutation == "bad_thresholds":
        thresholds = (0.50, 0.25)

    with pytest.raises((TypeError, ValueError)):
        build_selective_pair_targets(
            ious, valid, baseline, thresholds=thresholds
        )


@pytest.mark.parametrize("hidden_dim", [0, 64])
def test_residual_model_has_zero_initialized_two_head_output(hidden_dim):
    model = SelectiveResidualModel(
        input_dim=185, hidden_dim=hidden_dim, dropout=0.1
    )
    features = torch.randn(2, 5, 185)
    valid = torch.tensor([
        [True, True, False, False, False],
        [True, False, True, False, True],
    ])

    logits = model(features, valid)

    assert logits.shape == (2, 5, 2, 3)
    assert torch.equal(logits, torch.zeros_like(logits))
    assert torch.equal(model.head.weight, torch.zeros_like(model.head.weight))
    assert torch.equal(model.head.bias, torch.zeros_like(model.head.bias))
    if hidden_dim == 0:
        assert isinstance(model.encoder, torch.nn.Identity)
        assert model.head.in_features == 185
    else:
        assert model.head.in_features == 64


def test_residual_model_zero_fills_invalid_logits_after_training():
    model = SelectiveResidualModel(hidden_dim=0)
    with torch.no_grad():
        model.head.bias.fill_(2.0)
    features = torch.randn(1, 3, 185)
    valid = torch.tensor([[True, False, True]])

    logits = model(features, valid)

    assert logits[0, 0].eq(2.0).all()
    assert logits[0, 1].eq(0.0).all()


def test_selective_loss_is_row_balanced_and_reports_class_counts():
    logits = torch.zeros(2, 4, 2, 3, requires_grad=True)
    pair_valid = torch.tensor([
        [True, False, False, False],
        [True, True, True, False],
    ])
    targets = torch.ones(2, 4, 2, dtype=torch.long)
    targets[0, 0, 0] = 2
    targets[1, :3, 0] = 0

    loss, stats = compute_selective_residual_loss(
        logits, targets, pair_valid, break_cost=4.0
    )

    assert torch.allclose(loss, torch.tensor(2.0 * math.log(3.0)))
    assert {key: int(value.item()) for key, value in stats.items()} == {
        "informative_rows": 2,
        "break025": 3,
        "neutral025": 0,
        "fix025": 1,
        "break050": 0,
        "neutral050": 4,
        "fix050": 0,
    }
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_selective_loss_returns_differentiable_zero_without_alternatives():
    logits = torch.randn(2, 3, 2, 3, requires_grad=True)
    targets = torch.ones(2, 3, 2, dtype=torch.long)
    pair_valid = torch.zeros(2, 3, dtype=torch.bool)

    loss, stats = compute_selective_residual_loss(
        logits, targets, pair_valid, break_cost=2.0
    )

    assert loss.item() == 0.0
    assert all(value.item() == 0 for value in stats.values())
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize(
    "mutation",
    [
        "bad_logits_shape",
        "bad_logits_dtype",
        "nonfinite_logits",
        "bad_target_shape",
        "bad_target_dtype",
        "bad_target_class",
        "bad_valid_dtype",
        "bad_break_cost",
        "bad_weights",
    ],
)
def test_selective_loss_rejects_malformed_inputs(mutation):
    logits = torch.zeros(1, 2, 2, 3)
    targets = torch.ones(1, 2, 2, dtype=torch.long)
    valid = torch.tensor([[True, False]])
    break_cost = 2.0
    weights = (2.0, 1.0)
    if mutation == "bad_logits_shape":
        logits = logits[:, :, :, :2]
    elif mutation == "bad_logits_dtype":
        logits = logits.double()
    elif mutation == "nonfinite_logits":
        logits[0, 0, 0, 0] = float("nan")
    elif mutation == "bad_target_shape":
        targets = targets[:, :, :1]
    elif mutation == "bad_target_dtype":
        targets = targets.int()
    elif mutation == "bad_target_class":
        targets[0, 0, 0] = 3
    elif mutation == "bad_valid_dtype":
        valid = valid.long()
    elif mutation == "bad_break_cost":
        break_cost = 0.0
    elif mutation == "bad_weights":
        weights = (1.0,)

    with pytest.raises((TypeError, ValueError)):
        compute_selective_residual_loss(
            logits,
            targets,
            valid,
            break_cost=break_cost,
            threshold_weights=weights,
        )


def test_expected_gain_is_weighted_fix_minus_break_probability():
    probabilities = torch.tensor([[[
        [0.1, 0.2, 0.7],
        [0.4, 0.1, 0.5],
    ], [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ]]], dtype=torch.float32)

    gain = expected_selective_gain(probabilities.log())

    assert gain.shape == (1, 2)
    assert torch.allclose(gain, torch.tensor([[1.3, 0.0]]), atol=1e-6)


def test_selective_policy_promotes_stable_best_and_abstains_exactly():
    base_scores = torch.tensor([
        [0.9, 0.8, -float("inf"), 0.7],
        [0.2, 0.3, 0.1, -float("inf")],
    ])
    pair_gain = torch.tensor([
        [0.0, 0.4, 99.0, 0.4],
        [0.2, 0.0, 0.1, 99.0],
    ])
    pair_valid = torch.tensor([
        [False, True, False, True],
        [True, False, True, False],
    ])

    selected = apply_selective_policy(
        base_scores, pair_gain, pair_valid, margin=0.4
    )

    assert set(selected) == {
        "scores", "selected_indices", "switch_mask", "baseline_indices"
    }
    assert selected["baseline_indices"].tolist() == [0, 1]
    assert selected["selected_indices"].tolist() == [1, 1]
    assert selected["switch_mask"].tolist() == [True, False]
    assert selected["scores"][0, 1] == torch.nextafter(
        base_scores[0, 0], torch.tensor(float("inf"))
    )
    assert torch.equal(selected["scores"][0, [0, 2, 3]],
                       base_scores[0, [0, 2, 3]])
    assert torch.equal(selected["scores"][1], base_scores[1])
    assert torch.isneginf(selected["scores"][0, 2])
    assert torch.isneginf(selected["scores"][1, 3])


def test_selective_policy_nonpositive_or_infinite_margin_is_no_switch():
    base_scores = torch.tensor([[1.0, 0.5, -float("inf")]])
    pair_valid = torch.tensor([[False, True, False]])

    zero_gain = apply_selective_policy(
        base_scores, torch.tensor([[0.0, 0.0, 1.0]]), pair_valid, margin=0.0
    )
    sentinel = apply_selective_policy(
        base_scores, torch.tensor([[0.0, 1.0, 1.0]]), pair_valid,
        margin=float("inf")
    )

    assert not zero_gain["switch_mask"].item()
    assert not sentinel["switch_mask"].item()
    assert torch.equal(zero_gain["scores"], base_scores)
    assert torch.equal(sentinel["scores"], base_scores)


@pytest.mark.parametrize(
    "mutation",
    [
        "bad_score_dtype",
        "bad_gain_dtype",
        "bad_mask_dtype",
        "shape_mismatch",
        "nan_score",
        "positive_infinite_score",
        "empty_score_row",
        "pair_on_invalid_score",
        "baseline_pair_valid",
        "nonfinite_valid_gain",
        "negative_margin",
    ],
)
def test_selective_policy_rejects_malformed_inputs(mutation):
    scores = torch.tensor([[1.0, 0.5, -float("inf")]])
    gain = torch.tensor([[0.0, 0.2, 0.0]])
    valid = torch.tensor([[False, True, False]])
    margin = 0.1
    if mutation == "bad_score_dtype":
        scores = scores.double()
    elif mutation == "bad_gain_dtype":
        gain = gain.double()
    elif mutation == "bad_mask_dtype":
        valid = valid.long()
    elif mutation == "shape_mismatch":
        gain = gain[:, :-1]
    elif mutation == "nan_score":
        scores[0, 1] = float("nan")
    elif mutation == "positive_infinite_score":
        scores[0, 1] = float("inf")
    elif mutation == "empty_score_row":
        scores.fill_(-float("inf"))
    elif mutation == "pair_on_invalid_score":
        valid[0, 2] = True
    elif mutation == "baseline_pair_valid":
        valid[0, 0] = True
    elif mutation == "nonfinite_valid_gain":
        gain[0, 1] = float("nan")
    elif mutation == "negative_margin":
        margin = -0.1

    with pytest.raises((TypeError, ValueError)):
        apply_selective_policy(scores, gain, valid, margin)


def test_scene_folds_are_exact_scene_disjoint_and_order_independent():
    unique_scenes = ["scene{:02d}".format(index) for index in range(17)]
    scan_ids = []
    for index, scene_id in enumerate(unique_scenes):
        scan_ids.extend([scene_id] * (index % 4 + 1))

    mapping = build_residual_scene_folds(scan_ids, fold_count=5, seed=0)
    reordered = build_residual_scene_folds(
        list(reversed(scan_ids)), fold_count=5, seed=0
    )

    assert mapping == reordered
    assert set(mapping) == set(unique_scenes)
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
    expected_sha = (
        "1f9e633a2d7cc00ee73ab73e2544e986"
        "ada9b8702ef4e1415d0424bda7d0f064"
    )
    assert canonical_scene_fold_sha256(mapping) == expected_sha
    assert canonical_scene_fold_sha256(dict(reversed(list(mapping.items())))) \
        == expected_sha
    assert all(
        mapping[scene_id] == mapping[other]
        for scene_id, other in zip(scan_ids, scan_ids)
    )


@pytest.mark.parametrize(
    "scan_ids,fold_count,seed",
    [
        ([], 5, 0),
        (["a", "b", "c", "d"], 5, 0),
        (["a", "b", "c", "d", ""], 5, 0),
        (["a", "b", "c", "d", 5], 5, 0),
        (["a", "b", "c", "d", "e"], 4, 0),
        (["a", "b", "c", "d", "e"], 5, 1),
    ],
)
def test_scene_folds_reject_noncontract_inputs(scan_ids, fold_count, seed):
    with pytest.raises((TypeError, ValueError)):
        build_residual_scene_folds(
            scan_ids, fold_count=fold_count, seed=seed
        )


def _hit_bits_for_scene_deltas(scene_deltas):
    scan_ids = []
    baseline = []
    proposed = []
    for scene_index, delta in enumerate(scene_deltas):
        scene_id = "cluster{:02d}".format(scene_index)
        for row_index in range(4):
            scan_ids.append(scene_id)
            if delta >= 0:
                baseline.append(0)
                proposed.append(int(row_index < delta))
            else:
                baseline.append(int(row_index < -delta))
                proposed.append(0)
    return scan_ids, baseline, proposed


def test_scene_clustered_bootstrap_is_exact_repeated_and_not_row_based():
    scan_ids, baseline, proposed = _hit_bits_for_scene_deltas(
        [3, 2, 1, 0, -1, -2]
    )

    first = scene_clustered_hit_delta_bootstrap(
        scan_ids, baseline, proposed
    )
    second = scene_clustered_hit_delta_bootstrap(
        list(reversed(scan_ids)),
        list(reversed(baseline)),
        list(reversed(proposed)),
    )

    assert first == second
    assert first == {
        "confidence": 0.95,
        "delta_hits": 3,
        "lower_bound_95": -4,
        "replicates": 10000,
        "scene_count": 6,
        "seed": 0,
    }
    parameters = inspect.signature(
        scene_clustered_hit_delta_bootstrap
    ).parameters
    assert set(parameters) == {"scan_ids", "baseline_hits", "proposed_hits"}
    with pytest.raises(TypeError):
        scene_clustered_hit_delta_bootstrap(
            scan_ids, baseline, proposed, resample_unit="row"
        )


@pytest.mark.parametrize(
    "scan_ids,baseline,proposed",
    [
        (["a", "b"], [0], [1, 1]),
        (["a", "b"], [0, 2], [1, 1]),
        (["a", "b"], [0, 0], [1, -1]),
        (["a", ""], [0, 0], [1, 1]),
        (["a"], [0], [1]),
    ],
)
def test_scene_clustered_bootstrap_rejects_malformed_rows(
        scan_ids, baseline, proposed):
    with pytest.raises((TypeError, ValueError)):
        scene_clustered_hit_delta_bootstrap(
            scan_ids, baseline, proposed
        )


def _selection_candidate(
        percentile, margin, deltas025, deltas050=None, hidden_dim=0,
        weight_decay=1e-3, break_cost=8.0):
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
        "break_cost": break_cost,
        "margin_percentile": percentile,
        "margin": margin,
        "scan_ids": scan_ids,
        "baseline_hits025": baseline025,
        "proposed_hits025": proposed025,
        "baseline_hits050": baseline050,
        "proposed_hits050": proposed050,
        "switch_bits": switch_bits,
    }


def test_configuration_grid_is_closed_and_selection_uses_exact_gates():
    assert RESIDUAL_HIDDEN_DIMS == (0, 64)
    assert RESIDUAL_WEIGHT_DECAYS == (1e-4, 1e-3)
    assert RESIDUAL_BREAK_COSTS == (2.0, 4.0, 8.0)
    assert RESIDUAL_MARGIN_PERCENTILES == (
        50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 97.5, 99.0
    )
    percentile90 = _selection_candidate(
        90.0, 0.50, [1, 1, 1, 1, 1, 1]
    )
    percentile95 = _selection_candidate(
        95.0, 0.60, [1, 1, 1, 1, 1, 1]
    )
    lower_objective = _selection_candidate(
        99.0, 0.70, [1, 1, 1, 1, 1, 0]
    )
    fold_regression = _selection_candidate(
        80.0, 0.40, [2, -1, 2, 2, 2, 2]
    )

    choice = choose_selective_configuration([
        percentile90, lower_objective, fold_regression, percentile95
    ])

    assert choice["eligible"] is True
    assert choice["selected"] == "residual"
    assert choice["delta_hits025"] == 6
    assert choice["delta_hits025"] > 0
    assert choice["delta_hits050"] == 0
    assert choice["bootstrap025"]["lower_bound_95"] == 6
    assert choice["bootstrap050"]["lower_bound_95"] == 0
    assert choice["fold_deltas"]["0"]["hits050"] >= 0
    assert choice["margin_percentile"] == 95.0
    assert choice["margin"] == 0.60


def test_configuration_selection_rejects_fold_and_cluster_regressions():
    fold_regression = _selection_candidate(
        90.0, 0.5, [2, -1, 2, 2, 2, 2]
    )
    negative_lower_bound025 = _selection_candidate(
        95.0, 0.6, [0, 0, 0, 2, -1, 1]
    )
    negative_lower_bound050 = _selection_candidate(
        97.5, 0.7,
        [2, 2, 2, 2, 2, 2],
        [0, 0, 0, 2, -1, 1],
    )

    choice = choose_selective_configuration([
        fold_regression,
        negative_lower_bound025,
        negative_lower_bound050,
    ])

    assert choice["candidate_count"] == 3
    assert choice["eligible_candidate_count"] == 0
    assert choice["eligible"] is False
    assert choice["reason"] == "no-eligible-configuration"
    assert choice["selected"] == "baseline"
    assert len(choice["candidate_diagnostics"]) == 3
    for record in choice["candidate_diagnostics"]:
        predicates = record["eligibility_predicates"]
        assert set(predicates) == {
            "not_no_switch",
            "all_folds_nonnegative025",
            "all_folds_nonnegative050",
            "pooled_delta025_positive",
            "bootstrap025_lower_bound_positive",
            "bootstrap050_lower_bound_nonnegative",
        }
        assert record["failed_predicates"] == sorted(
            name for name, passed in predicates.items() if not passed
        )
        assert record["eligible"] is False
        assert record["sample_count"] == 24
        assert record["switches"] + record["abstentions"] == 24
        assert record["switch_rate"] == pytest.approx(
            record["switches"] / 24.0
        )
        for threshold in ("0.25", "0.50"):
            effects = record["effects"][threshold]
            assert set(effects) == {
                "fixes",
                "breaks",
                "neutral_switches",
                "kept_correct",
                "kept_wrong",
            }
            assert sum(effects.values()) == 24
            assert record["proposed"][threshold]["hits"] - \
                record["baseline"][threshold]["hits"] == \
                effects["fixes"] - effects["breaks"]


def test_configuration_ties_prefer_linear_weight_decay_and_break_cost():
    common = [1, 1, 1, 1, 1, 1]
    candidates = [
        _selection_candidate(
            95.0, 0.6, common, hidden_dim=64,
            weight_decay=1e-3, break_cost=8.0
        ),
        _selection_candidate(
            95.0, 0.6, common, hidden_dim=0,
            weight_decay=1e-4, break_cost=8.0
        ),
        _selection_candidate(
            95.0, 0.6, common, hidden_dim=0,
            weight_decay=1e-3, break_cost=4.0
        ),
        _selection_candidate(
            95.0, 0.6, common, hidden_dim=0,
            weight_decay=1e-3, break_cost=8.0
        ),
    ]

    choice = choose_selective_configuration(candidates)

    assert choice["hidden_dim"] == 0
    assert choice["weight_decay"] == 1e-3
    assert choice["break_cost"] == 8.0


def test_no_switch_sentinel_is_diagnostic_and_cannot_win():
    candidate = _selection_candidate(
        None, float("inf"), [0, 0, 0, 0, 0, 0]
    )

    choice = choose_selective_configuration([candidate])

    assert choice["eligible"] is False
    assert choice["selected"] == "baseline"
    assert choice["candidate_count"] == 1
    assert choice["eligible_candidate_count"] == 0
    assert len(choice["candidate_diagnostics"]) == 1
    diagnostic = choice["candidate_diagnostics"][0]
    assert diagnostic["no_switch"] is True
    assert diagnostic["margin"] is None
    assert diagnostic["sample_count"] == 24
    assert diagnostic["switches"] == 0
    assert diagnostic["abstentions"] == 24
    assert diagnostic["switch_rate"] == 0.0
    assert diagnostic["baseline"] == diagnostic["proposed"]
    assert diagnostic["effects"] == {
        "0.25": {
            "fixes": 0,
            "breaks": 0,
            "neutral_switches": 0,
            "kept_correct": 12,
            "kept_wrong": 12,
        },
        "0.50": {
            "fixes": 0,
            "breaks": 0,
            "neutral_switches": 0,
            "kept_correct": 6,
            "kept_wrong": 18,
        },
    }
    assert diagnostic["failed_predicates"] == [
        "bootstrap025_lower_bound_positive",
        "not_no_switch",
        "pooled_delta025_positive",
    ]


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("hidden_dim", 32),
        ("weight_decay", 1e-2),
        ("break_cost", 1.0),
        ("margin_percentile", 55.0),
        ("margin", -0.1),
        ("switch_bits", [2] * 24),
        ("proposed_hits025", [0] * 23),
    ],
)
def test_configuration_selection_rejects_noncontract_candidate(
        field, bad_value):
    candidate = _selection_candidate(
        95.0, 0.6, [1, 1, 1, 1, 1, 1]
    )
    candidate[field] = bad_value

    with pytest.raises((TypeError, ValueError)):
        choose_selective_configuration([candidate])


def test_configuration_selection_requires_identical_oof_baseline_rows():
    first = _selection_candidate(
        90.0, 0.5, [1, 1, 1, 1, 1, 1]
    )
    second = _selection_candidate(
        95.0, 0.6, [1, 1, 1, 1, 1, 1]
    )
    second["baseline_hits025"][2] = 1

    with pytest.raises(ValueError, match="same OOF baseline"):
        choose_selective_configuration([first, second])
