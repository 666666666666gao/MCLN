import pytest
import torch

import models.rec_mask_geometry as rec_mask_geometry
from models.rec_mask_geometry import (
    build_rec_mask_geometry_candidates,
    mask_logits_to_point_aabbs,
    normalize_mcln_mask_logits,
)


EXPECTED_DEFAULT_VARIANTS = (
    {
        "name": "regressed",
        "source": "regressed",
        "logit_threshold": 0.0,
        "quantile": 0.0,
        "regressed_weight": 1.0,
    },
    {
        "name": "fused_t0_exact",
        "source": "fused",
        "logit_threshold": 0.0,
        "quantile": 0.0,
        "regressed_weight": 0.0,
    },
    {
        "name": "fused_t0_q0.005",
        "source": "fused",
        "logit_threshold": 0.0,
        "quantile": 0.005,
        "regressed_weight": 0.0,
    },
    {
        "name": "query_t0_q0.005",
        "source": "query",
        "logit_threshold": 0.0,
        "quantile": 0.005,
        "regressed_weight": 0.0,
    },
    {
        "name": "fused_t0.5_q0.005",
        "source": "fused",
        "logit_threshold": 0.5,
        "quantile": 0.005,
        "regressed_weight": 0.0,
    },
    {
        "name": "blend_regressed_fused_q0.005",
        "source": "fused",
        "logit_threshold": 0.0,
        "quantile": 0.005,
        "regressed_weight": 0.5,
    },
    {
        "name": "blend_regressed_query_q0.005",
        "source": "query",
        "logit_threshold": 0.0,
        "quantile": 0.005,
        "regressed_weight": 0.5,
    },
)


def test_mask_geometry_schema_and_default_variants_are_stable():
    assert rec_mask_geometry.MASK_GEOMETRY_SCHEMA_VERSION == \
        "rec-mask-geometry-v1"
    assert rec_mask_geometry.DEFAULT_REC_MASK_GEOMETRY_VARIANTS == \
        EXPECTED_DEFAULT_VARIANTS


def test_normalize_mask_logits_gathers_original_queries_and_fuses_logits():
    text = torch.tensor([
        [4.0, -4.0, 1.0, -1.0],
        [0.0, 1.0, 2.0, 3.0],
        [-2.0, 2.0, -3.0, 3.0],
    ])
    query = torch.tensor([
        [-2.0, 2.0, -1.0, 1.0],
        [3.0, 2.0, 1.0, 0.0],
        [2.0, -2.0, 3.0, -3.0],
    ])
    end_points = {
        "last_pred_masks": [text.unsqueeze(0)],
        "sp_last_pred_masks": [query],
        "adaptive_weights": [torch.tensor([0.25])],
    }

    text_out, query_out, fused_out, alpha = normalize_mcln_mask_logits(
        end_points, batch_idx=0, query_indices=torch.tensor([2, 0])
    )

    assert text_out.shape == query_out.shape == fused_out.shape == (2, 4)
    assert torch.equal(text_out, text[[2, 0]])
    assert torch.equal(query_out, query[[2, 0]])
    assert alpha.dim() == 0
    assert alpha.item() == pytest.approx(0.25)
    assert torch.allclose(fused_out, 0.25 * text_out + 0.75 * query_out)
    probability_blend = (
        0.25 * text_out.sigmoid() + 0.75 * query_out.sigmoid()
    )
    assert not torch.allclose(fused_out, probability_blend)


def test_normalize_mask_logits_handles_batched_singleton_shapes():
    text = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    query = -text
    end_points = {
        "last_pred_masks": text.unsqueeze(1),
        "sp_last_pred_masks": query,
        "adaptive_weights": torch.tensor([[0.1], [0.6]]),
    }

    text_out, query_out, fused_out, alpha = normalize_mcln_mask_logits(
        end_points, batch_idx=1, query_indices=[1]
    )

    assert text_out.shape == query_out.shape == fused_out.shape == (1, 4)
    assert torch.equal(text_out[0], text[1, 1])
    assert torch.equal(query_out[0], query[1, 1])
    assert alpha.item() == pytest.approx(0.6)


def test_normalize_mask_logits_gathers_query_wise_adaptive_weight():
    logits = torch.zeros(2, 3)
    query = torch.ones(2, 3)
    end_points = {
        "last_pred_masks": [logits],
        "sp_last_pred_masks": [query],
        "adaptive_weights": [torch.tensor([0.25, 0.75])],
    }

    _, _, fused, alpha = normalize_mcln_mask_logits(
        end_points, 0, torch.tensor([1, 0])
    )

    assert alpha.shape == (2, 1)
    assert torch.equal(alpha[:, 0], torch.tensor([0.75, 0.25]))
    assert torch.equal(fused[:, 0], torch.tensor([0.25, 0.75]))


def test_normalize_mask_logits_rejects_wrong_query_weight_count():
    logits = torch.zeros(2, 3)
    end_points = {
        "last_pred_masks": [logits],
        "sp_last_pred_masks": [logits],
        "adaptive_weights": [torch.tensor([0.25, 0.50, 0.75])],
    }

    with pytest.raises(ValueError, match="adaptive weight"):
        normalize_mcln_mask_logits(end_points, 0, torch.tensor([0]))


@pytest.mark.parametrize(
    "adaptive_weight",
    [torch.tensor(0.25), torch.tensor([0.25]), torch.tensor([[0.25]])],
)
def test_normalize_mask_logits_accepts_scalar_singleton_weights(
        adaptive_weight):
    logits = torch.zeros(2, 3)
    end_points = {
        "last_pred_masks": [logits],
        "sp_last_pred_masks": [logits],
        "adaptive_weights": [adaptive_weight],
    }

    _, _, _, alpha = normalize_mcln_mask_logits(
        end_points, 0, torch.tensor([0])
    )

    assert alpha.dim() == 0
    assert alpha.item() == pytest.approx(0.25)


def test_mask_aabbs_preserve_raw_superpoint_ids_and_compute_quantiles():
    coords = torch.arange(10, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    superpoint_ids = torch.tensor([1] * 8 + [3] * 2)
    # Column zero is deliberately negative. Compacting [1, 3] to [0, 1]
    # would therefore produce the wrong selection.
    mask_logits = torch.tensor([[-100.0, 2.0, 100.0, -2.0]])

    boxes, valid, diagnostics = mask_logits_to_point_aabbs(
        coords,
        superpoint_ids,
        mask_logits,
        logit_threshold=0.0,
        quantiles=(0.0, 0.25),
        min_points=1,
        max_point_fraction=0.9,
    )

    assert boxes.shape == (1, 2, 6)
    assert valid.tolist() == [[True, True]]
    assert torch.allclose(
        boxes[0, 0], torch.tensor([3.5, 3.5, 3.5, 7.0, 7.0, 7.0])
    )
    assert torch.allclose(
        boxes[0, 1], torch.tensor([3.5, 3.5, 3.5, 3.5, 3.5, 3.5])
    )
    assert diagnostics["selected_point_counts"].tolist() == [8]
    assert diagnostics["selected_point_fractions"].item() == pytest.approx(0.8)
    assert diagnostics["selected_superpoint_counts"].tolist() == [1]
    assert diagnostics["selected_superpoint_fractions"].item() == pytest.approx(0.5)
    assert diagnostics["lower_bounds"].shape == (1, 2, 3)
    assert diagnostics["upper_bounds"].shape == (1, 2, 3)
    assert diagnostics["rejection_codes"].tolist() == [[0, 0]]


def test_mask_aabbs_use_a_strict_logit_threshold():
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [10.0, 10.0, 10.0],
        [12.0, 12.0, 12.0],
        [14.0, 14.0, 14.0],
    ])
    superpoint_ids = torch.tensor([1, 1, 1, 3, 3, 3])
    mask_logits = torch.tensor([[-5.0, 0.0, 5.0, 0.1]])

    boxes, valid, diagnostics = mask_logits_to_point_aabbs(
        coords,
        superpoint_ids,
        mask_logits,
        logit_threshold=0.0,
        quantiles=(0.0,),
        min_points=1,
        max_point_fraction=0.75,
    )

    assert valid.tolist() == [[True]]
    assert diagnostics["selected_point_counts"].tolist() == [3]
    assert torch.allclose(
        boxes[0, 0], torch.tensor([12.0, 12.0, 12.0, 4.0, 4.0, 4.0])
    )


@pytest.mark.parametrize(
    "case,positive_count,min_points,max_fraction,expected_rejection",
    [
        ("empty", 0, 1, 1.0, 1 | 2),
        ("too_few", 2, 3, 1.0, 2),
        ("full", 6, 1, 1.0, 4),
        ("over_fraction", 4, 1, 0.5, 8),
    ],
)
def test_mask_aabbs_reject_bad_point_counts(
        case, positive_count, min_points, max_fraction, expected_rejection):
    del case
    coords = torch.arange(6, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    superpoint_ids = torch.arange(6)
    logits = torch.full((1, 6), -1.0)
    logits[0, :positive_count] = 1.0

    boxes, valid, diagnostics = mask_logits_to_point_aabbs(
        coords,
        superpoint_ids,
        logits,
        logit_threshold=0.0,
        quantiles=(0.0,),
        min_points=min_points,
        max_point_fraction=max_fraction,
    )

    assert valid.tolist() == [[False]]
    assert torch.equal(boxes, torch.zeros_like(boxes))
    assert diagnostics["selected_point_counts"].tolist() == [positive_count]
    assert diagnostics["base_valid"].tolist() == [False]
    assert diagnostics["rejection_codes"][0, 0].item() == expected_rejection


def test_mask_aabbs_reject_nonfinite_logits_and_coordinates():
    coords = torch.arange(6, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    superpoint_ids = torch.arange(6)
    logits = torch.tensor([[1.0, 1.0, 1.0, -1.0, float("nan"), -1.0]])

    _, logit_valid, logit_diagnostics = mask_logits_to_point_aabbs(
        coords,
        superpoint_ids,
        logits,
        logit_threshold=0.0,
        quantiles=(0.0,),
        min_points=2,
        max_point_fraction=0.9,
    )

    bad_coords = coords.clone()
    bad_coords[1, 0] = float("inf")
    finite_logits = torch.tensor([[1.0, 1.0, 1.0, -1.0, -1.0, -1.0]])
    _, coord_valid, coord_diagnostics = mask_logits_to_point_aabbs(
        bad_coords,
        superpoint_ids,
        finite_logits,
        logit_threshold=0.0,
        quantiles=(0.0,),
        min_points=2,
        max_point_fraction=0.9,
    )

    assert logit_valid.tolist() == [[False]]
    assert logit_diagnostics["finite_logits"].tolist() == [False]
    assert coord_valid.tolist() == [[False]]
    assert coord_diagnostics["finite_coordinates"].tolist() == [False]


def test_mask_aabbs_reject_degenerate_boxes():
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 4.0, 0.0],
        [10.0, 10.0, 1.0],
    ])
    superpoint_ids = torch.arange(4)
    logits = torch.tensor([[1.0, 1.0, 1.0, -1.0]])

    boxes, valid, diagnostics = mask_logits_to_point_aabbs(
        coords,
        superpoint_ids,
        logits,
        logit_threshold=0.0,
        quantiles=(0.0,),
        min_points=2,
        max_point_fraction=0.9,
    )

    assert valid.tolist() == [[False]]
    assert torch.equal(boxes, torch.zeros_like(boxes))
    assert diagnostics["nondegenerate"].tolist() == [[False]]


def _geometry_fixture():
    first = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.5, 0.2],
        [2.0, 1.0, 0.4],
        [3.0, 1.5, 0.6],
        [4.0, 2.0, 0.8],
    ])
    second = first + 10.0
    background = torch.tensor([[-5.0, -5.0, -5.0], [30.0, 30.0, 30.0]])
    coords = torch.cat([first, second, background], dim=0)
    superpoint_ids = torch.tensor([1] * 5 + [3] * 5 + [0] * 2)

    text = torch.full((3, 4), -2.0)
    query = torch.full((3, 4), -2.0)
    text[0] = 2.0
    query[0] = 2.0
    text[2, 1] = 2.0
    query[2, 3] = 2.0
    end_points = {
        "last_pred_masks": [text.unsqueeze(0)],
        "sp_last_pred_masks": [query],
        "adaptive_weights": [torch.tensor(0.75)],
        "superpoints": superpoint_ids.unsqueeze(0),
    }
    inputs = {"point_clouds": coords.unsqueeze(0)}
    candidate_batch = {
        "boxes": torch.tensor([[
            [5.0, 4.0, 3.0, 4.0, 4.0, 4.0],
            [9.0, 9.0, 9.0, 2.0, 2.0, 2.0],
        ]]),
        "query_indices": torch.tensor([[2, 0]]),
        "valid_mask": torch.tensor([[True, True]]),
    }
    return end_points, inputs, candidate_batch


def test_build_geometry_candidates_maps_query_indices_and_keeps_fallback():
    end_points, inputs, candidate_batch = _geometry_fixture()

    result = build_rec_mask_geometry_candidates(
        end_points, inputs, candidate_batch
    )

    assert set((
        "boxes", "valid_mask", "geometry_features", "geometry_feature_names",
        "variant_names",
    )) \
        <= set(result)
    assert result["boxes"].shape == (1, 2, 7, 6)
    assert result["valid_mask"].shape == (1, 2, 7)
    assert result["geometry_features"].shape[:3] == (1, 2, 7)
    assert len(result["variant_names"]) == 7
    assert result["variant_names"][0] == "regressed"
    required_features = {
        "valid",
        "source_is_regressed",
        "source_is_query",
        "source_is_fused",
        "logit_threshold",
        "quantile",
        "regressed_weight",
        "selected_point_count",
        "selected_point_fraction",
        "selected_superpoint_count",
        "selected_superpoint_fraction",
        "foreground_logit_mean",
        "source_mask_to_regressed_volume_ratio",
        "center_delta_x_scene_norm",
        "size_delta_x_scene_norm",
        "source_mask_vs_regressed_iou",
    }
    assert required_features <= set(result["geometry_feature_names"])
    assert result["geometry_features"].shape[-1] == len(
        result["geometry_feature_names"]
    )
    assert torch.equal(result["boxes"][:, :, 0], candidate_batch["boxes"])
    assert result["valid_mask"][0, 0].all()
    assert result["valid_mask"][0, 1].tolist() == [
        True, False, False, False, False, False, False
    ]
    assert torch.isfinite(result["geometry_features"]).all()

    _, query_logits, fused_logits, _ = normalize_mcln_mask_logits(
        end_points, 0, torch.tensor([2])
    )
    fused_boxes, fused_valid, _ = mask_logits_to_point_aabbs(
        inputs["point_clouds"][0, :, :3],
        end_points["superpoints"][0],
        fused_logits,
        logit_threshold=0.0,
        quantiles=(0.0, 0.005),
    )
    query_boxes, query_valid, _ = mask_logits_to_point_aabbs(
        inputs["point_clouds"][0, :, :3],
        end_points["superpoints"][0],
        query_logits,
        logit_threshold=0.0,
        quantiles=(0.005,),
    )
    assert fused_valid.all() and query_valid.all()
    assert torch.allclose(result["boxes"][0, 0, 1], fused_boxes[0, 0])
    assert torch.allclose(result["boxes"][0, 0, 2], fused_boxes[0, 1])
    assert torch.allclose(result["boxes"][0, 0, 3], query_boxes[0, 0])
    assert torch.allclose(
        result["boxes"][0, 0, 5],
        0.5 * candidate_batch["boxes"][0, 0] + 0.5 * fused_boxes[0, 1],
    )
    assert torch.allclose(
        result["boxes"][0, 0, 6],
        0.5 * candidate_batch["boxes"][0, 0] + 0.5 * query_boxes[0, 0],
    )


def test_build_geometry_candidates_persists_default_schema_metadata():
    end_points, inputs, candidate_batch = _geometry_fixture()

    result = build_rec_mask_geometry_candidates(
        end_points, inputs, candidate_batch
    )

    assert result["schema_version"] == "rec-mask-geometry-v1"
    assert result["geometry_feature_names"] == \
        rec_mask_geometry.REC_MASK_GEOMETRY_FEATURE_NAMES
    assert result["variant_configs"] == EXPECTED_DEFAULT_VARIANTS
    assert result["variant_names"] == tuple(
        variant["name"] for variant in EXPECTED_DEFAULT_VARIANTS
    )
    assert result["min_points"] == 5
    assert result["max_point_fraction"] == pytest.approx(0.5)


def test_build_geometry_candidates_persists_effective_filter_overrides():
    end_points, inputs, candidate_batch = _geometry_fixture()
    custom_variants = EXPECTED_DEFAULT_VARIANTS[:2]

    result = build_rec_mask_geometry_candidates(
        end_points,
        inputs,
        candidate_batch,
        variant_config={
            "variants": custom_variants,
            "min_points": 2.0,
            "max_point_fraction": "0.8",
        },
    )

    assert result["variant_configs"] == custom_variants
    assert result["min_points"] == 2
    assert type(result["min_points"]) is int
    assert result["max_point_fraction"] == pytest.approx(0.8)
    assert type(result["max_point_fraction"]) is float


@pytest.mark.parametrize(
    "config_key,bad_value",
    [
        ("min_points", True),
        ("min_points", 0),
        ("min_points", -1),
        ("min_points", 2.5),
        ("max_point_fraction", 0.0),
        ("max_point_fraction", -0.1),
        ("max_point_fraction", 1.1),
        ("max_point_fraction", float("nan")),
        ("max_point_fraction", float("inf")),
    ],
)
def test_build_geometry_candidates_rejects_invalid_filters_without_masks(
        config_key, bad_value):
    end_points, inputs, candidate_batch = _geometry_fixture()
    config = {
        "variants": EXPECTED_DEFAULT_VARIANTS[:1],
        config_key: bad_value,
    }

    with pytest.raises(ValueError):
        build_rec_mask_geometry_candidates(
            end_points, inputs, candidate_batch, variant_config=config
        )


def test_build_geometry_candidates_needs_no_ground_truth_fields():
    end_points, inputs, candidate_batch = _geometry_fixture()
    forbidden = {
        "center_label", "size_gts", "box_label_mask", "gt_masks",
        "point_instance_label", "all_bboxes",
    }
    assert forbidden.isdisjoint(end_points)
    assert forbidden.isdisjoint(inputs)
    assert forbidden.isdisjoint(candidate_batch)

    result = build_rec_mask_geometry_candidates(
        end_points, inputs, candidate_batch
    )

    assert result["valid_mask"][0, :, 0].tolist() == [True, True]


def _target_geometry_fixture():
    model_inputs = {"features": torch.tensor([[[3.0, 4.0]]])}
    geometry = {
        "boxes": torch.tensor([[[
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [10.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 4.0, 1.0, 1.0],
            [10.0, 0.0, 0.0, 2.0, 1.0, 1.0],
        ]]]),
        "valid_mask": torch.tensor([[[True, True, False, True]]]),
        "model_inputs": model_inputs,
    }
    end_points = {
        "center_label": torch.tensor([[
            [0.0, 0.0, 0.0, 99.0],
            [10.0, 0.0, 0.0, 99.0],
        ]]),
        "size_gts": torch.tensor([[
            [4.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
        ]]),
        "box_label_mask": torch.tensor([[True, True]]),
    }
    return geometry, end_points, model_inputs


def test_attach_geometry_targets_distinguishes_root_only_from_all_gt():
    geometry, end_points, _ = _target_geometry_fixture()

    root_targets = rec_mask_geometry.attach_rec_mask_geometry_targets(
        geometry, end_points
    )
    all_targets = rec_mask_geometry.attach_rec_mask_geometry_targets(
        geometry, end_points, root_only=False
    )

    assert root_targets["geometry_ious"].shape == (1, 1, 4)
    assert torch.allclose(
        root_targets["geometry_ious"],
        torch.tensor([[[0.25, 0.0, 0.0, 0.0]]]),
    )
    assert torch.allclose(
        all_targets["geometry_ious"],
        torch.tensor([[[0.25, 0.50, 0.0, 1.0]]]),
    )


def test_attach_geometry_targets_zeroes_invalid_and_uses_strict_thresholds():
    geometry, end_points, _ = _target_geometry_fixture()

    targets = rec_mask_geometry.attach_rec_mask_geometry_targets(
        geometry, end_points, root_only=False
    )

    assert targets["threshold_labels"].shape == (1, 1, 4, 2)
    assert targets["threshold_labels"].tolist() == [[[
        [False, False],
        [True, False],
        [False, False],
        [True, True],
    ]]]
    assert targets["geometry_ious"][0, 0, 2].item() == 0.0


def test_attach_geometry_targets_does_not_mutate_geometry_or_model_inputs():
    geometry, end_points, model_inputs = _target_geometry_fixture()
    original_keys = set(geometry)
    original_boxes = geometry["boxes"].clone()
    original_valid = geometry["valid_mask"].clone()
    original_features = model_inputs["features"].clone()

    targets = rec_mask_geometry.attach_rec_mask_geometry_targets(
        geometry, end_points
    )

    assert targets is not geometry
    assert targets["model_inputs"] is model_inputs
    assert set(geometry) == original_keys
    assert "geometry_ious" not in geometry
    assert "threshold_labels" not in geometry
    assert torch.equal(geometry["boxes"], original_boxes)
    assert torch.equal(geometry["valid_mask"], original_valid)
    assert torch.equal(model_inputs["features"], original_features)


def _rejection_geometry_fixture():
    variant_configs = (
        {
            "name": "fused_exact",
            "source": "fused",
            "logit_threshold": 0.0,
            "quantile": 0.0,
        },
        {
            "name": "regressed",
            "source": "regressed",
            "logit_threshold": 0.0,
            "quantile": 0.0,
        },
        {
            "name": "query_trimmed",
            "source": "query",
            "logit_threshold": 0.5,
            "quantile": 0.005,
        },
        {
            "name": "fused_trimmed",
            "source": "fused",
            "logit_threshold": 0.0,
            "quantile": 0.005,
        },
    )
    sample_codes = (
        (
            torch.tensor([[11, 12], [13, 14]]),
            torch.tensor([[21, 22], [23, 24]]),
        ),
        (
            torch.tensor([[31, 32], [33, 34]]),
            torch.tensor([[41, 42], [43, 44]]),
        ),
    )
    diagnostics = tuple({
        "fused_t0": {
            "quantiles": torch.tensor([0.005, 0.0]),
            "rejection_codes": fused_codes,
        },
        "query_t0.5": {
            "quantiles": torch.tensor([0.0, 0.005]),
            "rejection_codes": query_codes,
        },
    } for fused_codes, query_codes in sample_codes)
    return {
        "boxes": torch.zeros(2, 2, 4, 6),
        "valid_mask": torch.ones(2, 2, 4, dtype=torch.bool),
        "variant_configs": variant_configs,
        "mask_diagnostics": diagnostics,
    }


def test_project_variant_rejection_codes_maps_groups_and_quantile_columns():
    geometry = _rejection_geometry_fixture()

    codes = rec_mask_geometry.project_variant_rejection_codes(geometry)

    assert codes.shape == (2, 2, 4)
    assert codes.dtype == torch.int16
    assert codes.device == geometry["boxes"].device
    assert torch.equal(codes, torch.tensor([
        [[12, 0, 22, 11], [14, 0, 24, 13]],
        [[32, 0, 42, 31], [34, 0, 44, 33]],
    ], dtype=torch.int16))


def test_project_variant_rejection_codes_rejects_missing_group():
    geometry = _rejection_geometry_fixture()
    del geometry["mask_diagnostics"][0]["query_t0.5"]

    with pytest.raises(ValueError, match="missing.*query_t0.5"):
        rec_mask_geometry.project_variant_rejection_codes(geometry)


@pytest.mark.parametrize(
    "quantiles",
    [torch.tensor([0.0, 0.25]), torch.tensor([0.005, 0.005])],
    ids=("missing", "ambiguous"),
)
def test_project_variant_rejection_codes_requires_one_matching_quantile(
        quantiles):
    geometry = _rejection_geometry_fixture()
    geometry["mask_diagnostics"][0]["query_t0.5"][
        "quantiles"
    ] = quantiles

    with pytest.raises(ValueError, match="quantile"):
        rec_mask_geometry.project_variant_rejection_codes(geometry)


@pytest.mark.parametrize("field", ("quantiles", "rejection_codes"))
def test_project_variant_rejection_codes_rejects_malformed_diagnostic_shape(
        field):
    geometry = _rejection_geometry_fixture()
    diagnostics = geometry["mask_diagnostics"][0]["fused_t0"]
    if field == "quantiles":
        diagnostics[field] = diagnostics[field].reshape(1, 2)
    else:
        diagnostics[field] = diagnostics[field].reshape(-1)

    with pytest.raises(ValueError, match="shape"):
        rec_mask_geometry.project_variant_rejection_codes(geometry)


@pytest.mark.parametrize(
    "axis,error_match",
    (("batch", "batch"), ("candidate", "candidate"), ("variant", "variant")),
)
def test_project_variant_rejection_codes_rejects_mismatched_geometry_axes(
        axis, error_match):
    geometry = _rejection_geometry_fixture()
    if axis == "batch":
        geometry["mask_diagnostics"] = geometry["mask_diagnostics"][:1]
    elif axis == "candidate":
        geometry["mask_diagnostics"][0]["fused_t0"][
            "rejection_codes"
        ] = torch.zeros(3, 2, dtype=torch.long)
    else:
        geometry["variant_configs"] = geometry["variant_configs"][:3]

    with pytest.raises(ValueError, match=error_match):
        rec_mask_geometry.project_variant_rejection_codes(geometry)


@pytest.mark.parametrize("bad_code", (-1, 32768))
def test_project_variant_rejection_codes_rejects_invalid_code_range(bad_code):
    geometry = _rejection_geometry_fixture()
    geometry["mask_diagnostics"][0]["fused_t0"][
        "rejection_codes"
    ][0, 0] = bad_code

    with pytest.raises(ValueError, match="int16"):
        rec_mask_geometry.project_variant_rejection_codes(geometry)
