import hashlib

import pytest
import torch

from models.rec_candidate_adapter import (
    FEATURE_SCHEMA_VERSION,
    attach_candidate_targets,
    build_full_rec_query_state,
    build_rec_candidate_batch,
    compact_rec_query_state,
    scatter_candidate_scores,
)


def _synthetic_batch():
    torch.manual_seed(11)
    batch_size = 2
    num_queries = 4
    num_tokens = 5
    proj_dim = 64
    num_superpoints = 3
    num_points = 6

    end_points = {
        "last_center": torch.tensor([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
             [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
             [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
        ]),
        "last_pred_size": torch.ones(batch_size, num_queries, 3),
        "last_sem_cls_scores": torch.randn(
            batch_size, num_queries, num_tokens
        ),
        "last_proj_queries": torch.randn(
            batch_size, num_queries, proj_dim
        ),
        "proj_tokens": torch.randn(batch_size, num_tokens, proj_dim),
        "seeds_obj_cls_logits": torch.randn(batch_size, 1, 8),
        "query_points_sample_inds": torch.tensor([
            [0, 2, 4, 6], [1, 3, 5, 7]
        ]),
        "last_pred_masks": [
            torch.randn(1, num_queries, num_superpoints)
            for _ in range(batch_size)
        ],
        "sp_last_pred_masks": [
            torch.randn(num_queries, num_superpoints)
            for _ in range(batch_size)
        ],
        "adaptive_weights": [torch.tensor(0.6), torch.tensor(0.4)],
    }
    inputs = {
        "point_clouds": torch.tensor([
            [[0.0, 0.0, 0.0, 0, 0, 0],
             [1.0, 1.0, 1.0, 0, 0, 0],
             [2.0, 2.0, 2.0, 0, 0, 0],
             [3.0, 3.0, 3.0, 0, 0, 0],
             [4.0, 4.0, 4.0, 0, 0, 0],
             [5.0, 5.0, 5.0, 0, 0, 0]],
            [[1.0, 1.0, 1.0, 0, 0, 0]] * num_points,
        ], dtype=torch.float32),
        "positive_map": torch.zeros(batch_size, 1, num_tokens),
        "modify_positive_map": torch.zeros(batch_size, 1, num_tokens),
        "pron_positive_map": torch.zeros(batch_size, 1, num_tokens),
        "other_entity_map": torch.zeros(batch_size, 1, num_tokens),
        "rel_positive_map": torch.zeros(batch_size, 1, num_tokens),
    }
    inputs["positive_map"][:, :, 0] = 1.0
    inputs["modify_positive_map"][:, :, 1] = 1.0
    inputs["pron_positive_map"][:, :, 2] = 1.0
    inputs["other_entity_map"][:, :, 3] = 1.0
    inputs["rel_positive_map"][:, :, 4] = 1.0
    return end_points, inputs


def _assert_states_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], dict):
            _assert_states_equal(actual[key], expected[key])
        elif torch.is_tensor(actual[key]):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key


def test_build_full_rec_query_state_keeps_the_full_query_axis():
    end_points, inputs = _synthetic_batch()

    full = build_full_rec_query_state(end_points, inputs)

    assert full["default_scores"].shape == (2, 4)
    assert full["contrastive_scores"].shape == (2, 4)
    assert full["boxes"].shape == (2, 4, 6)
    assert full["features"].shape == (2, 4, 152)
    assert full["num_queries"] == 4
    assert full["features"].shape[-1] == len(full["feature_names"])
    assert full["schema_version"] == FEATURE_SCHEMA_VERSION
    assert torch.isfinite(full["features"]).all()
    target_only_keys = {
        "center_label", "size_gts", "box_label_mask", "gt_masks",
        "candidate_ious", "threshold_labels",
    }
    assert target_only_keys.isdisjoint(full)


def test_compact_rec_query_state_matches_legacy_candidate_batch_elementwise():
    end_points, inputs = _synthetic_batch()
    legacy_tensor_sha256 = {
        "query_indices": (
            "ebe01130c3c3016af37b51bd57f187bf78f2ab7495591ea8443e9a2c573453ea"
        ),
        "valid_mask": (
            "04abc8821a06e5a30937967d11ad10221cb5ac3b5273e434f1284ee87129a061"
        ),
        "features": (
            "8860f5f62e505a873a07e2b923f6c721582ad86bb5896d2eef080d5bd01154e2"
        ),
        "boxes": (
            "3c8916661fd995109e8d34df7bb13303d2d1bdd3dc9e8e747fb261c1c0c49695"
        ),
        "default_scores": (
            "99f9e5296d79eba2b1a68772b6723d4782f7411164b71e8f7322bb6354c16198"
        ),
        "contrastive_scores": (
            "2b61ac6f46862a0ccb8f4f408c09d7d6773154b4fe650e1e522659adf2634e17"
        ),
        "default_top1_query_index": (
            "9b768a7138e147b4158a6b26c2e04ee536af084a18f7b751a9439af1a7cc0765"
        ),
    }
    legacy = build_rec_candidate_batch(
        end_points, inputs, topk_per_source=2, max_candidates=4
    )

    full = build_full_rec_query_state(end_points, inputs)
    compact = compact_rec_query_state(
        full, topk_per_source=2, max_candidates=4
    )

    assert compact.keys() == legacy.keys()
    for key in (
        "query_indices", "valid_mask", "features", "boxes",
        "default_scores", "contrastive_scores",
        "default_top1_query_index",
    ):
        assert torch.equal(compact[key], legacy[key]), key
        legacy_bytes = legacy[key].detach().cpu().contiguous().numpy().tobytes()
        assert hashlib.sha256(legacy_bytes).hexdigest() == (
            legacy_tensor_sha256[key]
        ), key
    assert compact["schema_version"] == legacy["schema_version"]
    assert compact["feature_names"] == legacy["feature_names"]
    assert compact["num_queries"] == full["num_queries"] == 4
    assert compact["num_queries"] == legacy["num_queries"]
    _assert_states_equal(compact["model_inputs"], legacy["model_inputs"])

    full_with_declared_query_count = dict(full)
    full_with_declared_query_count["num_queries"] = 17
    with pytest.raises(ValueError, match="num_queries"):
        compact_rec_query_state(full_with_declared_query_count)

    for invalid_num_queries in (0, 4.0):
        invalid_full = dict(full)
        invalid_full["num_queries"] = invalid_num_queries
        with pytest.raises(ValueError, match="num_queries"):
            compact_rec_query_state(invalid_full)

    for field in ("contrastive_scores", "features", "boxes"):
        invalid_full = dict(full)
        invalid_full[field] = invalid_full[field][:, :-1]
        with pytest.raises(ValueError, match=field):
            compact_rec_query_state(invalid_full)


def test_full_and_compact_query_states_ignore_target_only_sentinels():
    end_points, inputs = _synthetic_batch()
    target_only_keys = (
        "center_label", "size_gts", "box_label_mask", "gt_masks",
        "candidate_ious", "threshold_labels",
    )
    first_end_points = dict(end_points)
    first_inputs = dict(inputs)
    second_end_points = dict(end_points)
    second_inputs = dict(inputs)
    for key in target_only_keys:
        first_end_points[key] = torch.tensor([-101.0])
        first_inputs[key] = torch.tensor([-202.0])
        second_end_points[key] = torch.tensor([303.0, 404.0])
        second_inputs[key] = torch.tensor([505.0, 606.0])

    first_full = build_full_rec_query_state(first_end_points, first_inputs)
    second_full = build_full_rec_query_state(second_end_points, second_inputs)
    first_compact = compact_rec_query_state(first_full)
    second_compact = compact_rec_query_state(second_full)

    _assert_states_equal(first_full, second_full)
    _assert_states_equal(first_compact, second_compact)


def test_build_rec_candidate_batch_has_finite_deployable_features():
    end_points, inputs = _synthetic_batch()

    batch = build_rec_candidate_batch(
        end_points,
        inputs,
        topk_per_source=2,
        max_candidates=4,
    )

    assert batch["features"].shape[:2] == (2, 4)
    assert batch["boxes"].shape == (2, 4, 6)
    assert batch["query_indices"].shape == (2, 4)
    assert batch["valid_mask"].dtype == torch.bool
    assert batch["features"].shape[-1] == len(batch["feature_names"])
    assert batch["schema_version"] == FEATURE_SCHEMA_VERSION
    assert torch.isfinite(batch["features"]).all()
    assert set(batch["model_inputs"]) == {"features", "valid_mask"}
    assert not any(
        "gt" in key or "iou" in key for key in batch["model_inputs"]
    )


def test_build_rec_candidate_batch_gathers_boxes_by_query_index():
    end_points, inputs = _synthetic_batch()
    batch = build_rec_candidate_batch(
        end_points, inputs, topk_per_source=1, max_candidates=3
    )
    full_boxes = torch.cat(
        [end_points["last_center"], end_points["last_pred_size"]], dim=-1
    )
    expected = torch.gather(
        full_boxes,
        1,
        batch["query_indices"].unsqueeze(-1).expand(-1, -1, 6),
    )

    assert torch.equal(batch["boxes"], expected)


def test_attach_candidate_targets_is_separate_from_model_inputs():
    end_points, inputs = _synthetic_batch()
    batch = build_rec_candidate_batch(
        end_points, inputs, topk_per_source=2, max_candidates=4
    )
    end_points.update({
        "center_label": torch.tensor([
            [[0.0, 0.0, 0.0]], [[3.0, 1.0, 0.0]]
        ]),
        "size_gts": torch.ones(2, 1, 3),
        "box_label_mask": torch.ones(2, 1),
    })

    targeted = attach_candidate_targets(batch, end_points)

    assert "candidate_ious" not in batch
    assert targeted["candidate_ious"].shape == (2, 4)
    assert targeted["threshold_labels"].shape == (2, 4, 2)
    assert set(targeted["model_inputs"]) == {"features", "valid_mask"}
    assert not any(
        "gt" in key or "iou" in key for key in targeted["model_inputs"]
    )


def test_attach_candidate_targets_can_use_only_the_root_target():
    candidate_batch = {
        "boxes": torch.tensor([[
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [8.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]]),
        "valid_mask": torch.tensor([[True, True]]),
        "model_inputs": {
            "features": torch.zeros(1, 2, 1),
            "valid_mask": torch.tensor([[True, True]]),
        },
    }
    end_points = {
        "center_label": torch.tensor([[
            [0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ]]),
        "size_gts": torch.ones(1, 2, 3) * 2.0,
        "box_label_mask": torch.tensor([[True, True]]),
    }

    targeted = attach_candidate_targets(
        candidate_batch, end_points, root_only=True
    )

    assert torch.allclose(targeted["candidate_ious"], torch.tensor([[1.0, 0.0]]))


def test_scatter_candidate_scores_restores_full_query_axis():
    compact = torch.tensor([[0.7, 0.2, 0.9]])
    indices = torch.tensor([[2, 0, 3]])
    valid = torch.tensor([[True, False, True]])

    full = scatter_candidate_scores(
        compact, indices, valid, num_queries=5, fill_value=-100.0
    )

    assert torch.allclose(
        full,
        torch.tensor([[-100.0, -100.0, 0.7, 0.9, -100.0]]),
    )


def test_scatter_candidate_scores_excludes_unselected_queries_by_default():
    full = scatter_candidate_scores(
        torch.tensor([[-10001.0]]),
        torch.tensor([[1]]),
        torch.tensor([[True]]),
        num_queries=3,
    )

    assert full[0, 1].item() == -10001.0
    assert torch.isneginf(full[0, 0])
    assert torch.isneginf(full[0, 2])
