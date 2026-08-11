import pytest
import torch

from scripts.cache_scanrefer_joint_box_mask import (
    APPROVED_JOINT_ROW_KEYS,
    INFERENCE_FORBIDDEN_KEYS,
    estimate_cache_capacity,
    validate_joint_cache_manifest,
    validate_joint_cache_row,
)


def test_capacity_preflight_requires_four_gib_reserve():
    estimate = estimate_cache_capacity(1024, 256 << 20, 10 << 30)
    assert estimate["projected_bytes"] == 9 * (1 << 30)
    assert estimate["can_materialize"] is False
    assert estimate["reserve_bytes"] == 4 * (1 << 30)


def test_capacity_preflight_rejects_non_positive_inputs():
    with pytest.raises(ValueError, match="positive"):
        estimate_cache_capacity(0, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        estimate_cache_capacity(1, 0, 1)
    with pytest.raises(ValueError, match="free"):
        estimate_cache_capacity(1, 1, -1)


def _manifest(split="train"):
    return {
        "schema": "rec-joint-box-mask-cache-v1",
        "split": split,
        "sample_count": 2,
        "dataset_size": 2,
        "source_dataset_size": 2,
        "feature_schema_version": "rec-query-v1",
        "geometry_schema_version": "rec-geometry-flat-v1",
        "candidate_count": 16,
        "variant_count": 7,
        "source_count": 3,
        "thresholds": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "source_names": ["text", "query", "fused"],
        "base_cache_manifest_sha256": "a" * 64,
        "geometry_cache_manifest_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "shards": ["shard_000000.pt"],
        "complete": True,
        "validation_data_accessed": False,
    }


def _row(index=0):
    return {
        "dataset_index": index,
        "scan_id": "scene0000_00",
        "target_id": 1,
        "query_indices": torch.arange(16, dtype=torch.long),
        "candidate_valid": torch.ones(16, dtype=torch.bool),
        "mask_ious": torch.full((16, 3, 5), 0.5),
    }


def test_manifest_rejects_validation_split():
    with pytest.raises(ValueError, match="train"):
        validate_joint_cache_manifest(_manifest(split="val"), "train")


def test_manifest_rejects_unsealed_partial_cache():
    manifest = _manifest()
    del manifest["complete"]
    with pytest.raises(ValueError, match="complete"):
        validate_joint_cache_manifest(manifest, "train")


def test_row_has_exact_keys_and_no_inference_target_leak():
    row = _row()
    validate_joint_cache_row(row, 0)
    assert set(row) == APPROVED_JOINT_ROW_KEYS
    assert not (set(row) & INFERENCE_FORBIDDEN_KEYS)


@pytest.mark.parametrize("forbidden", [
    "gt_masks", "candidate_ious", "geometry_ious", "center_label",
    "box_label_mask", "target_ious",
])
def test_row_rejects_ground_truth_or_iou_fields(forbidden):
    row = _row()
    row[forbidden] = torch.zeros(1)
    with pytest.raises(ValueError, match="forbidden|target|schema"):
        validate_joint_cache_row(row, 0)


def test_row_rejects_nonfinite_or_invalid_mask_iou():
    row = _row()
    row["mask_ious"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_joint_cache_row(row, 0)
    row = _row()
    row["mask_ious"][0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="\[0, 1\]"):
        validate_joint_cache_row(row, 0)
