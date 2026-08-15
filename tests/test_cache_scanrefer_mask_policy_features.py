import copy

import pytest
import torch

from models.rec_joint_box_mask import (
    MASK_POLICY_FEATURE_SCHEMA_VERSION,
    build_mask_policy_feature_names,
)
from scripts.cache_scanrefer_mask_policy_features import (
    CACHE_SCHEMA,
    assert_runtime_candidate_identity,
    build_frozen_candidate_identity,
    canonical_json_sha256,
    validate_feature_manifest,
    validate_feature_row,
)


def _row(index=0):
    return {
        "dataset_index": index,
        "scan_id": "scene0000_00",
        "target_id": 3,
        "query_indices": torch.arange(16, dtype=torch.long),
        "candidate_valid": torch.ones(16, dtype=torch.bool),
        "mask_policy_features": torch.zeros(16, 52),
    }


def _manifest():
    value = {
        "schema": CACHE_SCHEMA,
        "version": 1,
        "split": "train",
        "complete": True,
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
        "sample_count": 2,
        "dataset_size": 2,
        "source_dataset_size": 10,
        "feature_schema_version": MASK_POLICY_FEATURE_SCHEMA_VERSION,
        "feature_dim": 52,
        "feature_names": build_mask_policy_feature_names(),
        "candidate_count": 16,
        "checkpoint_sha256": "a" * 64,
        "base_cache_manifest_sha256": "b" * 64,
        "joint_label_manifest_sha256": "c" * 64,
        "joint_label_content_sha256": "d" * 64,
        "shards": [{
            "name": "shard_000000.pt", "row_count": 2,
            "sha256": "e" * 64,
        }],
    }
    value["content_sha256"] = canonical_json_sha256(value)
    return value


def test_feature_row_rejects_supervision_and_nonzero_padding():
    row = _row()
    assert validate_feature_row(row, 0) is row
    supervised = dict(row, mask_ious=torch.zeros(16, 3, 5))
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_row(supervised, 0)
    padded = copy.deepcopy(row)
    padded["candidate_valid"][-1] = False
    padded["mask_policy_features"][-1, 0] = 1.0
    with pytest.raises(ValueError, match="zero"):
        validate_feature_row(padded, 0)


def test_manifest_distinguishes_smoke_from_full_coverage():
    manifest = _manifest()
    assert validate_feature_manifest(manifest, require_full=False) is manifest
    with pytest.raises(ValueError, match="coverage"):
        validate_feature_manifest(manifest, require_full=True)


def test_runtime_candidate_identity_is_exact():
    base = [_row()]
    base[0]["valid_mask"] = base[0].pop("candidate_valid")
    fresh = {
        "query_indices": base[0]["query_indices"].unsqueeze(0),
        "valid_mask": base[0]["valid_mask"].unsqueeze(0),
    }
    assert assert_runtime_candidate_identity(
        fresh, base, [0], ["scene0000_00"], torch.tensor([3])
    )
    bad = dict(fresh, query_indices=fresh["query_indices"].roll(1, dims=1))
    with pytest.raises(RuntimeError, match="identity"):
        assert_runtime_candidate_identity(
            bad, base, [0], ["scene0000_00"], torch.tensor([3])
        )


def test_frozen_candidate_identity_uses_cache_axis_without_reranking():
    base = [_row()]
    base[0]["valid_mask"] = base[0].pop("candidate_valid")
    frozen = build_frozen_candidate_identity(
        base, [0], ["scene0000_00"], torch.tensor([3]), torch.device("cpu")
    )
    assert torch.equal(
        frozen["query_indices"], base[0]["query_indices"].unsqueeze(0)
    )
    assert torch.equal(
        frozen["valid_mask"], base[0]["valid_mask"].unsqueeze(0)
    )
    with pytest.raises(RuntimeError, match="source identity"):
        build_frozen_candidate_identity(
            base, [0], ["wrong_scene"], torch.tensor([3]),
            torch.device("cpu"),
        )
