import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import scripts.rec_geometry_cache as rec_geometry_cache
from models.rec_mask_geometry import (
    DEFAULT_REC_MASK_GEOMETRY_VARIANTS,
    MASK_GEOMETRY_SCHEMA_VERSION,
    REC_MASK_GEOMETRY_FEATURE_NAMES,
)
from scripts.rec_geometry_cache import (
    GEOMETRY_CACHE_SCHEMA_VERSION,
    build_base_cache_binding,
    canonical_json_sha256,
    join_base_and_geometry_rows,
    load_bound_candidate_cache,
    sha256_file,
    validate_base_cache_binding,
    validate_geometry_manifest,
    validate_geometry_row,
)


def _base_row(index, num_candidates=3):
    query_indices = torch.arange(
        10 + index * num_candidates,
        10 + (index + 1) * num_candidates,
        dtype=torch.int64,
    )
    valid_mask = torch.tensor([True, True, False], dtype=torch.bool)
    boxes = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 1.5, 2.0],
        [1.0, 2.0, 3.0, 0.5, 0.75, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)
    return {
        "dataset_index": index,
        "scan_id": "scene{:04d}_00".format(index),
        "target_id": 100 + index,
        "features": torch.zeros(num_candidates, 2, dtype=torch.float32),
        "boxes": boxes,
        "query_indices": query_indices,
        "valid_mask": valid_mask,
        "default_scores": torch.tensor(
            [2.0, 1.0, 0.0], dtype=torch.float32
        ),
        "contrastive_scores": torch.tensor(
            [0.5, 0.25, 0.0], dtype=torch.float32
        ),
        "candidate_ious": torch.tensor(
            [0.75, 0.25, 0.0], dtype=torch.float32
        ),
        "default_top1_query_index": int(query_indices[0]),
    }


def _base_manifest(rows, split="train", shard_names=None):
    if shard_names is None:
        shard_names = ["shard_000000.pt"]
    return {
        "cache_schema_version": 1,
        "feature_schema_version": "rec-query-v1",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_epoch": 71,
        "split": split,
        "candidate_rule": {
            "topk_per_source": 8,
            "max_candidates": len(rows[0]["valid_mask"]),
        },
        "feature_dim": 2,
        "feature_names": ["feature_0", "feature_1"],
        "target_iou_policy": "root_only",
        "model_inputs": {
            "use_color": True,
            "use_height": False,
            "use_multiview": False,
            "butd": True,
            "butd_gt": False,
            "butd_cls": False,
        },
        "backbone_config": {
            "model": "MCLN",
            "num_target": 256,
            "num_decoder_layers": 6,
            "self_position_embedding": "loc_learned",
            "self_attend": True,
            "use_soft_token_loss": True,
            "use_contrastive_align": True,
            "detect_intermediate": True,
            "use_source_choice_selector": True,
            "source_choice_selector_sources": "default,contrastive",
            "source_choice_selector_hidden_dim": 288,
        },
        "dataset_size": len(rows),
        "source_dataset_size": len(rows),
        "deterministic": True,
        "sample_count": len(rows),
        "shards": list(shard_names),
    }


def _write_base_cache(cache_dir, split="train", row_count=2,
                      shard_sizes=None):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True)
    rows = [_base_row(index) for index in range(row_count)]
    if shard_sizes is None:
        midpoint = max(1, row_count // 2)
        chunks = [rows[:midpoint], rows[midpoint:]]
        chunks = [chunk for chunk in chunks if chunk]
    else:
        assert sum(shard_sizes) == row_count
        chunks = []
        start = 0
        for size in shard_sizes:
            assert size > 0
            chunks.append(rows[start:start + size])
            start += size
    shard_names = []
    for index, chunk in enumerate(chunks):
        name = "shard_{:06d}.pt".format(index)
        torch.save({"rows": chunk}, cache_dir / name)
        shard_names.append(name)
    manifest = _base_manifest(rows, split=split, shard_names=shard_names)
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return rows, manifest


def test_load_bound_candidate_cache_returns_one_consistent_snapshot(tmp_path):
    cache_dir = tmp_path / "base"
    expected_rows, expected_manifest = _write_base_cache(
        cache_dir, split="val", row_count=3
    )

    rows, manifest, binding = load_bound_candidate_cache(cache_dir, "val")

    assert manifest == expected_manifest
    assert binding == build_base_cache_binding(cache_dir, "val")
    assert [row["dataset_index"] for row in rows] == [0, 1, 2]
    assert torch.equal(rows[2]["boxes"], expected_rows[2]["boxes"])


def test_load_bound_candidate_cache_fails_closed_on_split_or_content_change(
        tmp_path, monkeypatch):
    cache_dir = tmp_path / "base"
    _write_base_cache(cache_dir, split="train", row_count=2)

    with pytest.raises(ValueError, match="split"):
        load_bound_candidate_cache(cache_dir, "val")

    original_verify = rec_geometry_cache._verify_stable_cache_snapshot

    def mutate_before_verify(*args, **kwargs):
        shard_path = cache_dir / "shard_000000.pt"
        shard_path.write_bytes(shard_path.read_bytes() + b"changed")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(
        rec_geometry_cache,
        "_verify_stable_cache_snapshot",
        mutate_before_verify,
    )
    with pytest.raises(ValueError, match="changed during validation"):
        load_bound_candidate_cache(cache_dir, "train")


def _refresh_binding_content_sha256(binding):
    content = {
        key: value for key, value in binding.items()
        if key not in ("path", "content_sha256")
    }
    binding["content_sha256"] = canonical_json_sha256(content)


_IMMUTABLE_GEOMETRY_MANIFEST_FIELDS = (
    "geometry_cache_schema_version",
    "geometry_schema_version",
    "geometry_feature_names",
    "variant_names",
    "variant_configs",
    "regressed_variant_index",
    "min_points",
    "max_point_fraction",
    "split",
    "dataset_size",
    "source_dataset_size",
    "candidate_rule",
    "target_iou_policy",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_epoch",
    "model_inputs",
    "backbone_config",
    "extraction_batch_size",
    "num_workers",
    "shard_size",
    "base_cache_binding",
    "annotation_sha256",
    "audit_provenance",
    "filter_non_gt_boxes",
)
_MUTABLE_GEOMETRY_MANIFEST_FIELDS = (
    "complete",
    "sample_count",
    "shards",
    "parity_maxima",
    "cache_content_digest",
)
_GEOMETRY_IMMUTABLE_DIGEST_FIELD = "immutable_metadata_digest"


def _geometry_immutable_metadata(manifest):
    return {
        field: copy.deepcopy(manifest[field])
        for field in _IMMUTABLE_GEOMETRY_MANIFEST_FIELDS
    }


def _refresh_geometry_manifest_digests(manifest):
    """Keep synthetic manifests cryptographically valid after mutation."""
    manifest[_GEOMETRY_IMMUTABLE_DIGEST_FIELD] = canonical_json_sha256(
        _geometry_immutable_metadata(manifest)
    )
    manifest["cache_content_digest"] = canonical_json_sha256({
        key: value for key, value in manifest.items()
        if key != "cache_content_digest"
    })
    return manifest


def _geometry_manifest(binding, filter_non_gt_boxes=False):
    variants = [dict(config) for config in DEFAULT_REC_MASK_GEOMETRY_VARIANTS]
    count = binding["sample_count"]
    manifest = {
        "geometry_cache_schema_version": GEOMETRY_CACHE_SCHEMA_VERSION,
        "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "geometry_feature_names": list(REC_MASK_GEOMETRY_FEATURE_NAMES),
        "variant_names": [config["name"] for config in variants],
        "variant_configs": variants,
        "regressed_variant_index": 0,
        "min_points": 5,
        "max_point_fraction": 0.5,
        "split": binding["split"],
        "dataset_size": count,
        "source_dataset_size": count,
        "candidate_rule": copy.deepcopy(binding["candidate_rule"]),
        "target_iou_policy": "root_only",
        "checkpoint_path": "/tmp/checkpoint.pth",
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "checkpoint_epoch": 71,
        "model_inputs": copy.deepcopy(binding["model_inputs"]),
        "backbone_config": copy.deepcopy(binding["backbone_config"]),
        "extraction_batch_size": 12,
        "num_workers": 2,
        "shard_size": 252,
        "base_cache_binding": copy.deepcopy(binding),
        "annotation_sha256": "b" * 64,
        "audit_provenance": {"panel": "synthetic", "sha256": "c" * 64},
        "filter_non_gt_boxes": filter_non_gt_boxes,
        "complete": True,
        "sample_count": count,
        "shards": [{
            "name": "shard_000000.pt",
            "row_count": count,
            "sha256": "d" * 64,
        }],
        "parity_maxima": {"boxes": 0.0, "ious": 0.0},
        "cache_content_digest": "e" * 64,
    }
    return _refresh_geometry_manifest_digests(manifest)


def _geometry_row(base_row, manifest):
    candidate_valid = base_row["valid_mask"].clone()
    k = candidate_valid.numel()
    g = len(manifest["variant_names"])
    f = len(manifest["geometry_feature_names"])
    geometry_valid = candidate_valid[:, None].expand(k, g).clone()
    boxes = torch.zeros(k, g, 6, dtype=torch.float32)
    boxes[geometry_valid] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float32
    )
    boxes[:, 0] = base_row["boxes"]
    ious = torch.zeros(k, g, dtype=torch.float32)
    ious[geometry_valid] = 0.1
    ious[:, 0] = base_row["candidate_ious"]
    return {
        "dataset_index": base_row["dataset_index"],
        "scan_id": base_row["scan_id"],
        "target_id": base_row["target_id"],
        "default_top1_query_index": base_row["default_top1_query_index"],
        "query_indices": base_row["query_indices"].clone(),
        "candidate_valid": candidate_valid,
        "geometry_boxes": boxes.contiguous(),
        "geometry_valid": geometry_valid.clone().contiguous(),
        "evaluator_valid": geometry_valid.clone().contiguous(),
        "geometry_features": torch.zeros(k, g, f, dtype=torch.float32),
        "geometry_ious": ious.contiguous(),
        "source_rejection_codes": torch.zeros(k, g, dtype=torch.int16),
    }


def _noncontiguous_same_shape(tensor):
    result = torch.stack((tensor, tensor), dim=-1)[..., 0]
    assert result.shape == tensor.shape
    assert not result.is_contiguous()
    return result


def _replace_path_after_open(monkeypatch, target_path, replacement_path,
                             should_replace):
    original_open = Path.open
    target_path = Path(target_path).resolve()
    replacement_path = Path(replacement_path).resolve()
    replacements = []

    def open_then_replace(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if (not replacements
                and should_replace()
                and mode == "rb"
                and Path(path).resolve() == target_path):
            os.replace(str(replacement_path), str(target_path))
            replacements.append(True)
        return handle

    monkeypatch.setattr(Path, "open", open_then_replace)
    return replacements


@pytest.fixture
def geometry_fixture(tmp_path):
    rows, base_manifest = _write_base_cache(tmp_path / "base")
    binding = build_base_cache_binding(tmp_path / "base", "train")
    manifest = _geometry_manifest(binding)
    geometry_rows = [_geometry_row(row, manifest) for row in rows]
    return tmp_path / "base", rows, base_manifest, manifest, geometry_rows


def _geometry_build_input(tmp_path, row_count, split="train",
                          base_shard_sizes=None):
    base_dir = tmp_path / "base"
    base_rows, base_manifest = _write_base_cache(
        base_dir,
        split=split,
        row_count=row_count,
        shard_sizes=base_shard_sizes,
    )
    binding = build_base_cache_binding(base_dir, split)
    template = _geometry_manifest(binding)
    return (
        base_dir,
        base_rows,
        base_manifest,
        _geometry_immutable_metadata(template),
    )


def _geometry_rows_for_indices(base_rows, manifest, start, stop):
    return [
        _geometry_row(base_rows[index], manifest)
        for index in range(start, stop)
    ]


def test_canonical_json_sha256_is_deterministic_and_rejects_nan(tmp_path):
    payload = {"unicode": "\u96ea", "nested": {"b": 2, "a": 1}}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    assert canonical_json_sha256(payload) == hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError):
        canonical_json_sha256({"bad": float("nan")})
    with pytest.raises(ValueError):
        canonical_json_sha256({"bad": float("inf")})
    assert canonical_json_sha256([1, 2]) != canonical_json_sha256([2, 1])
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()
    with pytest.raises(ValueError, match="regular file"):
        sha256_file(tmp_path / "missing.bin")


def test_hash_live_file_rejects_path_replaced_after_open(
        tmp_path, monkeypatch):
    path = tmp_path / "bytes.bin"
    replacement_path = tmp_path / "replacement.bin"
    path.write_bytes(b"abc")
    replacement_path.write_bytes(b"abc")
    replacements = _replace_path_after_open(
        monkeypatch, path, replacement_path, lambda: True
    )

    with pytest.raises(ValueError, match="changed during validation"):
        rec_geometry_cache._hash_live_file(path)

    assert replacements == [True]


def test_read_shard_snapshot_rejects_path_replaced_after_open(
        tmp_path, monkeypatch):
    cache_dir = tmp_path / "base"
    _write_base_cache(cache_dir, row_count=2)
    shard_path = cache_dir / "shard_000000.pt"
    replacement_path = tmp_path / "replacement.pt"
    replacement_path.write_bytes(shard_path.read_bytes())
    replacements = _replace_path_after_open(
        monkeypatch, shard_path, replacement_path, lambda: True
    )

    with pytest.raises(ValueError, match="changed during validation"):
        rec_geometry_cache._read_shard_snapshot(shard_path, shard_path.name)

    assert replacements == [True]


def test_read_manifest_snapshot_rejects_path_replaced_after_open(
        tmp_path, monkeypatch):
    cache_dir = tmp_path / "base"
    _write_base_cache(cache_dir, row_count=2)
    manifest_path = cache_dir / "manifest.json"
    replacement_path = tmp_path / "replacement.json"
    replacement_path.write_bytes(manifest_path.read_bytes())
    replacements = _replace_path_after_open(
        monkeypatch, manifest_path, replacement_path, lambda: True
    )

    with pytest.raises(ValueError, match="changed during validation"):
        rec_geometry_cache._read_manifest_snapshot(cache_dir)

    assert replacements == [True]


def test_base_binding_hashes_ordered_shards_and_detects_one_byte_mutation(
        tmp_path):
    _, manifest = _write_base_cache(tmp_path / "base", row_count=4)
    binding = build_base_cache_binding(tmp_path / "base", "train")

    assert binding["manifest_sha256"] == canonical_json_sha256(manifest)
    assert [entry["row_count"] for entry in binding["shards"]] == [2, 2]
    assert validate_base_cache_binding(
        tmp_path / "base", binding, "train"
    ) == binding

    shard = tmp_path / "base" / binding["shards"][0]["name"]
    before = sha256_file(shard)
    with shard.open("ab") as handle:
        handle.write(b"x")
    assert sha256_file(shard) != before
    changed = build_base_cache_binding(tmp_path / "base", "train")
    assert changed["content_sha256"] != binding["content_sha256"]
    with pytest.raises(ValueError, match="binding"):
        validate_base_cache_binding(tmp_path / "base", binding, "train")


def test_base_binding_rejects_reordered_descriptors_and_cross_split(tmp_path):
    _write_base_cache(tmp_path / "base", split="val", row_count=4)
    binding = build_base_cache_binding(tmp_path / "base", "val")
    reordered = copy.deepcopy(binding)
    reordered["shards"].reverse()
    with pytest.raises(ValueError, match="binding"):
        validate_base_cache_binding(tmp_path / "base", reordered, "val")
    with pytest.raises(ValueError, match="split"):
        validate_base_cache_binding(tmp_path / "base", binding, "train")


def test_base_binding_requires_checkpoint_sha256(tmp_path):
    _, manifest = _write_base_cache(tmp_path / "base")
    manifest["checkpoint_sha256"] = "not-a-sha256"
    (tmp_path / "base" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="SHA-256"):
        build_base_cache_binding(tmp_path / "base", "train")


@pytest.mark.parametrize(
    "field,error",
    [
        ("binding_version", "version"),
        ("cache_schema_version", "cache schema"),
    ],
)
def test_base_binding_rejects_boolean_schema_versions(
        tmp_path, field, error):
    _write_base_cache(tmp_path / "base")
    binding = build_base_cache_binding(tmp_path / "base", "train")
    binding[field] = True
    _refresh_binding_content_sha256(binding)

    with pytest.raises(ValueError, match=error):
        validate_base_cache_binding(tmp_path / "base", binding, "train")


def test_base_binding_recomputes_caller_content_digest_before_cache_scan(
        tmp_path):
    _write_base_cache(tmp_path / "base")
    binding = build_base_cache_binding(tmp_path / "base", "train")
    binding["shards"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="content digest"):
        validate_base_cache_binding(tmp_path / "base", binding, "train")


def test_base_binding_rejects_manifest_only_without_shard_descriptors(tmp_path):
    _write_base_cache(tmp_path / "base")
    binding = build_base_cache_binding(tmp_path / "base", "train")
    binding.pop("shards")
    _refresh_binding_content_sha256(binding)

    with pytest.raises(ValueError, match="binding"):
        validate_base_cache_binding(tmp_path / "base", binding, "train")


def test_base_binding_rejects_rehashed_tampered_shard_row_counts(tmp_path):
    _write_base_cache(tmp_path / "base", row_count=4)
    binding = build_base_cache_binding(tmp_path / "base", "train")
    binding["shards"][0]["row_count"] += 1
    binding["shards"][1]["row_count"] -= 1
    _refresh_binding_content_sha256(binding)

    with pytest.raises(ValueError, match="cache contents"):
        validate_base_cache_binding(tmp_path / "base", binding, "train")


def test_base_binding_rejects_aba_shard_snapshot_mismatch(
        tmp_path, monkeypatch):
    cache_dir = tmp_path / "base"
    _write_base_cache(cache_dir, row_count=2)
    shard_path = cache_dir / "shard_000000.pt"
    original_bytes = shard_path.read_bytes()
    replacement_path = tmp_path / "replacement.pt"
    replacement_payload = torch.load(shard_path, map_location="cpu")
    replacement_payload["rows"][0]["features"][0, 0] += 1.0
    torch.save(replacement_payload, replacement_path)
    replacement_bytes = replacement_path.read_bytes()
    original_read_shard_snapshot = rec_geometry_cache._read_shard_snapshot
    original_torch_load = rec_geometry_cache.torch.load
    restore_calls = []
    deserialization_inputs = []
    observed_snapshot_digests = []
    observed_features = []

    def read_and_record_snapshot(path, shard_name):
        if Path(path).resolve() != shard_path.resolve():
            return original_read_shard_snapshot(path, shard_name)
        snapshot, rows, identity = original_read_shard_snapshot(
            path, shard_name
        )
        observed_snapshot_digests.append(hashlib.sha256(snapshot).hexdigest())
        observed_features.append(rows[0]["features"].clone())
        return snapshot, rows, identity

    def restore_a_before_deserialization(source, *args, **kwargs):
        if not restore_calls:
            restore_calls.append(True)
            deserialization_inputs.append((
                isinstance(source, io.BytesIO),
                source.getvalue() if isinstance(source, io.BytesIO) else None,
            ))
            shard_path.write_bytes(original_bytes)
        return original_torch_load(source, *args, **kwargs)

    monkeypatch.setattr(
        rec_geometry_cache,
        "_read_shard_snapshot",
        read_and_record_snapshot,
    )
    monkeypatch.setattr(
        rec_geometry_cache.torch,
        "load",
        restore_a_before_deserialization,
    )

    shard_path.write_bytes(replacement_bytes)
    try:
        with pytest.raises(ValueError, match="changed during validation"):
            build_base_cache_binding(cache_dir, "train")
    finally:
        shard_path.write_bytes(original_bytes)

    assert restore_calls == [True]
    assert len(deserialization_inputs) == 1
    is_snapshot_buffer, deserialized_bytes = deserialization_inputs[0]
    assert is_snapshot_buffer
    assert deserialized_bytes == replacement_bytes
    assert observed_snapshot_digests == [hashlib.sha256(replacement_bytes).hexdigest()]
    assert torch.equal(
        observed_features[0], replacement_payload["rows"][0]["features"]
    )
    assert shard_path.read_bytes() == original_bytes


def test_base_binding_rejects_shard_replaced_after_snapshot(
        tmp_path, monkeypatch):
    cache_dir = tmp_path / "base"
    _write_base_cache(cache_dir, row_count=4)
    original_snapshot = rec_geometry_cache._snapshot_candidate_cache

    def snapshot_then_replace_shard(path, manifest, expected_split):
        descriptors, rows, identities = original_snapshot(
            path, manifest, expected_split
        )
        shard_path = Path(path) / manifest["shards"][0]
        payload = torch.load(shard_path, map_location="cpu")
        payload["rows"][0]["features"][0, 0] += 1.0
        torch.save(payload, shard_path)
        return descriptors, rows, identities

    monkeypatch.setattr(
        rec_geometry_cache, "_snapshot_candidate_cache",
        snapshot_then_replace_shard,
    )

    with pytest.raises(ValueError, match="changed during validation"):
        build_base_cache_binding(cache_dir, "train")


def test_base_binding_rejects_shard_replaced_after_final_hash(
        tmp_path, monkeypatch):
    cache_dir = tmp_path / "base"
    _write_base_cache(cache_dir, row_count=4)
    shard_path = cache_dir / "shard_000000.pt"
    replacement_path = tmp_path / "replacement.pt"
    replacement_path.write_bytes(shard_path.read_bytes())
    original_hash = rec_geometry_cache.sha256_file
    replaced = []

    def hash_then_replace(path):
        digest = original_hash(path)
        if Path(path).name == "shard_000001.pt" and not replaced:
            replaced.append(True)
            os.replace(str(replacement_path), str(shard_path))
        return digest

    monkeypatch.setattr(rec_geometry_cache, "sha256_file", hash_then_replace)

    with pytest.raises(ValueError, match="changed during validation"):
        build_base_cache_binding(cache_dir, "train")


def test_base_binding_rejects_manifest_rewritten_after_snapshot(
        tmp_path, monkeypatch):
    cache_dir = tmp_path / "base"
    _write_base_cache(cache_dir, row_count=2)
    original_snapshot = rec_geometry_cache._snapshot_candidate_cache

    def snapshot_then_rewrite_manifest(path, manifest, expected_split):
        result = original_snapshot(path, manifest, expected_split)
        (Path(path) / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result

    monkeypatch.setattr(
        rec_geometry_cache,
        "_snapshot_candidate_cache",
        snapshot_then_rewrite_manifest,
    )

    with pytest.raises(ValueError, match="changed during validation"):
        build_base_cache_binding(cache_dir, "train")


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_target", 256.0),
        ("self_attend", 1),
    ],
)
def test_base_binding_rejects_backbone_provenance_type_substitution(
        tmp_path, field, value):
    cache_dir = tmp_path / "base"
    _, manifest = _write_base_cache(cache_dir)
    manifest["backbone_config"][field] = value
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="backbone"):
        build_base_cache_binding(cache_dir, "train")


def test_geometry_manifest_accepts_complete_synthetic_cache(geometry_fixture):
    _, _, _, manifest, _ = geometry_fixture
    assert validate_geometry_manifest(manifest, "train", True) == manifest


def test_geometry_manifest_accepts_portable_extraction_runtime(
        geometry_fixture):
    manifest = copy.deepcopy(geometry_fixture[3])
    manifest.update({
        "extraction_batch_size": 36,
        "num_workers": 4,
        "shard_size": 252,
    })
    rec_geometry_cache._refresh_geometry_manifest_digests(manifest)

    assert validate_geometry_manifest(manifest, "train", True) == manifest


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda value: value.pop("annotation_sha256"), "annotation"),
        (lambda value: value.update(geometry_schema_version="other"), "schema"),
        (lambda value: value.update(split="val"), "split"),
        (lambda value: value.update(extraction_batch_size=8), "batch"),
        (lambda value: value.update(shard_size=250), "shard size"),
        (lambda value: value.update(complete=False), "complete"),
    ],
)
def test_geometry_manifest_rejects_missing_or_inconsistent_fields(
        geometry_fixture, mutation, error):
    manifest = copy.deepcopy(geometry_fixture[3])
    mutation(manifest)
    with pytest.raises(ValueError, match=error):
        validate_geometry_manifest(manifest, "train", True)


@pytest.mark.parametrize("field", _IMMUTABLE_GEOMETRY_MANIFEST_FIELDS)
def test_geometry_manifest_requires_every_immutable_field(
        geometry_fixture, field):
    manifest = copy.deepcopy(geometry_fixture[3])
    manifest.pop(field)

    with pytest.raises(ValueError):
        validate_geometry_manifest(manifest, "train", True)


@pytest.mark.parametrize("field", _MUTABLE_GEOMETRY_MANIFEST_FIELDS)
def test_geometry_manifest_requires_every_mutable_field(
        geometry_fixture, field):
    manifest = copy.deepcopy(geometry_fixture[3])
    manifest.pop(field)

    with pytest.raises(ValueError):
        validate_geometry_manifest(manifest, "train", True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("geometry_cache_schema_version", True),
        ("regressed_variant_index", False),
        ("extraction_batch_size", 12.0),
        ("num_workers", 2.0),
        ("shard_size", 252.0),
    ],
)
def test_geometry_manifest_fixed_integers_are_strict_ints(
        geometry_fixture, field, value):
    manifest = copy.deepcopy(geometry_fixture[3])
    manifest[field] = value

    with pytest.raises(ValueError):
        validate_geometry_manifest(manifest, "train", True)


def test_geometry_manifest_source_dataset_size_is_strict_int(geometry_fixture):
    manifest = copy.deepcopy(geometry_fixture[3])
    manifest["source_dataset_size"] = float(manifest["dataset_size"])

    with pytest.raises(ValueError, match="source dataset size"):
        validate_geometry_manifest(manifest, "train", True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_target", 256.0),
        ("self_attend", 1),
    ],
)
def test_geometry_manifest_rejects_backbone_provenance_type_substitution(
        geometry_fixture, field, value):
    manifest = copy.deepcopy(geometry_fixture[3])
    manifest["backbone_config"][field] = value

    with pytest.raises(ValueError, match="backbone"):
        validate_geometry_manifest(manifest, "train", True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("logit_threshold", False),
        ("quantile", False),
        ("regressed_weight", True),
    ],
)
def test_geometry_manifest_variant_configs_use_type_sensitive_equality(
        geometry_fixture, field, value):
    manifest = copy.deepcopy(geometry_fixture[3])
    manifest["variant_configs"][0][field] = value

    with pytest.raises(ValueError, match="variant configs"):
        validate_geometry_manifest(manifest, "train", True)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "nonzero"])
def test_geometry_manifest_requires_unique_g0_regressed_variant(
        geometry_fixture, mode):
    manifest = copy.deepcopy(geometry_fixture[3])
    if mode == "missing":
        manifest["variant_configs"][0]["source"] = "fused"
    elif mode == "duplicate":
        manifest["variant_configs"][1]["source"] = "regressed"
    else:
        manifest["regressed_variant_index"] = 1
    with pytest.raises(ValueError, match="regressed"):
        validate_geometry_manifest(manifest, "train", True)


def test_geometry_row_accepts_exact_cpu_contiguous_schema(geometry_fixture):
    _, base_rows, _, manifest, geometry_rows = geometry_fixture
    row = geometry_rows[0]
    assert validate_geometry_row(row, manifest, base_rows[0]) == row
    assert row["query_indices"].dtype == torch.int64
    assert row["candidate_valid"].dtype == torch.bool
    assert row["geometry_boxes"].dtype == torch.float32
    assert row["geometry_features"].shape[-1] == 25
    assert row["source_rejection_codes"].dtype == torch.int16
    assert all(
        value.device.type == "cpu" and value.is_contiguous()
        for value in row.values() if isinstance(value, torch.Tensor)
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("features", torch.zeros(3, 2), "base payload"),
        ("default_scores", torch.zeros(3), "base payload"),
        ("threshold_labels", torch.zeros(3, 7, 2), "base payload"),
    ],
)
def test_geometry_row_rejects_duplicated_base_or_target_payload(
        geometry_fixture, field, value, error):
    row = dict(geometry_fixture[4][0])
    row[field] = value
    with pytest.raises(ValueError, match=error):
        validate_geometry_row(row, geometry_fixture[3])


@pytest.mark.parametrize(
    "field,mutation",
    [
        ("query_indices", lambda tensor: tensor[:-1]),
        ("geometry_boxes", lambda tensor: tensor[:, :-1, :]),
        ("geometry_features", lambda tensor: tensor[..., :-1]),
    ],
    ids=["candidate-axis-k", "variant-axis-g", "feature-axis-f"],
)
def test_geometry_row_rejects_wrong_k_g_f_shapes(
        geometry_fixture, field, mutation):
    row = copy.deepcopy(geometry_fixture[4][0])
    row[field] = mutation(row[field]).contiguous()

    with pytest.raises(ValueError, match="shape"):
        validate_geometry_row(row, geometry_fixture[3])


@pytest.mark.parametrize(
    "field,dtype",
    [
        ("query_indices", torch.int32),
        ("candidate_valid", torch.uint8),
        ("geometry_boxes", torch.float64),
        ("geometry_valid", torch.uint8),
        ("evaluator_valid", torch.uint8),
        ("geometry_features", torch.float64),
        ("geometry_ious", torch.float64),
        ("source_rejection_codes", torch.int32),
    ],
)
def test_geometry_row_rejects_wrong_tensor_dtypes(
        geometry_fixture, field, dtype):
    row = copy.deepcopy(geometry_fixture[4][0])
    row[field] = row[field].to(dtype=dtype)

    with pytest.raises(ValueError, match="dtype"):
        validate_geometry_row(row, geometry_fixture[3])


@pytest.mark.parametrize(
    "field",
    [
        "query_indices",
        "candidate_valid",
        "geometry_boxes",
        "geometry_valid",
        "evaluator_valid",
        "geometry_features",
        "geometry_ious",
        "source_rejection_codes",
    ],
)
def test_geometry_row_rejects_noncontiguous_tensor_storage(
        geometry_fixture, field):
    row = copy.deepcopy(geometry_fixture[4][0])
    row[field] = _noncontiguous_same_shape(row[field])

    with pytest.raises(ValueError, match="contiguous"):
        validate_geometry_row(row, geometry_fixture[3])


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda row: row["geometry_features"].__setitem__(
            (0, 0, 0), float("nan")), "finite"),
        (lambda row: row["geometry_boxes"].__setitem__(
            (0, 1, 0), float("nan")), "finite"),
        (lambda row: row["geometry_ious"].__setitem__(
            (0, 1), float("inf")), "finite"),
        (lambda row: row["geometry_boxes"].__setitem__(
            (0, 1, slice(3, 6)), 0.0), "positive"),
        (lambda row: row["geometry_ious"].__setitem__(
            (0, 1), 1.1), "IoU"),
        (lambda row: row["geometry_ious"].__setitem__(
            (2, 1), 0.1), "invalid"),
        (lambda row: row["evaluator_valid"].__setitem__(
            (2, 1), True), "evaluator"),
    ],
)
def test_geometry_row_rejects_nonfinite_boxes_ious_and_evaluator_masks(
        geometry_fixture, mutation, error):
    row = copy.deepcopy(geometry_fixture[4][0])
    mutation(row)
    with pytest.raises(ValueError, match=error):
        validate_geometry_row(row, geometry_fixture[3])


def test_geometry_row_requires_evaluator_policy_and_nonempty_selection(
        geometry_fixture):
    manifest = geometry_fixture[3]
    row = copy.deepcopy(geometry_fixture[4][0])
    row["evaluator_valid"][0, 1] = False
    with pytest.raises(ValueError, match="equal geometry"):
        validate_geometry_row(row, manifest)

    filtered_manifest = copy.deepcopy(manifest)
    filtered_manifest["filter_non_gt_boxes"] = True
    _refresh_geometry_manifest_digests(filtered_manifest)
    row["evaluator_valid"].zero_()
    with pytest.raises(ValueError, match="at least one"):
        validate_geometry_row(row, filtered_manifest)


def test_geometry_row_rejects_query_and_regressed_validity_mismatches(
        geometry_fixture):
    manifest = geometry_fixture[3]
    row = copy.deepcopy(geometry_fixture[4][0])
    row["query_indices"][1] = row["query_indices"][0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_geometry_row(row, manifest)

    row = copy.deepcopy(geometry_fixture[4][0])
    row["geometry_valid"][1, 0] = False
    row["evaluator_valid"][1, 0] = False
    with pytest.raises(ValueError, match="regressed"):
        validate_geometry_row(row, manifest)


def test_geometry_row_requires_bit_exact_g0_base_clone(geometry_fixture):
    _, base_rows, _, manifest, geometry_rows = geometry_fixture
    row = copy.deepcopy(geometry_rows[0])
    row["geometry_boxes"][0, 0, 0] += 1e-7
    with pytest.raises(ValueError, match="regressed boxes"):
        validate_geometry_row(row, manifest, base_rows[0])

    row = copy.deepcopy(geometry_rows[0])
    row["geometry_ious"][0, 0] += 1e-7
    with pytest.raises(ValueError, match="regressed IoUs"):
        validate_geometry_row(row, manifest, base_rows[0])


def test_join_returns_noncopying_pairs_and_rejects_identity_or_query_mismatch(
        geometry_fixture):
    _, base_rows, base_manifest, manifest, geometry_rows = geometry_fixture
    joined = join_base_and_geometry_rows(
        base_rows, geometry_rows, base_manifest, manifest
    )
    assert joined[0]["base"] is base_rows[0]
    assert joined[0]["geometry"] is geometry_rows[0]

    bad_rows = copy.deepcopy(geometry_rows)
    bad_rows[0]["scan_id"] = "other-scene"
    with pytest.raises(ValueError, match="identity"):
        join_base_and_geometry_rows(
            base_rows, bad_rows, base_manifest, manifest
        )

    bad_rows = copy.deepcopy(geometry_rows)
    bad_rows[0]["query_indices"][0] += 100
    with pytest.raises(ValueError, match="query"):
        join_base_and_geometry_rows(
            base_rows, bad_rows, base_manifest, manifest
        )


def test_join_rejects_base_manifest_provenance_mismatch(geometry_fixture):
    _, base_rows, base_manifest, manifest, geometry_rows = geometry_fixture
    changed = copy.deepcopy(base_manifest)
    changed["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="manifest"):
        join_base_and_geometry_rows(
            base_rows, geometry_rows, changed, manifest
        )


def test_join_rejects_rows_changed_together_after_base_cache_load(
        geometry_fixture):
    _, base_rows, base_manifest, manifest, geometry_rows = geometry_fixture
    changed_base_rows = copy.deepcopy(base_rows)
    changed_geometry_rows = copy.deepcopy(geometry_rows)
    changed_base_rows[0]["target_id"] += 1
    changed_geometry_rows[0]["target_id"] += 1

    with pytest.raises(ValueError, match="bound base cache"):
        join_base_and_geometry_rows(
            changed_base_rows,
            changed_geometry_rows,
            base_manifest,
            manifest,
        )


def test_join_rejects_checkpoint_epoch_mismatch(geometry_fixture):
    _, base_rows, base_manifest, manifest, geometry_rows = geometry_fixture
    changed = copy.deepcopy(manifest)
    changed["checkpoint_epoch"] += 1
    _refresh_geometry_manifest_digests(changed)

    with pytest.raises(ValueError, match="checkpoint epoch"):
        join_base_and_geometry_rows(
            base_rows, geometry_rows, base_manifest, changed
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows, manifest: rows[0]["candidate_valid"].__setitem__(1, False),
        lambda rows, manifest: rows[0].update(
            default_top1_query_index=int(rows[0]["query_indices"][1])
        ),
        lambda rows, manifest: manifest["candidate_rule"].update(
            topk_per_source=7
        ),
        lambda rows, manifest: manifest.update(checkpoint_sha256="f" * 64),
        lambda rows, manifest: manifest["model_inputs"].update(
            use_color=False
        ),
        lambda rows, manifest: manifest["backbone_config"].update(
            num_target=128
        ),
        lambda rows, manifest: manifest.update(
            target_iou_policy="all_targets"
        ),
    ],
    ids=[
        "candidate-valid",
        "default-query",
        "candidate-rule",
        "checkpoint-sha",
        "model-inputs",
        "backbone-config",
        "target-policy",
    ],
)
def test_join_rejects_all_row_and_manifest_provenance_mismatches(
        geometry_fixture, mutation):
    _, base_rows, base_manifest, manifest, geometry_rows = geometry_fixture
    changed_rows = copy.deepcopy(geometry_rows)
    changed_manifest = copy.deepcopy(manifest)
    mutation(changed_rows, changed_manifest)

    with pytest.raises(ValueError):
        join_base_and_geometry_rows(
            base_rows, changed_rows, base_manifest, changed_manifest
        )


def test_geometry_cache_build_is_hidden_until_complete_and_final_noops(
        tmp_path):
    _, base_rows, _, metadata = _geometry_build_input(tmp_path, row_count=2)
    output_dir = tmp_path / "geometry"

    building = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )

    assert building["complete"] is False
    assert building["sample_count"] == 0
    assert not output_dir.exists()
    assert (tmp_path / "geometry.building" / "manifest.json").is_file()
    with pytest.raises(ValueError, match="directory|complete"):
        rec_geometry_cache.load_geometry_cache(output_dir, "train")

    finalized = rec_geometry_cache.finalize_geometry_cache(
        output_dir,
        building,
        _geometry_rows_for_indices(base_rows, building, 0, 2),
        {"boxes": 0.0, "ious": 0.0},
    )
    loaded_rows, loaded_manifest = rec_geometry_cache.load_geometry_cache(
        output_dir, "train"
    )
    assert len(loaded_rows) == 2
    assert loaded_manifest == finalized

    no_op = rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)
    assert no_op == finalized
    assert not (tmp_path / "geometry.building").exists()

    replacement = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata, overwrite=True
    )
    assert replacement["complete"] is False
    assert (tmp_path / "geometry.building").is_dir()
    assert rec_geometry_cache.load_geometry_cache(output_dir, "train")[1] == (
        finalized
    )


def test_preloaded_bound_base_snapshot_avoids_a_second_base_cache_read(
        tmp_path, monkeypatch):
    base_dir, base_rows, _, metadata = _geometry_build_input(
        tmp_path, row_count=2, split="val"
    )
    geometry_dir = tmp_path / "geometry"
    building = rec_geometry_cache.initialize_geometry_cache(
        geometry_dir, metadata
    )
    rec_geometry_cache.finalize_geometry_cache(
        geometry_dir,
        building,
        _geometry_rows_for_indices(base_rows, building, 0, 2),
        {"boxes": 0.0, "ious": 0.0},
    )
    loaded_base, base_manifest, binding = (
        rec_geometry_cache.load_bound_candidate_cache(base_dir, "val")
    )

    def reject_second_base_read(*_args, **_kwargs):
        raise AssertionError("bound base cache must be read exactly once")

    monkeypatch.setattr(
        rec_geometry_cache, "_build_base_cache_binding", reject_second_base_read
    )
    geometry_rows, geometry_manifest = rec_geometry_cache.load_geometry_cache(
        geometry_dir,
        "val",
        base_snapshot=(loaded_base, base_manifest, binding),
    )
    joined = rec_geometry_cache.join_base_and_geometry_rows(
        loaded_base,
        geometry_rows,
        base_manifest,
        geometry_manifest,
        verified_base_binding=binding,
    )

    assert [row["base"]["dataset_index"] for row in joined] == [0, 1]


@pytest.mark.parametrize(
    "split,row_count,tail",
    [
        ("train", 252 + 125, 125),
        ("val", 252 + 184, 184),
    ],
)
def test_geometry_cache_enforces_full_append_and_terminal_tail(
        tmp_path, split, row_count, tail):
    _, base_rows, _, metadata = _geometry_build_input(
        tmp_path, row_count=row_count, split=split
    )
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )

    assert manifest["shard_size"] % manifest["extraction_batch_size"] == 0
    with pytest.raises(ValueError, match="252|shard"):
        rec_geometry_cache.append_geometry_shard(
            output_dir,
            manifest,
            _geometry_rows_for_indices(base_rows, manifest, 0, 251),
            {"boxes": 0.0, "ious": 0.0},
        )

    manifest = rec_geometry_cache.append_geometry_shard(
        output_dir,
        manifest,
        _geometry_rows_for_indices(base_rows, manifest, 0, 252),
        {"boxes": 0.1, "ious": 0.01},
    )
    assert manifest["sample_count"] == 252
    assert manifest["complete"] is False

    with pytest.raises(ValueError, match="252|final"):
        rec_geometry_cache.append_geometry_shard(
            output_dir,
            manifest,
            _geometry_rows_for_indices(base_rows, manifest, 252, row_count),
            {"boxes": 0.1, "ious": 0.01},
        )
    with pytest.raises(ValueError, match="remaining|tail|row"):
        rec_geometry_cache.finalize_geometry_cache(
            output_dir,
            manifest,
            _geometry_rows_for_indices(base_rows, manifest, 252, row_count - 1),
            {"boxes": 0.1, "ious": 0.01},
        )

    finalized = rec_geometry_cache.finalize_geometry_cache(
        output_dir,
        manifest,
        _geometry_rows_for_indices(base_rows, manifest, 252, row_count),
        {"boxes": 0.1, "ious": 0.01},
    )
    assert finalized["complete"] is True
    assert finalized["sample_count"] == row_count
    assert [entry["row_count"] for entry in finalized["shards"]] == [252, tail]


def test_append_cannot_commit_an_exact_terminal_full_shard(tmp_path):
    _, base_rows, _, metadata = _geometry_build_input(tmp_path, row_count=252)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )

    with pytest.raises(ValueError, match="final|terminal"):
        rec_geometry_cache.append_geometry_shard(
            output_dir,
            manifest,
            _geometry_rows_for_indices(base_rows, manifest, 0, 252),
            {"boxes": 0.0, "ious": 0.0},
        )

    finalized = rec_geometry_cache.finalize_geometry_cache(
        output_dir,
        manifest,
        _geometry_rows_for_indices(base_rows, manifest, 0, 252),
        {"boxes": 0.0, "ious": 0.0},
    )
    assert finalized["complete"] is True
    assert finalized["shards"][0]["row_count"] == 252


def test_finalize_and_publication_do_not_retain_full_validated_row_lists(
        tmp_path, monkeypatch):
    _, base_rows, _, metadata = _geometry_build_input(tmp_path, row_count=2)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    retain_rows_calls = []
    original_validate = (
        rec_geometry_cache._validate_complete_geometry_directory
    )

    def record_retention(*args, **kwargs):
        retain_rows_calls.append(kwargs.get("retain_rows", True))
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(
        rec_geometry_cache,
        "_validate_complete_geometry_directory",
        record_retention,
    )

    finalized = rec_geometry_cache.finalize_geometry_cache(
        output_dir,
        manifest,
        _geometry_rows_for_indices(base_rows, manifest, 0, 2),
        {"boxes": 0.0, "ious": 0.0},
    )

    assert finalized["complete"] is True
    assert len(retain_rows_calls) >= 3
    assert retain_rows_calls == [False] * len(retain_rows_calls)


def test_geometry_manifest_digests_bind_descriptors_and_loader_files(tmp_path):
    _, base_rows, _, metadata = _geometry_build_input(tmp_path, row_count=2)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    finalized = rec_geometry_cache.finalize_geometry_cache(
        output_dir,
        manifest,
        _geometry_rows_for_indices(base_rows, manifest, 0, 2),
        {"boxes": 0.0, "ious": 0.0},
    )

    assert finalized[_GEOMETRY_IMMUTABLE_DIGEST_FIELD] == (
        canonical_json_sha256(metadata)
    )
    assert finalized["cache_content_digest"] == canonical_json_sha256({
        key: value for key, value in finalized.items()
        if key != "cache_content_digest"
    })

    manifest_path = output_dir / "manifest.json"
    pristine_manifest = manifest_path.read_bytes()
    tampered = json.loads(pristine_manifest.decode("utf-8"))
    tampered["shards"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content digest"):
        rec_geometry_cache.load_geometry_cache(output_dir, "train")

    manifest_path.write_bytes(pristine_manifest)
    shard_path = output_dir / "shard_000000.pt"
    shard_bytes = shard_path.read_bytes()
    shard_path.unlink()
    with pytest.raises(ValueError, match="shard"):
        rec_geometry_cache.load_geometry_cache(output_dir, "train")

    shard_path.write_bytes(shard_bytes)
    (output_dir / "shard_000001.pt").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="shard|unexpected"):
        rec_geometry_cache.load_geometry_cache(output_dir, "train")


def test_append_publishes_shard_before_manifest_and_rolls_back_on_failure(
        tmp_path, monkeypatch):
    _, base_rows, _, metadata = _geometry_build_input(tmp_path, row_count=377)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    original_replace = rec_geometry_cache.os.replace
    calls = []

    def record_replace(source, destination):
        calls.append((Path(source).name, Path(destination).name))
        return original_replace(source, destination)

    monkeypatch.setattr(rec_geometry_cache.os, "replace", record_replace)
    stale_manifest = copy.deepcopy(manifest)
    manifest = rec_geometry_cache.append_geometry_shard(
        output_dir,
        manifest,
        _geometry_rows_for_indices(base_rows, manifest, 0, 252),
        {"boxes": 0.0, "ious": 0.0},
    )
    shard_publish = calls.index(("shard_000000.pt.tmp", "shard_000000.pt"))
    manifest_publish = calls.index(("manifest.json.tmp", "manifest.json"))
    assert shard_publish < manifest_publish

    with pytest.raises(ValueError, match="stale|manifest"):
        rec_geometry_cache.append_geometry_shard(
            output_dir,
            stale_manifest,
            _geometry_rows_for_indices(base_rows, manifest, 252, 377),
            {"boxes": 0.0, "ious": 0.0},
        )


def test_append_rolls_back_only_a_not_yet_manifested_shard(tmp_path, monkeypatch):
    _, base_rows, _, metadata = _geometry_build_input(tmp_path, row_count=377)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    original_replace = rec_geometry_cache.os.replace

    def fail_manifest_replace(source, destination):
        if (Path(source).name == "manifest.json.tmp"
                and Path(destination).name == "manifest.json"):
            raise OSError("simulated manifest failure")
        return original_replace(source, destination)

    monkeypatch.setattr(
        rec_geometry_cache.os, "replace", fail_manifest_replace
    )
    with pytest.raises(OSError, match="simulated manifest failure"):
        rec_geometry_cache.append_geometry_shard(
            output_dir,
            manifest,
            _geometry_rows_for_indices(base_rows, manifest, 0, 252),
            {"boxes": 0.0, "ious": 0.0},
        )
    building_dir = tmp_path / "geometry.building"
    assert not (building_dir / "shard_000000.pt").exists()
    on_disk = json.loads((building_dir / "manifest.json").read_text(
        encoding="utf-8"
    ))
    assert on_disk["shards"] == []


def test_append_preserves_a_shard_when_manifest_commit_is_ambiguous(
        tmp_path, monkeypatch):
    _, base_rows, _, metadata = _geometry_build_input(tmp_path, row_count=377)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    original_replace = rec_geometry_cache.os.replace

    def replace_then_report_failure(source, destination):
        result = original_replace(source, destination)
        if (Path(source).name == "manifest.json.tmp"
                and Path(destination).name == "manifest.json"):
            raise OSError("ambiguous manifest commit")
        return result

    monkeypatch.setattr(
        rec_geometry_cache.os, "replace", replace_then_report_failure
    )
    committed = rec_geometry_cache.append_geometry_shard(
        output_dir,
        manifest,
        _geometry_rows_for_indices(base_rows, manifest, 0, 252),
        {"boxes": 0.0, "ious": 0.0},
    )
    assert committed["sample_count"] == 252
    assert (tmp_path / "geometry.building" / "shard_000000.pt").is_file()


def test_resume_removes_only_the_exact_next_orphan_and_rejects_future_one(
        tmp_path):
    _, _, _, metadata = _geometry_build_input(tmp_path, row_count=377)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    assert manifest["sample_count"] == 0
    building_dir = tmp_path / "geometry.building"
    next_orphan = building_dir / "shard_000000.pt"
    next_orphan.write_bytes(b"interrupted write")

    resumed = rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)
    assert resumed["sample_count"] == 0
    assert not next_orphan.exists()

    future_orphan = building_dir / "shard_000001.pt"
    future_orphan.write_bytes(b"must not be cleaned")
    with pytest.raises(ValueError, match="orphan|unexpected|future"):
        rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)
    assert future_orphan.is_file()


def test_geometry_loader_checks_payload_header_and_joins_base_rows_by_index(
        tmp_path):
    _, base_rows, base_manifest, metadata = _geometry_build_input(
        tmp_path,
        row_count=377,
        base_shard_sizes=[256, 121],
    )
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    manifest = rec_geometry_cache.append_geometry_shard(
        output_dir,
        manifest,
        _geometry_rows_for_indices(base_rows, manifest, 0, 252),
        {"boxes": 0.0, "ious": 0.0},
    )
    finalized = rec_geometry_cache.finalize_geometry_cache(
        output_dir,
        manifest,
        _geometry_rows_for_indices(base_rows, manifest, 252, 377),
        {"boxes": 0.0, "ious": 0.0},
    )
    geometry_rows, loaded_manifest = rec_geometry_cache.load_geometry_cache(
        output_dir, "train"
    )
    joined = join_base_and_geometry_rows(
        base_rows, geometry_rows, base_manifest, loaded_manifest
    )
    assert len(joined) == 377
    assert joined[251]["geometry"]["dataset_index"] == 251
    assert joined[252]["base"]["dataset_index"] == 252

    shard_path = output_dir / "shard_000001.pt"
    payload = torch.load(shard_path, map_location="cpu")
    payload["row_start"] += 1
    torch.save(payload, shard_path)
    changed = copy.deepcopy(finalized)
    changed["shards"][1]["sha256"] = sha256_file(shard_path)
    _refresh_geometry_manifest_digests(changed)
    (output_dir / "manifest.json").write_text(
        json.dumps(changed), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="row|header|range"):
        rec_geometry_cache.load_geometry_cache(output_dir, "train")


def test_join_resolves_rows_by_dataset_index_not_caller_list_position(
        geometry_fixture):
    _, base_rows, base_manifest, manifest, geometry_rows = geometry_fixture
    joined = join_base_and_geometry_rows(
        list(reversed(base_rows)),
        list(reversed(geometry_rows)),
        base_manifest,
        manifest,
    )

    assert [pair["base"]["dataset_index"] for pair in joined] == [0, 1]
    assert [pair["geometry"]["dataset_index"] for pair in joined] == [0, 1]


def test_second_full_append_skips_full_base_rescan(tmp_path, monkeypatch):
    _, base_rows, _, metadata = _geometry_build_input(tmp_path, row_count=629)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    manifest = rec_geometry_cache.append_geometry_shard(
        output_dir,
        manifest,
        _geometry_rows_for_indices(base_rows, manifest, 0, 252),
        {"boxes": 0.0, "ious": 0.0},
    )

    def fail_if_base_is_rescanned(_manifest):
        raise AssertionError("append must not rescan every bound base row")

    def fail_if_prior_geometry_is_loaded(*_args, **_kwargs):
        raise AssertionError("append must not deserialize prior geometry shards")

    monkeypatch.setattr(
        rec_geometry_cache, "_load_bound_base_rows", fail_if_base_is_rescanned
    )
    monkeypatch.setattr(
        rec_geometry_cache,
        "_validate_geometry_bundle",
        fail_if_prior_geometry_is_loaded,
    )
    monkeypatch.setattr(
        rec_geometry_cache.torch, "load", fail_if_prior_geometry_is_loaded
    )
    manifest = rec_geometry_cache.append_geometry_shard(
        output_dir,
        manifest,
        _geometry_rows_for_indices(base_rows, manifest, 252, 504),
        {"boxes": 0.0, "ious": 0.0},
    )
    assert manifest["sample_count"] == 504


def test_unlocked_persistent_build_lock_does_not_block_resume(tmp_path):
    _, _, _, metadata = _geometry_build_input(tmp_path, row_count=377)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    lock_path = tmp_path / "geometry.building.lock"

    assert lock_path.is_file()
    resumed = rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)
    assert resumed == manifest


def test_active_build_lock_rejects_a_second_writer(tmp_path):
    _, _, _, metadata = _geometry_build_input(tmp_path, row_count=377)
    output_dir = tmp_path / "geometry"
    rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)
    lock_path = tmp_path / "geometry.building.lock"
    descriptor = os.open(str(lock_path), os.O_RDWR)
    try:
        rec_geometry_cache.fcntl.flock(
            descriptor,
            rec_geometry_cache.fcntl.LOCK_EX | rec_geometry_cache.fcntl.LOCK_NB,
        )
        with pytest.raises(ValueError, match="lock"):
            rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)
    finally:
        rec_geometry_cache.fcntl.flock(descriptor, rec_geometry_cache.fcntl.LOCK_UN)
        os.close(descriptor)


@pytest.mark.parametrize(
    "reserved_name",
    [
        "geometry.building",
        "geometry.backup",
        "geometry.publish.json",
        "geometry.publish.json.tmp",
        "geometry.building.lock",
    ],
)
def test_public_cache_apis_reject_reserved_final_suffixes(
        tmp_path, reserved_name):
    _, _, _, metadata = _geometry_build_input(tmp_path, row_count=2)
    output_dir = tmp_path / reserved_name

    with pytest.raises(ValueError, match="reserved"):
        rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)
    with pytest.raises(ValueError, match="reserved"):
        rec_geometry_cache.load_geometry_cache(output_dir, "train")


def test_child_process_exit_releases_persistent_build_lock(tmp_path):
    _, _, _, metadata = _geometry_build_input(tmp_path, row_count=377)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    lock_path = tmp_path / "geometry.building.lock"
    child_script = (
        "import fcntl, os, sys; "
        "fd = os.open(sys.argv[1], os.O_RDWR); "
        "fcntl.flock(fd, fcntl.LOCK_EX); "
        "os.ftruncate(fd, 0); os.write(fd, b'crashed'); os.fsync(fd); "
        "os._exit(0)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", child_script, str(lock_path)],
        check=False,
        timeout=5,
    )
    assert completed.returncode == 0
    assert lock_path.read_bytes() == b"crashed"
    assert rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    ) == manifest


def test_resume_removes_manifest_temp_before_exact_next_orphan(tmp_path):
    _, _, _, metadata = _geometry_build_input(tmp_path, row_count=377)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    building_dir = tmp_path / "geometry.building"
    manifest_temp = building_dir / "manifest.json.tmp"
    next_shard = building_dir / "shard_000000.pt"
    manifest_temp.write_bytes(b"fsynced but not renamed")
    next_shard.write_bytes(b"published before manifest")

    resumed = rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)

    assert resumed == manifest
    assert not manifest_temp.exists()
    assert not next_shard.exists()


def test_resume_restarts_safe_initial_manifest_temp_state(tmp_path):
    _, _, _, metadata = _geometry_build_input(tmp_path, row_count=2)
    output_dir = tmp_path / "geometry"
    building_dir = tmp_path / "geometry.building"
    building_dir.mkdir()
    (building_dir / "manifest.json.tmp").write_bytes(
        b"initial manifest before rename"
    )

    resumed = rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)

    assert resumed["sample_count"] == 0
    assert (building_dir / "manifest.json").is_file()
    assert not (building_dir / "manifest.json.tmp").exists()


def test_resume_removes_safe_precommit_publication_temp(tmp_path):
    _, _, _, metadata = _geometry_build_input(tmp_path, row_count=377)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    transaction_temp = tmp_path / "geometry.publish.json.tmp"
    transaction_temp.write_bytes(b"prepared transaction before rename")

    resumed = rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)

    assert resumed == manifest
    assert not transaction_temp.exists()


def test_manifest_post_rename_fsync_failure_is_not_reported_as_success(
        tmp_path, monkeypatch):
    _, base_rows, _, metadata = _geometry_build_input(tmp_path, row_count=377)
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    building_dir = tmp_path / "geometry.building"
    original_fsync_directory = rec_geometry_cache._fsync_directory
    failures = []

    def fail_after_new_manifest_rename(directory):
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if (directory == building_dir and manifest_path.is_file()
                and not failures):
            on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
            if on_disk.get("sample_count") == 252:
                failures.append(True)
                raise OSError("simulated post-rename directory fsync failure")
        return original_fsync_directory(directory)

    monkeypatch.setattr(
        rec_geometry_cache, "_fsync_directory", fail_after_new_manifest_rename
    )
    with pytest.raises(ValueError, match="durability|uncertain"):
        rec_geometry_cache.append_geometry_shard(
            output_dir,
            manifest,
            _geometry_rows_for_indices(base_rows, manifest, 0, 252),
            {"boxes": 0.0, "ious": 0.0},
        )
    assert failures == [True]
    committed = json.loads((building_dir / "manifest.json").read_text(
        encoding="utf-8"
    ))
    assert committed["sample_count"] == 252
    assert (building_dir / "shard_000000.pt").is_file()

    monkeypatch.setattr(
        rec_geometry_cache, "_fsync_directory", original_fsync_directory
    )
    recovered = rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)
    assert recovered["sample_count"] == 252


def _complete_replacement_build_without_publication(
        output_dir, metadata, base_rows, monkeypatch):
    building = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata, overwrite=True
    )
    with monkeypatch.context() as isolated:
        isolated.setattr(
            rec_geometry_cache,
            "_publish_complete_geometry_bundle",
            lambda _paths, complete_manifest: complete_manifest,
        )
        completed = rec_geometry_cache.finalize_geometry_cache(
            output_dir,
            building,
            _geometry_rows_for_indices(base_rows, building, 0, len(base_rows)),
            {"boxes": 0.0, "ious": 0.0},
        )
    return completed


@pytest.mark.parametrize("state", ["pre_move", "old_moved", "new_installed"])
def test_recover_geometry_publication_preserves_or_completes_replacement(
        tmp_path, monkeypatch, state):
    _, base_rows, _, old_metadata = _geometry_build_input(tmp_path, row_count=2)
    output_dir = tmp_path / "geometry"
    initial = rec_geometry_cache.initialize_geometry_cache(
        output_dir, old_metadata
    )
    old_manifest = rec_geometry_cache.finalize_geometry_cache(
        output_dir,
        initial,
        _geometry_rows_for_indices(base_rows, initial, 0, 2),
        {"boxes": 0.0, "ious": 0.0},
    )
    new_metadata = copy.deepcopy(old_metadata)
    new_metadata["annotation_sha256"] = "d" * 64
    new_manifest = _complete_replacement_build_without_publication(
        output_dir, new_metadata, base_rows, monkeypatch
    )
    paths = rec_geometry_cache._geometry_cache_paths(output_dir)
    old_identity = canonical_json_sha256(old_manifest)
    new_identity = canonical_json_sha256(new_manifest)

    if state == "pre_move":
        record = rec_geometry_cache._publication_record(
            old_identity, new_identity, "prepared"
        )
    elif state == "old_moved":
        os.replace(paths["final"], paths["backup"])
        record = rec_geometry_cache._publication_record(
            old_identity, new_identity, "old_moved"
        )
    else:
        os.replace(paths["final"], paths["backup"])
        os.replace(paths["building"], paths["final"])
        record = rec_geometry_cache._publication_record(
            old_identity, new_identity, "new_installed"
        )
    rec_geometry_cache._atomic_write_json(paths["transaction"], record)
    transaction_temp = paths["transaction"].with_name(
        paths["transaction"].name + ".tmp"
    )
    if state == "pre_move":
        transaction_temp.write_bytes(b"stage update before rename")

    recovered = rec_geometry_cache.recover_geometry_publication(output_dir)
    if state == "pre_move":
        assert recovered == old_manifest
        assert paths["final"].is_dir()
        assert paths["building"].is_dir()
        assert not paths["backup"].exists()
    else:
        assert recovered == new_manifest
        assert paths["final"].is_dir()
        assert not paths["building"].exists()
        assert not paths["backup"].exists()
    assert not paths["transaction"].exists()
    assert not transaction_temp.exists()


def test_resume_rejects_tampered_payload_parity_and_changed_base_binding(
        tmp_path):
    base_dir, base_rows, _, metadata = _geometry_build_input(
        tmp_path, row_count=377
    )
    output_dir = tmp_path / "geometry"
    manifest = rec_geometry_cache.initialize_geometry_cache(
        output_dir, metadata
    )
    manifest = rec_geometry_cache.append_geometry_shard(
        output_dir,
        manifest,
        _geometry_rows_for_indices(base_rows, manifest, 0, 252),
        {"boxes": 0.1, "ious": 0.01},
    )
    building_manifest_path = tmp_path / "geometry.building" / "manifest.json"
    tampered = json.loads(building_manifest_path.read_text(encoding="utf-8"))
    tampered["parity_maxima"]["boxes"] = 0.2
    _refresh_geometry_manifest_digests(tampered)
    building_manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="parity"):
        rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)

    building_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    base_manifest_path = base_dir / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_manifest["checkpoint_sha256"] = "f" * 64
    base_manifest_path.write_text(json.dumps(base_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="binding|cache contents|checkpoint"):
        rec_geometry_cache.initialize_geometry_cache(output_dir, metadata)
