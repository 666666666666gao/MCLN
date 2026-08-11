import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.cache_scanrefer_rec_mask_geometry as extractor
from models.rec_mask_geometry import (
    DEFAULT_REC_MASK_GEOMETRY_VARIANTS,
    MASK_GEOMETRY_SCHEMA_VERSION,
    REC_MASK_GEOMETRY_FEATURE_NAMES,
)


def _backbone_config():
    return {
        "model": "MCLN",
        "num_target": 256,
        "num_decoder_layers": 6,
        "self_position_embedding": "loc_learned",
        "self_attend": True,
        "use_soft_token_loss": True,
        "use_contrastive_align": True,
        "detect_intermediate": True,
        "use_source_choice_selector": True,
        "source_choice_selector_sources": (
            "default,default_rank_blend_contrastive010"
        ),
        "source_choice_selector_hidden_dim": 288,
    }


def _model_inputs():
    return {
        "use_color": True,
        "use_height": False,
        "use_multiview": False,
        "butd": True,
        "butd_gt": False,
        "butd_cls": False,
    }


def _single_stage_model_inputs():
    result = _model_inputs()
    result["butd"] = False
    return result


def _config():
    values = {}
    values.update(_model_inputs())
    values.update(_backbone_config())
    return SimpleNamespace(**values)


def _production_source(split="train"):
    size = extractor.EXPECTED_DATASET_SIZES[split]
    shard_count = extractor.EXPECTED_BASE_SHARD_COUNTS[split]
    binding = {
        "path": "/tmp/base/{}".format(split),
        "split": split,
        "sample_count": size,
        "feature_dim": 152,
        "candidate_rule": {"topk_per_source": 8, "max_candidates": 16},
        "checkpoint_sha256": extractor.EXPECTED_CHECKPOINT_SHA256,
        "content_sha256": {
            "train": (
                "411ec7d5d80a7be9596de20b348667d529e6a8f568b8ab0c0e0922b8719f9045"
            ),
            "val": (
                "b2e6cf81ba8441d7e9ec04141e0d4fb73f61f069ea3ec99a2396ae68a6740ef3"
            ),
        }[split],
        "manifest_sha256": {
            "train": (
                "c8858036c3da0b25183f262c763e947a3dac77544ee3073623172716878cfabc"
            ),
            "val": (
                "695b5565de460f580a6d130d1b6465d309b65eef46502340f0fc8ea1235907d2"
            ),
        }[split],
        "model_inputs": _model_inputs(),
        "backbone_config": _backbone_config(),
        "shards": [
            {"name": "shard_{:06d}.pt".format(index)}
            for index in range(shard_count)
        ],
    }
    manifest = {
        "split": split,
        "sample_count": size,
        "dataset_size": size,
        "source_dataset_size": size,
        "checkpoint_sha256": extractor.EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_epoch": 71,
        "candidate_rule": copy.deepcopy(binding["candidate_rule"]),
        "feature_dim": 152,
        "target_iou_policy": "root_only",
        "deterministic": True,
        "data_root": "/tmp/data",
        "model_inputs": _model_inputs(),
        "backbone_config": _backbone_config(),
        "shards": [value["name"] for value in binding["shards"]],
    }
    return binding, manifest


def _base_rows(count=2, num_candidates=3):
    rows = []
    for index in range(count):
        query_indices = torch.arange(
            10 + 10 * index,
            10 + 10 * index + num_candidates,
            dtype=torch.int64,
        )
        valid = torch.ones(num_candidates, dtype=torch.bool)
        valid[-1] = False
        boxes = torch.zeros(num_candidates, 6, dtype=torch.float32)
        boxes[:, :3] = float(index) + torch.arange(
            num_candidates, dtype=torch.float32
        ).unsqueeze(1)
        boxes[:, 3:] = torch.tensor([1.0, 1.5, 2.0])
        boxes[~valid] = 0.0
        candidate_ious = torch.tensor(
            [0.60, 0.20] + [0.0] * max(num_candidates - 2, 0),
            dtype=torch.float32,
        )[:num_candidates]
        rows.append({
            "dataset_index": index,
            "scan_id": "scene{:04d}_00".format(index),
            "target_id": 100 + index,
            "default_top1_query_index": int(query_indices[0]),
            "features": torch.full(
                (num_candidates, 4), float(index), dtype=torch.float32
            ),
            "boxes": boxes,
            "query_indices": query_indices,
            "valid_mask": valid,
            "default_scores": torch.tensor(
                [0.9, 0.5] + [0.0] * max(num_candidates - 2, 0),
                dtype=torch.float32,
            )[:num_candidates],
            "contrastive_scores": torch.tensor(
                [0.8, 0.4] + [0.0] * max(num_candidates - 2, 0),
                dtype=torch.float32,
            )[:num_candidates],
            "candidate_ious": candidate_ious,
        })
    return rows


def _fresh_candidates(base_rows, box_drift=0.001):
    boxes = torch.stack([row["boxes"] for row in base_rows]).clone()
    valid = torch.stack([row["valid_mask"] for row in base_rows])
    boxes[valid] += box_drift
    return {
        "schema_version": "rec-query-v1",
        "feature_names": ["f0", "f1", "f2", "f3"],
        "features": torch.stack([row["features"] for row in base_rows]),
        "boxes": boxes,
        "query_indices": torch.stack([
            row["query_indices"] for row in base_rows
        ]),
        "valid_mask": valid,
        "default_scores": torch.stack([
            row["default_scores"] for row in base_rows
        ]),
        "contrastive_scores": torch.stack([
            row["contrastive_scores"] for row in base_rows
        ]),
        "default_top1_query_index": torch.tensor([
            row["default_top1_query_index"] for row in base_rows
        ], dtype=torch.int64),
        "model_inputs": {"runtime_only": torch.tensor(1)},
    }


def _geometry_batch(candidate_batch, with_targets=True):
    variants = tuple(dict(value) for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS)
    batch_size, num_candidates = candidate_batch["valid_mask"].shape
    num_variants = len(variants)
    boxes = torch.zeros(
        batch_size, num_candidates, num_variants, 6, dtype=torch.float32
    )
    parent = candidate_batch["boxes"]
    valid = candidate_batch["valid_mask"].unsqueeze(-1).expand(
        -1, -1, num_variants
    ).clone()
    for variant_index in range(num_variants):
        boxes[:, :, variant_index] = parent + float(variant_index) * 0.25
    boxes[~valid] = 0.0
    features = torch.arange(
        batch_size * num_candidates * num_variants
        * len(REC_MASK_GEOMETRY_FEATURE_NAMES),
        dtype=torch.float32,
    ).reshape(
        batch_size,
        num_candidates,
        num_variants,
        len(REC_MASK_GEOMETRY_FEATURE_NAMES),
    )
    result = {
        "schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
        "boxes": boxes,
        "valid_mask": valid,
        "geometry_features": features,
        "geometry_feature_names": REC_MASK_GEOMETRY_FEATURE_NAMES,
        "variant_names": tuple(value["name"] for value in variants),
        "variant_configs": variants,
        "variant_config": variants,
        "min_points": 5,
        "max_point_fraction": 0.5,
        "mask_diagnostics": tuple({} for _ in range(batch_size)),
    }
    if with_targets:
        ious = torch.full(valid.shape, 0.35, dtype=torch.float32)
        ious[~valid] = 0.0
        result["geometry_ious"] = ious
        result["threshold_labels"] = torch.stack(
            [ious > 0.25, ious > 0.50], dim=-1
        )
    return result


def _batch_data(base_rows):
    batch_size = len(base_rows)
    return {
        "scan_ids": [row["scan_id"] for row in base_rows],
        "target_id": torch.tensor([
            row["target_id"] for row in base_rows
        ]),
        "center_label": torch.zeros(batch_size, 1, 3),
        "size_gts": torch.ones(batch_size, 1, 3),
        "box_label_mask": torch.ones(batch_size, 1),
        "gt_masks": torch.ones(batch_size, 1, 4),
    }


def _zero_parity():
    return {name: 0.0 for name in extractor.PARITY_FIELDS}


def test_cli_enforces_fixed_production_values_and_stop_contract(tmp_path):
    required = [
        "--split", "val",
        "--data-root", str(tmp_path / "data"),
        "--checkpoint", str(tmp_path / "checkpoint.pth"),
        "--base-cache", str(tmp_path / "base"),
        "--output-dir", str(tmp_path / "geometry"),
        "--audit-provenance", str(tmp_path / "selection.json"),
    ]
    args = extractor.parse_args(required + [
        "--overwrite", "--restart-building", "--stop-after-shards", "1"
    ])

    assert args.batch_size == 12
    assert args.num_workers == 2
    assert args.shard_size == 252
    assert args.device == "cuda:0"
    assert args.restart_building
    assert args.stop_after_shards == 1

    for option, value in (
            ("--batch-size", "6"),
            ("--num-workers", "0"),
            ("--shard-size", "12"),
            ("--device", "cpu"),
            ("--stop-after-shards", "0")):
        with pytest.raises(SystemExit):
            extractor.parse_args(required + [option, value])
    with pytest.raises(SystemExit):
        extractor.parse_args(required + ["--restart-building"])


def test_portable_cli_requires_and_binds_audit_train_cache(tmp_path):
    required = [
        "--split", "val",
        "--data-root", str(tmp_path / "data"),
        "--checkpoint", str(tmp_path / "checkpoint.pth"),
        "--base-cache", str(tmp_path / "val"),
        "--output-dir", str(tmp_path / "geometry-val"),
        "--audit-provenance", str(tmp_path / "selection.json"),
        "--portable-provenance",
    ]
    with pytest.raises(SystemExit):
        extractor.parse_args(required)

    train_cache = tmp_path / "train"
    args = extractor.parse_args(
        required + [
            "--audit-train-cache", str(train_cache),
            "--batch-size", "36", "--num-workers", "4",
            "--shard-size", "252",
        ]
    )
    assert args.portable_provenance is True
    assert args.audit_train_cache == str(train_cache)
    assert args.batch_size == 36
    assert args.num_workers == 4


def test_portable_metadata_binds_dynamic_checkpoint_and_audit(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    train_cache = tmp_path / "v19" / "train"
    train_cache.mkdir(parents=True)
    checkpoint = tmp_path / "v19.pth"
    checkpoint.write_bytes(b"v19 checkpoint")
    checkpoint_sha256 = "2" * 64
    binding, manifest = _production_source("train")
    backbone = extractor._backbone_config_from_config(_config())
    binding.update({
        "path": str(train_cache.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "content_sha256": "3" * 64,
        "manifest_sha256": "4" * 64,
        "backbone_config": copy.deepcopy(backbone),
    })
    manifest.update({
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": 2,
        "data_root": str(data_root.resolve()),
        "backbone_config": copy.deepcopy(backbone),
    })
    (train_cache / "manifest.json").write_text(json.dumps(manifest))
    manifest_sha256 = extractor.canonical_json_sha256(manifest)
    variants = [
        dict(value) for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS
    ]
    audit = tmp_path / "audit" / "selection.json"
    audit.parent.mkdir()
    audit_payload = {
        "panel_schema_version": "rec-mask-geometry-audit-panel-v1",
        "population_estimate": False,
        "sample_count": 256,
        "cache_extraction_batch_size": 12,
        "checkpoint_sha256": checkpoint_sha256,
        "train_cache": str(train_cache.resolve()),
        "provenance": {
            "panel_schema_version": "rec-mask-geometry-audit-panel-v1",
            "population_estimate": False,
            "split": "train",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": 2,
            "data_root": str(data_root.resolve()),
            "train_cache": str(train_cache.resolve()),
            "train_cache_manifest_sha256": manifest_sha256,
            "cache_extraction_batch_size": 12,
            "candidate_rule": {
                "topk_per_source": 8, "max_candidates": 16,
            },
            "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
            "geometry_feature_names": list(
                REC_MASK_GEOMETRY_FEATURE_NAMES
            ),
            "variant_names": [value["name"] for value in variants],
            "variant_configs": variants,
            "min_points": 5,
            "max_point_fraction": 0.5,
        },
    }
    audit.write_text(json.dumps(audit_payload))
    args = SimpleNamespace(
        split="train",
        data_root=str(data_root),
        audit_provenance=str(audit),
        audit_train_cache=str(train_cache),
        portable_provenance=True,
        batch_size=12,
        num_workers=2,
        shard_size=252,
    )

    metadata = extractor.build_geometry_immutable_metadata(
        args,
        binding,
        manifest,
        checkpoint,
        checkpoint_sha256,
        2,
        _config(),
        annotation_sha256=extractor.EXPECTED_ANNOTATION_SHA256["train"],
    )

    assert metadata["checkpoint_sha256"] == checkpoint_sha256
    assert metadata["checkpoint_epoch"] == 2
    assert metadata["base_cache_binding"] == binding
    assert metadata["audit_provenance"]["panel"] == str(audit.resolve())


def test_stable_file_snapshot_rejects_path_replacement(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    replacement = tmp_path / "replacement.bin"
    source.write_bytes(b"authoritative bytes")
    replacement.write_bytes(b"replacement bytes")
    original_open = Path.open
    replaced = []

    def open_then_replace(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if (not replaced and mode == "rb"
                and Path(path).resolve() == source.resolve()):
            os.replace(str(replacement), str(source))
            replaced.append(True)
        return handle

    monkeypatch.setattr(Path, "open", open_then_replace)

    with pytest.raises(ValueError, match="changed during stable snapshot"):
        extractor._read_stable_file_snapshot(source, "test source")


def test_checkpoint_is_hashed_and_loaded_from_one_stable_byte_snapshot(
        tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"epoch": 71, "value": torch.tensor([3.0])}, checkpoint_path)
    expected_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    checkpoint, fingerprint = extractor._load_checkpoint_snapshot(
        checkpoint_path
    )

    assert checkpoint["epoch"] == 71
    assert torch.equal(checkpoint["value"], torch.tensor([3.0]))
    assert fingerprint == expected_sha


def test_dataset_build_rejects_annotation_identity_change(monkeypatch):
    expected_sha = extractor.EXPECTED_ANNOTATION_SHA256["train"]
    snapshots = iter([
        {"sha256": expected_sha, "identity": (1, 2, 3, 4, 5)},
        {"sha256": expected_sha, "identity": (1, 9, 3, 4, 5)},
    ])
    monkeypatch.setattr(
        extractor,
        "_validated_annotation_snapshot",
        lambda data_root, split: next(snapshots),
    )

    with pytest.raises(ValueError, match="changed during dataset construction"):
        extractor._build_validated_dataset(
            config=object(),
            split="train",
            data_root="/tmp/data",
            dataset_builder=lambda config, split: object(),
        )


def test_metadata_binds_actual_sources_without_replay_or_target_payload(
        tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    annotation = data_root / "scanrefer" / "ScanRefer_filtered_train.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_bytes(b"annotation bytes")
    audit = tmp_path / "audit" / "selection.json"
    audit.parent.mkdir()
    binding, base_manifest = _production_source("train")
    binding["path"] = str((tmp_path / "base").resolve())
    base_manifest["data_root"] = str(data_root.resolve())
    audit_payload = {
        "panel_schema_version": "rec-mask-geometry-audit-panel-v1",
        "population_estimate": False,
        "sample_count": 256,
        "cache_extraction_batch_size": 12,
        "checkpoint_sha256": extractor.EXPECTED_CHECKPOINT_SHA256,
        "train_cache": binding["path"],
        "provenance": {
            "panel_schema_version": "rec-mask-geometry-audit-panel-v1",
            "population_estimate": False,
            "split": "train",
            "checkpoint_sha256": extractor.EXPECTED_CHECKPOINT_SHA256,
            "checkpoint_epoch": 71,
            "data_root": str(data_root.resolve()),
            "train_cache": binding["path"],
            "train_cache_manifest_sha256": binding["manifest_sha256"],
            "cache_extraction_batch_size": 12,
            "candidate_rule": {"topk_per_source": 8, "max_candidates": 16},
            "geometry_schema_version": MASK_GEOMETRY_SCHEMA_VERSION,
            "geometry_feature_names": list(REC_MASK_GEOMETRY_FEATURE_NAMES),
            "variant_names": [
                value["name"] for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS
            ],
            "variant_configs": [
                dict(value) for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS
            ],
            "min_points": 5,
            "max_point_fraction": 0.5,
        },
    }
    audit_bytes = json.dumps(
        audit_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    audit.write_bytes(audit_bytes)
    monkeypatch.setitem(
        extractor.EXPECTED_ANNOTATION_SHA256,
        "train",
        hashlib.sha256(b"annotation bytes").hexdigest(),
    )
    monkeypatch.setattr(
        extractor,
        "EXPECTED_AUDIT_SELECTION_SHA256",
        hashlib.sha256(audit_bytes).hexdigest(),
    )
    checkpoint = tmp_path / "epoch71.pth"
    checkpoint.write_bytes(b"checkpoint bytes")
    args = SimpleNamespace(
        split="train",
        data_root=str(data_root),
        audit_provenance=str(audit),
        batch_size=12,
        num_workers=2,
        shard_size=252,
    )

    metadata = extractor.build_geometry_immutable_metadata(
        args=args,
        base_binding=binding,
        base_manifest=base_manifest,
        checkpoint_path=checkpoint,
        checkpoint_sha256=extractor.EXPECTED_CHECKPOINT_SHA256,
        checkpoint_epoch=71,
        config=_config(),
    )

    assert set(metadata) == set(extractor.GEOMETRY_IMMUTABLE_METADATA_FIELDS)
    assert metadata["base_cache_binding"] == binding
    assert metadata["dataset_size"] == 36665
    assert metadata["source_dataset_size"] == 36665
    assert metadata["checkpoint_path"] == str(checkpoint.resolve())
    assert metadata["annotation_sha256"] == hashlib.sha256(
        b"annotation bytes"
    ).hexdigest()
    assert metadata["audit_provenance"] == {
        "panel": str(audit.resolve()),
        "sha256": hashlib.sha256(audit_bytes).hexdigest(),
    }
    assert metadata["geometry_feature_names"] == list(
        REC_MASK_GEOMETRY_FEATURE_NAMES
    )
    assert metadata["variant_configs"] == [
        dict(value) for value in DEFAULT_REC_MASK_GEOMETRY_VARIANTS
    ]
    assert metadata["filter_non_gt_boxes"] is False
    assert "cache_extraction_batch_size" not in metadata
    assert "cache_replay_boundaries" not in metadata
    assert "geometry_ious" not in metadata


def test_source_validation_rejects_nonproduction_butd_and_nonfull_sizes():
    binding, manifest = _production_source("val")

    extractor.validate_source_provenance(
        split="val",
        base_binding=binding,
        base_manifest=manifest,
        checkpoint_sha256=extractor.EXPECTED_CHECKPOINT_SHA256,
        checkpoint_epoch=71,
        config=_config(),
        source_dataset_size=9508,
        data_root="/tmp/data",
    )

    bad_config = _config()
    bad_config.butd = False
    with pytest.raises(ValueError, match="butd"):
        extractor.validate_source_provenance(
            "val", binding, manifest,
            extractor.EXPECTED_CHECKPOINT_SHA256, 71,
            bad_config, 9508, "/tmp/data",
        )
    with pytest.raises(ValueError, match="full source dataset"):
        extractor.validate_source_provenance(
            "val", binding, manifest,
            extractor.EXPECTED_CHECKPOINT_SHA256, 71,
            _config(), 9507, "/tmp/data",
        )
    limited = copy.deepcopy(manifest)
    limited["dataset_size"] -= 1
    with pytest.raises(ValueError, match="complete full-dataset"):
        extractor.validate_source_provenance(
            "val", binding, limited,
            extractor.EXPECTED_CHECKPOINT_SHA256, 71,
            _config(), 9508, "/tmp/data",
        )


def test_source_validation_pins_base_content_and_resolved_data_root():
    binding, manifest = _production_source("train")

    tampered_binding = copy.deepcopy(binding)
    tampered_binding["content_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="content digest"):
        extractor.validate_source_provenance(
            "train", tampered_binding, manifest,
            extractor.EXPECTED_CHECKPOINT_SHA256, 71,
            _config(), 36665, "/tmp/data",
        )

    with pytest.raises(ValueError, match="data root"):
        extractor.validate_source_provenance(
            "train", binding, manifest,
            extractor.EXPECTED_CHECKPOINT_SHA256, 71,
            _config(), 36665, "/tmp/different-data",
        )


def test_metadata_rejects_unapproved_audit_or_annotation_sha(tmp_path):
    data_root = tmp_path / "data"
    annotation = data_root / "scanrefer" / "ScanRefer_filtered_train.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_bytes(b"not authoritative")
    audit = tmp_path / "selection.json"
    audit.write_text("{}")
    checkpoint = tmp_path / "epoch71.pth"
    checkpoint.write_bytes(b"checkpoint")
    binding, manifest = _production_source("train")
    binding["path"] = str((tmp_path / "base").resolve())
    manifest["data_root"] = str(data_root.resolve())
    args = SimpleNamespace(
        split="train", data_root=str(data_root),
        audit_provenance=str(audit), batch_size=12,
        num_workers=2, shard_size=252,
    )

    with pytest.raises(ValueError, match="annotation JSON SHA-256"):
        extractor.build_geometry_immutable_metadata(
            args, binding, manifest, checkpoint,
            extractor.EXPECTED_CHECKPOINT_SHA256, 71, _config()
        )

    annotation.write_bytes(b"annotation")
    expected_annotation = hashlib.sha256(b"annotation").hexdigest()
    original_annotation = extractor.EXPECTED_ANNOTATION_SHA256["train"]
    extractor.EXPECTED_ANNOTATION_SHA256["train"] = expected_annotation
    try:
        with pytest.raises(ValueError, match="audit selection SHA-256"):
            extractor.build_geometry_immutable_metadata(
                args, binding, manifest, checkpoint,
                extractor.EXPECTED_CHECKPOINT_SHA256, 71, _config()
            )
    finally:
        extractor.EXPECTED_ANNOTATION_SHA256["train"] = original_annotation


def test_row_construction_canonicalizes_g0_and_excludes_runtime_payload():
    base_rows = _base_rows()
    base_by_index = {row["dataset_index"]: row for row in base_rows}
    fresh = _fresh_candidates(base_rows)
    fresh["candidate_ious"] = torch.stack([
        row["candidate_ious"] for row in base_rows
    ])
    fresh["threshold_labels"] = torch.zeros(2, 3, 2, dtype=torch.bool)
    geometry = _geometry_batch(fresh)
    original_nonregressed = geometry["boxes"][:, :, 1:].clone()
    rejection_codes = torch.arange(
        2 * 3 * 7, dtype=torch.int16
    ).reshape(2, 3, 7)
    parity_calls = []

    def parity(candidate_batch, cached_rows, dataset_indices,
               scan_ids, target_ids, **kwargs):
        parity_calls.append(tuple(dataset_indices))
        assert candidate_batch is fresh
        assert cached_rows is base_by_index
        return {
            "boxes": 0.001,
            "candidate_ious": 0.0,
            "features": 0.25,
            "default_scores": 0.0,
            "contrastive_scores": 0.0,
        }

    rows, maxima = extractor.build_geometry_rows(
        dataset_indices=[0, 1],
        batch_data=_batch_data(base_rows),
        targeted_candidates=fresh,
        targeted_geometry=geometry,
        base_rows_by_index=base_by_index,
        rejection_codes=rejection_codes,
        parity_check=parity,
    )

    assert parity_calls == [(0, 1)]
    assert maxima["features"] == 0.25
    assert len(rows) == 2
    expected_keys = {
        "dataset_index", "scan_id", "target_id",
        "default_top1_query_index", "query_indices", "candidate_valid",
        "geometry_boxes", "geometry_valid", "evaluator_valid",
        "geometry_features", "geometry_ious", "source_rejection_codes",
    }
    excluded = {
        "features", "scores", "boxes", "valid_mask", "model_inputs",
        "center_label", "size_gts", "box_label_mask", "gt_masks",
        "threshold_labels", "candidate_ious",
    }
    for index, row in enumerate(rows):
        assert set(row) == expected_keys
        assert not (set(row) & excluded)
        assert torch.equal(row["query_indices"], base_rows[index][
            "query_indices"
        ])
        assert torch.equal(row["candidate_valid"], base_rows[index][
            "valid_mask"
        ])
        assert torch.equal(row["geometry_boxes"][:, 0], base_rows[index][
            "boxes"
        ])
        assert torch.equal(row["geometry_ious"][:, 0], base_rows[index][
            "candidate_ious"
        ])
        assert torch.equal(
            row["geometry_boxes"][:, 1:], original_nonregressed[index]
        )
        assert torch.equal(row["evaluator_valid"], row["geometry_valid"])
        assert torch.equal(
            row["source_rejection_codes"], rejection_codes[index]
        )
        assert bool((row["geometry_ious"][~row["geometry_valid"]] == 0).all())
        for value in row.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
                assert value.is_contiguous()
    assert rows[0]["query_indices"].dtype == torch.int64
    assert rows[0]["candidate_valid"].dtype == torch.bool
    assert rows[0]["geometry_boxes"].dtype == torch.float32
    assert rows[0]["geometry_features"].dtype == torch.float32
    assert rows[0]["geometry_ious"].dtype == torch.float32
    assert rows[0]["source_rejection_codes"].dtype == torch.int16


def test_parity_failure_happens_before_any_geometry_row_is_accessed():
    base_rows = _base_rows(count=1)
    fresh = _fresh_candidates(base_rows)
    fresh["candidate_ious"] = torch.stack([
        base_rows[0]["candidate_ious"]
    ])

    def fail_parity(*args, **kwargs):
        raise RuntimeError("parity rejected batch")

    with pytest.raises(RuntimeError, match="parity rejected"):
        extractor.build_geometry_rows(
            [0], _batch_data(base_rows), fresh,
            targeted_geometry={},
            base_rows_by_index={0: base_rows[0]},
            rejection_codes=None,
            parity_check=fail_parity,
        )


def test_gt_boundary_and_base_parent_are_enforced_before_target_attachment():
    base_rows = _base_rows()
    base_by_index = {row["dataset_index"]: row for row in base_rows}
    raw_candidates = _fresh_candidates(base_rows)
    batch_data = _batch_data(base_rows)
    inputs = {"point_clouds": torch.zeros(2, 4, 3), "train": False}
    calls = []
    geometry_seen = {}

    def model(model_inputs):
        calls.append("model")
        assert not (set(model_inputs) & extractor.FORBIDDEN_DEPLOYABLE_KEYS)
        return {"deployable": torch.tensor(1)}

    def candidate_builder(end_points, model_inputs, **kwargs):
        calls.append("candidate")
        assert not (set(end_points) & extractor.FORBIDDEN_DEPLOYABLE_KEYS)
        assert not (set(model_inputs) & extractor.FORBIDDEN_DEPLOYABLE_KEYS)
        return raw_candidates

    def geometry_builder(end_points, model_inputs, candidates, **kwargs):
        calls.append("geometry")
        assert "candidate_ious" not in candidates
        expected_boxes = torch.stack([row["boxes"] for row in base_rows])
        assert torch.equal(candidates["boxes"], expected_boxes)
        assert torch.equal(candidates["query_indices"], torch.stack([
            row["query_indices"] for row in base_rows
        ]))
        geometry_seen["value"] = _geometry_batch(
            candidates, with_targets=False
        )
        return geometry_seen["value"]

    def rejection_projector(geometry):
        calls.append("project")
        return torch.full(geometry["valid_mask"].shape, 9, dtype=torch.int16)

    def attach_candidates(candidates, targets, root_only):
        calls.append("attach_candidates")
        assert root_only is True
        result = dict(candidates)
        result["candidate_ious"] = torch.stack([
            row["candidate_ious"] for row in base_rows
        ])
        result["threshold_labels"] = torch.zeros(2, 3, 2, dtype=torch.bool)
        return result

    def attach_geometry(geometry, targets, root_only):
        calls.append("attach_geometry")
        assert root_only is True
        result = dict(geometry)
        ious = torch.full(geometry["valid_mask"].shape, 0.4)
        ious[~geometry["valid_mask"]] = 0.0
        result["geometry_ious"] = ious
        result["threshold_labels"] = torch.zeros(
            *ious.shape, 2, dtype=torch.bool
        )
        return result

    def parity(candidate_batch, *args, **kwargs):
        calls.append("parity")
        assert torch.equal(candidate_batch["boxes"], raw_candidates["boxes"])
        return _zero_parity()

    rows, maxima = extractor.extract_geometry_batch(
        model=model,
        inputs=inputs,
        batch_data=batch_data,
        dataset_indices=[0, 1],
        base_rows_by_index=base_by_index,
        candidate_builder=candidate_builder,
        geometry_builder=geometry_builder,
        candidate_target_attacher=attach_candidates,
        geometry_target_attacher=attach_geometry,
        rejection_projector=rejection_projector,
        parity_check=parity,
    )

    assert calls == [
        "model", "candidate", "geometry", "project",
        "attach_candidates", "attach_geometry", "parity",
    ]
    assert maxima == _zero_parity()
    assert torch.equal(
        rows[0]["geometry_boxes"][:, 5],
        geometry_seen["value"]["boxes"][0, :, 5],
    )
    assert torch.equal(
        rows[0]["source_rejection_codes"],
        torch.full((3, 7), 9, dtype=torch.int16),
    )


def test_portable_parity_uses_cached_candidate_identity():
    base_rows = _base_rows(count=1)
    base_by_index = {0: base_rows[0]}
    fresh = _fresh_candidates(base_rows)
    fresh["query_indices"] = fresh["query_indices"].flip(1)
    fresh["default_top1_query_index"] = fresh["query_indices"][:, 0]

    def geometry_builder(end_points, inputs, candidates, **kwargs):
        assert torch.equal(
            candidates["query_indices"],
            torch.stack([base_rows[0]["query_indices"]]),
        )
        return _geometry_batch(candidates, with_targets=False)

    def attach_candidates(candidates, targets, root_only):
        assert root_only is True
        assert torch.equal(
            candidates["query_indices"],
            torch.stack([base_rows[0]["query_indices"]]),
        )
        assert candidates["default_top1_query_index"].item() == (
            base_rows[0]["default_top1_query_index"]
        )
        result = dict(candidates)
        result["candidate_ious"] = torch.stack([
            base_rows[0]["candidate_ious"]
        ])
        result["threshold_labels"] = torch.zeros(
            1, 3, 2, dtype=torch.bool
        )
        return result

    def attach_geometry(geometry, targets, root_only):
        assert root_only is True
        result = dict(geometry)
        result["geometry_ious"] = torch.zeros_like(
            geometry["valid_mask"], dtype=torch.float32
        )
        return result

    def parity(candidate_batch, *args, **kwargs):
        assert torch.equal(
            candidate_batch["query_indices"],
            torch.stack([base_rows[0]["query_indices"]]),
        )
        assert candidate_batch["default_top1_query_index"].item() == (
            base_rows[0]["default_top1_query_index"]
        )
        return _zero_parity()

    rows, maxima = extractor.extract_geometry_batch(
        model=lambda inputs: {},
        inputs={"point_clouds": torch.zeros(1, 4, 3)},
        batch_data=_batch_data(base_rows),
        dataset_indices=[0],
        base_rows_by_index=base_by_index,
        candidate_builder=lambda *args, **kwargs: fresh,
        geometry_builder=geometry_builder,
        candidate_target_attacher=attach_candidates,
        geometry_target_attacher=attach_geometry,
        rejection_projector=lambda geometry: torch.zeros_like(
            geometry["valid_mask"], dtype=torch.int16
        ),
        parity_check=parity,
        canonicalize_candidate_parity=True,
    )

    assert len(rows) == 1
    assert maxima == _zero_parity()


def test_train_tester_get_inputs_is_gt_free():
    from train_dist_mod import TrainTester

    batch_data = {
        "point_clouds": torch.zeros(1, 4, 3),
        "utterances": ["the chair"],
        "all_detected_boxes": torch.zeros(1, 2, 6),
        "all_detected_bbox_label_mask": torch.ones(1, 2, dtype=torch.bool),
        "all_detected_class_ids": torch.zeros(1, 2, dtype=torch.long),
        "superpoint": torch.zeros(1, 4, dtype=torch.long),
        "center_label": torch.zeros(1, 1, 3),
        "size_gts": torch.ones(1, 1, 3),
        "box_label_mask": torch.ones(1, 1),
        "gt_masks": torch.ones(1, 1, 4),
    }

    inputs = TrainTester._get_inputs(batch_data)

    assert not (set(inputs) & extractor.FORBIDDEN_DEPLOYABLE_KEYS)
    assert set(inputs) == {
        "point_clouds", "text", "det_boxes", "det_bbox_label_mask",
        "det_class_ids", "superpoint",
    }


@pytest.mark.parametrize("forbidden_key", sorted(
    extractor.FORBIDDEN_DEPLOYABLE_KEYS
))
def test_forbidden_target_key_cannot_reach_candidate_builder(forbidden_key):
    base_row = _base_rows(count=1)[0]
    candidate_called = []

    def model(inputs):
        return {forbidden_key: torch.tensor(1)}

    def candidate_builder(*args, **kwargs):
        candidate_called.append(True)
        raise AssertionError("candidate builder received target payload")

    with pytest.raises(ValueError, match="deployable"):
        extractor.extract_geometry_batch(
            model=model,
            inputs={"point_clouds": torch.zeros(1, 4, 3)},
            batch_data=_batch_data([base_row]),
            dataset_indices=[0],
            base_rows_by_index={0: base_row},
            candidate_builder=candidate_builder,
        )
    assert candidate_called == []


def test_candidate_builder_cannot_inject_target_payload_into_geometry_builder():
    base_row = _base_rows(count=1)[0]
    candidates = _fresh_candidates([base_row])
    geometry_called = []

    def candidate_builder(end_points, inputs, **kwargs):
        end_points["candidate_ious"] = torch.ones(1, 3)
        return candidates

    def geometry_builder(*args, **kwargs):
        geometry_called.append(True)
        raise AssertionError("geometry builder received target payload")

    with pytest.raises(ValueError, match="deployable"):
        extractor.extract_geometry_batch(
            model=lambda inputs: {},
            inputs={"point_clouds": torch.zeros(1, 4, 3)},
            batch_data=_batch_data([base_row]),
            dataset_indices=[0],
            base_rows_by_index={0: base_row},
            candidate_builder=candidate_builder,
            geometry_builder=geometry_builder,
        )
    assert geometry_called == []


def test_schema_drift_fails_before_targets_or_rows_are_built():
    base_row = _base_rows(count=1)[0]
    candidates = _fresh_candidates([base_row])
    attach_calls = []

    def geometry_builder(end_points, inputs, canonical, **kwargs):
        geometry = _geometry_batch(canonical, with_targets=False)
        geometry["variant_names"] = tuple(geometry["variant_names"][:-1])
        return geometry

    def attach(*args, **kwargs):
        attach_calls.append(True)
        raise AssertionError("target attachment must not run")

    with pytest.raises(ValueError, match="variant names"):
        extractor.extract_geometry_batch(
            model=lambda inputs: {},
            inputs={"point_clouds": torch.zeros(1, 4, 3)},
            batch_data=_batch_data([base_row]),
            dataset_indices=[0],
            base_rows_by_index={0: base_row},
            candidate_builder=lambda *args, **kwargs: candidates,
            geometry_builder=geometry_builder,
            candidate_target_attacher=attach,
            geometry_target_attacher=attach,
        )
    assert attach_calls == []


def test_parity_maxima_merge_is_exact_and_rejects_schema_drift():
    first = {
        "boxes": 0.001,
        "candidate_ious": 0.004,
        "features": 0.2,
        "default_scores": 0.3,
        "contrastive_scores": 0.1,
    }
    second = {
        "boxes": 0.0005,
        "candidate_ious": 0.006,
        "features": 0.1,
        "default_scores": 0.4,
        "contrastive_scores": 0.2,
    }

    merged = extractor.merge_parity_maxima(first, second)

    assert merged == {
        "boxes": 0.001,
        "candidate_ious": 0.006,
        "features": 0.2,
        "default_scores": 0.4,
        "contrastive_scores": 0.2,
    }
    with pytest.raises(ValueError, match="parity fields"):
        extractor.merge_parity_maxima(first, {"boxes": 0.0})


def _loader_batch(size, start=0):
    return {
        "scan_ids": ["scene{:04d}_00".format(start + offset)
                     for offset in range(size)],
        "target_id": torch.arange(start, start + size),
    }


def _parity(value):
    return {name: float(value) for name in extractor.PARITY_FIELDS}


def test_resume_starts_at_aligned_global_index_and_stop_commits_one_shard():
    manifest = {
        "complete": False,
        "sample_count": 252,
        "dataset_size": 757,
        "shard_size": 252,
        "extraction_batch_size": 12,
        "parity_maxima": _parity(0.05),
        "shards": [{"name": "shard_000000.pt"}],
    }
    loader = [_loader_batch(12, 252 + 12 * index) for index in range(21)]
    seen_indices = []
    appended = []

    def extract_batch(**kwargs):
        indices = tuple(kwargs["dataset_indices"])
        seen_indices.append(indices)
        return ([{"dataset_index": index} for index in indices],
                _parity(indices[0] / 1000.0))

    def append(output_dir, current, rows, shard_parity):
        appended.append((list(rows), dict(shard_parity)))
        updated = copy.deepcopy(current)
        updated["sample_count"] += len(rows)
        updated["shards"].append({"name": "shard_000001.pt"})
        updated["parity_maxima"] = extractor.merge_parity_maxima(
            current["parity_maxima"], shard_parity
        )
        return updated

    result = extractor.process_geometry_loader(
        loader=loader,
        model=object(),
        base_rows_by_index={},
        output_dir="/tmp/geometry",
        manifest=manifest,
        get_inputs=lambda batch: {},
        device=torch.device("cpu"),
        stop_after_shards=1,
        extract_batch=extract_batch,
        append_shard=append,
        finalize_cache=lambda *args, **kwargs: pytest.fail(
            "stop-after-shards must not finalize"
        ),
        move_batch=lambda batch, device: batch,
    )

    assert seen_indices[0] == tuple(range(252, 264))
    assert seen_indices[-1] == tuple(range(492, 504))
    assert len(appended) == 1
    assert len(appended[0][0]) == 252
    assert appended[0][1] == _parity(0.492)
    assert result["sample_count"] == 504
    assert result["complete"] is False


def test_loader_allows_short_batch_only_at_true_dataset_end_and_finalizes():
    manifest = {
        "complete": False,
        "sample_count": 0,
        "dataset_size": 13,
        "shard_size": 252,
        "extraction_batch_size": 12,
        "parity_maxima": {},
        "shards": [],
    }
    finalized = []

    def extract_batch(**kwargs):
        indices = list(kwargs["dataset_indices"])
        return ([{"dataset_index": index} for index in indices], _parity(0.1))

    def finalize(output_dir, current, rows, maxima):
        finalized.append((rows, maxima))
        result = copy.deepcopy(current)
        result["sample_count"] += len(rows)
        result["complete"] = True
        result["parity_maxima"] = dict(maxima)
        return result

    result = extractor.process_geometry_loader(
        loader=[_loader_batch(12), _loader_batch(1, 12)],
        model=object(),
        base_rows_by_index={},
        output_dir="/tmp/geometry",
        manifest=manifest,
        get_inputs=lambda batch: {},
        device=torch.device("cpu"),
        extract_batch=extract_batch,
        append_shard=lambda *args, **kwargs: pytest.fail(
            "a terminal tail must use finalization"
        ),
        finalize_cache=finalize,
        move_batch=lambda batch, device: batch,
    )

    assert result["complete"] is True
    assert [row["dataset_index"] for row in finalized[0][0]] == list(range(13))
    assert finalized[0][1] == _parity(0.1)

    with pytest.raises(RuntimeError, match="short before dataset end"):
        extractor.process_geometry_loader(
            loader=[_loader_batch(11), _loader_batch(2, 11)],
            model=object(),
            base_rows_by_index={},
            output_dir="/tmp/geometry",
            manifest=manifest,
            get_inputs=lambda batch: {},
            device=torch.device("cpu"),
            extract_batch=extract_batch,
            append_shard=lambda *args, **kwargs: None,
            finalize_cache=finalize,
            move_batch=lambda batch, device: batch,
        )


def test_exact_252_row_terminal_shard_is_finalized_without_append():
    manifest = {
        "complete": False,
        "sample_count": 0,
        "dataset_size": 252,
        "shard_size": 252,
        "extraction_batch_size": 12,
        "parity_maxima": {},
        "shards": [],
    }
    finalized = []

    def extract_batch(**kwargs):
        indices = list(kwargs["dataset_indices"])
        return ([{"dataset_index": index} for index in indices], _parity(0.2))

    def finalize(output_dir, current, rows, maxima):
        finalized.append((list(rows), dict(maxima)))
        result = copy.deepcopy(current)
        result["complete"] = True
        result["sample_count"] = len(rows)
        result["parity_maxima"] = dict(maxima)
        return result

    result = extractor.process_geometry_loader(
        loader=[_loader_batch(12, 12 * index) for index in range(21)],
        model=object(),
        base_rows_by_index={},
        output_dir="/tmp/geometry",
        manifest=manifest,
        get_inputs=lambda batch: {},
        device=torch.device("cpu"),
        extract_batch=extract_batch,
        append_shard=lambda *args, **kwargs: pytest.fail(
            "terminal exact-252 rows must not use append"
        ),
        finalize_cache=finalize,
        move_batch=lambda batch, device: batch,
    )

    assert result["complete"] is True
    assert len(finalized) == 1
    assert [row["dataset_index"] for row in finalized[0][0]] == list(
        range(252)
    )
    assert finalized[0][1] == _parity(0.2)


def test_resume_rejects_unaligned_state_and_complete_cache_is_noop():
    manifest = {
        "complete": False,
        "sample_count": 12,
        "dataset_size": 300,
        "shard_size": 252,
        "extraction_batch_size": 12,
        "parity_maxima": {},
        "shards": [],
    }
    with pytest.raises(ValueError, match="252-row shard boundary"):
        extractor.process_geometry_loader(
            loader=[], model=object(), base_rows_by_index={},
            output_dir="/tmp/geometry", manifest=manifest,
            get_inputs=lambda batch: {}, device=torch.device("cpu"),
        )

    complete = dict(manifest, complete=True, sample_count=300)
    touched = []

    def loader():
        touched.append(True)
        yield _loader_batch(1)

    assert extractor.process_geometry_loader(
        loader=loader(), model=object(), base_rows_by_index={},
        output_dir="/tmp/geometry", manifest=complete,
        get_inputs=lambda batch: {}, device=torch.device("cpu"),
    ) is complete
    assert touched == []
