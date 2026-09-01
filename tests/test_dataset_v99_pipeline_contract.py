import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from scripts import build_v99_pareto_contextual_artifact as builder
from scripts import cleanup_dataset_v99_intermediates as cleanup
from scripts import finalize_dataset_v99_pipeline as finalizer
from scripts import preflight_dataset_v99_geometry_panel as panel_preflight
from scripts import run_dataset_v99_official as official
from scripts.train_scanrefer_rec_selective_residual import (
    split_residual_joined_rows,
)
from test_train_scanrefer_rec_selective_residual import _geometry_artifact


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _portable_oof(dataset="nr3d", variant_count=7):
    protected = {"same": True}
    return {
        "schema": "rec-pareto-contextual-hierarchical-train-only-v1",
        "validation_data_accessed": False,
        "contaminated_calibration_accessed": False,
        "protected_before": protected,
        "protected_after": protected,
        "oof": {
            "passed": True,
            "predicates": {"positive": True},
            "diagnostics": {
                "delta_hits025": 4,
                "delta_hits050": 3,
                "switches": 9,
            },
        },
        "dataset_contract": {
            "dataset": dataset,
            "dataset_only": True,
            "joint_training": False,
            "backbone_sha256": "a" * 64,
            "query_count": 16,
            "variant_count": variant_count,
            "flat_candidate_count": 16 * variant_count,
            "method": "V99-contextual-pareto-16x7",
            "v99_script_sha256": "b" * 64,
        },
    }


def test_builder_materialization_binds_portable_artifact_chain(monkeypatch):
    artifact = _geometry_artifact()
    artifact["checkpoint_sha256"] = "a" * 64
    artifact["parent_artifact_sha256"] = "b" * 64
    loaded = {
        "parent": object(),
        "geometry_model": object(),
        "geometry_artifact": artifact,
    }
    split = {"fit_rows": [object()]}
    captured = {}

    def materialize(rows, parent, geometry_model, geometry_artifact, **kwargs):
        kwargs["artifact_validator"](geometry_artifact)
        captured.update(kwargs)
        return ["materialized"]

    monkeypatch.setattr(builder, "materialize_hierarchical_rows", materialize)
    result = builder._materialize_fit_rows(
        split,
        loaded,
        "cpu",
        True,
        {
            "backbone": {"sha256": "a" * 64},
            "parent": {"sha256": "b" * 64},
        },
    )

    assert result == ["materialized"]
    assert captured["device"] == "cpu"
    assert captured["require_contiguous"] is False


def test_portable_v99_oof_requires_real_16x7_contract(tmp_path):
    path = tmp_path / "oof.json"
    path.write_text(json.dumps(_portable_oof()), encoding="ascii")
    result = builder._validate_oof_result(
        path,
        expected_sha256=_sha256(path),
        portable_dataset_contract=True,
        dataset="nr3d",
    )
    assert result["dataset_contract"]["flat_candidate_count"] == 112

    path.write_text(json.dumps(_portable_oof(variant_count=1)), encoding="ascii")
    with pytest.raises(ValueError, match="dataset contract"):
        builder._validate_oof_result(
            path,
            expected_sha256=_sha256(path),
            portable_dataset_contract=True,
            dataset="nr3d",
        )


def test_portable_joined_split_is_scene_disjoint_and_dynamic():
    rows = []
    for index in range(20):
        identity = {
            "dataset_index": index,
            "scan_id": "scene_{:02d}".format(index // 4),
        }
        geometry = dict(identity)
        geometry["evaluator_valid"] = torch.tensor([index != 0])
        rows.append({"base": dict(identity), "geometry": geometry})
    split = split_residual_joined_rows(rows, portable_provenance=True)
    fit = {row["base"]["scan_id"] for row in split["fit_rows"]}
    calibration = {
        row["base"]["scan_id"] for row in split["calibration_rows"]
    }
    assert fit.isdisjoint(calibration)
    assert len(split["fit_rows"]) + len(split["calibration_rows"]) == 19
    assert all(
        bool(row["geometry"]["evaluator_valid"].any().item())
        for row in split["fit_rows"] + split["calibration_rows"]
    )
    assert split["metadata"]["sample_count"] == 19
    assert split["metadata"]["scene_count"] == 5
    with pytest.raises(ValueError, match="authoritative"):
        split_residual_joined_rows(rows)


def test_official_command_uses_runtime_and_records_joint_gt_provenance(tmp_path):
    args = type("Args", (), {
        "python_bin": str(tmp_path / "python"),
        "master_port": 5200,
        "dataset": "sr3d",
        "data_root": str(tmp_path / "data"),
        "expected_sample_count": 17726,
        "experiment": "sr3d_v99_official",
        "backbone_joint_training": True,
        "inference_uses_ground_truth": True,
    })()
    artifacts = {
        name: {"path": str(tmp_path / (name + ".pth"))}
        for name in ("backbone", "parent", "geometry", "hierarchical")
    }
    command = official._build_command(args, artifacts, tmp_path / "official")
    official._validate_command(command, args)
    assert "--eval_use_rec_hierarchical_reranker_scores" in command
    assert "--joint_det" in command
    assert "--butd_cls" in command
    assert "--butd" not in command
    assert command[command.index("--dataset") + 1] == "sr3d"


def test_official_receipt_discovery_uses_one_new_nested_receipt(tmp_path):
    output = tmp_path / "official"
    stale = output / "nr3d" / "old" / "1" / "eval_metrics_epoch_0.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="ascii")
    before = official._metric_receipts(output)

    nested = output / "nr3d" / "exp" / "2" / "eval_metrics_epoch_0.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}", encoding="ascii")
    assert official._new_metric_receipt(output, before) == nested.resolve()

    extra = output / "nr3d" / "exp" / "3" / "eval_metrics_epoch_0.json"
    extra.parent.mkdir(parents=True)
    extra.write_text("{}", encoding="ascii")
    with pytest.raises(ValueError, match="exactly one new"):
        official._new_metric_receipt(output, before)


def test_official_main_receipts_real_nested_log_layout(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "train_dist_mod.py").write_text("# test\n", encoding="ascii")
    output = tmp_path / "official"
    artifacts = {}
    for name in ("backbone", "parent", "geometry", "hierarchical"):
        path = tmp_path / (name + ".pth")
        path.write_bytes(name.encode("ascii"))
        _protect(path)
        artifacts[name] = path

    def fake_run(command, cwd, check):
        assert cwd == str(root.resolve())
        assert check is False
        nested = (
            output / "nr3d" / "nr3d_v99_official" / "1234567890"
            / "eval_metrics_epoch_0.json"
        )
        nested.parent.mkdir(parents=True)
        nested.write_text(json.dumps({
            "sample_count": 10,
            "position": {
                "learned_selector": {"hits025": 6, "hits050": 5}
            },
        }), encoding="ascii")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    assert official.main([
        "--dataset", "nr3d",
        "--expected-sample-count", "10",
        "--python-bin", str(tmp_path / "python"),
        "--project-root", str(root),
        "--data-root", str(tmp_path / "data"),
        "--master-port", "5200",
        "--experiment", "nr3d_v99_official",
        "--backbone-checkpoint", str(artifacts["backbone"]),
        "--parent-artifact", str(artifacts["parent"]),
        "--geometry-artifact", str(artifacts["geometry"]),
        "--hierarchical-artifact", str(artifacts["hierarchical"]),
        "--output-dir", str(output),
    ]) == 0
    result = json.loads((output / "official_result.json").read_text())
    assert result["eval_receipt"]["path"].endswith(
        "nr3d/nr3d_v99_official/1234567890/eval_metrics_epoch_0.json"
    )
    assert result["metrics"]["sample_count"] == 10


def test_geometry_panel_preflight_counts_only_fully_stratified_scenes():
    records = []
    for scene, buckets in (
            ("eligible", ["fail025", "mid", "pass050", "pass050"]),
            ("missing_mid", ["fail025", "pass050", "pass050", "pass050"]),
            ("too_short", ["fail025", "mid", "pass050"])):
        for index, bucket in enumerate(buckets):
            records.append({
                "scan_id": scene,
                "dataset_index": len(records),
                "bucket": bucket,
            })
    assert panel_preflight._eligible_scene_count(records, 4) == 1


def _protect(path):
    os.chmod(path, 0o444)
    return path


def test_finalize_and_cleanup_preserve_only_selected_checkpoint(tmp_path):
    root = tmp_path / "v99"
    artifacts = root / "artifacts"
    official = root / "official"
    backbone_run = tmp_path / "backbone_run"
    artifacts.mkdir(parents=True)
    official.mkdir()
    backbone_run.mkdir()
    for name in ("candidate_cache", "geometry_cache", "geometry_audit"):
        directory = root / name
        directory.mkdir()
        (directory / "reconstructible.bin").write_bytes(b"cache")

    initialization = tmp_path / "initialization.pth"
    initialization.write_bytes(b"init")
    _protect(initialization)
    backbone = backbone_run / "ckpt_best_rec_acc025.pth"
    backbone.write_bytes(b"backbone")
    _protect(backbone)
    other = backbone_run / "ckpt_epoch_last.pth"
    other.write_bytes(b"other")
    backbone_sha = _sha256(backbone)

    parent = artifacts / "parent.pth"
    torch.save({"parent": True}, parent)
    _protect(parent)
    geometry = artifacts / "geometry.pth"
    torch.save({
        "checkpoint_sha256": backbone_sha,
        "parent_artifact_sha256": _sha256(parent),
        "variant_names": ["g{}".format(index) for index in range(7)],
    }, geometry)
    _protect(geometry)
    hierarchy = artifacts / "hierarchy.pth"
    torch.save({
        "schema": finalizer.V99_SCHEMA,
        "deployable": True,
        "input_sha256": {
            "backbone": backbone_sha,
            "parent": _sha256(parent),
            "geometry": _sha256(geometry),
        },
        "policy": {"aggregate_margin": 0.13312220573425293},
    }, hierarchy)
    _protect(hierarchy)
    oof = artifacts / "oof.json"
    oof.write_text(json.dumps({
        "dataset_contract": {
            "dataset": "nr3d",
            "dataset_only": True,
            "joint_training": False,
            "reranker_dataset_only": True,
            "backbone_training_dataset_only": True,
            "backbone_joint_training": False,
            "inference_uses_ground_truth": False,
            "variant_count": 7,
            "flat_candidate_count": 112,
            "backbone_sha256": backbone_sha,
        },
        "oof": {"passed": True},
    }), encoding="utf-8")
    artifact_receipt = artifacts / "artifact_receipt.json"
    artifact_receipt.write_text(json.dumps({
        "oof_result_sha256": _sha256(oof),
    }), encoding="utf-8")
    audit_panel_preflight = artifacts / "geometry_panel_preflight.json"
    audit_panel_preflight.write_text(json.dumps({
        "schema": "mcln-dataset-v99-geometry-panel-preflight-v1",
        "passed": True,
        "dataset": "nr3d",
        "dataset_only": True,
        "checkpoint_sha256": backbone_sha,
        "required_scene_count": 64,
        "eligible_scene_count": 70,
        "expressions_per_scene": 4,
        "selected_sample_count": 256,
    }), encoding="utf-8")
    eval_receipt = official / "eval_metrics_epoch_0.json"
    eval_receipt.write_text(json.dumps({
        "sample_count": 10,
        "position": {"learned_selector": {"hits025": 6, "hits050": 5}},
        "position_subgroups": {
            "unique": {
                "sample_count": 4, "hits025": 4, "hits050": 3,
            },
            "multiple": {
                "sample_count": 6, "hits025": 3, "hits050": 2,
            },
        },
    }), encoding="utf-8")
    eval_metrics = json.loads(eval_receipt.read_text())
    def official_snapshot(path):
        return {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size": path.stat().st_size,
            "mode": 0o444,
        }
    official_result = official / "official_result.json"
    official_result.write_text(json.dumps({
        "schema": "mcln-dataset-v99-official-result-v1",
        "dataset": "nr3d",
        "sample_count": 10,
        "dataset_only": True,
        "joint_training": False,
        "inference_uses_ground_truth": False,
        "artifacts": {
            "backbone": official_snapshot(backbone),
            "parent": official_snapshot(parent),
            "geometry": official_snapshot(geometry),
            "hierarchical": official_snapshot(hierarchy),
        },
        "eval_receipt": {
            "path": str(eval_receipt.resolve()),
            "sha256": _sha256(eval_receipt),
            "size": eval_receipt.stat().st_size,
        },
        "metrics": eval_metrics,
        "command": ["python", "train_dist_mod.py", "--eval"],
    }), encoding="utf-8")

    pipeline_receipt = root / "pipeline_receipt.json"
    assert finalizer.main([
        "--dataset", "nr3d",
        "--expected-sample-count", "10",
        "--initialization-checkpoint", str(initialization),
        "--expected-initialization-sha256", _sha256(initialization),
        "--backbone-checkpoint", str(backbone),
        "--parent-artifact", str(parent),
        "--geometry-artifact", str(geometry),
        "--hierarchical-artifact", str(hierarchy),
        "--audit-panel-preflight", str(audit_panel_preflight),
        "--oof-result", str(oof),
        "--artifact-receipt", str(artifact_receipt),
        "--eval-receipt", str(eval_receipt),
        "--official-result", str(official_result),
        "--output", str(pipeline_receipt),
    ]) == 0
    receipt = json.loads(pipeline_receipt.read_text())
    assert receipt["method"]["flat_candidate_count"] == 112
    assert receipt["metrics"]["rec"] == {
        "counter_source": "final_deployed_position_subgroups",
        "sample_count": 10,
        "hits025": 7,
        "hits050": 5,
        "acc025": 0.7,
        "acc050": 0.5,
    }

    cleanup_receipt = root / "cleanup_receipt.json"
    assert cleanup.main([
        "--pipeline-root", str(root),
        "--pipeline-receipt", str(pipeline_receipt),
        "--backbone-run-dir", str(backbone_run),
        "--keep-checkpoint", str(backbone),
        "--output", str(cleanup_receipt),
    ]) == 0
    assert backbone.is_file()
    assert not other.exists()
    assert not (root / "candidate_cache").exists()
    assert not (root / "geometry_cache").exists()
    assert not (root / "geometry_audit").exists()
