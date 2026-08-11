import os
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

import scripts.run_frozen_rec_joint_box_mask_official as official
from models.rec_joint_box_mask import (
    JointBoxMaskAdapter,
    LEGACY_MASK_POLICY_INDEX,
    MASK_LOGIT_THRESHOLDS,
    MASK_SOURCE_NAMES,
)
from scripts.train_scanrefer_joint_box_mask import TRAINER_SCHEMA


def _write_inputs(tmp_path):
    paths = {}
    for name, payload in (
        ("adapter", b"adapter"),
        ("backbone", b"backbone"),
        ("parent", b"parent"),
        ("geometry", b"geometry"),
    ):
        path = tmp_path / (name + ".pth")
        path.write_bytes(payload)
        if name != "adapter":
            os.chmod(str(path), 0o444)
        paths[name] = path
    model = JointBoxMaskAdapter(179, hidden_dim=8, dropout=0.0)
    artifact = {
        "schema": TRAINER_SCHEMA,
        "deployable": True,
        "selection": "joint_adapter",
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
        "feature_names": ["feature_{:03d}".format(i) for i in range(179)],
        "model_config": {"input_dim": 179, "hidden_dim": 8, "dropout": 0.0},
        "model_state_dict": model.state_dict(),
        "mask_policy_source_names": list(MASK_SOURCE_NAMES),
        "mask_policy_logit_thresholds": list(MASK_LOGIT_THRESHOLDS),
        "legacy_mask_policy_index": LEGACY_MASK_POLICY_INDEX,
        "feature_mean": torch.zeros(179),
        "feature_std": torch.ones(179),
        "switch_margin": 0.02,
        "box_margin": 0.05,
        "parent_artifact_sha256": hashlib.sha256(
            paths["parent"].read_bytes()
        ).hexdigest(),
        "geometry_artifact_sha256": hashlib.sha256(
            paths["geometry"].read_bytes()
        ).hexdigest(),
        "backbone_checkpoint_sha256": hashlib.sha256(
            paths["backbone"].read_bytes()
        ).hexdigest(),
    }
    torch.save(artifact, paths["adapter"])
    os.chmod(str(paths["adapter"]), 0o444)
    return paths


def _bind_protected(monkeypatch, paths):
    bindings = (
        ("OFFICIAL_CHECKPOINT_PATH", "OFFICIAL_CHECKPOINT_SHA256", paths["backbone"]),
        ("OFFICIAL_PARENT_ARTIFACT_PATH", "OFFICIAL_PARENT_ARTIFACT_SHA256", paths["parent"]),
        ("OFFICIAL_SELECTED_ARTIFACT_PATH", "OFFICIAL_SELECTED_ARTIFACT_SHA256", paths["geometry"]),
    )
    for path_name, sha_name, path in bindings:
        monkeypatch.setattr(official.geometry_official, path_name, path)
        monkeypatch.setattr(
            official.geometry_official,
            sha_name,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def _metrics_text():
    return "\n".join((
        "length of testing dataset: 9508",
        "last_ position alignment Acc0.25: Top-1: 0.63105, Top-5: 0.7",
        "last_ position alignment Acc0.50: Top-1: 0.48601, Top-5: 0.6",
        "mask_sem 0.45000",
        "overall25 0.59003",
        "overall50 0.50999",
        "inference_uses_ground_truth=false",
    ))


def test_metric_parser_recovers_integer_position_hits_and_mask_metrics():
    result = official._parse_metrics(_metrics_text())
    assert result["position_hits025"] == 6000
    assert result["position_hits050"] == 4621
    assert result["mask_acc025"] == pytest.approx(0.59003)
    assert result["mask_acc050"] == pytest.approx(0.50999)
    assert result["mask_miou"] == pytest.approx(0.45)
    assert result["mask_hits025"] == 5610
    assert result["mask_hits050"] == 4849


def test_metric_parser_rejects_duplicate_or_missing_mask_lines():
    with pytest.raises(ValueError, match="missing or duplicating"):
        official._parse_metrics(_metrics_text().replace("mask_sem 0.45000", ""))
    with pytest.raises(ValueError, match="missing or duplicating"):
        official._parse_metrics(_metrics_text() + "\noverall50 0.51")


def test_real_run_requires_one_population_line_and_explicit_gt_false(
        monkeypatch, tmp_path):
    paths = _write_inputs(tmp_path)
    claim = tmp_path / "claims" / "joint.claim"
    monkeypatch.setattr(official, "JOINT_CLAIM_PATH", claim)
    _bind_protected(monkeypatch, paths)

    def fake_run(_command, **kwargs):
        kwargs["stdout"].write(_metrics_text().encode("utf-8"))
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="ground-truth"):
        monkeypatch.setattr(
            official.subprocess, "run",
            lambda _command, **kwargs: (
                kwargs["stdout"].write(
                    _metrics_text().replace(
                        "inference_uses_ground_truth=false",
                        "inference_uses_ground_truth=true",
                    ).encode("utf-8")
                ),
                kwargs["stdout"].flush(),
                SimpleNamespace(returncode=0),
            )[-1],
        )
        official.run_official(paths["adapter"], tmp_path / "bad", dry_run=False)


def test_claim_is_exclusive_for_real_run_and_dry_run_does_not_claim(
        monkeypatch, tmp_path):
    paths = _write_inputs(tmp_path)
    claim = tmp_path / "claims" / "joint.claim"
    monkeypatch.setattr(official, "JOINT_CLAIM_PATH", claim)
    _bind_protected(monkeypatch, paths)
    dry_result = official.run_official(
        paths["adapter"], tmp_path / "dry-run", dry_run=True
    )
    assert dry_result["validation_data_accessed"] is False
    assert dry_result["acceptance_gate_pass"] is False
    assert not claim.exists()

    def fake_run(_command, **kwargs):
        kwargs["stdout"].write(_metrics_text().encode("utf-8"))
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(official.subprocess, "run", fake_run)
    result = official.run_official(paths["adapter"], tmp_path / "run")
    assert result["validation_data_accessed"] is True
    assert result["acceptance_gate_pass"] is True
    assert claim.is_file()
    with pytest.raises(FileExistsError, match="claim"):
        official.run_official(paths["adapter"], tmp_path / "run2")


def test_authoritative_command_contains_joint_flags(tmp_path):
    paths = _write_inputs(tmp_path)
    command = official.build_authoritative_command(paths["adapter"], tmp_path / "run")
    assert "--eval_use_rec_joint_box_mask" in command
    assert command[command.index("--rec_joint_box_mask_checkpoint") + 1] == str(paths["adapter"].resolve())


def test_runner_rejects_wrong_protected_sha_before_claim(monkeypatch, tmp_path):
    paths = _write_inputs(tmp_path)
    claim = tmp_path / "claims" / "joint.claim"
    monkeypatch.setattr(official, "JOINT_CLAIM_PATH", claim)
    _bind_protected(monkeypatch, paths)
    monkeypatch.setattr(
        official.geometry_official, "OFFICIAL_CHECKPOINT_SHA256", "0" * 64
    )
    with pytest.raises(ValueError, match="SHA"):
        official.run_official(paths["adapter"], tmp_path / "run", dry_run=True)
    assert not claim.exists()


def test_runner_rejects_non_deployable_adapter_before_claim(monkeypatch, tmp_path):
    paths = _write_inputs(tmp_path)
    claim = tmp_path / "claims" / "joint.claim"
    monkeypatch.setattr(official, "JOINT_CLAIM_PATH", claim)
    _bind_protected(monkeypatch, paths)
    artifact = torch.load(paths["adapter"], map_location="cpu")
    artifact["deployable"] = False
    torch.save(artifact, paths["adapter"])
    os.chmod(str(paths["adapter"]), 0o444)
    with pytest.raises(ValueError, match="adapter"):
        official.run_official(paths["adapter"], tmp_path / "run", dry_run=True)
    assert not claim.exists()


def test_post_claim_subprocess_error_writes_failure_receipt(monkeypatch, tmp_path):
    paths = _write_inputs(tmp_path)
    claim = tmp_path / "claims" / "joint.claim"
    output = tmp_path / "run"
    monkeypatch.setattr(official, "JOINT_CLAIM_PATH", claim)
    _bind_protected(monkeypatch, paths)
    monkeypatch.setattr(
        official.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("launch failed")),
    )
    with pytest.raises(OSError, match="launch failed"):
        official.run_official(paths["adapter"], output)
    failure = output / "official_failure.json"
    assert claim.is_file()
    assert failure.is_file()
    assert json.loads(failure.read_text())["status"] == "failure"
    assert oct(failure.stat().st_mode & 0o777) == "0o444"
