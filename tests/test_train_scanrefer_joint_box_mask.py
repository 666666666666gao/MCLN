import pytest
import torch
import hashlib
import json

from models.rec_joint_box_mask import (
    JointBoxMaskAdapter,
    LEGACY_MASK_POLICY_INDEX,
    MASK_LOGIT_THRESHOLDS,
    MASK_SOURCE_NAMES,
)
from scripts.train_scanrefer_joint_box_mask import (
    FEATURE_DIM,
    TRAINER_SCHEMA,
    _joined_rows,
    build_joint_feature_batch,
    deterministic_scene_split,
    evaluate_quality_policy,
    load_joint_adapter_artifact,
    publication_gate,
    save_artifact,
    select_quality_policy,
    train_adapter,
    _validate_joint_cache_bindings,
    write_trial_receipt,
)


def _row(index, scene):
    return {
        "dataset_index": index,
        "scan_id": scene,
        "target_id": index,
        "features": torch.arange(16 * 152, dtype=torch.float32).reshape(16, 152),
        "geometry_features": torch.zeros(16, 7, 25),
        "geometry_valid": torch.ones(16, 7, dtype=torch.bool),
        "geometry_ious": torch.full((16, 7), 0.6),
        "mask_ious": torch.full((16, 3, 5), 0.6),
        "query_indices": torch.arange(16, dtype=torch.long),
        "candidate_valid": torch.ones(16, dtype=torch.bool),
    }


def test_scene_split_is_disjoint_and_deterministic():
    rows = [_row(i, "scene{:02d}".format(i // 2)) for i in range(12)]
    fit_a, cal_a, digest_a = deterministic_scene_split(rows, seed=0)
    fit_b, cal_b, digest_b = deterministic_scene_split(rows, seed=0)
    assert digest_a == digest_b
    assert sorted(r["dataset_index"] for r in fit_a + cal_a) == list(range(12))
    assert set(r["scan_id"] for r in fit_a).isdisjoint(
        set(r["scan_id"] for r in cal_a)
    )
    assert [r["dataset_index"] for r in fit_a] == [
        r["dataset_index"] for r in fit_b
    ]


def test_joint_feature_batch_has_query_major_variant_minor_contract():
    out = build_joint_feature_batch(_row(0, "scene00"))
    assert out["features"].shape == (112, FEATURE_DIM)
    assert out["valid_mask"].shape == (112,)
    assert out["box_ious"].shape == (112,)
    assert out["mask_ious"].shape == (112,)
    assert torch.equal(out["query_positions"][:7], torch.zeros(7, dtype=torch.long))
    assert out["mask_ious"].max().item() == pytest.approx(0.6)


def test_quality_policy_switches_only_when_predicted_box_tier_is_protected():
    # Candidate 1 has a better mask prediction but a lower predicted box tier;
    # candidate 2 has both a better mask and a protected box tier.
    mask_pred = torch.tensor([[0.60, 0.95, 0.90]])
    box_logits = torch.tensor([
        [[5.0, 5.0], [-5.0, -5.0], [5.0, 5.0]]
    ])
    baseline_scores = torch.tensor([[0.9, 0.8, 0.1]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    selected = select_quality_policy(
        mask_pred, box_logits, baseline_scores, valid,
        switch_margin=0.01, box_margin=0.05,
    )
    assert selected["selected_flat_index"].item() == 2
    assert selected["selected_flat_index"].item() != 1


def test_quality_policy_falls_back_when_no_safe_switch():
    mask_pred = torch.tensor([[0.60, 0.99]])
    box_logits = torch.tensor([[[5.0, 5.0], [-5.0, -5.0]]])
    baseline_scores = torch.tensor([[0.9, 0.8]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    selected = select_quality_policy(
        mask_pred, box_logits, baseline_scores, valid,
        switch_margin=0.01, box_margin=0.01,
    )
    assert selected["selected_flat_index"].item() == 0
    assert selected["fallback_count"] == 1


def test_train_adapter_updates_mask_policy_head_from_train_only_labels():
    torch.manual_seed(5)
    sample_count = 4
    fit = {
        "features": torch.randn(sample_count, 112, FEATURE_DIM),
        "valid_mask": torch.ones(
            sample_count, 112, dtype=torch.bool
        ),
        "box_ious": torch.rand(sample_count, 112),
        "mask_ious": torch.rand(sample_count, 112),
        "mask_policy_ious": torch.zeros(sample_count, 16, 15),
    }
    fit["mask_policy_ious"][..., 0] = 1.0
    model, history = train_adapter(
        fit,
        torch.zeros(FEATURE_DIM),
        torch.ones(FEATURE_DIM),
        hidden_dim=8,
        dropout=0.0,
        epochs=2,
        batch_size=2,
        seed=0,
        device="cpu",
    )

    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(history)))
    assert model.mask_policy_head.weight.abs().sum().item() > 0.0


def test_evaluate_quality_policy_reports_strict_metrics():
    baseline_box = torch.tensor([[0.6, 0.2]])
    selected_box = torch.tensor([[0.6, 0.7]])
    baseline_mask = torch.tensor([[0.4, 0.4]])
    selected_mask = torch.tensor([[0.4, 0.6]])
    selected_indices = torch.tensor([1])
    out = evaluate_quality_policy(
        baseline_box, selected_box, baseline_mask, selected_mask,
        selected_indices,
    )
    assert out["baseline_position_hits025"] == 1
    assert out["selected_position_hits025"] == 1
    assert out["baseline_mask_hits050"] == 0
    assert out["selected_mask_hits050"] == 1
    assert out["delta_mask_miou"] == pytest.approx(0.2)


def test_publication_gate_requires_all_registered_thresholds():
    passing = {
        "delta_position_acc025": 0.0,
        "delta_position_acc050": 0.0,
        "delta_mask_acc025": 0.0,
        "delta_mask_acc050": 0.02,
        "delta_mask_miou": 0.03,
        "position025_bootstrap_lcb": 0.0,
        "position050_bootstrap_lcb": 0.0,
    }
    assert publication_gate(passing)["pass"] is True
    failing = dict(passing, delta_mask_miou=0.029)
    assert publication_gate(failing)["pass"] is False


def test_joined_rows_preserve_nested_cache_contract_and_reject_query_drift():
    base = _row(0, "scene00")
    base["valid_mask"] = base.pop("candidate_valid")
    geometry = {
        "dataset_index": 0,
        "scan_id": "scene00",
        "target_id": 0,
        "query_indices": base["query_indices"].clone(),
        "candidate_valid": base["valid_mask"].clone(),
        "geometry_features": base["geometry_features"],
        "geometry_valid": base["geometry_valid"],
        "evaluator_valid": base["geometry_valid"],
        "geometry_ious": base["geometry_ious"],
    }
    mask = {
        "dataset_index": 0,
        "scan_id": "scene00",
        "target_id": 0,
        "query_indices": base["query_indices"].clone(),
        "candidate_valid": base["valid_mask"].clone(),
        "mask_ious": base["mask_ious"],
    }
    joined = _joined_rows([base], [geometry], [mask])[0]
    assert joined["_base_row"] is base
    assert joined["_geometry_row"] is geometry
    assert torch.equal(joined["candidate_valid"], base["valid_mask"])

    mask["query_indices"][0] = 99
    with pytest.raises(ValueError, match="query identities"):
        _joined_rows([base], [geometry], [mask])


def test_joint_cache_bindings_reject_stale_manifest_and_checkpoint(tmp_path):
    base_manifest_path = tmp_path / "base-manifest.json"
    geometry_manifest_path = tmp_path / "geometry-manifest.json"
    base_manifest_path.write_text("base", encoding="utf-8")
    geometry_manifest_path.write_text("geometry", encoding="utf-8")
    file_sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    base_manifest = {
        "checkpoint_sha256": "c" * 64,
        "sample_count": 2,
        "dataset_size": 2,
        "source_dataset_size": 2,
    }
    geometry_manifest = dict(base_manifest, cache_content_digest="g" * 64)
    joint_manifest = {
        "base_cache_manifest_sha256": file_sha(base_manifest_path),
        "geometry_cache_manifest_sha256": file_sha(geometry_manifest_path),
        "checkpoint_sha256": "c" * 64,
        "geometry_cache_content_digest": "g" * 64,
        "sample_count": 2,
        "dataset_size": 2,
        "source_dataset_size": 2,
        "complete": True,
    }
    _validate_joint_cache_bindings(
        joint_manifest, base_manifest, geometry_manifest,
        base_manifest_path, geometry_manifest_path,
    )
    joint_manifest["checkpoint_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        _validate_joint_cache_bindings(
            joint_manifest, base_manifest, geometry_manifest,
            base_manifest_path, geometry_manifest_path,
        )


def _deployable_artifact():
    model = JointBoxMaskAdapter(FEATURE_DIM, hidden_dim=8, dropout=0.0)
    return {
        "schema": TRAINER_SCHEMA,
        "deployable": True,
        "selection": "joint_adapter",
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
        "feature_names": ["feature_{:03d}".format(i) for i in range(FEATURE_DIM)],
        "model_config": {"input_dim": FEATURE_DIM, "hidden_dim": 8, "dropout": 0.0},
        "mask_policy_source_names": list(MASK_SOURCE_NAMES),
        "mask_policy_logit_thresholds": list(MASK_LOGIT_THRESHOLDS),
        "legacy_mask_policy_index": LEGACY_MASK_POLICY_INDEX,
        "model_state_dict": model.state_dict(),
        "feature_mean": torch.zeros(FEATURE_DIM),
        "feature_std": torch.ones(FEATURE_DIM),
        "switch_margin": 0.02,
        "box_margin": 0.05,
        "parent_artifact_sha256": "a" * 64,
        "geometry_artifact_sha256": "b" * 64,
        "backbone_checkpoint_sha256": "c" * 64,
    }


def test_joint_artifact_loader_accepts_only_deployable_frozen_artifact(tmp_path):
    path = tmp_path / "joint.pth"
    save_artifact(path, _deployable_artifact())
    model, artifact = load_joint_adapter_artifact(path)
    assert artifact["deployable"] is True
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert len(model._artifact_sha256) == 64

    rejected = tmp_path / "rejected.pth"
    payload = _deployable_artifact()
    payload["deployable"] = False
    payload["selection"] = "baseline"
    save_artifact(rejected, payload)
    with pytest.raises(ValueError, match="provenance"):
        load_joint_adapter_artifact(rejected)


def test_trial_receipt_keeps_metrics_when_baseline_is_selected(tmp_path):
    artifact = _deployable_artifact()
    artifact.update({
        "deployable": False,
        "selection": "baseline",
        "calibration_metrics": {"delta_mask_miou": 0.01},
        "calibration_gate": {"pass": False, "observed": {"delta_mask_miou": 0.01}},
        "split_digest": "d" * 64,
        "fit_sample_count": 10,
        "calibration_sample_count": 2,
        "training": {"seed": 0, "epochs": 1},
    })
    receipt_path = tmp_path / "trial.json"
    receipt = write_trial_receipt(receipt_path, artifact, tmp_path / "model.pth")
    assert receipt["selection"] == "baseline"
    assert receipt["calibration_metrics"]["delta_mask_miou"] == pytest.approx(0.01)
    assert json.loads(receipt_path.read_text())["calibration_gate"]["pass"] is False
