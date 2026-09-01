import copy
import importlib.util
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "decide_density_target_box_scene_audit.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "density_scene_decision_under_test", str(_SCRIPT)
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _slice(count, selected025, selected050, top025, top050, iou):
    return {
        "sample_count": count,
        "target_point_count_sum": count * 100,
        "target_point_count_mean": 100.0,
        "selected_hits025": selected025,
        "selected_hits050": selected050,
        "selected_acc025": selected025 / float(count or 1),
        "selected_acc050": selected050 / float(count or 1),
        "top16_hits025": top025,
        "top16_hits050": top050,
        "top16_acc025": top025 / float(count or 1),
        "top16_acc050": top050 / float(count or 1),
        "matched_iou_sum": iou * count,
        "matched_iou_mean": iou,
        "matched_hits025": min(top025, count),
        "matched_hits050": min(top050, count),
        "matched_acc025": min(top025, count) / float(count or 1),
        "matched_acc050": min(top050, count) / float(count or 1),
        "matched_center_l1_sum": count * 0.1,
        "matched_center_l1_mean": 0.1,
        "matched_size_l1_sum": count * 0.2,
        "matched_size_l1_mean": 0.2,
    }


def _receipt(role):
    sparse_count = 1600
    dense_count = 4700
    zero_count = 29
    metrics = {
        "schema": "mcln-density-target-box-scene-metrics-v1",
        "sample_count": 6329,
        "sample_identity_count": 6329,
        "sample_identity_unique_count": 6329,
        "sample_identity_sha256": "a" * 64,
        "candidate_filter": "butd-cls-overlap-iou-strictly-greater-0.25",
        "ranking": "selected-source-scores-desc-query-index-asc",
        "top_k": 16,
        "slices": {
            "overall": _slice(6329, 4000, 3200, 5000, 4300, 0.60),
            "active_sparse": _slice(
                sparse_count, 800, 600, 1200, 1000, 0.50
            ),
            "dense": _slice(
                dense_count, 3180, 2580, 3780, 3280, 0.64
            ),
            "zero_point": _slice(
                zero_count, 20, 20, 20, 20, 0.30
            ),
        },
    }
    return {
        "schema": _MODULE.ROLE_SCHEMA,
        "role": role,
        "epoch": 57 if role == "parent" else 58,
        "checkpoint_path": "/protected/e57.pth",
        "checkpoint_sha256": _MODULE.PROTECTED_E57_SHA256,
        "checkpoint_epoch": 57,
        "density_aware_target_box_loss_weight": (
            1.0 if role == "method" else 0.0
        ),
        "split": {
            "schema": "mcln-density-target-box-nr3d-scene-fold-v1",
            "fold": 2,
            "fold_count": 5,
            "total_scenes": 511,
            "total_samples": 32919,
            "fit_scenes": 408,
            "holdout_scenes": 103,
            "fit_samples": 26590,
            "holdout_samples": 6329,
            "fit_sample_identity_sha256": "b" * 64,
            "holdout_sample_identity_sha256": "a" * 64,
        },
        "training": None if role == "parent" else {
            "batch_count": 100,
            "optimizer_step_count": 100,
            "sample_count": 1600,
            "sample_identity_count": 1600,
            "sample_identity_unique_count": 1600,
            "sample_identity_sha256": "c" * 64,
        },
        "evaluation": {
            "density_aware_target_box_scene_audit": metrics
        },
        "generated_weights": [],
        "formal_validation_accessed": False,
        "audit_only": True,
        "long_training_authorized": False,
    }


def _records():
    return {
        role: {"path": "/{}.json".format(role), "sha256": role[0] * 64}
        for role in ("parent", "control", "method")
    }


def _provenance_record():
    return {
        "path": "/pre_audit_provenance.json",
        "sha256": "e" * 64,
        "payload": {
            "schema": "mcln-density-target-box-scene-provenance-v1",
            "audit_only": True,
            "long_training_authorized": False,
        },
    }


def test_decision_passes_only_strict_sparse_proposal_gain_and_safety():
    parent = _receipt("parent")
    control = _receipt("control")
    method = _receipt("method")
    method_sparse = method["evaluation"][
        "density_aware_target_box_scene_audit"
    ]["slices"]["active_sparse"]
    method_sparse["top16_hits025"] = 1201
    method_sparse["top16_acc025"] = 1201 / 1600.0
    method_sparse["matched_iou_sum"] = 0.51 * 1600
    method_sparse["matched_iou_mean"] = 0.51

    decision = _MODULE.build_decision(
        parent, control, method, _records(), _provenance_record()
    )
    assert decision["density_gate_passed"] is True
    assert decision["next_allowed_step"] == "independent_review_only"
    assert decision["long_training_authorized"] is False
    assert decision["provenance"]["sha256"] == "e" * 64


def test_decision_seals_when_sparse_top16_does_not_strictly_improve():
    decision = _MODULE.build_decision(
        _receipt("parent"), _receipt("control"), _receipt("method"),
        _records(), _provenance_record(),
    )
    assert decision["density_gate_passed"] is False
    assert decision["next_allowed_step"] == "seal_method"
    assert decision["long_training_authorized"] is False


def test_decision_rejects_fit_identity_or_formal_access_drift():
    parent = _receipt("parent")
    control = _receipt("control")
    method = _receipt("method")
    method["training"]["sample_identity_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="fit row identities"):
        _MODULE.build_decision(
            parent, control, method, _records(), _provenance_record()
        )

    method = _receipt("method")
    method["formal_validation_accessed"] = True
    with pytest.raises(ValueError, match="formal validation"):
        _MODULE.build_decision(
            parent, control, method, _records(), _provenance_record()
        )
