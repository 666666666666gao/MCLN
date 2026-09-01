from pathlib import Path
import copy
import hashlib

from scripts.validate_nr3d_fpr_tv_av4_scene_fold4 import (
    validate_scene_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_nr3d_fpr_tv_av4_scene_disjoint_fold4.sh"
VALIDATOR = (
    ROOT / "scripts" / "validate_nr3d_fpr_tv_av4_scene_fold4.py"
)
SPEC = ROOT / "FPR_TV_AV4_SCENE_DISJOINT_FOLD4_SPEC_2026-09-01.md"


def _launcher_text():
    return LAUNCHER.read_text(encoding="utf-8")


def _validator_text():
    return VALIDATOR.read_text(encoding="utf-8")


def test_fold4_launcher_has_only_the_preregistered_scene_audit_contract():
    text = _launcher_text()

    launcher_required = (
        "readonly FOLD=4",
        "readonly EXPECTED_FIT_SAMPLES=27004",
        "readonly EXPECTED_HOLDOUT_SAMPLES=5915",
        "readonly EXPECTED_OPTIMIZER_STEPS=1688",
        "readonly MIN_FREE_GB=5",
        "--fpr_scene_disjoint_audit",
        "--fpr_scene_disjoint_av4_audit",
        "--parent_relative_text_verifier_counterfactual_training",
        'fpr_scene_disjoint_audit_fold_${FOLD}_epoch_${AUDIT_EPOCH}.json',
        'eval_metrics_epoch_${AUDIT_EPOCH}.json',
        'readonly DECISION="${AUDIT_ROOT}/decision.json"',
    )
    validator_required = (
        'SCHEMA = "mcln-fpr-tv-av4-scene-fold4-decision-v1"',
        '"schema": SCHEMA',
        '"audit_only": True',
        '"formal_validation_accessed": False',
        '"long_training_authorized": False',
    )
    for marker in launcher_required:
        assert marker in text
    validator = _validator_text()
    for marker in validator_required:
        assert marker in validator

    forbidden = (
        "${FAILURE_EVIDENCE",
        "${AUDIT_BATCHES",
        '-name "train_audit_receipt_epoch_${AUDIT_EPOCH}.json"',
        'readonly DECISION="${AUDIT_ROOT}/counterfactual_parent_decision.json"',
        "formal_exact_sample_count",
        "7899",
    )
    for marker in forbidden:
        assert marker not in text


def test_fold4_wrapper_recomputes_the_fixed_integer_gate():
    text = _validator_text()

    assert 'if switch_count <= 0:' in text
    assert 'threshold025["fix_count"] <= threshold025["break_count"]' in text
    assert 'threshold050["fix_count"] < threshold050["break_count"]' in text
    assert 'receipt.get("fold_gate_pass") is gate_pass' in text
    assert 'receipt.get("gate_failures") == gate_failures' in text
    assert 'if not decision["fold_gate_pass"]:' in text
    assert 'raise SystemExit(20)' in text


def test_fold4_wrapper_binds_all_consumed_history_and_never_chains_runs():
    text = _launcher_text()

    history = (
        "fpr_v1_fold0_decision",
        "fpr_v1_fold0_receipt",
        "fpr_v2_fold1_decision",
        "fpr_v2_fold1_receipt",
        "density_fold2_decision",
        "fpr_v3_fold3_decision",
        "fpr_v3_fold3_receipt",
        "av4_mechanism_decision",
        "av4_mechanism_receipt",
    )
    for name in history:
        assert name in text
    assert "verify_preregistered_history absent" in text
    assert "verify_preregistered_history consumed" in text
    assert 'current_root_state == "absent"' in text
    assert 'current_root_state == "consumed"' in text
    assert '$$(' not in text
    assert (
        '"${AUDIT_ROOT}/runtime_output" \\\n'
        '  "${GROUPFREE_CHECKPOINT}" \\\n'
        '  "${SOURCE_CHECKPOINT}"'
    ) in text

    validator = _validator_text()
    assert "next_step_if_passed" in validator
    assert "independent_review_only" in validator
    assert "run_nr3d" not in validator.split("next_step_if_passed", 1)[1]
    assert "sr3d" not in text.lower()


def test_fold4_spec_preserves_current_project_targets_and_exclusions():
    text = SPEC.read_text(encoding="utf-8")

    assert "5 GiB" in text
    assert "0.893 GiB" in text
    assert "strictly above 60.0%" in text
    assert "strictly above 68.9%" in text
    assert "4740/7899" in text
    assert "12214/17726" in text
    assert "baseline reproduction" in text
    assert "Section/Experiment 7" in text
    assert "Section/Experiment 8" in text
    assert "E0--E7" in text


def test_fold4_static_executor_is_reproducibly_bound_to_the_launcher():
    binary = ROOT / "scripts" / "mcln_fpr_tv_av4_scene_fold4_static_exec.x86_64"
    receipt_path = (
        ROOT / "scripts" / "mcln_fpr_tv_av4_scene_fold4_static_exec.build_receipt"
    )
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    receipt = dict(
        line.split("=", 1)
        for line in receipt_path.read_text(encoding="utf-8").splitlines()
        if line
    )

    assert digest == "15ab2d486f1b231ff28eb50fedbeaed1744913a172d18ce139fef50f184c0972"
    assert receipt["artifact_sha256"] == digest
    assert receipt["artifact_size"] == str(binary.stat().st_size)
    assert receipt["artifact_mode"] == "0755"
    assert receipt["source_sha256_lf"] == (
        "0bf6cfcfb015a91474579ba0c0f186c49c6a38695601d904d3216724cc67dcdc"
    )
    assert receipt["trust_root"] == "/root/mcln_fpr_av4_scene_fold4_trust/v1"
    assert receipt["shared_gpu_lock"] == (
        "/root/autodl-tmp/mcln_v99_backbone_gpu0.lock"
    )
    assert digest in _launcher_text()


def _transition(threshold, fix, break_count, kept_correct, kept_wrong):
    sample_count = fix + break_count + kept_correct + kept_wrong
    parent_hits = break_count + kept_correct
    selected_hits = fix + kept_correct
    return {
        "threshold": threshold,
        "fix_count": fix,
        "break_count": break_count,
        "kept_correct_count": kept_correct,
        "kept_wrong_count": kept_wrong,
        "parent_hits": parent_hits,
        "selected_hits": selected_hits,
        "net_hits": fix - break_count,
        "parent_accuracy": parent_hits / float(sample_count),
        "selected_accuracy": selected_hits / float(sample_count),
    }


def _passing_artifacts():
    identity = "1" * 64
    diagnostics = {
        "schema": "mcln-fpr-tv-decision-counts-v1",
        "sample_count": 5915,
        "switch_count": 5,
        "thresholds": {
            "025": _transition(0.25, 2, 1, 4000, 1912),
            "050": _transition(0.50, 1, 1, 4000, 1913),
        },
    }
    metrics = {
        "schema": "mcln-retrain-metrics-v1",
        "sample_count": 5915,
        "parent_relative_text_verifier_scene_audit": diagnostics,
    }
    receipt = {
        "schema": "mcln-fpr-tv-scene-disjoint-audit-v1",
        "epoch": 58,
        "checkpoint_epoch": 57,
        "checkpoint_sha256": "a" * 64,
        "audit_only": True,
        "formal_validation_accessed": False,
        "long_training_authorized": False,
        "generated_weights": [],
        "split": {
            "schema": "mcln-fpr-tv-nr3d-scene-fold-v1",
            "fold": 4,
            "fold_count": 5,
            "hash": "sha256-prefix32-mod5",
            "total_scenes": 511,
            "total_samples": 32919,
            "fit_scenes": 417,
            "fit_samples": 27004,
            "holdout_scenes": 94,
            "holdout_samples": 5915,
            "fit_sample_identity_sha256": identity,
            "holdout_sample_identity_sha256": "2" * 64,
        },
        "frozen_config": {
            "schema": "mcln-fpr-tv-av4-scene-fold-config-v1",
            "sha256": "b" * 64,
            "values": {
                "fpr_scene_disjoint_av4_audit": True,
                "parent_relative_text_verifier_counterfactual_training": True,
            },
        },
        "training": {
            "schema": "mcln-train-loss-epoch-v1",
            "batch_count": 1688,
            "optimizer_step_count": 1688,
            "sample_count": 27004,
            "sample_identity_count": 27004,
            "sample_identity_unique_count": 27004,
            "sample_identity_sha256": identity,
            "loss_means": {"loss": 1.0},
            "stat_means": {
                "grad_norm": 1.0,
                "parent_relative_text_verifier_actual_sample_count": 1.0,
                "parent_relative_text_verifier_actual_selected_score_gradient_l1": 0.1,
                "parent_relative_text_verifier_actual_nonfinite_count": 0.0,
                "parent_relative_text_verifier_counterfactual_sample_count": 1.0,
                "parent_relative_text_verifier_counterfactual_selected_score_gradient_l1": 0.1,
                "parent_relative_text_verifier_counterfactual_nonfinite_count": 0.0,
                "parent_relative_text_verifier_counterfactual_view_count": 1.0,
            },
        },
        "evaluation": metrics,
        "state_integrity": {
            "frozen_exact": True,
            "trainable_changed": True,
            "before": {
                "frozen": {"sha256": "3" * 64},
                "trainable": {"sha256": "4" * 64},
            },
            "after": {
                "frozen": {"sha256": "3" * 64},
                "trainable": {"sha256": "5" * 64},
            },
        },
        "output_integrity": {
            "exact": True,
            "before": {"combined_sha256": "6" * 64},
            "after": {"combined_sha256": "6" * 64},
        },
        "fold_gate_pass": True,
        "gate_failures": [],
        "next_stage": "independent_review_only",
    }
    return receipt, metrics


def test_fold4_artifact_validator_accepts_only_the_fixed_positive_gate():
    receipt, metrics = _passing_artifacts()
    diagnostics, gate_pass, failures = validate_scene_artifacts(
        receipt,
        metrics,
        {"checkpoint_sha256": "a" * 64, "config_sha256": "b" * 64},
    )
    assert diagnostics["switch_count"] == 5
    assert gate_pass is True
    assert failures == []

    failed = copy.deepcopy(receipt)
    failed_metrics = copy.deepcopy(metrics)
    failed_record = _transition(0.25, 1, 1, 4000, 1913)
    failed_metrics["parent_relative_text_verifier_scene_audit"][
        "thresholds"
    ]["025"] = failed_record
    failed["evaluation"] = failed_metrics
    failed["fold_gate_pass"] = False
    failed["gate_failures"] = ["acc025_fix_not_greater_than_break"]
    failed["next_stage"] = "method_sealed"
    _, gate_pass, failures = validate_scene_artifacts(
        failed,
        failed_metrics,
        {"checkpoint_sha256": "a" * 64, "config_sha256": "b" * 64},
    )
    assert gate_pass is False
    assert failures == ["acc025_fix_not_greater_than_break"]
