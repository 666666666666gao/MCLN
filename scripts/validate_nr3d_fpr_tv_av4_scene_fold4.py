from __future__ import print_function

import argparse
import hashlib
import json
import math
import os


SCHEMA = "mcln-fpr-tv-av4-scene-fold4-decision-v1"
PRE_SCHEMA = "mcln-fpr-tv-av4-scene-fold4-pre-audit-v1"
RECEIPT_SCHEMA = "mcln-fpr-tv-scene-disjoint-audit-v1"
METRICS_SCHEMA = "mcln-retrain-metrics-v1"
COUNTS_SCHEMA = "mcln-fpr-tv-decision-counts-v1"
SPLIT_SCHEMA = "mcln-fpr-tv-nr3d-scene-fold-v1"
CONFIG_SCHEMA = "mcln-fpr-tv-av4-scene-fold-config-v1"
EXPECTED_PRE_PATH_KEYS = {
    "launcher",
    "train_entry",
    "main_utils",
    "losses",
    "parent_relative_text_verifier",
    "parent_relative_text_verifier_tests",
    "scene_audit_tests",
    "finite_training_tests",
    "fold4_contract_tests",
    "fold4_spec",
    "fold4_validator",
    "model",
    "source_choice_selector",
    "dataset",
    "snapshot_executor",
    "resume_checkpoint",
    "groupfree_checkpoint",
    "data_manifest",
    "runtime_manifest",
    "fpr_v1_fold0_decision",
    "fpr_v1_fold0_receipt",
    "fpr_v2_fold1_decision",
    "fpr_v2_fold1_receipt",
    "density_fold2_decision",
    "fpr_v3_fold3_decision",
    "fpr_v3_fold3_receipt",
    "av4_mechanism_decision",
    "av4_mechanism_receipt",
    "static_executor",
    "static_source",
    "train_command",
}


def load_json_with_sha(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_transition(record, threshold, sample_count):
    _require(isinstance(record, dict), "missing threshold record")
    integer_names = (
        "fix_count",
        "break_count",
        "kept_correct_count",
        "kept_wrong_count",
        "parent_hits",
        "selected_hits",
        "net_hits",
    )
    for name in integer_names:
        _require(
            isinstance(record.get(name), int)
            and not isinstance(record.get(name), bool),
            "{} must be an integer".format(name),
        )
    fix_count = record["fix_count"]
    break_count = record["break_count"]
    kept_correct = record["kept_correct_count"]
    kept_wrong = record["kept_wrong_count"]
    _require(
        min(fix_count, break_count, kept_correct, kept_wrong) >= 0,
        "transition count is negative",
    )
    _require(
        fix_count + break_count + kept_correct + kept_wrong == sample_count,
        "transition counts do not partition held-out rows",
    )
    _require(
        record["parent_hits"] == break_count + kept_correct,
        "parent hit identity changed",
    )
    _require(
        record["selected_hits"] == fix_count + kept_correct,
        "selected hit identity changed",
    )
    _require(
        record["net_hits"] == fix_count - break_count,
        "net-hit identity changed",
    )
    _require(
        math.isclose(
            float(record.get("threshold")),
            threshold,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "threshold changed",
    )
    _require(
        math.isclose(
            float(record.get("parent_accuracy")),
            record["parent_hits"] / float(sample_count),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "parent accuracy is inconsistent",
    )
    _require(
        math.isclose(
            float(record.get("selected_accuracy")),
            record["selected_hits"] / float(sample_count),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "selected accuracy is inconsistent",
    )
    return record


def validate_scene_artifacts(receipt, metrics, expected):
    _require(receipt.get("schema") == RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt.get("epoch") == 58, "audit epoch changed")
    _require(receipt.get("checkpoint_epoch") == 57, "source epoch changed")
    _require(
        receipt.get("checkpoint_sha256") == expected["checkpoint_sha256"],
        "source checkpoint SHA changed",
    )
    _require(receipt.get("audit_only") is True, "receipt is not audit-only")
    _require(
        receipt.get("formal_validation_accessed") is False,
        "formal validation was accessed",
    )
    _require(
        receipt.get("long_training_authorized") is False,
        "receipt authorized long training",
    )
    _require(receipt.get("generated_weights") == [], "weight was generated")
    _require(receipt.get("evaluation") == metrics, "metrics bytes disagree")

    split = receipt.get("split", {})
    expected_split = {
        "schema": SPLIT_SCHEMA,
        "fold": 4,
        "fold_count": 5,
        "total_scenes": 511,
        "total_samples": 32919,
        "fit_scenes": 417,
        "fit_samples": 27004,
        "holdout_scenes": 94,
        "holdout_samples": 5915,
        "hash": "sha256-prefix32-mod5",
    }
    for name, value in expected_split.items():
        _require(split.get(name) == value, "split {} changed".format(name))
    for name in (
        "fit_sample_identity_sha256",
        "holdout_sample_identity_sha256",
    ):
        value = split.get(name)
        _require(
            isinstance(value, str) and len(value) == 64,
            "split identity SHA is missing",
        )

    config = receipt.get("frozen_config", {})
    _require(config.get("schema") == CONFIG_SCHEMA, "config schema changed")
    _require(
        config.get("sha256") == expected["config_sha256"],
        "config SHA changed",
    )
    values = config.get("values", {})
    _require(
        values.get("fpr_scene_disjoint_av4_audit") is True,
        "A-V4 audit flag is missing",
    )
    _require(
        values.get("parent_relative_text_verifier_counterfactual_training")
        is True,
        "counterfactual training flag is missing",
    )

    training = receipt.get("training", {})
    required_training = {
        "schema": "mcln-train-loss-epoch-v1",
        "batch_count": 1688,
        "optimizer_step_count": 1688,
        "sample_count": 27004,
        "sample_identity_count": 27004,
        "sample_identity_unique_count": 27004,
        "sample_identity_sha256": split["fit_sample_identity_sha256"],
    }
    for name, value in required_training.items():
        _require(
            training.get(name) == value,
            "training {} changed".format(name),
        )
    for section in ("loss_means", "stat_means"):
        values = training.get(section)
        _require(isinstance(values, dict) and values, "missing " + section)
        _require(
            all(finite_number(value) for value in values.values()),
            "non-finite " + section,
        )
    stats = training["stat_means"]
    for axis in ("actual", "counterfactual"):
        prefix = "parent_relative_text_verifier_{}_".format(axis)
        _require(
            float(stats.get(prefix + "sample_count", 0.0)) > 0.0,
            axis + " supervision is absent",
        )
        _require(
            float(stats.get(prefix + "selected_score_gradient_l1", 0.0))
            > 0.0,
            axis + " score-axis gradient is absent",
        )
        _require(
            float(stats.get(prefix + "nonfinite_count", -1.0)) == 0.0,
            axis + " non-finite count is nonzero",
        )
    _require(
        float(stats.get(
            "parent_relative_text_verifier_counterfactual_view_count", 0.0
        )) > 0.0,
        "counterfactual views were not exercised",
    )
    _require(float(stats.get("grad_norm", 0.0)) > 0.0, "gradient is absent")

    state = receipt.get("state_integrity", {})
    outputs = receipt.get("output_integrity", {})
    _require(state.get("frozen_exact") is True, "frozen state changed")
    _require(
        state.get("trainable_changed") is True,
        "A-V4 trainable state did not change",
    )
    _require(outputs.get("exact") is True, "frozen outputs changed")
    _require(
        state.get("before", {}).get("frozen", {}).get("sha256")
        == state.get("after", {}).get("frozen", {}).get("sha256"),
        "frozen state hashes disagree",
    )
    _require(
        state.get("before", {}).get("trainable", {}).get("sha256")
        != state.get("after", {}).get("trainable", {}).get("sha256"),
        "trainable state hashes did not change",
    )
    _require(
        outputs.get("before", {}).get("combined_sha256")
        == outputs.get("after", {}).get("combined_sha256"),
        "output hashes disagree",
    )

    _require(metrics.get("schema") == METRICS_SCHEMA, "metrics schema changed")
    _require(metrics.get("sample_count") == 5915, "held-out count changed")
    diagnostics = metrics.get(
        "parent_relative_text_verifier_scene_audit", {}
    )
    _require(
        diagnostics.get("schema") == COUNTS_SCHEMA,
        "decision counts schema changed",
    )
    _require(diagnostics.get("sample_count") == 5915, "decision count changed")
    switch_count = diagnostics.get("switch_count")
    _require(
        isinstance(switch_count, int) and not isinstance(switch_count, bool),
        "switch count is not an integer",
    )
    _require(0 <= switch_count <= 5915, "switch count is out of range")
    threshold025 = validate_transition(
        diagnostics.get("thresholds", {}).get("025"), 0.25, 5915
    )
    threshold050 = validate_transition(
        diagnostics.get("thresholds", {}).get("050"), 0.50, 5915
    )
    gate_failures = []
    if not state["frozen_exact"]:
        gate_failures.append("frozen_model_state_drift")
    if not state["trainable_changed"]:
        gate_failures.append("trainable_state_unchanged")
    if not outputs["exact"]:
        gate_failures.append("box_mask_parent_output_drift")
    if switch_count <= 0:
        gate_failures.append("no_heldout_switch")
    if threshold025["fix_count"] <= threshold025["break_count"]:
        gate_failures.append("acc025_fix_not_greater_than_break")
    if threshold050["fix_count"] < threshold050["break_count"]:
        gate_failures.append("acc050_net_negative")
    gate_pass = not gate_failures
    _require(
        receipt.get("fold_gate_pass") is gate_pass,
        "receipt fold gate disagrees with fixed gate",
    )
    _require(
        receipt.get("gate_failures") == gate_failures,
        "receipt gate failures disagree with fixed gate",
    )
    _require(
        receipt.get("next_stage")
        == ("independent_review_only" if gate_pass else "method_sealed"),
        "receipt next stage changed",
    )
    return diagnostics, gate_pass, gate_failures


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--pre-audit", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--static-executor-sha256", required=True)
    parser.add_argument("--static-source-sha256", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--groupfree-sha256", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    receipt, receipt_sha = load_json_with_sha(args.receipt)
    metrics, metrics_sha = load_json_with_sha(args.metrics)
    pre, pre_sha = load_json_with_sha(args.pre_audit)
    _require(pre.get("schema") == PRE_SCHEMA, "pre-audit schema changed")
    _require(
        set(pre.get("paths", {})) == EXPECTED_PRE_PATH_KEYS,
        "pre-audit path closure changed",
    )
    current = {
        name: sha256_file(path) for name, path in pre.get("paths", {}).items()
    }
    _require(current == pre.get("observed_sha256"), "artifact drift detected")
    fixed = {
        "launcher": args.launcher_sha256,
        "resume_checkpoint": args.checkpoint_sha256,
        "groupfree_checkpoint": args.groupfree_sha256,
        "data_manifest": args.data_manifest_sha256,
        "runtime_manifest": args.runtime_manifest_sha256,
        "static_executor": args.static_executor_sha256,
        "static_source": args.static_source_sha256,
    }
    for name, expected_sha in fixed.items():
        _require(current.get(name) == expected_sha, name + " SHA changed")
    expected = {
        "checkpoint_sha256": args.checkpoint_sha256,
        "config_sha256": args.config_sha256,
    }
    diagnostics, gate_pass, gate_failures = validate_scene_artifacts(
        receipt, metrics, expected
    )
    decision = {
        "schema": SCHEMA,
        "audit_only": True,
        "formal_validation_accessed": False,
        "long_training_authorized": False,
        "fold": 4,
        "fit_samples": 27004,
        "holdout_samples": 5915,
        "optimizer_step_count": 1688,
        "fold_gate_pass": gate_pass,
        "gate_failures": gate_failures,
        "switch_count": diagnostics["switch_count"],
        "thresholds": diagnostics["thresholds"],
        "next_step_if_passed": "independent_review_only",
        "receipt": os.path.realpath(args.receipt),
        "receipt_sha256": receipt_sha,
        "metrics": os.path.realpath(args.metrics),
        "metrics_sha256": metrics_sha,
        "pre_audit_provenance": os.path.realpath(args.pre_audit),
        "pre_audit_provenance_sha256": pre_sha,
        "code_and_input_sha256": current,
        "checkpoint_sha256": args.checkpoint_sha256,
        "config_sha256": args.config_sha256,
        "launcher_sha256": args.launcher_sha256,
        "static_executor_sha256": args.static_executor_sha256,
        "static_source_sha256": args.static_source_sha256,
        "runtime_manifest_sha256": args.runtime_manifest_sha256,
        "data_manifest_sha256": args.data_manifest_sha256,
        "groupfree_sha256": args.groupfree_sha256,
    }
    payload = (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(
        args.decision, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(
        os.path.dirname(os.path.realpath(args.decision)),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps(decision, sort_keys=True))
    if not decision["fold_gate_pass"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
