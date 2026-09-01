"""Fail-closed paired decision for the density target-box scene audit."""

import argparse
import hashlib
import json
import math
import os
import tempfile


ROLE_SCHEMA = "mcln-density-target-box-scene-disjoint-role-v1"
DECISION_SCHEMA = "mcln-density-target-box-scene-disjoint-decision-v1"
PROTECTED_E57_SHA256 = (
    "fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
)


def _load_json_with_sha(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("invalid audit receipt {}: {}".format(path, exc))
    if not isinstance(payload, dict):
        raise ValueError("audit receipt must be a JSON object")
    return payload, digest


def _integer(value, name, lower=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        raise ValueError("{} must be an integer >= {}".format(name, lower))
    return int(value)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be numeric".format(name))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    return value


def _validate_metrics(receipt, role):
    evaluation = receipt.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("{} evaluation is missing".format(role))
    metrics = evaluation.get("density_aware_target_box_scene_audit")
    if (
            not isinstance(metrics, dict)
            or metrics.get("schema")
            != "mcln-density-target-box-scene-metrics-v1"):
        raise ValueError("{} scene metrics are incompatible".format(role))
    if _integer(metrics.get("sample_count"), "sample_count", 1) != 6329:
        raise ValueError("{} holdout sample count drifted".format(role))
    if (
            metrics.get("sample_identity_count") != 6329
            or metrics.get("sample_identity_unique_count") != 6329):
        raise ValueError("{} holdout identity coverage drifted".format(role))
    identity = metrics.get("sample_identity_sha256")
    if (
            not isinstance(identity, str) or len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)):
        raise ValueError("{} holdout identity digest is invalid".format(role))
    slices = metrics.get("slices")
    if not isinstance(slices, dict) or set(slices) != {
        "overall", "active_sparse", "dense", "zero_point"
    }:
        raise ValueError("{} metric slices are incomplete".format(role))
    for slice_name, values in slices.items():
        if not isinstance(values, dict):
            raise ValueError("{} {} slice is invalid".format(role, slice_name))
        count = _integer(
            values.get("sample_count"),
            "{} {} sample_count".format(role, slice_name),
        )
        for prefix in ("selected", "top16", "matched"):
            hits025 = _integer(
                values.get(prefix + "_hits025"),
                "{} {} {}_hits025".format(role, slice_name, prefix),
            )
            hits050 = _integer(
                values.get(prefix + "_hits050"),
                "{} {} {}_hits050".format(role, slice_name, prefix),
            )
            if hits050 > hits025 or hits025 > count:
                raise ValueError("{} {} hits are not nested".format(
                    role, slice_name
                ))
        for name in (
                "target_point_count_mean",
                "selected_acc025", "selected_acc050",
                "top16_acc025", "top16_acc050",
                "matched_iou_mean", "matched_acc025", "matched_acc050",
                "matched_center_l1_mean", "matched_size_l1_mean"):
            _finite(
                values.get(name),
                "{} {} {}".format(role, slice_name, name),
            )
    if (
            slices["active_sparse"]["sample_count"]
            + slices["dense"]["sample_count"]
            + slices["zero_point"]["sample_count"]
            != 6329):
        raise ValueError("{} holdout slices do not partition rows".format(role))
    return metrics


def _validate_role(receipt, expected_role):
    if receipt.get("schema") != ROLE_SCHEMA:
        raise ValueError("{} receipt schema is incompatible".format(
            expected_role
        ))
    if receipt.get("role") != expected_role:
        raise ValueError("{} receipt role drifted".format(expected_role))
    if receipt.get("checkpoint_sha256") != PROTECTED_E57_SHA256:
        raise ValueError("{} checkpoint identity drifted".format(expected_role))
    if receipt.get("checkpoint_epoch") != 57:
        raise ValueError("{} checkpoint epoch drifted".format(expected_role))
    if receipt.get("generated_weights") != []:
        raise ValueError("{} generated a checkpoint".format(expected_role))
    if receipt.get("formal_validation_accessed") is not False:
        raise ValueError("{} accessed formal validation".format(expected_role))
    if receipt.get("audit_only") is not True:
        raise ValueError("{} is not marked audit-only".format(expected_role))
    if receipt.get("long_training_authorized") is not False:
        raise ValueError("{} incorrectly authorizes long training".format(
            expected_role
        ))
    expected_weight = 1.0 if expected_role == "method" else 0.0
    if _finite(
            receipt.get("density_aware_target_box_loss_weight"),
            "{} density weight".format(expected_role),
    ) != expected_weight:
        raise ValueError("{} density weight drifted".format(expected_role))
    expected_epoch = 57 if expected_role == "parent" else 58
    if receipt.get("epoch") != expected_epoch:
        raise ValueError("{} audit epoch drifted".format(expected_role))
    split = receipt.get("split")
    if not isinstance(split, dict) or any((
        split.get("fold") != 2,
        split.get("fit_scenes") != 408,
        split.get("holdout_scenes") != 103,
        split.get("fit_samples") != 26590,
        split.get("holdout_samples") != 6329,
        split.get("total_scenes") != 511,
        split.get("total_samples") != 32919,
    )):
        raise ValueError("{} split metadata drifted".format(expected_role))
    training = receipt.get("training")
    if expected_role == "parent":
        if training is not None:
            raise ValueError("parent role must not train")
    else:
        if not isinstance(training, dict):
            raise ValueError("{} training receipt is missing".format(
                expected_role
            ))
        expected_training = {
            "batch_count": 100,
            "optimizer_step_count": 100,
            "sample_count": 1600,
            "sample_identity_count": 1600,
            "sample_identity_unique_count": 1600,
        }
        for name, expected in expected_training.items():
            if training.get(name) != expected:
                raise ValueError("{} training {} drifted".format(
                    expected_role, name
                ))
        identity = training.get("sample_identity_sha256")
        if not isinstance(identity, str) or len(identity) != 64:
            raise ValueError("{} fit identity is invalid".format(expected_role))
    return split, training, _validate_metrics(receipt, expected_role)


def build_decision(parent, control, method, receipt_records,
                   provenance_record=None):
    parent_split, parent_training, parent_metrics = _validate_role(
        parent, "parent"
    )
    control_split, control_training, control_metrics = _validate_role(
        control, "control"
    )
    method_split, method_training, method_metrics = _validate_role(
        method, "method"
    )
    if not (parent_split == control_split == method_split):
        raise ValueError("role split metadata differs")
    if parent_training is not None:
        raise ValueError("parent training receipt must be absent")
    if (
            control_training["sample_identity_sha256"]
            != method_training["sample_identity_sha256"]):
        raise ValueError("control and method fit row identities differ")
    holdout_identities = {
        parent_metrics["sample_identity_sha256"],
        control_metrics["sample_identity_sha256"],
        method_metrics["sample_identity_sha256"],
    }
    if len(holdout_identities) != 1:
        raise ValueError("role holdout row identities differ")

    parent_overall = parent_metrics["slices"]["overall"]
    parent_sparse = parent_metrics["slices"]["active_sparse"]
    control_overall = control_metrics["slices"]["overall"]
    control_sparse = control_metrics["slices"]["active_sparse"]
    method_overall = method_metrics["slices"]["overall"]
    method_sparse = method_metrics["slices"]["active_sparse"]
    gates = {
        "active_sparse_top16_hits025_strict_gain_vs_control": (
            method_sparse["top16_hits025"]
            > control_sparse["top16_hits025"]
        ),
        "active_sparse_matched_iou_strict_gain_vs_control": (
            method_sparse["matched_iou_mean"]
            > control_sparse["matched_iou_mean"]
        ),
        "overall_selected025_non_degradation_vs_control": (
            method_overall["selected_hits025"]
            >= control_overall["selected_hits025"]
        ),
        "overall_selected050_non_degradation_vs_control": (
            method_overall["selected_hits050"]
            >= control_overall["selected_hits050"]
        ),
        "active_sparse_selected025_non_degradation_vs_control": (
            method_sparse["selected_hits025"]
            >= control_sparse["selected_hits025"]
        ),
        "not_jointly_worse_than_parent": not (
            method_overall["selected_hits025"]
            < parent_overall["selected_hits025"]
            and method_sparse["top16_hits025"]
            < parent_sparse["top16_hits025"]
        ),
        "fit_and_holdout_identity_exact": True,
        "zero_generated_weights_and_zero_formal_validation": True,
    }
    gate_passed = all(gates.values())
    decision = {
        "schema": DECISION_SCHEMA,
        "audit_only": True,
        "density_gate_passed": gate_passed,
        "gates": gates,
        "split": parent_split,
        "fit_sample_identity_sha256": control_training[
            "sample_identity_sha256"
        ],
        "holdout_sample_identity_sha256": next(iter(holdout_identities)),
        "receipts": receipt_records,
        "parent": parent_metrics,
        "control": control_metrics,
        "method": method_metrics,
        "next_allowed_step": (
            "independent_review_only" if gate_passed else "seal_method"
        ),
        "long_training_authorized": False,
    }
    if provenance_record is not None:
        if not isinstance(provenance_record, dict):
            raise ValueError("provenance record must be a mapping")
        decision["provenance"] = provenance_record
    return decision


def _publish_no_overwrite(payload, output_path):
    output_path = os.path.abspath(output_path)
    output_directory = os.path.dirname(output_path)
    if not os.path.isdir(output_directory):
        raise ValueError("decision output directory must already exist")
    if os.path.lexists(output_path):
        raise FileExistsError(output_path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".density-scene-decision-", suffix=".tmp",
        dir=output_directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output_path)
        directory_fd = os.open(output_directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payloads = {}
    records = {}
    for role in ("parent", "control", "method"):
        path = os.path.abspath(getattr(args, role))
        payload, digest = _load_json_with_sha(path)
        payloads[role] = payload
        records[role] = {"path": path, "sha256": digest}
    provenance_path = os.path.abspath(args.provenance)
    provenance, provenance_sha256 = _load_json_with_sha(provenance_path)
    if provenance.get("schema") != (
            "mcln-density-target-box-scene-provenance-v1"):
        raise ValueError("scene-audit provenance schema is incompatible")
    if provenance.get("audit_only") is not True:
        raise ValueError("scene-audit provenance is not audit-only")
    if provenance.get("long_training_authorized") is not False:
        raise ValueError("scene-audit provenance authorizes long training")
    decision = build_decision(
        payloads["parent"], payloads["control"], payloads["method"],
        records,
        {
            "path": provenance_path,
            "sha256": provenance_sha256,
            "payload": provenance,
        },
    )
    _publish_no_overwrite(decision, args.output)
    print(json.dumps({
        "density_gate_passed": decision["density_gate_passed"],
        "long_training_authorized": False,
        "output": os.path.abspath(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
