#!/usr/bin/env python3
"""Build a fail-closed summary for parallel joint-query smoke runs."""

import argparse
import json
import math
import os
import re
import tempfile


SCHEMAS = {
    "v41": "mcln-v41-smoke-panel-v1",
    "v42": "mcln-v42-smoke-panel-v1",
    "v43": "mcln-v43-smoke-panel-v1",
    "v43_selector": "mcln-v43-selector-smoke-panel-v1",
    "v46": "mcln-v46-smoke-panel-v1",
    "v48": "mcln-v48-smoke-panel-v1",
    "v49": "mcln-v49-smoke-panel-v1",
    "v50_sacr": "mcln-v50-sacr-smoke-panel-v1",
}
BASE_DIAGNOSTIC_NAMES = (
    "joint_query_quality_residual_abs_mean",
    "joint_query_quality_residual_abs_max",
    "joint_query_quality_residual_query_std",
    "joint_query_quality_switch_ratio",
)
MASK_CALIBRATION_DIAGNOSTIC_NAMES = (
    "joint_query_quality_mask_alpha_residual_abs_mean",
    "joint_query_quality_mask_logit_bias_abs_mean",
    "joint_query_quality_mask_logit_bias_abs_max",
    "joint_query_quality_mask_weight_std_mean",
)
SOURCE_MASK_EVIDENCE_DIAGNOSTIC_NAMES = (
    "joint_query_quality_source_mask_evidence_query_std",
    "joint_query_quality_source_mask_disagreement_mean",
)
GATE_EVIDENCE_DIAGNOSTIC_NAMES = (
    "joint_query_quality_gate_evidence_query_std",
    "joint_query_quality_gate_candidate_ratio",
)
CANDIDATE_MASK_DIAGNOSTIC_NAMES = (
    "joint_query_quality_candidate_mask_query_ratio",
)
CANDIDATE_LOVASZ_DIAGNOSTIC_NAMES = (
    "joint_query_quality_candidate_lovasz_loss",
)
SPATIAL_MASK_DIAGNOSTIC_NAMES = (
    "joint_query_quality_mask_spatial_residual_abs_mean",
    "joint_query_quality_mask_spatial_residual_abs_max",
    "joint_query_quality_mask_spatial_superpoint_std_mean",
    "joint_query_quality_mask_spatial_query_std_mean",
)
SOURCE_MIX_DIAGNOSTIC_NAMES = (
    "joint_query_quality_source_mix_alignment_loss",
    "joint_query_quality_source_mix_alignment_target_top1_acc",
    "joint_query_quality_source_mix_alignment_target_effective_count_mean",
    "joint_query_quality_source_mix_residual_abs_mean",
    "joint_query_quality_source_mix_router_residual_abs_mean",
    "joint_query_quality_source_mix_weight_query_std_mean",
    "joint_query_quality_source_mix_effective_count_mean",
)
SACR_DIAGNOSTIC_NAMES = (
    "sacr_residual_scale_value",
    "sacr_valid_ratio",
    "sacr_relation_active_ratio",
)
DIAGNOSTIC_NAMES = (
    BASE_DIAGNOSTIC_NAMES + MASK_CALIBRATION_DIAGNOSTIC_NAMES
    + SOURCE_MASK_EVIDENCE_DIAGNOSTIC_NAMES
    + GATE_EVIDENCE_DIAGNOSTIC_NAMES
    + SPATIAL_MASK_DIAGNOSTIC_NAMES
    + SOURCE_MIX_DIAGNOSTIC_NAMES
    + SACR_DIAGNOSTIC_NAMES
    + CANDIDATE_MASK_DIAGNOSTIC_NAMES
    + CANDIDATE_LOVASZ_DIAGNOSTIC_NAMES
)
DIAGNOSTIC_PATTERN = re.compile(
    r"({})\s+([0-9eE+.-]+)".format(
        "|".join(re.escape(name) for name in DIAGNOSTIC_NAMES)
    )
)


def _load_json(path, label):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError("{} is unreadable: {}".format(label, error))
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(label))
    return value


def _read_diagnostics(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as error:
        raise ValueError("launcher log is unreadable: {}".format(error))
    diagnostics = {}
    for name, value in DIAGNOSTIC_PATTERN.findall(text):
        diagnostics[name] = float(value)
    return diagnostics


def summarize_record(name, run_dir, log_path, epoch, expected_sample_count,
                     profile="v41", require_candidate_mask=False,
                     require_candidate_lovasz=False):
    if not isinstance(name, str) or not name:
        raise ValueError("variant name must be non-empty")
    if profile not in SCHEMAS:
        raise ValueError("unknown joint-query smoke profile")
    receipt = _load_json(
        os.path.join(run_dir, "eval_metrics_epoch_{}.json".format(epoch)),
        "evaluation receipt",
    )
    audit = _load_json(
        os.path.join(run_dir, "audit_{}.json".format(profile)),
        "{} checkpoint audit".format(profile.upper()),
    )
    diagnostics = _read_diagnostics(log_path)
    required_diagnostics = BASE_DIAGNOSTIC_NAMES
    if profile in (
            "v42", "v43", "v43_selector", "v46", "v48", "v49",
            "v50_sacr"):
        required_diagnostics += MASK_CALIBRATION_DIAGNOSTIC_NAMES
    if profile in (
            "v43", "v43_selector", "v46", "v48", "v49", "v50_sacr"):
        required_diagnostics += SOURCE_MASK_EVIDENCE_DIAGNOSTIC_NAMES
    if profile == "v46":
        required_diagnostics += GATE_EVIDENCE_DIAGNOSTIC_NAMES
    if profile in ("v48", "v49", "v50_sacr"):
        required_diagnostics += SPATIAL_MASK_DIAGNOSTIC_NAMES
    if profile in ("v49", "v50_sacr"):
        required_diagnostics += SOURCE_MIX_DIAGNOSTIC_NAMES
    if profile == "v50_sacr":
        required_diagnostics += SACR_DIAGNOSTIC_NAMES
    if require_candidate_mask:
        required_diagnostics += CANDIDATE_MASK_DIAGNOSTIC_NAMES
    if require_candidate_lovasz:
        required_diagnostics += CANDIDATE_LOVASZ_DIAGNOSTIC_NAMES
    missing = [
        name for name in required_diagnostics if name not in diagnostics
    ]
    finite = all(
        math.isfinite(diagnostics.get(name, float("nan")))
        for name in required_diagnostics
    )
    calibration_nonzero = profile == "v41" or all(
        diagnostics.get(name, 0.0) > 0.0
        for name in (
            "joint_query_quality_mask_alpha_residual_abs_mean",
            "joint_query_quality_mask_logit_bias_abs_mean",
            "joint_query_quality_mask_weight_std_mean",
        )
    )
    source_evidence_nonzero = profile not in (
        "v43", "v43_selector", "v46", "v48", "v49", "v50_sacr"
    ) or all(
        diagnostics.get(name, 0.0) > 0.0
        for name in SOURCE_MASK_EVIDENCE_DIAGNOSTIC_NAMES
    )
    gate_evidence_nonzero = profile != "v46" or all(
        diagnostics.get(name, 0.0) > 0.0
        for name in GATE_EVIDENCE_DIAGNOSTIC_NAMES
    )
    spatial_mask_nonzero = profile not in (
        "v48", "v49", "v50_sacr"
    ) or all(
        diagnostics.get(name, 0.0) > 0.0
        for name in SPATIAL_MASK_DIAGNOSTIC_NAMES
    )
    source_mix_nonzero = profile not in ("v49", "v50_sacr") or (
        diagnostics.get(
            "joint_query_quality_source_mix_alignment_loss", 0.0
        ) > 0.0
        and diagnostics.get(
            "joint_query_quality_source_mix_alignment_target_effective_count_mean",
            0.0,
        ) >= 1.0
        and diagnostics.get(
            "joint_query_quality_source_mix_residual_abs_mean", 0.0
        ) > 0.0
        and diagnostics.get(
            "joint_query_quality_source_mix_router_residual_abs_mean", 0.0
        ) > 0.0
        and diagnostics.get(
            "joint_query_quality_source_mix_weight_query_std_mean", 0.0
        ) > 0.0
        and diagnostics.get(
            "joint_query_quality_source_mix_effective_count_mean", 0.0
        ) >= 1.0
    )
    sacr_nonzero = profile != "v50_sacr" or (
        abs(diagnostics.get("sacr_residual_scale_value", 0.0)) > 0.0
        and diagnostics.get("sacr_valid_ratio", 0.0) > 0.0
        and diagnostics.get("sacr_relation_active_ratio", 0.0) > 0.0
    )
    candidate_mask_nonzero = not require_candidate_mask or all(
        diagnostics.get(name, 0.0) > 0.0
        for name in CANDIDATE_MASK_DIAGNOSTIC_NAMES
    )
    candidate_lovasz_nonzero = not require_candidate_lovasz or all(
        diagnostics.get(name, 0.0) > 0.0
        for name in CANDIDATE_LOVASZ_DIAGNOSTIC_NAMES
    )
    passed = (
        receipt.get("schema") == "mcln-retrain-metrics-v1"
        and receipt.get("sample_count") == expected_sample_count
        and audit.get("schema") == "mcln-source-moe-checkpoint-audit-v1"
        and audit.get("profile") == profile
        and audit.get("pass") is True
        and not missing
        and finite
        and diagnostics["joint_query_quality_residual_abs_mean"] > 0.0
        and diagnostics["joint_query_quality_residual_query_std"] > 0.0
        and calibration_nonzero
        and source_evidence_nonzero
        and gate_evidence_nonzero
        and spatial_mask_nonzero
        and source_mix_nonzero
        and sacr_nonzero
        and candidate_mask_nonzero
        and candidate_lovasz_nonzero
    )
    return {
        "variant": name,
        "run_dir": run_dir,
        "receipt": receipt,
        "audit": audit,
        "diagnostics": diagnostics,
        "missing_diagnostics": missing,
        "candidate_mask_required": bool(require_candidate_mask),
        "candidate_lovasz_required": bool(require_candidate_lovasz),
        "pass": passed,
    }


def build_summary(records, epoch, expected_sample_count, profile="v41"):
    if not records:
        raise ValueError("at least one smoke record is required")
    if profile not in SCHEMAS:
        raise ValueError("unknown joint-query smoke profile")
    return {
        "schema": SCHEMAS[profile],
        "profile": profile,
        "epoch": epoch,
        "sample_count": expected_sample_count,
        "records": records,
        "pass": all(record.get("pass") is True for record in records),
    }


def atomic_write_json(path, value):
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".v41-panel-", suffix=".json", dir=output_dir
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--expected-sample-count", type=int, required=True)
    parser.add_argument("--profile", choices=tuple(SCHEMAS), default="v41")
    parser.add_argument("--require-candidate-mask", action="store_true")
    parser.add_argument(
        "--require-lovasz-variant", action="append", default=[]
    )
    parser.add_argument(
        "--record", action="append", nargs=3,
        metavar=("NAME", "RUN_DIR", "LOG_PATH"), required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epoch <= 0 or args.expected_sample_count <= 0:
        raise SystemExit("epoch and expected sample count must be positive")
    record_names = [record[0] for record in args.record]
    unknown_lovasz = sorted(
        set(args.require_lovasz_variant) - set(record_names)
    )
    if unknown_lovasz:
        raise SystemExit(
            "unknown Lovasz variants: {}".format(
                ", ".join(unknown_lovasz)
            )
        )
    records = [
        summarize_record(
            name, run_dir, log_path,
            args.epoch, args.expected_sample_count, args.profile,
            args.require_candidate_mask,
            name in set(args.require_lovasz_variant),
        )
        for name, run_dir, log_path in args.record
    ]
    summary = build_summary(
        records, args.epoch, args.expected_sample_count, args.profile
    )
    atomic_write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit("one or more {} smoke gates failed".format(
            args.profile.upper()
        ))


if __name__ == "__main__":
    main()
