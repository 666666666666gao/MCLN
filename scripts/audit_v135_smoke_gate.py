#!/usr/bin/env python3
"""Build the fail-closed V135 scene-disjoint smoke gate."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import re
import stat
from pathlib import Path

import torch

from v133_receipt_utils import atomic_write_new_json


SCHEMA = "mcln-v135-formal-admission-v2"
CONTRACT_SCHEMA = "mcln-v135-relation-counterfactual-contract-v1"
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def evidence(path):
    path = Path(path).resolve()
    require(path.is_file(), "V135 evidence is missing: {}".format(path))
    return {"path": str(path), "sha256": sha256(path)}


def epoch_files(run_dir, prefix):
    paths = list(Path(run_dir).glob("{}_epoch_*.json".format(prefix)))
    return sorted(paths, key=lambda path: int(re.search(
        r"epoch_(\d+)\.json$", path.name
    ).group(1)))


def validate_metric(path):
    data = read_json(path)
    require(data.get("schema") == "mcln-retrain-metrics-v1",
            "unexpected V135 metric schema")
    require(data.get("sample_count") == 128,
            "V135 smoke metrics must contain 128 samples")
    return data


def validate_diagnostic(path):
    data = read_json(path)
    require(data.get("schema") == "mcln-source-choice-diagnostics-v1",
            "unexpected V135 diagnostic schema")
    require(data.get("sample_count") == 128,
            "V135 smoke diagnostics must contain 128 samples")
    for key in (
            "sacr_parent", "sacr_feasible_oracle_headroom",
            "sacr_parent_effects"):
        require(key in data, "V135 diagnostics are missing {}".format(key))
    return data


def validate_config(config, parent_checkpoint, run_dir):
    exact = {
        "batch_size": 8,
        "dataset": ["scanrefer"],
        "debug": True,
        "debug_train_holdout": True,
        "eval_use_selector_choice_scores": True,
        "expected_eval_sample_count": 128,
        "max_epoch": 2,
        "sacr_score_refiner_train_only": True,
        "sacr_score_use_parent_relative_abstention": False,
        "sacr_score_use_relation_counterfactual": True,
        "sacr_score_max_delta": 0.25,
        "sacr_score_promotion_margin": 0.01,
        "sacr_counterfactual_parent_top_k": 16,
        "sacr_counterfactual_target_tolerance": 0.05,
        "sacr_counterfactual_attribute_tolerance": 0.05,
        "sacr_counterfactual_geometry_threshold": 0.08,
        "sacr_counterfactual_iou_gap": 0.10,
        "sacr_counterfactual_correct_iou_threshold": 0.25,
        "sacr_counterfactual_pair_margin": 0.25,
        "sacr_counterfactual_max_negatives": 4,
        "sacr_counterfactual_relation_scale": 4.0,
        "sacr_counterfactual_deployment_threshold": 0.05,
        "test_dataset": "scanrefer",
        "use_sacr_score_refiner": True,
        "use_source_choice_selector": True,
        "val_freq": 1,
    }
    mismatches = [
        "{} expected {!r}, got {!r}".format(key, expected, config.get(key))
        for key, expected in exact.items() if config.get(key) != expected
    ]
    if Path(config.get("checkpoint_path", "")).resolve() != parent_checkpoint:
        mismatches.append("checkpoint_path differs from bound parent")
    if Path(config.get("log_dir", "")).resolve() != run_dir:
        mismatches.append("log_dir differs from audited run")
    require(not mismatches, "V135 smoke config differs: " + "; ".join(mismatches))


def parse_training_stats(log_text):
    rows = []
    for line in log_text.splitlines():
        if "[source_moe]" not in line:
            continue
        values = {
            key: float(value)
            for key, value in re.findall(
                r"(sacr_score_[A-Za-z0-9_]+)\s+(-?[0-9.eE+]+)", line
            )
        }
        if "sacr_score_hard_negative_row_ratio" in values:
            rows.append(values)
    require(rows, "V135 hard-negative training statistics are missing")
    for row in rows:
        for key, value in row.items():
            require(math.isfinite(value), "V135 {} is non-finite".format(key))
    require(max(row["sacr_score_hard_negative_row_ratio"] for row in rows) > 0,
            "V135 did not mine any hard-negative row")
    require(max(row.get("sacr_score_selected_negative_count_mean", 0.0)
                for row in rows) > 0,
            "V135 did not select any counterfactual negative")
    require(max(row.get("sacr_score_parent_drift_abs_max", 0.0)
                for row in rows) == 0.0,
            "V135 changed a parent score")
    require(max(row.get("sacr_score_residual_abs_max", 0.0)
                for row in rows) <= 0.25,
            "V135 exceeded its residual bound")
    return {
        "row_count": len(rows),
        "hard_negative_row_ratio_max": max(
            row["sacr_score_hard_negative_row_ratio"] for row in rows
        ),
        "selected_negative_count_mean_max": max(
            row.get("sacr_score_selected_negative_count_mean", 0.0)
            for row in rows
        ),
    }


def validate_checkpoint(run_dir):
    path = run_dir / "ckpt_epoch_last.pth"
    require(path.is_file(), "V135 smoke checkpoint is missing")
    checkpoint = torch.load(str(path), map_location="cpu")
    config = checkpoint.get("config")
    flag = (
        config.get("sacr_score_use_relation_counterfactual", False)
        if isinstance(config, dict)
        else getattr(config, "sacr_score_use_relation_counterfactual", False)
    )
    require(flag is True, "V135 checkpoint lost its deployment flag")
    return {"path": str(path), "sha256": sha256(path)}


def build(args):
    repo_root = Path(args.repo_root).resolve()
    require(repo_root == SCRIPT_REPO_ROOT,
            "--repo-root must contain the executing V135 audit script")
    run_dir = Path(args.run_dir).resolve()
    parent = Path(args.parent_checkpoint).resolve()
    contract_path = Path(args.contract_receipt).resolve()
    baseline_path = Path(args.baseline_receipt).resolve()
    launch_log = Path(args.launch_log).resolve()
    for path in (run_dir, parent, contract_path, baseline_path, launch_log):
        require(path.exists(), "V135 smoke input is missing: {}".format(path))
    require(stat.S_IMODE(contract_path.stat().st_mode) == 0o444,
            "V135 contract receipt must be read-only")
    contract = read_json(contract_path)
    require(contract.get("schema") == CONTRACT_SCHEMA
            and contract.get("verdict") == "pass",
            "V135 contract receipt did not pass")
    for relative, expected in contract["source_sha256"].items():
        require(sha256(repo_root / relative) == expected,
                "V135 source changed: {}".format(relative))

    metrics = [validate_metric(path) for path in epoch_files(
        run_dir, "eval_metrics"
    )]
    diagnostics = [validate_diagnostic(path) for path in epoch_files(
        run_dir, "source_choice_diagnostics"
    )]
    require(len(metrics) == 2 and len(diagnostics) == 2,
            "V135 smoke must contain two evaluated epochs")
    baseline = validate_metric(baseline_path)
    validate_config(read_json(run_dir / "config.json"), parent, run_dir)
    log_text = (run_dir / "log.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    launch_text = launch_log.read_text(encoding="utf-8", errors="replace")
    require("finished v135_relation_counterfactual_smoke" in launch_text,
            "V135 smoke launch did not finish")
    require("overlap=0" in log_text + "\n" + launch_text,
            "V135 scene-disjoint evidence is missing")
    training_stats = parse_training_stats(log_text)

    audits = []
    safe_epochs = []
    efficacious_epochs = []
    for epoch, (metric, diagnostic) in enumerate(
            zip(metrics, diagnostics), start=1):
        parent_row = diagnostic["sacr_parent"]
        learned = metric["position"]["learned_selector"]
        effects = diagnostic["sacr_parent_effects"]
        effect_non_degradation_pass = True
        effect_improvement_pass = True
        for suffix, hit_key in (("025", "hits025"), ("050", "hits050")):
            values = effects[suffix]
            require(
                learned[hit_key] == parent_row[hit_key]
                + values["sacr_parent_fix"] - values["sacr_parent_break"],
                "V135 fix/break accounting differs from deployed REC",
            )
            effect_non_degradation_pass = (
                effect_non_degradation_pass
                and values["sacr_parent_fix"] >= values["sacr_parent_break"]
            )
            effect_improvement_pass = effect_improvement_pass and (
                values["sacr_parent_fix"] > values["sacr_parent_break"]
            )
        headroom = diagnostic["sacr_feasible_oracle_headroom"]
        oracle_efficacy_testable = (
            headroom["hits025"] >= 1 and headroom["hits050"] >= 1
        )
        mask = metric["mask"]
        baseline_mask = baseline["mask"]
        mask_pass = (
            mask["hits025"] >= baseline_mask["hits025"] - 1
            and mask["hits050"] >= baseline_mask["hits050"] - 1
            and mask["miou"] >= baseline_mask["miou"] - 0.005
        )
        safety_pass = effect_non_degradation_pass and mask_pass
        efficacy_pass = (
            safety_pass
            and oracle_efficacy_testable
            and effect_improvement_pass
        )
        if safety_pass:
            safe_epochs.append(epoch)
        if efficacy_pass:
            efficacious_epochs.append(epoch)
        audits.append({
            "epoch": epoch,
            "parent": parent_row,
            "learned": learned,
            "effects": effects,
            "proposal_oracle_headroom": headroom,
            "effect_non_degradation_pass": effect_non_degradation_pass,
            "effect_improvement_pass": effect_improvement_pass,
            "oracle_efficacy_testable": oracle_efficacy_testable,
            "mask_pass": mask_pass,
            "safety_pass": safety_pass,
            "efficacy_pass": efficacy_pass,
        })
    require(safe_epochs, "no V135 smoke epoch passed the safety gate")
    rec025_testable = any(
        audit["proposal_oracle_headroom"]["hits025"] >= 1
        for audit in audits
    )
    efficacy_status = (
        "demonstrated" if efficacious_epochs
        else "inconclusive_no_rec025_oracle_headroom"
        if not rec025_testable
        else "not_demonstrated_in_smoke"
    )
    config_path = run_dir / "config.json"
    log_path = run_dir / "log.txt"
    checkpoint = validate_checkpoint(run_dir)
    payload = {
        "schema": SCHEMA,
        "verdict": "formal_admission_pass",
        "scope": "safety_and_execution_only",
        "efficacy_status": efficacy_status,
        "rec025_efficacy_testable": rec025_testable,
        "safe_epochs": safe_epochs,
        "efficacious_epochs": efficacious_epochs,
        "epochs": audits,
        "training_stats": training_stats,
        "checkpoint": checkpoint,
        "checkpoint_retention": "sha_record_only_may_be_removed_after_gate",
        "parent_checkpoint": {"path": str(parent), "sha256": sha256(parent)},
        "contract_receipt": {"path": str(contract_path),
                             "sha256": sha256(contract_path)},
        "run_dir": str(run_dir),
        "evidence": {
            "baseline": evidence(baseline_path),
            "config": evidence(config_path),
            "log": evidence(log_path),
            "launch_log": evidence(launch_log),
            "metrics": [evidence(path) for path in epoch_files(
                run_dir, "eval_metrics"
            )],
            "diagnostics": [evidence(path) for path in epoch_files(
                run_dir, "source_choice_diagnostics"
            )],
        },
    }
    atomic_write_new_json(payload, args.output)
    print(json.dumps(payload, sort_keys=True))


def verify(args):
    repo_root = Path(args.repo_root).resolve()
    require(repo_root == SCRIPT_REPO_ROOT,
            "--repo-root must contain the executing V135 audit script")
    gate_path = Path(args.gate).resolve()
    parent = Path(args.parent_checkpoint).resolve()
    contract_path = Path(args.contract_receipt).resolve()
    for path in (gate_path, parent, contract_path):
        require(path.is_file(), "V135 formal input is missing: {}".format(path))
    require(stat.S_IMODE(gate_path.stat().st_mode) == 0o444,
            "V135 formal admission must be read-only")
    gate = read_json(gate_path)
    require(gate.get("schema") == SCHEMA,
            "unexpected V135 formal-admission schema")
    require(gate.get("verdict") == "formal_admission_pass",
            "V135 smoke did not grant formal admission")
    require(gate.get("scope") == "safety_and_execution_only",
            "V135 formal-admission scope changed")
    require(gate["parent_checkpoint"]["sha256"] == sha256(parent),
            "V135 formal parent differs from smoke parent")
    require(gate["contract_receipt"]["sha256"] == sha256(contract_path),
            "V135 formal contract differs from smoke contract")
    contract = read_json(contract_path)
    require(contract.get("schema") == CONTRACT_SCHEMA
            and contract.get("verdict") == "pass",
            "V135 formal contract did not pass")
    for relative, expected in contract["source_sha256"].items():
        require(sha256(repo_root / relative) == expected,
                "V135 source changed after smoke: {}".format(relative))
    for record in gate["evidence"].values():
        records = record if isinstance(record, list) else [record]
        for item in records:
            require(sha256(item["path"]) == item["sha256"],
                    "V135 smoke evidence changed: {}".format(item["path"]))
    checkpoint_path = Path(gate["checkpoint"]["path"])
    if checkpoint_path.exists():
        require(sha256(checkpoint_path) == gate["checkpoint"]["sha256"],
                "V135 smoke checkpoint changed")
    else:
        require(
            gate.get("checkpoint_retention")
            == "sha_record_only_may_be_removed_after_gate",
            "V135 smoke checkpoint is unexpectedly missing",
        )
    print(json.dumps({
        "schema": SCHEMA,
        "verdict": "formal_admission_verified",
        "efficacy_status": gate["efficacy_status"],
        "gate_sha256": sha256(gate_path),
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "verify"), default="build")
    parser.add_argument("--output")
    parser.add_argument("--gate")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--contract-receipt", required=True)
    parser.add_argument("--baseline-receipt")
    parser.add_argument("--launch-log")
    args = parser.parse_args()
    if args.mode == "build":
        for name in ("output", "run_dir", "baseline_receipt", "launch_log"):
            require(getattr(args, name) is not None,
                    "--{} is required in build mode".format(
                        name.replace("_", "-")
                    ))
        build(args)
    else:
        require(args.gate is not None, "--gate is required in verify mode")
        verify(args)


if __name__ == "__main__":
    main()
