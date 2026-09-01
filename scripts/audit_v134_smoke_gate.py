#!/usr/bin/env python3
"""Build and verify the fail-closed V134 scene-holdout smoke gate."""

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


SCHEMA = "mcln-v134-smoke-gate-v1"
CONTRACT_SCHEMA = "mcln-v134-parent-relative-contract-v1"
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def require_read_only(path, label):
    mode = stat.S_IMODE(Path(path).stat().st_mode)
    require(mode == 0o444, "{} must have mode 0444".format(label))


def resolve_repo_root(value):
    repo_root = Path(value).resolve()
    require(repo_root == SCRIPT_REPO_ROOT,
            "--repo-root must contain the executing V134 audit script")
    return repo_root


def epoch_files(run_dir, prefix):
    files = list(Path(run_dir).glob("{}_epoch_*.json".format(prefix)))

    def epoch(path):
        match = re.search(r"epoch_(\d+)\.json$", path.name)
        require(match is not None, "invalid epoch receipt name")
        return int(match.group(1))

    return sorted(files, key=epoch)


def validate_metrics(path):
    data = read_json(path)
    require(data.get("schema") == "mcln-retrain-metrics-v1",
            "unexpected V134 metric schema")
    require(data.get("sample_count") == 128,
            "V134 smoke metric receipt must contain 128 samples")
    position = data.get("position", {})
    for source in ("fixed_default", "learned_selector"):
        for key in ("hits025", "hits050"):
            require(isinstance(position.get(source, {}).get(key), int),
                    "V134 smoke REC hits are missing")
    mask = data.get("mask", {})
    for key in ("hits025", "hits050"):
        require(isinstance(mask.get(key), int),
                "V134 smoke mask hits are missing")
    require(math.isfinite(float(mask.get("miou"))),
            "V134 smoke mask mIoU is missing/non-finite")
    return data


def validate_diagnostics(path):
    data = read_json(path)
    require(data.get("schema") == "mcln-source-choice-diagnostics-v1",
            "unexpected V134 source-choice diagnostic schema")
    require(data.get("sample_count") == 128,
            "V134 source-choice diagnostics must contain 128 samples")
    for key in (
            "sacr_parent", "sacr_feasible_oracle",
            "sacr_feasible_oracle_headroom", "sacr_parent_effects"):
        require(key in data, "V134 diagnostics are missing {}".format(key))
    return data


def validate_config(config, parent_checkpoint, run_dir):
    exact = {
        "batch_size": 8,
        "checkpoint_metric_retention": True,
        "dataset": ["scanrefer"],
        "debug": True,
        "debug_train_holdout": True,
        "eval": False,
        "eval_use_selector_choice_scores": True,
        "expected_eval_sample_count": 128,
        "max_epoch": 2,
        "model": "MCLN",
        "sacr_score_dense_weight": 0.25,
        "sacr_score_gate_weight": 0.05,
        "sacr_score_mask_tolerance": 0.02,
        "sacr_score_mask_weight": 0.25,
        "sacr_score_max_delta": 0.25,
        "sacr_score_min_box_advantage": 0.03,
        "sacr_score_parent_gate_hidden_dim": 32,
        "sacr_score_preserve_weight": 1.0,
        "sacr_score_promotion_margin": 0.01,
        "sacr_score_raw_margin": 0.1,
        "sacr_score_refiner_loss_weight": 1.0,
        "sacr_score_refiner_lr": 0.0003,
        "sacr_score_refiner_train_only": True,
        "sacr_score_saturation_weight": 0.05,
        "sacr_score_temperature": 0.1,
        "sacr_score_use_parent_relative_abstention": True,
        "test_dataset": "scanrefer",
        "use_sacr_score_refiner": True,
        "use_source_choice_selector": True,
        "val_freq": 1,
    }
    mismatches = []
    for key, expected in exact.items():
        if config.get(key) != expected:
            mismatches.append(
                "{} expected {!r}, got {!r}".format(
                    key, expected, config.get(key)
                )
            )
    if Path(config.get("checkpoint_path", "")).resolve() != parent_checkpoint:
        mismatches.append("checkpoint_path differs from the bound parent")
    if Path(config.get("log_dir", "")).resolve() != run_dir:
        mismatches.append("log_dir differs from the audited run")
    require(not mismatches, "V134 smoke config differs: " + "; ".join(mismatches))


def parse_contract_stats(log_text):
    rows = []
    for line in log_text.splitlines():
        if "[source_moe]" not in line:
            continue
        values = {
            key: float(value)
            for key, value in re.findall(
                r"(sacr_score_[A-Za-z0-9_]+)\s+(-?[0-9.]+)", line
            )
        }
        if "sacr_score_parent_drift_abs_max" in values:
            rows.append(values)
    require(rows, "V134 smoke contract statistics are missing")
    for row in rows:
        require(row["sacr_score_parent_drift_abs_max"] == 0.0,
                "V134 changed a parent score")
        for key in (
                "sacr_score_residual_saturation_ratio",
                "sacr_score_residual_abs_max",
                "sacr_score_sample_gate_mean"):
            require(key in row and math.isfinite(row[key]),
                    "V134 {} is missing/non-finite".format(key))
        require(row["sacr_score_residual_saturation_ratio"] <= 0.05,
                "V134 residual saturation ratio exceeds 5%")
        require(row["sacr_score_residual_abs_max"] < 0.25,
                "V134 residual reached the hard bound")
    return {
        "row_count": len(rows),
        "parent_drift_abs_max": max(
            row["sacr_score_parent_drift_abs_max"] for row in rows
        ),
        "residual_saturation_ratio_max": max(
            row["sacr_score_residual_saturation_ratio"] for row in rows
        ),
        "residual_abs_max": max(
            row["sacr_score_residual_abs_max"] for row in rows
        ),
        "sample_gate_mean_max": max(
            row["sacr_score_sample_gate_mean"] for row in rows
        ),
    }


def validate_checkpoint_state(run_dir):
    checkpoint_path = run_dir / "ckpt_epoch_last.pth"
    require(checkpoint_path.is_file(), "V134 smoke checkpoint is missing")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model")
    require(isinstance(state, dict), "V134 smoke checkpoint model state is invalid")

    def canonical(name):
        return name[7:] if name.startswith("module.") else name

    names = {canonical(name) for name in state}
    require("sacr_score_gate" not in names,
            "V134 smoke checkpoint retained the global scalar gate")
    require(any(
        name.startswith("sacr_parent_relative_gate.") for name in names
    ), "V134 smoke checkpoint has no per-sample parent gate")
    config = checkpoint.get("config")
    flag = (
        config.get("sacr_score_use_parent_relative_abstention", False)
        if isinstance(config, dict)
        else getattr(config, "sacr_score_use_parent_relative_abstention", False)
    )
    require(flag is True, "V134 smoke checkpoint config lost its deployment flag")
    return {
        "path": str(checkpoint_path),
        "sha256": sha256(checkpoint_path),
        "global_scalar_present": False,
        "parent_gate_present": True,
    }


def build(args):
    repo_root = resolve_repo_root(args.repo_root)
    run_dir = Path(args.run_dir).resolve()
    parent = Path(args.parent_checkpoint).resolve()
    contract_path = Path(args.contract_receipt).resolve()
    baseline_path = Path(args.baseline_receipt).resolve()
    launch_log = Path(args.launch_log).resolve()
    for path, label in (
            (run_dir, "run directory"), (parent, "parent checkpoint"),
            (contract_path, "contract receipt"),
            (baseline_path, "baseline receipt"),
            (launch_log, "launch log")):
        require(path.exists(), "V134 {} is missing".format(label))
    require_read_only(contract_path, "V134 contract receipt")
    contract = read_json(contract_path)
    require(
        contract.get("schema") == CONTRACT_SCHEMA
        and contract.get("verdict") == "pass",
        "V134 contract receipt did not pass",
    )
    source_hashes = contract.get("source_sha256", {})
    for relative, expected in source_hashes.items():
        require(sha256(repo_root / relative) == expected,
                "V134 source changed after contract audit: {}".format(relative))
    metrics_paths = epoch_files(run_dir, "eval_metrics")
    diagnostics_paths = epoch_files(run_dir, "source_choice_diagnostics")
    require(len(metrics_paths) == 2 and len(diagnostics_paths) == 2,
            "V134 smoke must contain two metric and diagnostic receipts")
    metrics = [validate_metrics(path) for path in metrics_paths]
    diagnostics = [validate_diagnostics(path) for path in diagnostics_paths]
    baseline = validate_metrics(baseline_path)
    config_path = run_dir / "config.json"
    log_path = run_dir / "log.txt"
    require(config_path.is_file() and log_path.is_file(),
            "V134 smoke config/log is missing")
    validate_config(read_json(config_path), parent, run_dir)
    launch_text = launch_log.read_text(encoding="utf-8", errors="replace")
    require("finished v134_parent_relative_smoke" in launch_text,
            "V134 smoke launch did not finish")
    require(
        "Debug train holdout: train=128 examples/128 scenes; "
        "holdout=128 examples/120 scenes; overlap=0" in (
            log_path.read_text(encoding="utf-8", errors="replace")
            + "\n" + launch_text
        ),
        "V134 smoke scene-disjoint evidence is missing",
    )
    contract_stats = parse_contract_stats(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    epoch_audits = []
    passing_epochs = []
    for epoch_index, (metric, diagnostic) in enumerate(
            zip(metrics, diagnostics), start=1):
        parent = diagnostic["sacr_parent"]
        learned = metric["position"]["learned_selector"]
        require(
            parent["hits025"]
            == metric["position"]["fixed_default"]["hits025"]
            and parent["hits050"]
            == metric["position"]["fixed_default"]["hits050"],
            "V134 smoke parent differs from the frozen default source",
        )
        effects = diagnostic["sacr_parent_effects"]
        effect_pass = True
        for suffix, hit_key in (("025", "hits025"), ("050", "hits050")):
            row = effects[suffix]
            require(
                learned[hit_key]
                == parent[hit_key]
                + row["sacr_parent_fix"]
                - row["sacr_parent_break"],
                "V134 fix/break accounting does not match deployed REC",
            )
            effect_pass = effect_pass and (
                row["sacr_parent_fix"] > row["sacr_parent_break"]
            )
        headroom = diagnostic["sacr_feasible_oracle_headroom"]
        oracle_pass = headroom["hits025"] >= 1 and headroom["hits050"] >= 1
        mask = metric["mask"]
        baseline_mask = baseline["mask"]
        mask_pass = (
            mask["hits025"] >= baseline_mask["hits025"] - 1
            and mask["hits050"] >= baseline_mask["hits050"] - 1
            and mask["miou"] >= baseline_mask["miou"] - 0.005
        )
        passed = effect_pass and oracle_pass and mask_pass
        if passed:
            passing_epochs.append(epoch_index)
        epoch_audits.append({
            "epoch": epoch_index,
            "parent": parent,
            "learned": learned,
            "effects": effects,
            "feasible_oracle_headroom": headroom,
            "mask": mask,
            "effect_pass": effect_pass,
            "oracle_pass": oracle_pass,
            "mask_pass": mask_pass,
            "passed": passed,
        })
    require(passing_epochs,
            "no V134 smoke epoch passed fix/break, oracle, and mask gates")
    checkpoint = validate_checkpoint_state(run_dir)
    evidence_paths = (
        tuple(metrics_paths) + tuple(diagnostics_paths)
        + (baseline_path, contract_path, config_path, log_path, launch_log)
    )
    payload = {
        "schema": SCHEMA,
        "verdict": "pass",
        "parent_checkpoint": str(Path(args.parent_checkpoint).resolve()),
        "parent_sha256": sha256(args.parent_checkpoint),
        "contract_receipt": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "baseline_receipt": str(baseline_path),
        "baseline_mask": baseline["mask"],
        "run_dir": str(run_dir),
        "launch_log": str(launch_log),
        "contract_stats": contract_stats,
        "checkpoint": checkpoint,
        "passing_epochs": passing_epochs,
        "epochs": epoch_audits,
        "execution_source_sha256": {
            "scripts/audit_v134_smoke_gate.py": sha256(
                repo_root / "scripts/audit_v134_smoke_gate.py"
            ),
            "scripts/run_v134_sacr_parent_relative.sh": sha256(
                repo_root / "scripts/run_v134_sacr_parent_relative.sh"
            ),
        },
        "evidence_sha256": {
            str(path): sha256(path) for path in evidence_paths
        },
    }
    output = atomic_write_new_json(payload, args.output)
    print(json.dumps({
        "receipt": str(output),
        "sha256": sha256(output),
        "verdict": "pass",
        "passing_epochs": passing_epochs,
    }, sort_keys=True))


def verify(args):
    repo_root = resolve_repo_root(args.repo_root)
    receipt_path = Path(args.receipt).resolve()
    require_read_only(receipt_path, "V134 smoke gate receipt")
    receipt = read_json(receipt_path)
    require(
        receipt.get("schema") == SCHEMA
        and receipt.get("verdict") == "pass",
        "V134 smoke gate did not pass",
    )
    parent = Path(args.parent_checkpoint).resolve()
    contract_path = Path(args.contract_receipt).resolve()
    require(receipt.get("parent_checkpoint") == str(parent),
            "V134 smoke gate parent path changed")
    require(receipt.get("parent_sha256") == sha256(parent),
            "V134 smoke gate parent hash changed")
    require(receipt.get("contract_receipt") == str(contract_path),
            "V134 smoke gate contract path changed")
    require(receipt.get("contract_sha256") == sha256(contract_path),
            "V134 smoke gate contract hash changed")
    contract = read_json(contract_path)
    for relative, expected in contract.get("source_sha256", {}).items():
        require(sha256(repo_root / relative) == expected,
                "V134 source changed after smoke: {}".format(relative))
    for relative, expected in receipt.get(
            "execution_source_sha256", {}).items():
        require(sha256(repo_root / relative) == expected,
                "V134 execution source changed after smoke: {}".format(
                    relative
                ))
    for path, expected in receipt.get("evidence_sha256", {}).items():
        require(sha256(path) == expected,
                "V134 smoke evidence changed: {}".format(path))
    print(json.dumps({
        "receipt": str(receipt_path),
        "sha256": sha256(receipt_path),
        "verdict": "verified",
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--run-dir", required=True)
    build_parser.add_argument("--baseline-receipt", required=True)
    build_parser.add_argument("--launch-log", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--repo-root", required=True)
    build_parser.add_argument("--parent-checkpoint", required=True)
    build_parser.add_argument("--contract-receipt", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--receipt", required=True)
    verify_parser.add_argument("--repo-root", required=True)
    verify_parser.add_argument("--parent-checkpoint", required=True)
    verify_parser.add_argument("--contract-receipt", required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
