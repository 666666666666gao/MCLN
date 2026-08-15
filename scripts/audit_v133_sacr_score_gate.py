#!/usr/bin/env python3
"""Build and verify the fail-closed V133 identity/smoke gate receipt."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

from v133_receipt_utils import atomic_write_new_json


SCHEMA = "mcln-v133-review3-gate-v1"
BINDING_SCHEMA = "mcln-v133-launch-binding-v1"
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA256 = (
    "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208"
)
RUNNER_FILE = "scripts/run_v133_sacr_score_refiner.sh"
EXCLUDED_SOURCE_PARTS = {".git", ".pytest_cache", "__pycache__"}
FIXED_CONFIG = {
    "batch_size": 8,
    "max_delta": 0.25,
    "mask_weight": 0.25,
    "max_formal_epochs": 4,
    "nproc_per_node": 1,
    "refiner_lr": 0.0003,
    "temperature": 0.1,
}


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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_repo_root(value):
    repo_root = Path(value).resolve()
    require(
        repo_root == SCRIPT_REPO_ROOT,
        "--repo-root must contain the executing V133 audit script",
    )
    return repo_root


def require_read_only(path, label):
    mode = stat.S_IMODE(os.stat(str(path)).st_mode)
    require(mode == 0o444, "{} must have mode 0444".format(label))


def receipt_files(run_dir):
    paths = list(Path(run_dir).glob("eval_metrics_epoch_*.json"))

    def epoch(path):
        match = re.search(r"epoch_(\d+)\.json$", path.name)
        require(match is not None, "invalid evaluation receipt name")
        return int(match.group(1))

    return sorted(paths, key=epoch)


def validate_common_config(
        config, max_epoch, audit, eval_mode, parent_checkpoint):
    exact = {
        "batch_size": 8,
        "checkpoint_metric_retention": True,
        "dataloader_prefetch_factor": 1,
        "dataset": ["scanrefer"],
        "debug": True,
        "debug_train_holdout": True,
        "eval": eval_mode,
        "eval_use_selector_choice_scores": True,
        "expected_eval_sample_count": 128,
        "local_rank": 0,
        "max_epoch": max_epoch,
        "model": "MCLN",
        "num_workers": 2,
        "print_freq": 1,
        "rng_seed": 0,
        "sacr_geo_dim": 16,
        "sacr_hidden_dim": 288,
        "sacr_max_pairs": 3,
        "sacr_min_parse_confidence": 0.0,
        "sacr_score_contract_audit": audit,
        "sacr_score_mask_weight": 0.25,
        "sacr_score_max_delta": 0.25,
        "sacr_score_refiner_loss_weight": 1.0,
        "sacr_score_refiner_lr": 0.0003,
        "sacr_score_refiner_train_only": True,
        "sacr_score_temperature": 0.1,
        "sacr_top_k_anchors": 16,
        "sacr_top_m_targets": 32,
        "save_freq": 1,
        "start_epoch": 1,
        "source_choice_selector_choice_target": (
            "precision_gain_default_sourcewise_focal_bce"
        ),
        "source_choice_selector_default_source": "default",
        "source_choice_selector_hidden_dim": 288,
        "source_choice_selector_min_iou_gap": 0.03,
        "source_choice_selector_sources": (
            "default,default_rank_blend_contrastive010"
        ),
        "test_dataset": "scanrefer",
        "use_sacr_score_refiner": True,
        "use_source_choice_selector": True,
        "val_freq": 1,
    }
    mismatches = []
    for key, expected in exact.items():
        actual = config.get(key)
        if actual != expected:
            mismatches.append(
                "{} expected {!r}, got {!r}".format(key, expected, actual)
            )
    checkpoint_path = config.get("checkpoint_path")
    if (
            not checkpoint_path
            or Path(checkpoint_path).resolve() != parent_checkpoint):
        mismatches.append(
            "checkpoint_path expected {!r}, got {!r}".format(
                str(parent_checkpoint), checkpoint_path
            )
        )
    require(not mismatches, "run config differs: " + "; ".join(mismatches))


def validate_launch_binding(
        binding_path, mode, repo_root, parent_checkpoint, launch_log):
    binding_path = Path(binding_path).resolve()
    require(binding_path.is_file(), "{} launch binding is missing".format(mode))
    require_read_only(binding_path, "{} launch binding".format(mode))
    binding = read_json(binding_path)
    require(binding.get("schema") == BINDING_SCHEMA,
            "{} launch binding schema changed".format(mode))
    require(binding.get("verdict") == "bound",
            "{} launch binding verdict changed".format(mode))
    require(binding.get("mode") == mode,
            "{} launch binding mode changed".format(mode))
    require(binding.get("fixed_config") == FIXED_CONFIG,
            "{} launch binding config changed".format(mode))
    require(binding.get("parent_sha256") == PARENT_SHA256,
            "{} launch binding parent hash changed".format(mode))
    require(
        Path(binding.get("parent_checkpoint", "")).resolve()
        == parent_checkpoint,
        "{} launch binding parent path changed".format(mode),
    )
    require(
        Path(binding.get("launch_log", "")).resolve() == launch_log,
        "{} launch binding log path changed".format(mode),
    )
    require(binding.get("source_sha256") == source_hashes(repo_root),
            "{} launch binding source changed".format(mode))
    return binding_path


def validate_metric_receipt(path):
    data = read_json(path)
    require(data.get("schema") == "mcln-retrain-metrics-v1",
            "unexpected metric receipt schema")
    require(data.get("sample_count") == 128,
            "identity/smoke receipt must contain 128 samples")
    position = data.get("position", {})
    fixed = position.get("fixed_default", {})
    learned = position.get("learned_selector", {})
    for key in ("hits025", "hits050"):
        require(isinstance(fixed.get(key), int), "fixed hits are missing")
        require(isinstance(learned.get(key), int), "learned hits are missing")
    mask = data.get("mask", {})
    require(isinstance(mask.get("hits025"), int), "mask@.25 is missing")
    require(isinstance(mask.get("hits050"), int), "mask@.50 is missing")
    require(math.isfinite(float(mask.get("miou"))), "mask mIoU is not finite")
    return data


def parse_stat_lines(log_text):
    rows = []
    for line in log_text.splitlines():
        if "[source_moe]" not in line or "sacr_score_gate_value" not in line:
            continue
        values = {}
        for name, value in re.findall(r"(sacr_[A-Za-z0-9_]+)\s+(-?[0-9.]+)", line):
            values[name] = float(value)
        rows.append(values)
    return rows


def validate_identity(
        run_dir, launch_log, binding_path, repo_root, parent_checkpoint):
    run_dir = Path(run_dir).resolve()
    launch_log = Path(launch_log).resolve()
    require(launch_log.is_file(), "identity launch log is missing")
    binding_path = validate_launch_binding(
        binding_path, "baseline", repo_root, parent_checkpoint, launch_log
    )
    receipts = receipt_files(run_dir)
    require(len(receipts) == 1, "identity run must have one metric receipt")
    metrics = validate_metric_receipt(receipts[0])
    fixed = metrics["position"]["fixed_default"]
    learned = metrics["position"]["learned_selector"]
    require(fixed == learned, "zero-gate identity changed REC hits")
    config_path = run_dir / "config.json"
    log_path = run_dir / "log.txt"
    require(config_path.is_file() and log_path.is_file(),
            "identity config/log is missing")
    config = read_json(config_path)
    validate_common_config(config, 1, True, True, parent_checkpoint)
    require(Path(config.get("log_dir", "")).resolve() == run_dir,
            "identity config does not name its run directory")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    launch_text = launch_log.read_text(encoding="utf-8", errors="replace")
    require(str(config_path) in launch_text,
            "identity launch log does not name its config receipt")
    require("finished v133_sacr_score_refiner" in launch_text,
            "identity launch did not finish")
    stat_rows = parse_stat_lines(log_text)
    require(stat_rows, "identity SACR statistics are missing")
    final = stat_rows[-1]
    require(final.get("sacr_score_contract_audit_pass") == 1.0,
            "tensor-level identity audit did not pass")
    require(final.get("sacr_score_contract_audit_tensor_count") == 7.0,
            "tensor-level identity audit did not cover seven frozen outputs")
    require(final.get("sacr_score_gate_value") == 0.0,
            "identity gate is not exactly zero")
    require(final.get("sacr_score_residual_abs_max") == 0.0,
            "identity residual is not exactly zero")
    return {
        "run_dir": str(run_dir),
        "receipt": str(receipts[0]),
        "rec_hits": learned,
        "mask": metrics["mask"],
        "launch_binding": str(binding_path),
        "launch_log": str(launch_log),
    }, (receipts[0], config_path, log_path, launch_log, binding_path)


def validate_smoke(
        run_dir, launch_log, binding_path, repo_root, parent_checkpoint):
    run_dir = Path(run_dir).resolve()
    launch_log = Path(launch_log).resolve()
    require(launch_log.is_file(), "smoke launch log is missing")
    binding_path = validate_launch_binding(
        binding_path, "smoke", repo_root, parent_checkpoint, launch_log
    )
    receipts = receipt_files(run_dir)
    require(len(receipts) == 2, "smoke run must have two metric receipts")
    metrics = [validate_metric_receipt(path) for path in receipts]
    config_path = run_dir / "config.json"
    log_path = run_dir / "log.txt"
    require(config_path.is_file() and log_path.is_file(),
            "smoke config/log is missing")
    config = read_json(config_path)
    validate_common_config(config, 2, False, False, parent_checkpoint)
    require(Path(config.get("log_dir", "")).resolve() == run_dir,
            "smoke config does not name its run directory")
    net_deltas = []
    for epoch_metrics in metrics:
        fixed = epoch_metrics["position"]["fixed_default"]
        learned = epoch_metrics["position"]["learned_selector"]
        epoch_delta = {}
        for key in ("hits025", "hits050"):
            delta = learned[key] - fixed[key]
            require(delta >= 0, "smoke fix<break at {}".format(key))
            epoch_delta[key] = delta
        net_deltas.append(epoch_delta)
    run_log_text = log_path.read_text(encoding="utf-8", errors="replace")
    launch_text = launch_log.read_text(encoding="utf-8", errors="replace")
    require(str(config_path) in launch_text,
            "smoke launch log does not name its config receipt")
    require("finished v133_sacr_score_refiner" in launch_text,
            "smoke launch did not finish")
    log_text = run_log_text + "\n" + launch_text
    require(
        re.search(
            r"Debug train holdout: train=128 examples/128 scenes; "
            r"holdout=128 examples/120 scenes; overlap=0",
            log_text,
        ) is not None,
        "smoke scene cardinality/disjointness evidence is missing",
    )
    transitions = re.findall(
        r"Acc0\.(25|50) selector_fix: ([0-9.]+), selector_break: ([0-9.]+)",
        log_text,
    )
    require(len(transitions) >= 4, "smoke fix/break logs are incomplete")
    for threshold, fixed_ratio, broken_ratio in transitions:
        require(float(fixed_ratio) + 1e-12 >= float(broken_ratio),
                "smoke log fix<break at 0.{}".format(threshold))
    stat_rows = parse_stat_lines(log_text)
    require(len(stat_rows) >= 2, "smoke SACR statistics are incomplete")
    for row in stat_rows:
        for key in (
                "sacr_score_gate_value",
                "sacr_score_residual_abs_max",
                "sacr_score_residual_abs_mean"):
            require(key in row and math.isfinite(row[key]),
                    "smoke {} is missing/non-finite".format(key))
        require(row["sacr_score_residual_abs_max"] < 0.25,
                "smoke residual reached the hard bound")
    return {
        "run_dir": str(run_dir),
        "receipts": [str(path) for path in receipts],
        "net_hit_deltas": net_deltas,
        "epochs": [
            {
                "position": item["position"],
                "mask": item["mask"],
            } for item in metrics
        ],
        "launch_binding": str(binding_path),
        "launch_log": str(launch_log),
    }, tuple(receipts) + (
        config_path, log_path, launch_log, binding_path
    )


def source_hashes(repo_root):
    repo_root = Path(repo_root).resolve()
    candidates = []
    for path in repo_root.rglob("*.py"):
        relative = path.relative_to(repo_root)
        if any(
                part in EXCLUDED_SOURCE_PARTS or part.startswith(".v")
                for part in relative.parts):
            continue
        if path.is_file():
            candidates.append(relative.as_posix())
    candidates.append(RUNNER_FILE)
    result = {}
    for relative in sorted(set(candidates)):
        path = repo_root / relative
        require(path.is_file(), "source file is missing: {}".format(relative))
        result[relative] = sha256(path)
    require(len(result) > 8, "source binding is unexpectedly narrow")
    return result


def bind(args):
    repo_root = resolve_repo_root(args.repo_root)
    parent = Path(args.parent_checkpoint).resolve()
    require(parent.is_file(), "parent checkpoint is missing")
    require(sha256(parent) == PARENT_SHA256, "parent checkpoint hash changed")
    launch_log = Path(args.launch_log).resolve()
    require(not launch_log.exists(), "launch log already exists")
    payload = {
        "schema": BINDING_SCHEMA,
        "verdict": "bound",
        "mode": args.mode,
        "fixed_config": FIXED_CONFIG,
        "parent_checkpoint": str(parent),
        "parent_sha256": PARENT_SHA256,
        "launch_log": str(launch_log),
        "source_sha256": source_hashes(repo_root),
    }
    output = atomic_write_new_json(payload, args.output)
    print(json.dumps({
        "binding": str(output),
        "sha256": sha256(output),
        "verdict": "bound",
    }, sort_keys=True))


def build(args):
    repo_root = resolve_repo_root(args.repo_root)
    parent = Path(args.parent_checkpoint).resolve()
    require(parent.is_file(), "parent checkpoint is missing")
    require(sha256(parent) == PARENT_SHA256, "parent checkpoint hash changed")
    identity, identity_files = validate_identity(
        args.baseline_run_dir,
        args.baseline_launch_log,
        args.baseline_binding,
        repo_root,
        parent,
    )
    smoke, smoke_files = validate_smoke(
        args.smoke_run_dir,
        args.smoke_launch_log,
        args.smoke_binding,
        repo_root,
        parent,
    )
    contract_path = Path(args.contract_receipt).resolve()
    require(contract_path.is_file(), "supervision contract receipt is missing")
    require_read_only(contract_path, "supervision contract receipt")
    contract = read_json(contract_path)
    require(
        contract.get("schema") == "mcln-v133-supervision-contract-v1"
        and contract.get("verdict") == "pass",
        "supervision contract did not pass",
    )
    expected_contract_sources = {
        "models/losses.py": sha256(repo_root / "models/losses.py"),
        "scripts/audit_v133_sacr_supervision_contract.py": sha256(
            repo_root / "scripts/audit_v133_sacr_supervision_contract.py"
        ),
        "scripts/v133_receipt_utils.py": sha256(
            repo_root / "scripts/v133_receipt_utils.py"
        ),
    }
    require(contract.get("source_sha256") == expected_contract_sources,
            "supervision contract source binding changed")
    evidence_files = {}
    for path in identity_files + smoke_files + (contract_path,):
        evidence_files[str(Path(path).resolve())] = sha256(path)
    payload = {
        "schema": SCHEMA,
        "verdict": "pass",
        "parent_checkpoint": str(parent),
        "parent_sha256": PARENT_SHA256,
        "fixed_config": FIXED_CONFIG,
        "source_sha256": source_hashes(repo_root),
        "evidence_sha256": evidence_files,
        "identity": identity,
        "smoke": smoke,
        "supervision_contract": {
            "receipt": str(contract_path),
            "sha256": sha256(contract_path),
            "cross_dataset": contract["cross_dataset"],
            "ddp": contract["ddp"],
        },
    }
    output = atomic_write_new_json(payload, args.output)
    print(json.dumps({
        "receipt": str(output),
        "sha256": sha256(output),
        "verdict": "pass",
    }, sort_keys=True))


def verify(args):
    receipt = Path(args.receipt).resolve()
    require(receipt.is_file(), "gate receipt is missing")
    require_read_only(receipt, "gate receipt")
    payload = read_json(receipt)
    require(payload.get("schema") == SCHEMA, "gate schema changed")
    require(payload.get("verdict") == "pass", "gate verdict is not pass")
    require(payload.get("fixed_config") == FIXED_CONFIG,
            "gate fixed configuration changed")
    repo_root = resolve_repo_root(args.repo_root)
    parent = Path(args.parent_checkpoint).resolve()
    require(parent.is_file(), "parent checkpoint is missing")
    require(sha256(parent) == PARENT_SHA256, "parent checkpoint hash changed")
    require(payload.get("parent_sha256") == PARENT_SHA256,
            "gate parent hash changed")
    require(
        Path(payload.get("parent_checkpoint", "")).resolve() == parent,
        "gate parent path changed",
    )
    require(
        payload.get("source_sha256") == source_hashes(repo_root),
        "current source differs from the gated source",
    )
    for path_text, expected in payload.get("evidence_sha256", {}).items():
        path = Path(path_text)
        require(path.is_file(), "gated evidence is missing: {}".format(path))
        require(sha256(path) == expected,
                "gated evidence changed: {}".format(path))
    require(payload.get("evidence_sha256"), "gate has no bound evidence")
    for mode, section in (("baseline", "identity"), ("smoke", "smoke")):
        evidence = payload.get(section, {})
        validate_launch_binding(
            evidence.get("launch_binding", ""),
            mode,
            repo_root,
            parent,
            Path(evidence.get("launch_log", "")).resolve(),
        )
    contract = payload.get("supervision_contract", {})
    contract_path = Path(contract.get("receipt", "")).resolve()
    require(contract_path.is_file(), "gated supervision contract is missing")
    require_read_only(contract_path, "gated supervision contract")
    require(sha256(contract_path) == contract.get("sha256"),
            "gated supervision contract changed")
    print(json.dumps({
        "receipt": str(receipt),
        "sha256": sha256(receipt),
        "verdict": "pass",
    }, sort_keys=True))


def parser():
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command")
    bind_parser = subparsers.add_parser("bind")
    bind_parser.add_argument(
        "--mode", required=True, choices=("baseline", "smoke", "formal")
    )
    bind_parser.add_argument("--output", required=True)
    bind_parser.add_argument("--repo-root", required=True)
    bind_parser.add_argument("--parent-checkpoint", required=True)
    bind_parser.add_argument("--launch-log", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--baseline-run-dir", required=True)
    build_parser.add_argument("--baseline-launch-log", required=True)
    build_parser.add_argument("--baseline-binding", required=True)
    build_parser.add_argument("--smoke-run-dir", required=True)
    build_parser.add_argument("--smoke-launch-log", required=True)
    build_parser.add_argument("--smoke-binding", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--repo-root", required=True)
    build_parser.add_argument("--parent-checkpoint", required=True)
    build_parser.add_argument("--contract-receipt", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--receipt", required=True)
    verify_parser.add_argument("--repo-root", required=True)
    verify_parser.add_argument("--parent-checkpoint", required=True)
    return root


def main():
    args = parser().parse_args()
    require(args.command in ("bind", "build", "verify"),
            "command is required")
    if args.command == "bind":
        bind(args)
    elif args.command == "build":
        build(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
