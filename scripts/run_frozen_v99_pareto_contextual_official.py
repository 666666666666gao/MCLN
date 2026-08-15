#!/usr/bin/env python
"""Run and seal the sole frozen V99 ScanRefer validation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_frozen_rec_geometry_official as geometry_official
from scripts.build_v99_pareto_contextual_artifact import (
    V99_RESULT_SHA256,
    load_v99_artifact,
)


SAMPLE_COUNT = 9508
EXPERIMENT = "epoch71_v99_pareto_contextual_official"
MASTER_PORT = 29699
ARTIFACT_PATH = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
    "v99_artifacts/pareto_contextual_h128_seed0_fullfit.pth"
)
ARTIFACT_SHA256 = (
    "9752990c393fa6e45173a9dd129c4de4bb740924094dcbbec2f3121cbf39d1f2"
)
CLAIM_PATH = ARTIFACT_PATH.parent / "v99_official_validation_once.claim.json"
RESULT_SCHEMA = "rec-v99-pareto-contextual-official-validation-result-v1"
MIN_REC_HITS025 = 5610
MIN_REC_HITS050 = 4659
MASK_GEOMETRY_BASELINE_HITS025 = 5676
MASK_GEOMETRY_BASELINE_HITS050 = 4662
MASK_GEOMETRY_BASELINE_MIOU = 0.4176762145248869
MASK_V19_BEST_HITS025 = 5688
MASK_V19_BEST_HITS050 = 4672
MASK_V19_BEST_MIOU = 0.4186131

POSITION_RE = re.compile(
    r"last_ position alignment Acc0\.(25|50): Top-1: "
    r"((?:0|1)\.[0-9]{5})(?=,|\r?$)",
    re.MULTILINE,
)
MASK_RE = {
    "mask_acc025": re.compile(r"^.*?overall25\s+([01]?\.[0-9]+)\s*$", re.MULTILINE),
    "mask_acc050": re.compile(r"^.*?overall50\s+([01]?\.[0-9]+)\s*$", re.MULTILINE),
    "mask_miou": re.compile(r"^.*?mask_sem\s+([01]?\.[0-9]+)\s*$", re.MULTILINE),
}
FORBIDDEN = {
    "--eval_train", "--eval_use_ground_truth", "--use_gt_masks",
    "--use_gt_boxes", "--gt_masks", "--gt_boxes", "--butd_gt",
}


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path, label, require_readonly=False):
    path = Path(path).expanduser().absolute()
    try:
        entry = os.lstat(str(path))
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
            raise ValueError("{} must be a regular non-symlink file".format(label))
        descriptor = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            before = os.fstat(descriptor)
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        live = os.stat(str(path), follow_symlinks=False)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("could not snapshot {}".format(label)) from error
    identity = lambda value: (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )
    if identity(before) != identity(after) or identity(after) != identity(live):
        raise ValueError("{} changed during stable snapshot".format(label))
    mode = stat.S_IMODE(live.st_mode)
    if require_readonly and mode != 0o444:
        raise ValueError("{} must have mode 0444".format(label))
    return {
        "path": str(path.resolve(strict=True)),
        "device": int(live.st_dev),
        "inode": int(live.st_ino),
        "mode": mode,
        "size": int(live.st_size),
        "mtime_ns": int(live.st_mtime_ns),
        "ctime_ns": int(live.st_ctime_ns),
        "sha256": digest.hexdigest(),
    }


def _canonical_json_bytes(value):
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("ascii")


def _exclusive_json(path, value):
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(value)
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise OSError("official JSON write made no progress")
            offset += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _snapshot(path, "published JSON", require_readonly=True)


def _replace_option(command, option, value):
    command = list(command)
    if command.count(option) != 1:
        raise ValueError("authoritative command option is missing or duplicated")
    index = command.index(option)
    if index + 1 >= len(command):
        raise ValueError("authoritative command option lacks a value")
    command[index + 1] = str(value)
    return command


def build_authoritative_command(output_dir):
    command = geometry_official.build_authoritative_command()
    command = _replace_option(command, "--master_port", MASTER_PORT)
    command = _replace_option(command, "--log_dir", Path(output_dir).resolve())
    command = _replace_option(command, "--exp", EXPERIMENT)
    command.extend([
        "--rec_hierarchical_reranker_checkpoint", str(ARTIFACT_PATH),
        "--eval_use_rec_hierarchical_reranker_scores",
    ])
    return command


def validate_authoritative_command(command, output_dir):
    expected = build_authoritative_command(output_dir)
    if list(command) != expected:
        raise ValueError("V99 official command changed")
    required = {
        "--eval", "--eval_use_rec_reranker_scores",
        "--eval_use_rec_geometry_reranker_scores",
        "--eval_use_rec_hierarchical_reranker_scores",
    }
    if not required.issubset(command) or FORBIDDEN.intersection(command):
        raise ValueError("V99 official command policy is invalid")
    if "--eval_use_rec_selective_residual_scores" in command:
        raise ValueError("V99 cannot combine the selective residual policy")
    return list(command)


def _recover_mask_hits(token):
    if not isinstance(token, str) or re.fullmatch(r"(?:0|1)?\.[0-9]+", token) is None:
        raise ValueError("mask metric token is invalid")
    value = float(token)
    decimals = len(token.split(".", 1)[1])
    candidate = int(round(value * SAMPLE_COUNT))
    tolerance = 0.5 * 10.0 ** (-decimals) + 1e-12
    if (not 0 <= candidate <= SAMPLE_COUNT
            or abs(value - candidate / float(SAMPLE_COUNT)) > tolerance):
        raise ValueError("mask rate does not identify an integer hit count")
    return candidate


def parse_metrics(text):
    positions = POSITION_RE.findall(text)
    if len(positions) != 2 or {item[0] for item in positions} != {"25", "50"}:
        raise ValueError("official output has missing or duplicate REC metrics")
    tokens = {threshold: token for threshold, token in positions}
    result = {
        "rec_acc025": tokens["25"],
        "rec_acc050": tokens["50"],
        "rec_hits025": geometry_official.recover_exact_hits(
            tokens["25"], SAMPLE_COUNT
        ),
        "rec_hits050": geometry_official.recover_exact_hits(
            tokens["50"], SAMPLE_COUNT
        ),
    }
    mask_tokens = {}
    for name, pattern in MASK_RE.items():
        matches = pattern.findall(text)
        if len(matches) != 1:
            raise ValueError("official output has missing or duplicate {}".format(name))
        mask_tokens[name] = matches[0]
    result.update({
        "mask_acc025": float(mask_tokens["mask_acc025"]),
        "mask_acc050": float(mask_tokens["mask_acc050"]),
        "mask_miou": float(mask_tokens["mask_miou"]),
        "mask_hits025": _recover_mask_hits(mask_tokens["mask_acc025"]),
        "mask_hits050": _recover_mask_hits(mask_tokens["mask_acc050"]),
    })
    if (not 0 <= result["rec_hits050"] <= result["rec_hits025"] <= SAMPLE_COUNT
            or not 0 <= result["mask_hits050"] <= result["mask_hits025"] <= SAMPLE_COUNT
            or not 0.0 <= result["mask_miou"] <= 1.0):
        raise ValueError("official metric threshold nesting is invalid")
    return result


def _python_tree_snapshot():
    root = ROOT.resolve(strict=True)
    records = {}
    for path in sorted(root.rglob("*.py")):
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        if path.is_symlink():
            raise ValueError("Python source tree contains a symlink")
        relative = str(path.relative_to(root))
        records[relative] = {
            "size": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
    if not records:
        raise ValueError("Python source tree is empty")
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return {
        "root": str(root),
        "file_count": len(records),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "files": records,
    }


def _protected_snapshots():
    result = {
        "backbone": _snapshot(
            geometry_official.OFFICIAL_CHECKPOINT_PATH, "backbone", True
        ),
        "parent": _snapshot(
            geometry_official.OFFICIAL_PARENT_ARTIFACT_PATH, "parent", True
        ),
        "geometry": _snapshot(
            geometry_official.OFFICIAL_SELECTED_ARTIFACT_PATH, "geometry", True
        ),
        "v99": _snapshot(ARTIFACT_PATH, "V99 artifact", True),
    }
    expected = {
        "backbone": geometry_official.OFFICIAL_CHECKPOINT_SHA256,
        "parent": geometry_official.OFFICIAL_PARENT_ARTIFACT_SHA256,
        "geometry": geometry_official.OFFICIAL_SELECTED_ARTIFACT_SHA256,
        "v99": ARTIFACT_SHA256,
    }
    if any(result[name]["sha256"] != sha for name, sha in expected.items()):
        raise ValueError("official artifact SHA binding changed")
    model, artifact = load_v99_artifact(
        ARTIFACT_PATH,
        device="cpu",
        expected_artifact_sha256=ARTIFACT_SHA256,
        parent_sha256=expected["parent"],
        geometry_sha256=expected["geometry"],
    )
    if (artifact.get("deployable") is not True
            or artifact.get("validation_data_accessed") is not False
            or artifact.get("oof_evidence", {}).get("sha256")
            != V99_RESULT_SHA256
            or getattr(model, "_artifact_sha256", None) != ARTIFACT_SHA256):
        raise ValueError("V99 deployable artifact contract is invalid")
    return result


def run_official(output_dir, dry_run=False):
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("official output directory already exists")
    if not dry_run and (CLAIM_PATH.exists() or CLAIM_PATH.is_symlink()):
        raise FileExistsError("V99 official validation claim already exists")
    protected_before = _protected_snapshots()
    code_before = _python_tree_snapshot()
    command = validate_authoritative_command(
        build_authoritative_command(output), output
    )
    environment = os.environ.copy()
    for key in ("PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        environment.pop(key, None)
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "pointnet2"),
    })
    preflight = {
        "schema": RESULT_SCHEMA + "-preflight-v1",
        "created_at_utc": _utc_now(),
        "sample_count": SAMPLE_COUNT,
        "command": command,
        "environment": {
            key: environment.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH")
        },
        "protected": protected_before,
        "code": code_before,
        "inference_uses_ground_truth": False,
        "validation_data_accessed": False,
    }
    if dry_run:
        return preflight
    claim = _exclusive_json(CLAIM_PATH, preflight)
    os.mkdir(str(output), 0o700)
    stdout_path = output / "official_stdout.log"
    descriptor = os.open(
        str(stdout_path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    returncode = None
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            completed = subprocess.run(
                command, cwd=str(ROOT), env=environment,
                stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
            returncode = int(completed.returncode)
    finally:
        os.close(descriptor)
    os.chmod(str(stdout_path), 0o444)
    protected_after = _protected_snapshots()
    code_after = _python_tree_snapshot()
    if protected_after != protected_before:
        raise RuntimeError("protected artifact changed during V99 official run")
    if code_after != code_before:
        raise RuntimeError("Python source tree changed during V99 official run")
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    if returncode != 0:
        raise RuntimeError("V99 official subprocess failed with code {}".format(returncode))
    populations = re.findall(r"length of testing dataset:\s*(\d+)", text)
    if len(populations) != 1 or populations[0] != str(SAMPLE_COUNT):
        raise ValueError("V99 official validation population is not 9508")
    metrics = parse_metrics(text)
    gates = {
        "rec025_at_least_059": metrics["rec_hits025"] >= MIN_REC_HITS025,
        "rec050_at_least_049": metrics["rec_hits050"] >= MIN_REC_HITS050,
        "mask025_preserves_geometry_baseline": (
            metrics["mask_hits025"] >= MASK_GEOMETRY_BASELINE_HITS025
        ),
        "mask050_preserves_geometry_baseline": (
            metrics["mask_hits050"] >= MASK_GEOMETRY_BASELINE_HITS050
        ),
        "mask_miou_preserves_geometry_baseline": (
            metrics["mask_miou"] >= MASK_GEOMETRY_BASELINE_MIOU
        ),
        "mask025_reaches_v19_best": (
            metrics["mask_hits025"] >= MASK_V19_BEST_HITS025
        ),
        "mask050_reaches_v19_best": (
            metrics["mask_hits050"] >= MASK_V19_BEST_HITS050
        ),
        "mask_miou_reaches_v19_best": (
            metrics["mask_miou"] >= MASK_V19_BEST_MIOU
        ),
    }
    timestamp_runs = [
        path for path in (output / "scanrefer" / EXPERIMENT).iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if len(timestamp_runs) != 1:
        raise ValueError("V99 official run directory is ambiguous")
    run_path = timestamp_runs[0]
    config = _snapshot(run_path / "config.json", "official config")
    log = _snapshot(run_path / "log.txt", "official log")
    result = {
        "schema": RESULT_SCHEMA,
        "version": 1,
        "created_at_utc": _utc_now(),
        "sample_count": SAMPLE_COUNT,
        "returncode": returncode,
        "validation_data_accessed": True,
        "inference_uses_ground_truth": False,
        "metrics": metrics,
        "gates": gates,
        "rec_target_pass": gates["rec025_at_least_059"] and gates[
            "rec050_at_least_049"
        ],
        "mask_baseline_preservation_pass": gates[
            "mask025_preserves_geometry_baseline"
        ] and gates["mask050_preserves_geometry_baseline"] and gates[
            "mask_miou_preserves_geometry_baseline"
        ],
        "mask_v19_best_pass": gates["mask025_reaches_v19_best"] and gates[
            "mask050_reaches_v19_best"
        ] and gates["mask_miou_reaches_v19_best"],
        "all_goals_pass": gates["rec025_at_least_059"] and gates[
            "rec050_at_least_049"
        ] and gates["mask025_preserves_geometry_baseline"] and gates[
            "mask050_preserves_geometry_baseline"
        ] and gates["mask_miou_preserves_geometry_baseline"],
        "thresholds": {
            "min_rec_hits025": MIN_REC_HITS025,
            "min_rec_hits050": MIN_REC_HITS050,
            "mask_geometry_baseline_hits025": MASK_GEOMETRY_BASELINE_HITS025,
            "mask_geometry_baseline_hits050": MASK_GEOMETRY_BASELINE_HITS050,
            "mask_geometry_baseline_miou": MASK_GEOMETRY_BASELINE_MIOU,
            "mask_v19_best_hits025": MASK_V19_BEST_HITS025,
            "mask_v19_best_hits050": MASK_V19_BEST_HITS050,
            "mask_v19_best_miou": MASK_V19_BEST_MIOU,
        },
        "command": command,
        "environment": preflight["environment"],
        "claim": claim,
        "stdout": _snapshot(stdout_path, "official stdout", True),
        "run": {"path": str(run_path), "config": config, "log": log},
        "protected_before": protected_before,
        "protected_after": protected_after,
        "code_before": code_before,
        "code_after": code_after,
    }
    _exclusive_json(output / "official_result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run_official(args.output_dir, dry_run=args.dry_run)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
