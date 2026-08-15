#!/usr/bin/env python
"""Run and seal the sole frozen V113 ScanRefer mesh-superpoint validation."""

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
from scripts.build_v113_meshsp_asymmetric_risk_artifact import (
    EXPECTED_POLICY,
    load_v113_artifact,
)


SAMPLE_COUNT = 9508
EXPERIMENT = "epoch71_v113_meshsp_asymmetric_risk_committee_full9508"
MASTER_PORT = 29791
V113_BASE = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp"
)
PARENT_ARTIFACT_PATH = V113_BASE / "v108_artifacts/parent_h256_seed0.pth"
GEOMETRY_ARTIFACT_PATH = V113_BASE / "v108_artifacts/geometry_h256_seed0.pth"
ARTIFACT_PATH = V113_BASE / (
    "v113_artifacts/asymmetric_risk_committee_h128_seeds0_1_2_fullfit.pth"
)
PARITY_PATH = V113_BASE / "v113_train_runtime_parity.json"
PARENT_ARTIFACT_SHA256 = (
    "7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f"
)
GEOMETRY_ARTIFACT_SHA256 = (
    "20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972"
)
ARTIFACT_SHA256 = (
    "45f96279794da73c9d21f5f7e817bb47def03a86a30ab7db092c1b1c0275a37b"
)
V113_RESULT_SHA256 = (
    "ced399bca041cfa1f4213671100347f4a2423783aee4936ce7a82f785605e61d"
)
PARITY_SHA256 = (
    "53e86c392e86a7cb8813041d3a978413cc3c1784f741ded82a9444aba8ac4a81"
)
CLAIM_PATH = ARTIFACT_PATH.parent / (
    "v113_meshsp_official_once_after_train_runtime_parity.claim.json"
)
RESULT_SCHEMA = "rec-v113-meshsp-official-validation-result-v1"
MESHSP_DATA_ROOT = "/root/autodl-tmp/DATA_ROOT_mcln_meshsp/"
MIN_REC_HITS025 = 5610
MIN_REC_HITS050 = 4659
MASK_V99_MESHSP_HITS025 = 5690
MASK_V99_MESHSP_HITS050 = 4976
MASK_V99_MESHSP_MIOU = 0.4593026020554575
MASK_USER_BASELINE_ACC025 = 0.5870
MASK_USER_BASELINE_ACC050 = 0.5070
MASK_USER_BASELINE_MIOU = 0.4472

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
POSITION_SUBGROUP_RE = re.compile(
    r"position subgroup (unique|multiple) Acc0\.(25|50): "
    r"hits=(\d+), total=(\d+), accuracy=([01]?\.[0-9]+)",
)
MASK_SUBGROUP_RE = re.compile(
    r"^.*?(unique25|unique50|multi25|multi50)\s+([01]?\.[0-9]+)\s*$",
    re.MULTILINE,
)
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
    command = _replace_option(command, "--data_root", MESHSP_DATA_ROOT)
    command = _replace_option(
        command, "--rec_reranker_checkpoint", PARENT_ARTIFACT_PATH
    )
    command = _replace_option(
        command, "--rec_geometry_reranker_checkpoint",
        GEOMETRY_ARTIFACT_PATH,
    )
    command.extend([
        "--rec_hierarchical_reranker_checkpoint", str(ARTIFACT_PATH),
        "--eval_use_rec_hierarchical_reranker_scores",
    ])
    return command


def validate_authoritative_command(command, output_dir):
    expected = build_authoritative_command(output_dir)
    if list(command) != expected:
        raise ValueError("V113 meshsp official command changed")
    required = {
        "--eval", "--eval_use_rec_reranker_scores",
        "--eval_use_rec_geometry_reranker_scores",
        "--eval_use_rec_hierarchical_reranker_scores",
    }
    if not required.issubset(command) or FORBIDDEN.intersection(command):
        raise ValueError("V113 meshsp official command policy is invalid")
    if "--eval_use_rec_selective_residual_scores" in command:
        raise ValueError("V113 cannot combine the selective residual policy")
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


def _recover_subgroup_hits(token, total, label):
    if not isinstance(token, str) or re.fullmatch(
            r"(?:0|1)?\.[0-9]+", token) is None:
        raise ValueError("{} rate token is invalid".format(label))
    value = float(token)
    decimals = len(token.split(".", 1)[1])
    candidate = int(round(value * total))
    tolerance = 0.5 * 10.0 ** (-decimals) + 1e-12
    if (not 0 <= candidate <= total
            or abs(value - candidate / float(total)) > tolerance):
        raise ValueError("{} rate does not identify integer hits".format(label))
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
    position_matches = POSITION_SUBGROUP_RE.findall(text)
    if len(position_matches) != 4:
        raise ValueError("official output lacks exact REC unique/multiple metrics")
    position_subgroups = {}
    for group, threshold, hits, total, accuracy in position_matches:
        row = position_subgroups.setdefault(group, {})
        if "total" in row and row["total"] != int(total):
            raise ValueError("REC subgroup thresholds use different totals")
        row["total"] = int(total)
        row["hits" + threshold] = int(hits)
        row["acc" + threshold] = float(accuracy)
    if (set(position_subgroups) != {"unique", "multiple"}
            or sum(row["total"] for row in position_subgroups.values())
            != SAMPLE_COUNT):
        raise ValueError("REC unique/multiple totals do not partition validation")
    for threshold in ("25", "50"):
        if sum(
                row["hits" + threshold]
                for row in position_subgroups.values()
        ) != result["rec_hits0" + threshold]:
            raise ValueError("REC subgroup hits do not partition overall")

    mask_matches = MASK_SUBGROUP_RE.findall(text)
    if len(dict(mask_matches)) != 4:
        raise ValueError("official output lacks exact Mask unique/multiple metrics")
    mask_rates = dict(mask_matches)
    mask_subgroups = {}
    for source_group, result_group in (("unique", "unique"), ("multi", "multiple")):
        total = position_subgroups[result_group]["total"]
        row = {"total": total}
        for threshold in ("25", "50"):
            token = mask_rates[source_group + threshold]
            row["acc" + threshold] = float(token)
            row["hits" + threshold] = _recover_subgroup_hits(
                token, total, "mask {}{}".format(result_group, threshold)
            )
        if row["hits50"] > row["hits25"]:
            raise ValueError("Mask subgroup threshold nesting is invalid")
        mask_subgroups[result_group] = row
    for threshold in ("25", "50"):
        if sum(
                row["hits" + threshold] for row in mask_subgroups.values()
        ) != result["mask_hits0" + threshold]:
            raise ValueError("Mask subgroup hits do not partition overall")
    result["position_subgroups"] = position_subgroups
    result["mask_subgroups"] = mask_subgroups
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
            PARENT_ARTIFACT_PATH, "parent", True
        ),
        "geometry": _snapshot(
            GEOMETRY_ARTIFACT_PATH, "geometry", True
        ),
        "v113": _snapshot(ARTIFACT_PATH, "V113 artifact", True),
        "parity": _snapshot(PARITY_PATH, "V113 parity report", True),
    }
    expected = {
        "backbone": geometry_official.OFFICIAL_CHECKPOINT_SHA256,
        "parent": PARENT_ARTIFACT_SHA256,
        "geometry": GEOMETRY_ARTIFACT_SHA256,
        "v113": ARTIFACT_SHA256,
        "parity": PARITY_SHA256,
    }
    if any(result[name]["sha256"] != sha for name, sha in expected.items()):
        raise ValueError("official artifact SHA binding changed")
    parity = json.loads(PARITY_PATH.read_text(encoding="ascii"))
    if (parity.get("schema") != "rec-v113-train-runtime-parity-audit-v1"
            or parity.get("row_count") != 36665
            or parity.get("all_equal") is not True
            or parity.get("validation_data_accessed") is not False
            or parity.get("weights_modified") is not False
            or parity.get("artifact_sha256") != ARTIFACT_SHA256
            or parity.get("protected_before")
            != parity.get("protected_after")):
        raise ValueError("V113 train/runtime parity contract is invalid")
    model, artifact = load_v113_artifact(
        ARTIFACT_PATH,
        device="cpu",
        expected_artifact_sha256=ARTIFACT_SHA256,
        parent_sha256=expected["parent"],
        geometry_sha256=expected["geometry"],
    )
    if (artifact.get("deployable") is not True
            or artifact.get("validation_data_accessed") is not False
            or artifact.get("oof_evidence", {}).get("sha256")
            != V113_RESULT_SHA256
            or artifact.get("policy") != EXPECTED_POLICY
            or getattr(model, "_artifact_sha256", None) != ARTIFACT_SHA256):
        raise ValueError("V113 deployable artifact contract is invalid")
    return result


def run_official(output_dir, dry_run=False):
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("official output directory already exists")
    if not dry_run and (CLAIM_PATH.exists() or CLAIM_PATH.is_symlink()):
        raise FileExistsError("V113 meshsp validation claim already exists")
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
        raise RuntimeError("protected artifact changed during V113 official run")
    if code_after != code_before:
        raise RuntimeError("Python source tree changed during V113 official run")
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    if returncode != 0:
        raise RuntimeError("V113 official subprocess failed with code {}".format(returncode))
    populations = re.findall(r"length of testing dataset:\s*(\d+)", text)
    if len(populations) != 1 or populations[0] != str(SAMPLE_COUNT):
        raise ValueError("V113 official validation population is not 9508")
    metrics = parse_metrics(text)
    gates = {
        "rec025_at_least_059": metrics["rec_hits025"] >= MIN_REC_HITS025,
        "rec050_at_least_049": metrics["rec_hits050"] >= MIN_REC_HITS050,
        "mask025_preserves_user_baseline": (
            metrics["mask_acc025"] >= MASK_USER_BASELINE_ACC025
        ),
        "mask050_preserves_user_baseline": (
            metrics["mask_acc050"] >= MASK_USER_BASELINE_ACC050
        ),
        "mask_miou_preserves_user_baseline": (
            metrics["mask_miou"] >= MASK_USER_BASELINE_MIOU
        ),
        "mask025_preserves_v99_meshsp": (
            metrics["mask_hits025"] >= MASK_V99_MESHSP_HITS025
        ),
        "mask050_preserves_v99_meshsp": (
            metrics["mask_hits050"] >= MASK_V99_MESHSP_HITS050
        ),
        "mask_miou_preserves_v99_meshsp": (
            metrics["mask_miou"] >= MASK_V99_MESHSP_MIOU
        ),
    }
    timestamp_runs = [
        path for path in (output / "scanrefer" / EXPERIMENT).iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if len(timestamp_runs) != 1:
        raise ValueError("V113 official run directory is ambiguous")
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
        "mask_user_baseline_preservation_pass": gates[
            "mask025_preserves_user_baseline"
        ] and gates["mask050_preserves_user_baseline"] and gates[
            "mask_miou_preserves_user_baseline"
        ],
        "mask_v99_meshsp_preservation_pass": gates[
            "mask025_preserves_v99_meshsp"
        ] and gates["mask050_preserves_v99_meshsp"] and gates[
            "mask_miou_preserves_v99_meshsp"
        ],
        "all_goals_pass": gates["rec025_at_least_059"] and gates[
            "rec050_at_least_049"
        ] and gates["mask025_preserves_user_baseline"] and gates[
            "mask050_preserves_user_baseline"
        ] and gates["mask_miou_preserves_user_baseline"],
        "thresholds": {
            "min_rec_hits025": MIN_REC_HITS025,
            "min_rec_hits050": MIN_REC_HITS050,
            "mask_user_baseline_acc025": MASK_USER_BASELINE_ACC025,
            "mask_user_baseline_acc050": MASK_USER_BASELINE_ACC050,
            "mask_user_baseline_miou": MASK_USER_BASELINE_MIOU,
            "mask_v99_meshsp_hits025": MASK_V99_MESHSP_HITS025,
            "mask_v99_meshsp_hits050": MASK_V99_MESHSP_HITS050,
            "mask_v99_meshsp_miou": MASK_V99_MESHSP_MIOU,
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
