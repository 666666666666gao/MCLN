#!/usr/bin/env python
"""Run the single official validation for a train-gated joint adapter."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_frozen_rec_geometry_official as geometry_official


SAMPLE_COUNT = 9508
RESULT_SCHEMA = "rec-joint-box-mask-official-validation-result-v1"
JOINT_CLAIM_PATH = Path(
    "/root/autodl-tmp/DATA_ROOT/output/scanrefer_joint_box_mask/"
    "epoch71_joint_box_mask_official_validation_once.claim.json"
)
POSITION_RE = re.compile(
    r"last_ position alignment Acc0\.(25|50): Top-1: "
    r"((?:0|1)\.[0-9]{5})"
)
MASK_RE = {
    "mask_acc025": re.compile(r"overall25\s+([01]?\.[0-9]+)"),
    "mask_acc050": re.compile(r"overall50\s+([01]?\.[0-9]+)"),
    "mask_miou": re.compile(r"mask_sem\s+([01]?\.[0-9]+)"),
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path, label, require_readonly=False):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("{} is not a regular file: {}".format(label, path))
    mode = stat.S_IMODE(path.stat().st_mode)
    if require_readonly and mode != 0o444:
        raise ValueError("{} must have mode 0444: {}".format(label, path))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size": int(path.stat().st_size),
        "mode": mode,
    }


def _canonical_json_bytes(value):
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("utf-8")


def _write_exclusive_json(path, value):
    """Create an immutable claim without allowing a second official run."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags, 0o444)
    except FileExistsError:
        raise FileExistsError("official validation claim already exists: {}".format(path))
    payload = _canonical_json_bytes(value)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating official claim")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(str(path), 0o444)
    return path


def _utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _replace_option(command, option, value):
    command = list(command)
    try:
        index = command.index(option)
    except ValueError:
        raise ValueError("authoritative command is missing {}".format(option))
    if index + 1 >= len(command):
        raise ValueError("authoritative command has no value for {}".format(option))
    command[index + 1] = str(value)
    return command


def build_authoritative_command(adapter_path, output_dir, master_port=29672):
    """Derive the approved geometry command and add the joint adapter flags."""
    command = geometry_official.build_authoritative_command()
    command = _replace_option(command, "--master_port", master_port)
    command = _replace_option(command, "--log_dir", output_dir)
    command = _replace_option(command, "--exp", "epoch71_joint_box_mask_official")
    command.extend([
        "--rec_joint_box_mask_checkpoint", str(Path(adapter_path).resolve()),
        "--eval_use_rec_joint_box_mask",
    ])
    return command


def _parse_metrics(text):
    positions = {}
    for threshold, token in POSITION_RE.findall(text):
        key = "position_acc0{}".format(threshold)
        if key in positions:
            raise ValueError("official output contains duplicate {}".format(key))
        positions[key] = token
    if set(positions) != {"position_acc025", "position_acc050"}:
        raise ValueError("official output is missing position metrics")
    result = dict(positions)
    for key, pattern in MASK_RE.items():
        matches = pattern.findall(text)
        if len(matches) != 1:
            raise ValueError("official output is missing or duplicating {}".format(key))
        result[key] = float(matches[0])
        if not 0.0 <= result[key] <= 1.0:
            raise ValueError("official {} is outside [0,1]".format(key))
    result["mask_hits025"] = _recover_rate_hits(
        MASK_RE["mask_acc025"].findall(text)[0], SAMPLE_COUNT
    )
    result["mask_hits050"] = _recover_rate_hits(
        MASK_RE["mask_acc050"].findall(text)[0], SAMPLE_COUNT
    )
    result["position_hits025"] = geometry_official.recover_exact_hits(
        result["position_acc025"], SAMPLE_COUNT
    )
    result["position_hits050"] = geometry_official.recover_exact_hits(
        result["position_acc050"], SAMPLE_COUNT
    )
    if result["position_hits050"] > result["position_hits025"]:
        raise ValueError("position @0.50 hits cannot exceed @0.25 hits")
    if result["mask_hits050"] > result["mask_hits025"]:
        raise ValueError("mask @0.50 hits cannot exceed @0.25 hits")
    for key in ("mask_hits025", "mask_hits050"):
        if not 0 <= result[key] <= SAMPLE_COUNT:
            raise ValueError("official {} is outside the sample population".format(key))
    result["position_acc025_float"] = result["position_hits025"] / float(SAMPLE_COUNT)
    result["position_acc050_float"] = result["position_hits050"] / float(SAMPLE_COUNT)
    return result


def _recover_rate_hits(token, sample_count):
    """Recover a count only when the printed decimal identifies it."""
    if not isinstance(token, str) or not re.fullmatch(
            r"(?:0|1)?\.[0-9]+", token):
        raise ValueError("metric rate token is invalid")
    decimals = len(token.split(".", 1)[1])
    value = float(token)
    candidate = int(round(value * float(sample_count)))
    tolerance = 0.5 * (10.0 ** (-decimals)) + 1e-12
    if (candidate < 0 or candidate > sample_count
            or abs(value - candidate / float(sample_count)) > tolerance):
        raise ValueError("metric rate does not identify an integer hit count")
    return candidate


def _validate_authoritative_command(command):
    expected = build_authoritative_command(
        command[command.index("--rec_joint_box_mask_checkpoint") + 1],
        command[command.index("--log_dir") + 1],
        master_port=int(command[command.index("--master_port") + 1]),
    )
    if list(command) != expected:
        raise ValueError("official command is not the authoritative joint command")
    forbidden = {
        "--eval_train", "--eval_use_ground_truth", "--use_gt_masks",
        "--use_gt_boxes", "--gt_masks", "--gt_boxes",
    }
    if any(value in forbidden for value in command):
        raise ValueError("official command contains a ground-truth inference flag")
    return list(command)


def _source_snapshot():
    files = (
        "main_utils.py",
        "train_dist_mod.py",
        "src/grounding_evaluator.py",
        "models/rec_joint_box_mask.py",
        "scripts/train_scanrefer_joint_box_mask.py",
        "scripts/run_frozen_rec_joint_box_mask_official.py",
        "scripts/run_frozen_rec_geometry_official.py",
    )
    return {
        name: _snapshot(ROOT / name, "source " + name)
        for name in files
    }


def _validate_protected_shas(protected):
    expected = {
        "backbone": geometry_official.OFFICIAL_CHECKPOINT_SHA256,
        "parent": geometry_official.OFFICIAL_PARENT_ARTIFACT_SHA256,
        "geometry": geometry_official.OFFICIAL_SELECTED_ARTIFACT_SHA256,
    }
    for name, sha256 in expected.items():
        if protected.get(name, {}).get("sha256") != sha256:
            raise ValueError("authoritative {} SHA mismatch".format(name))
    return protected


def _validate_adapter_artifact(path, snapshot, protected):
    from scripts.train_scanrefer_joint_box_mask import (
        load_joint_adapter_artifact,
    )
    if snapshot.get("mode") != 0o444:
        raise ValueError("joint adapter must have mode 0444 before official run")
    try:
        model, artifact = load_joint_adapter_artifact(path, device="cpu")
    except Exception as error:
        raise ValueError("joint adapter validation failed: {}".format(error))
    if getattr(model, "_artifact_sha256", None) != snapshot.get("sha256"):
        raise ValueError("joint adapter stable SHA validation failed")
    bindings = {
        "backbone_checkpoint_sha256": protected["backbone"]["sha256"],
        "parent_artifact_sha256": protected["parent"]["sha256"],
        "geometry_artifact_sha256": protected["geometry"]["sha256"],
    }
    if any(artifact.get(key) != value for key, value in bindings.items()):
        raise ValueError("joint adapter protected-artifact binding mismatch")
    return {
        "schema": artifact["schema"],
        "selection": artifact["selection"],
        "deployable": artifact["deployable"],
        "validation_data_accessed": artifact["validation_data_accessed"],
        "inference_uses_ground_truth": artifact["inference_uses_ground_truth"],
        "bindings": bindings,
    }


def _run_official_impl(adapter_path, output_dir, dry_run=False):
    adapter = _snapshot(adapter_path, "joint adapter", True)
    protected = {
        "backbone": _snapshot(
            geometry_official.OFFICIAL_CHECKPOINT_PATH, "backbone", True
        ),
        "parent": _snapshot(
            geometry_official.OFFICIAL_PARENT_ARTIFACT_PATH, "parent", True
        ),
        "geometry": _snapshot(
            geometry_official.OFFICIAL_SELECTED_ARTIFACT_PATH, "geometry", True
        ),
    }
    _validate_protected_shas(protected)
    adapter_contract = _validate_adapter_artifact(
        adapter_path, adapter, protected
    )
    sources_before = _source_snapshot()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("official output directory must be new: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    command = build_authoritative_command(adapter_path, output_dir)
    _validate_authoritative_command(command)
    environment = os.environ.copy()
    for key in ("PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        environment.pop(key, None)
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "pointnet2"),
    })
    stdout_path = output_dir / "official_stdout.log"
    returncode = None
    metrics = None
    claim = None
    if not dry_run:
        claim = _write_exclusive_json(JOINT_CLAIM_PATH, {
            "schema": RESULT_SCHEMA + "-claim-v1",
            "goal": "scanrefer_joint_box_mask_official_validation_once",
            "created_at_utc": _utc_now(),
            "command": command,
            "sample_count": SAMPLE_COUNT,
            "inference_uses_ground_truth": False,
        })
    if not dry_run:
        with stdout_path.open("wb") as handle:
            completed = subprocess.run(
                command, cwd=str(ROOT), env=environment,
                stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        returncode = int(completed.returncode)
        text = stdout_path.read_text(errors="replace")
        if returncode == 0:
            metrics = _parse_metrics(text)
            population = re.findall(
                r"length of testing dataset:\s*(\d+)", text
            )
            if len(population) != 1 or population[0] != str(SAMPLE_COUNT):
                raise ValueError("official validation population is not 9508")
            markers = re.findall(
                r"inference_uses_ground_truth\s*=\s*(true|false)",
                text.lower(),
            )
            if markers and (len(markers) != 1 or markers[0] != "false"):
                raise ValueError("official output claims ground-truth inference")
    after = {
        name: _snapshot(value["path"], name, True)
        for name, value in protected.items()
    }
    if after != protected:
        raise RuntimeError("protected artifact changed during official run")
    adapter_after = _snapshot(adapter["path"], "joint adapter", True)
    if adapter_after != adapter:
        raise RuntimeError("joint adapter changed during official run")
    sources_after = _source_snapshot()
    if sources_after != sources_before:
        raise RuntimeError("official runtime sources changed during execution")
    result = {
        "schema": RESULT_SCHEMA,
        "sample_count": SAMPLE_COUNT,
        "dry_run": bool(dry_run),
        "returncode": returncode,
        "created_at_utc": _utc_now(),
        "claim_path": str(claim) if claim is not None else None,
        "acceptance_gate_pass": bool(
            metrics is not None
            and metrics["position_hits025"] >= 5610
            and metrics["position_hits050"] >= 4621
            and metrics["mask_hits025"] >= 5582
            and metrics["mask_hits050"] >= 4821
            and metrics["mask_miou"] > 0.4472
            and returncode == 0
        ) if metrics is not None else False,
        "validation_data_accessed": not bool(dry_run),
        "inference_uses_ground_truth": False,
        "command": command,
        "environment": {
            key: environment.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH")
        },
        "adapter": adapter,
        "adapter_after": adapter_after,
        "adapter_contract": adapter_contract,
        "protected": protected,
        "protected_after": after,
        "protected_unchanged": after == protected,
        "sources": sources_before,
        "sources_after": sources_after,
        "metrics": metrics,
        "stdout_path": str(stdout_path),
        "stdout_sha256": _sha256(stdout_path) if stdout_path.is_file() else None,
    }
    temporary = output_dir / "official_result.json.tmp"
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(output_dir / "official_result.json"))
    if stdout_path.is_file():
        os.chmod(str(stdout_path), 0o444)
    os.chmod(str(output_dir / "official_result.json"), 0o444)
    if not dry_run and returncode != 0:
        raise RuntimeError(
            "official validation subprocess failed with return code {}".format(
                returncode
            )
        )
    if metrics is not None and not result["acceptance_gate_pass"]:
        raise RuntimeError("official joint adapter metrics do not meet the requested gate")
    return result


def _publish_failure_receipt(output_dir, claim_path, error):
    output_dir = Path(output_dir).expanduser().resolve()
    claim_path = Path(claim_path).expanduser().resolve()
    stdout_path = output_dir / "official_stdout.log"
    result_path = output_dir / "official_result.json"
    payload = {
        "schema": RESULT_SCHEMA + "-failure-v1",
        "status": "failure",
        "created_at_utc": _utc_now(),
        "error_type": type(error).__name__,
        "error": str(error),
        "claim": _snapshot(claim_path, "official claim", True),
        "stdout": _snapshot(stdout_path, "official stdout")
        if stdout_path.is_file() else None,
        "result": _snapshot(result_path, "official result")
        if result_path.is_file() else None,
    }
    _write_exclusive_json(output_dir / "official_failure.json", payload)
    if stdout_path.is_file():
        os.chmod(str(stdout_path), 0o444)
    if result_path.is_file():
        os.chmod(str(result_path), 0o444)
    return payload


def run_official(adapter_path, output_dir, dry_run=False):
    claim_path = Path(JOINT_CLAIM_PATH).expanduser().resolve()
    claim_preexisting = claim_path.exists() or claim_path.is_symlink()
    try:
        return _run_official_impl(adapter_path, output_dir, dry_run=dry_run)
    except Exception as error:
        if (not dry_run and not claim_preexisting and claim_path.is_file()
                and Path(output_dir).expanduser().resolve().is_dir()):
            _publish_failure_receipt(output_dir, claim_path, error)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run_official(args.adapter, args.output_dir, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
