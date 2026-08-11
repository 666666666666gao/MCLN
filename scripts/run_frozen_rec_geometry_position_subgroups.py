#!/usr/bin/env python
"""Report position-alignment subgroups for the frozen best REC geometry run."""

import argparse
from datetime import datetime
import json
import os
import re
from pathlib import Path
import stat
import subprocess
import sys
import uuid

if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from scripts import run_frozen_rec_geometry_official as official


REPORT_SAMPLE_COUNT = 9508
REPORT_EXPECTED_HITS025 = 5542
REPORT_EXPECTED_HITS050 = 4621
REPORT_EXPERIMENT = "epoch71_geometry_position_subgroups"
REPORT_MASTER_PORT = 29673
REPORT_SCHEMA = "rec-geometry-position-subgroup-report-v1"
REPORT_VERSION = 1
REPORT_RESULT_NAME = "position_subgroup_report.json"
REPORT_ROOT = Path(
    "/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16/"
    "geometry_position_subgroup_reports"
)
REPORT_GROUPS = (
    "unique",
    "multiple",
    "easy",
    "hard",
    "view_dependent",
    "view_independent",
)
REPORT_THRESHOLDS = (0.25, 0.50)
REPORT_PARTITIONS = (
    ("unique", "multiple"),
    ("easy", "hard"),
    ("view_dependent", "view_independent"),
)
REPORT_CODE_ROOT = Path(os.path.abspath(__file__)).parents[1]
REPORT_DESIGN_PATH = REPORT_CODE_ROOT / (
    "docs/superpowers/specs/"
    "2026-07-20-scanrefer-rec-hierarchical-risk-controlled-reranking-design.md"
)
REPORT_PROTECTED_ARTIFACTS = {
    "backbone": (
        official.OFFICIAL_CHECKPOINT_PATH,
        official.OFFICIAL_CHECKPOINT_SHA256,
    ),
    "parent": (
        official.OFFICIAL_PARENT_ARTIFACT_PATH,
        official.OFFICIAL_PARENT_ARTIFACT_SHA256,
    ),
    "geometry": (
        official.OFFICIAL_SELECTED_ARTIFACT_PATH,
        official.OFFICIAL_SELECTED_ARTIFACT_SHA256,
    ),
}
REPORT_FIELDS = frozenset((
    "schema",
    "version",
    "created_at_utc",
    "report_only",
    "eligible_for_model_selection",
    "selection_uses_validation",
    "inference_uses_ground_truth",
    "authoritative",
    "sample_count",
    "overall",
    "position_subgroups",
    "artifacts_before",
    "artifacts_after",
    "code",
    "python",
    "design",
    "files",
    "run",
    "preflight",
))

_SUBGROUP_PATTERN = re.compile(
    r"position subgroup ({}).*?"
    r"Acc(0\.(?:25|50)): hits=([0-9]+), total=([0-9]+), "
    r"accuracy=((?:0|1)\.[0-9]{{12}})(?=\r?$)".format(
        "|".join(REPORT_GROUPS)
    ),
    re.MULTILINE,
)


def _parse_subgroup_rendering(text, label):
    if not isinstance(text, str):
        raise ValueError("{} subgroup evidence must be text".format(label))
    values = {}
    counts = {}
    for group, threshold_token, hits_token, total_token, accuracy_token in (
            _SUBGROUP_PATTERN.findall(text)):
        threshold = float(threshold_token)
        key = (group, threshold)
        counts[key] = counts.get(key, 0) + 1
        hits = int(hits_token)
        total = int(total_token)
        if total <= 0 or hits < 0 or hits > total:
            raise ValueError("{} subgroup counts are invalid".format(label))
        expected_accuracy = "{:.12f}".format(hits / float(total))
        if accuracy_token != expected_accuracy:
            raise ValueError("{} subgroup accuracy is inconsistent".format(label))
        values[key] = {
            "hits": hits,
            "total": total,
            "accuracy": hits / float(total),
            "printed_accuracy": accuracy_token,
            "five_decimal_accuracy": "{:.5f}".format(hits / float(total)),
        }
    expected = {
        (group, threshold)
        for group in REPORT_GROUPS
        for threshold in REPORT_THRESHOLDS
    }
    if set(values) != expected or any(
            counts.get(key) != 1 for key in expected):
        raise ValueError(
            "{} must contain each position subgroup exactly once".format(
                label
            )
        )
    return values


def parse_position_subgroups(log_text, stdout_text):
    """Parse two independent exact-count renderings of subgroup metrics."""
    log_values = _parse_subgroup_rendering(log_text, "evaluation log")
    stdout_values = _parse_subgroup_rendering(
        stdout_text, "captured stdout"
    )
    if log_values != stdout_values:
        raise ValueError("position subgroup renderings do not match")
    return log_values


def validate_subgroup_reconciliation(subgroups, totals):
    """Require each ScanRefer partition to reproduce the overall counts."""
    expected_keys = {
        (group, threshold)
        for group in REPORT_GROUPS
        for threshold in REPORT_THRESHOLDS
    }
    if not isinstance(subgroups, dict) or set(subgroups) != expected_keys:
        raise ValueError("position subgroup schema does not reconcile")
    if not isinstance(totals, dict):
        raise ValueError("position subgroup totals do not reconcile")
    total_hits = {
        0.25: totals.get("hits025"),
        0.50: totals.get("hits050"),
    }
    if any(type(value) is not int for value in total_hits.values()):
        raise ValueError("position subgroup totals do not reconcile")
    for group in REPORT_GROUPS:
        if (subgroups[(group, 0.25)]["total"]
                != subgroups[(group, 0.50)]["total"]):
            raise ValueError(
                "position subgroup threshold denominators do not reconcile"
            )
        if (subgroups[(group, 0.50)]["hits"]
                > subgroups[(group, 0.25)]["hits"]):
            raise ValueError(
                "position subgroup threshold hits do not reconcile"
            )
    for threshold in REPORT_THRESHOLDS:
        for left, right in REPORT_PARTITIONS:
            left_value = subgroups[(left, threshold)]
            right_value = subgroups[(right, threshold)]
            if left_value["total"] + right_value["total"] != REPORT_SAMPLE_COUNT:
                raise ValueError(
                    "position subgroup denominators do not reconcile"
                )
            if left_value["hits"] + right_value["hits"] != total_hits[threshold]:
                raise ValueError("position subgroup hits do not reconcile")
    return subgroups


def _replace_argument(command, flag, value):
    positions = [
        index for index, token in enumerate(command)
        if token == flag
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError("frozen command has an invalid {} binding".format(flag))
    command[positions[0] + 1] = str(value)


def build_report_command(run_root):
    """Return the official frozen argv with only report bindings changed."""
    root = Path(run_root)
    if not root.is_absolute():
        raise ValueError("position subgroup output path must be absolute")
    if root.exists() or root.is_symlink():
        raise FileExistsError(
            "position subgroup output already exists: {}".format(root)
        )
    resolved = root.resolve()
    command = list(official.build_authoritative_command())
    _replace_argument(command, "--master_port", REPORT_MASTER_PORT)
    _replace_argument(command, "--log_dir", resolved)
    _replace_argument(command, "--exp", REPORT_EXPERIMENT)
    return command


def _canonical_json_bytes(value):
    return official._canonical_json_bytes(value)


def _utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _is_sha256(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _without_bytes(snapshot):
    return {
        key: value for key, value in snapshot.items()
        if key != "bytes" and key != "value"
    }


def _snapshot_protected_artifacts():
    if (not isinstance(REPORT_PROTECTED_ARTIFACTS, dict)
            or set(REPORT_PROTECTED_ARTIFACTS)
            != {"backbone", "parent", "geometry"}):
        raise ValueError("protected artifact bindings are invalid")
    result = {}
    for label in ("backbone", "parent", "geometry"):
        binding = REPORT_PROTECTED_ARTIFACTS[label]
        if (not isinstance(binding, (tuple, list)) or len(binding) != 2
                or not _is_sha256(binding[1])):
            raise ValueError("protected {} binding is invalid".format(label))
        snapshot = official._stable_snapshot(
            binding[0], "protected {} artifact".format(label)
        )
        if snapshot["sha256"] != binding[1]:
            raise ValueError(
                "protected {} changed from approved SHA".format(label)
            )
        if snapshot["mode"] != 0o444:
            raise ValueError(
                "protected {} changed from mode 0444".format(label)
            )
        result[label] = _without_bytes(snapshot)
    return result


def _snapshot_design():
    return _without_bytes(official._stable_snapshot(
        REPORT_DESIGN_PATH, "approved position subgroup design"
    ))


def _snapshot_python():
    return official.snapshot_authoritative_python()


def _preflight_frozen_runtime():
    official._authoritative_input_snapshots()
    return {
        "selection_uses_validation": False,
        "inference_uses_ground_truth": False,
    }


def _prepare_report_root(output_dir):
    if output_dir is None:
        parent = official._reject_symlink_components(
            REPORT_ROOT, "position subgroup report root"
        )
        parent.mkdir(parents=True, exist_ok=True)
        official._reject_symlink_components(
            parent, "position subgroup report root"
        )
        root = parent / ("position_subgroups." + uuid.uuid4().hex)
    else:
        root = Path(output_dir)
        if not root.is_absolute():
            raise ValueError("position subgroup output path must be absolute")
        parent = official._reject_symlink_components(
            root.parent, "position subgroup report parent"
        )
        parent.mkdir(parents=True, exist_ok=True)
        official._reject_symlink_components(
            parent, "position subgroup report parent"
        )
    command = build_report_command(root)
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(
            "position subgroup output already exists: {}".format(root)
        )
    return root.resolve(), command


def _open_exclusive_stdout(path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags, 0o600)
    except FileExistsError:
        raise FileExistsError(
            "position subgroup stdout already exists: {}".format(path)
        )
    return os.fdopen(descriptor, "wb", buffering=0)


def _discover_timestamp_run(run_root):
    parent = run_root / official.OFFICIAL_DATASET / REPORT_EXPERIMENT
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(
            "position subgroup launch must create exactly one timestamp run"
        )
    runs = [
        child for child in parent.iterdir()
        if child.name.isdigit() and child.is_dir() and not child.is_symlink()
    ]
    if len(runs) != 1:
        raise ValueError(
            "position subgroup launch must create exactly one timestamp run"
        )
    return runs[0]


def _decode_snapshot(snapshot, label):
    try:
        return snapshot["bytes"].decode("utf-8")
    except (KeyError, UnicodeDecodeError) as error:
        raise ValueError("{} is not UTF-8 evidence".format(label)) from error


def _nested_subgroups(subgroups):
    return {
        group: {
            "{:.2f}".format(threshold): dict(subgroups[(group, threshold)])
            for threshold in REPORT_THRESHOLDS
        }
        for group in REPORT_GROUPS
    }


def _flatten_subgroups(nested):
    if not isinstance(nested, dict) or set(nested) != set(REPORT_GROUPS):
        raise ValueError("position subgroup report schema is invalid")
    flattened = {}
    for group in REPORT_GROUPS:
        values = nested[group]
        if not isinstance(values, dict) or set(values) != {"0.25", "0.50"}:
            raise ValueError("position subgroup report schema is invalid")
        for threshold in REPORT_THRESHOLDS:
            value = values["{:.2f}".format(threshold)]
            if (not isinstance(value, dict)
                    or set(value) != {
                        "hits", "total", "accuracy", "printed_accuracy",
                        "five_decimal_accuracy",
                    }):
                raise ValueError("position subgroup report schema is invalid")
            hits = value["hits"]
            total = value["total"]
            if (type(hits) is not int or type(total) is not int
                    or total <= 0 or not 0 <= hits <= total):
                raise ValueError("position subgroup report counts are invalid")
            accuracy = hits / float(total)
            if (value["accuracy"] != accuracy
                    or value["printed_accuracy"] != "{:.12f}".format(accuracy)
                    or value["five_decimal_accuracy"]
                    != "{:.5f}".format(accuracy)):
                raise ValueError("position subgroup report accuracy is invalid")
            flattened[(group, threshold)] = dict(value)
    return flattened


def _validate_snapshot_map(snapshots, label):
    if (not isinstance(snapshots, dict)
            or set(snapshots) != {"backbone", "parent", "geometry"}):
        raise ValueError("{} protected snapshots are invalid".format(label))
    for name, snapshot in snapshots.items():
        if (not isinstance(snapshot, dict)
                or not _is_sha256(snapshot.get("sha256"))
                or snapshot.get("mode") != 0o444
                or type(snapshot.get("size")) is not int
                or not isinstance(snapshot.get("path"), str)
                or not isinstance(snapshot.get("identity"), list)
                or len(snapshot["identity"]) != 5):
            raise ValueError(
                "{} {} protected snapshot is invalid".format(label, name)
            )


def validate_position_subgroup_report(record):
    """Validate a sealed report without treating it as selection evidence."""
    if (not isinstance(record, dict) or set(record) != set(REPORT_FIELDS)
            or record.get("schema") != REPORT_SCHEMA
            or record.get("version") != REPORT_VERSION
            or record.get("report_only") is not True
            or record.get("eligible_for_model_selection") is not False
            or record.get("selection_uses_validation") is not False
            or record.get("inference_uses_ground_truth") is not False
            or record.get("sample_count") != REPORT_SAMPLE_COUNT
            or not isinstance(record.get("created_at_utc"), str)):
        raise ValueError("position subgroup report contract is invalid")
    overall = record.get("overall")
    if (not isinstance(overall, dict)
            or set(overall) != {
                "printed_acc025", "printed_acc050", "hits025", "hits050"
            }
            or type(overall["hits025"]) is not int
            or type(overall["hits050"]) is not int
            or not 0 <= overall["hits050"] <= overall["hits025"] <= 9508):
        raise ValueError("position subgroup overall metrics are invalid")
    subgroups = _flatten_subgroups(record.get("position_subgroups"))
    validate_subgroup_reconciliation(subgroups, overall)
    expected_authoritative = (
        overall["hits025"] == REPORT_EXPECTED_HITS025
        and overall["hits050"] == REPORT_EXPECTED_HITS050
    )
    if record.get("authoritative") is not expected_authoritative:
        raise ValueError("position subgroup authority flag is invalid")
    _validate_snapshot_map(record.get("artifacts_before"), "before")
    _validate_snapshot_map(record.get("artifacts_after"), "after")
    if record["artifacts_before"] != record["artifacts_after"]:
        raise ValueError("protected artifacts changed during report run")
    code = record.get("code")
    design = record.get("design")
    if (not isinstance(code, dict) or not _is_sha256(code.get("sha256"))
            or not isinstance(code.get("root"), str)
            or not isinstance(code.get("files"), dict)
            or not isinstance(design, dict)
            or not _is_sha256(design.get("sha256"))):
        raise ValueError("position subgroup source evidence is invalid")
    files = record.get("files")
    if (not isinstance(files, dict)
            or set(files) != {"stdout", "log", "config"}
            or any(not isinstance(value, dict)
                   or not _is_sha256(value.get("sha256"))
                   for value in files.values())):
        raise ValueError("position subgroup file evidence is invalid")
    run = record.get("run")
    if (not isinstance(run, dict)
            or run.get("returncode") != 0
            or not isinstance(run.get("command"), list)
            or not isinstance(run.get("environment"), dict)
            or not isinstance(run.get("output_root"), str)
            or not isinstance(run.get("timestamp_path"), str)):
        raise ValueError("position subgroup run evidence is invalid")
    preflight = record.get("preflight")
    if (not isinstance(preflight, dict)
            or preflight.get("selection_uses_validation") is not False
            or preflight.get("inference_uses_ground_truth") is not False):
        raise ValueError("position subgroup preflight evidence is invalid")
    if not isinstance(record.get("python"), dict):
        raise ValueError("position subgroup interpreter evidence is invalid")
    return record


def run_position_subgroup_report(output_dir=None):
    """Run one fresh report-only evaluation and seal its exact subgroup data."""
    run_root, command = _prepare_report_root(output_dir)
    sys.stdout.write("REPORT_RUN_ROOT={}\n".format(run_root))
    sys.stdout.flush()

    preflight = _preflight_frozen_runtime()
    artifacts_before = _snapshot_protected_artifacts()
    code_before = official.snapshot_code_tree(REPORT_CODE_ROOT)
    python_before = _snapshot_python()
    design_before = _snapshot_design()
    environment = official._authoritative_environment(REPORT_CODE_ROOT)

    stdout_path = run_root / "evaluation_stdout.log"
    with _open_exclusive_stdout(stdout_path) as stdout_handle:
        completed = subprocess.run(
            command,
            cwd=str(Path(REPORT_CODE_ROOT).resolve()),
            env=environment,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stdout_handle.flush()
        os.fsync(stdout_handle.fileno())
    os.chmod(str(stdout_path), 0o444)
    if type(completed.returncode) is not int:
        raise ValueError("position subgroup subprocess return code is invalid")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)

    timestamp_path = _discover_timestamp_run(run_root)
    log_path = timestamp_path / "log.txt"
    config_path = timestamp_path / "config.json"
    stdout_snapshot = official._stable_snapshot(
        stdout_path, "position subgroup stdout"
    )
    log_snapshot = official._stable_snapshot(log_path, "position subgroup log")
    config_snapshot = official._stable_snapshot(
        config_path, "position subgroup config"
    )
    stdout_text = _decode_snapshot(stdout_snapshot, "captured stdout")
    log_text = _decode_snapshot(log_snapshot, "evaluation log")
    overall = official.parse_official_metrics(log_text, stdout_text)
    subgroups = parse_position_subgroups(log_text, stdout_text)
    validate_subgroup_reconciliation(subgroups, overall)

    artifacts_after = _snapshot_protected_artifacts()
    code_after = official.snapshot_code_tree(REPORT_CODE_ROOT)
    python_after = _snapshot_python()
    design_after = _snapshot_design()
    if artifacts_after != artifacts_before:
        raise ValueError("protected artifacts changed during report run")
    if code_after != code_before:
        raise ValueError("runtime code changed during report run")
    if python_after != python_before:
        raise ValueError("authoritative interpreter changed during report run")
    if design_after != design_before:
        raise ValueError("approved design changed during report run")

    critical_environment = {
        key: environment.get(key)
        for key in official._CRITICAL_ENVIRONMENT_KEYS
    }
    record = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "created_at_utc": _utc_now(),
        "report_only": True,
        "eligible_for_model_selection": False,
        "selection_uses_validation": False,
        "inference_uses_ground_truth": False,
        "authoritative": (
            overall["hits025"] == REPORT_EXPECTED_HITS025
            and overall["hits050"] == REPORT_EXPECTED_HITS050
        ),
        "sample_count": REPORT_SAMPLE_COUNT,
        "overall": overall,
        "position_subgroups": _nested_subgroups(subgroups),
        "artifacts_before": artifacts_before,
        "artifacts_after": artifacts_after,
        "code": code_before,
        "python": python_before,
        "design": design_before,
        "files": {
            "stdout": _without_bytes(stdout_snapshot),
            "log": _without_bytes(log_snapshot),
            "config": _without_bytes(config_snapshot),
        },
        "run": {
            "output_root": str(run_root),
            "timestamp_path": str(timestamp_path.resolve()),
            "returncode": completed.returncode,
            "command": list(command),
            "environment": critical_environment,
        },
        "preflight": preflight,
    }
    validate_position_subgroup_report(record)
    result_path = run_root / REPORT_RESULT_NAME
    official._write_exclusive_json(
        result_path, record, "position subgroup report"
    )
    published = official._require_published_json(
        result_path, record, "position subgroup report"
    )["value"]
    validate_position_subgroup_report(published)
    if published != record:
        raise ValueError("position subgroup report changed during publication")
    return published


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run the frozen REC geometry position subgroup report."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv=None):
    args = _build_argument_parser().parse_args(argv)
    record = run_position_subgroup_report(args.output_dir)
    summary = {
        "authoritative": record["authoritative"],
        "output_root": record["run"]["output_root"],
        "report_path": str(
            Path(record["run"]["output_root"]) / REPORT_RESULT_NAME
        ),
        "overall": record["overall"],
        "position_subgroups": {
            group: record["position_subgroups"][group]
            for group in ("unique", "multiple")
        },
    }
    sys.stdout.write(_canonical_json_bytes(summary).decode("utf-8"))
    return 0


__all__ = [
    "REPORT_EXPERIMENT",
    "REPORT_GROUPS",
    "REPORT_MASTER_PORT",
    "REPORT_PARTITIONS",
    "REPORT_RESULT_NAME",
    "REPORT_ROOT",
    "REPORT_SCHEMA",
    "REPORT_SAMPLE_COUNT",
    "REPORT_THRESHOLDS",
    "build_report_command",
    "parse_position_subgroups",
    "run_position_subgroup_report",
    "validate_position_subgroup_report",
    "validate_subgroup_reconciliation",
]


if __name__ == "__main__":
    raise SystemExit(main())
