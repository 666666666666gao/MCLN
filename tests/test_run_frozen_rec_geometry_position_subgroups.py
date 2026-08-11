import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_frozen_rec_geometry_official as official
from scripts import run_frozen_rec_geometry_position_subgroups as report


GROUPS = (
    "unique",
    "multiple",
    "easy",
    "hard",
    "view_dependent",
    "view_independent",
)
THRESHOLDS = (0.25, 0.50)


def _exact_counts():
    return {
        ("unique", 0.25): (1000, 1419),
        ("unique", 0.50): (800, 1419),
        ("multiple", 0.25): (4542, 8089),
        ("multiple", 0.50): (3821, 8089),
        ("easy", 0.25): (3600, 6000),
        ("easy", 0.50): (3000, 6000),
        ("hard", 0.25): (1942, 3508),
        ("hard", 0.50): (1621, 3508),
        ("view_dependent", 0.25): (1600, 3000),
        ("view_dependent", 0.50): (1300, 3000),
        ("view_independent", 0.25): (3942, 6508),
        ("view_independent", 0.50): (3321, 6508),
    }


def _subgroup_lines(counts=None):
    counts = _exact_counts() if counts is None else counts
    lines = []
    for threshold in THRESHOLDS:
        for group in GROUPS:
            hits, total = counts[(group, threshold)]
            lines.append(
                "position subgroup {} Acc{:.2f}: "
                "hits={}, total={}, accuracy={:.12f}".format(
                    group, threshold, hits, total, hits / float(total)
                )
            )
    return "\n".join(lines) + "\n"


def _overall_totals():
    return {
        "printed_acc025": "0.58288",
        "printed_acc050": "0.48601",
        "hits025": 5542,
        "hits050": 4621,
    }


def test_parse_position_subgroups_preserves_exact_counts_and_ratios():
    text = _subgroup_lines()

    parsed = report.parse_position_subgroups(text, text)

    assert parsed[("unique", 0.25)] == {
        "hits": 1000,
        "total": 1419,
        "accuracy": 1000 / 1419.0,
        "printed_accuracy": "0.704721634954",
        "five_decimal_accuracy": "0.70472",
    }
    assert parsed[("multiple", 0.50)]["hits"] == 3821
    assert parsed[("multiple", 0.50)]["total"] == 8089
    report.validate_subgroup_reconciliation(parsed, _overall_totals())


@pytest.mark.parametrize("failure", [
    "duplicate",
    "missing",
    "rendering_mismatch",
    "bad_ratio",
    "hits_above_total",
])
def test_parse_position_subgroups_rejects_inexact_evidence(failure):
    log_text = _subgroup_lines()
    stdout_text = log_text
    first_line = log_text.splitlines()[0] + "\n"
    if failure == "duplicate":
        log_text += first_line
        stdout_text += first_line
    elif failure == "missing":
        log_text = log_text.replace(first_line, "")
        stdout_text = stdout_text.replace(first_line, "")
    elif failure == "rendering_mismatch":
        stdout_text = stdout_text.replace(
            "hits=1000, total=1419, accuracy=0.704721634954",
            "hits=999, total=1419, accuracy=0.703312191684",
        )
    elif failure == "bad_ratio":
        log_text = log_text.replace("0.704721634954", "0.700000000000")
        stdout_text = log_text
    else:
        log_text = log_text.replace(
            "hits=1000, total=1419, accuracy=0.704721634954",
            "hits=1420, total=1419, accuracy=1.000704721635",
        )
        stdout_text = log_text

    with pytest.raises(ValueError, match="subgroup"):
        report.parse_position_subgroups(log_text, stdout_text)


@pytest.mark.parametrize("failure", [
    "denominator",
    "hits",
    "threshold_denominator",
])
def test_position_subgroup_reconciliation_rejects_inconsistent_partitions(
        failure):
    counts = _exact_counts()
    if failure == "denominator":
        counts[("multiple", 0.25)] = (4542, 8088)
    elif failure == "hits":
        counts[("multiple", 0.25)] = (4541, 8089)
    else:
        counts[("unique", 0.50)] = (800, 1418)
    parsed = report.parse_position_subgroups(
        _subgroup_lines(counts), _subgroup_lines(counts)
    )

    with pytest.raises(ValueError, match="reconcile"):
        report.validate_subgroup_reconciliation(parsed, _overall_totals())


def _value_after(command, flag):
    index = command.index(flag)
    return command[index + 1]


def test_report_command_changes_only_output_experiment_and_port(tmp_path):
    run_root = tmp_path / "fresh-report"
    expected = official.build_authoritative_command()
    actual = report.build_report_command(run_root)

    assert _value_after(actual, "--log_dir") == str(run_root.resolve())
    assert _value_after(actual, "--exp") == report.REPORT_EXPERIMENT
    assert _value_after(actual, "--master_port") == str(
        report.REPORT_MASTER_PORT
    )

    for flag in ("--log_dir", "--exp", "--master_port"):
        expected[expected.index(flag) + 1] = _value_after(actual, flag)
    assert actual == expected
    assert "--eval_use_rec_reranker_scores" in actual
    assert "--eval_use_rec_geometry_reranker_scores" in actual
    assert "--eval" in actual


def test_report_command_rejects_relative_or_existing_output(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        report.build_report_command(Path("relative-report"))

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        report.build_report_command(existing)


def test_direct_cli_bootstraps_repository_root_without_pythonpath(tmp_path):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-B", str(Path(report.__file__).resolve()), "--help"],
        cwd=str(tmp_path),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "position subgroup report" in completed.stdout.lower()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _overall_lines(hits025=5542, hits050=4621):
    return (
        "length of testing dataset: 9508\n"
        "last_ position alignment Acc0.25: Top-1: {:.5f}, Top-5: 0.60000\n"
        "last_ position alignment Acc0.50: Top-1: {:.5f}, Top-5: 0.50000\n"
    ).format(hits025 / 9508.0, hits050 / 9508.0)


def _counts_for_totals(hits025=5542, hits050=4621):
    counts = _exact_counts()
    delta025 = hits025 - 5542
    delta050 = hits050 - 4621
    for group in ("multiple", "hard", "view_independent"):
        old_hits, total = counts[(group, 0.25)]
        counts[(group, 0.25)] = (old_hits + delta025, total)
        old_hits, total = counts[(group, 0.50)]
        counts[(group, 0.50)] = (old_hits + delta050, total)
    return counts


def _bind_synthetic_runtime(monkeypatch, tmp_path):
    protected = {}
    for label in ("backbone", "parent", "geometry"):
        path = tmp_path / (label + ".pth")
        path.write_bytes((label + " frozen bytes\n").encode("ascii"))
        path.chmod(0o444)
        protected[label] = (path, _sha256(path))
    design = tmp_path / "design.md"
    design.write_text("# Approved design\n", encoding="utf-8")
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "train_dist_mod.py").write_text(
        "# frozen entrypoint\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        report, "REPORT_PROTECTED_ARTIFACTS", protected, raising=False
    )
    monkeypatch.setattr(report, "REPORT_DESIGN_PATH", design, raising=False)
    monkeypatch.setattr(report, "REPORT_CODE_ROOT", code_root, raising=False)
    monkeypatch.setattr(
        report,
        "_snapshot_python",
        lambda: {
            "logical_path": "/synthetic/python",
            "sha256": "1" * 64,
            "size": 1,
            "mode": 0o755,
            "identity": [1, 2, 1, 3, 4],
            "link_identity": [1, 5, 1, 3, 4],
            "link_target": "python3.7",
            "resolved_path": "/synthetic/python3.7",
        },
        raising=False,
    )
    monkeypatch.setattr(
        report,
        "_preflight_frozen_runtime",
        lambda: {
            "selection_uses_validation": False,
            "inference_uses_ground_truth": False,
        },
        raising=False,
    )
    return {
        "protected": protected,
        "design": design,
        "code_root": code_root,
    }


def _install_synthetic_launch(
        monkeypatch, hits025=5542, hits050=4621, returncode=0,
        mutate=None, extra_run=False):
    counts = _counts_for_totals(hits025, hits050)
    rendered = _overall_lines(hits025, hits050) + _subgroup_lines(counts)

    def fake_run(command, **kwargs):
        output = Path(_value_after(command, "--log_dir"))
        run = (
            output / "scanrefer" / report.REPORT_EXPERIMENT / "1700000001"
        )
        run.mkdir(parents=True)
        (run / "config.json").write_text("{}\n", encoding="utf-8")
        (run / "log.txt").write_text(rendered, encoding="utf-8")
        if extra_run:
            (run.parent / "1700000002").mkdir()
        kwargs["stdout"].write(rendered.encode("utf-8"))
        kwargs["stdout"].flush()
        if mutate is not None:
            mutate()
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(report.subprocess, "run", fake_run)


def test_synthetic_launch_seals_canonical_authoritative_report(
        monkeypatch, tmp_path):
    runtime = _bind_synthetic_runtime(monkeypatch, tmp_path)
    _install_synthetic_launch(monkeypatch)
    output = tmp_path / "position-report"

    record = report.run_position_subgroup_report(output)

    report_path = output / report.REPORT_RESULT_NAME
    report_bytes = report_path.read_bytes()
    assert report_bytes == report._canonical_json_bytes(record)
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o444
    assert record["report_only"] is True
    assert record["eligible_for_model_selection"] is False
    assert record["selection_uses_validation"] is False
    assert record["inference_uses_ground_truth"] is False
    assert record["authoritative"] is True
    assert record["sample_count"] == 9508
    assert record["overall"]["hits025"] == 5542
    assert record["overall"]["hits050"] == 4621
    assert record["position_subgroups"]["unique"]["0.25"]["hits"] == 1000
    assert record["position_subgroups"]["multiple"]["0.50"]["hits"] == 3821
    assert record["artifacts_before"] == record["artifacts_after"]
    assert record["code"]["root"] == str(runtime["code_root"].resolve())
    assert record["design"]["sha256"] == _sha256(runtime["design"])
    assert report.validate_position_subgroup_report(record) == record
    evidence_mtime = max(
        Path(record["files"][name]["path"]).stat().st_mtime_ns
        for name in ("stdout", "log", "config")
    )
    assert report_path.stat().st_mtime_ns >= evidence_mtime


def test_synthetic_launch_seals_total_mismatch_as_non_authoritative(
        monkeypatch, tmp_path):
    _bind_synthetic_runtime(monkeypatch, tmp_path)
    _install_synthetic_launch(monkeypatch, hits025=5541, hits050=4620)

    record = report.run_position_subgroup_report(
        tmp_path / "non-authoritative"
    )

    assert record["overall"]["hits025"] == 5541
    assert record["overall"]["hits050"] == 4620
    assert record["authoritative"] is False


def test_synthetic_launch_nonzero_exit_writes_no_completion_report(
        monkeypatch, tmp_path):
    _bind_synthetic_runtime(monkeypatch, tmp_path)
    _install_synthetic_launch(monkeypatch, returncode=7)
    output = tmp_path / "failed"

    with pytest.raises(Exception, match="7"):
        report.run_position_subgroup_report(output)

    assert output.is_dir()
    assert not (output / report.REPORT_RESULT_NAME).exists()


def test_synthetic_launch_rejects_multiple_timestamp_runs(
        monkeypatch, tmp_path):
    _bind_synthetic_runtime(monkeypatch, tmp_path)
    _install_synthetic_launch(monkeypatch, extra_run=True)
    output = tmp_path / "ambiguous"

    with pytest.raises(ValueError, match="exactly one"):
        report.run_position_subgroup_report(output)

    assert not (output / report.REPORT_RESULT_NAME).exists()


@pytest.mark.parametrize("mutation", ["artifact", "source"])
def test_synthetic_launch_rejects_provenance_mutation(
        monkeypatch, tmp_path, mutation):
    runtime = _bind_synthetic_runtime(monkeypatch, tmp_path)
    if mutation == "artifact":
        path = runtime["protected"]["geometry"][0]

        def mutate():
            path.chmod(0o644)
            path.write_bytes(b"changed geometry\n")
    else:
        path = runtime["code_root"] / "train_dist_mod.py"

        def mutate():
            path.write_text("# changed source\n", encoding="utf-8")
    _install_synthetic_launch(monkeypatch, mutate=mutate)
    output = tmp_path / ("mutated-" + mutation)

    with pytest.raises(ValueError, match="changed"):
        report.run_position_subgroup_report(output)

    assert not (output / report.REPORT_RESULT_NAME).exists()


def test_synthetic_launch_never_reuses_output_directory(
        monkeypatch, tmp_path):
    _bind_synthetic_runtime(monkeypatch, tmp_path)
    output = tmp_path / "already-there"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        report.run_position_subgroup_report(output)
