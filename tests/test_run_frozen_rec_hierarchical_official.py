import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.run_frozen_rec_hierarchical_official as official


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json(path, value):
    path.write_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii"))


def _snapshot(path):
    metadata = Path(path).stat()
    return {
        "path": str(Path(path).absolute()),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode & 0o777,
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "sha256": _sha256(path),
    }


def _metric_text(hits025=5705, hits050=4469, subgroups=True):
    acc025 = "%.5f" % (hits025 / float(official.OFFICIAL_SAMPLE_COUNT))
    acc050 = "%.5f" % (hits050 / float(official.OFFICIAL_SAMPLE_COUNT))
    lines = [
        "length of testing dataset: 9508",
        "last_ position alignment Acc0.25: Top-1: {}, Top-5: {}".format(
            acc025, acc025
        ),
        "last_ position alignment Acc0.50: Top-1: {}, Top-5: {}".format(
            acc050, acc050
        ),
    ]
    if subgroups:
        unique025 = min(1300, hits025)
        unique050 = min(1000, hits050)
        groups = (
            ("unique", "0.25", unique025, 1419),
            ("multiple", "0.25", hits025 - unique025, 8089),
            ("unique", "0.50", unique050, 1419),
            ("multiple", "0.50", hits050 - unique050, 8089),
        )
        for group, threshold, hits, total in groups:
            lines.append(
                "position subgroup {} Acc{}: hits={}, total={}, "
                "accuracy={:.12f}".format(
                    group, threshold, hits, total, hits / float(total)
                )
            )
    return "\n".join(lines) + "\n"


def _input_case(monkeypatch, tmp_path):
    online_dir = tmp_path / "online"
    online_dir.mkdir()
    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()
    paths = {
        "backbone": frozen_dir / "backbone.pth",
        "parent": frozen_dir / "parent.pth",
        "geometry": frozen_dir / "geometry.pth",
        "staged_hierarchical": frozen_dir / "selected_hierarchical.pth",
        "staged_result_receipt": frozen_dir / "result-receipt.json",
        "source_gate_baseline": frozen_dir / "source-gate.json",
        "hierarchical": online_dir / "deployed_hierarchical_reranker.pth",
        "online_receipt": online_dir / "online-calibration.json",
    }
    for name in (
            "backbone", "parent", "geometry", "staged_hierarchical",
            "source_gate_baseline", "hierarchical"):
        paths[name].write_bytes(name.encode("ascii"))
    staged_receipt = {
        "selected": "staged_hierarchical",
        "deployable": False,
        "validation_data_accessed": False,
        "artifact": {
            "name": paths["staged_hierarchical"].name,
            "sha256": _sha256(paths["staged_hierarchical"]),
        },
        "calibration": {"gate": {"passed": True}},
    }
    _canonical_json(paths["staged_result_receipt"], staged_receipt)
    for path in paths.values():
        if path != paths["online_receipt"]:
            path.chmod(0o444)
    protected = {
        name: _snapshot(paths[name])
        for name in (
            "backbone", "parent", "geometry", "staged_hierarchical",
            "staged_result_receipt", "source_gate_baseline")
    }
    online_record = {
        "schema": "rec-hierarchical-online-calibration-v1",
        "version": 1,
        "sample_count": 3625,
        "gate": {
            "passed": True,
            "failures": [],
            "required_hits025": 3524,
            "required_hits050": 3316,
        },
        "staged_artifact": {
            "path": str(paths["staged_hierarchical"].absolute()),
            "sha256": _sha256(paths["staged_hierarchical"]),
            "deployable": False,
        },
        "deployed_artifact": {
            "path": str(paths["hierarchical"].absolute()),
            "sha256": _sha256(paths["hierarchical"]),
            "deployable": True,
        },
        "provenance": {
            "protected_before": protected,
            "protected_after": copy.deepcopy(protected),
            "staged_result_receipt": {
                "path": str(paths["staged_result_receipt"].absolute()),
                "sha256": _sha256(paths["staged_result_receipt"]),
            },
            "source_gate_baseline_receipt": {
                "path": str(paths["source_gate_baseline"].absolute()),
                "sha256": _sha256(paths["source_gate_baseline"]),
            },
        },
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
    }
    _canonical_json(paths["online_receipt"], online_record)
    paths["online_receipt"].chmod(0o444)
    monkeypatch.setattr(
        official, "AUTHORITATIVE_BACKBONE_SHA256", _sha256(paths["backbone"])
    )
    monkeypatch.setattr(
        official, "AUTHORITATIVE_PARENT_ARTIFACT_SHA256",
        _sha256(paths["parent"]),
    )
    monkeypatch.setattr(
        official, "AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256",
        _sha256(paths["geometry"]),
    )
    return paths, online_record, staged_receipt


def _preflight(monkeypatch, tmp_path):
    paths, online_record, staged_receipt = _input_case(monkeypatch, tmp_path)
    run_root = tmp_path / "official"
    hierarchy = torch.nn.Linear(1, 1)
    hierarchy._artifact_sha256 = _sha256(paths["hierarchical"])
    calls = []

    def hierarchy_loader(path, **kwargs):
        calls.append("hierarchy")
        assert Path(path) == paths["hierarchical"].absolute()
        assert kwargs["expected_artifact_sha256"] == _sha256(
            paths["hierarchical"]
        )
        assert kwargs["expected_deployable"] is True
        assert kwargs["parent_sha256"] == _sha256(paths["parent"])
        assert kwargs["geometry_sha256"] == _sha256(paths["geometry"])
        return hierarchy, {
            "deployable": True,
            "validation_data_accessed": False,
        }

    context = official.preflight_official_inputs(
        paths["online_receipt"],
        run_root,
        online_receipt_loader=lambda _path: online_record,
        online_record_validator=lambda value, **_kwargs: value,
        staged_receipt_loader=lambda _path: staged_receipt,
        staged_receipt_validator=lambda value: value,
        hierarchy_loader=hierarchy_loader,
        code_manifest_builder=lambda _root: {
            "root": str(Path(official.__file__).parents[1]),
            "files": {"train_dist_mod.py": {
                "sha256": "1" * 64,
                "size": 1,
                "identity": [1, 2, 1, 3, 4],
            }},
            "sha256": "2" * 64,
        },
        python_manifest_builder=lambda: {"sha256": "3" * 64},
        environment_builder=lambda _root: {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": "/repo:/repo/pointnet2",
            "PYTHONHOME": None,
            "LD_PRELOAD": None,
            "LD_LIBRARY_PATH": None,
        },
    )
    assert calls == ["hierarchy"]
    return paths, context


def test_authoritative_command_adds_only_hierarchy_runtime_surface(
        monkeypatch, tmp_path):
    paths, context = _preflight(monkeypatch, tmp_path)
    command = context["command"]

    assert command[command.index("--checkpoint_path") + 1] == str(
        paths["backbone"].absolute()
    )
    assert command[command.index("--rec_reranker_checkpoint") + 1] == str(
        paths["parent"].absolute()
    )
    assert command[
        command.index("--rec_geometry_reranker_checkpoint") + 1
    ] == str(paths["geometry"].absolute())
    assert command[
        command.index("--rec_hierarchical_reranker_checkpoint") + 1
    ] == str(paths["hierarchical"].absolute())
    assert "--eval_use_rec_reranker_scores" in command
    assert "--eval_use_rec_geometry_reranker_scores" in command
    assert "--eval_use_rec_hierarchical_reranker_scores" in command
    assert "--eval_use_rec_selective_residual_scores" not in command
    assert "--butd_gt" not in command
    assert "--butd_cls" not in command
    assert command[command.index("--nproc_per_node") + 1] == "1"


def test_preflight_binds_online_cache_receipts_and_exact_four_artifacts(
        monkeypatch, tmp_path):
    paths, context = _preflight(monkeypatch, tmp_path)

    assert set(context["artifacts"]) >= {
        "backbone", "parent", "geometry", "hierarchical",
        "staged_hierarchical", "staged_result_receipt",
        "source_gate_baseline", "online_calibration",
    }
    assert context["artifacts"]["backbone"]["sha256"] == _sha256(
        paths["backbone"]
    )
    assert context["artifacts"]["hierarchical"]["sha256"] == _sha256(
        paths["hierarchical"]
    )
    assert context["claim_path"] == paths["online_receipt"].parent / \
        official.OFFICIAL_CLAIM_NAME
    assert not context["run_root"].exists()
    assert not context["claim_path"].exists()

    paths["hierarchical"].chmod(0o644)
    paths["hierarchical"].write_bytes(b"tampered")
    paths["hierarchical"].chmod(0o444)
    with pytest.raises(ValueError, match="SHA"):
        official.preflight_official_inputs(
            paths["online_receipt"], context["run_root"],
            online_receipt_loader=lambda _path: context["online_record"],
            online_record_validator=lambda value, **_kwargs: value,
            staged_receipt_loader=lambda _path: context["staged_receipt"],
            staged_receipt_validator=lambda value: value,
            hierarchy_loader=lambda *_args, **_kwargs: pytest.fail(
                "tampered hierarchy must fail before deserialization"
            ),
            code_manifest_builder=lambda _root: context["code"],
            python_manifest_builder=lambda: context["python"],
            environment_builder=lambda _root: context["environment"],
        )


def test_metric_parser_recovers_integer_hits_and_reconciles_subgroups():
    text = _metric_text()
    parsed = official.parse_official_evidence(text, text)

    assert parsed["printed_acc025"] == "0.60002"
    assert parsed["printed_acc050"] == "0.47003"
    assert parsed["hits025"] == 5705
    assert parsed["hits050"] == 4469
    assert parsed["position_subgroups"]["unique"]["0.25"] == {
        "hits": 1300, "total": 1419,
    }
    assert sum(
        parsed["position_subgroups"][group]["0.50"]["hits"]
        for group in ("unique", "multiple")
    ) == 4469
    assert official.parse_official_evidence(
        _metric_text(subgroups=False), _metric_text(subgroups=False)
    )["position_subgroups"] is None
    with pytest.raises(ValueError, match="exactly once"):
        official.parse_official_evidence(text + text, text)
    partial = text.replace(
        next(line for line in text.splitlines()
             if "position subgroup multiple Acc0.50" in line) + "\n",
        "",
    )
    with pytest.raises(ValueError, match="subgroup"):
        official.parse_official_evidence(partial, partial)


def test_acceptance_gate_uses_exact_counts_and_no_ground_truth():
    assert official.acceptance_gate_pass(5705, 4469, 9508, False)
    assert not official.acceptance_gate_pass(5704, 4469, 9508, False)
    assert not official.acceptance_gate_pass(5705, 4468, 9508, False)
    assert not official.acceptance_gate_pass(5705, 4469, 9507, False)
    assert not official.acceptance_gate_pass(5705, 4469, 9508, True)


@pytest.mark.parametrize("hits025,hits050,expected", [
    (5705, 4469, True),
    (5705, 4468, False),
])
def test_runner_claims_once_seals_immutable_evidence_and_preserves_inputs(
        monkeypatch, tmp_path, hits025, hits050, expected):
    paths, context = _preflight(monkeypatch, tmp_path)
    run_path = (
        context["run_root"] / official.OFFICIAL_DATASET
        / official.OFFICIAL_EXPERIMENT / "1700000001"
    )
    before = {name: path.read_bytes() for name, path in paths.items()}
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        run_path.mkdir(parents=True)
        config = dict(official.OFFICIAL_CONFIG_VALUES)
        config.update({
            "log_dir": str(run_path),
            "data_root": str(official.OFFICIAL_DATA_ROOT),
            "checkpoint_path": str(paths["backbone"].absolute()),
            "rec_reranker_checkpoint": str(paths["parent"].absolute()),
            "rec_geometry_reranker_checkpoint": str(
                paths["geometry"].absolute()
            ),
            "rec_hierarchical_reranker_checkpoint": str(
                paths["hierarchical"].absolute()
            ),
        })
        (run_path / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        evidence = _metric_text(hits025, hits050)
        (run_path / "log.txt").write_text(evidence, encoding="utf-8")
        kwargs["stdout"].write(evidence.encode("utf-8"))
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    result = official.run_official_evaluation(
        paths["online_receipt"],
        context["run_root"],
        preflight_builder=lambda *_args, **_kwargs: context,
        subprocess_runner=fake_run,
        postflight_verifier=lambda value: value,
        utc_now=lambda: "2026-07-20T00:00:00Z",
    )

    assert len(calls) == 1
    assert result["hits025"] == hits025
    assert result["hits050"] == hits050
    assert result["acceptance_gate_pass"] is expected
    assert result["inference_uses_ground_truth"] is False
    assert result["sample_count"] == 9508
    assert set(result["artifacts"]) >= {
        "backbone", "parent", "geometry", "hierarchical",
    }
    assert {name: path.read_bytes() for name, path in paths.items()} == before
    for path in (
            context["claim_path"], context["run_root"] / "official_stdout.log",
            run_path / "config.json", run_path / "log.txt",
            context["run_root"] / "official_result.json"):
        assert path.stat().st_mode & 0o777 == 0o444
    assert json.loads(
        (context["run_root"] / "official_result.json").read_text(
            encoding="ascii"
        )
    ) == result

    with pytest.raises(FileExistsError, match="claim"):
        official.run_official_evaluation(
            paths["online_receipt"],
            context["run_root"],
            preflight_builder=lambda *_args, **_kwargs: context,
            subprocess_runner=lambda *_args, **_kwargs: pytest.fail(
                "one-shot subprocess cannot run twice"
            ),
            postflight_verifier=lambda value: value,
        )


def test_cli_has_only_receipt_and_fresh_run_root_paths():
    parsed = official.parse_args([
        "--online-calibration-receipt", "/sealed/online-calibration.json",
        "--run-root", "/fresh/official",
    ])
    assert vars(parsed) == {
        "online_calibration_receipt": "/sealed/online-calibration.json",
        "run_root": "/fresh/official",
    }
    for forbidden in (
            "--hits025", "--threshold", "--artifact-sha256",
            "--validation-cache", "--checkpoint-path"):
        with pytest.raises(SystemExit):
            official.parse_args([
                "--online-calibration-receipt",
                "/sealed/online-calibration.json",
                "--run-root", "/fresh/official",
                forbidden, "x",
            ])
