import hashlib
import json
import math
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.tuning import mcln_retrain_provenance as provenance


def test_sha256_file_streams_exact_content(tmp_path):
    path = tmp_path / "artifact.bin"
    content = (b"mcln-provenance" * 1000) + b"tail"
    path.write_bytes(content)

    assert provenance.sha256_file(path, chunk_size=17) == (
        hashlib.sha256(content).hexdigest()
    )


def test_verify_file_contract_requires_exact_size_mode_and_hash(tmp_path):
    path = tmp_path / "epoch54.pth"
    path.write_bytes(b"protected checkpoint")
    path.chmod(0o444)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    actual = provenance.verify_file_contract(
        path, sha256=digest, size=20, mode=0o444
    )

    assert actual == {
        "path": str(path.resolve()),
        "size": 20,
        "mode": 0o444,
        "sha256": digest,
    }

    for overrides in (
        {"sha256": "0" * 64},
        {"size": 19},
        {"mode": 0o644},
    ):
        expected = {"sha256": digest, "size": 20, "mode": 0o444}
        expected.update(overrides)
        with pytest.raises(ValueError):
            provenance.verify_file_contract(path, **expected)


def test_atomic_json_dump_is_sorted_and_rejects_nan(tmp_path):
    path = tmp_path / "receipt.json"
    provenance.atomic_json_dump(path, {"z": 1, "a": 2})

    assert path.read_text() == '{\n  "a": 2,\n  "z": 1\n}\n'

    before = path.read_bytes()
    with pytest.raises(ValueError):
        provenance.atomic_json_dump(path, {"bad": math.nan})
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp.*")) == []


def _make_source_tree(root):
    (root / "models").mkdir(parents=True)
    (root / "models" / "model.py").write_text("MODEL = True\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "run.sh").write_text("#!/bin/sh\n")
    (root / "docs").mkdir()
    (root / "docs" / "note.md").write_text("reproducible\n")
    (root / "weights.pth").write_bytes(b"large")
    (root / "adapter.pt").write_bytes(b"large")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "model.pyc").write_bytes(b"cache")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache" / "state").write_text("cache")
    (root / "output").mkdir()
    (root / "output" / "trial.json").write_text("generated")


def test_source_manifest_is_sorted_and_excludes_weights_caches_and_outputs(
        tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _make_source_tree(root)

    manifest = provenance.build_source_manifest(
        root, excluded_roots=[root / "output"]
    )

    paths = [entry["path"] for entry in manifest["files"]]
    assert paths == sorted(paths)
    assert paths == [
        "docs/note.md", "models/model.py", "scripts/run.sh"
    ]
    assert manifest["file_count"] == 3
    assert manifest["total_size"] == sum(
        (root / path).stat().st_size for path in paths
    )
    assert len(manifest["manifest_sha256"]) == 64


def test_source_snapshot_contains_manifested_files_and_canonical_manifest(
        tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _make_source_tree(root)
    output = tmp_path / "artifacts"
    archive = output / "source.tar.gz"
    manifest_path = output / "source_manifest.json"

    result = provenance.create_source_snapshot(
        root,
        archive,
        manifest_path,
        excluded_roots=[root / "output", output],
    )

    assert result["archive"] == str(archive.resolve())
    assert result["manifest"] == str(manifest_path.resolve())
    assert result["manifest_sha256"] == json.loads(
        manifest_path.read_text()
    )["manifest_sha256"]
    with tarfile.open(archive, "r:gz") as handle:
        names = sorted(handle.getnames())
    assert names == [
        "SOURCE_MANIFEST.json",
        "docs/note.md",
        "models/model.py",
        "scripts/run.sh",
    ]


def test_snapshot_rejects_output_inside_source_without_exclusion(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "model.py").write_text("x = 1\n")

    with pytest.raises(ValueError):
        provenance.create_source_snapshot(
            root,
            root / "snapshot.tar.gz",
            root / "manifest.json",
        )


def test_snapshot_source_mutation_publishes_neither_manifest_nor_archive(
        tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "model.py"
    source.write_text("VALUE = 1\n")
    archive = tmp_path / "snapshot.tar.gz"
    manifest_path = tmp_path / "manifest.json"
    real_builder = provenance.build_source_manifest

    def mutate_after_manifest(*args, **kwargs):
        manifest = real_builder(*args, **kwargs)
        source.write_text("VALUE = 2\n")
        return manifest

    monkeypatch.setattr(
        provenance, "build_source_manifest", mutate_after_manifest
    )

    with pytest.raises(ValueError, match="mutat|SHA-256|size"):
        provenance.create_source_snapshot(root, archive, manifest_path)
    assert not archive.exists()
    assert not manifest_path.exists()


def test_prepare_run_resume_rejects_changed_source_without_replacing_artifacts(
        tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "model.py").write_text("VERSION = 1\n")
    base = tmp_path / "epoch54.pth"
    base.write_bytes(b"base checkpoint")
    base.chmod(0o444)
    pointnet = tmp_path / "pointnet.pth"
    pointnet.write_bytes(b"pointnet checkpoint")
    args = SimpleNamespace(
        output_root=str(tmp_path / "study"),
        repo_root=str(root),
        data_root=str(tmp_path / "data"),
        base_checkpoint=str(base),
        base_sha256=provenance.sha256_file(base),
        base_size=base.stat().st_size,
        base_mode=0o444,
        pp_checkpoint=str(pointnet),
        python_bin="/usr/bin/python3",
    )
    monkeypatch.setattr(
        provenance,
        "capture_environment",
        lambda _python_bin: {"environment": "stable"},
    )

    provenance.prepare_run(args)
    provenance_root = Path(args.output_root) / "provenance"
    protected = {
        path.name: path.read_bytes()
        for path in provenance_root.iterdir()
        if path.is_file()
    }

    (root / "model.py").write_text("VERSION = 2\n")
    with pytest.raises(ValueError, match="source|provenance"):
        provenance.prepare_run(args)

    assert {
        path.name: path.read_bytes()
        for path in provenance_root.iterdir()
        if path.is_file()
    } == protected


def test_prepare_run_recaptures_same_stable_environment_on_resume(
        tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "model.py").write_text("VERSION = 1\n")
    base = tmp_path / "epoch54.pth"
    base.write_bytes(b"base checkpoint")
    base.chmod(0o444)
    pointnet = tmp_path / "pointnet.pth"
    pointnet.write_bytes(b"pointnet checkpoint")
    args = SimpleNamespace(
        output_root=str(tmp_path / "study"),
        repo_root=str(root),
        data_root=str(tmp_path / "data"),
        base_checkpoint=str(base),
        base_sha256=provenance.sha256_file(base),
        base_size=base.stat().st_size,
        base_mode=0o444,
        pp_checkpoint=str(pointnet),
        python_bin="/usr/bin/python3",
    )
    captures = []

    def stable_capture(_python_bin):
        captures.append(True)
        return {"environment": "stable"}

    monkeypatch.setattr(provenance, "capture_environment", stable_capture)

    first = provenance.prepare_run(args)
    second = provenance.prepare_run(args)

    assert first == second
    assert len(captures) == 2


def test_prepare_run_rejects_environment_drift_without_replacing_artifacts(
        tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "model.py").write_text("VERSION = 1\n")
    base = tmp_path / "epoch54.pth"
    base.write_bytes(b"base checkpoint")
    base.chmod(0o444)
    pointnet = tmp_path / "pointnet.pth"
    pointnet.write_bytes(b"pointnet checkpoint")
    args = SimpleNamespace(
        output_root=str(tmp_path / "study"),
        repo_root=str(root),
        data_root=str(tmp_path / "data"),
        base_checkpoint=str(base),
        base_sha256=provenance.sha256_file(base),
        base_size=base.stat().st_size,
        base_mode=0o444,
        pp_checkpoint=str(pointnet),
        python_bin="/usr/bin/python3",
    )
    captures = iter((
        {"python": "3.7", "driver": "stable"},
        {"python": "3.7", "driver": "changed"},
    ))
    monkeypatch.setattr(
        provenance,
        "capture_environment",
        lambda _python_bin: next(captures),
    )

    provenance.prepare_run(args)
    provenance_root = Path(args.output_root) / "provenance"
    protected = {
        path.name: path.read_bytes()
        for path in provenance_root.iterdir()
        if path.is_file()
    }

    with pytest.raises(ValueError, match="environment"):
        provenance.prepare_run(args)

    assert {
        path.name: path.read_bytes()
        for path in provenance_root.iterdir()
        if path.is_file()
    } == protected


def test_capture_environment_queries_only_stable_gpu_identity_fields(
        monkeypatch):
    commands = []
    python_bin = "/usr/bin/python3"
    python_payload = {
        "python_executable": str(Path(python_bin).resolve()),
        "python_version": "3.7.fake",
        "platform": "fake-platform",
        "machine": "x86_64",
        "numpy": "1.0",
        "optuna": "4.0",
        "torch": {
            "version": "1.10",
            "cuda": "11.1",
            "cudnn": 8000,
            "cuda_available": True,
        },
        "packages": ["numpy==1.0", "torch==1.10"],
        "environment": {"PYTHONPATH": "/workspace"},
    }

    def fake_capture(command):
        commands.append(command)
        if command[0] == "nvidia-smi":
            return {"returncode": 0, "output": "stable-gpu\n"}
        return {"returncode": 0, "output": json.dumps(python_payload)}

    monkeypatch.setattr(provenance, "_run_capture", fake_capture)

    result = provenance.capture_environment(python_bin)

    nvidia_command = next(command for command in commands if command[0] == "nvidia-smi")
    assert "-q" not in nvidia_command
    query = next(value for value in nvidia_command if value.startswith("--query-gpu="))
    for field in (
            "uuid", "name", "driver_version", "memory.total", "compute_cap"):
        assert field in query
    python_commands = [command for command in commands if command[0] == python_bin]
    assert len(python_commands) == 1
    assert result["python_version"] == "3.7.fake"
    assert result["packages"] == ["numpy==1.0", "torch==1.10"]
    assert result["nvidia_smi"]["output"] == "stable-gpu\n"


def test_capture_environment_rejects_wrong_python_identity(monkeypatch):
    requested = "/opt/requested/python"
    payload = {
        "python_executable": "/opt/other/python",
        "python_version": "3.7.fake",
        "platform": "fake",
        "machine": "x86_64",
        "numpy": "1",
        "optuna": "4",
        "torch": {},
        "packages": [],
        "environment": {},
    }

    def fake_capture(command):
        if command[0] == "nvidia-smi":
            return {"returncode": 0, "output": "gpu\n"}
        return {"returncode": 0, "output": json.dumps(payload)}

    monkeypatch.setattr(provenance, "_run_capture", fake_capture)

    with pytest.raises(ValueError, match="python executable|identity"):
        provenance.capture_environment(requested)
