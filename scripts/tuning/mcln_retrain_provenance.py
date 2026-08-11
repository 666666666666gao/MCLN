#!/usr/bin/env python
"""Provenance, input protection, and source snapshots for MCLN retraining."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


MANIFEST_SCHEMA = "mcln-retrain-source-manifest-v1"
SNAPSHOT_EXCLUDED_DIRECTORY_NAMES = frozenset((
    ".git",
    ".pytest_cache",
    "__pycache__",
))
SNAPSHOT_EXCLUDED_SUFFIXES = frozenset((
    ".pth",
    ".pt",
    ".pyc",
    ".pyo",
))


def sha256_file(path, chunk_size=1024 * 1024):
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_contract(path, sha256, size, mode):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    stat_result = path.stat()
    actual = {
        "path": str(path),
        "size": int(stat_result.st_size),
        "mode": int(stat.S_IMODE(stat_result.st_mode)),
        "sha256": sha256_file(path),
    }
    expected = {
        "size": int(size),
        "mode": int(mode),
        "sha256": str(sha256),
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(
                "file contract mismatch for {}: {}".format(path, key)
            )
    return actual


def atomic_json_dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_dump_no_replace(path, payload, label):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if _load_json(path) != payload:
            raise ValueError("existing {} differs from requested payload".format(
                label
            ))
        return False
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary), str(path))
        except FileExistsError:
            if _load_json(path) != payload:
                raise ValueError(
                    "concurrent {} differs from requested payload".format(
                        label
                    )
                )
            return False
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_publish_no_replace(path, temporary, label):
    path = Path(path)
    temporary = Path(temporary)
    if path.is_file():
        if sha256_file(path) != sha256_file(temporary):
            raise ValueError("existing {} differs from requested artifact".format(
                label
            ))
        return False
    try:
        os.link(str(temporary), str(path))
    except FileExistsError:
        if sha256_file(path) != sha256_file(temporary):
            raise ValueError(
                "concurrent {} differs from requested artifact".format(label)
            )
        return False
    return True


def _is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolved_exclusions(excluded_roots):
    result = []
    for root in excluded_roots:
        result.append(Path(root).resolve())
    return tuple(result)


def _source_file_allowed(path, root, excluded_roots):
    if path.is_symlink() or not path.is_file():
        return False
    relative = path.relative_to(root)
    if any(part in SNAPSHOT_EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
        return False
    if path.suffix.lower() in SNAPSHOT_EXCLUDED_SUFFIXES:
        return False
    resolved = path.resolve()
    if any(_is_within(resolved, excluded) for excluded in excluded_roots):
        return False
    return True


def _manifest_digest_payload(manifest_without_digest):
    return json.dumps(
        manifest_without_digest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def build_source_manifest(root, excluded_roots=()):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("source root does not exist")
    excluded = _resolved_exclusions(excluded_roots)
    entries = []
    for path in sorted(root.rglob("*")):
        if not _source_file_allowed(path, root, excluded):
            continue
        relative = path.relative_to(root).as_posix()
        entries.append({
            "path": relative,
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    payload = {
        "schema": MANIFEST_SCHEMA,
        "file_count": len(entries),
        "total_size": sum(entry["size"] for entry in entries),
        "files": entries,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        _manifest_digest_payload(payload)
    ).hexdigest()
    return payload


def _path_is_explicitly_excluded(path, excluded_roots):
    resolved = Path(path).resolve()
    return any(
        resolved == excluded or _is_within(resolved, excluded)
        for excluded in _resolved_exclusions(excluded_roots)
    )


def _canonical_manifest_bytes(manifest):
    return (
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
    ).encode("ascii")


def _tar_info(name, size, mode=0o444):
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def create_source_snapshot(
        root, archive_path, manifest_path, excluded_roots=()):
    root = Path(root).resolve()
    archive_path = Path(archive_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    for output_path in (archive_path, manifest_path):
        if _is_within(output_path, root) and not _path_is_explicitly_excluded(
                output_path, excluded_roots):
            raise ValueError(
                "snapshot outputs inside source root must be excluded"
            )

    manifest = build_source_manifest(root, excluded_roots=excluded_roots)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(archive_path.name),
        suffix=".tmp",
        dir=str(archive_path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as archive:
                    for entry in manifest["files"]:
                        source = root / entry["path"]
                        if source.is_symlink() or not source.is_file():
                            raise ValueError(
                                "source mutated after manifest creation"
                            )
                        source_bytes = source.read_bytes()
                        post_read = source.stat()
                        if (
                            len(source_bytes) != entry["size"]
                            or int(post_read.st_size) != entry["size"]
                        ):
                            raise ValueError(
                                "source size mutated after manifest creation"
                            )
                        if hashlib.sha256(source_bytes).hexdigest() != entry[
                                "sha256"]:
                            raise ValueError(
                                "source SHA-256 mutated after manifest creation"
                            )
                        info = _tar_info(
                            entry["path"], entry["size"],
                            mode=stat.S_IMODE(post_read.st_mode),
                        )
                        archive.addfile(info, _BytesReader(source_bytes))
                    manifest_bytes = _canonical_manifest_bytes(manifest)
                    archive.addfile(
                        _tar_info(
                            "SOURCE_MANIFEST.json", len(manifest_bytes)
                        ),
                        _BytesReader(manifest_bytes),
                    )
        if manifest_path.is_file() and _load_json(manifest_path) != manifest:
            raise ValueError(
                "existing source manifest differs from requested payload"
            )
        if (
            archive_path.is_file()
            and sha256_file(archive_path) != sha256_file(temporary)
        ):
            raise ValueError(
                "existing source snapshot differs from requested artifact"
            )
        _file_publish_no_replace(
            archive_path, temporary, "source snapshot"
        )
        _json_dump_no_replace(
            manifest_path, manifest, "source manifest"
        )
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "file_count": manifest["file_count"],
        "total_size": manifest["total_size"],
    }


class _BytesReader:
    def __init__(self, payload):
        self.payload = payload
        self.offset = 0

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self.payload) - self.offset
        start = self.offset
        end = min(len(self.payload), start + size)
        self.offset = end
        return self.payload[start:end]


def _run_capture(command):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"returncode": None, "output": str(error)}
    return {
        "returncode": result.returncode,
        "output": result.stdout,
        "stderr": result.stderr,
    }


def capture_environment(python_bin=sys.executable):
    capture_script = r'''
import json
import os
import platform
import sys


def module_version(name):
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception as error:
        return "unavailable: {}".format(error)


try:
    import torch
    torch_record = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
except Exception as error:
    torch_record = {"error": str(error)}

try:
    import pkg_resources
    packages = sorted(
        "{}=={}".format(item.project_name, item.version)
        for item in pkg_resources.working_set
    )
except Exception as error:
    packages = ["unavailable: {}".format(error)]

payload = {
    "python_executable": os.path.realpath(sys.executable),
    "python_version": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "numpy": module_version("numpy"),
    "optuna": module_version("optuna"),
    "torch": torch_record,
    "packages": packages,
    "environment": {
        key: os.environ.get(key)
        for key in (
            "CONDA_PREFIX",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "PYTHONPATH",
            "TOKENIZERS_PARALLELISM",
        )
    },
}
print(json.dumps(payload, sort_keys=True, allow_nan=False))
'''.strip()
    python = _run_capture([python_bin, "-c", capture_script])
    if python.get("returncode") != 0:
        raise RuntimeError(
            "environment capture through requested python failed: {}".format(
                python.get("stderr") or python.get("output")
            )
        )
    try:
        payload = json.loads(python.get("output", ""))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "requested python emitted invalid environment JSON: {}".format(
                error
            )
        )
    required_fields = {
        "python_executable",
        "python_version",
        "platform",
        "machine",
        "numpy",
        "optuna",
        "torch",
        "packages",
        "environment",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError("requested python environment schema is invalid")
    for field in (
            "python_executable", "python_version", "platform", "machine",
            "numpy", "optuna"):
        if not isinstance(payload[field], str):
            raise ValueError(
                "requested python environment {} is invalid".format(field)
            )
    if not isinstance(payload["torch"], dict):
        raise ValueError("requested python torch environment is invalid")
    if (
        not isinstance(payload["packages"], list)
        or any(not isinstance(item, str) for item in payload["packages"])
    ):
        raise ValueError("requested python package environment is invalid")
    if not isinstance(payload["environment"], dict):
        raise ValueError("requested python process environment is invalid")
    requested_python = str(Path(python_bin).resolve())
    reported_python = str(Path(payload["python_executable"]).resolve())
    if reported_python != requested_python:
        raise ValueError("requested python executable identity differs")

    nvidia = _run_capture([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    result = dict(payload)
    result["nvidia_smi"] = nvidia
    return result


def prepare_run(args):
    output_root = Path(args.output_root).resolve()
    provenance_root = output_root / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)
    base = verify_file_contract(
        args.base_checkpoint,
        sha256=args.base_sha256,
        size=args.base_size,
        mode=args.base_mode,
    )
    pp_path = Path(args.pp_checkpoint).resolve()
    if not pp_path.is_file():
        raise FileNotFoundError(str(pp_path))
    pp = {
        "path": str(pp_path),
        "size": int(pp_path.stat().st_size),
        "mode": int(stat.S_IMODE(pp_path.stat().st_mode)),
        "sha256": sha256_file(pp_path),
    }
    snapshot = create_source_snapshot(
        args.repo_root,
        provenance_root / "source_snapshot.tar.gz",
        provenance_root / "source_manifest.json",
        excluded_roots=[output_root],
    )
    environment_path = provenance_root / "environment.json"
    environment = capture_environment(args.python_bin)
    _json_dump_no_replace(
        environment_path, environment, "environment receipt"
    )
    inputs = {
        "schema": "mcln-retrain-inputs-v1",
        "data_root": str(Path(args.data_root).resolve()),
        "base_checkpoint": base,
        "pointnet_checkpoint": pp,
    }
    inputs_path = provenance_root / "inputs.json"
    _json_dump_no_replace(inputs_path, inputs, "input receipt")
    receipt = {
        "schema": "mcln-retrain-run-provenance-v1",
        "repo_root": str(Path(args.repo_root).resolve()),
        "output_root": str(output_root),
        "data_root": str(Path(args.data_root).resolve()),
        "base_checkpoint": base,
        "pointnet_checkpoint": pp,
        "source_snapshot": snapshot,
        "environment": str(environment_path),
        "environment_sha256": sha256_file(environment_path),
        "inputs": str(inputs_path),
        "inputs_sha256": sha256_file(inputs_path),
    }
    _json_dump_no_replace(
        provenance_root / "run_manifest.json",
        receipt,
        "run provenance manifest",
    )
    return receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--base-size", required=True, type=int)
    parser.add_argument("--base-mode", default="0444")
    parser.add_argument("--pp-checkpoint", required=True)
    parser.add_argument("--python-bin", required=True)
    args = parser.parse_args(argv)
    try:
        args.base_mode = int(args.base_mode, 8)
    except ValueError:
        raise ValueError("base mode must be an octal integer")
    return args


def main():
    receipt = prepare_run(parse_args())
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
