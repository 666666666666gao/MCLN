#!/usr/bin/env python
"""Generate an utterance-only cache for legacy Scene_graph_parse."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import glob
import hashlib
try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata
import inspect
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import spacy
import sng_parser
from sng_parser import database as sng_database
from sng_parser.backends.backend import ParserBackend

from src import scannet_classes
from src.joint_det_dataset import Scene_graph_parse
from src.legacy_scene_graph_cache import (
    ALLOWED_PARSER_INPUTS,
    BUNDLE_SCHEMA,
    FORBIDDEN_PARSER_INPUTS,
    SUPPORTED_TARGET_SELECTIONS,
    legacy_scene_graph_cache_key,
    sha256_file,
)


def _referit3d_path(data_root, filename):
    candidates = [
        os.path.join(data_root, "refer_it_3d", filename),
        os.path.join(data_root, "ReferIt3D", filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.realpath(path)
    raise FileNotFoundError(", ".join(candidates))


def _scanrefer_path(data_root, split):
    filename = "ScanRefer_filtered_{}.json".format(split)
    candidates = [
        os.path.join(data_root, "scanrefer", filename),
        os.path.join(data_root, "ScanRefer", filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.realpath(path)
    raise FileNotFoundError(", ".join(candidates))


def _referit3d_utterances(path, dataset):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        try:
            utterance_index = header.index("utterance")
        except ValueError:
            raise ValueError("{} has no utterance column".format(path))
        for row in reader:
            yield dataset, row[utterance_index]


def _scanrefer_utterances(path):
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    for record in records:
        tokens = record.get("token")
        if not isinstance(tokens, list) or not all(
                isinstance(token, str) for token in tokens):
            raise ValueError("{} has an invalid token field".format(path))
        yield "scanrefer", " ".join(tokens)


def _collect_source_paths(data_root, datasets):
    sources = []
    for dataset in datasets:
        if dataset in ("nr3d", "sr3d", "sr3d+"):
            path = _referit3d_path(data_root, "{}.csv".format(dataset))
            sources.append((dataset, path))
        elif dataset == "scanrefer":
            for split in ("train", "val"):
                path = _scanrefer_path(data_root, split)
                sources.append((dataset, path))
        else:
            raise ValueError("unsupported dataset {}".format(dataset))
    return sources


def _collect_inputs(sources):
    unique = {}
    for dataset, path in sources:
        if dataset in ("nr3d", "sr3d", "sr3d+"):
            stream = _referit3d_utterances(path, dataset)
        else:
            stream = _scanrefer_utterances(path)
        for dataset, utterance in stream:
            key = legacy_scene_graph_cache_key(dataset, utterance)
            value = (dataset, utterance)
            previous = unique.get(key)
            if previous is not None and previous != value:
                raise ValueError("SHA-256 collision in parser inputs")
            unique[key] = value
    return unique


def _model_version():
    try:
        return importlib_metadata.version("en-core-web-sm")
    except importlib_metadata.PackageNotFoundError:
        raise RuntimeError("en-core-web-sm package version is unavailable")


def _snapshot_file(path, role=None):
    path = os.path.realpath(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    record = {
        "path": path,
        "size": size,
        "sha256": digest.hexdigest(),
    }
    if role is not None:
        record["role"] = role
    return record


def _runtime_dependency_paths():
    spacy_backend_module = importlib.import_module(
        "sng_parser.backends.spacy_parser"
    )
    paths = [
        ("generator", os.path.realpath(__file__)),
        ("scene_graph_wrapper", inspect.getsourcefile(Scene_graph_parse)),
        ("sng_dispatcher", inspect.getsourcefile(sng_parser.parse)),
        ("sng_backend", inspect.getsourcefile(spacy_backend_module)),
        ("sng_backend_base", inspect.getsourcefile(ParserBackend)),
        ("sng_database", inspect.getsourcefile(sng_database)),
        ("mapping_full2rio27", os.path.join(
            PROJECT_ROOT, "mapping_full2rio27.json"
        )),
        ("scannet_classes", inspect.getsourcefile(scannet_classes)),
    ]
    for path in sorted(glob.glob(os.path.join(
            PROJECT_ROOT, "sng_parser", "_data", "*.txt"))):
        paths.append(("sng_data:" + os.path.basename(path), path))
    if not any(role.startswith("sng_data:") for role, _ in paths):
        raise RuntimeError("no sng parser data dependencies found")
    return [(role, os.path.realpath(path)) for role, path in paths]


def _assert_snapshots_unchanged(records, label):
    for record in records:
        current = _snapshot_file(
            record["path"], record.get("role")
        )
        if current != record:
            raise RuntimeError(
                "{} changed during generation: {}".format(
                    label, record["path"]
                )
            )


def _publish_no_overwrite(temporary, output):
    os.chmod(temporary, 0o444)
    try:
        os.link(temporary, output)
    except FileExistsError:
        raise FileExistsError("cache bundle already exists: {}".format(output))
    directory_fd = os.open(os.path.dirname(output), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def generate(args):
    output = os.path.realpath(args.output)
    if os.path.exists(output) or os.path.exists(output + ".manifest.json"):
        raise FileExistsError("cache bundle or legacy manifest already exists")
    directory = os.path.dirname(output)
    os.makedirs(directory, exist_ok=True)
    dependency_paths = _runtime_dependency_paths()
    dependencies = [
        _snapshot_file(path, role)
        for role, path in dependency_paths
    ]
    source_specs = _collect_source_paths(
        os.path.realpath(args.data_root), args.datasets
    )
    sources = [
        _snapshot_file(path) for _, path in source_specs
    ]
    unique = _collect_inputs(source_specs)
    _assert_snapshots_unchanged(dependencies, "runtime dependency")
    _assert_snapshots_unchanged(sources, "input source")
    ordered = sorted(unique.items())
    records_handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=directory,
        prefix=".raw-scene-graph-records-", suffix=".tmp", delete=False,
    )
    records_temporary = records_handle.name
    bundle_temporary = None
    counts = Counter()
    records_digest = hashlib.sha256()
    try:
        with records_handle as handle:
            for start in range(0, len(ordered), args.chunk_size):
                chunk = ordered[start:start + args.chunk_size]
                annos = [
                    {"dataset": dataset, "utterance": utterance}
                    for _, (dataset, utterance) in chunk
                ]
                Scene_graph_parse(
                    annos, target_selection=args.target_selection
                )
                for (key, (dataset, source_utterance)), anno in zip(
                        chunk, annos):
                    record = {
                        "key": key,
                        "dataset": dataset,
                        "source_utterance": source_utterance,
                        "parsed_utterance": anno["utterance"],
                        "graph_node": anno["graph_node"],
                        "graph_edge": anno["graph_edge"],
                        "auxi_entity": anno["auxi_entity"],
                    }
                    payload = (json.dumps(
                        record, ensure_ascii=False, sort_keys=True
                    ) + "\n").encode("utf-8")
                    handle.write(payload)
                    records_digest.update(payload)
                    counts[dataset] += 1
            handle.flush()
            os.fsync(handle.fileno())
        _assert_snapshots_unchanged(dependencies, "runtime dependency")
        _assert_snapshots_unchanged(sources, "input source")
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "allowed_parser_inputs": ALLOWED_PARSER_INPUTS,
            "forbidden_parser_inputs": FORBIDDEN_PARSER_INPUTS,
            "gt_assistance": False,
            "manual_review_applied": False,
            "target_selection": args.target_selection,
            "record_count": len(ordered),
            "records_by_dataset": dict(sorted(counts.items())),
            "records_sha256": records_digest.hexdigest(),
            "runtime_dependencies": dependencies,
            "input_sources": sources,
            "generation_command": [sys.executable] + sys.argv,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "spacy_version": spacy.__version__,
            "en_core_web_sm_version": _model_version(),
        }
        bundle_handle = tempfile.NamedTemporaryFile(
            mode="wb", dir=directory,
            prefix=".raw-scene-graph-bundle-", suffix=".tmp", delete=False,
        )
        bundle_temporary = bundle_handle.name
        with bundle_handle as handle:
            handle.write((json.dumps(
                manifest, ensure_ascii=False, sort_keys=True
            ) + "\n").encode("utf-8"))
            with open(records_temporary, "rb") as records_source:
                shutil.copyfileobj(records_source, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_snapshots_unchanged(dependencies, "runtime dependency")
        _assert_snapshots_unchanged(sources, "input source")
        _publish_no_overwrite(bundle_temporary, output)
        print(json.dumps({
            "bundle": output,
            "bundle_sha256": sha256_file(output),
            "records_sha256": manifest["records_sha256"],
            "record_count": manifest["record_count"],
            "records_by_dataset": manifest["records_by_dataset"],
            "target_selection": manifest["target_selection"],
            "spacy_version": manifest["spacy_version"],
            "en_core_web_sm_version": manifest["en_core_web_sm_version"],
        }, indent=2, sort_keys=True))
    finally:
        for path in (records_temporary, bundle_temporary):
            if path and os.path.exists(path):
                os.unlink(path)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--datasets", nargs="+",
        choices=("nr3d", "sr3d", "sr3d+", "scanrefer"), required=True,
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument(
        "--target_selection", required=True,
        choices=tuple(sorted(SUPPORTED_TARGET_SELECTIONS)),
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.chunk_size < 1 or args.chunk_size > 4096:
        raise ValueError("chunk_size must be in [1, 4096]")
    generate(args)


if __name__ == "__main__":
    main()
