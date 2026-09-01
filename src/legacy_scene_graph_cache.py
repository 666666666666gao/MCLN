"""Fail-closed cache for legacy Scene_graph_parse outputs.

The cache is deliberately limited to the two parser inputs already used by
MCLN: the dataset name and the raw utterance. It does not expose structured
spans, target labels, object ids, or new model features.
"""

import copy
import hashlib
import io
try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata
import json
import os
import platform
import sys

import spacy


CACHE_SCHEMA = "mcln-legacy-scene-graph-cache-v1"
MANIFEST_SCHEMA = "mcln-legacy-scene-graph-cache-manifest-v1"
BUNDLE_SCHEMA = "mcln-raw-scene-graph-bundle-v1"
ALLOWED_PARSER_INPUTS = ["dataset", "utterance"]
FORBIDDEN_PARSER_INPUTS = [
    "target_id", "object_id", "object_name", "instance_type",
    "target_label", "target_class", "distractor_ids", "anchor_ids",
    "anchors_types", "correct_guess", "mentions_target_class",
    "reference_type", "scene_id", "scene_geometry", "object_boxes",
    "target_box", "target_mask", "gt_box", "gt_boxes", "gt_mask",
    "gt_masks", "gt_labels", "gt_geometry", "unique_multiple",
    "unique", "multiple", "subgroup", "subgroup_labels",
    "position_subgroups", "model_features", "model_logits",
    "model_scores", "model_outputs", "predicted_boxes",
    "predicted_masks", "predicted_scores", "detector_outputs",
    "source_scores", "tokens", "structured_spans", "graph_node",
    "graph_edge", "auxi_entity", "manual_target", "manual_annotations",
]
SUPPORTED_TARGET_SELECTIONS = frozenset({
    "first_object", "conservative_syntax_v1",
})
REQUIRED_DEPENDENCY_ROLES = frozenset({
    "generator", "mapping_full2rio27", "scene_graph_wrapper",
    "scannet_classes", "sng_backend", "sng_database", "sng_dispatcher",
})
RECORD_FIELDS = frozenset({
    "key", "dataset", "source_utterance", "parsed_utterance",
    "graph_node", "graph_edge", "auxi_entity",
})


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def legacy_scene_graph_cache_key(dataset, utterance):
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("legacy cache dataset must be a non-empty string")
    if not isinstance(utterance, str):
        raise ValueError("legacy cache utterance must be a string")
    payload = json.dumps(
        [dataset, utterance], ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_version():
    try:
        return importlib_metadata.version("en-core-web-sm")
    except importlib_metadata.PackageNotFoundError:
        raise ValueError("en-core-web-sm package version is unavailable")


def _validate_sha_record(record, label, require_role):
    if not isinstance(record, dict):
        raise ValueError("{} provenance entry must be a mapping".format(label))
    fields = {"path", "size", "sha256"}
    if require_role:
        fields.add("role")
    if set(record) != fields:
        raise ValueError(
            "{} provenance fields changed: {}".format(label, sorted(record))
        )
    path = record.get("path")
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError("{} provenance path must be absolute".format(label))
    size = record.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("{} provenance size is invalid".format(label))
    expected_sha = record.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("{} provenance SHA-256 is invalid".format(label))
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            actual_size = os.fstat(handle.fileno()).st_size
            if actual_size != size:
                raise ValueError(
                    "{} provenance size drift: expected {}, got {}".format(
                        label, size, actual_size
                    )
                )
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError:
        raise FileNotFoundError(path)
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            "{} provenance SHA-256 drift: expected {}, got {}".format(
                label, expected_sha, actual_sha
            )
        )


def _validate_bundle_manifest(manifest, expected_target_selection):
    if not isinstance(manifest, dict):
        raise ValueError("raw scene-graph bundle manifest must be a mapping")
    required_fields = {
        "schema", "allowed_parser_inputs", "forbidden_parser_inputs",
        "gt_assistance", "manual_review_applied", "target_selection",
        "record_count", "records_by_dataset", "records_sha256",
        "runtime_dependencies", "input_sources", "generation_command",
        "generated_at_utc", "python_version", "platform",
        "spacy_version", "en_core_web_sm_version",
    }
    if set(manifest) != required_fields:
        raise ValueError(
            "raw scene-graph bundle manifest fields changed: {}".format(
                sorted(manifest)
            )
        )
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported raw scene-graph bundle schema")
    if manifest.get("allowed_parser_inputs") != ALLOWED_PARSER_INPUTS:
        raise ValueError("bundle parser inputs must be dataset and utterance")
    if manifest.get("forbidden_parser_inputs") != FORBIDDEN_PARSER_INPUTS:
        raise ValueError("bundle forbidden parser inputs changed")
    if manifest.get("gt_assistance") is not False:
        raise ValueError("bundle must explicitly disable GT assistance")
    if manifest.get("manual_review_applied") is not False:
        raise ValueError("bundle must explicitly disable manual review")
    target_selection = manifest.get("target_selection")
    if target_selection not in SUPPORTED_TARGET_SELECTIONS:
        raise ValueError("bundle target_selection is unsupported")
    if (expected_target_selection is not None
            and target_selection != expected_target_selection):
        raise ValueError(
            "bundle target_selection mismatch: expected {}, got {}".format(
                expected_target_selection, target_selection
            )
        )
    record_count = manifest.get("record_count")
    if (not isinstance(record_count, int) or isinstance(record_count, bool)
            or record_count < 0):
        raise ValueError("bundle record_count is invalid")
    records_by_dataset = manifest.get("records_by_dataset")
    if (not isinstance(records_by_dataset, dict)
            or not records_by_dataset
            or any(not isinstance(key, str) or not key for key in records_by_dataset)
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or value < 0 for value in records_by_dataset.values())
            or sum(records_by_dataset.values()) != record_count):
        raise ValueError("bundle records_by_dataset is invalid")
    records_sha = manifest.get("records_sha256")
    if not isinstance(records_sha, str) or len(records_sha) != 64:
        raise ValueError("bundle records_sha256 is invalid")
    command = manifest.get("generation_command")
    if (not isinstance(command, list) or not command
            or not all(isinstance(value, str) for value in command)):
        raise ValueError("bundle generation_command is invalid")
    if manifest.get("python_version") != platform.python_version():
        raise ValueError("bundle Python version drift")
    if manifest.get("platform") != platform.platform():
        raise ValueError("bundle platform drift")
    if manifest.get("spacy_version") != spacy.__version__:
        raise ValueError("bundle spaCy version drift")
    if manifest.get("en_core_web_sm_version") != _model_version():
        raise ValueError("bundle en_core_web_sm version drift")

    dependencies = manifest.get("runtime_dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("bundle runtime_dependencies is invalid")
    roles = []
    for index, dependency in enumerate(dependencies):
        _validate_sha_record(
            dependency, "runtime dependency {}".format(index), True
        )
        role = dependency.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("runtime dependency role is invalid")
        roles.append(role)
    if len(roles) != len(set(roles)):
        raise ValueError("runtime dependency roles must be unique")
    if not REQUIRED_DEPENDENCY_ROLES.issubset(set(roles)):
        raise ValueError("bundle runtime dependency roles are incomplete")
    if not any(role.startswith("sng_data:") for role in roles):
        raise ValueError("bundle has no sng parser data dependency")

    input_sources = manifest.get("input_sources")
    if not isinstance(input_sources, list) or not input_sources:
        raise ValueError("bundle input_sources is invalid")
    for index, source in enumerate(input_sources):
        _validate_sha_record(source, "input source {}".format(index), False)


def _validate_record(record, line_number):
    if not isinstance(record, dict):
        raise ValueError(
            "legacy cache line {} must contain an object".format(line_number)
        )
    fields = frozenset(record)
    if fields != RECORD_FIELDS:
        raise ValueError(
            "legacy cache line {} fields changed: {}".format(
                line_number, sorted(fields)
            )
        )
    expected_key = legacy_scene_graph_cache_key(
        record["dataset"], record["source_utterance"]
    )
    if record["key"] != expected_key:
        raise ValueError(
            "legacy cache line {} key does not match parser inputs".format(
                line_number
            )
        )
    if not isinstance(record["parsed_utterance"], str):
        raise ValueError("legacy cache parsed_utterance must be a string")
    if not isinstance(record["graph_node"], list):
        raise ValueError("legacy cache graph_node must be a list")
    if not isinstance(record["graph_edge"], list):
        raise ValueError("legacy cache graph_edge must be a list")
    if (record["auxi_entity"] is not None
            and not isinstance(record["auxi_entity"], dict)):
        raise ValueError("legacy cache auxi_entity must be an object or null")


def _load_bundle(handle, first_line, expected_target_selection):
    try:
        manifest = json.loads(first_line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("invalid raw scene-graph bundle manifest") from error
    _validate_bundle_manifest(manifest, expected_target_selection)
    records = {}
    counts = {}
    records_digest = hashlib.sha256()
    line_count = 0
    for line_number, raw_line in enumerate(handle, 2):
        if not raw_line.strip():
            raise ValueError(
                "raw scene-graph bundle has a blank line at {}".format(
                    line_number
                )
            )
        records_digest.update(raw_line)
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(
                "invalid raw scene-graph record at line {}".format(line_number)
            ) from error
        _validate_record(record, line_number)
        key = record["key"]
        if key in records:
            raise ValueError(
                "duplicate raw scene-graph key at line {}".format(line_number)
            )
        records[key] = record
        dataset = record["dataset"]
        counts[dataset] = counts.get(dataset, 0) + 1
        line_count += 1
    if records_digest.hexdigest() != manifest["records_sha256"]:
        raise ValueError("raw scene-graph records SHA-256 mismatch")
    if line_count != manifest["record_count"]:
        raise ValueError("raw scene-graph bundle record count mismatch")
    if counts != manifest["records_by_dataset"]:
        raise ValueError("raw scene-graph records_by_dataset mismatch")
    return records, manifest


def _load_legacy_from_handle(handle, cache_path, expected_target_selection):
    if expected_target_selection is not None:
        raise ValueError(
            "legacy v1 cache cannot prove expected target_selection"
        )
    manifest_path = cache_path + ".manifest.json"
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)
    with open(manifest_path, encoding="utf-8") as manifest_handle:
        manifest = json.load(manifest_handle)
    expected_sha = manifest.get("cache_sha256")
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        raise ValueError("legacy cache SHA-256 mismatch")
    handle.seek(0)
    _validate_manifest_without_cache_hash(manifest)
    records = {}
    line_count = 0
    text_handle = io.TextIOWrapper(handle, encoding="utf-8")
    for line_number, line in enumerate(text_handle, 1):
        if not line.strip():
            raise ValueError(
                "legacy cache contains a blank line at {}".format(line_number)
            )
        record = json.loads(line)
        _validate_record(record, line_number)
        key = record["key"]
        if key in records:
            raise ValueError(
                "duplicate legacy cache key at line {}".format(line_number)
            )
        records[key] = record
        line_count += 1
    if line_count != manifest["record_count"]:
        raise ValueError(
            "legacy cache record count mismatch: manifest={}, file={}".format(
                manifest["record_count"], line_count
            )
        )
    return records, manifest


def _validate_manifest_without_cache_hash(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("legacy cache manifest must be a mapping")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported legacy cache manifest schema")
    if manifest.get("cache_schema") != CACHE_SCHEMA:
        raise ValueError("unsupported legacy cache record schema")
    if manifest.get("allowed_parser_inputs") != ALLOWED_PARSER_INPUTS:
        raise ValueError(
            "legacy cache parser inputs must be exactly dataset and utterance"
        )
    if manifest.get("gt_assistance") is not False:
        raise ValueError("legacy cache must explicitly disable GT assistance")
    if manifest.get("manual_review_applied") is not False:
        raise ValueError("legacy cache must explicitly disable manual review")
    record_count = manifest.get("record_count")
    if (not isinstance(record_count, int) or isinstance(record_count, bool)
            or record_count < 0):
        raise ValueError("legacy cache record_count must be non-negative")


def load_legacy_scene_graph_cache(
        cache_path, expected_target_selection=None,
        expected_bundle_sha256=None):
    if not isinstance(cache_path, str) or not cache_path:
        raise ValueError("legacy cache path must be a non-empty string")
    cache_path = os.path.realpath(cache_path)
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(cache_path)
    with open(cache_path, "rb") as handle:
        bundle_digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            bundle_digest.update(chunk)
        actual_bundle_sha256 = bundle_digest.hexdigest()
        handle.seek(0)
        if expected_bundle_sha256 is not None:
            if (not isinstance(expected_bundle_sha256, str)
                    or len(expected_bundle_sha256) != 64):
                raise ValueError("expected bundle SHA-256 is invalid")
            if actual_bundle_sha256 != expected_bundle_sha256:
                raise ValueError(
                    "raw scene-graph bundle SHA-256 mismatch: expected {}, "
                    "got {}".format(
                        expected_bundle_sha256, actual_bundle_sha256
                    )
                )
        first_line = handle.readline()
        try:
            first_object = json.loads(first_line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            first_object = None
        if (isinstance(first_object, dict)
                and first_object.get("schema") == BUNDLE_SCHEMA):
            if expected_bundle_sha256 is None:
                raise ValueError(
                    "raw scene-graph bundle requires an expected SHA-256"
                )
            return _load_bundle(
                handle, first_line, expected_target_selection
            )
        if expected_bundle_sha256 is not None:
            raise ValueError(
                "expected bundle SHA-256 requires raw bundle schema"
            )
        handle.seek(0)
        return _load_legacy_from_handle(
            handle, cache_path, expected_target_selection
        )


def apply_legacy_scene_graph_cache(anno, records):
    dataset = anno.get("dataset")
    utterance = anno.get("utterance")
    key = legacy_scene_graph_cache_key(dataset, utterance)
    record = records.get(key)
    if record is None:
        return False
    if (record["dataset"] != dataset
            or record["source_utterance"] != utterance):
        raise ValueError("legacy cache key collision")
    anno["utterance"] = record["parsed_utterance"]
    anno["graph_node"] = copy.deepcopy(record["graph_node"])
    anno["graph_edge"] = copy.deepcopy(record["graph_edge"])
    anno["auxi_entity"] = copy.deepcopy(record["auxi_entity"])
    return True
