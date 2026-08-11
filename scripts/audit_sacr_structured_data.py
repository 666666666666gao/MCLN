#!/usr/bin/env python3
"""Audit SACR sidecar coverage and post-normalization token alignment."""

import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys

import torch
from transformers import RobertaTokenizerFast


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.structured_source import build_token_span_tensors
from src.joint_det_dataset import Joint3DDataset, Scene_graph_parse
from src.structured_annotations import (
    build_scanrefer_structured_lookup,
    build_structured_annotation,
    realign_structured_annotation,
    scanrefer_annotation_key,
)


EXPECTED_SPLIT_COUNTS = {
    "train": {
        "scanrefer": 36665,
        "nr3d": 32919,
        "sr3d": 65846,
    },
    "val": {
        "scanrefer": 9508,
        "nr3d": 7899,
        "sr3d": 17726,
    },
}


def _atomic_write_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _read_scan_ids(path):
    path = Path(path)
    if path.suffix == ".txt" and path.name.startswith("ScanRefer"):
        return {
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    value = ast.literal_eval(path.read_text(encoding="utf-8"))
    if not isinstance(value, (list, set, tuple)):
        raise ValueError("split file must contain a sequence of scan ids")
    return set(value)


def _scanrefer_directory(data_root):
    for name in ("scanrefer", "ScanRefer"):
        candidate = Path(data_root) / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("ScanRefer directory is unavailable")


def _referit3d_directory(data_root):
    for name in ("refer_it_3d", "ReferIt3D"):
        candidate = Path(data_root) / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("ReferIt3D directory is unavailable")


def _scanrefer_annotations(data_root, split):
    data_dir = _scanrefer_directory(data_root)
    split_name = "val" if split in ("val", "test") else split
    scan_ids = _read_scan_ids(
        data_dir / "ScanRefer_filtered_{}.txt".format(split_name)
    )
    base_path = data_dir / "ScanRefer_filtered_{}.json".format(split_name)
    sidecar_path = data_dir / (
        "ScanRefer_filtered_{}_spacy.json".format(split_name)
    )
    base_records = json.loads(base_path.read_text(encoding="utf-8"))
    sidecar_records = json.loads(sidecar_path.read_text(encoding="utf-8"))
    base_records = [
        record for record in base_records if record["scene_id"] in scan_ids
    ]
    selected_sidecar = [
        record for record in sidecar_records
        if record.get("scene_id") in scan_ids
    ]
    sidecar_keys = [
        scanrefer_annotation_key(record) for record in selected_sidecar
    ]
    if len(sidecar_keys) != len(set(sidecar_keys)):
        raise ValueError("ScanRefer SACR sidecar contains duplicate keys")
    lookup = build_scanrefer_structured_lookup(selected_sidecar)
    annotations = []
    hit_keys = []
    missing_examples = []
    for record in base_records:
        key = scanrefer_annotation_key(record)
        sidecar = lookup.get(key, {})
        utterance = " ".join(record.get("token", []))
        annotation = {
            "dataset": "scanrefer",
            "utterance": utterance,
            "_structured_source_utterance": utterance,
        }
        annotation.update(build_structured_annotation(sidecar, utterance))
        annotation["structured_annotation_hit"] = bool(sidecar)
        annotations.append(annotation)
        if sidecar:
            hit_keys.append(key)
        elif len(missing_examples) < 10:
            missing_examples.append(list(key))
    base_keys = [scanrefer_annotation_key(record) for record in base_records]
    return annotations, {
        "base_path": str(base_path),
        "sidecar_path": str(sidecar_path),
        "base_sample_count": len(base_records),
        "sidecar_record_count": len(selected_sidecar),
        "sidecar_unique_key_count": len(set(sidecar_keys)),
        "joined_sample_count": len(annotations),
        "lookup_hit_count": len(hit_keys),
        "lookup_missing_count": len(base_records) - len(hit_keys),
        "lookup_missing_examples": missing_examples,
        "sidecar_extra_key_count": len(set(sidecar_keys) - set(base_keys)),
        "reused_sidecar_count": len(hit_keys) - len(set(hit_keys)),
    }


def _referit3d_selected(row, dataset, split, scan_ids):
    if row.get("scan_id") not in scan_ids:
        return False
    if dataset == "nr3d":
        return (
            split != "test"
            or str(row.get("correct_guess", "")).lower() == "true"
        )
    return str(row.get("mentions_target_class", "")).lower() == "true"


def _referit3d_annotations(data_root, dataset, split):
    split_name = "test" if split == "val" else split
    scan_ids = _read_scan_ids(
        ROOT_DIR / "data" / "meta_data" /
        "{}_{}_scans.txt".format(dataset, split_name)
    )
    data_dir = _referit3d_directory(data_root)
    base_path = data_dir / "{}.csv".format(dataset)
    sidecar_path = data_dir / "{}_spacy.csv".format(dataset)
    with sidecar_path.open(encoding="utf-8") as handle:
        selected_sidecar = [
            row for row in csv.DictReader(handle)
            if _referit3d_selected(
                row, dataset, split_name, scan_ids
            )
        ]
    sidecar_keys = [
        Joint3DDataset._referit3d_structured_key(record, dataset)
        for record in selected_sidecar
    ]
    if len(sidecar_keys) != len(set(sidecar_keys)):
        raise ValueError(
            "{} SACR sidecar contains conflicting duplicate keys".format(
                dataset
            )
        )
    lookup = dict(zip(sidecar_keys, selected_sidecar))
    annotations = []
    base_keys = []
    hit_keys = []
    missing_examples = []
    with base_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not _referit3d_selected(
                    row, dataset, split_name, scan_ids):
                continue
            key = Joint3DDataset._referit3d_structured_key(row, dataset)
            sidecar = lookup.get(key, {})
            utterance = row.get("utterance", "")
            annotation = {
                "dataset": dataset,
                "utterance": utterance,
                "_structured_source_utterance": utterance,
            }
            annotation.update(build_structured_annotation(sidecar, utterance))
            annotation["structured_annotation_hit"] = bool(sidecar)
            annotations.append(annotation)
            base_keys.append(key)
            if sidecar:
                hit_keys.append(key)
            elif len(missing_examples) < 10:
                missing_examples.append(list(key))
    return annotations, {
        "base_path": str(base_path),
        "sidecar_path": str(sidecar_path),
        "base_sample_count": len(base_keys),
        "sidecar_record_count": len(selected_sidecar),
        "sidecar_unique_key_count": len(set(sidecar_keys)),
        "joined_sample_count": len(annotations),
        "lookup_hit_count": len(hit_keys),
        "lookup_missing_count": len(base_keys) - len(hit_keys),
        "lookup_missing_examples": missing_examples,
        "sidecar_extra_key_count": len(set(sidecar_keys) - set(base_keys)),
        "reused_sidecar_count": len(hit_keys) - len(set(hit_keys)),
        "base_duplicate_key_count": len(base_keys) - len(set(base_keys)),
    }


def _span_valid(span, text):
    if not isinstance(span, dict):
        return False
    try:
        start = int(span.get("start", -1))
        end = int(span.get("end", -1))
    except (TypeError, ValueError):
        return False
    return (
        0 <= start < end <= len(text)
        and text[start:end].lower() == str(span.get("text", "")).lower()
    )


def _token_rows(tokenized, annotations, texts, key):
    spans = [annotation.get(key, []) for annotation in annotations]
    return build_token_span_tensors(
        tokenized, spans, texts, torch.device("cpu")
    )


def _audit_alignment(annotations, tokenizer, batch_size):
    counters = Counter()
    examples = {
        "lost_target": [],
        "invalid_target_offset": [],
        "invalid_target_token_alignment": [],
        "invalid_relation_anchor_pair": [],
    }
    for offset in range(0, len(annotations), batch_size):
        batch = annotations[offset:offset + batch_size]
        before_has_target = [
            bool(annotation.get("target_spans")) for annotation in batch
        ]
        source_texts = [
            annotation.pop(
                "_structured_source_utterance",
                annotation.get("utterance", ""),
            )
            for annotation in batch
        ]
        with redirect_stdout(io.StringIO()):
            Scene_graph_parse(batch)
        for annotation, source_text in zip(batch, source_texts):
            realign_structured_annotation(
                annotation, source_text, annotation.get("utterance", "")
            )
        texts = [annotation.get("utterance", "") for annotation in batch]
        tokenized = tokenizer(
            texts,
            padding="longest",
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        target_tokens = _token_rows(
            tokenized, batch, texts, "target_spans"
        )
        entity_tokens = _token_rows(
            tokenized, batch, texts, "entity_spans"
        )
        relation_tokens = _token_rows(
            tokenized, batch, texts, "rel_spans"
        )
        for row_index, annotation in enumerate(batch):
            absolute_index = offset + row_index
            text = texts[row_index]
            target_spans = annotation.get("target_spans", [])
            entity_spans = annotation.get("entity_spans", [])
            relation_spans = annotation.get("rel_spans", [])
            anchor_ids = annotation.get("structured_anchor_ids", [])
            counters["record_count"] += 1
            counters["text_changed_count"] += int(
                source_texts[row_index] != text
            )
            counters["target_present_before_count"] += int(
                before_has_target[row_index]
            )
            counters["target_present_after_count"] += int(bool(target_spans))
            if before_has_target[row_index] and not target_spans:
                counters["lost_target_count"] += 1
                if len(examples["lost_target"]) < 10:
                    examples["lost_target"].append(absolute_index)
            target_offsets_valid = bool(target_spans) and all(
                _span_valid(span, text) for span in target_spans
            )
            counters["target_offset_valid_count"] += int(
                target_offsets_valid
            )
            target_token_valid = bool(target_spans) and bool(
                (target_tokens[row_index, :len(target_spans)] >= 0)
                .all(dim=-1).all().item()
            )
            counters["target_token_valid_count"] += int(target_token_valid)
            if target_spans and not target_offsets_valid:
                if len(examples["invalid_target_offset"]) < 10:
                    examples["invalid_target_offset"].append(absolute_index)
            if target_spans and not target_token_valid:
                if len(examples["invalid_target_token_alignment"]) < 10:
                    examples["invalid_target_token_alignment"].append(
                        absolute_index
                    )
            usable = (
                bool(annotation.get("structured_annotation_available"))
                and float(annotation.get("parse_confidence", 0.0)) > 0.0
                and not bool(annotation.get("decomp_global_only_mask", True))
                and target_token_valid
            )
            counters["sacr_usable_record_count"] += int(usable)
            counters["relation_record_count"] += int(bool(relation_spans))
            row_has_valid_pair = False
            for relation_index, relation_span in enumerate(relation_spans):
                counters["relation_span_count"] += 1
                anchor_id = (
                    int(anchor_ids[relation_index])
                    if relation_index < len(anchor_ids)
                    else -1
                )
                if anchor_id < 0:
                    counters["relation_without_anchor_count"] += 1
                    continue
                counters["anchored_relation_count"] += 1
                relation_valid = bool(
                    (relation_tokens[row_index, relation_index] >= 0)
                    .all().item()
                )
                anchor_valid = (
                    anchor_id < len(entity_spans)
                    and _span_valid(entity_spans[anchor_id], text)
                    and bool(
                        (entity_tokens[row_index, anchor_id] >= 0)
                        .all().item()
                    )
                )
                pair_valid = relation_valid and anchor_valid
                counters["valid_relation_anchor_pair_count"] += int(
                    pair_valid
                )
                row_has_valid_pair |= pair_valid
                if not pair_valid and len(
                        examples["invalid_relation_anchor_pair"]) < 10:
                    examples["invalid_relation_anchor_pair"].append({
                        "record_index": absolute_index,
                        "relation_index": relation_index,
                        "anchor_id": anchor_id,
                    })
            counters["valid_relation_pair_record_count"] += int(
                row_has_valid_pair
            )
    target_denominator = counters["target_present_after_count"]
    pair_denominator = counters["anchored_relation_count"]
    result = dict(counters)
    result.update({
        "target_offset_valid_rate": (
            counters["target_offset_valid_count"] / target_denominator
            if target_denominator else 0.0
        ),
        "target_token_valid_rate": (
            counters["target_token_valid_count"] / target_denominator
            if target_denominator else 0.0
        ),
        "relation_anchor_pair_valid_rate": (
            counters["valid_relation_anchor_pair_count"] / pair_denominator
            if pair_denominator else 0.0
        ),
        "examples": examples,
    })
    return result


def audit_dataset(data_root, dataset, split, tokenizer, batch_size):
    if dataset == "scanrefer":
        annotations, data = _scanrefer_annotations(data_root, split)
    else:
        annotations, data = _referit3d_annotations(
            data_root, dataset, split
        )
    alignment = _audit_alignment(annotations, tokenizer, batch_size)
    contract_split = "val" if split == "test" else split
    expected_count = EXPECTED_SPLIT_COUNTS.get(
        contract_split, {}
    ).get(dataset)
    data["lookup_hit_rate"] = (
        data["lookup_hit_count"] / data["base_sample_count"]
        if data["base_sample_count"] else 0.0
    )
    data["dataset_count_unchanged"] = (
        data["joined_sample_count"] == data["base_sample_count"]
    )
    data["expected_sample_count"] = expected_count
    data["expected_sample_count_match"] = (
        expected_count is None or data["base_sample_count"] == expected_count
    )
    passed = (
        data["dataset_count_unchanged"]
        and data["expected_sample_count_match"]
        and data["lookup_missing_count"] == 0
        and data["sidecar_extra_key_count"] == 0
        and alignment.get("lost_target_count", 0) == 0
        and alignment.get("target_offset_valid_rate", 0.0) == 1.0
        and alignment.get("target_token_valid_rate", 0.0) == 1.0
        and alignment.get("valid_relation_anchor_pair_count", 0)
        == alignment.get("anchored_relation_count", 0)
    )
    return {"data": data, "alignment": alignment, "pass": passed}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--datasets", default="scanrefer,nr3d,sr3d"
    )
    parser.add_argument("--split", choices=("train", "val", "test"),
                        default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    datasets = tuple(
        name.strip() for name in args.datasets.split(",") if name.strip()
    )
    if not datasets or any(
            name not in ("scanrefer", "nr3d", "sr3d")
            for name in datasets):
        raise ValueError("datasets must be scanrefer, nr3d, and/or sr3d")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        str(Path(args.data_root) / "roberta-base"),
        local_files_only=True,
    )
    results = {}
    for dataset in datasets:
        results[dataset] = audit_dataset(
            args.data_root,
            dataset,
            args.split,
            tokenizer,
            args.batch_size,
        )
    receipt = {
        "schema": "mcln-sacr-structured-data-audit-v1",
        "split": args.split,
        "datasets": results,
        "pass": all(result["pass"] for result in results.values()),
    }
    _atomic_write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
