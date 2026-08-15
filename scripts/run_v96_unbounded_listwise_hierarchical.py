#!/usr/bin/env python
"""V96 wrapper: V95 target with unbounded raw-logit ranking utility."""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

import scripts.run_v95_threshold_aligned_listwise_hierarchical as base
import scripts.train_scanrefer_rec_hierarchical_reranker as train_module
from models.rec_hierarchical_reranker import QUERY_COUNT, VARIANT_COUNT


def raw_utility(logits):
    if logits.dtype != torch.float32 or logits.shape[-1] != 2:
        raise ValueError("raw utility logits must be float32 [...,2]")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("raw utility logits must be finite")
    return 2.0 * logits[..., 0] + logits[..., 1]


def graded_listwise_loss(outputs, candidate_ious, query_valid, variant_valid):
    quality = base.graded_quality(candidate_ious)
    query_quality = quality.masked_fill(
        ~variant_valid, -float("inf")
    ).max(dim=2).values
    query_score = raw_utility(outputs["query_logits"])
    variant_score = raw_utility(outputs["variant_logits"])
    query_loss = base.masked_soft_listwise_cross_entropy(
        query_score, query_quality, query_valid, dim=1
    )
    batch_size, query_count, variant_count = quality.shape
    valid_query_rows = query_valid.reshape(-1)
    variant_loss = base.masked_soft_listwise_cross_entropy(
        variant_score.reshape(batch_size * query_count, variant_count)[
            valid_query_rows
        ],
        quality.reshape(batch_size * query_count, variant_count)[
            valid_query_rows
        ],
        variant_valid.reshape(batch_size * query_count, variant_count)[
            valid_query_rows
        ],
        dim=1,
    )
    return query_loss + variant_loss, {
        "query_loss": float(query_loss.detach().item()),
        "variant_loss": float(variant_loss.detach().item()),
    }


def predict_raw_proposals(model, records, statistics, device):
    resolved = torch.device(device)
    model.to(resolved).eval()
    proposals = []
    gains = []
    with torch.no_grad():
        for start in range(0, len(records), base.HIERARCHICAL_BATCH_SIZE):
            row_batch = records[start:start + base.HIERARCHICAL_BATCH_SIZE]
            model_batch = base._normalized_hierarchical_model_batch(
                row_batch, statistics, resolved
            )
            outputs = model(**model_batch)
            query_score = raw_utility(outputs["query_logits"])
            variant_score = raw_utility(outputs["variant_logits"])
            selected_query = query_score.masked_fill(
                ~model_batch["query_valid"], -float("inf")
            ).argmax(dim=1)
            rows = torch.arange(len(row_batch), device=resolved)
            selected_variant = variant_score[rows, selected_query].masked_fill(
                ~model_batch["variant_valid"][rows, selected_query],
                -float("inf"),
            ).argmax(dim=1)
            selected_indices = (
                selected_query * VARIANT_COUNT + selected_variant
            )
            flat_score = variant_score.reshape(
                len(row_batch), QUERY_COUNT * VARIANT_COUNT
            )
            baseline_indices = torch.tensor([
                record["baseline_index"] for record in row_batch
            ], dtype=torch.long, device=resolved)
            gain = (
                flat_score[rows, selected_indices]
                - flat_score[rows, baseline_indices]
            )
            flat_valid = model_batch["variant_valid"].reshape(
                len(row_batch), -1
            )
            if not bool(flat_valid[rows, selected_indices].all().item()):
                raise RuntimeError("V96 proposed an invalid variant")
            proposals.append(selected_indices.detach().cpu().long())
            gains.append(gain.detach().cpu().float())
    return torch.cat(proposals), torch.cat(gains)


def exclusive_write(path, payload):
    descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--output" not in arguments:
        raise ValueError("V96 requires --output")
    output_index = arguments.index("--output") + 1
    output = Path(arguments[output_index]).expanduser().absolute()
    internal = Path(str(output) + ".internal_v95_schema.json")
    if output.exists() or internal.exists():
        raise FileExistsError(str(output))
    arguments[output_index] = str(internal)

    original_loss = base.graded_listwise_loss
    original_predict = base._predict_hierarchical_proposals
    original_training_predict = train_module._predict_hierarchical_proposals
    base.graded_listwise_loss = graded_listwise_loss
    base._predict_hierarchical_proposals = predict_raw_proposals
    train_module._predict_hierarchical_proposals = predict_raw_proposals
    try:
        base.main(arguments)
    finally:
        base.graded_listwise_loss = original_loss
        base._predict_hierarchical_proposals = original_predict
        train_module._predict_hierarchical_proposals = original_training_predict

    report = json.loads(internal.read_text(encoding="ascii"))
    report["schema"] = "rec-unbounded-listwise-hierarchical-train-only-v1"
    report["training_contract"] = copy.deepcopy(report["training_contract"])
    report["training_contract"]["objective"] = (
        "two-level-threshold-aligned-unbounded-soft-listwise-cross-entropy"
    )
    report["training_contract"]["ranking_utility"] = (
        "2*raw_logit25+raw_logit50"
    )
    report["internal_evidence"] = {
        "path": str(internal),
        "sha256": hashlib.sha256(internal.read_bytes()).hexdigest(),
    }
    payload = json.dumps(
        report,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    exclusive_write(output, payload)
    print(json.dumps({
        "output": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "oof": report["oof"],
        "calibration_status": report["calibration"]["status"],
        "calibration_passed": report["calibration"].get("passed"),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
