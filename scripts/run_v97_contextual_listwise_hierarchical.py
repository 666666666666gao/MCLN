#!/usr/bin/env python
"""V97 wrapper: V95 objective with contextual query-set attention."""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn

import scripts.run_v95_threshold_aligned_listwise_hierarchical as base
from models.rec_hierarchical_reranker import (
    VARIANT_COUNT,
    HierarchicalQueryVariantReranker,
)


class ContextualHierarchicalQueryVariantReranker(
        HierarchicalQueryVariantReranker):
    """Compare query candidates with a masked permutation-equivariant layer."""

    def __init__(self, hidden_dim, dropout):
        super().__init__(hidden_dim=hidden_dim, dropout=dropout)
        if hidden_dim != 128:
            raise ValueError("V97 requires hidden_dim=128")
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.query_context = nn.TransformerEncoder(layer, num_layers=1)

    def forward(
            self, query_features, variant_features,
            query_aux_continuous, query_aux_binary,
            variant_aux_continuous, variant_aux_binary,
            query_valid, variant_valid):
        self._validate_inputs(
            query_features,
            variant_features,
            query_aux_continuous,
            query_aux_binary,
            variant_aux_continuous,
            variant_aux_binary,
            query_valid,
            variant_valid,
        )
        query_mask = query_valid.unsqueeze(-1)
        variant_mask = variant_valid.unsqueeze(-1)
        safe_variant_features = torch.where(
            variant_mask, variant_features, torch.zeros_like(variant_features)
        )
        variant_embedding = self.variant_encoder(safe_variant_features)
        variant_embedding = torch.where(
            variant_mask,
            variant_embedding,
            torch.zeros_like(variant_embedding),
        )
        variant_count = variant_valid.sum(dim=2, keepdim=True).clamp_min(1)
        variant_mean = variant_embedding.sum(dim=2) / variant_count.to(
            variant_embedding.dtype
        )
        variant_max = variant_embedding.masked_fill(
            ~variant_mask, -float("inf")
        ).max(dim=2).values
        variant_max = torch.where(
            query_mask, variant_max, torch.zeros_like(variant_max)
        )

        safe_query_features = torch.where(
            query_mask, query_features, torch.zeros_like(query_features)
        )
        safe_query_aux = torch.where(
            query_mask,
            query_aux_continuous,
            torch.zeros_like(query_aux_continuous),
        )
        query_input = torch.cat((
            safe_query_features,
            safe_query_aux,
            query_aux_binary.to(query_features.dtype),
            variant_mean,
            variant_max,
        ), dim=-1)
        query_embedding = self.query_encoder(query_input)
        query_embedding = torch.where(
            query_mask, query_embedding, torch.zeros_like(query_embedding)
        )
        contextual = self.query_context(
            query_embedding,
            src_key_padding_mask=~query_valid,
        )
        contextual = torch.where(
            query_mask, contextual, torch.zeros_like(contextual)
        )
        query_logits = self.query_head(contextual)
        query_logits = torch.where(
            query_mask, query_logits, torch.zeros_like(query_logits)
        )

        safe_variant_aux = torch.where(
            variant_mask,
            variant_aux_continuous,
            torch.zeros_like(variant_aux_continuous),
        )
        expanded_context = contextual.unsqueeze(2).expand(
            -1, -1, VARIANT_COUNT, -1
        )
        variant_input = torch.cat((
            expanded_context,
            variant_embedding,
            safe_variant_aux,
            variant_aux_binary.to(query_features.dtype),
        ), dim=-1)
        variant_logits = self.variant_head(variant_input)
        variant_logits = torch.where(
            variant_mask, variant_logits, torch.zeros_like(variant_logits)
        )
        return {
            "query_logits": query_logits,
            "variant_logits": variant_logits,
            "query_embedding": contextual,
            "variant_embedding": variant_embedding,
        }


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
        raise ValueError("V97 requires --output")
    output_index = arguments.index("--output") + 1
    output = Path(arguments[output_index]).expanduser().absolute()
    internal = Path(str(output) + ".internal_v95_schema.json")
    if output.exists() or internal.exists():
        raise FileExistsError(str(output))
    arguments[output_index] = str(internal)

    original_model = base.HierarchicalQueryVariantReranker
    base.HierarchicalQueryVariantReranker = (
        ContextualHierarchicalQueryVariantReranker
    )
    try:
        base.main(arguments)
    finally:
        base.HierarchicalQueryVariantReranker = original_model

    report = json.loads(internal.read_text(encoding="ascii"))
    report["schema"] = "rec-contextual-listwise-hierarchical-train-only-v1"
    report["training_contract"] = copy.deepcopy(report["training_contract"])
    report["training_contract"]["architecture"] = {
        "query_context_layers": 1,
        "attention_heads": 4,
        "feedforward_dim": 256,
        "activation": "gelu",
        "permutation_equivariant": True,
        "padding_masked": True,
    }
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

