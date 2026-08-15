#!/usr/bin/env python
"""Nested scene-OOF screen of the V120 switch-outcome classifier."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

import scripts.run_v99_pareto_contextual_hierarchical as v99
import scripts.run_v95_threshold_aligned_listwise_hierarchical as v95
import models.rec_anchored_spatial_adapter as v115_model
import models.rec_pairwise_switch_risk as v118_model
import models.rec_pairwise_switch_classifier as v120_model
from models.rec_hierarchical_reranker import (
    build_hierarchical_scene_folds,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    HIERARCHICAL_BATCH_SIZE,
    HIERARCHICAL_DROPOUT,
    HIERARCHICAL_EPOCHS,
    HIERARCHICAL_GRAD_CLIP_NORM,
    HIERARCHICAL_LEARNING_RATE,
    _normalized_hierarchical_model_batch,
    _set_hierarchical_seed,
    capture_immutable_artifact_identities,
    fit_hierarchical_normalization,
    materialize_hierarchical_rows,
    split_residual_joined_rows,
)
from scripts.train_rec_geometry_reranker import (
    load_geometry_reranker_artifact,
    load_geometry_training_data,
)


V108_REPORT_SHA256 = (
    "72ca54b2db0bca829011a2f480c458c0a3e450a492dd77de9d8411e84f3e9162"
)
V115_REPORT_SHA256 = (
    "cae35808390c5f8c86b5ed3eeb73219ac226a20c7be091e36986fa25cf5f423f"
)
V116_REPORT_SHA256 = (
    "18fddfca24719062cc83b6b8e1c11183b04bb4e1b09da263d7e8b0db938ccdb9"
)
V117_REPORT_SHA256 = (
    "6e43afe461745ba4c65956d39f4e2fed7c62fd17d59a4641118549b1e1fc6c00"
)
V118_REPORT_SHA256 = (
    "8611a9bd24ab6e4d09e05dc37833f8e5d9dfc34e4c1be647ec66ecd4f10958da"
)
V119_REPORT_SHA256 = (
    "731d110af0c8954d2b6ff5a5e8930c9b4897eaf4f71d96a89c720c7bbbd2ee8a"
)
EXPECTED_SAMPLE_COUNT = 36665
EXPECTED_SCENE_COUNT = 562
MIN_DELTA_025 = 105
MIN_DELTA_050 = 225
MIN_BOOTSTRAP_LOWER_025 = 60
MIN_BOOTSTRAP_LOWER_050 = 170
MIN_CORRECTED_BOOTSTRAP_LOWER_025 = 35
MIN_CORRECTED_BOOTSTRAP_LOWER_050 = 115
MIN_REGULAR_BOOTSTRAP_LOWER_025 = 8
MIN_REGULAR_BOOTSTRAP_LOWER_050 = 25
MAX_SWITCH_RATE = 0.13
V118_RISK_EPOCHS = 200
V118_RISK_LEARNING_RATE = 1e-3
V118_RISK_WEIGHT_DECAY = 1e-3
V120_BREAK_COST = 4.0
V120_EVENT_WEIGHT = 4.0
V120_BREAK_CLASS = 0
V120_NEUTRAL_CLASS = 1
V120_FIX_CLASS = 2
V120_CLASS_WEIGHTS = (
    V120_BREAK_COST * V120_EVENT_WEIGHT,
    1.0,
    V120_EVENT_WEIGHT,
)
FALLBACK_MANIFEST_SHA256 = (
    "caf63109bdf9f19cd8132b3c70eb1f2467d70fc605d174c6ec801b34c1c31079"
)
EXPECTED_FALLBACK_SCENE_COUNT = 789
EXPECTED_CORRECTED_SCENE_COUNT = 361
EXPECTED_REGULAR_SCENE_COUNT = 201
EXPECTED_CANDIDATE_RECEIPT_SHA256 = (
    "bfe2a650e22459d09dcbca6f525cbcda136787b456bf77d734c1e2f76b67caaa"
)
EXPECTED_GEOMETRY_RECEIPT_SHA256 = (
    "e45adaafb3730f45dabcea7f0c4f4492a6ea6360b7f07bdb164270bd934d9443"
)
EXPECTED_PARENT_ARTIFACT_SHA256 = (
    "7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f"
)
EXPECTED_GEOMETRY_ARTIFACT_SHA256 = (
    "20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972"
)
EXPECTED_BACKBONE_SHA256 = (
    "3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208"
)
EXPECTED_BASE_CACHE_CONTENT_SHA256 = (
    "0ff6855d9e144d07447aa6d6b1fd26b7dd6c4a52168f2b2a8d6729940c60f637"
)
EXPECTED_GEOMETRY_CACHE_CONTENT_SHA256 = (
    "7bd0634bb7a6faeece7399e81dc98987e562dc8eea2ee701de8e9535f9bbc91f"
)
EXPECTED_GEOMETRY_METADATA_SHA256 = (
    "d596a574342af7966fcd33380832aabd1c2f369a353f233fcc1c63bbd7196798"
)


def acceptance_gate(diagnostics, subgroups):
    folds = diagnostics["fold_deltas"].values()
    return {
        "delta025_at_least_oracle_scaled_gap": (
            diagnostics["delta_hits025"] >= MIN_DELTA_025
        ),
        "delta050_at_least_oracle_scaled_gap": (
            diagnostics["delta_hits050"] >= MIN_DELTA_050
        ),
        "all_folds_strictly_positive025": all(
            row["hits025"] > 0 for row in folds
        ),
        "all_folds_strictly_positive050": all(
            row["hits050"] > 0
            for row in diagnostics["fold_deltas"].values()
        ),
        "bootstrap025_lower_at_least_frozen_floor": (
            diagnostics["bootstrap025"]["lower_bound_95"]
            >= MIN_BOOTSTRAP_LOWER_025
        ),
        "bootstrap050_lower_at_least_frozen_floor": (
            diagnostics["bootstrap050"]["lower_bound_95"]
            >= MIN_BOOTSTRAP_LOWER_050
        ),
        "corrected_bootstrap025_lower_at_least_frozen_floor": (
            subgroups["corrected"]["diagnostics"]["bootstrap025"][
                "lower_bound_95"] >= MIN_CORRECTED_BOOTSTRAP_LOWER_025
        ),
        "corrected_bootstrap050_lower_at_least_frozen_floor": (
            subgroups["corrected"]["diagnostics"]["bootstrap050"][
                "lower_bound_95"] >= MIN_CORRECTED_BOOTSTRAP_LOWER_050
        ),
        "regular_bootstrap025_lower_at_least_frozen_floor": (
            subgroups["regular"]["diagnostics"]["bootstrap025"][
                "lower_bound_95"] >= MIN_REGULAR_BOOTSTRAP_LOWER_025
        ),
        "regular_bootstrap050_lower_at_least_frozen_floor": (
            subgroups["regular"]["diagnostics"]["bootstrap050"][
                "lower_bound_95"] >= MIN_REGULAR_BOOTSTRAP_LOWER_050
        ),
        "switch_rate_at_most_frozen_ceiling": (
            diagnostics["switch_rate"] <= MAX_SWITCH_RATE
        ),
    }


def fit_v115(records, device):
    """Fit V99 first, freeze it, then fit only the bounded V115 adapter."""
    anchor, statistics, anchor_epochs = v99.fit_v97(records, device)
    resolved = torch.device(device)
    _set_hierarchical_seed(resolved)
    model = v115_model.AnchoredLanguageSpatialResidualAdapter(
        anchor,
        hidden_dim=128,
        dropout=HIERARCHICAL_DROPOUT,
        residual_scale=v115_model.V115_RESIDUAL_SCALE,
    ).to(resolved)
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad
                            for parameter in model.anchor.parameters()):
        raise RuntimeError("V118 anchor/trainable parameter contract changed")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=HIERARCHICAL_LEARNING_RATE,
        weight_decay=1e-3,
    )
    shuffle_state = random.Random(0)
    adapter_epochs = []
    for epoch in range(HIERARCHICAL_EPOCHS):
        model.train()
        if model.anchor.training:
            raise RuntimeError("V118 frozen anchor entered training mode")
        order = list(range(len(records)))
        shuffle_state.shuffle(order)
        totals = {"loss": 0.0, "query_loss": 0.0, "variant_loss": 0.0}
        batches = 0
        for start in range(0, len(order), HIERARCHICAL_BATCH_SIZE):
            indices = order[start:start + HIERARCHICAL_BATCH_SIZE]
            row_batch = [records[index] for index in indices]
            model_batch = _normalized_hierarchical_model_batch(
                row_batch, statistics, resolved
            )
            candidate_ious = torch.stack([
                record["candidate_ious"] for record in row_batch
            ]).to(resolved)
            outputs = model(**model_batch)
            loss, stats = v95.graded_listwise_loss(
                outputs,
                candidate_ious,
                model_batch["query_valid"],
                model_batch["variant_valid"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(trainable, HIERARCHICAL_GRAD_CLIP_NORM)
            optimizer.step()
            totals["loss"] += float(loss.detach().item())
            totals["query_loss"] += stats["query_loss"]
            totals["variant_loss"] += stats["variant_loss"]
            batches += 1
        adapter_epochs.append({
            "epoch": epoch + 1,
            "loss": totals["loss"] / batches,
            "query_loss": totals["query_loss"] / batches,
            "variant_loss": totals["variant_loss"] / batches,
        })
    model.eval().requires_grad_(False)
    return model, statistics, {
        "anchor_epochs": anchor_epochs,
        "adapter_epochs": adapter_epochs,
    }


def build_pairwise_features(
        records, proposals, baselines, proposal_probability,
        baseline_probability, head_gain):
    """Build deployable proposal-vs-baseline features without IoU labels."""
    count = len(records)
    tensors = (
        (proposals, torch.long, (count,), "proposals"),
        (baselines, torch.long, (count,), "baselines"),
        (proposal_probability, torch.float32, (count, 2), "proposal_probability"),
        (baseline_probability, torch.float32, (count, 2), "baseline_probability"),
        (head_gain, torch.float32, (count, 2), "head_gain"),
    )
    for value, dtype, shape, name in tensors:
        if (not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != dtype
                or tuple(value.shape) != shape):
            raise ValueError("V118 {} layout changed".format(name))
    rows = torch.arange(count)
    proposal_query = proposals.div(7, rounding_mode="floor")
    proposal_variant = proposals.remainder(7)
    baseline_query = baselines.div(7, rounding_mode="floor")
    baseline_variant = baselines.remainder(7)
    query_features = torch.stack([
        record["query_features"] for record in records
    ])
    query_aux = torch.stack([
        record["query_aux_continuous"] for record in records
    ])
    variant_aux = torch.stack([
        record["variant_aux_continuous"] for record in records
    ])
    baseline_scores = torch.stack([
        record["baseline_scores"] for record in records
    ])
    proposal_center = query_features[rows, proposal_query, 128:131]
    baseline_center = query_features[rows, baseline_query, 128:131]
    center_delta = (proposal_center - baseline_center).abs()
    proposal_size = query_features[rows, proposal_query, 131:134]
    baseline_size = query_features[rows, baseline_query, 131:134]
    size_delta = (proposal_size - baseline_size).abs()
    proposal_query_aux = query_aux[rows, proposal_query]
    baseline_query_aux = query_aux[rows, baseline_query]
    proposal_variant_aux = variant_aux[
        rows, proposal_query, proposal_variant
    ]
    baseline_variant_aux = variant_aux[
        rows, baseline_query, baseline_variant
    ]
    aggregate_gain = (
        2.0 * head_gain[:, 0] + head_gain[:, 1]
    ).unsqueeze(1)
    score_delta = (
        baseline_scores[rows, proposals]
        - baseline_scores[rows, baselines]
    ).unsqueeze(1)
    query_changed = proposal_query.ne(baseline_query).float().unsqueeze(1)
    variant_changed = (
        proposal_query.eq(baseline_query)
        & proposal_variant.ne(baseline_variant)
    ).float().unsqueeze(1)
    features = torch.cat((
        head_gain,
        proposal_probability,
        baseline_probability,
        aggregate_gain,
        score_delta,
        query_changed,
        variant_changed,
        center_delta,
        center_delta.square().sum(dim=1, keepdim=True).sqrt(),
        size_delta,
        proposal_query_aux - baseline_query_aux,
        proposal_variant_aux - baseline_variant_aux,
    ), dim=1).float().contiguous()
    if (tuple(features.shape)
            != (count, v118_model.V118_FEATURE_DIM)
            or not bool(torch.isfinite(features).all().item())):
        raise RuntimeError("V118 deployable feature contract changed")
    return features


def predict_pair_components(model, records, statistics, device):
    """Return the standard V115 proposal plus deployable pair features."""
    resolved = torch.device(device)
    model.to(resolved).eval()
    proposals = []
    baselines = []
    proposal_probability = []
    baseline_probability = []
    head_gain = []
    with torch.no_grad():
        for start in range(0, len(records), HIERARCHICAL_BATCH_SIZE):
            row_batch = records[start:start + HIERARCHICAL_BATCH_SIZE]
            batch = _normalized_hierarchical_model_batch(
                row_batch, statistics, resolved
            )
            outputs = model(**batch)
            selected = v99.select_hierarchical_proposal(
                outputs["query_logits"],
                outputs["variant_logits"],
                batch["query_valid"],
                batch["variant_valid"],
            )
            probabilities = v99.monotone_hit_probabilities(
                outputs["variant_logits"]
            ).reshape(len(row_batch), -1, 2)
            baseline = torch.tensor([
                record["baseline_index"] for record in row_batch
            ], dtype=torch.long, device=resolved)
            proposal = selected["flat_indices"]
            rows = torch.arange(len(row_batch), device=resolved)
            proposal_prob = probabilities[rows, proposal]
            baseline_prob = probabilities[rows, baseline]
            proposals.append(proposal.cpu().long())
            baselines.append(baseline.cpu().long())
            proposal_probability.append(proposal_prob.cpu().float())
            baseline_probability.append(baseline_prob.cpu().float())
            head_gain.append((proposal_prob - baseline_prob).cpu().float())
    proposals = torch.cat(proposals)
    baselines = torch.cat(baselines)
    proposal_probability = torch.cat(proposal_probability)
    baseline_probability = torch.cat(baseline_probability)
    head_gain = torch.cat(head_gain)
    aggregate_gain = 2.0 * head_gain[:, 0] + head_gain[:, 1]
    accepted = v99.pareto_accept_mask(
        proposals, baselines, aggregate_gain, head_gain
    )
    features = build_pairwise_features(
        records, proposals, baselines, proposal_probability,
        baseline_probability, head_gain,
    )
    return {
        "proposals": proposals,
        "baselines": baselines,
        "proposal_probability": proposal_probability,
        "baseline_probability": baseline_probability,
        "head_gain": head_gain,
        "aggregate_gain": aggregate_gain,
        "base_accepted": accepted,
        "features": features,
    }


def switch_outcome_targets(records, proposals, baselines):
    """Build training-only break=0, neutral=1, fix=2 class labels."""
    candidate_ious = torch.stack([
        record["candidate_ious"].reshape(-1) for record in records
    ])
    rows = torch.arange(len(records))
    proposal_iou = candidate_ious[rows, proposals]
    baseline_iou = candidate_ious[rows, baselines]
    targets = []
    counts = {}
    for suffix, threshold in (("025", 0.25), ("050", 0.50)):
        proposal_hit = proposal_iou.gt(threshold)
        baseline_hit = baseline_iou.gt(threshold)
        fixes = ~baseline_hit & proposal_hit
        breaks = baseline_hit & ~proposal_hit
        target = torch.full(
            (len(records),), V120_NEUTRAL_CLASS, dtype=torch.long
        )
        target[breaks] = V120_BREAK_CLASS
        target[fixes] = V120_FIX_CLASS
        targets.append(target)
        counts[suffix] = {
            "fix": int(fixes.sum().item()),
            "break": int(breaks.sum().item()),
            "neutral": int((~fixes & ~breaks).sum().item()),
        }
    return torch.stack(targets, dim=1), counts


def fit_pairwise_outcome_classifier(features, targets, device):
    """Fit one fixed three-class head on inner-held scene examples."""
    if (features.dtype != torch.float32
            or targets.dtype != torch.long
            or features.dim() != 2
            or tuple(targets.shape) != (features.shape[0], 2)
            or features.shape[0] < 100
            or not bool(torch.isfinite(features).all().item())
            or bool((targets < 0).any().item())
            or bool((targets >= 3).any().item())):
        raise ValueError("V120 classifier training tensors changed")
    for column in range(2):
        values = set(targets[:, column].unique().tolist())
        if values != {
                V120_BREAK_CLASS, V120_NEUTRAL_CLASS, V120_FIX_CLASS}:
            raise ValueError("V120 inner calibration lacks outcome classes")
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(
        v120_model.V120_MIN_STD
    )
    resolved = torch.device(device)
    _set_hierarchical_seed(resolved)
    model = v120_model.PairwiseSwitchOutcomeClassifier(
        mean, std
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=V118_RISK_LEARNING_RATE,
        weight_decay=V118_RISK_WEIGHT_DECAY,
    )
    train_features = features.to(resolved)
    train_targets = targets.to(resolved)
    class_weights = torch.tensor(
        V120_CLASS_WEIGHTS, dtype=torch.float32, device=resolved
    )
    final_loss = None
    for _ in range(V118_RISK_EPOCHS):
        model.train()
        logits = model(train_features)
        loss = F.cross_entropy(
            logits.reshape(-1, v120_model.V120_CLASS_COUNT),
            train_targets.reshape(-1),
            weight=class_weights,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach().item())
    model.eval().requires_grad_(False)
    return model, {
        "epochs": V118_RISK_EPOCHS,
        "final_loss": final_loss,
        "feature_mean_sha256": v99.tensor_sha256(mean),
        "feature_std_sha256": v99.tensor_sha256(std),
        "state_sha256": v99.tensor_sha256(*[
            value for _, value in sorted(model.state_dict().items())
        ]),
    }


def predict_pairwise_outcomes(model, features, device):
    resolved = torch.device(device)
    model.to(resolved).eval()
    with torch.no_grad():
        logits = model(features.to(resolved)).cpu().float()
    if (tuple(logits.shape) != (features.shape[0], 2, 3)
            or not bool(torch.isfinite(logits).all().item())):
        raise RuntimeError("V120 outcome prediction contract changed")
    return logits


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_bound_receipt(path, expected_sha256, expected_schema):
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("V118 receipt must be a regular non-symlink file")
    if file_sha256(resolved) != expected_sha256:
        raise ValueError("V118 receipt SHA-256 changed: {}".format(resolved))
    receipt = json.loads(resolved.read_text(encoding="ascii"))
    if (receipt.get("schema") != expected_schema
            or receipt.get("validation_data_accessed") is not False
            or receipt.get("sample_count") != EXPECTED_SAMPLE_COUNT
            or receipt.get("scene_count") != EXPECTED_SCENE_COUNT):
        raise ValueError("V118 receipt policy changed: {}".format(resolved))
    return receipt


def load_v108_meshsp_training_inputs(
        base_cache, geometry_cache, parent_artifact_path,
        geometry_artifact_path, device="cuda:0"):
    """Load only the frozen MeshSP V108 caches and fitted sidecars."""
    base_path = Path(base_cache).expanduser().resolve(strict=True)
    geometry_path = Path(geometry_cache).expanduser().resolve(strict=True)
    parent_path = Path(parent_artifact_path).expanduser().resolve(strict=True)
    geometry_artifact_path = Path(
        geometry_artifact_path
    ).expanduser().resolve(strict=True)
    if (base_path.name != "train"
            or geometry_path.name != "geometry_train"
            or base_path.parent != geometry_path.parent
            or parent_path.parent != base_path.parent / "v108_artifacts"
            or geometry_artifact_path.parent != parent_path.parent):
        raise ValueError("V108 inputs do not have the frozen train-only layout")

    candidate_receipt_path = base_path.parent / "candidate_train_receipt.json"
    geometry_receipt_path = base_path.parent / "geometry_train_receipt.json"
    candidate_receipt = _load_bound_receipt(
        candidate_receipt_path,
        EXPECTED_CANDIDATE_RECEIPT_SHA256,
        "mcln-v108-meshsp-train-candidate-cache-receipt-v1",
    )
    geometry_receipt = _load_bound_receipt(
        geometry_receipt_path,
        EXPECTED_GEOMETRY_RECEIPT_SHA256,
        "mcln-v108-meshsp-train-geometry-cache-receipt-v1",
    )
    parent_sha256 = file_sha256(parent_path)
    geometry_sha256 = file_sha256(geometry_artifact_path)
    if parent_sha256 != EXPECTED_PARENT_ARTIFACT_SHA256:
        raise ValueError("V108 parent artifact SHA-256 changed")
    if geometry_sha256 != EXPECTED_GEOMETRY_ARTIFACT_SHA256:
        raise ValueError("V108 geometry artifact SHA-256 changed")

    joined, base_manifest, geometry_manifest, parent = (
        load_geometry_training_data(
            base_path, geometry_path, parent_path
        )
    )
    geometry_model, geometry_artifact = load_geometry_reranker_artifact(
        geometry_artifact_path,
        device=device,
        parent_artifact_path=parent_path,
        base_manifest=base_manifest,
        geometry_manifest=geometry_manifest,
    )
    base_binding = geometry_manifest.get("base_cache_binding", {})
    expected = {
        "backbone": base_manifest.get("checkpoint_sha256"),
        "base_cache_content": base_binding.get("content_sha256"),
        "geometry_cache_content": geometry_manifest.get(
            "cache_content_digest"
        ),
        "geometry_metadata": geometry_manifest.get(
            "immutable_metadata_digest"
        ),
    }
    if (base_manifest.get("split") != "train"
            or geometry_manifest.get("split") != "train"
            or len(joined) != EXPECTED_SAMPLE_COUNT
            or base_manifest.get("sample_count") != EXPECTED_SAMPLE_COUNT
            or geometry_manifest.get("sample_count") != EXPECTED_SAMPLE_COUNT
            or candidate_receipt.get("checkpoint_sha256")
            != expected["backbone"]
            or candidate_receipt.get("manifest_sha256")
            != file_sha256(base_path / "manifest.json")
            or geometry_receipt.get("manifest_sha256")
            != file_sha256(geometry_path / "manifest.json")
            or geometry_receipt.get("base_cache_content_sha256")
            != expected["base_cache_content"]
            or geometry_receipt.get("cache_content_digest")
            != expected["geometry_cache_content"]
            or geometry_artifact.get("parent_artifact_sha256")
            != parent_sha256
            or geometry_artifact.get("train_base_cache_content_digest")
            != expected["base_cache_content"]
            or geometry_artifact.get("train_geometry_cache_content_digest")
            != expected["geometry_cache_content"]
            or geometry_artifact.get(
                "train_geometry_immutable_metadata_digest"
            ) != expected["geometry_metadata"]
            or float(geometry_artifact.get("geometry_weight", -1.0)) != 0.9
            or getattr(parent[0], "_artifact_sha256", None) != parent_sha256
            or getattr(geometry_model, "_artifact_sha256", None)
            != geometry_sha256):
        raise ValueError("V108 cache/artifact provenance binding changed")
    return {
        "joined_rows": joined,
        "base_manifest": base_manifest,
        "geometry_manifest": geometry_manifest,
        "parent": parent,
        "geometry_model": geometry_model,
        "geometry_artifact": geometry_artifact,
        "input_sha256": {
            **expected,
            "parent": parent_sha256,
            "geometry": geometry_sha256,
            "candidate_receipt": EXPECTED_CANDIDATE_RECEIPT_SHA256,
            "geometry_receipt": EXPECTED_GEOMETRY_RECEIPT_SHA256,
        },
        "validation_data_accessed": False,
    }


def validate_v108_materialization_artifact(artifact):
    """Accept only the already strict-loaded, frozen V108 geometry artifact."""
    expected = {
        "checkpoint_sha256": EXPECTED_BACKBONE_SHA256,
        "parent_artifact_sha256": EXPECTED_PARENT_ARTIFACT_SHA256,
        "train_base_cache_content_digest": (
            EXPECTED_BASE_CACHE_CONTENT_SHA256
        ),
        "train_geometry_cache_content_digest": (
            EXPECTED_GEOMETRY_CACHE_CONTENT_SHA256
        ),
        "train_geometry_immutable_metadata_digest": (
            EXPECTED_GEOMETRY_METADATA_SHA256
        ),
        "geometry_weight": 0.9,
        "regressed_variant_index": 0,
    }
    if (not isinstance(artifact, dict)
            or any(artifact.get(key) != value
                   for key, value in expected.items())):
        raise ValueError("V108 geometry materialization contract changed")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v108-report", required=True)
    parser.add_argument("--v115-report", required=True)
    parser.add_argument("--v116-report", required=True)
    parser.add_argument("--v117-report", required=True)
    parser.add_argument("--v118-report", required=True)
    parser.add_argument("--v119-report", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--geometry-artifact", required=True)
    parser.add_argument("--fallback-scenes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(str(output))
    source_path = Path(args.v108_report).expanduser().absolute()
    source_sha256 = file_sha256(source_path)
    if source_sha256 != V108_REPORT_SHA256:
        raise ValueError("V108 report SHA-256 changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_oof = source.get("oof", {})
    source_diagnostics = source_oof.get("diagnostics", {})
    if (source.get("schema")
            != "rec-v108-meshsp-pareto-full-train-scene-oof-v1"
            or source.get("validation_data_accessed") is not False
            or source_diagnostics.get("delta_hits025") != 70
            or source_diagnostics.get("delta_hits050") != 245):
        raise ValueError("V108 source contract changed")
    design_sources = {}
    for name, path_value, expected_sha256, expected_schema, expected_delta in (
            (
                "v115", args.v115_report, V115_REPORT_SHA256,
                "rec-v115-anchored-spatial-residual-scene-oof-v1", (75, 263),
            ),
            (
                "v116", args.v116_report, V116_REPORT_SHA256,
                "rec-v116-primary025-policy-scene-oof-v1", (72, 264),
            ),
            (
                "v117", args.v117_report, V117_REPORT_SHA256,
                "rec-v117-calibrated-anchored-adapter-scene-oof-v1", (70, 243),
            ),
            (
                "v118", args.v118_report, V118_REPORT_SHA256,
                "rec-v118-nested-pairwise-switch-risk-scene-oof-v1", (63, 202),
            ),
            (
                "v119", args.v119_report, V119_REPORT_SHA256,
                "rec-v119-nested-midpoint-break-veto-scene-oof-v1", (71, 256),
            )):
        path = Path(path_value).expanduser().absolute()
        if file_sha256(path) != expected_sha256:
            raise ValueError("{} report SHA-256 changed".format(name))
        report = json.loads(path.read_text(encoding="ascii"))
        oof = report.get("oof", {})
        diagnostics = oof.get("diagnostics", {})
        if (report.get("schema") != expected_schema
                or report.get("validation_data_accessed") is not False
                or oof.get("passed") is not False
                or diagnostics.get("delta_hits025") != expected_delta[0]
                or diagnostics.get("delta_hits050") != expected_delta[1]):
            raise ValueError("{} design-source contract changed".format(name))
        design_sources[name] = {
            "path": str(path),
            "sha256": expected_sha256,
            "oof": copy.deepcopy(oof),
        }
    fallback_path = Path(args.fallback_scenes).expanduser().absolute()
    if file_sha256(fallback_path) != FALLBACK_MANIFEST_SHA256:
        raise ValueError("MeshSP fallback-scene manifest SHA-256 changed")
    fallback_scenes = {
        line.strip() for line in fallback_path.read_text(
            encoding="ascii"
        ).splitlines() if line.strip()
    }
    if len(fallback_scenes) != EXPECTED_FALLBACK_SCENE_COUNT:
        raise ValueError("MeshSP fallback-scene manifest count changed")

    protected_paths = {
        "backbone": Path(AUTHORITATIVE_BACKBONE_PATH).resolve(),
        "parent": Path(args.parent_artifact).expanduser().resolve(),
        "geometry": Path(args.geometry_artifact).expanduser().resolve(),
    }
    protected_metadata_paths = {
        "candidate_receipt": (
            Path(args.base_cache).expanduser().resolve().parent
            / "candidate_train_receipt.json"
        ),
        "geometry_receipt": (
            Path(args.base_cache).expanduser().resolve().parent
            / "geometry_train_receipt.json"
        ),
        "base_manifest": (
            Path(args.base_cache).expanduser().resolve() / "manifest.json"
        ),
        "geometry_manifest": (
            Path(args.geometry_cache).expanduser().resolve() / "manifest.json"
        ),
    }
    protected_before = capture_immutable_artifact_identities(protected_paths)
    protected_metadata_before = {
        name: file_sha256(path)
        for name, path in protected_metadata_paths.items()
    }
    loaded = load_v108_meshsp_training_inputs(
        Path(args.base_cache), Path(args.geometry_cache),
        Path(args.parent_artifact), Path(args.geometry_artifact),
        device=args.device,
    )
    split = split_residual_joined_rows(loaded["joined_rows"])
    joined_rows = loaded["joined_rows"]
    scan_ids = [row["base"]["scan_id"] for row in joined_rows]
    if len(joined_rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("V108 full-train sample count changed")
    if len(set(scan_ids)) != EXPECTED_SCENE_COUNT:
        raise ValueError("V108 full-train scene count changed")
    training_scenes = set(scan_ids)
    corrected_scenes = training_scenes & fallback_scenes
    regular_scenes = training_scenes - fallback_scenes
    if (len(corrected_scenes) != EXPECTED_CORRECTED_SCENE_COUNT
            or len(regular_scenes) != EXPECTED_REGULAR_SCENE_COUNT
            or corrected_scenes & regular_scenes
            or corrected_scenes | regular_scenes != training_scenes):
        raise ValueError("V108 corrected/regular train-scene partition changed")
    reconstructed_rows = split["fit_rows"] + split["calibration_rows"]
    if len(reconstructed_rows) != len(joined_rows):
        raise RuntimeError("fit and calibration rows do not reconstruct train rows")
    if set(scan_ids) != set(
            row["base"]["scan_id"] for row in reconstructed_rows):
        raise RuntimeError("split rows do not reconstruct all train scenes")
    records = materialize_hierarchical_rows(
        joined_rows, loaded["parent"], loaded["geometry_model"],
        loaded["geometry_artifact"], device=args.device,
        require_contiguous=True,
        artifact_validator=validate_v108_materialization_artifact,
    )
    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("materialization dropped full-train rows")
    scene_folds = build_hierarchical_scene_folds([
        record["scan_id"] for record in records
    ])
    fold_indices = {
        fold: [
            index for index, record in enumerate(records)
            if scene_folds[record["scan_id"]] == fold
        ] for fold in range(5)
    }
    if sorted(index for rows in fold_indices.values() for index in rows) != list(
            range(EXPECTED_SAMPLE_COUNT)):
        raise RuntimeError("OOF folds do not partition all full-train rows")

    baselines = torch.tensor([
        record["baseline_index"] for record in records
    ], dtype=torch.long)
    proposals = baselines.clone()
    selected = baselines.clone()
    aggregate_gain = torch.zeros(EXPECTED_SAMPLE_COUNT, dtype=torch.float32)
    head_gain = torch.zeros(EXPECTED_SAMPLE_COUNT, 2, dtype=torch.float32)
    accepted = torch.zeros(EXPECTED_SAMPLE_COUNT, dtype=torch.bool)
    base_accepted = torch.zeros(EXPECTED_SAMPLE_COUNT, dtype=torch.bool)
    risk_logits = torch.zeros(
        EXPECTED_SAMPLE_COUNT, 2, 3, dtype=torch.float32
    )
    folds = []
    for held in range(5):
        outer_train_indices = sorted([
            index for fold in range(5) if fold != held
            for index in fold_indices[fold]
        ])
        held_indices = fold_indices[held]
        inner_calibration_fold = (held + 1) % 5
        inner_calibration_indices = fold_indices[inner_calibration_fold]
        inner_fit_indices = sorted([
            index for fold in range(5)
            if fold not in (held, inner_calibration_fold)
            for index in fold_indices[fold]
        ])
        if set(records[index]["scan_id"] for index in outer_train_indices) & set(
                records[index]["scan_id"] for index in held_indices):
            raise RuntimeError("scene leakage between train and held fold")
        if (set(records[index]["scan_id"] for index in inner_fit_indices)
                & set(records[index]["scan_id"]
                      for index in inner_calibration_indices)):
            raise RuntimeError("scene leakage into V120 inner calibration")

        inner_model, inner_statistics, inner_training = fit_v115(
            [records[index] for index in inner_fit_indices], args.device
        )
        inner_records = [
            records[index] for index in inner_calibration_indices
        ]
        inner_prediction = predict_pair_components(
            inner_model, inner_records, inner_statistics, args.device
        )
        inner_mask = inner_prediction["base_accepted"]
        inner_targets, _ = switch_outcome_targets(
            inner_records,
            inner_prediction["proposals"],
            inner_prediction["baselines"],
        )
        risk_features = inner_prediction["features"][inner_mask]
        risk_targets = inner_targets[inner_mask]
        target_counts = {}
        for column, suffix in enumerate(("025", "050")):
            values = risk_targets[:, column]
            target_counts[suffix] = {
                "fix": int(values.eq(V120_FIX_CLASS).sum().item()),
                "break": int(values.eq(V120_BREAK_CLASS).sum().item()),
                "neutral": int(values.eq(V120_NEUTRAL_CLASS).sum().item()),
            }
        risk_model, risk_training = fit_pairwise_outcome_classifier(
            risk_features, risk_targets, args.device
        )
        del inner_model
        torch.cuda.empty_cache()

        model, statistics, training = fit_v115(
            [records[index] for index in outer_train_indices], args.device
        )
        held_records = [records[index] for index in held_indices]
        prediction = predict_pair_components(
            model,
            held_records,
            statistics,
            args.device,
        )
        if not torch.equal(prediction["baselines"], baselines[held_indices]):
            raise RuntimeError("V120 OOF baseline identity changed")
        fold_risk_logits = predict_pairwise_outcomes(
            risk_model, prediction["features"], args.device
        )
        predicted_class = fold_risk_logits.argmax(dim=2)
        fold_accepted = (
            prediction["base_accepted"]
            & predicted_class[:, 0].ne(V120_BREAK_CLASS)
            & predicted_class[:, 1].ne(V120_BREAK_CLASS)
        )
        proposals[held_indices] = prediction["proposals"]
        selected[held_indices] = torch.where(
            fold_accepted,
            prediction["proposals"],
            prediction["baselines"],
        )
        aggregate_gain[held_indices] = prediction["aggregate_gain"]
        head_gain[held_indices] = prediction["head_gain"]
        base_accepted[held_indices] = prediction["base_accepted"]
        accepted[held_indices] = fold_accepted
        risk_logits[held_indices] = fold_risk_logits
        fold_record = {
            "held_fold": held,
            "inner_calibration_fold": inner_calibration_fold,
            "outer_train_row_count": len(outer_train_indices),
            "held_row_count": len(held_indices),
            "outer_train_scene_count": len(set(
                records[index]["scan_id"] for index in outer_train_indices
            )),
            "held_scene_count": len(set(
                records[index]["scan_id"] for index in held_indices
            )),
            "inner_fit_row_count": len(inner_fit_indices),
            "inner_fit_scene_count": len(set(
                records[index]["scan_id"] for index in inner_fit_indices
            )),
            "inner_calibration_row_count": len(inner_calibration_indices),
            "inner_calibration_scene_count": len(set(
                records[index]["scan_id"]
                for index in inner_calibration_indices
            )),
            "inner_base_switch_examples": int(inner_mask.sum().item()),
            "inner_target_counts": target_counts,
            "inner_normalization_sha256": inner_statistics["sha256"],
            "inner_anchor_final_epoch": inner_training["anchor_epochs"][-1],
            "inner_adapter_final_epoch": inner_training["adapter_epochs"][-1],
            "risk_training": risk_training,
            "base_accepted_switches": int(
                prediction["base_accepted"].sum().item()
            ),
            "risk_accepted_switches": int(fold_accepted.sum().item()),
            "normalization_sha256": statistics["sha256"],
            "anchor_final_epoch": training["anchor_epochs"][-1],
            "adapter_final_epoch": training["adapter_epochs"][-1],
        }
        folds.append(fold_record)
        print(json.dumps({"completed_fold": fold_record}, sort_keys=True), flush=True)
        del model, risk_model
        torch.cuda.empty_cache()

    diagnostics = v99.build_diagnostics(records, selected, baselines)
    subgroup_scene_sets = {
        "corrected": corrected_scenes,
        "regular": regular_scenes,
    }
    subgroups = {}
    for name, scenes in subgroup_scene_sets.items():
        indices = [
            index for index, record in enumerate(records)
            if record["scan_id"] in scenes
        ]
        subgroup_records = [records[index] for index in indices]
        subgroup_diagnostics = v99.build_diagnostics(
            subgroup_records, selected[indices], baselines[indices]
        )
        subgroups[name] = {
            "scene_count": len(scenes),
            "row_count": len(indices),
            "scene_sha256": hashlib.sha256(
                ("\n".join(sorted(scenes)) + "\n").encode("ascii")
            ).hexdigest(),
            "diagnostics": subgroup_diagnostics,
        }
    if sum(row["row_count"] for row in subgroups.values()) != len(records):
        raise RuntimeError("V108 subgroup rows do not partition train rows")
    predicates = acceptance_gate(diagnostics, subgroups)
    raw_switch = proposals.ne(baselines)
    original_margin_pass = raw_switch & aggregate_gain.ge(v99.V97_MARGIN)
    pareto_veto = original_margin_pass & ~base_accepted
    outcome_veto = base_accepted & ~accepted
    veto_reasons = {
        "original_margin_switches": int(original_margin_pass.sum().item()),
        "base_pareto_switches": int(base_accepted.sum().item()),
        "accepted_switches": int(accepted.sum().item()),
        "pareto_vetoes": int(pareto_veto.sum().item()),
        "nested_outcome_classifier_vetoes": int(outcome_veto.sum().item()),
        "nonpositive_delta025": int(
            (original_margin_pass & head_gain[:, 0].le(0.0)).sum().item()
        ),
        "nonpositive_delta050": int(
            (original_margin_pass & head_gain[:, 1].le(0.0)).sum().item()
        ),
    }
    protected_after = capture_immutable_artifact_identities(protected_paths)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V120")
    protected_metadata_after = {
        name: file_sha256(path)
        for name, path in protected_metadata_paths.items()
    }
    if protected_metadata_after != protected_metadata_before:
        raise RuntimeError("protected V120 cache metadata changed during OOF")
    script_sources = {
        "v120": str(Path(__file__).resolve()),
        "v115_model": str(Path(v115_model.__file__).resolve()),
        "v118_model": str(Path(v118_model.__file__).resolve()),
        "v120_model": str(Path(v120_model.__file__).resolve()),
        "v108": str(
            Path(__file__).resolve().parent / "run_v108_meshsp_pareto_oof.py"
        ),
        "v99": str(Path(v99.__file__).resolve()),
    }
    report = {
        "schema": "rec-v120-nested-switch-outcome-classifier-scene-oof-v1",
        "version": 1,
        "validation_data_accessed": False,
        "prior_calibration_used_for_selection": False,
        "prior_train_oof_used_for_protocol_design": True,
        "deployable": False,
        "source": {
            "path": str(source_path),
            "sha256": source_sha256,
            "v108_oof": copy.deepcopy(source["oof"]),
            "protocol_design": design_sources,
        },
        "protocol": {
            "architecture": (
                "V115 frozen-anchor language-spatial adapter followed by a "
                "general proposal-vs-baseline outcome classifier"
            ),
            "adapter_objective": "frozen V95 bounded threshold-aligned listwise",
            "risk_objective": (
                "weighted three-class cross entropy over break, neutral, "
                "and fix outcomes at both REC thresholds"
            ),
            "proposal_margin": v99.V97_MARGIN,
            "acceptance": (
                "V115 Pareto acceptance unless either threshold predicts "
                "the break class by argmax"
            ),
            "grid_search": False,
            "selection_data": (
                "outer scene-disjoint 5-fold OOF; each outer fold uses the "
                "next complete fold as inner calibration and fits its "
                "proposal model on the remaining three folds"
            ),
            "sample_count": EXPECTED_SAMPLE_COUNT,
            "scene_count": EXPECTED_SCENE_COUNT,
            "relation_coordinates": (
                "frozen query feature columns center_x/y/z_norm"
            ),
            "language_condition": "frozen 64D target_text_proj mean",
            "anchor": "fold-local V99 frozen before adapter optimization",
            "residual_scale": v115_model.V115_RESIDUAL_SCALE,
            "initial_output_contract": "exact V99 logits via zero delta heads",
            "risk_feature_dim": v120_model.V120_FEATURE_DIM,
            "risk_hidden_dim": v120_model.V120_HIDDEN_DIM,
            "risk_epochs": V118_RISK_EPOCHS,
            "risk_learning_rate": V118_RISK_LEARNING_RATE,
            "risk_weight_decay": V118_RISK_WEIGHT_DECAY,
            "risk_class_order": ["break", "neutral", "fix"],
            "risk_class_weights": list(V120_CLASS_WEIGHTS),
            "risk_acceptance": "argmax_not_break_at_both_thresholds",
            "risk_uses_iou_at_inference": False,
            "subgroup_policy": (
                "corrected361_and_regular201_bootstrap_lower floors"
            ),
        },
        "folds": folds,
        "veto_diagnostics": veto_reasons,
        "prediction_sha256": v99.tensor_sha256(
            proposals, selected, aggregate_gain, head_gain,
            base_accepted, risk_logits, accepted
        ),
        "oof": {
            "diagnostics": diagnostics,
            "subgroups": subgroups,
            "required_delta_hits025": MIN_DELTA_025,
            "required_delta_hits050": MIN_DELTA_050,
            "required_bootstrap_lower": {
                "025": MIN_BOOTSTRAP_LOWER_025,
                "050": MIN_BOOTSTRAP_LOWER_050,
            },
            "required_subgroup_bootstrap_lower": {
                "corrected": {
                    "025": MIN_CORRECTED_BOOTSTRAP_LOWER_025,
                    "050": MIN_CORRECTED_BOOTSTRAP_LOWER_050,
                },
                "regular": {
                    "025": MIN_REGULAR_BOOTSTRAP_LOWER_025,
                    "050": MIN_REGULAR_BOOTSTRAP_LOWER_050,
                },
            },
            "maximum_switch_rate": MAX_SWITCH_RATE,
            "predicates": predicates,
            "passed": all(predicates.values()),
        },
        "input_sha256": copy.deepcopy(loaded["input_sha256"]),
        "historical_split_metadata": copy.deepcopy(split["metadata"]),
        "source_sha256": {
            name: file_sha256(path) for name, path in script_sources.items()
        },
        "fallback_scene_manifest": {
            "path": str(fallback_path),
            "sha256": FALLBACK_MANIFEST_SHA256,
            "scene_count": EXPECTED_FALLBACK_SCENE_COUNT,
        },
        "protected_before": protected_before,
        "protected_after": protected_after,
        "protected_metadata_before": protected_metadata_before,
        "protected_metadata_after": protected_metadata_after,
    }
    payload = json.dumps(
        report, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(output), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("V120 output write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({
        "output": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "veto_diagnostics": veto_reasons,
        "oof": report["oof"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
