#!/usr/bin/env python
"""Scene-disjoint train-only OOF gate for the V102 mask-only policy."""

import argparse
import hashlib
import io
import json
import math
import os
import random
import stat
from pathlib import Path

import torch

from models.rec_joint_box_mask import (
    LEGACY_MASK_POLICY_INDEX,
    MASK_POLICY_COUNT,
    build_mask_policy_feature_names,
)
from models.rec_query_mask_policy import (
    QueryMaskPolicyPostprocessor,
    compute_mask_policy_loss,
    select_mask_only_policy,
)
from scripts.cache_scanrefer_mask_policy_features import (
    load_mask_policy_feature_cache,
)
from scripts.train_scanrefer_joint_box_mask import (
    _compute_baseline_scores,
    _load_all_training_sources,
)
from scripts.train_scanrefer_rec_hierarchical_reranker import (
    AUTHORITATIVE_BACKBONE_PATH,
    capture_immutable_artifact_identities,
)


SCHEMA = "rec-v102-mask-only-scene-oof-v1"
VERSION = 1
EXPECTED_ROW_COUNT = 36665
EXPECTED_SCENE_COUNT = 562
EXPECTED_FOLD_COUNT = 5
EXPECTED_V101_PREDICTION_SHA256 = (
    "b81664e65d64dad7058f8f252d990d4ab11dd8c00746c64a918bb120b6434c99"
)
HIDDEN_DIM = 128
DROPOUT = 0.1
EPOCHS = 12
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP_NORM = 1.0
TARGET_TEMPERATURE = 0.25
AGGREGATE_MARGIN = 0.02
MIN_STD = 1e-6
BOOTSTRAP_SAMPLES = 10000
GATE_DELTA_HITS025 = 49
GATE_DELTA_HITS050 = 79
GATE_DELTA_MIOU = 0.0023


def file_sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def capture_readonly_file_identity(path, name):
    """Capture immutable evidence for one additional protected input."""
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("protected {} must be a regular file".format(name))
    metadata = path.stat()
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != 0o444:
        raise ValueError("protected {} must have mode 0444".format(name))
    return {
        "path": str(path),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": mode,
        "size": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
        "sha256": file_sha256(path),
    }


def tensor_sha256(*values):
    digest = hashlib.sha256()
    for value in values:
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _streaming_statistics(features, valid, indices, chunk_size=256):
    if (not isinstance(features, torch.Tensor)
            or features.dtype != torch.float32
            or features.dim() < 3
            or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool
            or valid.shape != features.shape[:-1]):
        raise ValueError("normalization tensors are malformed")
    indices = torch.as_tensor(indices, dtype=torch.long).cpu()
    if indices.dim() != 1 or indices.numel() == 0:
        raise ValueError("normalization indices must be non-empty")
    dimension = features.shape[-1]
    count = 0
    total = torch.zeros(dimension, dtype=torch.float64)
    square = torch.zeros(dimension, dtype=torch.float64)
    for start in range(0, int(indices.numel()), int(chunk_size)):
        selected = indices[start:start + int(chunk_size)]
        batch = features.index_select(0, selected)
        batch_valid = valid.index_select(0, selected)
        values = batch[batch_valid].double()
        count += int(values.shape[0])
        total += values.sum(dim=0)
        square += values.square().sum(dim=0)
    if count <= 0:
        raise ValueError("normalization has no valid values")
    mean = total / float(count)
    variance = (square / float(count) - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(float(MIN_STD))
    return {
        "count": count,
        "mean": mean.float(),
        "std": std.float(),
    }


def fit_fold_normalization(geometry_features, variant_valid,
                           mask_features, query_valid, indices):
    geometry = _streaming_statistics(
        geometry_features, variant_valid, indices
    )
    mask = _streaming_statistics(mask_features, query_valid, indices)
    result = {"geometry": geometry, "mask": mask}
    result["sha256"] = tensor_sha256(
        geometry["mean"], geometry["std"],
        mask["mean"], mask["std"],
    )
    return result


def normalize_batch(geometry, mask, valid, statistics):
    geometry_mean = statistics["geometry"]["mean"].to(geometry.device)
    geometry_std = statistics["geometry"]["std"].to(geometry.device)
    mask_mean = statistics["mask"]["mean"].to(mask.device)
    mask_std = statistics["mask"]["std"].to(mask.device)
    query_valid = valid.any(dim=2)
    normalized_geometry = (geometry - geometry_mean) / geometry_std
    normalized_geometry = torch.where(
        valid.unsqueeze(-1), normalized_geometry,
        torch.zeros_like(normalized_geometry),
    )
    normalized_mask = (mask - mask_mean) / mask_std
    normalized_mask = torch.where(
        query_valid.unsqueeze(-1), normalized_mask,
        torch.zeros_like(normalized_mask),
    )
    return normalized_geometry, normalized_mask


def set_deterministic(seed, device):
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def fit_fold_model(geometry_features, variant_valid, mask_features,
                   policy_ious, fit_indices, statistics, device):
    set_deterministic(0, device)
    model = QueryMaskPolicyPostprocessor(
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    fit_indices = torch.as_tensor(fit_indices, dtype=torch.long).cpu()
    history = []
    for epoch in range(EPOCHS):
        model.train()
        order = fit_indices[torch.randperm(
            int(fit_indices.numel()), generator=generator
        )]
        totals = {name: 0.0 for name in (
            "loss", "iou", "hit", "listwise", "regret"
        )}
        batches = 0
        for start in range(0, int(order.numel()), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            geometry = geometry_features.index_select(0, indices).to(device)
            valid = variant_valid.index_select(0, indices).to(device)
            mask = mask_features.index_select(0, indices).to(device)
            labels = policy_ious.index_select(0, indices).to(device)
            geometry, mask = normalize_batch(
                geometry, mask, valid, statistics
            )
            outputs = model(geometry, mask, valid)
            loss, components = compute_mask_policy_loss(
                outputs, labels, valid.any(dim=2), TARGET_TEMPERATURE
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP_NORM
            )
            optimizer.step()
            totals["loss"] += float(loss.detach().item())
            for name, value in components.items():
                totals[name] += float(value.detach().item())
            batches += 1
        history.append({
            "epoch": epoch + 1,
            **{name: value / float(batches)
               for name, value in totals.items()},
        })
    model.eval().requires_grad_(False)
    return model, history


def predict_fold(model, geometry_features, variant_valid, mask_features,
                 selected_parent_positions, held_indices, statistics, device):
    held_indices = torch.as_tensor(held_indices, dtype=torch.long).cpu()
    selected = []
    proposals = []
    accepted = []
    gains = []
    deltas_iou = []
    deltas_hits = []
    parents = []
    with torch.inference_mode():
        for start in range(0, int(held_indices.numel()), BATCH_SIZE):
            indices = held_indices[start:start + BATCH_SIZE]
            geometry = geometry_features.index_select(0, indices).to(device)
            valid = variant_valid.index_select(0, indices).to(device)
            mask = mask_features.index_select(0, indices).to(device)
            parent = selected_parent_positions.index_select(
                0, indices
            ).to(device)
            geometry, mask = normalize_batch(
                geometry, mask, valid, statistics
            )
            outputs = model(geometry, mask, valid)
            policy = select_mask_only_policy(
                outputs, parent, aggregate_margin=AGGREGATE_MARGIN
            )
            if not torch.equal(policy["selected_parent_positions"], parent):
                raise RuntimeError("V102 changed the frozen REC parent query")
            selected.append(policy["selected_policy_indices"].cpu())
            proposals.append(policy["proposal_policy_indices"].cpu())
            accepted.append(policy["accepted"].cpu())
            gains.append(policy["aggregate_gain"].cpu())
            deltas_iou.append(policy["delta_iou"].cpu())
            deltas_hits.append(policy["delta_hits"].cpu())
            parents.append(policy["selected_parent_positions"].cpu())
    return {
        "selected": torch.cat(selected).long(),
        "proposals": torch.cat(proposals).long(),
        "accepted": torch.cat(accepted).bool(),
        "aggregate_gain": torch.cat(gains).float(),
        "delta_iou": torch.cat(deltas_iou).float(),
        "delta_hits": torch.cat(deltas_hits).float(),
        "parents": torch.cat(parents).long(),
    }


def metric_delta(before, after):
    before = torch.as_tensor(before, dtype=torch.float64).reshape(-1)
    after = torch.as_tensor(after, dtype=torch.float64).reshape(-1)
    if before.shape != after.shape or before.numel() == 0:
        raise ValueError("metric vectors must align and be non-empty")
    result = {"count": int(before.numel())}
    for suffix, threshold in (("025", 0.25), ("050", 0.50)):
        before_hits = int(before.gt(threshold).sum().item())
        after_hits = int(after.gt(threshold).sum().item())
        result.update({
            "before_hits" + suffix: before_hits,
            "after_hits" + suffix: after_hits,
            "delta_hits" + suffix: after_hits - before_hits,
            "before_acc" + suffix: before_hits / float(before.numel()),
            "after_acc" + suffix: after_hits / float(after.numel()),
            "delta_acc" + suffix: (
                after_hits - before_hits
            ) / float(after.numel()),
        })
    result.update({
        "before_miou": float(before.mean().item()),
        "after_miou": float(after.mean().item()),
        "delta_miou": float((after - before).mean().item()),
        "improved_iou": int(after.gt(before).sum().item()),
        "equal_iou": int(after.eq(before).sum().item()),
        "degraded_iou": int(after.lt(before).sum().item()),
    })
    return result


def scene_block_bootstrap_lower_bounds(before, after, scene_ids,
                                       seed=0, samples=BOOTSTRAP_SAMPLES):
    before = torch.as_tensor(before, dtype=torch.float64).reshape(-1)
    after = torch.as_tensor(after, dtype=torch.float64).reshape(-1)
    if before.shape != after.shape or len(scene_ids) != before.numel():
        raise ValueError("bootstrap inputs do not align")
    scene_names = sorted(set(str(scene) for scene in scene_ids))
    if len(scene_names) != EXPECTED_SCENE_COUNT:
        raise ValueError("bootstrap scene coverage changed")
    differences = torch.stack((
        after.gt(0.25).double() - before.gt(0.25).double(),
        after.gt(0.50).double() - before.gt(0.50).double(),
        after - before,
    ), dim=-1)
    groups = {scene: [] for scene in scene_names}
    for index, scene in enumerate(scene_ids):
        groups[str(scene)].append(index)
    scene_sums = torch.stack([
        differences[groups[scene]].sum(dim=0) for scene in scene_names
    ])
    scene_counts = torch.tensor([
        len(groups[scene]) for scene in scene_names
    ], dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = torch.empty(int(samples), 3, dtype=torch.float64)
    for start in range(0, int(samples), 256):
        size = min(256, int(samples) - start)
        draw = torch.randint(
            0, len(scene_names), (size, len(scene_names)),
            generator=generator,
        )
        numerator = scene_sums[draw].sum(dim=1)
        denominator = scene_counts[draw].sum(dim=1, keepdim=True)
        values[start:start + size] = numerator / denominator
    ordered = values.sort(dim=0).values
    rank = max(0, int(math.ceil(0.025 * int(samples))) - 1)
    lower = ordered[rank]
    return {
        "samples": int(samples),
        "quantile": 0.025,
        "delta_acc025_lower_bound_95": float(lower[0].item()),
        "delta_acc050_lower_bound_95": float(lower[1].item()),
        "delta_miou_lower_bound_95": float(lower[2].item()),
    }


def acceptance_gate(metrics, folds, bootstrap, rec_identity_unchanged):
    per_fold = all(
        fold["delta_hits025"] >= 0
        and fold["delta_hits050"] >= 0
        and fold["delta_miou"] >= 0.0
        and (fold["delta_hits025"] > 0
             or fold["delta_hits050"] > 0
             or fold["delta_miou"] > 0.0)
        for fold in folds
    )
    predicates = {
        "rec_identity_digest_unchanged": bool(rec_identity_unchanged),
        "delta_hits025_at_least_49": (
            metrics["delta_hits025"] >= GATE_DELTA_HITS025
        ),
        "delta_hits050_at_least_79": (
            metrics["delta_hits050"] >= GATE_DELTA_HITS050
        ),
        "delta_miou_at_least_0p0023": (
            metrics["delta_miou"] >= GATE_DELTA_MIOU
        ),
        "all_folds_nondegrading_and_one_strict": per_fold,
        "bootstrap025_lower_bound_positive": (
            bootstrap["delta_acc025_lower_bound_95"] > 0.0
        ),
        "bootstrap050_lower_bound_positive": (
            bootstrap["delta_acc050_lower_bound_95"] > 0.0
        ),
        "bootstrap_miou_lower_bound_positive": (
            bootstrap["delta_miou_lower_bound_95"] > 0.0
        ),
    }
    return {"passed": all(predicates.values()), "predicates": predicates}


def write_exclusive_json(path, value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
    descriptor = os.open(
        str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444
    )
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise OSError("V102 report write made no progress")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def write_exclusive_torch(path, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    payload = buffer.getvalue()
    descriptor = os.open(
        str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444
    )
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise OSError("V102 decision sidecar write made no progress")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--joint-cache", required=True)
    parser.add_argument("--mask-feature-cache", required=True)
    parser.add_argument("--v101-sidecar", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decision-output")
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    parser.add_argument("--runtime-batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    if args.runtime_batch_size <= 0:
        parser.error("--runtime-batch-size must be positive")
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(str(output))
    decision_output = None
    if args.decision_output is not None:
        decision_output = Path(args.decision_output).expanduser().absolute()
        if decision_output.exists() or decision_output.is_symlink():
            raise FileExistsError(str(decision_output))
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise ValueError("V102 OOF requires CUDA")
    protected_paths = {
        "backbone": Path(AUTHORITATIVE_BACKBONE_PATH).resolve(),
        "parent": Path(args.parent_checkpoint).expanduser().resolve(),
        "geometry": Path(args.geometry_checkpoint).expanduser().resolve(),
    }
    protected_before = capture_immutable_artifact_identities(protected_paths)
    protected_before["v101_sidecar"] = capture_readonly_file_identity(
        args.v101_sidecar, "v101_sidecar"
    )

    joined_rows, base_manifest, geometry_manifest, joint_manifest = (
        _load_all_training_sources(args)
    )
    all_state = _compute_baseline_scores(
        joined_rows, args, args.device, base_manifest, geometry_manifest
    )
    feature_rows, feature_manifest = load_mask_policy_feature_cache(
        args.mask_feature_cache, require_full=True
    )
    sidecar_path = Path(args.v101_sidecar).expanduser().absolute()
    sidecar = torch.load(sidecar_path, map_location="cpu")
    if (sidecar.get("schema") != "rec-v101-oof-row-decisions-v1"
            or sidecar.get("validation_data_accessed") is not False
            or sidecar.get("prediction_sha256")
            != EXPECTED_V101_PREDICTION_SHA256
            or sidecar.get("row_count") != EXPECTED_ROW_COUNT
            or sidecar.get("scene_count") != EXPECTED_SCENE_COUNT):
        raise ValueError("V101 OOF sidecar provenance changed")
    if (len(joined_rows) != EXPECTED_ROW_COUNT
            or len(feature_rows) != EXPECTED_ROW_COUNT
            or all_state["features"].shape != (
                EXPECTED_ROW_COUNT, 16 * 7, 179)
            or all_state["valid_mask"].shape != (
                EXPECTED_ROW_COUNT, 16 * 7)
            or all_state["mask_policy_ious"].shape != (
                EXPECTED_ROW_COUNT, 16, MASK_POLICY_COUNT)):
        raise ValueError("V102 train source coverage changed")
    expected_indices = torch.as_tensor(
        sidecar["dataset_indices"], dtype=torch.long
    )
    for index, (joined, feature) in enumerate(zip(joined_rows, feature_rows)):
        identity = (
            int(joined["dataset_index"]), str(joined["scan_id"]),
            int(joined["target_id"]),
        )
        if (identity != (
                int(feature["dataset_index"]), feature["scan_id"],
                int(feature["target_id"]))
                or identity[0] != int(expected_indices[index].item())
                or not torch.equal(
                    torch.as_tensor(joined["query_indices"]).long(),
                    feature["query_indices"],
                )
                or not torch.equal(
                    torch.as_tensor(joined["candidate_valid"]).bool(),
                    feature["candidate_valid"],
                )):
            raise ValueError("V102 cache row alignment changed")

    geometry_features = all_state["features"].reshape(
        EXPECTED_ROW_COUNT, 16, 7, 179
    ).float().contiguous()
    variant_valid = all_state["valid_mask"].reshape(
        EXPECTED_ROW_COUNT, 16, 7
    ).bool().contiguous()
    mask_features = torch.stack([
        row["mask_policy_features"] for row in feature_rows
    ]).float().contiguous()
    policy_ious = all_state["mask_policy_ious"].float().contiguous()
    selected_parents = torch.as_tensor(
        sidecar["selected_parent_positions"], dtype=torch.long
    ).contiguous()
    fold_ids = torch.as_tensor(sidecar["fold_ids"], dtype=torch.long)
    scenes = list(all_state["scene_ids"])
    if (len(set(scenes)) != EXPECTED_SCENE_COUNT
            or set(fold_ids.tolist()) != set(range(EXPECTED_FOLD_COUNT))
            or not torch.equal(variant_valid.any(dim=2), torch.stack([
                row["candidate_valid"] for row in feature_rows
            ]))):
        raise ValueError("V102 scene/fold/query validity changed")

    selected_policies = torch.full(
        (EXPECTED_ROW_COUNT,), LEGACY_MASK_POLICY_INDEX, dtype=torch.long
    )
    proposal_policies = selected_policies.clone()
    accepted = torch.zeros(EXPECTED_ROW_COUNT, dtype=torch.bool)
    aggregate_gain = torch.zeros(EXPECTED_ROW_COUNT, dtype=torch.float32)
    delta_iou = torch.zeros(EXPECTED_ROW_COUNT, dtype=torch.float32)
    delta_hits = torch.zeros(EXPECTED_ROW_COUNT, 2, dtype=torch.float32)
    predicted_parents = torch.full_like(selected_parents, -1)
    fold_records = []
    for held in range(EXPECTED_FOLD_COUNT):
        fit_indices = fold_ids.ne(held).nonzero(as_tuple=False).reshape(-1)
        held_indices = fold_ids.eq(held).nonzero(as_tuple=False).reshape(-1)
        fit_scenes = {scenes[index] for index in fit_indices.tolist()}
        held_scenes = {scenes[index] for index in held_indices.tolist()}
        if fit_scenes & held_scenes:
            raise RuntimeError("V102 scene leakage between fit and held fold")
        statistics = fit_fold_normalization(
            geometry_features, variant_valid, mask_features,
            variant_valid.any(dim=2), fit_indices,
        )
        model, history = fit_fold_model(
            geometry_features, variant_valid, mask_features, policy_ious,
            fit_indices, statistics, device,
        )
        prediction = predict_fold(
            model, geometry_features, variant_valid, mask_features,
            selected_parents, held_indices, statistics, device,
        )
        selected_policies[held_indices] = prediction["selected"]
        proposal_policies[held_indices] = prediction["proposals"]
        accepted[held_indices] = prediction["accepted"]
        aggregate_gain[held_indices] = prediction["aggregate_gain"]
        delta_iou[held_indices] = prediction["delta_iou"]
        delta_hits[held_indices] = prediction["delta_hits"]
        predicted_parents[held_indices] = prediction["parents"]
        rows = torch.arange(int(held_indices.numel()))
        held_parent = selected_parents[held_indices]
        held_labels = policy_ious[held_indices, held_parent]
        before = held_labels[:, LEGACY_MASK_POLICY_INDEX]
        after = held_labels[rows, prediction["selected"]]
        metrics = metric_delta(before, after)
        fold_record = {
            "held_fold": held,
            "fit_rows": int(fit_indices.numel()),
            "held_rows": int(held_indices.numel()),
            "fit_scenes": len(fit_scenes),
            "held_scenes": len(held_scenes),
            "accepted_switches": int(prediction["accepted"].sum().item()),
            "normalization_sha256": statistics["sha256"],
            "final_epoch": history[-1],
            **metrics,
        }
        fold_records.append(fold_record)
        print(json.dumps({"completed_fold": fold_record}, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()
    if not torch.equal(predicted_parents, selected_parents):
        raise RuntimeError("V102 changed REC parent identity in OOF")
    rows = torch.arange(EXPECTED_ROW_COUNT)
    selected_labels = policy_ious[rows, selected_parents]
    before = selected_labels[:, LEGACY_MASK_POLICY_INDEX]
    after = selected_labels[rows, selected_policies]
    metrics = metric_delta(before, after)
    bootstrap = scene_block_bootstrap_lower_bounds(
        before, after, scenes, seed=0, samples=BOOTSTRAP_SAMPLES
    )
    rec_identity_digest = tensor_sha256(
        selected_parents, torch.as_tensor(sidecar["selected_indices"])
    )
    rec_identity_digest_after = tensor_sha256(
        predicted_parents, torch.as_tensor(sidecar["selected_indices"])
    )
    gate = acceptance_gate(
        metrics, fold_records, bootstrap,
        rec_identity_digest == rec_identity_digest_after,
    )
    protected_after = capture_immutable_artifact_identities(protected_paths)
    protected_after["v101_sidecar"] = capture_readonly_file_identity(
        args.v101_sidecar, "v101_sidecar"
    )
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during V102 OOF")
    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "deployable": bool(gate["passed"]),
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
        "protocol": {
            "selection": "frozen_v101_parent_then_mask_policy_only",
            "source_policy_count": MASK_POLICY_COUNT,
            "legacy_policy_index": LEGACY_MASK_POLICY_INDEX,
            "aggregate_margin": AGGREGATE_MARGIN,
            "seed": 0,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "target_temperature": TARGET_TEMPERATURE,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "runtime_materialization_batch_size": args.runtime_batch_size,
            "grid_search": False,
        },
        "coverage": {
            "rows": EXPECTED_ROW_COUNT,
            "scenes": EXPECTED_SCENE_COUNT,
            "folds": EXPECTED_FOLD_COUNT,
        },
        "metrics": metrics,
        "folds": fold_records,
        "bootstrap": bootstrap,
        "gate": gate,
        "diagnostics": {
            "accepted_switches": int(accepted.sum().item()),
            "proposal_changes": int(proposal_policies.ne(
                LEGACY_MASK_POLICY_INDEX
            ).sum().item()),
            "selected_policy_counts": [
                int(value) for value in torch.bincount(
                    selected_policies, minlength=MASK_POLICY_COUNT
                ).tolist()
            ],
            "prediction_sha256": tensor_sha256(
                proposal_policies, selected_policies, accepted,
                aggregate_gain, delta_iou, delta_hits,
            ),
            "rec_identity_sha256_before": rec_identity_digest,
            "rec_identity_sha256_after": rec_identity_digest_after,
            "v101_prediction_sha256": EXPECTED_V101_PREDICTION_SHA256,
        },
        "input_sha256": {
            "backbone": all_state["geometry_artifact"].get(
                "checkpoint_sha256"
            ),
            "parent": all_state["parent_artifact_sha256"],
            "geometry": all_state["geometry_artifact_sha256"],
            "v101_sidecar": file_sha256(sidecar_path),
            "base_manifest": file_sha256(
                Path(args.base_cache) / "manifest.json"
            ),
            "geometry_manifest": file_sha256(
                Path(args.geometry_cache) / "manifest.json"
            ),
            "joint_manifest": file_sha256(
                Path(args.joint_cache) / "manifest.json"
            ),
            "mask_feature_manifest": file_sha256(
                Path(args.mask_feature_cache) / "manifest.json"
            ),
        },
        "feature_contract": {
            "mask_feature_names": build_mask_policy_feature_names(),
            "mask_feature_manifest_content_sha256": feature_manifest[
                "content_sha256"
            ],
            "joint_label_manifest_content_sha256": joint_manifest[
                "content_sha256"
            ],
        },
        "source_sha256": {
            "driver": file_sha256(__file__),
            "model": file_sha256(
                Path(__file__).resolve().parents[1]
                / "models" / "rec_query_mask_policy.py"
            ),
            "feature_cache": file_sha256(
                Path(__file__).with_name(
                    "cache_scanrefer_mask_policy_features.py"
                )
            ),
        },
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    decision_sha = None
    if decision_output is not None:
        selected_policy_ious = policy_ious[rows, selected_parents]
        decision_sidecar = {
            "schema": "rec-v102-mask-only-oof-decisions-v1",
            "version": 1,
            "validation_data_accessed": False,
            "inference_uses_ground_truth": False,
            "row_count": EXPECTED_ROW_COUNT,
            "scene_count": EXPECTED_SCENE_COUNT,
            "fold_count": EXPECTED_FOLD_COUNT,
            "dataset_indices": expected_indices.clone(),
            "scene_ids": list(scenes),
            "fold_ids": fold_ids.clone(),
            "selected_parent_positions": selected_parents.clone(),
            "proposal_policy_indices": proposal_policies.clone(),
            "selected_policy_indices": selected_policies.clone(),
            "accepted": accepted.clone(),
            "aggregate_gain": aggregate_gain.clone(),
            "predicted_delta_iou": delta_iou.clone(),
            "predicted_delta_hits": delta_hits.clone(),
            "selected_parent_policy_ious": selected_policy_ious.clone(),
            "before_ious": before.clone(),
            "after_ious": after.clone(),
            "prediction_sha256": report["diagnostics"]["prediction_sha256"],
            "rec_identity_sha256": rec_identity_digest,
            "input_sha256": dict(report["input_sha256"]),
            "source_sha256": dict(report["source_sha256"]),
        }
        decision_output.parent.mkdir(parents=True, exist_ok=True)
        decision_sha = write_exclusive_torch(
            decision_output, decision_sidecar
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_sha = write_exclusive_json(output, report)
    print(json.dumps({
        "output": str(output), "sha256": output_sha,
        "decision_output": (
            str(decision_output) if decision_output is not None else None
        ),
        "decision_sha256": decision_sha,
        "deployable": report["deployable"], "metrics": metrics,
        "bootstrap": bootstrap, "gate": gate,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
