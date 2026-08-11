#!/usr/bin/env python
"""Train and publish the train-only joint REC box/mask quality adapter."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rec_joint_box_mask import (
    JointBoxMaskAdapter,
    LEGACY_MASK_POLICY_INDEX,
    MASK_LOGIT_THRESHOLDS,
    MASK_POLICY_COUNT,
    MASK_SOURCE_NAMES,
)


TRAINER_SCHEMA = "rec-joint-box-mask-adapter-v2"
FEATURE_DIM = 179
BASE_FEATURE_DIM = 152
GEOMETRY_FEATURE_DIM = 25
CANDIDATE_COUNT = 16
VARIANT_COUNT = 7
FLAT_COUNT = CANDIDATE_COUNT * VARIANT_COUNT
MASK_SOURCE_INDEX = 2
MASK_THRESHOLD_INDEX = 2
DEFAULT_HIDDEN_DIM = 128
DEFAULT_DROPOUT = 0.1
DEFAULT_CALIBRATION_FRACTION = 0.10
MIN_STD = 1e-6
REGISTERED_GATE = {
    "delta_position_acc025": 0.0,
    "delta_position_acc050": 0.0,
    "delta_mask_acc025": 0.0,
    "delta_mask_acc050": 0.02,
    "delta_mask_miou": 0.03,
    "position025_bootstrap_lcb": 0.0,
    "position050_bootstrap_lcb": 0.0,
}
FEATURE_NAMES_SUFFIX = (
    "parent_score", "parent_is_deployed_top1",
)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(value):
    return _sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8"))


def _tensor_sha256(value):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_scene_split(rows, seed=0,
                              calibration_fraction=DEFAULT_CALIBRATION_FRACTION):
    """Split rows by scene while preserving dataset order and return a digest."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("rows must be a non-empty sequence")
    fraction = float(calibration_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration_fraction must lie in (0,1)")
    scenes = sorted(set(str(row["scan_id"]) for row in rows))
    if len(scenes) < 2:
        raise ValueError("scene split requires at least two scenes")
    shuffled = list(scenes)
    random.Random(int(seed)).shuffle(shuffled)
    count = max(1, min(len(scenes) - 1, int(round(len(scenes) * fraction))))
    calibration_scenes = set(shuffled[:count])
    fit = [row for row in rows if row["scan_id"] not in calibration_scenes]
    calibration = [row for row in rows if row["scan_id"] in calibration_scenes]
    digest = canonical_json_sha256({
        "seed": int(seed),
        "calibration_fraction": fraction,
        "fit_scenes": sorted(set(row["scan_id"] for row in fit)),
        "calibration_scenes": sorted(calibration_scenes),
        "fit_indices": [int(row["dataset_index"]) for row in fit],
        "calibration_indices": [int(row["dataset_index"]) for row in calibration],
    })
    return fit, calibration, digest


def build_joint_feature_batch(row):
    """Flatten one joined cache row to query-major/variant-minor tensors."""
    required = (
        "features", "geometry_features", "geometry_valid", "geometry_ious",
        "mask_ious", "candidate_valid",
    )
    if any(key not in row for key in required):
        raise ValueError("joined row is missing joint feature fields")
    base = torch.as_tensor(row["features"]).float()
    geometry = torch.as_tensor(row["geometry_features"]).float()
    geometry_valid = torch.as_tensor(row["geometry_valid"]).bool()
    if geometry_valid.shape != (CANDIDATE_COUNT, VARIANT_COUNT):
        raise ValueError("geometry_valid must have shape [16,7]")
    if base.shape != (CANDIDATE_COUNT, BASE_FEATURE_DIM):
        raise ValueError("base features must have shape [16,152]")
    if geometry.shape != (CANDIDATE_COUNT, VARIANT_COUNT, GEOMETRY_FEATURE_DIM):
        raise ValueError("geometry features must have shape [16,7,25]")
    candidate_valid = torch.as_tensor(row["candidate_valid"]).bool()
    if candidate_valid.shape != (CANDIDATE_COUNT,):
        raise ValueError("candidate_valid must have shape [16]")
    box_ious = torch.as_tensor(row["geometry_ious"]).float()
    if box_ious.shape != geometry_valid.shape:
        raise ValueError("geometry_ious must match geometry_valid")
    mask_ious = torch.as_tensor(row["mask_ious"]).float()
    if mask_ious.shape != (CANDIDATE_COUNT, 3, 5):
        raise ValueError("mask_ious must have shape [16,3,5]")
    parent_scores = torch.as_tensor(
        row.get("parent_scores", torch.zeros(CANDIDATE_COUNT))
    ).float()
    parent_top1 = torch.as_tensor(
        row.get("parent_top1_mask", torch.zeros(CANDIDATE_COUNT))
    ).bool()
    if parent_scores.shape != (CANDIDATE_COUNT,):
        raise ValueError("parent_scores must have shape [16]")
    if parent_top1.shape != (CANDIDATE_COUNT,):
        raise ValueError("parent_top1_mask must have shape [16]")
    if not bool(torch.isfinite(parent_scores).all().item()):
        raise ValueError("parent_scores must be finite")
    valid = geometry_valid & candidate_valid.unsqueeze(-1)
    base_expanded = base.unsqueeze(1).expand(-1, VARIANT_COUNT, -1)
    parent_expanded = parent_scores.view(CANDIDATE_COUNT, 1, 1).expand(
        -1, VARIANT_COUNT, -1
    )
    top1_expanded = parent_top1.to(base.dtype).view(
        CANDIDATE_COUNT, 1, 1
    ).expand(-1, VARIANT_COUNT, -1)
    features = torch.cat((
        base_expanded, geometry, parent_expanded, top1_expanded,
    ), dim=-1).reshape(FLAT_COUNT, FEATURE_DIM)
    flat_valid = valid.reshape(FLAT_COUNT)
    features = torch.where(
        flat_valid.unsqueeze(-1), features, torch.zeros_like(features)
    )
    fused_mask = mask_ious[:, MASK_SOURCE_INDEX, MASK_THRESHOLD_INDEX]
    mask_policy_ious = mask_ious.reshape(
        CANDIDATE_COUNT, MASK_POLICY_COUNT
    )
    flat_mask = fused_mask.unsqueeze(-1).expand(-1, VARIANT_COUNT).reshape(-1)
    flat_box = box_ious.reshape(-1)
    flat_mask = flat_mask.masked_fill(~flat_valid, 0.0)
    flat_box = flat_box.masked_fill(~flat_valid, 0.0)
    return {
        "features": features.contiguous(),
        "valid_mask": flat_valid.contiguous(),
        "box_ious": flat_box.contiguous(),
        "mask_ious": flat_mask.contiguous(),
        "mask_policy_ious": mask_policy_ious.contiguous(),
        "query_positions": torch.arange(CANDIDATE_COUNT).view(
            CANDIDATE_COUNT, 1
        ).expand(-1, VARIANT_COUNT).reshape(-1),
        "variant_indices": torch.arange(VARIANT_COUNT).view(
            1, VARIANT_COUNT
        ).expand(CANDIDATE_COUNT, -1).reshape(-1),
    }


def _stable_top1(scores, valid):
    if scores.dim() != 2 or valid.shape != scores.shape:
        raise ValueError("scores and valid must share shape [B,C]")
    masked = scores.masked_fill(~valid, -float("inf"))
    values = masked.max(dim=1, keepdim=True).values
    ties = valid & masked.eq(values)
    indices = torch.arange(scores.shape[1], device=scores.device).view(1, -1)
    indices = indices.expand_as(scores).masked_fill(~ties, scores.shape[1])
    return indices.min(dim=1).values


def _rank_normalize(scores, valid):
    output = torch.full_like(scores, -float("inf"), dtype=torch.float32)
    for index in range(scores.shape[0]):
        valid_indices = valid[index].nonzero(as_tuple=False).reshape(-1)
        ordered = sorted(
            valid_indices.detach().cpu().tolist(),
            key=lambda item: (-float(scores[index, item].item()), item),
        )
        denominator = float(max(len(ordered) - 1, 1))
        for rank, item in enumerate(ordered):
            output[index, item] = 1.0 - rank / denominator
    return output


def select_quality_policy(mask_pred, box_logits, baseline_scores, valid,
                          switch_margin=0.02, box_margin=0.05):
    """Select a mask-quality candidate under a conservative box-tier gate."""
    tensors = (mask_pred, box_logits, baseline_scores, valid)
    if (not isinstance(mask_pred, torch.Tensor) or mask_pred.dim() != 2
            or not isinstance(box_logits, torch.Tensor)
            or box_logits.shape != mask_pred.shape + (2,)
            or not isinstance(baseline_scores, torch.Tensor)
            or baseline_scores.shape != mask_pred.shape
            or not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.shape != mask_pred.shape):
        raise ValueError("quality policy tensors have incompatible shapes")
    if not all(value.device == mask_pred.device for value in tensors):
        raise ValueError("quality policy tensors must share a device")
    if not bool(valid.any(dim=1).all().item()):
        raise ValueError("every policy row needs a valid candidate")
    if not math.isfinite(float(switch_margin)) or switch_margin < 0.0:
        raise ValueError("switch_margin must be finite and non-negative")
    if not math.isfinite(float(box_margin)) or box_margin < 0.0:
        raise ValueError("box_margin must be finite and non-negative")
    baseline = _stable_top1(baseline_scores, valid)
    p025 = box_logits[..., 0].sigmoid()
    p050 = box_logits[..., 1].sigmoid()
    rows = torch.arange(mask_pred.shape[0], device=mask_pred.device)
    base025 = p025[rows, baseline].unsqueeze(1)
    base050 = p050[rows, baseline].unsqueeze(1)
    eligible = valid & (p025 + float(box_margin) >= base025) & (
        p050 + float(box_margin) >= base050
    )
    utility = mask_pred + 0.02 * _rank_normalize(baseline_scores, valid)
    utility = utility.masked_fill(~eligible, -float("inf"))
    proposal = _stable_top1(utility, eligible)
    proposal_gain = mask_pred[rows, proposal] - mask_pred[rows, baseline]
    switched = proposal_gain >= float(switch_margin)
    selected = torch.where(switched, proposal, baseline)
    return {
        "selected_flat_index": selected,
        "baseline_flat_index": baseline,
        "proposal_flat_index": proposal,
        "switched": switched,
        "fallback_count": int((~switched).sum().item()),
        "eligible_count": eligible.sum(dim=1),
        "selected_mask_prediction": mask_pred[rows, selected],
    }


def evaluate_quality_policy(baseline_box, selected_box, baseline_mask,
                            selected_mask, selected_indices):
    """Return strict integer counts and deltas for one policy evaluation."""
    values = (baseline_box, selected_box, baseline_mask, selected_mask)
    if any(not isinstance(value, torch.Tensor) or value.dim() != 2 for value in values):
        raise ValueError("policy metric tensors must have shape [B,C]")
    if not all(value.shape == baseline_box.shape for value in values):
        raise ValueError("policy metric tensors must share shape")
    if not isinstance(selected_indices, torch.Tensor) or selected_indices.shape != (baseline_box.shape[0],):
        raise ValueError("selected_indices must have shape [B]")
    rows = torch.arange(baseline_box.shape[0], device=baseline_box.device)
    selected_box_values = selected_box[rows, selected_indices]
    selected_mask_values = selected_mask[rows, selected_indices]
    baseline_box_values = baseline_box[rows, baseline_box.argmax(dim=1)]
    baseline_mask_values = baseline_mask[rows, baseline_box.argmax(dim=1)]
    # Callers pass a baseline candidate axis; argmax is only a fallback for
    # synthetic tests.  Runtime training supplies its exact baseline values.
    result = {"sample_count": int(baseline_box.shape[0])}
    for name, before, after in (
            ("position", baseline_box_values, selected_box_values),
            ("mask", baseline_mask_values, selected_mask_values)):
        for suffix, threshold in (("025", 0.25), ("050", 0.50)):
            before_hits = int((before > threshold).sum().item())
            after_hits = int((after > threshold).sum().item())
            result["baseline_{}_hits{}".format(name, suffix)] = before_hits
            result["selected_{}_hits{}".format(name, suffix)] = after_hits
            result["baseline_{}_acc{}".format(name, suffix)] = before_hits / float(before.numel())
            result["selected_{}_acc{}".format(name, suffix)] = after_hits / float(after.numel())
            result["delta_{}_acc{}".format(name, suffix)] = (after_hits - before_hits) / float(after.numel())
    result["baseline_mask_miou"] = float(baseline_mask_values.mean().item())
    result["selected_mask_miou"] = float(selected_mask_values.mean().item())
    result["delta_mask_miou"] = result["selected_mask_miou"] - result["baseline_mask_miou"]
    return result


def evaluate_selected_values(baseline_box_values, selected_box_values,
                              baseline_mask_values, selected_mask_values):
    """Evaluate already-gathered Top-1 values (used by the trainer)."""
    before_box = torch.as_tensor(baseline_box_values).reshape(-1).float()
    after_box = torch.as_tensor(selected_box_values).reshape(-1).float()
    before_mask = torch.as_tensor(baseline_mask_values).reshape(-1).float()
    after_mask = torch.as_tensor(selected_mask_values).reshape(-1).float()
    if not (before_box.shape == after_box.shape == before_mask.shape == after_mask.shape):
        raise ValueError("selected policy values must align")
    out = {"sample_count": int(before_box.numel())}
    for name, before, after in (
            ("position", before_box, after_box), ("mask", before_mask, after_mask)):
        for suffix, threshold in (("025", 0.25), ("050", 0.50)):
            bh = int((before > threshold).sum().item())
            ah = int((after > threshold).sum().item())
            out["baseline_{}_hits{}".format(name, suffix)] = bh
            out["selected_{}_hits{}".format(name, suffix)] = ah
            out["baseline_{}_acc{}".format(name, suffix)] = bh / float(before.numel())
            out["selected_{}_acc{}".format(name, suffix)] = ah / float(after.numel())
            out["delta_{}_acc{}".format(name, suffix)] = (ah - bh) / float(after.numel())
    out["baseline_mask_miou"] = float(before_mask.mean().item())
    out["selected_mask_miou"] = float(after_mask.mean().item())
    out["delta_mask_miou"] = out["selected_mask_miou"] - out["baseline_mask_miou"]
    return out


def scene_block_bootstrap_lcb(before, after, scene_ids, seed=0,
                              samples=400, quantile=0.05):
    """Deterministic scene-block lower bound for threshold hit deltas."""
    before = torch.as_tensor(before).reshape(-1).float()
    after = torch.as_tensor(after).reshape(-1).float()
    if before.shape != after.shape or len(scene_ids) != before.numel():
        raise ValueError("bootstrap inputs do not align")
    groups = {}
    for index, scene in enumerate(scene_ids):
        groups.setdefault(str(scene), []).append(index)
    if not groups:
        raise ValueError("bootstrap requires scenes")
    generator = random.Random(int(seed))
    scene_names = sorted(groups)
    deltas = []
    for _ in range(int(samples)):
        chosen = [scene_names[generator.randrange(len(scene_names))]
                  for _ in scene_names]
        indices = [idx for scene in chosen for idx in groups[scene]]
        deltas.append(float((after[indices] - before[indices]).mean().item()))
    deltas.sort()
    position = max(0, min(len(deltas) - 1, int(math.floor(float(quantile) * len(deltas)))))
    return float(deltas[position])


def publication_gate(metrics):
    """Apply the pre-registered train-only publication thresholds."""
    observed = {}
    for key, threshold in REGISTERED_GATE.items():
        value = metrics.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("gate metric {} is invalid".format(key))
        if not math.isfinite(value):
            raise ValueError("gate metric {} is non-finite".format(key))
        observed[key] = value
    return {
        "pass": all(observed[key] >= threshold for key, threshold in REGISTERED_GATE.items()),
        "thresholds": dict(REGISTERED_GATE),
        "observed": observed,
    }


def compute_feature_stats(features, valid_mask, min_std=MIN_STD):
    features = torch.as_tensor(features).float()
    valid_mask = torch.as_tensor(valid_mask).bool()
    if features.dim() != 3 or valid_mask.shape != features.shape[:2]:
        raise ValueError("feature stats inputs are malformed")
    values = features[valid_mask]
    if values.numel() == 0:
        raise ValueError("feature stats have no valid rows")
    mean = values.mean(0)
    std = values.std(0, unbiased=False).clamp(min=float(min_std))
    return mean, std


def normalize_feature_batch(features, valid_mask, mean, std):
    features = features.float()
    mean = torch.as_tensor(mean, device=features.device, dtype=features.dtype)
    std = torch.as_tensor(std, device=features.device, dtype=features.dtype)
    normalized = (features - mean) / std
    return torch.where(valid_mask.unsqueeze(-1), normalized, torch.zeros_like(normalized))


def _policy_metrics_from_batch(
        adapter, features, valid, box_ious, mask_ious, mask_policy_ious,
        baseline_scores, switch_margin, box_margin):
    batch_size = features.shape[0]
    structured = features.reshape(batch_size, CANDIDATE_COUNT, VARIANT_COUNT, FEATURE_DIM)
    structured_valid = valid.reshape(batch_size, CANDIDATE_COUNT, VARIANT_COUNT)
    outputs = adapter(structured, structured_valid)
    policy = select_quality_policy(
        outputs["mask_iou"].reshape(batch_size, FLAT_COUNT),
        outputs["box_logits"].reshape(batch_size, FLAT_COUNT, 2),
        baseline_scores, valid, switch_margin=switch_margin,
        box_margin=box_margin,
    )
    rows = torch.arange(batch_size, device=features.device)
    baseline_idx = policy["baseline_flat_index"]
    selected_idx = policy["selected_flat_index"]
    before_box = box_ious[rows, baseline_idx]
    after_box = box_ious[rows, selected_idx]
    baseline_query = torch.div(
        baseline_idx, VARIANT_COUNT, rounding_mode="floor"
    )
    selected_query = torch.div(
        selected_idx, VARIANT_COUNT, rounding_mode="floor"
    )
    selected_policy = outputs["mask_policy_logits"][
        rows, selected_query
    ].argmax(dim=-1)
    before_mask = mask_policy_ious[
        rows, baseline_query, LEGACY_MASK_POLICY_INDEX
    ]
    after_mask = mask_policy_ious[
        rows, selected_query, selected_policy
    ]
    policy = dict(policy)
    policy.update({
        "baseline_parent_query": baseline_query,
        "selected_parent_query": selected_query,
        "selected_mask_policy_index": selected_policy,
        "selected_mask_source_index": torch.div(
            selected_policy,
            len(MASK_LOGIT_THRESHOLDS),
            rounding_mode="floor",
        ),
        "selected_mask_threshold_index": torch.remainder(
            selected_policy, len(MASK_LOGIT_THRESHOLDS)
        ),
    })
    return policy, before_box.detach().cpu(), after_box.detach().cpu(), before_mask.detach().cpu(), after_mask.detach().cpu()


def _joined_rows(base_rows, geometry_rows, mask_rows):
    if not (len(base_rows) == len(geometry_rows) == len(mask_rows)):
        raise ValueError("cache row populations do not align")
    joined = []
    for base, geometry, mask in zip(base_rows, geometry_rows, mask_rows):
        identity = (base["dataset_index"], base["scan_id"], base["target_id"])
        if identity != (geometry["dataset_index"], geometry["scan_id"], geometry["target_id"]) \
                or identity != (mask["dataset_index"], mask["scan_id"], mask["target_id"]):
            raise ValueError("cache row identities differ")
        if ("query_indices" in geometry
                and not torch.equal(
                    torch.as_tensor(mask["query_indices"]).long(),
                    torch.as_tensor(geometry["query_indices"]).long(),
                )):
            raise ValueError("mask and geometry query identities differ")
        if ("candidate_valid" in geometry
                and not torch.equal(
                    torch.as_tensor(mask["candidate_valid"]).bool(),
                    torch.as_tensor(geometry["candidate_valid"]).bool(),
                )):
            raise ValueError("mask and geometry candidate validity differs")
        row = dict(base)
        row.update({
            "geometry_features": geometry["geometry_features"],
            "geometry_valid": geometry.get("evaluator_valid", geometry["geometry_valid"]),
            "geometry_ious": geometry["geometry_ious"],
            "mask_ious": mask["mask_ious"],
            "candidate_valid": base["valid_mask"],
            # Keep the source rows available for exact baseline reconstruction;
            # these private references are never serialized into artifacts.
            "_base_row": base,
            "_geometry_row": geometry,
        })
        joined.append(row)
    return joined


def _prepare_training_tensors(rows, indices, device="cpu"):
    batches = [build_joint_feature_batch(rows[index]) for index in indices]
    return {
        key: torch.stack([batch[key] for batch in batches]).to(device)
        for key in (
            "features", "valid_mask", "box_ious", "mask_ious",
            "mask_policy_ious",
        )
    }


def train_adapter(fit_tensors, feature_mean, feature_std, hidden_dim=DEFAULT_HIDDEN_DIM,
                  dropout=DEFAULT_DROPOUT, lr=1e-3, weight_decay=1e-4,
                  epochs=18, batch_size=128, seed=0, device="cuda:0"):
    """Fit one deterministic multi-task adapter on the fit scenes."""
    device = torch.device(device)
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = JointBoxMaskAdapter(FEATURE_DIM, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    features = normalize_feature_batch(fit_tensors["features"].to(device), fit_tensors["valid_mask"].to(device), feature_mean.to(device), feature_std.to(device))
    valid = fit_tensors["valid_mask"].to(device)
    box = fit_tensors["box_ious"].to(device)
    mask = fit_tensors["mask_ious"].to(device)
    mask_policy = fit_tensors["mask_policy_ious"].to(device)
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(int(seed))
    count = features.shape[0]
    history = []
    for epoch in range(int(epochs)):
        order = torch.randperm(count, generator=order_generator)
        total = 0.0
        for start in range(0, count, int(batch_size)):
            batch_indices = order[start:start + int(batch_size)].to(device)
            x = features.index_select(0, batch_indices).reshape(-1, CANDIDATE_COUNT, VARIANT_COUNT, FEATURE_DIM)
            v = valid.index_select(0, batch_indices).reshape(-1, CANDIDATE_COUNT, VARIANT_COUNT)
            b = box.index_select(0, batch_indices).reshape(-1, CANDIDATE_COUNT, VARIANT_COUNT)
            m = mask.index_select(0, batch_indices).reshape(-1, CANDIDATE_COUNT, VARIANT_COUNT)
            mp = mask_policy.index_select(0, batch_indices).reshape(
                -1, CANDIDATE_COUNT, MASK_POLICY_COUNT
            )
            outputs = model(x, v)
            box_target = torch.stack((b > 0.25, b > 0.50), dim=-1).float()
            mask_target = torch.stack((m > 0.25, m > 0.50), dim=-1).float()
            box_loss = F.binary_cross_entropy_with_logits(outputs["box_logits"], box_target, reduction="none")
            mask_loss = F.smooth_l1_loss(outputs["mask_iou"], m, reduction="none")
            mask_threshold_loss = F.binary_cross_entropy_with_logits(
                outputs["mask_logits"], mask_target, reduction="none"
            )
            probability = outputs["mask_iou"]
            ranking = F.relu(0.03 - (probability - probability.mean(dim=(1, 2), keepdim=True)) * (m - m.mean(dim=(1, 2), keepdim=True)))
            weights = v.float()
            loss = (box_loss * weights.unsqueeze(-1)).sum() / weights.sum().clamp(min=1.0)
            loss = loss + 2.0 * (mask_loss * weights).sum() / weights.sum().clamp(min=1.0)
            loss = loss + 0.5 * (
                mask_threshold_loss * weights.unsqueeze(-1)
            ).sum() / weights.sum().clamp(min=1.0)
            loss = loss + 0.1 * (ranking * weights).sum() / weights.sum().clamp(min=1.0)
            query_valid = v.any(dim=2)
            policy_logits = outputs["mask_policy_logits"]
            policy_target = mp.argmax(dim=-1)
            policy_ce = F.cross_entropy(
                policy_logits.reshape(-1, MASK_POLICY_COUNT),
                policy_target.reshape(-1),
                reduction="none",
            ).reshape_as(query_valid)
            policy_expected_iou = (
                policy_logits.softmax(dim=-1) * mp
            ).sum(dim=-1)
            policy_regret = (
                mp.max(dim=-1).values - policy_expected_iou
            ).clamp(min=0.0)
            query_weights = query_valid.float()
            loss = loss + (
                policy_ce * query_weights
            ).sum() / query_weights.sum().clamp(min=1.0)
            loss = loss + 2.0 * (
                policy_regret * query_weights
            ).sum() / query_weights.sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().item())
        history.append(total / max(1, (count + int(batch_size) - 1) // int(batch_size)))
    model.eval().requires_grad_(False)
    return model, history


def _load_all_training_sources(args):
    from scripts.rec_geometry_cache import load_bound_candidate_cache, load_geometry_cache
    from scripts.cache_scanrefer_joint_box_mask import load_joint_cache
    base_rows, base_manifest, base_binding = load_bound_candidate_cache(args.base_cache, "train")
    geometry_rows, geometry_manifest = load_geometry_cache(
        args.geometry_cache, "train", base_snapshot=(base_rows, base_manifest, base_binding)
    )
    mask_rows, mask_manifest = load_joint_cache(args.joint_cache)
    if len(base_rows) != len(geometry_rows) or len(base_rows) != len(mask_rows):
        raise ValueError("train source cache sizes differ")
    if mask_manifest.get("sample_count") != len(base_rows):
        raise ValueError("joint mask cache is not complete")
    _validate_joint_cache_bindings(
        mask_manifest,
        base_manifest,
        geometry_manifest,
        Path(args.base_cache) / "manifest.json",
        Path(args.geometry_cache) / "manifest.json",
    )
    return _joined_rows(base_rows, geometry_rows, mask_rows), base_manifest, geometry_manifest, mask_manifest


def _validate_joint_cache_bindings(mask_manifest, base_manifest,
                                   geometry_manifest, base_manifest_path,
                                   geometry_manifest_path):
    """Reject labels built from a different frozen cache or backbone."""
    if not isinstance(mask_manifest, dict) or mask_manifest.get("complete") is not True:
        raise ValueError("joint mask cache must be marked complete")
    counts = (
        mask_manifest.get("sample_count"),
        mask_manifest.get("dataset_size"),
        mask_manifest.get("source_dataset_size"),
    )
    if not (counts[0] == counts[1] == counts[2]):
        raise ValueError("joint mask cache population is incomplete")
    if not isinstance(base_manifest, dict) or not isinstance(geometry_manifest, dict):
        raise ValueError("source cache manifests are invalid")
    base_manifest_path = Path(base_manifest_path).expanduser().resolve()
    geometry_manifest_path = Path(geometry_manifest_path).expanduser().resolve()
    if (not base_manifest_path.is_file() or not geometry_manifest_path.is_file()):
        raise ValueError("source cache manifests are missing")
    expected_base = _sha256_file(base_manifest_path)
    expected_geometry = _sha256_file(geometry_manifest_path)
    if mask_manifest.get("base_cache_manifest_sha256") != expected_base:
        raise ValueError("joint cache base manifest binding mismatch")
    if mask_manifest.get("geometry_cache_manifest_sha256") != expected_geometry:
        raise ValueError("joint cache geometry manifest binding mismatch")
    checkpoint_values = {
        mask_manifest.get("checkpoint_sha256"),
        base_manifest.get("checkpoint_sha256"),
        geometry_manifest.get("checkpoint_sha256"),
    }
    if len(checkpoint_values) != 1:
        raise ValueError("joint cache checkpoint binding mismatch")
    geometry_digest = geometry_manifest.get("cache_content_digest")
    if (not geometry_digest
            or mask_manifest.get("geometry_cache_content_digest") != geometry_digest):
        raise ValueError("joint cache geometry content binding mismatch")
    for manifest in (base_manifest, geometry_manifest):
        for key in ("sample_count", "dataset_size", "source_dataset_size"):
            if manifest.get(key) != counts[0]:
                raise ValueError("joint cache source population mismatch")
    return True


def _compute_baseline_scores(rows, args, device, base_manifest=None,
                             geometry_manifest=None):
    """Reconstruct the protected geometry baseline from immutable caches."""
    from scripts.train_rec_geometry_reranker import load_parent_reranker_snapshot, load_geometry_reranker_artifact, normalize_features
    from scripts.train_rec_geometry_reranker import build_geometry_training_batch
    from models.rec_geometry_reranker import blend_rec_geometry_scores
    parent = load_parent_reranker_snapshot(args.parent_checkpoint, device=device)
    geometry_model, geometry_artifact = load_geometry_reranker_artifact(
        args.geometry_checkpoint,
        device=device,
        parent_artifact_path=args.parent_checkpoint,
        base_manifest=base_manifest,
        geometry_manifest=geometry_manifest,
    )
    if float(geometry_artifact.get("geometry_weight", 0.0)) == 0.0:
        raise ValueError(
            "joint adapter requires the flat geometry scorer (geometry_weight > 0)"
        )
    all_features = []
    all_valid = []
    all_box = []
    all_mask = []
    all_mask_policy = []
    all_scores = []
    all_scenes = []
    for start in range(0, len(rows), int(args.runtime_batch_size)):
        chunk = rows[start:start + int(args.runtime_batch_size)]
        pairs = [{
            "base": row["_base_row"],
            "geometry": row["_geometry_row"],
        } for row in chunk]
        batch = build_geometry_training_batch(pairs, parent)
        valid = batch["valid_mask"].to(device)
        normalized = normalize_features(batch["features"].to(device).float(), valid, geometry_artifact["feature_mean"], geometry_artifact["feature_std"])
        outputs = geometry_model(normalized, valid)
        geometry_valid = valid.reshape(-1, CANDIDATE_COUNT, VARIANT_COUNT)
        parent_state = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch["parent_state"].items()
        }
        blended = blend_rec_geometry_scores(parent_state, outputs["ranking_logits"].float(), geometry_valid, geometry_artifact["geometry_weight"], geometry_artifact["regressed_variant_index"])
        all_features.append(batch["features"].cpu())
        all_valid.append(batch["valid_mask"].cpu())
        all_box.append(batch["candidate_ious"].cpu())
        all_mask.append(torch.stack([build_joint_feature_batch(row)["mask_ious"] for row in chunk]))
        all_mask_policy.append(torch.stack([
            build_joint_feature_batch(row)["mask_policy_ious"]
            for row in chunk
        ]))
        all_scores.append(blended["flat_scores"].detach().cpu())
        all_scenes.extend(row["scan_id"] for row in chunk)
    return {
        "features": torch.cat(all_features), "valid_mask": torch.cat(all_valid),
        "box_ious": torch.cat(all_box), "mask_ious": torch.cat(all_mask),
        "mask_policy_ious": torch.cat(all_mask_policy),
        "baseline_scores": torch.cat(all_scores), "scene_ids": all_scenes,
        "parent_artifact_sha256": getattr(parent[0], "_artifact_sha256", None),
        "geometry_artifact_sha256": getattr(geometry_model, "_artifact_sha256", None),
        "geometry_artifact": geometry_artifact,
    }


def _train_and_calibrate(args):
    rows, base_manifest, geometry_manifest, mask_manifest = _load_all_training_sources(args)
    fit_rows, calibration_rows, split_digest = deterministic_scene_split(rows, args.seed, args.calibration_fraction)
    fit_indices = [int(row["dataset_index"]) for row in fit_rows]
    calibration_indices = [int(row["dataset_index"]) for row in calibration_rows]
    all_state = _compute_baseline_scores(
        rows, args, args.device, base_manifest, geometry_manifest
    )
    tensor_keys = (
        "features", "valid_mask", "box_ious", "mask_ious",
        "mask_policy_ious",
    )
    fit_tensors = {key: all_state[key].index_select(0, torch.tensor(fit_indices)) for key in tensor_keys}
    cal_tensors = {key: all_state[key].index_select(0, torch.tensor(calibration_indices)) for key in tensor_keys}
    feature_mean, feature_std = compute_feature_stats(fit_tensors["features"], fit_tensors["valid_mask"])
    adapter, history = train_adapter(fit_tensors, feature_mean, feature_std, hidden_dim=args.hidden_dim, dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, batch_size=args.train_batch_size, seed=args.model_seed, device=args.device)
    cal_features = normalize_feature_batch(cal_tensors["features"].to(args.device), cal_tensors["valid_mask"].to(args.device), feature_mean.to(args.device), feature_std.to(args.device))
    cal_valid = cal_tensors["valid_mask"].to(args.device)
    cal_box = cal_tensors["box_ious"].to(args.device)
    cal_mask = cal_tensors["mask_ious"].to(args.device)
    cal_mask_policy = cal_tensors["mask_policy_ious"].to(args.device)
    cal_scores = all_state["baseline_scores"].index_select(0, torch.tensor(calibration_indices)).to(args.device)
    best = None
    for switch_margin in args.switch_margin_grid:
        for box_margin in args.box_margin_grid:
            with torch.inference_mode():
                policy, before_box, after_box, before_mask, after_mask = _policy_metrics_from_batch(
                    adapter, cal_features, cal_valid, cal_box, cal_mask,
                    cal_mask_policy, cal_scores, switch_margin, box_margin,
                )
            metrics = evaluate_selected_values(before_box, after_box, before_mask, after_mask)
            metrics["position025_bootstrap_lcb"] = scene_block_bootstrap_lcb((before_box > 0.25).float(), (after_box > 0.25).float(), [row["scan_id"] for row in calibration_rows], seed=args.seed)
            metrics["position050_bootstrap_lcb"] = scene_block_bootstrap_lcb((before_box > 0.5).float(), (after_box > 0.5).float(), [row["scan_id"] for row in calibration_rows], seed=args.seed)
            gate = publication_gate(metrics)
            key = (gate["pass"], metrics["delta_mask_miou"], metrics["delta_mask_acc050"], -metrics["delta_position_acc025"], -metrics["delta_position_acc050"])
            if best is None or key > best["key"]:
                best = {"key": key, "metrics": metrics, "gate": gate, "switch_margin": float(switch_margin), "box_margin": float(box_margin), "policy": policy}
    if best is None:
        raise RuntimeError("no calibration policy was evaluated")
    artifact = {
        "schema": TRAINER_SCHEMA,
        "deployable": bool(best["gate"]["pass"]),
        "selection": "joint_adapter" if best["gate"]["pass"] else "baseline",
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
        "input_dim": FEATURE_DIM,
        "feature_names": (
            list(base_manifest.get("feature_names", ()))
            + list(geometry_manifest.get("geometry_feature_names", ()))
            + list(FEATURE_NAMES_SUFFIX)
        ),
        "model_config": {"input_dim": FEATURE_DIM, "hidden_dim": args.hidden_dim, "dropout": args.dropout},
        "mask_policy_source_names": list(MASK_SOURCE_NAMES),
        "mask_policy_logit_thresholds": list(MASK_LOGIT_THRESHOLDS),
        "legacy_mask_policy_index": LEGACY_MASK_POLICY_INDEX,
        "model_state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "feature_mean": feature_mean.cpu(), "feature_std": feature_std.cpu(),
        "switch_margin": best["switch_margin"], "box_margin": best["box_margin"],
        "calibration_metrics": best["metrics"], "calibration_gate": best["gate"],
        "split_digest": split_digest, "fit_scene_count": len(set(row["scan_id"] for row in fit_rows)),
        "calibration_scene_count": len(set(row["scan_id"] for row in calibration_rows)),
        "fit_sample_count": len(fit_rows), "calibration_sample_count": len(calibration_rows),
        "base_cache_manifest_sha256": canonical_json_sha256(base_manifest),
        "geometry_cache_manifest_sha256": canonical_json_sha256(geometry_manifest),
        "joint_cache_manifest_sha256": canonical_json_sha256(mask_manifest),
        "parent_artifact_sha256": all_state["parent_artifact_sha256"],
        "geometry_artifact_sha256": all_state["geometry_artifact_sha256"],
        "backbone_checkpoint_sha256": geometry_manifest.get("checkpoint_sha256"),
        "training": {"seed": args.model_seed, "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs, "history": history},
    }
    return artifact


def save_artifact(path, artifact):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError("adapter artifact already exists: {}".format(path))
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, temporary)
    os.replace(str(temporary), str(path))
    return path


def write_trial_receipt(path, artifact, model_path):
    """Persist train-only selection evidence, including failed gates."""
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError("trial receipt already exists: {}".format(path))
    model_path = Path(model_path).expanduser().resolve()
    payload = {
        "schema": TRAINER_SCHEMA + "-trial-receipt-v1",
        "selection": artifact.get("selection"),
        "deployable": artifact.get("deployable") is True,
        "validation_data_accessed": artifact.get("validation_data_accessed"),
        "inference_uses_ground_truth": artifact.get(
            "inference_uses_ground_truth"
        ),
        "calibration_metrics": artifact.get("calibration_metrics", {}),
        "calibration_gate": artifact.get("calibration_gate", {}),
        "split_digest": artifact.get("split_digest"),
        "fit_scene_count": artifact.get("fit_scene_count"),
        "calibration_scene_count": artifact.get("calibration_scene_count"),
        "fit_sample_count": artifact.get("fit_sample_count"),
        "calibration_sample_count": artifact.get("calibration_sample_count"),
        "training": artifact.get("training", {}),
        "model_path": str(model_path),
        "model_sha256": _sha256_file(model_path) if model_path.is_file() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        os.chmod(str(path), 0o444)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def load_joint_adapter_artifact(path, device="cpu"):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("joint adapter artifact does not exist")
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict) or artifact.get("schema") != TRAINER_SCHEMA:
        raise ValueError("joint adapter artifact schema is invalid")
    if (artifact.get("deployable") is not True
            or artifact.get("selection") != "joint_adapter"
            or artifact.get("validation_data_accessed") is not False
            or artifact.get("inference_uses_ground_truth") is not False):
        raise ValueError("joint adapter artifact provenance is invalid")
    config = artifact.get("model_config")
    if (not isinstance(config, dict)
            or config.get("input_dim") != FEATURE_DIM
            or not isinstance(config.get("hidden_dim"), int)
            or isinstance(config.get("hidden_dim"), bool)
            or config["hidden_dim"] <= 0):
        raise ValueError("joint adapter model config is invalid")
    if (artifact.get("mask_policy_source_names")
            != list(MASK_SOURCE_NAMES)
            or artifact.get("mask_policy_logit_thresholds")
            != list(MASK_LOGIT_THRESHOLDS)
            or artifact.get("legacy_mask_policy_index")
            != LEGACY_MASK_POLICY_INDEX):
        raise ValueError("joint adapter mask policy schema is invalid")
    names = artifact.get("feature_names")
    if (not isinstance(names, list) or len(names) != FEATURE_DIM
            or len(set(names)) != FEATURE_DIM
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("joint adapter feature schema is invalid")
    mean = artifact.get("feature_mean")
    std = artifact.get("feature_std")
    if (not isinstance(mean, torch.Tensor)
            or not isinstance(std, torch.Tensor)
            or tuple(mean.shape) != (FEATURE_DIM,)
            or tuple(std.shape) != (FEATURE_DIM,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or bool((std <= 0).any().item())):
        raise ValueError("joint adapter normalization is invalid")
    for key in (
            "parent_artifact_sha256", "geometry_artifact_sha256",
            "backbone_checkpoint_sha256"):
        value = artifact.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("joint adapter {} is invalid".format(key))
        try:
            int(value, 16)
        except ValueError:
            raise ValueError("joint adapter {} is invalid".format(key))
    for key in ("switch_margin", "box_margin"):
        value = artifact.get(key)
        if (not isinstance(value, (float, int)) or isinstance(value, bool)
                or not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError("joint adapter {} is invalid".format(key))
    model = JointBoxMaskAdapter(FEATURE_DIM, hidden_dim=int(config["hidden_dim"]), dropout=float(config["dropout"]))
    model.load_state_dict(artifact.get("model_state_dict"), strict=True)
    model.to(torch.device(device)).eval().requires_grad_(False)
    model._artifact_sha256 = _sha256_file(path)
    return model, artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the ScanRefer joint mask quality adapter on train only.")
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--joint-cache", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--calibration-fraction", type=float, default=DEFAULT_CALIBRATION_FRACTION)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--runtime-batch-size", type=int, default=256)
    parser.add_argument("--switch-margin-grid", type=float, nargs="+", default=[0.01, 0.02, 0.04, 0.06])
    parser.add_argument("--box-margin-grid", type=float, nargs="+", default=[0.02, 0.05, 0.08, 0.12])
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.train_batch_size <= 0 or args.runtime_batch_size <= 0:
        parser.error("training sizes and epochs must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("calibration fraction must lie in (0,1)")
    return args


def main(argv=None):
    args = parse_args(argv)
    artifact = _train_and_calibrate(args)
    receipt_path = Path(str(args.output) + ".receipt.json")
    if artifact["selection"] != "joint_adapter":
        write_trial_receipt(receipt_path, artifact, args.output)
        print("Train gate selected protected baseline; no deployable adapter published")
        print(json.dumps({"calibration_gate": artifact["calibration_gate"],
                          "receipt": str(receipt_path)}, indent=2, sort_keys=True))
        return 2
    save_artifact(args.output, artifact)
    write_trial_receipt(receipt_path, artifact, args.output)
    print(json.dumps({
        "schema": artifact["schema"], "selection": artifact["selection"],
        "calibration_gate": artifact["calibration_gate"],
        "output": str(Path(args.output).expanduser().resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
