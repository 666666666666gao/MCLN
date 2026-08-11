#!/usr/bin/env python
"""Probe train-calibrated signed utility evidence without validation access."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_v40_shared_ordinal_evidence import (
    actual_transition_utility,
    fallback_position,
    select_feature_indices,
    split_rows_by_scene,
)
from scripts.train_rec_reranker import (
    deterministic_scene_split,
    load_candidate_cache,
)


class SignedUtilityEvidenceProbe(nn.Module):
    """Predict signed decision utility and heteroscedastic evidence scale."""

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.scale_head = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        hidden = self.trunk(features)
        utility = 2.0 * self.utility_head(hidden).squeeze(-1).tanh()
        scale = F.softplus(self.scale_head(hidden).squeeze(-1)) + 1e-3
        return utility, scale


def flatten_pair_utility(rows, feature_indices, break_cost):
    features = []
    utilities = []
    for row in rows:
        valid = row["valid_mask"].bool()
        fallback_index = fallback_position(row)
        active = valid.clone()
        active[fallback_index] = False
        candidate = row["features"][active][:, feature_indices].float()
        fallback = row["features"][fallback_index, feature_indices].float()
        fallback = fallback.unsqueeze(0).expand_as(candidate)
        features.append(torch.cat((
            candidate,
            fallback,
            candidate - fallback,
            candidate * fallback,
        ), dim=-1))
        candidate_iou = row["candidate_ious"][active].float()
        fallback_iou = row["candidate_ious"][fallback_index].float()
        utilities.append(actual_transition_utility(
            candidate_iou, fallback_iou, break_cost
        ))
    return torch.cat(features), torch.cat(utilities)


def _class_balanced_mean(values, targets):
    groups = (
        (targets > 0.0, 1.0),
        (targets < 0.0, 1.0),
        (targets == 0.0, 0.25),
    )
    terms = [
        values[mask].mean() * weight
        for mask, weight in groups if bool(mask.any().item())
    ]
    if not terms:
        return values.sum() * 0.0
    return torch.stack(terms).sum() / sum(
        weight for mask, weight in groups if bool(mask.any().item())
    )


def signed_utility_loss(prediction, scale, target):
    residual = prediction - target
    regression = F.smooth_l1_loss(
        prediction, target, reduction="none", beta=0.25
    )
    gaussian_nll = 0.5 * residual.square() / scale.square() + scale.log()
    nonneutral = target != 0.0
    if bool(nonneutral.any().item()):
        sign_target = (target[nonneutral] > 0.0).to(prediction.dtype)
        sign_loss = F.binary_cross_entropy_with_logits(
            prediction[nonneutral], sign_target, reduction="none"
        )
        positive = sign_target > 0.5
        sign_terms = []
        if bool(positive.any().item()):
            sign_terms.append(sign_loss[positive].mean())
        if bool((~positive).any().item()):
            sign_terms.append(sign_loss[~positive].mean())
        balanced_sign = torch.stack(sign_terms).mean()
    else:
        balanced_sign = prediction.sum() * 0.0
    return (
        _class_balanced_mean(regression, target)
        + 0.25 * _class_balanced_mean(gaussian_nll, target)
        + 0.5 * balanced_sign
    )


def fit_model(features, targets, mean, std, args, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = SignedUtilityEvidenceProbe(
        features.shape[1], args.hidden_dim, args.dropout
    ).to(device)
    dataset = TensorDataset((features - mean) / std, targets)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []
    for _ in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(
                device, non_blocking=device.type == "cuda"
            )
            batch_targets = batch_targets.to(
                device, non_blocking=device.type == "cuda"
            )
            prediction, scale = model(batch_features)
            loss = signed_utility_loss(prediction, scale, batch_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach().item()) * batch_features.shape[0]
            count += batch_features.shape[0]
        history.append(loss_sum / float(max(count, 1)))
    return model, history


def predict(model, features, mean, std, batch_size, device):
    utilities = []
    scales = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = (features[start:start + batch_size] - mean) / std
            utility, scale = model(batch.to(device))
            utilities.append(utility.cpu())
            scales.append(scale.cpu())
    return torch.cat(utilities), torch.cat(scales)


def calibrated_multipliers(model, rows, feature_indices, mean, std, args,
                           device):
    features, targets = flatten_pair_utility(
        rows, feature_indices, args.break_cost
    )
    prediction, scale = predict(
        model, features, mean, std, args.batch_size, device
    )
    harmful = targets < 0.0
    if not bool(harmful.any().item()):
        raise ValueError("calibration split contains no harmful candidates")
    ratios = (prediction[harmful] / scale[harmful]).clamp(min=0.0)
    return {
        "conformal_{:.3f}".format(coverage): float(torch.quantile(
            ratios, torch.tensor(float(coverage))
        ).item())
        for coverage in args.coverages
    }


def evaluate(model, rows, feature_indices, mean, std, multipliers, args,
             device):
    row_features = torch.stack(tuple(
        row["features"][:, feature_indices].float() for row in rows
    ))
    row_ious = torch.stack(tuple(
        row["candidate_ious"].float() for row in rows
    ))
    valid = torch.stack(tuple(row["valid_mask"].bool() for row in rows))
    fallback_indices = torch.tensor(tuple(
        fallback_position(row) for row in rows
    ), dtype=torch.long)
    row_index = torch.arange(len(rows))
    fallback_match = torch.zeros_like(valid)
    fallback_match[row_index, fallback_indices] = True
    active = valid & ~fallback_match
    fallback_features = row_features[
        row_index, fallback_indices
    ].unsqueeze(1).expand_as(row_features)
    pair_features = torch.cat((
        row_features,
        fallback_features,
        row_features - fallback_features,
        row_features * fallback_features,
    ), dim=-1)
    prediction, scale = predict(
        model, pair_features.flatten(0, 1), mean, std,
        args.batch_size, device,
    )
    prediction = prediction.view_as(valid)
    scale = scale.view_as(prediction)
    fallback_iou = row_ious[row_index, fallback_indices]
    actual_utility = actual_transition_utility(
        row_ious, fallback_iou.unsqueeze(1), args.break_cost
    ).masked_fill(~active, -1e4)
    oracle = actual_utility.max(dim=1).values > 0.0

    def metrics(margin):
        margin = margin.masked_fill(~active, -1e4)
        best_margin, selected_index = margin.max(dim=1)
        switch = best_margin > 0.0
        selected_utility = actual_utility[row_index, selected_index]
        beneficial = switch & (selected_utility > 0.0)
        harmful = switch & (selected_utility < 0.0)
        neutral = switch & (selected_utility == 0.0)
        selected_iou = torch.where(
            switch, row_ious[row_index, selected_index], fallback_iou
        )
        switch_count = int(switch.sum().item())
        beneficial_count = int(beneficial.sum().item())
        oracle_count = int(oracle.sum().item())
        return {
            "acc025": float((selected_iou > 0.25).float().mean().item()),
            "acc050": float((selected_iou > 0.50).float().mean().item()),
            "switches": switch_count,
            "beneficial_switches": beneficial_count,
            "harmful_switches": int(harmful.sum().item()),
            "neutral_switches": int(neutral.sum().item()),
            "switch_precision": beneficial_count / float(max(switch_count, 1)),
            "oracle_recall": beneficial_count / float(max(oracle_count, 1)),
        }

    policies = {"mean_utility": metrics(prediction)}
    for name, multiplier in multipliers.items():
        policies[name] = metrics(prediction - float(multiplier) * scale)
    for multiplier in args.fixed_multipliers:
        policies["fixed_{:.2f}_scale".format(multiplier)] = metrics(
            prediction - float(multiplier) * scale
        )
    return {
        "rows": len(rows),
        "baseline": {
            "acc025": float((fallback_iou > 0.25).float().mean().item()),
            "acc050": float((fallback_iou > 0.50).float().mean().item()),
        },
        "oracle_positive_rows": int(oracle.sum().item()),
        "policies": policies,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--scene-modulus", type=int, default=5)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    parser.add_argument("--calibration-fraction", type=float, default=0.125)
    parser.add_argument("--break-cost", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--coverages", type=float, nargs="+", default=(0.95, 0.99, 1.0)
    )
    parser.add_argument(
        "--fixed-multipliers", type=float, nargs="+",
        default=(0.5, 1.0, 1.5, 2.0),
    )
    parser.add_argument("--include-projections", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if (args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0
            or not 0.0 <= args.dropout < 1.0
            or not 0.0 < args.calibration_fraction < 1.0
            or any(not 0.0 < value <= 1.0 for value in args.coverages)
            or any(value < 0.0 for value in args.fixed_multipliers)):
        raise ValueError("probe arguments are invalid")
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.set_num_threads(1)
    rows, manifest = load_candidate_cache(args.cache, expected_split="train")
    train_rows, holdout_rows, holdout_scenes = split_rows_by_scene(
        rows, args.scene_modulus, args.holdout_remainder
    )
    fit_rows, calibration_rows = deterministic_scene_split(
        train_rows, args.seed, args.calibration_fraction
    )
    feature_indices = select_feature_indices(
        manifest["feature_names"], args.include_projections
    )
    fit_features, fit_targets = flatten_pair_utility(
        fit_rows, feature_indices, args.break_cost
    )
    mean = fit_features.mean(dim=0)
    std = fit_features.std(dim=0, unbiased=False).clamp(min=1e-6)
    model, history = fit_model(
        fit_features, fit_targets, mean, std, args, device
    )
    multipliers = calibrated_multipliers(
        model, calibration_rows, feature_indices, mean, std, args, device
    )
    results = evaluate(
        model, holdout_rows, feature_indices, mean, std, multipliers,
        args, device,
    )
    report = {
        "schema_version": 1,
        "purpose": "train-only-v40-signed-utility-evidence-probe",
        "validation_cache_opened": False,
        "cache": str(Path(args.cache).expanduser().resolve()),
        "cache_checkpoint_sha256": manifest["checkpoint_sha256"],
        "fit_rows": len(fit_rows),
        "calibration_rows": len(calibration_rows),
        "holdout_rows": len(holdout_rows),
        "holdout_scene_count": len(holdout_scenes),
        "feature_dim": len(feature_indices),
        "pair_feature_dim": 4 * len(feature_indices),
        "break_cost": args.break_cost,
        "calibrated_multipliers": multipliers,
        "fit_loss": history,
        "settings": vars(args),
        "results": results,
    }
    report["settings"]["cache"] = str(
        Path(args.cache).expanduser().resolve()
    )
    report["settings"]["output"] = str(
        Path(args.output).expanduser().resolve()
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
