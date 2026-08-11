#!/usr/bin/env python3
"""Train-only structural probe for shared ordinal query evidence.

The probe never opens a validation cache.  It fits token-level IoU threshold
probabilities on a scene-disjoint portion of one training cache, then evaluates
a fixed-zero candidate-versus-fallback expected-utility policy on held-out
training scenes.  The result is diagnostic evidence for an online architecture;
it is not a release artifact or a ScanRefer validation calibration step.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_rec_reranker import load_candidate_cache


THRESHOLDS = (0.25, 0.50)
THRESHOLD_WEIGHTS = (2.0 / 3.0, 1.0 / 3.0)


class SharedOrdinalEvidenceProbe(nn.Module):
    """Predict shared token quality on either independent or ordinal axes."""

    def __init__(self, input_dim, hidden_dim, mode):
        super().__init__()
        if mode not in ("independent", "ordinal"):
            raise ValueError("mode must be independent or ordinal")
        self.mode = mode
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, features):
        logits = self.network(features)
        probability_025 = torch.sigmoid(logits[:, 0])
        conditional_050 = torch.sigmoid(logits[:, 1])
        probability_050 = (
            conditional_050
            if self.mode == "independent"
            else probability_025 * conditional_050
        )
        return torch.stack((probability_025, probability_050), dim=1), None


class SharedBetaEvidenceProbe(nn.Module):
    """Predict one Beta evidence distribution for each IoU threshold."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 4),
        )

    def forward(self, features):
        evidence = F.softplus(self.network(features)).view(-1, 2, 2)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1)
        probabilities = alpha[..., 1] / strength
        variance = (
            probabilities * (1.0 - probabilities) / (strength + 1.0)
        )
        return probabilities, variance


class PairEvidentialTransitionProbe(nn.Module):
    """Predict break/neutral/fix evidence relative to one row fallback."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 6),
        )

    def forward(self, features):
        evidence = F.softplus(self.network(features)).view(-1, 2, 3)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        probabilities = alpha / strength
        variance = probabilities * (1.0 - probabilities) / (strength + 1.0)
        return probabilities, variance


class HierarchicalPairEvidentialTransitionProbe(nn.Module):
    """Factor transition evidence into change and conditional direction."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 8),
        )

    def forward(self, features):
        evidence = F.softplus(self.network(features)).view(-1, 2, 2, 2)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1)
        probability = alpha[..., 1] / strength
        variance = probability * (1.0 - probability) / (strength + 1.0)
        change_probability = probability[..., 0]
        fix_given_change = probability[..., 1]
        change_variance = variance[..., 0]
        direction_variance = variance[..., 1]
        break_probability = change_probability * (1.0 - fix_given_change)
        fix_probability = change_probability * fix_given_change
        neutral_probability = 1.0 - change_probability
        break_variance = (
            change_variance * direction_variance
            + change_variance * (1.0 - fix_given_change).pow(2)
            + direction_variance * change_probability.pow(2)
        )
        fix_variance = (
            change_variance * direction_variance
            + change_variance * fix_given_change.pow(2)
            + direction_variance * change_probability.pow(2)
        )
        probabilities = torch.stack((
            break_probability, neutral_probability, fix_probability,
        ), dim=-1)
        variances = torch.stack((
            break_variance, change_variance, fix_variance,
        ), dim=-1)
        return probabilities, variances


def dirichlet_kl_to_uniform(alpha):
    """KL(Dir(alpha) || Dir(1)) for the final class axis."""
    strength = alpha.sum(dim=-1, keepdim=True)
    class_count = alpha.shape[-1]
    log_normalizer = (
        torch.lgamma(strength)
        - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
        - torch.lgamma(alpha.new_tensor(float(class_count)))
    )
    expectation = (
        (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength))
    ).sum(dim=-1, keepdim=True)
    return (log_normalizer + expectation).squeeze(-1)


def evidential_binary_loss(model, features, targets, annealing):
    raw = model.network(features).view(-1, 2, 2)
    alpha = F.softplus(raw) + 1.0
    strength = alpha.sum(dim=-1, keepdim=True)
    one_hot = torch.stack((1.0 - targets, targets), dim=-1)
    fit = (
        one_hot * (torch.digamma(strength) - torch.digamma(alpha))
    ).sum(dim=-1)
    alpha_without_true_evidence = one_hot + (1.0 - one_hot) * alpha
    regularizer = dirichlet_kl_to_uniform(alpha_without_true_evidence)
    return (fit + float(annealing) * regularizer).mean()


def evidential_transition_loss(model, features, targets, annealing):
    raw = model.network(features).view(-1, 2, 3)
    alpha = F.softplus(raw) + 1.0
    strength = alpha.sum(dim=-1, keepdim=True)
    one_hot = F.one_hot(targets, num_classes=3).to(dtype=alpha.dtype)
    fit = (
        one_hot * (torch.digamma(strength) - torch.digamma(alpha))
    ).sum(dim=-1)
    alpha_without_true_evidence = one_hot + (1.0 - one_hot) * alpha
    regularizer = dirichlet_kl_to_uniform(alpha_without_true_evidence)
    return (fit + float(annealing) * regularizer).mean()


def hierarchical_evidential_transition_loss(
        model, features, targets, annealing):
    raw = model.network(features).view(-1, 2, 2, 2)
    alpha = F.softplus(raw) + 1.0
    strength = alpha.sum(dim=-1, keepdim=True)
    change_targets = targets.ne(1).long()
    direction_targets = targets.eq(2).long()
    task_targets = torch.stack((change_targets, direction_targets), dim=2)
    one_hot = F.one_hot(task_targets, num_classes=2).to(dtype=alpha.dtype)
    fit = (
        one_hot * (torch.digamma(strength) - torch.digamma(alpha))
    ).sum(dim=-1)
    alpha_without_true_evidence = one_hot + (1.0 - one_hot) * alpha
    regularizer = dirichlet_kl_to_uniform(alpha_without_true_evidence)
    task_loss = fit + float(annealing) * regularizer
    direction_active = change_targets.bool()
    change_loss = task_loss[..., 0].mean()
    direction_loss = (
        task_loss[..., 1][direction_active].mean()
        if bool(direction_active.any().item())
        else task_loss[..., 1].sum() * 0.0
    )
    return 0.5 * (change_loss + direction_loss)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe shared ordinal evidence on training scenes only."
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--scene-modulus", type=int, default=5)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    parser.add_argument("--break-cost", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--modes", nargs="+",
        choices=(
            "independent", "ordinal", "evidential", "pair_evidential",
            "hierarchical_pair_evidential",
        ),
        default=("independent", "ordinal", "evidential"),
    )
    parser.add_argument(
        "--confidence-weights", type=float, nargs="+",
        default=(0.25, 0.50, 0.75, 1.0),
        help="Fixed train-only Beta standard-deviation multipliers.",
    )
    parser.add_argument(
        "--include-projections", action="store_true",
        help="Include query/text projection features in addition to scalars.",
    )
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def split_rows_by_scene(rows, modulus, remainder):
    if modulus < 2:
        raise ValueError("scene modulus must be at least two")
    if remainder < 0 or remainder >= modulus:
        raise ValueError("holdout remainder must be in [0, scene_modulus)")
    scenes = sorted({row["scan_id"] for row in rows})
    holdout_scenes = {
        scene for index, scene in enumerate(scenes)
        if index % modulus == remainder
    }
    fit_rows = [row for row in rows if row["scan_id"] not in holdout_scenes]
    holdout_rows = [row for row in rows if row["scan_id"] in holdout_scenes]
    if not fit_rows or not holdout_rows:
        raise ValueError("scene split produced an empty partition")
    return fit_rows, holdout_rows, holdout_scenes


def select_feature_indices(feature_names, include_projections):
    if include_projections:
        return list(range(len(feature_names)))
    prefixes = ("query_proj_", "target_text_proj_")
    selected = [
        index for index, name in enumerate(feature_names)
        if not name.startswith(prefixes)
    ]
    if not selected:
        raise ValueError("feature selection removed every cache feature")
    return selected


def flatten_tokens(rows, feature_indices):
    features = []
    ious = []
    for row in rows:
        valid = row["valid_mask"].bool()
        features.append(row["features"][valid][:, feature_indices].float())
        ious.append(row["candidate_ious"][valid].float())
    return torch.cat(features, dim=0), torch.cat(ious, dim=0)


def fallback_position(row):
    match = row["valid_mask"].bool() & row["query_indices"].long().eq(
        int(row["default_top1_query_index"])
    )
    positions = match.nonzero(as_tuple=False).flatten()
    if positions.numel() != 1:
        raise ValueError("each row must contain its default fallback exactly once")
    return int(positions.item())


def transition_targets(candidate_iou, fallback_iou):
    labels = []
    for threshold in THRESHOLDS:
        candidate_ok = candidate_iou > threshold
        fallback_ok = fallback_iou > threshold
        target = torch.ones_like(candidate_iou, dtype=torch.long)
        target = torch.where(
            fallback_ok & ~candidate_ok, torch.zeros_like(target), target
        )
        target = torch.where(
            ~fallback_ok & candidate_ok,
            torch.full_like(target, 2),
            target,
        )
        labels.append(target)
    return torch.stack(tuple(labels), dim=-1)


def flatten_pair_examples(rows, feature_indices):
    features = []
    targets = []
    for row in rows:
        valid = row["valid_mask"].bool()
        fallback_index = fallback_position(row)
        active = valid.clone()
        active[fallback_index] = False
        candidate = row["features"][active][:, feature_indices].float()
        fallback = row["features"][fallback_index, feature_indices].float()
        fallback = fallback.unsqueeze(0).expand_as(candidate)
        features.append(torch.cat((
            candidate, fallback, candidate - fallback, candidate * fallback,
        ), dim=-1))
        candidate_iou = row["candidate_ious"][active].float()
        fallback_iou = row["candidate_ious"][fallback_index].float()
        targets.append(transition_targets(candidate_iou, fallback_iou))
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def threshold_targets(ious):
    return torch.stack(
        tuple(ious > threshold for threshold in THRESHOLDS), dim=1
    ).float()


def binary_calibration_error(probabilities, targets, bins=15):
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    error = probabilities.new_zeros(())
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if bool(selected.any().item()):
            fraction = selected.float().mean()
            error = error + fraction * (
                probabilities[selected].mean() - targets[selected].mean()
            ).abs()
    return float(error.item())


def fit_probe(mode, fit_features, fit_targets, mean, std, args, device):
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = (
        SharedBetaEvidenceProbe(fit_features.shape[1], args.hidden_dim)
        if mode == "evidential"
        else SharedOrdinalEvidenceProbe(
            fit_features.shape[1], args.hidden_dim, mode
        )
    ).to(device)
    normalized = (fit_features - mean) / std
    dataset = TensorDataset(normalized, fit_targets)
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
    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        example_count = 0
        for features, targets in loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            if mode == "evidential":
                annealing = min(1.0, 2.0 * (epoch + 1) / float(args.epochs))
                loss = evidential_binary_loss(
                    model, features, targets, annealing
                )
            else:
                probabilities, _ = model(features)
                loss = F.binary_cross_entropy(
                    probabilities.clamp(min=1e-6, max=1.0 - 1e-6), targets
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * features.shape[0]
            example_count += features.shape[0]
        history.append(loss_sum / float(max(example_count, 1)))
    return model, history


def fit_pair_probe(mode, fit_features, fit_targets, mean, std, args, device):
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = (
        HierarchicalPairEvidentialTransitionProbe(
            fit_features.shape[1], args.hidden_dim
        )
        if mode == "hierarchical_pair_evidential"
        else PairEvidentialTransitionProbe(
            fit_features.shape[1], args.hidden_dim
        )
    ).to(device)
    dataset = TensorDataset((fit_features - mean) / std, fit_targets)
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
    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        example_count = 0
        for features, targets in loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            annealing = min(1.0, 2.0 * (epoch + 1) / float(args.epochs))
            loss_builder = (
                hierarchical_evidential_transition_loss
                if mode == "hierarchical_pair_evidential"
                else evidential_transition_loss
            )
            loss = loss_builder(model, features, targets, annealing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * features.shape[0]
            example_count += features.shape[0]
        history.append(loss_sum / float(max(example_count, 1)))
    return model, history


def predict_tokens(model, features, mean, std, batch_size, device):
    probabilities = []
    variances = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = (features[start:start + batch_size] - mean) / std
            probability, variance = model(batch.to(device))
            probabilities.append(probability.cpu())
            if variance is not None:
                variances.append(variance.cpu())
    return (
        torch.cat(probabilities, dim=0),
        torch.cat(variances, dim=0) if variances else None,
    )


def evaluate_pair_policy(model, rows, feature_indices, mean, std, args, device):
    row_features = torch.stack(
        tuple(row["features"][:, feature_indices].float() for row in rows)
    )
    row_ious = torch.stack(tuple(row["candidate_ious"].float() for row in rows))
    valid_mask = torch.stack(tuple(row["valid_mask"].bool() for row in rows))
    fallback_indices = torch.tensor(
        tuple(fallback_position(row) for row in rows), dtype=torch.long
    )
    row_index = torch.arange(len(rows))
    fallback_match = torch.zeros_like(valid_mask)
    fallback_match[row_index, fallback_indices] = True
    active = valid_mask & ~fallback_match
    fallback_features = row_features[
        row_index, fallback_indices
    ].unsqueeze(1).expand_as(row_features)
    pair_features = torch.cat((
        row_features,
        fallback_features,
        row_features - fallback_features,
        row_features * fallback_features,
    ), dim=-1)
    probabilities, variances = predict_tokens(
        model, pair_features.flatten(0, 1), mean, std,
        args.batch_size, device,
    )
    probabilities = probabilities.view(len(rows), row_features.shape[1], 2, 3)
    variances = variances.view(len(rows), row_features.shape[1], 2, 3)
    weights = probabilities.new_tensor(THRESHOLD_WEIGHTS)
    expected_utility = (
        (probabilities[..., 2] - args.break_cost * probabilities[..., 0])
        * weights
    ).sum(dim=-1).masked_fill(~active, -1e4)

    fallback_iou = row_ious[row_index, fallback_indices]
    actual_utility = actual_transition_utility(
        row_ious, fallback_iou.unsqueeze(1), args.break_cost
    ).masked_fill(~active, -1e4)
    oracle = actual_utility.max(dim=1).values > 0.0

    def policy_metrics(candidate_margin):
        best_margin, selected_index = candidate_margin.max(dim=1)
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
            "oracle_positive_rows": oracle_count,
            "oracle_recall": beneficial_count / float(max(oracle_count, 1)),
            "mean_selected_margin": float(
                best_margin[switch].mean().item() if switch_count else 0.0
            ),
        }

    policies = {"expected_utility": policy_metrics(expected_utility)}
    for confidence_weight in args.confidence_weights:
        fix_lower = (
            probabilities[..., 2]
            - float(confidence_weight)
            * variances[..., 2].clamp(min=0.0).sqrt()
        ).clamp(min=0.0, max=1.0)
        break_upper = (
            probabilities[..., 0]
            + float(confidence_weight)
            * variances[..., 0].clamp(min=0.0).sqrt()
        ).clamp(min=0.0, max=1.0)
        margin = (
            (fix_lower - args.break_cost * break_upper) * weights
        ).sum(dim=-1).masked_fill(~active, -1e4)
        policies["lower_confidence_{:.2f}_sigma".format(
            float(confidence_weight)
        )] = policy_metrics(margin)

    active_probabilities = probabilities[active]
    active_targets = transition_targets(
        row_ious[active],
        fallback_iou.unsqueeze(1).expand_as(row_ious)[active],
    )
    classification = {}
    for index, suffix in enumerate(("025", "050")):
        target = active_targets[:, index]
        prediction = active_probabilities[:, index].argmax(dim=-1)
        classification[suffix] = {
            "accuracy": float(prediction.eq(target).float().mean().item()),
            "break_ratio": float(target.eq(0).float().mean().item()),
            "neutral_ratio": float(target.eq(1).float().mean().item()),
            "fix_ratio": float(target.eq(2).float().mean().item()),
        }
    return {
        "rows": len(rows),
        "baseline": {
            "acc025": float((fallback_iou > 0.25).float().mean().item()),
            "acc050": float((fallback_iou > 0.50).float().mean().item()),
        },
        "candidate_oracle": {
            "acc025": float((row_ious.masked_fill(~valid_mask, -1.0).max(dim=1).values > 0.25).float().mean().item()),
            "acc050": float((row_ious.masked_fill(~valid_mask, -1.0).max(dim=1).values > 0.50).float().mean().item()),
            "positive_rows": int(oracle.sum().item()),
        },
        "policies": policies,
        "transition_classification": classification,
    }


def actual_transition_utility(candidate_iou, fallback_iou, break_cost):
    utility = torch.zeros_like(candidate_iou, dtype=torch.float32)
    for threshold, weight in zip(THRESHOLDS, THRESHOLD_WEIGHTS):
        candidate_ok = candidate_iou > threshold
        fallback_ok = fallback_iou > threshold
        utility = utility + float(weight) * torch.where(
            ~fallback_ok & candidate_ok,
            torch.ones_like(utility),
            torch.where(
                fallback_ok & ~candidate_ok,
                torch.full_like(utility, -float(break_cost)),
                torch.zeros_like(utility),
            ),
        )
    return utility


def expected_transition_utility(candidate_probability, fallback_probability,
                                break_cost):
    fix_probability = (1.0 - fallback_probability) * candidate_probability
    break_probability = fallback_probability * (1.0 - candidate_probability)
    per_threshold = fix_probability - float(break_cost) * break_probability
    weights = per_threshold.new_tensor(THRESHOLD_WEIGHTS)
    return (per_threshold * weights).sum(dim=-1)


def evaluate_policy(model, rows, feature_indices, mean, std, args, device):
    row_features = torch.stack(
        tuple(row["features"][:, feature_indices].float() for row in rows)
    )
    row_ious = torch.stack(tuple(row["candidate_ious"].float() for row in rows))
    valid_mask = torch.stack(tuple(row["valid_mask"].bool() for row in rows))
    query_indices = torch.stack(tuple(row["query_indices"].long() for row in rows))
    fallback_query = torch.tensor(
        tuple(row["default_top1_query_index"] for row in rows), dtype=torch.long
    )
    fallback_match = valid_mask & query_indices.eq(fallback_query.unsqueeze(1))
    if not bool(fallback_match.sum(dim=1).eq(1).all().item()):
        raise ValueError("each row must contain its default fallback exactly once")
    fallback_index = fallback_match.float().argmax(dim=1)
    row_index = torch.arange(len(rows))
    active = valid_mask & ~fallback_match

    flat_probabilities, flat_variances = predict_tokens(
        model, row_features.flatten(0, 1), mean, std,
        args.batch_size, device,
    )
    probabilities = flat_probabilities.view(len(rows), row_features.shape[1], 2)
    fallback_probability = probabilities[row_index, fallback_index].unsqueeze(1)
    expected_utility = expected_transition_utility(
        probabilities, fallback_probability, args.break_cost
    ).masked_fill(~active, -1e4)

    fallback_iou = row_ious[row_index, fallback_index]
    actual_utility = actual_transition_utility(
        row_ious, fallback_iou.unsqueeze(1), args.break_cost
    ).masked_fill(~active, -1e4)
    oracle = actual_utility.max(dim=1).values > 0.0

    def policy_metrics(candidate_margin):
        best_margin, selected_index = candidate_margin.max(dim=1)
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
            "oracle_positive_rows": oracle_count,
            "oracle_recall": beneficial_count / float(max(oracle_count, 1)),
            "mean_selected_margin": float(
                best_margin[switch].mean().item() if switch_count else 0.0
            ),
        }

    policies = {"expected_utility": policy_metrics(expected_utility)}
    if flat_variances is not None:
        variances = flat_variances.view(
            len(rows), row_features.shape[1], 2
        )
        fallback_variance = variances[
            row_index, fallback_index
        ].unsqueeze(1)
        for confidence_weight in args.confidence_weights:
            candidate_lower = (
                probabilities
                - float(confidence_weight) * variances.clamp(min=0.0).sqrt()
            ).clamp(min=0.0, max=1.0)
            fallback_upper = (
                fallback_probability
                + float(confidence_weight)
                * fallback_variance.clamp(min=0.0).sqrt()
            ).clamp(min=0.0, max=1.0)
            lower_confidence_utility = expected_transition_utility(
                candidate_lower, fallback_upper, args.break_cost
            ).masked_fill(~active, -1e4)
            key = "lower_confidence_{:.2f}_sigma".format(
                float(confidence_weight)
            )
            policies[key] = policy_metrics(lower_confidence_utility)

    token_valid = valid_mask.flatten()
    token_probabilities = probabilities.flatten(0, 1)[token_valid]
    token_targets = threshold_targets(row_ious.flatten()[token_valid])
    token_metrics = {}
    for index, suffix in enumerate(("025", "050")):
        target = token_targets[:, index].numpy()
        probability = token_probabilities[:, index].numpy()
        token_metrics[suffix] = {
            "auroc": float(roc_auc_score(target, probability)),
            "average_precision": float(average_precision_score(target, probability)),
            "brier": float(((token_probabilities[:, index] - token_targets[:, index]) ** 2).mean().item()),
            "ece_15": binary_calibration_error(
                token_probabilities[:, index], token_targets[:, index]
            ),
            "positive_ratio": float(token_targets[:, index].mean().item()),
        }

    oracle_count = int(oracle.sum().item())
    return {
        "rows": len(rows),
        "baseline": {
            "acc025": float((fallback_iou > 0.25).float().mean().item()),
            "acc050": float((fallback_iou > 0.50).float().mean().item()),
        },
        "policies": policies,
        "candidate_oracle": {
            "acc025": float((row_ious.masked_fill(~valid_mask, -1.0).max(dim=1).values > 0.25).float().mean().item()),
            "acc050": float((row_ious.masked_fill(~valid_mask, -1.0).max(dim=1).values > 0.50).float().mean().item()),
            "positive_rows": oracle_count,
        },
        "token_quality": token_metrics,
    }


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1:
        raise ValueError("epochs, batch size, and hidden dim must be positive")
    if not math.isfinite(args.break_cost) or args.break_cost < 1.0:
        raise ValueError("break cost must be finite and at least one")
    if (not args.confidence_weights
            or any(not math.isfinite(value) or value < 0.0
                   for value in args.confidence_weights)):
        raise ValueError("confidence weights must be finite and non-negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = resolve_device(args.device)

    rows, manifest = load_candidate_cache(args.cache, expected_split="train")
    fit_rows, holdout_rows, holdout_scenes = split_rows_by_scene(
        rows, args.scene_modulus, args.holdout_remainder
    )
    feature_indices = select_feature_indices(
        manifest["feature_names"], args.include_projections
    )
    feature_names = [manifest["feature_names"][index] for index in feature_indices]
    results = {}
    for mode in args.modes:
        if mode in (
                "pair_evidential", "hierarchical_pair_evidential"):
            fit_features, fit_targets = flatten_pair_examples(
                fit_rows, feature_indices
            )
            mean = fit_features.mean(dim=0)
            std = fit_features.std(dim=0, unbiased=False).clamp(min=1e-6)
            model, history = fit_pair_probe(
                mode, fit_features, fit_targets, mean, std, args, device
            )
            results[mode] = evaluate_pair_policy(
                model, holdout_rows, feature_indices, mean, std, args, device
            )
        else:
            fit_features, fit_ious = flatten_tokens(fit_rows, feature_indices)
            fit_targets = threshold_targets(fit_ious)
            mean = fit_features.mean(dim=0)
            std = fit_features.std(dim=0, unbiased=False).clamp(min=1e-6)
            model, history = fit_probe(
                mode, fit_features, fit_targets, mean, std, args, device
            )
            results[mode] = evaluate_policy(
                model, holdout_rows, feature_indices, mean, std, args, device
            )
        results[mode]["fit_loss"] = history

    report = {
        "schema_version": 1,
        "purpose": "train-only-v40-structural-probe",
        "validation_cache_opened": False,
        "cache": str(Path(args.cache).expanduser().resolve()),
        "cache_checkpoint_sha256": manifest["checkpoint_sha256"],
        "cache_sample_count": manifest["sample_count"],
        "fit_rows": len(fit_rows),
        "holdout_rows": len(holdout_rows),
        "holdout_scene_count": len(holdout_scenes),
        "scene_split": {
            "modulus": args.scene_modulus,
            "holdout_remainder": args.holdout_remainder,
        },
        "thresholds": list(THRESHOLDS),
        "threshold_weights": list(THRESHOLD_WEIGHTS),
        "break_cost": args.break_cost,
        "fixed_deployment_boundary": 0.0,
        "feature_names": feature_names,
        "settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "modes": list(args.modes),
            "confidence_weights": list(args.confidence_weights),
        },
        "results": results,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
