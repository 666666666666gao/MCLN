"""Pure contracts for the train-only MCLN Optuna retraining study."""

from __future__ import annotations

import copy
import math
import os
import shutil
from pathlib import Path

from optuna.trial import TrialState


METRICS_SCHEMA = "mcln-retrain-metrics-v1"
EXPECTED_CALIBRATION_COUNT = 3625
MIN_FREE_BYTES = 8 * 1024 ** 3

DELTA_NAMES = (
    "position025",
    "position050",
    "mask025",
    "mask050",
    "mask_miou",
)

_SEED_PRESETS = (
    {
        "decoder_lr": 2e-5,
        "mask_head_lr_multiplier": 1.0,
        "selector_lr": 5e-4,
        "mask_loss_scale": 1.0,
        "consistency_loss_scale": 1.0,
        "selector_loss_weight": 0.5,
        "selector_min_iou_gap": 0.03,
    },
    {
        "decoder_lr": 1e-5,
        "mask_head_lr_multiplier": 4.0,
        "selector_lr": 5e-4,
        "mask_loss_scale": 2.0,
        "consistency_loss_scale": 0.5,
        "selector_loss_weight": 0.5,
        "selector_min_iou_gap": 0.03,
    },
    {
        "decoder_lr": 8e-6,
        "mask_head_lr_multiplier": 2.0,
        "selector_lr": 1e-3,
        "mask_loss_scale": 2.0,
        "consistency_loss_scale": 0.25,
        "selector_loss_weight": 0.2,
        "selector_min_iou_gap": 0.05,
    },
)


def seed_presets():
    """Return defensive copies of the three predeclared parameter presets."""
    return tuple(copy.deepcopy(preset) for preset in _SEED_PRESETS)


def suggest_trial_params(trial):
    """Resolve the exact approved seven-dimensional search space."""
    return {
        "decoder_lr": trial.suggest_float(
            "decoder_lr", 5e-6, 4e-5, log=True
        ),
        "mask_head_lr_multiplier": trial.suggest_categorical(
            "mask_head_lr_multiplier", [1.0, 2.0, 4.0]
        ),
        "selector_lr": trial.suggest_float(
            "selector_lr", 2e-4, 2e-3, log=True
        ),
        "mask_loss_scale": trial.suggest_float(
            "mask_loss_scale", 0.5, 4.0, log=True
        ),
        "consistency_loss_scale": trial.suggest_float(
            "consistency_loss_scale", 0.1, 2.0, log=True
        ),
        "selector_loss_weight": trial.suggest_float(
            "selector_loss_weight", 0.1, 1.0, log=True
        ),
        "selector_min_iou_gap": trial.suggest_categorical(
            "selector_min_iou_gap", [0.02, 0.03, 0.05, 0.08]
        ),
    }


def _require_exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("{} fields do not match the metrics schema".format(label))


def _require_nonnegative_integer(value, label, maximum=None):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("{} must be a non-negative integer".format(label))
    if maximum is not None and value > maximum:
        raise ValueError("{} exceeds sample_count".format(label))
    return value


def _require_finite_float(value, label, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValueError("{} must be finite".format(label))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("{} must be finite".format(label))
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(label))
    if minimum is not None and result < minimum:
        raise ValueError("{} is below its valid range".format(label))
    if maximum is not None and result > maximum:
        raise ValueError("{} exceeds its valid range".format(label))
    return result


def validate_metrics_receipt(
        receipt, expected_sample_count=EXPECTED_CALIBRATION_COUNT):
    """Validate and defensively copy one exact evaluator metrics receipt."""
    _require_exact_keys(
        receipt, ("schema", "sample_count", "position", "mask"), "receipt"
    )
    if receipt["schema"] != METRICS_SCHEMA:
        raise ValueError("metrics receipt schema is invalid")

    expected_sample_count = _require_nonnegative_integer(
        expected_sample_count, "expected_sample_count"
    )
    if expected_sample_count <= 0:
        raise ValueError("expected_sample_count must be positive")
    sample_count = _require_nonnegative_integer(
        receipt["sample_count"], "sample_count"
    )
    if sample_count != expected_sample_count:
        raise ValueError(
            "metrics receipt contains {} samples, expected {}".format(
                sample_count, expected_sample_count
            )
        )

    position = receipt["position"]
    _require_exact_keys(
        position, ("fixed_default", "learned_selector"), "position"
    )
    for source_name in ("fixed_default", "learned_selector"):
        source = position[source_name]
        _require_exact_keys(source, ("hits025", "hits050"), source_name)
        hits025 = _require_nonnegative_integer(
            source["hits025"], "{} hits025".format(source_name), sample_count
        )
        hits050 = _require_nonnegative_integer(
            source["hits050"], "{} hits050".format(source_name), sample_count
        )
        if hits050 > hits025:
            raise ValueError(
                "{} hits050 cannot exceed hits025".format(source_name)
            )

    mask = receipt["mask"]
    _require_exact_keys(
        mask, ("hits025", "hits050", "iou_sum", "miou"), "mask"
    )
    mask_hits025 = _require_nonnegative_integer(
        mask["hits025"], "mask hits025", sample_count
    )
    mask_hits050 = _require_nonnegative_integer(
        mask["hits050"], "mask hits050", sample_count
    )
    if mask_hits050 > mask_hits025:
        raise ValueError("mask hits050 cannot exceed hits025")
    iou_sum = _require_finite_float(
        mask["iou_sum"], "mask iou_sum", 0.0, float(sample_count)
    )
    miou = _require_finite_float(mask["miou"], "mask miou", 0.0, 1.0)
    expected_miou = iou_sum / float(sample_count)
    if not math.isclose(miou, expected_miou, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("mask miou is inconsistent with iou_sum")
    return copy.deepcopy(receipt)


def objective_from_deltas(deltas):
    """Return the approved weakest-link plus balanced-mean objective."""
    _require_exact_keys(deltas, DELTA_NAMES, "deltas")
    values = tuple(
        _require_finite_float(deltas[name], "delta {}".format(name))
        for name in DELTA_NAMES
    )
    return 100.0 * (min(values) + 0.25 * sum(values) / len(values))


def assess_trial_metrics(baseline, trial_metrics):
    """Assess one epoch-56 trial against the epoch-54 calibration baseline."""
    baseline = validate_metrics_receipt(baseline)
    trial_metrics = validate_metrics_receipt(trial_metrics)
    count = float(EXPECTED_CALIBRATION_COUNT)

    deltas = {
        "position025": (
            trial_metrics["position"]["learned_selector"]["hits025"]
            - baseline["position"]["fixed_default"]["hits025"]
        ) / count,
        "position050": (
            trial_metrics["position"]["learned_selector"]["hits050"]
            - baseline["position"]["fixed_default"]["hits050"]
        ) / count,
        "mask025": (
            trial_metrics["mask"]["hits025"]
            - baseline["mask"]["hits025"]
        ) / count,
        "mask050": (
            trial_metrics["mask"]["hits050"]
            - baseline["mask"]["hits050"]
        ) / count,
        "mask_miou": (
            trial_metrics["mask"]["miou"] - baseline["mask"]["miou"]
        ),
    }

    failures = []
    trial_position = trial_metrics["position"]
    for suffix in ("025", "050"):
        if (
            trial_position["learned_selector"]["hits" + suffix]
            < trial_position["fixed_default"]["hits" + suffix]
        ):
            failures.append(
                "selector_position{}_below_fixed_default".format(suffix)
            )
    for name in DELTA_NAMES:
        if deltas[name] < 0.0:
            failures.append("{}_below_baseline".format(name))

    feasible = not failures
    return {
        "feasible": feasible,
        "objective": objective_from_deltas(deltas) if feasible else None,
        "deltas": deltas,
        "constraint_failures": tuple(failures),
    }


def _validated_selection_candidate(candidate):
    if not isinstance(candidate, dict):
        raise ValueError("selection candidate must be a mapping")
    trial_number = _require_nonnegative_integer(
        candidate.get("trial_number"), "trial_number"
    )
    if candidate.get("feasible") is not True:
        return None
    metrics = validate_metrics_receipt(candidate.get("metrics"))
    deltas = candidate.get("deltas")
    expected_objective = objective_from_deltas(deltas)
    objective = _require_finite_float(
        candidate.get("objective"), "candidate objective"
    )
    if not math.isclose(
        objective, expected_objective, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("candidate objective is inconsistent with deltas")
    result = copy.deepcopy(candidate)
    result["trial_number"] = trial_number
    result["metrics"] = metrics
    result["objective"] = objective
    return result


def select_best_trial(candidates):
    """Choose a feasible trial using the approved deterministic tie-break."""
    feasible = []
    for candidate in candidates:
        validated = _validated_selection_candidate(candidate)
        if validated is not None:
            feasible.append(validated)
    if not feasible:
        return None

    def selection_key(candidate):
        metrics = candidate["metrics"]
        return (
            candidate["objective"],
            metrics["position"]["learned_selector"]["hits025"],
            metrics["position"]["learned_selector"]["hits050"],
            metrics["mask"]["hits050"],
            metrics["mask"]["miou"],
            -candidate["trial_number"],
        )

    return max(feasible, key=selection_key)


def count_completed_trials(trials, receipt_is_valid=None):
    """Count unique COMPLETE trials whose receipt passes structural checks."""
    if receipt_is_valid is None:
        receipt_is_valid = lambda trial: True
    if not callable(receipt_is_valid):
        raise ValueError("receipt_is_valid must be callable")

    completed_numbers = set()
    for trial in trials:
        if getattr(trial, "state", None) != TrialState.COMPLETE:
            continue
        number = _require_nonnegative_integer(
            getattr(trial, "number", None), "trial number"
        )
        if receipt_is_valid(trial):
            completed_numbers.add(number)
    return len(completed_numbers)


def require_minimum_free_space(path, reported_free_bytes=None):
    """Fail closed unless the target filesystem has at least eight GiB free."""
    path = Path(path)
    if reported_free_bytes is None:
        reported_free_bytes = shutil.disk_usage(str(path)).free
    if (
        not isinstance(reported_free_bytes, int)
        or isinstance(reported_free_bytes, bool)
        or reported_free_bytes < 0
    ):
        raise ValueError("reported free space must be a non-negative integer")
    if reported_free_bytes < MIN_FREE_BYTES:
        raise ValueError(
            "free space {} is below required {}".format(
                reported_free_bytes, MIN_FREE_BYTES
            )
        )
    return reported_free_bytes


def _normalized_path(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def cleanup_trial_checkpoints(trial_directory, keep_checkpoint=None):
    """Remove non-best trial weights without Python 3.8 ``missing_ok``."""
    trial_directory = Path(trial_directory)
    if not trial_directory.is_dir():
        raise ValueError("trial checkpoint directory does not exist")

    keep_normalized = None
    if keep_checkpoint is not None:
        keep_checkpoint = Path(keep_checkpoint)
        if not keep_checkpoint.exists():
            raise ValueError("global best checkpoint does not exist")
        keep_normalized = _normalized_path(keep_checkpoint)

    checkpoints = [
        path for path in sorted(trial_directory.glob("*.pth"))
        if path.exists()
    ]
    removed = []
    for checkpoint in checkpoints:
        if (
            keep_normalized is not None
            and _normalized_path(checkpoint) == keep_normalized
        ):
            continue
        if not checkpoint.exists():
            continue
        try:
            checkpoint.unlink()
        except FileNotFoundError:
            continue
        removed.append(checkpoint)
    return tuple(removed)
