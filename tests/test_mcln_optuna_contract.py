import math
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
from optuna.trial import TrialState

from scripts.tuning.mcln_optuna_contract import (
    EXPECTED_CALIBRATION_COUNT,
    METRICS_SCHEMA,
    MIN_FREE_BYTES,
    assess_trial_metrics,
    cleanup_trial_checkpoints,
    count_completed_trials,
    require_minimum_free_space,
    seed_presets,
    select_best_trial,
    suggest_trial_params,
    validate_metrics_receipt,
)


def _metrics(
        fixed025=2200,
        fixed050=1700,
        learned025=2200,
        learned050=1700,
        mask025=2250,
        mask050=1850,
        iou_sum=1600.0,
        sample_count=EXPECTED_CALIBRATION_COUNT):
    return {
        "schema": METRICS_SCHEMA,
        "sample_count": sample_count,
        "position": {
            "fixed_default": {
                "hits025": fixed025,
                "hits050": fixed050,
            },
            "learned_selector": {
                "hits025": learned025,
                "hits050": learned050,
            },
        },
        "mask": {
            "hits025": mask025,
            "hits050": mask050,
            "iou_sum": iou_sum,
            "miou": iou_sum / float(sample_count),
        },
    }


def _baseline():
    return _metrics()


def _improved_trial(**overrides):
    values = {
        "fixed025": 2210,
        "fixed050": 1710,
        "learned025": 2220,
        "learned050": 1720,
        "mask025": 2260,
        "mask050": 1860,
        "iou_sum": 1610.0,
    }
    values.update(overrides)
    return _metrics(**values)


def _candidate(trial_number, metrics):
    assessment = assess_trial_metrics(_baseline(), metrics)
    return {
        "trial_number": trial_number,
        "metrics": metrics,
        "feasible": assessment["feasible"],
        "objective": assessment["objective"],
        "deltas": assessment["deltas"],
    }


def test_validate_metrics_receipt_accepts_exact_calibration_receipt():
    receipt = _baseline()

    validated = validate_metrics_receipt(receipt)

    assert validated == receipt
    assert validated is not receipt
    assert validated["position"] is not receipt["position"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema="wrong"),
        lambda value: value.update(sample_count=3624),
        lambda value: value.update(unexpected=True),
        lambda value: value["position"].pop("fixed_default"),
        lambda value: value["mask"].update(miou=math.nan),
        lambda value: value["mask"].update(iou_sum=math.inf),
        lambda value: value["mask"].update(miou=0.123),
        lambda value: value["mask"].update(hits025=1800, hits050=1801),
        lambda value: value["position"]["fixed_default"].update(
            hits025=1600, hits050=1601
        ),
        lambda value: value["position"]["learned_selector"].update(
            hits025=-1
        ),
        lambda value: value["position"]["learned_selector"].update(
            hits025=1.5
        ),
    ],
)
def test_validate_metrics_receipt_rejects_schema_and_counter_drift(mutate):
    receipt = _baseline()
    mutate(receipt)

    with pytest.raises(ValueError):
        validate_metrics_receipt(receipt)


def test_assessment_uses_all_five_deltas_and_balanced_objective():
    assessment = assess_trial_metrics(_baseline(), _improved_trial())

    assert assessment["feasible"] is True
    assert set(assessment["deltas"]) == {
        "position025",
        "position050",
        "mask025",
        "mask050",
        "mask_miou",
    }
    expected = 100.0 * (
        min(assessment["deltas"].values())
        + 0.25 * sum(assessment["deltas"].values()) / 5.0
    )
    assert assessment["objective"] == pytest.approx(expected, abs=1e-12)
    assert assessment["constraint_failures"] == ()


@pytest.mark.parametrize(
    "overrides,expected_failure",
    [
        ({"learned025": 2209}, "selector_position025_below_fixed_default"),
        ({"learned050": 1709}, "selector_position050_below_fixed_default"),
        ({"learned025": 2199}, "position025_below_baseline"),
        ({"learned050": 1699}, "position050_below_baseline"),
        ({"mask025": 2249}, "mask025_below_baseline"),
        ({"mask050": 1849}, "mask050_below_baseline"),
        ({"iou_sum": 1599.0}, "mask_miou_below_baseline"),
    ],
)
def test_assessment_rejects_each_protection_constraint(
        overrides, expected_failure):
    assessment = assess_trial_metrics(
        _baseline(), _improved_trial(**overrides)
    )

    assert assessment["feasible"] is False
    assert assessment["objective"] is None
    assert expected_failure in assessment["constraint_failures"]


def test_select_best_trial_uses_approved_tie_break_order():
    worse = _candidate(
        3, _improved_trial(learned025=2221, learned050=1720)
    )
    better = _candidate(
        4, _improved_trial(learned025=2222, learned050=1719)
    )
    tied_later = _candidate(
        9, _improved_trial(learned025=2222, learned050=1719)
    )

    assert worse["objective"] == pytest.approx(better["objective"])

    assert select_best_trial([worse, better, tied_later])["trial_number"] == 4


def test_select_best_trial_ignores_infeasible_trials_and_rejects_no_best():
    infeasible = _candidate(1, _improved_trial(mask050=1849))
    feasible = _candidate(2, _improved_trial())

    assert select_best_trial([infeasible, feasible])["trial_number"] == 2
    assert select_best_trial([infeasible]) is None


APPROVED_PRESETS = (
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


def test_seed_presets_are_exact_and_defensive():
    first = seed_presets()
    second = seed_presets()

    assert first == APPROVED_PRESETS
    assert second == APPROVED_PRESETS
    assert first is not second
    assert all(a is not b for a, b in zip(first, second))


@pytest.mark.parametrize("preset", APPROVED_PRESETS)
def test_suggest_trial_params_has_exact_names_bounds_and_distributions(preset):
    trial = optuna.trial.FixedTrial(dict(preset))

    assert suggest_trial_params(trial) == preset
    assert set(trial.distributions) == set(preset)

    for name, low, high in (
        ("decoder_lr", 5e-6, 4e-5),
        ("selector_lr", 2e-4, 2e-3),
        ("mask_loss_scale", 0.5, 4.0),
        ("consistency_loss_scale", 0.1, 2.0),
        ("selector_loss_weight", 0.1, 1.0),
    ):
        distribution = trial.distributions[name]
        assert distribution.low == low
        assert distribution.high == high
        assert distribution.log is True

    assert trial.distributions["mask_head_lr_multiplier"].choices == (
        1.0, 2.0, 4.0
    )
    assert trial.distributions["selector_min_iou_gap"].choices == (
        0.02, 0.03, 0.05, 0.08
    )


def test_count_completed_trials_requires_complete_state_and_valid_receipt():
    trials = [
        SimpleNamespace(state=TrialState.COMPLETE, number=0),
        SimpleNamespace(state=TrialState.FAIL, number=1),
        SimpleNamespace(state=TrialState.RUNNING, number=2),
        SimpleNamespace(state=TrialState.COMPLETE, number=3),
    ]

    assert count_completed_trials(
        trials, receipt_is_valid=lambda trial: trial.number == 3
    ) == 1


@pytest.mark.parametrize(
    "reported_free_bytes",
    [MIN_FREE_BYTES - 1, -1, math.nan, math.inf, "lots"],
)
def test_disk_preflight_rejects_invalid_or_low_reported_space(
        tmp_path, reported_free_bytes):
    with pytest.raises(ValueError):
        require_minimum_free_space(
            tmp_path, reported_free_bytes=reported_free_bytes
        )


def test_disk_preflight_accepts_exact_minimum(tmp_path):
    assert require_minimum_free_space(
        tmp_path, reported_free_bytes=MIN_FREE_BYTES
    ) == MIN_FREE_BYTES


def test_cleanup_keeps_only_global_best_and_uses_python37_unlink(
        tmp_path, monkeypatch):
    best = tmp_path / "best.pth"
    stale = tmp_path / "stale.pth"
    duplicate = tmp_path / "last.pth"
    unrelated = tmp_path / "metrics.json"
    for path in (best, stale, duplicate, unrelated):
        path.write_bytes(path.name.encode("ascii"))

    real_unlink = Path.unlink
    calls = []

    def recording_unlink(path, *args, **kwargs):
        calls.append((path, args, kwargs))
        assert args == ()
        assert kwargs == {}
        return real_unlink(path)

    monkeypatch.setattr(Path, "unlink", recording_unlink)

    removed = cleanup_trial_checkpoints(tmp_path, keep_checkpoint=best)

    assert removed == (duplicate, stale)
    assert best.exists()
    assert unrelated.exists()
    assert not stale.exists()
    assert not duplicate.exists()
    assert all(call[2] == {} for call in calls)


def test_cleanup_tolerates_a_checkpoint_disappearing_before_unlink(
        tmp_path, monkeypatch):
    vanishing = tmp_path / "vanishing.pth"
    vanishing.write_bytes(b"checkpoint")
    real_exists = Path.exists
    checks = {vanishing: 0}

    def disappearing_exists(path):
        if path == vanishing:
            checks[path] += 1
            if checks[path] == 2:
                real_unlink = Path.unlink
                real_unlink(path)
                return False
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", disappearing_exists)

    assert cleanup_trial_checkpoints(tmp_path) == ()
    assert not real_exists(vanishing)
