import scripts.run_v101_full_train_pareto_oof as v101


def _diagnostics(delta025=72, delta050=74, fold025=1, fold050=1,
                 lower025=1, lower050=1):
    return {
        "delta_hits025": delta025,
        "delta_hits050": delta050,
        "fold_deltas": {
            str(i): {"hits025": fold025, "hits050": fold050}
            for i in range(5)
        },
        "bootstrap025": {"lower_bound_95": lower025},
        "bootstrap050": {"lower_bound_95": lower050},
    }


def test_fixed_preregistered_constants():
    assert v101.EXPECTED_SAMPLE_COUNT == 36665
    assert v101.EXPECTED_SCENE_COUNT == 562
    assert v101.MIN_DELTA_025 == 72
    assert v101.MIN_DELTA_050 == 74
    assert v101.v99.V97_MARGIN == 0.13312220573425293


def test_acceptance_gate_passes_only_all_predicates():
    predicates = v101.acceptance_gate(_diagnostics())
    assert all(predicates.values())


def test_acceptance_gate_rejects_zero_fold_delta():
    predicates = v101.acceptance_gate(_diagnostics(fold025=0))
    assert not predicates["all_folds_strictly_positive025"]
    assert not all(predicates.values())


def test_acceptance_gate_rejects_below_scaled_gap():
    predicates = v101.acceptance_gate(_diagnostics(delta050=73))
    assert not predicates["delta050_at_least_oracle_scaled_gap"]
    assert not all(predicates.values())


def test_acceptance_gate_rejects_nonpositive_bootstrap_bound():
    predicates = v101.acceptance_gate(_diagnostics(lower025=0))
    assert not predicates["bootstrap025_lower_bound_positive"]
    assert not all(predicates.values())
