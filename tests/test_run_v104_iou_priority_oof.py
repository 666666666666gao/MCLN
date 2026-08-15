import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "run_v104_iou_priority_oof.py"
)
SPEC = importlib.util.spec_from_file_location("v104_oof", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v104_inherits_every_training_and_gate_constant():
    assert MODULE.SEEDS == (0, 1, 2)
    assert MODULE.EPOCHS == 12
    assert MODULE.TARGET_TEMPERATURE == 0.25
    assert MODULE.AGGREGATE_MARGIN == 0.02
    assert MODULE.ALLOWED_MASK_POLICY_INDICES == tuple(range(4, 15))
    assert MODULE.BOOTSTRAP_SAMPLES == 10000


def test_report_patch_records_only_preregistered_selector_change():
    report = {
        "schema": "old",
        "protocol": {"selection": "old"},
        "source_sha256": {"driver": "old"},
    }
    MODULE._patch_report(report)
    assert report["schema"] == MODULE.SCHEMA
    assert report["protocol"]["selection"] == (
        "frozen_v101_parent_then_three_seed_worst_case_"
        "relative_mask_policy_iou_priority"
    )
    assert report["protocol"]["eligible_ranking"] == [
        "worst_delta_iou", "worst_aggregate", "lowest_original_policy_index"
    ]
    assert report["protocol"]["adaptive_development_iteration"] is True
    assert set(report["source_sha256"]) == {
        "driver", "v103_training_driver"
    }


def test_sidecar_patch_has_distinct_v104_schema_and_sources():
    sidecar = {"schema": "old", "source_sha256": {"driver": "old"}}
    MODULE._patch_sidecar(sidecar)
    assert sidecar["schema"] == "rec-v104-iou-priority-oof-decisions-v1"
    assert set(sidecar["source_sha256"]) == {
        "driver", "v103_training_driver"
    }
