#!/usr/bin/env python
"""V104 OOF: V103 training with preregistered IoU-priority selection."""

from pathlib import Path

from models.rec_relative_mask_policy import (
    ALLOWED_MASK_POLICY_INDICES,
    select_relative_mask_policy_iou_priority_ensemble,
)
from scripts import run_v103_relative_mask_oof as v103


SCHEMA = "rec-v104-iou-priority-relative-mask-scene-oof-v1"
SEEDS = v103.SEEDS
EPOCHS = v103.EPOCHS
TARGET_TEMPERATURE = v103.TARGET_TEMPERATURE
AGGREGATE_MARGIN = v103.AGGREGATE_MARGIN
BOOTSTRAP_SAMPLES = v103.BOOTSTRAP_SAMPLES


def _v104_selection_from_seed_outputs(seed_outputs, parents):
    prediction = select_relative_mask_policy_iou_priority_ensemble(
        seed_outputs,
        parents,
        aggregate_margin=AGGREGATE_MARGIN,
    )
    if not prediction.get("ranking") == "worst_delta_iou_then_aggregate":
        raise RuntimeError("V104 selector ranking contract changed")
    if not prediction["selected_parent_positions"].equal(parents):
        raise RuntimeError("V104 changed the frozen REC parent query")
    return prediction


def _source_identities(source_sha256):
    source_sha256["driver"] = v103.file_sha256(__file__)
    source_sha256["v103_training_driver"] = v103.file_sha256(
        Path(v103.__file__).resolve()
    )


def _patch_report(report):
    """Make the inherited report explicitly and audibly V104."""
    report["schema"] = SCHEMA
    report["protocol"]["selection"] = (
        "frozen_v101_parent_then_three_seed_worst_case_"
        "relative_mask_policy_iou_priority"
    )
    report["protocol"]["eligible_ranking"] = [
        "worst_delta_iou",
        "worst_aggregate",
        "lowest_original_policy_index",
    ]
    report["protocol"]["adaptive_development_iteration"] = True
    report["protocol"]["eligibility_changed_from_v103"] = False
    report["protocol"]["training_changed_from_v103"] = False
    _source_identities(report["source_sha256"])


def _patch_sidecar(sidecar):
    sidecar["schema"] = "rec-v104-iou-priority-oof-decisions-v1"
    _source_identities(sidecar["source_sha256"])


def main(argv=None):
    """Inject only the frozen V104 ranking into the audited V103 driver."""
    original_schema = v103.SCHEMA
    original_selection = v103._selection_from_seed_outputs
    original_json_writer = v103.write_exclusive_json
    original_torch_writer = v103.write_exclusive_torch

    def write_report(path, value):
        _patch_report(value)
        return original_json_writer(path, value)

    def write_sidecar(path, value):
        _patch_sidecar(value)
        return original_torch_writer(path, value)

    v103.SCHEMA = SCHEMA
    v103._selection_from_seed_outputs = _v104_selection_from_seed_outputs
    v103.write_exclusive_json = write_report
    v103.write_exclusive_torch = write_sidecar
    try:
        return v103.main(argv)
    finally:
        v103.SCHEMA = original_schema
        v103._selection_from_seed_outputs = original_selection
        v103.write_exclusive_json = original_json_writer
        v103.write_exclusive_torch = original_torch_writer


if __name__ == "__main__":
    main()
