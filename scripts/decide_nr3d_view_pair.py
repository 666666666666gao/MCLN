"""Compare the two complete G0 role receipts without configuration selection."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nr3d_view_pair_contract import CHECKPOINT_SHA, EXPECTED, compare_rows, digest_ids


def decide(old, fixed):
    for role, receipt in (("old", old), ("fixed", fixed)):
        if (receipt["role"] != role or receipt["status"] != "complete" or
                receipt["checkpoint_sha256"] != CHECKPOINT_SHA or
                receipt["weights_written"] != 0 or
                receipt["formal_validation_dataset_constructed"] is not False):
            raise ValueError("role completion/input contract failed")
        if (receipt["training"]["sample_count"] != 25768 or
                receipt["training"]["sample_identity_sha256"] != EXPECTED["fit"][3] or
                receipt["training"]["optimizer_steps"] != 1611 or
                receipt["training"]["batch_count"] != 1611 or
                len(receipt["rows"]) != 7151 or
                digest_ids(row["id"] for row in receipt["rows"]) != EXPECTED["holdout"][3] or
                sum(row["view_dependent"] for row in receipt["rows"]) != 2718):
            raise ValueError("role does not cover the frozen complete pair")
    for field in ("census", "optimizer_groups", "scheduler_milestones",
                  "optimizer_initialization", "fit_batches", "holdout_batches"):
        if old[field] != fixed[field]:
            raise ValueError("paired contract mismatch: " + field)
    if old["training"]["sample_order_sha256"] != fixed["training"]["sample_order_sha256"]:
        raise ValueError("actual paired sampler order differs")
    result = compare_rows(old["rows"], fixed["rows"])
    result.update(schema="mcln-nr3d-view-pair-decision-v2", integrity_pass=True,
                  formal_validation_evaluated=False, weights_written=0,
                  next_stage=("G1_preregistration" if result["scientific_gate_pass"]
                              else "seal_performance_route_keep_data_fix"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    args = parser.parse_args()
    old = json.loads((args.pair_root / "old/receipt.json").read_text())
    fixed = json.loads((args.pair_root / "fixed/receipt.json").read_text())
    decision = decide(old, fixed)
    with open(str(args.pair_root / "decision.json"), "x") as stream:
        json.dump(decision, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(decision, indent=2))
