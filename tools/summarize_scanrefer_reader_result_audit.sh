#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import json
from pathlib import Path
import sys

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits")
base = root / "b2_fixed_audit_20261103_lrpositive"
audit = root / "reader_result_20261104"
assert (audit / "complete.json").is_file()
assert (audit / "pipeline_exit_code").read_text().strip() == "0"
sys.path.insert(0, str(base))
from summarize_pvground_fixed_audit import IDENTITY_KEYS, read_complete

baseline, left = read_complete(
    base / "none.jsonl", "none",
    "46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8")
treatment, right = read_complete(
    audit / "none.jsonl", "none",
    "67cef44cd6ec253806eced03b1ca79f115d13d4e681c9da69730ca767b0fda35")
for a, b in zip(left, right):
    assert all(a[key] == b[key] for key in IDENTITY_KEYS), a["sample_index"]

results = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "Different trained checkpoints, exact input pairing; not an isolated Reader intervention or matched trained-control comparison.",
    "paired_samples": len(left),
    "exact_order_identity_points_and_gt_match": True,
    "baseline": baseline, "reader_trial": treatment, "transitions": {},
}
for threshold, suffix in ((0.25, "025"), (0.5, "050")):
    transitions = {}
    for label, wanted in (("gained", (False, True)), ("lost", (True, False))):
        pairs = [(a, b) for a, b in zip(left, right)
                 if (a["selected_iou"] > threshold, b["selected_iou"] > threshold) == wanted]
        failed = [a if label == "gained" else b for a, b in pairs]
        transitions[label] = {
            "samples": len(pairs),
            "unique": sum(a["is_unique"] for a, b in pairs),
            "multiple": sum(not a["is_unique"] for a, b in pairs),
            "failed_checkpoint_no_valid_candidate": sum(
                row["oracle_iou"] <= threshold for row in failed),
            "failed_checkpoint_valid_candidate_outside_top10": sum(
                row["oracle_iou"] > threshold
                and not any(iou > threshold for iou in row["top10_ious"]) for row in failed),
            "failed_checkpoint_valid_candidate_in_top10": sum(
                any(iou > threshold for iou in row["top10_ious"]) for row in failed),
        }
    transitions["net_hits"] = transitions["gained"]["samples"] - transitions["lost"]["samples"]
    assert transitions["net_hits"] == treatment[f"hits{suffix}"] - baseline[f"hits{suffix}"]
    results["transitions"][suffix] = transitions
output = audit / "paired-summary.json"
output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
print(json.dumps({"output": str(output), "transitions": results["transitions"]}))
PY
