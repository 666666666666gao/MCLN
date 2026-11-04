#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import random


def intersection_size(low, high, target_low, target_high):
    return max(0., min(high, target_high) - max(low, target_low))


def gate_iou(native, reference, target, gates):
    intersection = volume = target_volume = 1.
    for axis, gate in enumerate(gates):
        center = native[axis] + gate * (reference[axis] - native[axis])
        size = native[axis + 3] + gate * (reference[axis + 3] - native[axis + 3])
        intersection *= intersection_size(
            center - size / 2, center + size / 2,
            target[axis] - target[axis + 3] / 2, target[axis] + target[axis + 3] / 2)
        volume *= size
        target_volume *= target[axis + 3]
    return intersection / (volume + target_volume - intersection)


def continuous_oracle(native, reference, target):
    options = []
    for axis in range(3):
        native_low = native[axis] - native[axis + 3] / 2
        native_high = native[axis] + native[axis + 3] / 2
        reference_low = reference[axis] - reference[axis + 3] / 2
        reference_high = reference[axis] + reference[axis + 3] / 2
        target_low = target[axis] - target[axis + 3] / 2
        target_high = target[axis] + target[axis + 3] / 2
        gates = {0., 1.}
        for start, end in ((native_low, reference_low), (native_high, reference_high)):
            if start != end:
                for boundary in (target_low, target_high):
                    gate = (boundary - start) / (end - start)
                    if 0 < gate < 1:
                        gates.add(gate)
        axis_options = []
        for gate in sorted(gates):
            low = native_low + gate * (reference_low - native_low)
            high = native_high + gate * (reference_high - native_high)
            size = native[axis + 3] + gate * (reference[axis + 3] - native[axis + 3])
            axis_options.append((intersection_size(low, high, target_low, target_high), size))
        options.append(axis_options)
    target_volume = target[3] * target[4] * target[5]
    assert target_volume > 1e-8
    best = 0.
    for x, y, z in itertools.product(*options):
        intersection = x[0] * y[0] * z[0]
        volume = x[1] * y[1] * z[1]
        best = max(best, intersection / (volume + target_volume - intersection))
    return best


# Remote numerical witnesses supplement the coordinatewise fractional proof.
assert abs(continuous_oracle([-1, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1],
                             [.13, 0, 0, 1, 1, 1]) - 1) < 1e-12
fixed = [0, 0, 0, 1, 1, 1]
assert continuous_oracle(fixed, fixed, fixed) == 1.
generator = random.Random(0)
largest_grid_excess = 0.
for _ in range(64):
    boxes = [[generator.uniform(-2, 2) for _ in range(3)]
             + [generator.uniform(.05, 2) for _ in range(3)] for _ in range(3)]
    optimum = continuous_oracle(*boxes)
    grid = max(gate_iou(*boxes, gates)
               for gates in itertools.product([i / 10 for i in range(11)], repeat=3))
    largest_grid_excess = max(largest_grid_excess, grid - optimum)
    assert grid <= optimum + 1e-12

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits")
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "GT-only continuous gate oracle for archived fixed selected queries. No model import, inference, training, or claimed attainable accuracy.",
    "algorithm": "Evaluate Cartesian products of endpoint/GT-face crossing gates. Within each cell, IoU is linear-fractional in each coordinate with positive denominator; moving each coordinate to an endpoint cannot decrease the optimum.",
    "numerical_witness": {"random_cases": 64, "grid_points_each": 1331,
                          "largest_grid_excess": largest_grid_excess,
                          "constant_box_and_interior_exact_match_passed": True},
    "conditions": {},
}
for name, directory in (
    ("b2", root / "b2_fixed_audit_20261103_lrpositive"),
    ("reader", root / "reader_result_20261104"),
):
    path = directory / "none.jsonl"
    rows, footer = [], None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "semantic_sample":
                rows.append(row)
            elif row["kind"] == "forward_complete":
                footer = row
    assert footer and footer["verified_against_evaluator"] and len(rows) == 9508
    counts = {suffix: {"actual_hits": 0, "continuous_gate_oracle_hits": 0,
                      "oracle_gains": 0, "numerically_near_threshold_1e_minus9": 0,
                      "minimum_threshold_distance": 1.}
              for suffix in ("025", "050")}
    for row in rows:
        native = row["native_box"][:3] + [max(value, 1e-6) for value in row["native_box"][3:]]
        reference = row["reference_box"][:3] + [max(value, 1e-6) for value in row["reference_box"][3:]]
        target = row["gt_box"]
        optimum = continuous_oracle(native, reference, target)
        observed = gate_iou(native, reference, target, row["selected_clamped_gate"])
        assert optimum >= observed - 1e-12
        for suffix, threshold in (("025", .25), ("050", .5)):
            actual = row["selected_iou"] > threshold
            oracle = optimum > threshold
            current = counts[suffix]
            current["actual_hits"] += actual
            current["continuous_gate_oracle_hits"] += oracle
            current["oracle_gains"] += oracle and not actual
            current["numerically_near_threshold_1e_minus9"] += abs(optimum - threshold) <= 1e-9
            current["minimum_threshold_distance"] = min(
                current["minimum_threshold_distance"], abs(optimum - threshold))
    for suffix, current in counts.items():
        assert current["actual_hits"] == footer["semantic_hits" + suffix]
    result["conditions"][name] = {
        "samples": len(rows), "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "thresholds": counts,
    }
output = root / "reader_result_20261104/span-continuous-oracle-summary.json"
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
