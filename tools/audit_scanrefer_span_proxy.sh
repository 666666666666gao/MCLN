#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import itertools
import json
from pathlib import Path

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits")
grid = tuple(itertools.product((0., .25, .5, .75, 1.), repeat=3))

def iou(box, target):
    intersection = 1.
    for axis in range(3):
        low = max(box[axis] - box[axis + 3] / 2, target[axis] - target[axis + 3] / 2)
        high = min(box[axis] + box[axis + 3] / 2, target[axis] + target[axis + 3] / 2)
        intersection *= max(0., high - low)
    volume = box[3] * box[4] * box[5]
    target_volume = target[3] * target[4] * target[5]
    return intersection / max(volume + target_volume - intersection, 1e-8)

def mix(native, reference, gates):
    return [(1 - gates[axis % 3]) * native[axis] + gates[axis % 3] * reference[axis]
            for axis in range(6)]

result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "GT-based selected-query geometry diagnostic only; grid/proxy use GT and are not inference, achievable gains, or trained-method results.",
    "grid_values": [0., .25, .5, .75, 1.],
    "conditions": {},
}
for name, directory in (
    ("b2", root / "b2_fixed_audit_20261103_lrpositive"),
    ("reader", root / "reader_result_20261104"),
):
    path = directory / "none.jsonl"
    rows = []
    footer = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "semantic_sample":
                rows.append(row)
            elif row["kind"] == "forward_complete":
                footer = row
    assert footer and footer["verified_against_evaluator"] and len(rows) == 9508
    counts = {suffix: {key: 0 for key in (
        "final_hits", "proxy_hits", "grid_oracle_hits",
        "proxy_gains", "proxy_losses",
        "reference_valid_but_proxy_fails", "grid_fixes_proxy_failure",
        "proxy_gains_with_raw_gate_outside_clamp",
        "proxy_gains_with_all_raw_gates_outside_clamp",
        "proxy_gains_with_outside_axis_target_different",
    )} for suffix in ("025", "050")}
    saturation = {
        "axes": len(rows) * 3, "raw_below_zero": 0, "raw_above_one": 0,
        "samples_with_any_outside_axis": 0,
        "outside_axes_with_projection_target_different": 0,
    }
    max_reconstruction_error = 0.
    for row in rows:
        native = row["native_box"][:3] + [max(value, 1e-6) for value in row["native_box"][3:]]
        reference = row["reference_box"][:3] + [max(value, 1e-6) for value in row["reference_box"][3:]]
        target = row["gt_box"]
        reconstructed = mix(native, reference, row["selected_clamped_gate"])
        max_reconstruction_error = max(max_reconstruction_error, max(
            abs(a - b) for a, b in zip(reconstructed, row["final_box"])))
        gates = []
        for axis in range(3):
            dc = reference[axis] - native[axis]
            ds = reference[axis + 3] - native[axis + 3]
            projection = ((target[axis] - native[axis]) * dc
                          + (target[axis + 3] - native[axis + 3]) * ds)
            gates.append(min(1., max(0., projection / max(dc * dc + ds * ds, 1e-8))))
        proxy_iou = iou(mix(native, reference, gates), target)
        grid_iou = max(iou(mix(native, reference, gate), target) for gate in grid)
        raw = row["selected_raw_gate"]
        outside = [value < 0 or value > 1 for value in raw]
        different = [
            out and abs(gate - observed) > 1e-6
            for out, gate, observed in zip(outside, gates, row["selected_clamped_gate"])
        ]
        saturation["raw_below_zero"] += sum(value < 0 for value in raw)
        saturation["raw_above_one"] += sum(value > 1 for value in raw)
        saturation["samples_with_any_outside_axis"] += any(outside)
        saturation["outside_axes_with_projection_target_different"] += sum(different)
        for threshold, suffix in ((.25, "025"), (.5, "050")):
            final = row["selected_iou"] > threshold
            proxy = proxy_iou > threshold
            oracle = grid_iou > threshold
            current = counts[suffix]
            current["final_hits"] += final
            current["proxy_hits"] += proxy
            current["grid_oracle_hits"] += oracle
            current["proxy_gains"] += proxy and not final
            current["proxy_losses"] += final and not proxy
            current["reference_valid_but_proxy_fails"] += row["reference_iou"] > threshold and not proxy
            current["grid_fixes_proxy_failure"] += oracle and not proxy
            gained = proxy and not final
            current["proxy_gains_with_raw_gate_outside_clamp"] += gained and any(outside)
            current["proxy_gains_with_all_raw_gates_outside_clamp"] += gained and all(outside)
            current["proxy_gains_with_outside_axis_target_different"] += gained and any(different)
    assert max_reconstruction_error < 1e-4, max_reconstruction_error
    for suffix in counts:
        assert counts[suffix]["final_hits"] == footer["semantic_hits" + suffix]
    result["conditions"][name] = {
        "samples": len(rows), "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "max_final_box_reconstruction_error": max_reconstruction_error,
        "selected_gate_saturation": saturation,
        "thresholds": counts,
    }
result["interpretation"] = (
    "Outside-clamp counts describe zero derivatives through the clamp in isolation. "
    "They do not measure total training gradients, Hungarian eligibility, causality, "
    "or attainable gains. Projection uses GT; differences at most 1e-6 are ignored."
)
output = root / "reader_result_20261104/span-proxy-saturation-summary.json"
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
