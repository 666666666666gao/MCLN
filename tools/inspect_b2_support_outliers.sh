#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("/root/autodl-tmp/mcln_pvground_ft_20261101")
audit = root / "offline_audits/b2_fixed_audit_20261103_lrpositive"
source = root / "scanrefer_restart_20261102/source/models/pvground_innovations.py"
rows = []
footer = None
with (audit / "none.jsonl").open() as stream:
    for line in stream:
        row = json.loads(line)
        if row["kind"] == "semantic_sample":
            rows.append(row)
        elif row["kind"] == "forward_complete":
            footer = row
assert len(rows) == 9508 and footer["verified_against_evaluator"]
groups = {
    "all": rows,
    "native_nonpositive_axis": [r for r in rows if min(r["native_box"][3:]) <= 0],
    "native_axis_below_1e_minus6_floor": [r for r in rows if min(r["native_box"][3:]) < 1e-6],
    "selected_support_mean_delta_gt_1000": [
        r for r in rows if r["selected_support_delta_abs_mean_before_bypass"] > 1000
    ],
}
report = {
    "scope": "Read-only selected-query outlier association in completed B2 none condition; not a causal diagnosis.",
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "source_scale_lines": [
        {"line": i, "text": line.strip()}
        for i, line in enumerate(source.read_text().splitlines(), 1)
        if "scale = size.detach()" in line or "(value[None] - center.detach()" in line
    ],
    "groups": {},
}
for name, members in groups.items():
    report["groups"][name] = {
        "count": len(members),
        "nonpositive_native_size_count": sum(min(r["native_box"][3:]) <= 0 for r in members),
        "hits025": sum(r["selected_iou"] > 0.25 for r in members),
        "hits050": sum(r["selected_iou"] > 0.5 for r in members),
        "delta_gt_1000_count": sum(
            r["selected_support_delta_abs_mean_before_bypass"] > 1000 for r in members
        ),
    }
report["largest_20_selected_corrections"] = [
    {key: row[key] for key in (
        "sample_index", "scan_id", "target_id", "selected_query",
        "selected_support_delta_abs_mean_before_bypass",
        "selected_support_delta_abs_max_before_bypass",
        "selected_support_foreground_flips_before_bypass",
        "native_box", "reference_box", "final_box", "gt_box",
        "native_iou", "reference_iou", "final_iou",
    )}
    for row in sorted(
        rows, key=lambda r: r["selected_support_delta_abs_mean_before_bypass"], reverse=True
    )[:20]
]
output = audit / "support-outliers.json"
output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps({key: report[key] for key in ("scope", "source_sha256", "source_scale_lines", "groups")}))
PY
