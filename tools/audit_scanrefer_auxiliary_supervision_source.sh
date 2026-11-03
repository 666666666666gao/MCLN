#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import ast
import datetime
import hashlib
import json
from pathlib import Path

trial = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_gate_comparison_20261103")
source = trial / "source"
worker = Path("/proc/240217")
assert worker.exists() and (worker / "cwd").resolve() == source
run = trial / "span/scanrefer/scanrefer_span_gate_span_20261103/1793733129"
config = json.loads((run / "config.json").read_text())
receipt = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "worker_pid": 240217,
    "worker_cwd": str(source),
    "scope": "Read-only current source AST/text and config; no model/dataset import, tensor load, forward, evaluation, test, training or source mutation.",
    "files": {},
    "config": {key: value for key, value in config.items()
               if any(token in key for token in (
                   "counterfactual", "anchor", "aux_loss", "train_only", "sacr"))},
    "feature_switches": {key: value for key, value in config.items()
                         if key.startswith("use_") and isinstance(value, bool)},
}
for relative in ("models/losses.py", "main_utils.py"):
    data = (source / relative).read_bytes()
    lines = data.decode("utf-8").splitlines()
    tree = ast.parse("\n".join(lines))
    blocks = []
    usage = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        block = "\n".join(lines[node.lineno - 1:node.end_lineno])
        if node.name == "_get_inputs" or (
                node.name == "forward" and "auxi_indices = None" in block):
            blocks.append({
                "name": node.name, "first_line": node.lineno,
                "last_line": node.end_lineno, "source": block,
            })
        if node.name in ("loss_pos_align", "loss_boxes", "loss_masks", "loss_sem_align"):
            usage.append({
                "name": node.name, "first_line": node.lineno,
                "last_line": node.end_lineno,
                "auxi_indices_body_name_uses": sum(
                    isinstance(child, ast.Name) and child.id == "auxi_indices"
                    for child in ast.walk(node)),
                "auxiliary_field_body_references": [
                    {"line": child.lineno, "field": child.value}
                    for child in ast.walk(node)
                    if isinstance(child, ast.Constant)
                    and child.value in ("auxi_box", "auxi_entity_positive_map")
                ],
            })
    receipt["files"][relative] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "blocks": blocks, "native_loss_auxiliary_usage": usage,
        "batch_label_attachment_context": [
            {"first_line": max(0, i - 6) + 1,
             "source": "\n".join(lines[max(0, i - 6):i + 4])}
            for i, line in enumerate(lines)
            if "end_points[key] = batch_data[key]" in line
        ],
    }
destination = trial / "auxiliary_supervision_source_audit.json"
destination.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
