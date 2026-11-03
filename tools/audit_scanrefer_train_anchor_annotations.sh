#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import ast
from collections import Counter
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
dataset_path = source / "src/joint_det_dataset.py"
data = dataset_path.read_bytes()
text = data.decode("utf-8")
lines = text.splitlines()
tree = ast.parse(text)
blocks = [
    {
        "name": node.name,
        "first_line": node.lineno,
        "last_line": node.end_lineno,
        "source": "\n".join(lines[node.lineno - 1:node.end_lineno]),
    }
    for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef) and node.name == "load_scanrefer_annos"
]
assert len(blocks) == 1
auxiliary_blocks = [
    {
        "name": node.name,
        "first_line": node.lineno,
        "last_line": node.end_lineno,
        "source": "\n".join(lines[node.lineno - 1:node.end_lineno]),
    }
    for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef) and (
        node.name == "_get_auxi_boxes"
        or 'anno["auxi_entity"] = auxi_entity' in "\n".join(
            lines[node.lineno - 1:node.end_lineno]
        )
    )
]
train_paths = list(Path(config["data_root"]).glob("[Ss]can[Rr]efer/ScanRefer_filtered_train.json"))
assert len(train_paths) == 1
train_path = train_paths[0]
train_data = train_path.read_bytes()
records = json.loads(train_data)
assert isinstance(records, list) and records
fields = Counter(key for record in records for key in record)
anchor_fields = {
    key: {
        "present": fields[key],
        "nonempty": sum(bool(record.get(key)) for record in records),
    }
    for key in ("anchors", "anchor_ids", "anchors_types", "anchor_boxes", "path")
}
relevant = [
    {"line": i + 1, "source": line}
    for i, line in enumerate(lines)
    if any(token in line for token in ("auxi_entity", "anchor_ids", "auxi_box", "use_sacr_source"))
]
receipt = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "Read-only active-source and raw TRAIN annotation fields. No model/dataset import, tensor load, forward, evaluation, training or source mutation. No validation annotation read.",
    "worker_pid": 240217,
    "worker_cwd": str(source),
    "config": {
        key: config.get(key)
        for key in ("data_root", "use_sacr_source", "augment_det", "dataset", "test_dataset")
    },
    "dataset_source_sha256": hashlib.sha256(data).hexdigest(),
    "scanrefer_loader": blocks[0],
    "auxiliary_source_blocks": auxiliary_blocks,
    "related_source_lines": relevant,
    "train_annotation_path": str(train_path),
    "train_annotation_sha256": hashlib.sha256(train_data).hexdigest(),
    "train_records": len(records),
    "field_counts": dict(sorted(fields.items())),
    "explicit_anchor_field_counts": anchor_fields,
    "interpretation_limit": "Absent explicit raw fields do not prove that parsed text, other caches or auxiliary heuristics contain no entity information. They do not establish object-ID alignment, annotation correctness or trained method benefit.",
}
destination = trial / "train_anchor_annotation_audit.json"
destination.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({
    key: receipt[key]
    for key in (
        "observed_utc", "scope", "worker_pid", "config",
        "train_annotation_path", "train_annotation_sha256", "train_records",
        "field_counts", "explicit_anchor_field_counts", "auxiliary_source_blocks",
        "interpretation_limit",
    )
}))
PY
