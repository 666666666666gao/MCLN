#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import ast
import datetime
import hashlib
import json
from pathlib import Path

trial = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_gate_comparison_20261103")
source = trial / "source"
worker = Path("/proc/236606")
assert worker.exists()
assert (worker / "cwd").resolve() == source
selections = {
    "models/encoder_decoder_layers.py": {
        "calc_pairwise_locs", "MultiHeadAttentionSpatial", "BiDecoderLayer",
    },
    "models/candidate_local_visual.py": {"CandidateLocalVisual"},
    "models/mcln.py": set(),
    "models/losses.py": {"loss_sem_align"},
    "src/grounding_evaluator.py": {"evaluate_bbox_by_sem_align"},
}
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "Read-only source and active-process cwd audit. No model import, tensor load, forward, evaluation, test or training.",
    "worker_pid": 236606,
    "worker_cwd": str((worker / "cwd").resolve()),
    "files": {},
}
for relative, names in selections.items():
    data = (source / relative).read_bytes()
    lf_data = data.replace(b"\r\n", b"\n")
    text = data.decode("utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    blocks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            blocks.append({
                "name": node.name, "first_line": node.lineno,
                "last_line": node.end_lineno,
                "source": "\n".join(lines[node.lineno - 1:node.end_lineno]),
            })
    assert {block["name"] for block in blocks} == names
    result["files"][relative] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob_sha1": hashlib.sha1(
            b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        ).hexdigest(),
        "git_blob_sha1_lf_normalized": hashlib.sha1(
            b"blob " + str(len(lf_data)).encode("ascii") + b"\0" + lf_data
        ).hexdigest(),
        "bytes": len(data), "blocks": blocks,
    }
destination = trial / "reader_bbf_source_audit_git_normalized_20261103.json"
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({
    "destination": str(destination), "worker_cwd": result["worker_cwd"],
    "files": {
        name: {
            "sha256": details["sha256"],
            "git_blob_sha1": details["git_blob_sha1"],
            "git_blob_sha1_lf_normalized": details["git_blob_sha1_lf_normalized"],
            "blocks": [block["name"] for block in details["blocks"]],
        }
        for name, details in result["files"].items()
    },
}))
PY
