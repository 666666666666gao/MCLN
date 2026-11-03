#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
stage="$root/scanrefer_restart_20261102"
staging="$root/offline_audits/observations_v2"
cd "$stage/source"
CUDA_VISIBLE_DEVICES="" STAGE="$stage" STAGING="$staging" \
PYTHONPATH="$staging:$stage/source/pointnet2:$stage/source" \
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy

stage = Path(os.environ["STAGE"])
staging = Path(os.environ["STAGING"])
source = stage / "source"
entry = staging / "tools/eval_pvground_audit.py"
runpy.run_path(str(entry), run_name="audit_import_preflight")

import torch
import train_dist_mod
from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from src.grounding_evaluator import GroundingEvaluator
from tools.pvground_forward_audit import PVGroundForwardAudit
from tools.pvground_semantic_audit import PVGroundSemanticAudit

objects = {
    "trainer": train_dist_mod,
    "evaluator": GroundingEvaluator,
    "iou": _iou3d_par,
    "box_converter": box_cxcyczwhd_to_xyzxyz,
    "forward_audit": PVGroundForwardAudit,
    "semantic_audit": PVGroundSemanticAudit,
}
paths = {name: Path(inspect.getfile(value)).resolve() for name, value in objects.items()}
for name in ("trainer", "evaluator", "iou", "box_converter"):
    assert paths[name].is_relative_to(source), (name, paths[name])
for name in ("forward_audit", "semantic_audit"):
    assert paths[name].is_relative_to(staging), (name, paths[name])
assert not torch.cuda.is_initialized()
receipt = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "import_preflight_passed",
    "source_root": str(source), "entry": str(entry),
    "entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
    "imports": {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in paths.items()
    },
    "cuda_initialized": torch.cuda.is_initialized(),
    "scope": "Remote import resolution only; no model construction, checkpoint load, forward or evaluation.",
}
(stage / "current_audit_import_preflight.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
PY
