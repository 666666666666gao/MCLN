#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/mcln_pvground_ft_20261101
archive="$root/pvground_offline_audit_c8437a3.tar.gz"
test "$(sha256sum "$archive" | cut -d' ' -f1)" = 34a234ad96bbaea6b74a48550d0aeb868154d019118943961c921472f7cbca67
staging="$root/offline_audits/c8437a3"
mkdir -p "$staging"
tar -xzf "$archive" -C "$staging"
cd "$root/source"
PYTHONPATH="$staging:$root/source/pointnet2:$root/source" \
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import inspect
import json
import torch
from tools.pvground_forward_audit import PVGroundForwardAudit
from tools.pvground_semantic_audit import PVGroundSemanticAudit
import train_dist_mod as trainer
from src.grounding_evaluator import GroundingEvaluator
from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz

print(json.dumps({
    "status": "import_preflight_passed",
    "torch": torch.__version__,
    "trainer": inspect.getfile(trainer),
    "forward_audit": inspect.getfile(PVGroundForwardAudit),
    "semantic_audit": inspect.getfile(PVGroundSemanticAudit),
    "get_model": str(inspect.signature(trainer.TrainTester.get_model)),
    "build_evaluator": str(inspect.signature(trainer.TrainTester._build_grounding_evaluator)),
    "semantic_evaluator": str(inspect.signature(GroundingEvaluator.evaluate_bbox_by_sem_align)),
    "cuda_initialized": torch.cuda.is_initialized(),
    "limitation": "No model construction, checkpoint load, forward or evaluation.",
}, sort_keys=True))
PY
