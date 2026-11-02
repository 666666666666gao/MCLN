#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/PV-Ground/env/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import torch

directory = Path("/root/autodl-tmp/mcln_pvground_ft_20261101/runs/scanrefer/scanrefer/mcln_pvground_scanrefer_20261101/1793579788")
path = directory / "semantic_best.pth"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
prefixes = (
    "decoder.5.local_visual.",
    "pvground_support_corrector.",
    "pvground_span_mixer.",
)
rows = {}
for saved_name, tensor in checkpoint["model"].items():
    name = saved_name.removeprefix("module.")
    if not name.startswith(prefixes):
        continue
    value = tensor.detach().float()
    rows[name] = {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all()),
        "nonzero": int(torch.count_nonzero(value)),
        "elements": value.numel(),
        "norm": float(value.norm()),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
    }
steps = []
for state in checkpoint["optimizer"]["state"].values():
    if "step" in state:
        steps.append(float(state["step"]))
digest = hashlib.sha256()
with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
print(json.dumps({
    "checkpoint": str(path), "sha256": digest.hexdigest(),
    "epoch": checkpoint["epoch"], "tensors": rows,
    "optimizer_step_min": min(steps), "optimizer_step_max": max(steps),
    "limitation": "Parameter audit only; no sample gate occupancy, causal ablation or accuracy claim.",
}, sort_keys=True))
PY
