#!/usr/bin/env bash
# One read-only integration probe after G0 and both P1 receipts are complete.
set -euo pipefail
readonly PAIR_ROOT="$1"
readonly PYTHON=/root/miniconda3/envs/bdetr/bin/python
readonly ADDON="$PAIR_ROOT/pair_readout_adapter_v1"
readonly CHECKPOINT=/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/official_best_rec025_epoch_57_0p56652741.pth

exec 9>/root/autodl-tmp/mcln_v99_backbone_gpu0.lock
flock 9
"$PYTHON" - "$PAIR_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
decision = json.loads((root / "results/decision.json").read_text())
assert decision["integrity_pass"] is True
# This is a diagnostic, so it does not require the augmentation performance gate.
for name in ["padding_identity_receipt.json", "candidate_contract_v2_receipt.json"]:
    receipt = json.loads((root / name).read_text())
    assert receipt["checkpoint_sha256"] == "76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1"
    assert receipt["fit_row_ids"] == [0, 1, 3, 4]
    assert receipt["optimizer_steps"] == receipt["weights_written"] == 0
    assert receipt["formal_validation_evaluated"] is False
addon = root / "pair_readout_adapter_v1"
manifest = json.loads((addon / "scene_input_manifest.json").read_text())
for relative, expected in manifest["files"].items():
    assert hashlib.sha256((addon / relative).read_bytes()).hexdigest() == expected
used = int(subprocess.check_output([
    "nvidia-smi", "--id=0", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"]).decode().strip())
assert used < 500, used
PY
cd "$PAIR_ROOT/inputs_v3/fixed_source"
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
exec "$PYTHON" -u "$ADDON/scripts/audit_pair_readout_scene.py" \
  --checkpoint "$CHECKPOINT" \
  --output "$PAIR_ROOT/pair_readout_scene_receipt.json"
