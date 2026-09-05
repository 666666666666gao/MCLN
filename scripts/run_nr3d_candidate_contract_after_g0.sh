#!/usr/bin/env bash
# One frozen forward on the same four fit rows; no optimizer or rule changes.
set -euo pipefail
readonly PAIR_ROOT="$1"
readonly PYTHON=/root/miniconda3/envs/bdetr/bin/python
readonly CHECKPOINT=/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/official_best_rec025_epoch_57_0p56652741.pth

exec 9>/root/autodl-tmp/mcln_v99_backbone_gpu0.lock
flock 9
test -f "$PAIR_ROOT/results/decision.json"
"$PYTHON" - <<'PY'
import subprocess
used = int(subprocess.check_output([
    'nvidia-smi', '--id=0', '--query-gpu=memory.used',
    '--format=csv,noheader,nounits']).decode().strip())
assert used < 500, used
PY
cd "$PAIR_ROOT/inputs_v3/fixed_source"
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PAIR_ROOT/candidate_contract:$PWD"
exec "$PYTHON" -u "$PAIR_ROOT/candidate_contract/scripts/audit_nr3d_padding_scene.py" \
  --checkpoint "$CHECKPOINT" \
  --masked-layer-source "$PAIR_ROOT/padding_encoder_decoder_layers.py" \
  --candidate-contract-only \
  --output "$PAIR_ROOT/candidate_contract_receipt.json"
