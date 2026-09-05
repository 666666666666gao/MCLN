#!/usr/bin/env bash
set -euo pipefail
MODE="$1"
ADDON="$(cd "$(dirname "$0")/.." && pwd)"
PAIR=/root/autodl-tmp/mcln_g0_view_pair_20260905
SOURCE="$PAIR/inputs_v3/fixed_source"
PYTHON=/root/miniconda3/envs/bdetr/bin/python
CHECKPOINT=/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/official_best_rec025_epoch_57_0p56652741.pth
exec 9>/root/autodl-tmp/mcln_v99_backbone_gpu0.lock
flock -n 9
test "$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 500
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$SOURCE"
cd "$SOURCE"
case "$MODE" in
  preflight)
    exec "$PYTHON" "$ADDON/scripts/run_nr3d_pair_readout.py" --checkpoint "$CHECKPOINT" --output "$ADDON/preflight" --preflight-only
    ;;
  train)
    "$PYTHON" - "$ADDON" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
smoke = json.loads((root / 'preflight/smoke.json').read_text())
assert smoke['optimizer_steps'] == {'global': 0, 'pair': 0}
assert smoke['covered_rows'] > 0 and smoke['heldout_batches'] == 0 and smoke['weights_written'] == 0
assert all(smoke[mode]['gradient_norm'] > 0 for mode in ['global', 'pair'])
assert json.loads((root / 'cpu_receipt.json').read_text())['pytest_exit_code'] == 0
PY
    exec "$PYTHON" "$ADDON/scripts/run_nr3d_pair_readout.py" --checkpoint "$CHECKPOINT" --output "$ADDON/results"
    ;;
  *) exit 2 ;;
esac
