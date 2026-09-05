#!/usr/bin/env bash
# Usage: bash scripts/run_nr3d_view_pair.sh /absolute/path/to/pair_root
set -euo pipefail
readonly PAIR_ROOT="$1"
readonly PYTHON=/root/miniconda3/envs/bdetr/bin/python
readonly CHECKPOINT=/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/control/official_rec_monitor/official_best_rec025_epoch_57_0p56652741.pth

exec 9>/root/autodl-tmp/mcln_v99_backbone_gpu0.lock
flock -n 9
"$PYTHON" - "$PAIR_ROOT" <<'PY'
import json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
manifests = {}
preflights = {}
for role in ('old', 'fixed'):
    smoke = json.loads((root / ('smoke_v2_' + role) / 'smoke.json').read_text())
    assert smoke['backward'] and smoke['optimizer_steps'] == 0
    assert smoke['heldout_batches'] == 0 and smoke['weights_written'] == 0
    preflights[role] = json.loads((root / ('smoke_v2_' + role) / 'preflight.json').read_text())
    manifests[role] = json.loads((root / 'inputs_v2' / (role + '_source') / 'g0_source_manifest.json').read_text())['files']
assert set(manifests['old']) == set(manifests['fixed'])
assert [k for k in manifests['old'] if manifests['old'][k] != manifests['fixed'][k]] == ['src/joint_det_dataset.py']
assert manifests['old']['src/joint_det_dataset.py'] == '800bac2caf9b7a319bdc200f60386000e4e374a559d9581113a8eb57d525f9f0'
assert manifests['fixed']['src/joint_det_dataset.py'] == '4a8edacf2c59c8ded76697153d96e8eb078b70146222c187e98b0e34f64bc77e'
for key in ('census', 'optimizer_groups', 'scheduler_milestones', 'fit_batches', 'holdout_batches'):
    assert preflights['old'][key] == preflights['fixed'][key], key
used = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode().strip())
assert used < 500, used
print('G0 matched source/smoke/GPU checks PASS', flush=True)
PY

export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29572
mkdir "$PAIR_ROOT/results"
for ROLE in old fixed; do
  cd "$PAIR_ROOT/inputs_v2/${ROLE}_source"
  "$PYTHON" -u scripts/run_nr3d_view_pair_role.py \
    --role "$ROLE" --checkpoint "$CHECKPOINT" \
    --output "$PAIR_ROOT/results/$ROLE" \
    > "$PAIR_ROOT/results/${ROLE}.log" 2>&1
done
"$PYTHON" scripts/decide_nr3d_view_pair.py --pair-root "$PAIR_ROOT/results"
