#!/usr/bin/env bash
set -euo pipefail

REPO=/tmp/mcln_repo
PYTHON=/root/miniconda3/envs/bdetr/bin/python
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
V109=$BASE/v109_meshsp_nested_policy_oof.json
V112=$BASE/v112_meshsp_anchor_committee_tradeoff_oof.json
CACHE=$BASE/v112_meshsp_anchor_committee_train_oof_predictions.json.gz
PARENT=$BASE/v108_artifacts/parent_h256_seed0.pth
GEOMETRY=$BASE/v108_artifacts/geometry_h256_seed0.pth
FALLBACK=/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train_missing789.txt
OUTPUT=$BASE/v113_meshsp_asymmetric_risk_replay.json
LOG=$BASE/v113_meshsp_asymmetric_risk_replay.log
EXIT=$BASE/v113_meshsp_asymmetric_risk_replay.exit
SESSION=v113_asymmetric_risk_replay
SELF=$(readlink -f "$0")

require_sha() {
    local expected=$1 path=$2 actual
    actual=$(sha256sum "$path" | awk '{print $1}')
    if [[ "$actual" != "$expected" ]]; then
        echo "SHA-256 mismatch: $path expected=$expected actual=$actual" >&2
        exit 73
    fi
}

if [[ "${1:-}" == "--inner" ]]; then
    cd "$REPO"
    set +e
    env PYTHONPATH="$REPO:$REPO/pointnet2" \
        "$PYTHON" scripts/run_v113_meshsp_asymmetric_risk_replay.py \
        --v109-report "$V109" \
        --v112-report "$V112" \
        --prediction-cache "$CACHE" \
        --parent-artifact "$PARENT" \
        --geometry-artifact "$GEOMETRY" \
        --fallback-scenes "$FALLBACK" \
        --output "$OUTPUT" >"$LOG" 2>&1
    rc=$?
    set -e
    chmod 0444 "$LOG"
    printf '%s\n' "$rc" >"$EXIT"
    chmod 0444 "$EXIT"
    exit "$rc"
fi

cd "$REPO"
require_sha 439c75c081c3f445564ad36a55dfb4ab92443061ee889301297081ab4b4a2ee3 \
    scripts/run_v113_meshsp_asymmetric_risk_replay.py
require_sha 5f03702325d6ed93f7fe15348a0161032e95dac676bc06f35548f57131b3ce1b \
    scripts/run_v112_meshsp_anchor_committee_tradeoff_oof.py
require_sha 128ce636d27234db7fca4fb23bd5d30945928d9ac9dcd1cf8139c38670a41b96 "$V112"
require_sha 1123df3d312e433bf14b83874de99742906907738802bf878056ca07caa7ffdd "$CACHE"
require_sha 37680aaa34757cf9bb2376e93629ae6b89aa6b8fac16960ac091305cc20146a1 "$V109"
require_sha caf63109bdf9f19cd8132b3c70eb1f2467d70fc605d174c6ec801b34c1c31079 "$FALLBACK"
require_sha 7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f "$PARENT"
require_sha 20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972 "$GEOMETRY"
require_sha 3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208 \
    /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth
for path in "$OUTPUT" "$LOG" "$EXIT"; do
    if [[ -e "$path" ]]; then
        echo "Refusing to overwrite existing V113 output: $path" >&2
        exit 74
    fi
done
if [[ $(nvidia-smi -L | wc -l) -ne 1 ]]; then
    echo "V113 requires the frozen single-GPU topology" >&2
    exit 75
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | grep -Eq '^[[:space:]]*[0-9]+'; then
    echo "GPU already has a compute process" >&2
    exit 75
fi
if pgrep -af '[r]un_v113_meshsp_asymmetric_risk_replay.py' >/dev/null; then
    echo "V113 replay process already exists" >&2
    exit 75
fi
if screen -ls 2>/dev/null | grep -q "[.]$SESSION"; then
    echo "V113 screen session already exists" >&2
    exit 75
fi
screen -DmS "$SESSION" /bin/bash "$SELF" --inner
sleep 1
if ! screen -ls 2>/dev/null | grep -q "[.]$SESSION" \
        && [[ ! -e "$EXIT" ]]; then
    echo "V113 screen failed to remain alive and produced no exit receipt" >&2
    exit 75
fi
echo "launched session=$SESSION output=$OUTPUT log=$LOG exit=$EXIT"
