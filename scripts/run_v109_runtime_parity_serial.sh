#!/usr/bin/env bash
set -uo pipefail

ROOT=/tmp/mcln_repo
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
TRAIN_CACHE="$BASE/train"
GEOMETRY_CACHE="$BASE/geometry_train"
PARENT="$BASE/v108_artifacts/parent_h256_seed0.pth"
GEOMETRY="$BASE/v108_artifacts/geometry_h256_seed0.pth"
ARTIFACT="$BASE/v109_artifacts/nested_policy_h128_seed0_fullfit.pth"
OUTPUT="$BASE/v109_train_runtime_parity.json"
LOG="$BASE/v109_train_runtime_parity.log"
EXIT_FILE="$BASE/v109_train_runtime_parity_exitcode.txt"
PY=/root/miniconda3/envs/bdetr/bin/python

finish() {
    local rc="$1"
    printf '%s\n' "$rc" >"$EXIT_FILE"
    for path in "$OUTPUT" "$LOG" "$EXIT_FILE"; do
        [[ ! -e "$path" ]] || chmod 0444 "$path"
    done
    exit "$rc"
}

for path in "$OUTPUT" "$LOG" "$EXIT_FILE"; do
    if [[ -e "$path" ]]; then
        echo "V109 parity output already exists: $path" >&2
        exit 64
    fi
done

exec >"$LOG" 2>&1
echo "policy=single_gpu_serial_v109_train_runtime_parity"
date -Is

expected_sha256() {
    local expected="$1"
    local path="$2"
    [[ -f "$path" && ! -L "$path" ]] || return 1
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]]
}

expected_sha256 7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f "$PARENT" || finish 80
expected_sha256 20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972 "$GEOMETRY" || finish 81
expected_sha256 20db69ddc27680a035384277bc48cd44109215e3d7d1158cdc4a4f21ff7c785b "$ARTIFACT" || finish 82
expected_sha256 bfe2a650e22459d09dcbca6f525cbcda136787b456bf77d734c1e2f76b67caaa "$BASE/candidate_train_receipt.json" || finish 83
expected_sha256 e45adaafb3730f45dabcea7f0c4f4492a6ea6360b7f07bdb164270bd934d9443 "$BASE/geometry_train_receipt.json" || finish 84
expected_sha256 691a7aa969bc2fb277f9807bda578b20dcf5de1cf827ad37e4808e2b92c794fc "$ROOT/train_dist_mod.py" || finish 85
expected_sha256 d108fc146b80646b7ab0479d7a03d2f7f7cf69ed45bea597232b46f9b836f9fe "$ROOT/models/rec_pareto_contextual_hierarchy.py" || finish 86
expected_sha256 011a5de2881545a965df801db265a05f52002d9c55229cf7218e53975e70ff16 "$ROOT/scripts/audit_v109_runtime_parity_train.py" || finish 87
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
    echo "GPU0 is not idle before V109 parity"
    finish 88
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/pointnet2"
"$PY" scripts/audit_v109_runtime_parity_train.py \
    --base-cache "$TRAIN_CACHE" \
    --geometry-cache "$GEOMETRY_CACHE" \
    --parent-artifact "$PARENT" \
    --geometry-artifact "$GEOMETRY" \
    --v109-artifact "$ARTIFACT" \
    --output "$OUTPUT" \
    --device cuda:0
rc=$?
echo "V109 train/runtime parity rc=$rc"
date -Is
finish "$rc"
