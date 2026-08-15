#!/usr/bin/env bash
set -uo pipefail

ROOT=/tmp/mcln_repo
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
OUTPUT=/root/autodl-tmp/DATA_ROOT/output/v109_meshsp_official_20260814
PARENT="$BASE/v108_artifacts/parent_h256_seed0.pth"
GEOMETRY="$BASE/v108_artifacts/geometry_h256_seed0.pth"
ARTIFACT="$BASE/v109_artifacts/nested_policy_h128_seed0_fullfit.pth"
PARITY="$BASE/v109_train_runtime_parity.json"
PREFLIGHT="$BASE/v109_official_preflight_dryrun.json"
CLAIM="$BASE/v109_artifacts/v109_meshsp_official_once_after_train_runtime_parity.claim.json"
DRIVER_LOG="$BASE/v109_official_driver.log"
EXIT_FILE="$BASE/v109_official_exitcode.txt"
PY=/root/miniconda3/envs/bdetr/bin/python

finish() {
    local rc="$1"
    printf '%s\n' "$rc" >"$EXIT_FILE"
    for path in "$DRIVER_LOG" "$EXIT_FILE"; do
        [[ ! -e "$path" ]] || chmod 0444 "$path"
    done
    exit "$rc"
}

for path in "$OUTPUT" "$CLAIM" "$DRIVER_LOG" "$EXIT_FILE"; do
    if [[ -e "$path" ]]; then
        echo "V109 official output already exists: $path" >&2
        exit 64
    fi
done

exec >"$DRIVER_LOG" 2>&1
echo "policy=single_gpu_once_v109_meshsp_official"
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
expected_sha256 5bc46bbf1146a34ee834b4241f934b244f1d8b2287fb931ec963c720350a9c46 "$PARITY" || finish 83
expected_sha256 f7bdb28a4c77cea40f1e4621bbec127cab876377ba1a7f53b612605ec27025a2 "$PREFLIGHT" || finish 84
expected_sha256 691a7aa969bc2fb277f9807bda578b20dcf5de1cf827ad37e4808e2b92c794fc "$ROOT/train_dist_mod.py" || finish 85
expected_sha256 d108fc146b80646b7ab0479d7a03d2f7f7cf69ed45bea597232b46f9b836f9fe "$ROOT/models/rec_pareto_contextual_hierarchy.py" || finish 86
expected_sha256 3095cdd1746d4e99fe120a5b2f35284483d448f2ef020433f19c0d0bf9ca286b "$ROOT/scripts/run_frozen_v109_meshsp_official.py" || finish 87
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
    echo "GPU0 is not idle before V109 official validation"
    finish 88
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/pointnet2"
"$PY" scripts/run_frozen_v109_meshsp_official.py \
    --output-dir "$OUTPUT"
rc=$?
echo "V109 official validation rc=$rc"
date -Is
finish "$rc"
