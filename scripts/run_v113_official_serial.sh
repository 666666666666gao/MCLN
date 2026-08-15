#!/usr/bin/env bash
set -uo pipefail

ROOT=/tmp/mcln_repo
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
OUTPUT=/root/autodl-tmp/DATA_ROOT/output/v113_meshsp_official_20260815
PARENT="$BASE/v108_artifacts/parent_h256_seed0.pth"
GEOMETRY="$BASE/v108_artifacts/geometry_h256_seed0.pth"
ARTIFACT="$BASE/v113_artifacts/asymmetric_risk_committee_h128_seeds0_1_2_fullfit.pth"
PARITY="$BASE/v113_train_runtime_parity.json"
PREFLIGHT="$BASE/v113_official_preflight_dryrun.json"
CLAIM="$BASE/v113_artifacts/v113_meshsp_official_once_after_train_runtime_parity.claim.json"
DRIVER_LOG="$BASE/v113_official_driver.log"
EXIT_FILE="$BASE/v113_official_exitcode.txt"
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
        echo "V113 official output already exists: $path" >&2
        exit 64
    fi
done

exec >"$DRIVER_LOG" 2>&1
echo "policy=single_gpu_once_v113_meshsp_official"
date -Is

expected_sha256() {
    local expected="$1"
    local path="$2"
    [[ -f "$path" && ! -L "$path" ]] || return 1
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]]
}

expected_sha256 7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f "$PARENT" || finish 80
expected_sha256 20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972 "$GEOMETRY" || finish 81
expected_sha256 45f96279794da73c9d21f5f7e817bb47def03a86a30ab7db092c1b1c0275a37b "$ARTIFACT" || finish 82
expected_sha256 53e86c392e86a7cb8813041d3a978413cc3c1784f741ded82a9444aba8ac4a81 "$PARITY" || finish 83
expected_sha256 05a5b23e2b3ac9c4f4d21e808cf88c61d64295a62301d94791413e77a09724f9 "$PREFLIGHT" || finish 84
expected_sha256 9916a5df1cf07d9a83d72108520b9b5617bb7991ecc3d526261eb07c4488a238 "$ROOT/train_dist_mod.py" || finish 85
expected_sha256 fa56d3da22b9ce0c8c6389173ff4f45c3407818d7a73c2aeab9f44ce81722d4a "$ROOT/models/rec_pareto_contextual_hierarchy.py" || finish 86
expected_sha256 bc56b06a2b00acf554fecbdbd0b41afe08cdb7d58536fc46a31ba8e2fa0d3f82 "$ROOT/scripts/run_frozen_v113_meshsp_official.py" || finish 87
if [[ "$(nvidia-smi -L | wc -l)" -ne 1 ]]; then
    echo "V113 official validation requires exactly one physical GPU"
    finish 88
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
    echo "GPU0 is not idle before V113 official validation"
    finish 89
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/pointnet2"
"$PY" scripts/run_frozen_v113_meshsp_official.py \
    --output-dir "$OUTPUT"
rc=$?
echo "V113 official validation rc=$rc"
date -Is
finish "$rc"
