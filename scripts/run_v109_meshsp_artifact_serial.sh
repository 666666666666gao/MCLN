#!/usr/bin/env bash
set -uo pipefail

ROOT=/tmp/mcln_repo
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
TRAIN_CACHE="$BASE/train"
GEOMETRY_CACHE="$BASE/geometry_train"
PARENT="$BASE/v108_artifacts/parent_h256_seed0.pth"
GEOMETRY="$BASE/v108_artifacts/geometry_h256_seed0.pth"
OOF="$BASE/v109_meshsp_nested_policy_oof.json"
ARTIFACT_DIR="$BASE/v109_artifacts"
ARTIFACT="$ARTIFACT_DIR/nested_policy_h128_seed0_fullfit.pth"
ARTIFACT_RECEIPT="$ARTIFACT_DIR/artifact_receipt.json"
LOG="$BASE/v109_artifact_build.log"
PIPELINE_LOG="$BASE/v109_artifact_pipeline.log"
EXIT_FILE="$BASE/v109_artifact_pipeline_exitcode.txt"
PY=/root/miniconda3/envs/bdetr/bin/python

finish() {
    local rc="$1"
    printf '%s\n' "$rc" >"$EXIT_FILE"
    for path in "$LOG" "$PIPELINE_LOG" "$EXIT_FILE"; do
        [[ ! -e "$path" ]] || chmod 0444 "$path"
    done
    exit "$rc"
}

for path in "$ARTIFACT_DIR" "$LOG" "$PIPELINE_LOG" "$EXIT_FILE"; do
    if [[ -e "$path" ]]; then
        echo "V109 artifact output already exists: $path" >&2
        exit 64
    fi
done

exec >"$PIPELINE_LOG" 2>&1
echo "policy=single_gpu_v109_fullfit_artifact"
date -Is

expected_sha256() {
    local expected="$1"
    local path="$2"
    [[ -f "$path" && ! -L "$path" ]] || return 1
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]]
}

expected_sha256 37680aaa34757cf9bb2376e93629ae6b89aa6b8fac16960ac091305cc20146a1 "$OOF" || finish 80
expected_sha256 07af9c6b331e808f86d16e62ae92a1106e86321c3f0734c3f2cb6ede46b94986 "$BASE/v109_nested_policy_pipeline_receipt.json" || finish 81
expected_sha256 7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f "$PARENT" || finish 82
expected_sha256 20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972 "$GEOMETRY" || finish 83
expected_sha256 aa77fadd55d8e579ad65748f8e1a078cbb756eaf53b061ec0339b73495a17c33 "$ROOT/scripts/run_v109_meshsp_nested_policy_oof.py" || finish 84
expected_sha256 93d5dca1b284d5cb4a34902b69a21a287c589a06e8c0ee43f12e375c608e5b98 "$ROOT/scripts/build_v109_meshsp_nested_policy_artifact.py" || finish 85
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
    echo "GPU0 is not idle before V109 artifact fit"
    finish 86
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/pointnet2"
echo "stage=full_train_fit_and_strict_reload"
date -Is
"$PY" scripts/build_v109_meshsp_nested_policy_artifact.py \
    --v109-result "$OOF" \
    --v109-script "$ROOT/scripts/run_v109_meshsp_nested_policy_oof.py" \
    --base-cache "$TRAIN_CACHE" \
    --geometry-cache "$GEOMETRY_CACHE" \
    --parent-artifact "$PARENT" \
    --geometry-artifact "$GEOMETRY" \
    --artifact-output "$ARTIFACT" \
    --receipt-output "$ARTIFACT_RECEIPT" \
    --device cuda:0 >"$LOG" 2>&1
rc=$?
if [[ "$rc" -ne 0 ]]; then
    echo "V109 artifact build failed: rc=$rc"
    finish "$rc"
fi
if [[ ! -f "$ARTIFACT" || ! -f "$ARTIFACT_RECEIPT" ]]; then
    echo "V109 artifact build did not publish both outputs"
    finish 87
fi
chmod 0444 "$ARTIFACT" "$ARTIFACT_RECEIPT" "$LOG"
chmod 0555 "$ARTIFACT_DIR"
printf '0\n' >"$EXIT_FILE"
chmod 0444 "$EXIT_FILE"
echo "V109 artifact pipeline rc=0"
date -Is
chmod 0444 "$PIPELINE_LOG"
exit 0
