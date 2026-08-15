#!/usr/bin/env bash
set -uo pipefail

ROOT=/tmp/mcln_repo
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
TRAIN_CACHE="$BASE/train"
GEOMETRY_CACHE="$BASE/geometry_train"
PARENT="$BASE/v108_artifacts/parent_h256_seed0.pth"
GEOMETRY="$BASE/v108_artifacts/geometry_h256_seed0.pth"
ARTIFACT="$BASE/v113_artifacts/asymmetric_risk_committee_h128_seeds0_1_2_fullfit.pth"
ARTIFACT_RECEIPT="$BASE/v113_artifacts/artifact_receipt.json"
PARITY="$BASE/v113_train_runtime_parity.json"
LOG="$BASE/v113_train_runtime_parity.log"
PIPELINE_LOG="$BASE/v113_train_runtime_parity_pipeline.log"
EXIT_FILE="$BASE/v113_train_runtime_parity_exitcode.txt"
PY=/root/miniconda3/envs/bdetr/bin/python
ARTIFACT_SHA256=45f96279794da73c9d21f5f7e817bb47def03a86a30ab7db092c1b1c0275a37b
ARTIFACT_RECEIPT_SHA256=1af664eac2be45cbd6032f1a9340c7043f24a2ab91c09284d267eda0bbc9097d

finish() {
    local rc="$1"
    printf '%s\n' "$rc" >"$EXIT_FILE"
    for path in "$LOG" "$PIPELINE_LOG" "$EXIT_FILE"; do
        [[ ! -e "$path" ]] || chmod 0444 "$path"
    done
    exit "$rc"
}

for path in "$PARITY" "$LOG" "$PIPELINE_LOG" "$EXIT_FILE"; do
    if [[ -e "$path" || -L "$path" ]]; then
        echo "V113 parity output already exists: $path" >&2
        exit 64
    fi
done

exec >"$PIPELINE_LOG" 2>&1
echo "policy=single_gpu_v113_full_train_offline_runtime_exact_parity"
date -Is

expected_sha256() {
    local expected="$1"
    local path="$2"
    [[ -f "$path" && ! -L "$path" ]] || return 1
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]]
}

expected_sha256 "$ARTIFACT_SHA256" "$ARTIFACT" || finish 80
expected_sha256 "$ARTIFACT_RECEIPT_SHA256" "$ARTIFACT_RECEIPT" || finish 81
expected_sha256 7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f "$PARENT" || finish 82
expected_sha256 20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972 "$GEOMETRY" || finish 83
expected_sha256 7c989cbdc1dd73aeeea482130b028be69bc5f1d570889f5a1b5493de87f9d938 "$ROOT/scripts/audit_v113_runtime_parity_train.py" || finish 84
expected_sha256 9916a5df1cf07d9a83d72108520b9b5617bb7991ecc3d526261eb07c4488a238 "$ROOT/train_dist_mod.py" || finish 85
expected_sha256 fa56d3da22b9ce0c8c6389173ff4f45c3407818d7a73c2aeab9f44ce81722d4a "$ROOT/models/rec_pareto_contextual_hierarchy.py" || finish 86
expected_sha256 9875fa881b4d81aae92fc4f1f033c06de252073b93de2e9e76c85ba53fde8a8f "$ROOT/scripts/build_v113_meshsp_asymmetric_risk_artifact.py" || finish 87
expected_sha256 bfe2a650e22459d09dcbca6f525cbcda136787b456bf77d734c1e2f76b67caaa "$BASE/candidate_train_receipt.json" || finish 88
expected_sha256 e45adaafb3730f45dabcea7f0c4f4492a6ea6360b7f07bdb164270bd934d9443 "$BASE/geometry_train_receipt.json" || finish 89
expected_sha256 fc4ef0c019c4b4cdabfa7e68613605aeae9d35b47f9165114b741217f2ad6d6d "$TRAIN_CACHE/manifest.json" || finish 90
expected_sha256 5b97b82e99ba5512852be6f9db4184aaf0959d77be3217da4856ef9271f93ec2 "$GEOMETRY_CACHE/manifest.json" || finish 91

if [[ "$(nvidia-smi -L | wc -l)" -ne 1 ]]; then
    echo "V113 parity requires exactly one visible physical GPU"
    finish 92
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
    echo "GPU0 is not idle before V113 parity"
    finish 93
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/pointnet2"
echo "stage=full_36665_row_offline_runtime_exact_parity"
date -Is
"$PY" scripts/audit_v113_runtime_parity_train.py \
    --base-cache "$TRAIN_CACHE" \
    --geometry-cache "$GEOMETRY_CACHE" \
    --parent-artifact "$PARENT" \
    --geometry-artifact "$GEOMETRY" \
    --v113-artifact "$ARTIFACT" \
    --expected-artifact-sha256 "$ARTIFACT_SHA256" \
    --output "$PARITY" \
    --device cuda:0 >"$LOG" 2>&1
rc=$?
if [[ "$rc" -ne 0 ]]; then
    echo "V113 train/runtime parity failed: rc=$rc"
    finish "$rc"
fi
if [[ ! -f "$PARITY" || -L "$PARITY" ]]; then
    echo "V113 parity did not publish a regular report"
    finish 94
fi
chmod 0444 "$PARITY" "$LOG"
printf '0\n' >"$EXIT_FILE"
chmod 0444 "$EXIT_FILE"
echo "V113 parity pipeline rc=0"
date -Is
chmod 0444 "$PIPELINE_LOG"
exit 0
