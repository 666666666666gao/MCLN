#!/usr/bin/env bash
set -uo pipefail

ROOT=/tmp/mcln_repo
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
TRAIN_CACHE="$BASE/train"
GEOMETRY_CACHE="$BASE/geometry_train"
PARENT="$BASE/v108_artifacts/parent_h256_seed0.pth"
GEOMETRY="$BASE/v108_artifacts/geometry_h256_seed0.pth"
V109_ANCHOR="$BASE/v109_artifacts/nested_policy_h128_seed0_fullfit.pth"
OOF="$BASE/v113_meshsp_asymmetric_risk_replay.json"
OOF_SCRIPT="$ROOT/scripts/run_v113_meshsp_asymmetric_risk_replay.py"
ARTIFACT_DIR="$BASE/v113_artifacts"
ARTIFACT="$ARTIFACT_DIR/asymmetric_risk_committee_h128_seeds0_1_2_fullfit.pth"
ARTIFACT_RECEIPT="$ARTIFACT_DIR/artifact_receipt.json"
LOG="$BASE/v113_artifact_build.log"
PIPELINE_LOG="$BASE/v113_artifact_pipeline.log"
EXIT_FILE="$BASE/v113_artifact_pipeline_exitcode.txt"
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
    if [[ -e "$path" || -L "$path" ]]; then
        echo "V113 artifact output already exists: $path" >&2
        exit 64
    fi
done

exec >"$PIPELINE_LOG" 2>&1
echo "policy=single_gpu_v113_asymmetric_risk_committee_fullfit_artifact"
date -Is

expected_sha256() {
    local expected="$1"
    local path="$2"
    [[ -f "$path" && ! -L "$path" ]] || return 1
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]]
}

expected_sha256 ced399bca041cfa1f4213671100347f4a2423783aee4936ce7a82f785605e61d "$OOF" || finish 80
expected_sha256 439c75c081c3f445564ad36a55dfb4ab92443061ee889301297081ab4b4a2ee3 "$OOF_SCRIPT" || finish 81
expected_sha256 128ce636d27234db7fca4fb23bd5d30945928d9ac9dcd1cf8139c38670a41b96 "$BASE/v112_meshsp_anchor_committee_tradeoff_oof.json" || finish 82
expected_sha256 1123df3d312e433bf14b83874de99742906907738802bf878056ca07caa7ffdd "$BASE/v112_meshsp_anchor_committee_train_oof_predictions.json.gz" || finish 83
expected_sha256 20db69ddc27680a035384277bc48cd44109215e3d7d1158cdc4a4f21ff7c785b "$V109_ANCHOR" || finish 84
expected_sha256 7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f "$PARENT" || finish 85
expected_sha256 20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972 "$GEOMETRY" || finish 86
expected_sha256 bfe2a650e22459d09dcbca6f525cbcda136787b456bf77d734c1e2f76b67caaa "$BASE/candidate_train_receipt.json" || finish 87
expected_sha256 e45adaafb3730f45dabcea7f0c4f4492a6ea6360b7f07bdb164270bd934d9443 "$BASE/geometry_train_receipt.json" || finish 88
expected_sha256 fc4ef0c019c4b4cdabfa7e68613605aeae9d35b47f9165114b741217f2ad6d6d "$TRAIN_CACHE/manifest.json" || finish 89
expected_sha256 5b97b82e99ba5512852be6f9db4184aaf0959d77be3217da4856ef9271f93ec2 "$GEOMETRY_CACHE/manifest.json" || finish 90
expected_sha256 9875fa881b4d81aae92fc4f1f033c06de252073b93de2e9e76c85ba53fde8a8f "$ROOT/scripts/build_v113_meshsp_asymmetric_risk_artifact.py" || finish 91
expected_sha256 fa56d3da22b9ce0c8c6389173ff4f45c3407818d7a73c2aeab9f44ce81722d4a "$ROOT/models/rec_pareto_contextual_hierarchy.py" || finish 92
expected_sha256 eed77d2b4b75923eb74e37dd7af5565b4459170e8c28614bc4971544bc03d89d "$ROOT/scripts/run_v111_meshsp_anchor_committee_oof.py" || finish 93
expected_sha256 e26f64dc98b1e34a9bf648b3b4a9179a9e948a375e493c1953e760bd6c0595b1 "$ROOT/scripts/run_v95_threshold_aligned_listwise_hierarchical.py" || finish 94
expected_sha256 ca12f2a832e93089cfb856882ba535b75807a2537fdaecfa21b5cc6f2227a94f "$ROOT/scripts/run_v97_contextual_listwise_hierarchical.py" || finish 95
expected_sha256 af5ce0419b89a58d11bbcfd27e4dfb40b163e65009877b0ed83a576a95956efa "$ROOT/scripts/build_v108_meshsp_pareto_artifact.py" || finish 96
expected_sha256 93d5dca1b284d5cb4a34902b69a21a287c589a06e8c0ee43f12e375c608e5b98 "$ROOT/scripts/build_v109_meshsp_nested_policy_artifact.py" || finish 97
expected_sha256 94dbce107e8412e00ac777cdea732cf6f93d6a7985550258ef7ae46763053c8d "$ROOT/scripts/run_v108_meshsp_pareto_oof.py" || finish 98
expected_sha256 948075df82a17685e102c1913eff44f2ee032cc3e999bd5c3341aaa1b689aff3 "$ROOT/scripts/train_scanrefer_rec_hierarchical_reranker.py" || finish 99

if [[ "$(nvidia-smi -L | wc -l)" -ne 1 ]]; then
    echo "V113 requires exactly one visible physical GPU"
    finish 100
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
    echo "GPU0 is not idle before V113 artifact fit"
    finish 101
fi
free_bytes=$(df --output=avail -B1 /root/autodl-tmp | tail -n1 | tr -d ' ')
if [[ ! "$free_bytes" =~ ^[0-9]+$ ]] || (( free_bytes < 2684354560 )); then
    echo "V113 requires at least 2.5 GiB free under /root/autodl-tmp"
    finish 102
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/pointnet2"
echo "stage=three_member_full_train_fit_and_strict_reload"
date -Is
"$PY" scripts/build_v113_meshsp_asymmetric_risk_artifact.py \
    --v113-result "$OOF" \
    --v113-script "$OOF_SCRIPT" \
    --v109-anchor-artifact "$V109_ANCHOR" \
    --base-cache "$TRAIN_CACHE" \
    --geometry-cache "$GEOMETRY_CACHE" \
    --parent-artifact "$PARENT" \
    --geometry-artifact "$GEOMETRY" \
    --artifact-output "$ARTIFACT" \
    --receipt-output "$ARTIFACT_RECEIPT" \
    --device cuda:0 >"$LOG" 2>&1
rc=$?
if [[ "$rc" -ne 0 ]]; then
    echo "V113 artifact build failed: rc=$rc"
    finish "$rc"
fi
if [[ ! -f "$ARTIFACT" || -L "$ARTIFACT"
        || ! -f "$ARTIFACT_RECEIPT" || -L "$ARTIFACT_RECEIPT" ]]; then
    echo "V113 artifact build did not publish both regular outputs"
    finish 103
fi
chmod 0444 "$ARTIFACT" "$ARTIFACT_RECEIPT" "$LOG"
chmod 0555 "$ARTIFACT_DIR"
printf '0\n' >"$EXIT_FILE"
chmod 0444 "$EXIT_FILE"
echo "V113 artifact pipeline rc=0"
date -Is
chmod 0444 "$PIPELINE_LOG"
exit 0
