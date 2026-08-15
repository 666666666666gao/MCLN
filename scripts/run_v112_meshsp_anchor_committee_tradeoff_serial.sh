#!/usr/bin/env bash
set -euo pipefail

REPO=/tmp/mcln_repo
PYTHON=/root/miniconda3/envs/bdetr/bin/python
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
BASE_CACHE=$BASE/train
GEOMETRY_CACHE=$BASE/geometry_train
V109_REPORT=$BASE/v109_meshsp_nested_policy_oof.json
V110_REPORT=$BASE/v110_meshsp_uncertainty_ensemble_oof.json
V111_REPORT=$BASE/v111_meshsp_anchor_committee_oof.json
PARENT=$BASE/v108_artifacts/parent_h256_seed0.pth
GEOMETRY=$BASE/v108_artifacts/geometry_h256_seed0.pth
FALLBACK=/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train_missing789.txt
OUTPUT=$BASE/v112_meshsp_anchor_committee_tradeoff_oof.json
CACHE=$BASE/v112_meshsp_anchor_committee_train_oof_predictions.json.gz
LOG=$BASE/v112_meshsp_anchor_committee_tradeoff_oof.log
EXIT=$BASE/v112_meshsp_anchor_committee_tradeoff_oof.exit
SESSION=v112_anchor_tradeoff_oof
SELF=$(readlink -f "$0")

require_sha() {
    local expected=$1
    local path=$2
    local actual
    actual=$(sha256sum "$path" | awk '{print $1}')
    if [[ "$actual" != "$expected" ]]; then
        echo "SHA-256 mismatch: $path" >&2
        echo "expected=$expected actual=$actual" >&2
        exit 73
    fi
}

if [[ "${1:-}" == "--inner" ]]; then
    cd "$REPO"
    set +e
    env CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH="$REPO:$REPO/pointnet2" \
        "$PYTHON" scripts/run_v112_meshsp_anchor_committee_tradeoff_oof.py \
        --v109-report "$V109_REPORT" \
        --v110-report "$V110_REPORT" \
        --v111-report "$V111_REPORT" \
        --base-cache "$BASE_CACHE" \
        --geometry-cache "$GEOMETRY_CACHE" \
        --parent-artifact "$PARENT" \
        --geometry-artifact "$GEOMETRY" \
        --fallback-scenes "$FALLBACK" \
        --output "$OUTPUT" \
        --prediction-cache "$CACHE" \
        --device cuda:0 >"$LOG" 2>&1
    rc=$?
    set -e
    chmod 0444 "$LOG"
    printf '%s\n' "$rc" >"$EXIT"
    chmod 0444 "$EXIT"
    exit "$rc"
fi

cd "$REPO"
require_sha 5f03702325d6ed93f7fe15348a0161032e95dac676bc06f35548f57131b3ce1b \
    scripts/run_v112_meshsp_anchor_committee_tradeoff_oof.py
require_sha eed77d2b4b75923eb74e37dd7af5565b4459170e8c28614bc4971544bc03d89d \
    scripts/run_v111_meshsp_anchor_committee_oof.py
require_sha 95ef5e9d69308bb104186bdee1badfc4134671ce339ac72ff134b7d1d268f596 \
    scripts/run_v110_meshsp_uncertainty_ensemble_oof.py
require_sha aa77fadd55d8e579ad65748f8e1a078cbb756eaf53b061ec0339b73495a17c33 \
    scripts/run_v109_meshsp_nested_policy_oof.py
require_sha 78ff3fb141ab9aa8334285cd1d9e3c37845c7769710b166ee0fce00c33fac4a9 \
    scripts/run_v99_pareto_contextual_hierarchical.py
require_sha e26f64dc98b1e34a9bf648b3b4a9179a9e948a375e493c1953e760bd6c0595b1 \
    scripts/run_v95_threshold_aligned_listwise_hierarchical.py
require_sha 94dbce107e8412e00ac777cdea732cf6f93d6a7985550258ef7ae46763053c8d \
    scripts/run_v108_meshsp_pareto_oof.py
require_sha 948075df82a17685e102c1913eff44f2ee032cc3e999bd5c3341aaa1b689aff3 \
    scripts/train_scanrefer_rec_hierarchical_reranker.py
require_sha ca12f2a832e93089cfb856882ba535b75807a2537fdaecfa21b5cc6f2227a94f \
    scripts/run_v97_contextual_listwise_hierarchical.py
require_sha dc55bb4f6f2e827ba75811a0139bd0e2d97141e157a31fee43eb6ac45d89a60b \
    models/rec_hierarchical_reranker.py
require_sha 37680aaa34757cf9bb2376e93629ae6b89aa6b8fac16960ac091305cc20146a1 \
    "$V109_REPORT"
require_sha 7970a54bbf8a26ca09370be6a2413e436dbcb408dbc1dc1eec0a162ee40f8d48 \
    "$V110_REPORT"
require_sha 1455ec6044104932c1ecfd89c3bcc17a1e0a31cb85c518ca15db209390575a58 \
    "$V111_REPORT"
require_sha bfe2a650e22459d09dcbca6f525cbcda136787b456bf77d734c1e2f76b67caaa \
    "$BASE/candidate_train_receipt.json"
require_sha e45adaafb3730f45dabcea7f0c4f4492a6ea6360b7f07bdb164270bd934d9443 \
    "$BASE/geometry_train_receipt.json"
require_sha fc4ef0c019c4b4cdabfa7e68613605aeae9d35b47f9165114b741217f2ad6d6d \
    "$BASE_CACHE/manifest.json"
require_sha 5b97b82e99ba5512852be6f9db4184aaf0959d77be3217da4856ef9271f93ec2 \
    "$GEOMETRY_CACHE/manifest.json"
require_sha caf63109bdf9f19cd8132b3c70eb1f2467d70fc605d174c6ec801b34c1c31079 \
    "$FALLBACK"
require_sha 7b8956e854df3e2030a091e45e0b17ff2a9b56555d4bef660200f94d0c3b616f \
    "$PARENT"
require_sha 20f33cf46d3e296529aa817f58729bf73783a6637b0e4bc8221ff730e9897972 \
    "$GEOMETRY"
require_sha 3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208 \
    /root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth

for path in "$OUTPUT" "$CACHE" "$LOG" "$EXIT"; do
    if [[ -e "$path" ]]; then
        echo "Refusing to overwrite existing V112 output: $path" >&2
        exit 74
    fi
done
if [[ $(nvidia-smi -L | wc -l) -ne 1 ]]; then
    echo "V112 requires the frozen single-GPU topology" >&2
    exit 75
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | grep -Eq '^[[:space:]]*[0-9]+'; then
    echo "GPU already has a compute process" >&2
    exit 75
fi
if pgrep -af '[r]un_v112_meshsp_anchor_committee_tradeoff_oof.py' >/dev/null; then
    echo "V112 process already exists" >&2
    exit 75
fi
if screen -ls 2>/dev/null | grep -q "[.]$SESSION"; then
    echo "V112 screen session already exists" >&2
    exit 75
fi
available_kb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
if [[ "$available_kb" -lt 2621440 ]]; then
    echo "V112 requires at least 2.5 GiB free on /root/autodl-tmp" >&2
    exit 75
fi

screen -DmS "$SESSION" /bin/bash "$SELF" --inner
sleep 1
if ! screen -ls 2>/dev/null | grep -q "[.]$SESSION" \
        && [[ ! -e "$EXIT" ]]; then
    echo "V112 screen failed to remain alive and produced no exit receipt" >&2
    exit 75
fi
echo "launched session=$SESSION output=$OUTPUT log=$LOG exit=$EXIT"
