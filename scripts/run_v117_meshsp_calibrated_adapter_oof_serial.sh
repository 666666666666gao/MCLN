#!/usr/bin/env bash
set -euo pipefail

REPO=/tmp/mcln_repo
PYTHON=/root/miniconda3/envs/bdetr/bin/python
BASE=/root/autodl-tmp/DATA_ROOT/output/rec_reranker/e71_top16_meshsp
BASE_CACHE=$BASE/train
GEOMETRY_CACHE=$BASE/geometry_train
V108_REPORT=$BASE/v108_meshsp_pareto_oof.json
V115_REPORT=$BASE/v115r1_meshsp_anchored_spatial_adapter_oof.json
V116_REPORT=$BASE/v116_meshsp_primary025_policy_oof.json
PARENT=$BASE/v108_artifacts/parent_h256_seed0.pth
GEOMETRY=$BASE/v108_artifacts/geometry_h256_seed0.pth
FALLBACK=/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train_missing789.txt
OUTPUT=$BASE/v117_meshsp_calibrated_adapter_oof.json
LOG=$BASE/v117_meshsp_calibrated_adapter_oof.log
EXIT=$BASE/v117_meshsp_calibrated_adapter_oof.exit
SESSION=v117_calibrated_adapter_oof
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
        "$PYTHON" scripts/run_v117_meshsp_calibrated_adapter_oof.py \
        --v108-report "$V108_REPORT" \
        --v115-report "$V115_REPORT" \
        --v116-report "$V116_REPORT" \
        --base-cache "$BASE_CACHE" \
        --geometry-cache "$GEOMETRY_CACHE" \
        --parent-artifact "$PARENT" \
        --geometry-artifact "$GEOMETRY" \
        --fallback-scenes "$FALLBACK" \
        --output "$OUTPUT" \
        --device cuda:0 >"$LOG" 2>&1
    rc=$?
    set -e
    chmod 0444 "$LOG"
    printf '%s\n' "$rc" >"$EXIT"
    chmod 0444 "$EXIT"
    exit "$rc"
fi

cd "$REPO"
require_sha b9729032fb3771092ee0d51d88e589050b4092e9ede6b60f499fe5d4b3035f65 \
    scripts/run_v117_meshsp_calibrated_adapter_oof.py
require_sha 45cc25d9209d1300f5c6f5ff2f9620acefdf47e8166451504aa7fdfb37c9b0cd \
    models/rec_anchored_spatial_adapter.py
require_sha fa56d3da22b9ce0c8c6389173ff4f45c3407818d7a73c2aeab9f44ce81722d4a \
    models/rec_pareto_contextual_hierarchy.py
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
require_sha 6be03732f8067a6d6138d7a4f33378e7448df61ed9bda641db9c590160edd1d0 \
    models/encoder_decoder_layers.py
require_sha 72ca54b2db0bca829011a2f480c458c0a3e450a492dd77de9d8411e84f3e9162 \
    "$V108_REPORT"
require_sha cae35808390c5f8c86b5ed3eeb73219ac226a20c7be091e36986fa25cf5f423f \
    "$V115_REPORT"
require_sha 18fddfca24719062cc83b6b8e1c11183b04bb4e1b09da263d7e8b0db938ccdb9 \
    "$V116_REPORT"
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

for path in "$OUTPUT" "$LOG" "$EXIT"; do
    if [[ -e "$path" ]]; then
        echo "Refusing to overwrite existing V117 output: $path" >&2
        exit 74
    fi
done
if [[ $(nvidia-smi -L | wc -l) -ne 1 ]]; then
    echo "V117 requires the frozen single-GPU topology" >&2
    exit 75
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | grep -Eq '^[[:space:]]*[0-9]+'; then
    echo "GPU already has a compute process" >&2
    exit 75
fi
if pgrep -af '[r]un_v117_meshsp_calibrated_adapter_oof.py' >/dev/null; then
    echo "V117 process already exists" >&2
    exit 75
fi
if screen -ls 2>/dev/null | grep -q "[.]$SESSION"; then
    echo "V117 screen session already exists" >&2
    exit 75
fi
available_kb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
if [[ "$available_kb" -lt 2621440 ]]; then
    echo "V117 requires at least 2.5 GiB free on /root/autodl-tmp" >&2
    exit 75
fi

screen -DmS "$SESSION" /bin/bash "$SELF" --inner
sleep 1
if ! screen -ls 2>/dev/null | grep -q "[.]$SESSION" \
        && [[ ! -e "$EXIT" ]]; then
    echo "V117 screen failed to remain alive and produced no exit receipt" >&2
    exit 75
fi
echo "launched session=$SESSION output=$OUTPUT log=$LOG exit=$EXIT"
