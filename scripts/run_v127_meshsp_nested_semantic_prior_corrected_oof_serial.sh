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
V117_REPORT=$BASE/v117_meshsp_calibrated_adapter_oof.json
V118_REPORT=$BASE/v118_meshsp_nested_pairwise_risk_oof.json
V119_REPORT=$BASE/v119_meshsp_nested_break_veto_oof.json
V120_REPORT=$BASE/v120_meshsp_nested_outcome_classifier_oof.json
V121_REPORT=$BASE/v121_meshsp_nested_semantic_critic_oof.json
V122_REPORT=$BASE/v122_meshsp_nested_semantic_pairwise_oof.json
V123_REPORT=$BASE/v123_meshsp_nested_semantic_antisymmetric_oof.json
V124_REPORT=$BASE/v124_meshsp_nested_semantic_learned_pareto_oof.json
V125_REPORT=$BASE/v125_meshsp_nested_semantic_all_candidate_oof.json
V126_REPORT=$BASE/v126_meshsp_nested_semantic_listwise_oof.json
PARENT=$BASE/v108_artifacts/parent_h256_seed0.pth
GEOMETRY=$BASE/v108_artifacts/geometry_h256_seed0.pth
FALLBACK=/root/autodl-tmp/DATA_ROOT/superpoints_mesh_official/train_missing789.txt
OUTPUT=$BASE/v127_meshsp_nested_semantic_prior_corrected_oof.json
LOG=$BASE/v127_meshsp_nested_semantic_prior_corrected_oof.log
EXIT=$BASE/v127_meshsp_nested_semantic_prior_corrected_oof.exit
SESSION=v127_nested_semantic_prior_corrected_oof
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
        "$PYTHON" scripts/run_v127_meshsp_nested_semantic_prior_corrected_oof.py \
        --v108-report "$V108_REPORT" \
        --v115-report "$V115_REPORT" \
        --v116-report "$V116_REPORT" \
        --v117-report "$V117_REPORT" \
        --v118-report "$V118_REPORT" \
        --v119-report "$V119_REPORT" \
        --v120-report "$V120_REPORT" \
        --v121-report "$V121_REPORT" \
        --v122-report "$V122_REPORT" \
        --v123-report "$V123_REPORT" \
        --v124-report "$V124_REPORT" \
        --v125-report "$V125_REPORT" \
        --v126-report "$V126_REPORT" \
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
require_sha 0da18187a7690ad9e5d57feedbc6bb9ef69fe723dd228ee7969e1f5c4277fe12 \
    scripts/run_v127_meshsp_nested_semantic_prior_corrected_oof.py
require_sha 21f49b070bf72af67a33c0087246c8367955b1ef531f5b8cc83b242a12c8499b \
    models/rec_semantic_antisymmetric_utility.py
require_sha d16798f49c94a9fd36b03e22002dff1a0bdbf7120bb1760badc721426e0d5a6f \
    models/rec_semantic_candidate_critic.py
require_sha b650cd738e004a3d2febba0bcf23d852bc1ca1fb9e845bfe1530dd466a2bf3cc \
    models/rec_pairwise_switch_classifier.py
require_sha 15b93e0c7978e461c73221c0142070641c51bf252633f0d8fbacff47ee65cffd \
    models/rec_pairwise_switch_risk.py
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
require_sha 6e43afe461745ba4c65956d39f4e2fed7c62fd17d59a4641118549b1e1fc6c00 \
    "$V117_REPORT"
require_sha 8611a9bd24ab6e4d09e05dc37833f8e5d9dfc34e4c1be647ec66ecd4f10958da \
    "$V118_REPORT"
require_sha 731d110af0c8954d2b6ff5a5e8930c9b4897eaf4f71d96a89c720c7bbbd2ee8a \
    "$V119_REPORT"
require_sha bd8272d8c5337b60136f3d74dd50233c0a8c0e82b8f17287daf8505cd96afd87 \
    "$V120_REPORT"
require_sha af29255b71e19b41879e287abfeb99b368b9fb9eec81cfb5377045bf950ded0d \
    "$V121_REPORT"
require_sha 95fad62c6b1e8df313292dcf88f07c3d3bfde33a395c05d254162b7f4a9b2321 \
    "$V122_REPORT"
require_sha be05e2e5ac077c19852981dcc1280ff18a27bfe5253cc2900a9ca6c272155b2d \
    "$V123_REPORT"
require_sha cf430277b6e5f3c2f45ef8622fcb6b66df3022fa24d9252fb584c5c4377bc172 \
    "$V124_REPORT"
require_sha b52c96f7a641a069714dc36bf6cc2dbe055b6937693a441291bc0b1fc0e1806b \
    "$V125_REPORT"
require_sha 899a884cdd362e13f9d1772e53bd8cca01af9bf849d66417eee33df6fd7e8c53 \
    "$V126_REPORT"
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
        echo "Refusing to overwrite existing V127 output: $path" >&2
        exit 74
    fi
done
if [[ $(nvidia-smi -L | wc -l) -ne 1 ]]; then
    echo "V127 requires the frozen single-GPU topology" >&2
    exit 75
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | grep -Eq '^[[:space:]]*[0-9]+'; then
    echo "GPU already has a compute process" >&2
    exit 75
fi
if pgrep -af '[r]un_v127_meshsp_nested_semantic_prior_corrected_oof.py' >/dev/null; then
    echo "V127 process already exists" >&2
    exit 75
fi
if screen -ls 2>/dev/null | grep -q "[.]$SESSION"; then
    echo "V127 screen session already exists" >&2
    exit 75
fi
available_kb=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
if [[ "$available_kb" -lt 2621440 ]]; then
    echo "V127 requires at least 2.5 GiB free on /root/autodl-tmp" >&2
    exit 75
fi

screen -DmS "$SESSION" /bin/bash "$SELF" --inner
sleep 1
if ! screen -ls 2>/dev/null | grep -q "[.]$SESSION" \
        && [[ ! -e "$EXIT" ]]; then
    echo "V127 screen failed to remain alive and produced no exit receipt" >&2
    exit 75
fi
echo "launched session=$SESSION output=$OUTPUT log=$LOG exit=$EXIT"
