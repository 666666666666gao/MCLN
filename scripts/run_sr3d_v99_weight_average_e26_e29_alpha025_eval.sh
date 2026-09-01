#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly GROUPFREE_CHECKPOINT="${DATA_ROOT%/}/gf_detector_l6o256.pth"
readonly GROUPFREE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v99_baseline_gt/sr3d"
readonly BASE_CHECKPOINT="${OUTPUT_ROOT}/control/official_rec_monitor/official_best_rec025_epoch_26_0p68391064.pth"
readonly OTHER_CHECKPOINT="${OUTPUT_ROOT}/backbone/sr3d_mcln_joint_butdcls_v99_official_e25_plateau_lr100_e26_e29_b12a2_w4p2_20260828_174605/sr3d/sr3d_mcln_joint_butdcls_v99_official_e25_plateau_lr100_e26_e29_b12a2_w4p2/1787910373/ckpt_epoch_29.pth"
readonly BASE_SHA256="4ac72dd3d33bb6aa13278e4e67208d98f006a9863396b3f2ab3713a9c904fd1d"
readonly OTHER_SHA256="a5e964ac24867bd1a02874fa7362bf0cba95ef2573acb97b489a5c7f7224442f"
readonly MAIN_UTILS_SHA256="68760df2095b44711395cc87b9f23b258637d50b5738c389acfb2f1db09367ce"
readonly TRAIN_ENTRY_SHA256="0669b5535eeb75954c76422c333556679faa149d8e2c289e60668a5377e2c7cc"
readonly MODEL_SHA256="3b9a2b88e9fa36c6f94b1595ae3812818ca5fdeb4b1db64c68f467912c04c9b1"
readonly LOSSES_SHA256="1a08a7febc9bd94e1d389d2ef7f987908dd796a5f25cebccd433e155bbaa5aaf"
readonly DATASET_SHA256="49af3c83091970ab307d0ca2e95a68ae42a6fcf52ddf72566b9da966ce6ad7a2"
readonly EVALUATOR_SHA256="0173b31a7a818f872c210b01a4e5d17601c4e5f10ec8d97f78c7e537fa44e062"
readonly BUILDER_SHA256="07cfb23ea758f3b8c3c0c2237855840b411a477414236d119f53ea523dca44c2"
readonly EXPECTED_SAMPLE_COUNT=17726
readonly BASE_HITS025=12123
readonly TARGET_HITS025=12125
readonly OTHER_WEIGHT="0.25"
readonly MASTER_PORT=5517
readonly MIN_FREE_GB=3
readonly EXP="sr3d_mcln_joint_butdcls_v99_weight_average_e26_075_e29_025_eval"
readonly RUN_ROOT="${OUTPUT_ROOT}/evaluation/${EXP}_one_shot"
readonly LOG_DIR="${RUN_ROOT}/sr3d/${EXP}"
readonly AVG_CHECKPOINT="${RUN_ROOT}/candidate_e26_075_e29_025.pth"
readonly AVG_MANIFEST="${RUN_ROOT}/weight_average_manifest.json"

require_sha() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; exit 3; }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA-256 changed: ${actual}" >&2
    exit 3
  }
}

cd "${ROOT_DIR}"
require_sha "${BASE_CHECKPOINT}" "${BASE_SHA256}" "E26 checkpoint"
require_sha "${OTHER_CHECKPOINT}" "${OTHER_SHA256}" "E29 checkpoint"
require_sha "${ROOT_DIR}/main_utils.py" "${MAIN_UTILS_SHA256}" "main_utils.py"
require_sha "${ROOT_DIR}/train_dist_mod.py" "${TRAIN_ENTRY_SHA256}" "train_dist_mod.py"
require_sha "${ROOT_DIR}/models/mcln.py" "${MODEL_SHA256}" "models/mcln.py"
require_sha "${ROOT_DIR}/models/losses.py" "${LOSSES_SHA256}" "models/losses.py"
require_sha "${ROOT_DIR}/src/joint_det_dataset.py" "${DATASET_SHA256}" "dataset"
require_sha "${ROOT_DIR}/src/grounding_evaluator.py" "${EVALUATOR_SHA256}" "grounding evaluator"
require_sha "${ROOT_DIR}/scripts/build_weight_average_checkpoint.py" "${BUILDER_SHA256}" "weight-average builder"
require_sha "${GROUPFREE_CHECKPOINT}" "${GROUPFREE_SHA256}" "GroupFree checkpoint"

free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
gpu_used="$(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 |
    tr -d ' '
)"
if ((gpu_used >= 500)); then
  echo "GPU0 is busy (${gpu_used} MiB)" >&2
  exit 4
fi
if ((free_gb < MIN_FREE_GB)); then
  echo "need at least ${MIN_FREE_GB} GiB free under DATA_ROOT" >&2
  exit 5
fi

readonly LOCK_FILE="${DATA_ROOT%/}/output/network_v99/single_gpu.lock"
mkdir -p "$(dirname "${LOCK_FILE}")"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another V99 job owns ${LOCK_FILE}" >&2
  exit 6
fi

if [[ -e "${RUN_ROOT}" ]]; then
  echo "one-shot Sr3D candidate was already claimed: ${RUN_ROOT}" >&2
  exit 7
fi
mkdir -p "${LOG_DIR}"
"${PYTHON_BIN}" - \
  "${RUN_ROOT}/one_shot_claim.json" "${ROOT_DIR}" \
  "${BASH_SOURCE[0]}" "${GROUPFREE_CHECKPOINT}" \
  "${EXPECTED_SAMPLE_COUNT}" "${BASE_HITS025}" "${TARGET_HITS025}" \
  "${BUILDER_SHA256}" "${MAIN_UTILS_SHA256}" \
  "${TRAIN_ENTRY_SHA256}" "${MODEL_SHA256}" "${LOSSES_SHA256}" \
  "${DATASET_SHA256}" "${EVALUATOR_SHA256}" "${GROUPFREE_SHA256}" <<'PY'
from __future__ import print_function
import hashlib
import json
import os
import sys

(
    claim_path,
    root_dir,
    launcher_path,
    groupfree_path,
    sample_count,
    base_hits025,
    target_hits025,
    expected_builder_sha,
    expected_main_utils_sha,
    expected_train_entry_sha,
    expected_model_sha,
    expected_losses_sha,
    expected_dataset_sha,
    expected_evaluator_sha,
    expected_groupfree_sha,
) = sys.argv[1:]

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

paths = {
    "launcher": os.path.realpath(launcher_path),
    "builder": os.path.join(root_dir, "scripts", "build_weight_average_checkpoint.py"),
    "main_utils": os.path.join(root_dir, "main_utils.py"),
    "train_entry": os.path.join(root_dir, "train_dist_mod.py"),
    "model": os.path.join(root_dir, "models", "mcln.py"),
    "losses": os.path.join(root_dir, "models", "losses.py"),
    "dataset": os.path.join(root_dir, "src", "joint_det_dataset.py"),
    "evaluator": os.path.join(root_dir, "src", "grounding_evaluator.py"),
    "groupfree_checkpoint": os.path.realpath(groupfree_path),
}
expected_sha256 = {
    "builder": expected_builder_sha,
    "main_utils": expected_main_utils_sha,
    "train_entry": expected_train_entry_sha,
    "model": expected_model_sha,
    "losses": expected_losses_sha,
    "dataset": expected_dataset_sha,
    "evaluator": expected_evaluator_sha,
    "groupfree_checkpoint": expected_groupfree_sha,
}
current_sha256 = {name: sha256_file(path) for name, path in paths.items()}
for name, expected in expected_sha256.items():
    if current_sha256[name] != expected:
        raise SystemExit(
            "{} drifted between fixed gate and one-shot claim".format(name)
        )
claim = {
    "schema": "mcln-v99-weight-average-one-shot-claim-v1",
    "candidate": "0.75*E26+0.25*E29",
    "evaluation_only": True,
    "sample_count": int(sample_count),
    "base_hits025": int(base_hits025),
    "strict_target_hits025": int(target_hits025),
    "code_paths": paths,
    "expected_sha256": expected_sha256,
    "code_sha256": current_sha256,
}
with open(claim_path, "x") as handle:
    json.dump(claim, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
chmod 0444 "${RUN_ROOT}/one_shot_claim.json"

"${PYTHON_BIN}" scripts/build_weight_average_checkpoint.py \
  --base "${BASE_CHECKPOINT}" --base-sha256 "${BASE_SHA256}" \
  --other "${OTHER_CHECKPOINT}" --other-sha256 "${OTHER_SHA256}" \
  --other-weight "${OTHER_WEIGHT}" \
  --output "${AVG_CHECKPOINT}" --manifest "${AVG_MANIFEST}"
chmod 0444 "${AVG_CHECKPOINT}" "${AVG_MANIFEST}"
ln "${AVG_CHECKPOINT}" "${LOG_DIR}/ckpt_epoch_26.pth"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/pointnet2:${PYTHONPATH:-}"

eval_command=(
  "${PYTHON_BIN}" -m torch.distributed.launch
  --nproc_per_node 1 --master_port "${MASTER_PORT}" \
  train_dist_mod.py \
  --eval --checkpoint_path "${AVG_CHECKPOINT}" --checkpoint_start_epoch 26 \
  --num_decoder_layers 6 --num_target 256 \
  --use_color --weight_decay 0.0005 \
  --data_root "${DATA_ROOT}" --batch_size 12 \
  --num_workers 4 --dataloader_prefetch_factor 2 --persistent_train_workers \
  --dataset sr3d --test_dataset sr3d --joint_det --butd_cls \
  --lr_backbone 1e-3 --lr 1e-4 --lr_decay_epochs 30 40 --warmup-epoch 0 \
  --detect_intermediate --use_soft_token_loss --use_contrastive_align \
  --log_dir "${LOG_DIR}" \
  --pp_checkpoint "${GROUPFREE_CHECKPOINT}" \
  --self_attend --skip_missing_superpoints \
  --start_epoch 1 --max_epoch 26 --model MCLN --exp "${EXP}" \
  --use_source_choice_selector --eval_use_selector_choice_scores \
  --source_choice_selector_sources default,default_rank_blend_contrastive010 \
  --source_choice_selector_default_source default \
  --source_choice_selector_hidden_dim 288 \
  --source_choice_selector_lr 1.25e-4 \
  --source_choice_selector_loss_weight 0.5 \
  --source_choice_selector_choice_target precision_gain_default_sourcewise_focal_bce \
  --source_choice_selector_min_iou_gap 0.03 \
  --expected_eval_sample_count "${EXPECTED_SAMPLE_COUNT}"
)
printf '%q ' "${eval_command[@]}" >"${RUN_ROOT}/eval_command.txt"
printf '\n' >>"${RUN_ROOT}/eval_command.txt"
chmod 0444 "${RUN_ROOT}/eval_command.txt"
"${PYTHON_BIN}" - \
  "${RUN_ROOT}/one_shot_claim.json" "${AVG_MANIFEST}" \
  "${AVG_CHECKPOINT}" "${RUN_ROOT}/eval_command.txt" \
  "${RUN_ROOT}/pre_eval_provenance.json" <<'PY'
from __future__ import print_function
import hashlib
import json
import os
import sys

claim_path, manifest_path, checkpoint_path, command_path, output_path = sys.argv[1:]

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_json_with_sha(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()

claim, claim_sha = load_json_with_sha(claim_path)
manifest, manifest_sha = load_json_with_sha(manifest_path)
current_code = {
    name: sha256_file(path)
    for name, path in claim["code_paths"].items()
}
if current_code != claim["code_sha256"]:
    raise SystemExit("code or GroupFree drifted after the one-shot claim")
checkpoint_sha = sha256_file(checkpoint_path)
if checkpoint_sha != manifest.get("output_sha256"):
    raise SystemExit("candidate checkpoint does not match its manifest")
pre_eval = {
    "schema": "mcln-v99-weight-average-pre-eval-provenance-v1",
    "claim_sha256": claim_sha,
    "manifest_sha256": manifest_sha,
    "candidate_checkpoint_sha256": checkpoint_sha,
    "eval_command_sha256": sha256_file(command_path),
    "code_sha256": current_code,
}
with open(output_path, "x") as handle:
    json.dump(pre_eval, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
chmod 0444 "${RUN_ROOT}/pre_eval_provenance.json"
"${eval_command[@]}" 2>&1 | tee "${RUN_ROOT}/eval.log"

readonly RECEIPT="${LOG_DIR}/eval_metrics_epoch_26.json"
readonly DECISION="${RUN_ROOT}/decision.json"
"${PYTHON_BIN}" - "${RECEIPT}" "${DECISION}" \
  "${AVG_MANIFEST}" "${AVG_CHECKPOINT}" \
  "${RUN_ROOT}/one_shot_claim.json" "${RUN_ROOT}/eval_command.txt" \
  "${RUN_ROOT}/pre_eval_provenance.json" \
  "${ROOT_DIR}" "${BASH_SOURCE[0]}" "${GROUPFREE_CHECKPOINT}" \
  "${EXPECTED_SAMPLE_COUNT}" "${BASE_HITS025}" "${TARGET_HITS025}" <<'PY'
from __future__ import print_function
import hashlib
import json
import math
import os
import sys

(
    receipt_path,
    decision_path,
    manifest_path,
    checkpoint_path,
    claim_path,
    command_path,
    pre_eval_path,
    root_dir,
    launcher_path,
    groupfree_path,
    expected_sample_count,
    base_hits025,
    target_hits025,
) = sys.argv[1:]
expected_sample_count = int(expected_sample_count)
base_hits025 = int(base_hits025)
target_hits025 = int(target_hits025)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_json_with_sha(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()

receipt, receipt_sha = load_json_with_sha(receipt_path)
manifest, manifest_sha = load_json_with_sha(manifest_path)
claim, claim_sha = load_json_with_sha(claim_path)
pre_eval, pre_eval_sha = load_json_with_sha(pre_eval_path)
current_code = {
    name: sha256_file(path)
    for name, path in claim["code_paths"].items()
}
if current_code != claim.get("code_sha256"):
    raise SystemExit("code or GroupFree drifted during evaluation")
if current_code != pre_eval.get("code_sha256"):
    raise SystemExit("post-eval code hashes differ from pre-eval hashes")
actual_checkpoint_sha = sha256_file(checkpoint_path)
if actual_checkpoint_sha != manifest.get("output_sha256"):
    raise SystemExit("post-eval candidate checkpoint SHA mismatch")
if actual_checkpoint_sha != pre_eval.get("candidate_checkpoint_sha256"):
    raise SystemExit("candidate checkpoint drifted during evaluation")
actual_manifest_sha = manifest_sha
if actual_manifest_sha != pre_eval.get("manifest_sha256"):
    raise SystemExit("weight-average manifest drifted during evaluation")
actual_claim_sha = claim_sha
if actual_claim_sha != pre_eval.get("claim_sha256"):
    raise SystemExit("one-shot claim drifted during evaluation")
actual_command_sha = sha256_file(command_path)
if actual_command_sha != pre_eval.get("eval_command_sha256"):
    raise SystemExit("evaluation command drifted during evaluation")
metric = receipt.get("position_subgroups", {}).get("multiple", {})
if receipt.get("schema") != "mcln-retrain-metrics-v1":
    raise SystemExit("evaluation receipt schema mismatch")
if (
        receipt.get("sample_count") != expected_sample_count
        or metric.get("sample_count") != expected_sample_count):
    raise SystemExit("evaluation sample-count mismatch")
hits025 = int(metric.get("hits025", -1))
hits050 = int(metric.get("hits050", -1))
acc025 = float(metric.get("acc025", float("nan")))
acc050 = float(metric.get("acc050", float("nan")))
if not math.isclose(
        acc025, hits025 / float(expected_sample_count),
        rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("REC@0.25 hits/accuracy mismatch")
if not math.isclose(
        acc050, hits050 / float(expected_sample_count),
        rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("REC@0.50 hits/accuracy mismatch")
decision = {
    "schema": "mcln-v99-weight-average-eval-v1",
    "sample_count": expected_sample_count,
    "hits025": hits025,
    "hits050": hits050,
    "acc025": acc025,
    "acc050": acc050,
    "base_hits025": base_hits025,
    "strict_target_hits025": target_hits025,
    "improves_protected_best": hits025 > base_hits025,
    "strict_target_reached": hits025 >= target_hits025,
    "checkpoint": os.path.realpath(checkpoint_path),
    "checkpoint_sha256": actual_checkpoint_sha,
    "receipt": os.path.realpath(receipt_path),
    "receipt_sha256": receipt_sha,
    "weight_average_manifest": os.path.realpath(manifest_path),
    "weight_average_manifest_sha256": actual_manifest_sha,
    "one_shot_claim": os.path.realpath(claim_path),
    "one_shot_claim_sha256": actual_claim_sha,
    "eval_command": os.path.realpath(command_path),
    "eval_command_sha256": actual_command_sha,
    "pre_eval_provenance": os.path.realpath(pre_eval_path),
    "pre_eval_provenance_sha256": pre_eval_sha,
    "code_sha256": current_code,
}
temporary = decision_path + ".tmp.{}".format(os.getpid())
with open(temporary, "w") as handle:
    json.dump(decision, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, decision_path)
print(json.dumps(decision, sort_keys=True))
PY
chmod 0444 "${RECEIPT}" "${DECISION}"
echo "SR3D_V99_WEIGHT_AVERAGE_EVAL=COMPLETE run_root=${RUN_ROOT}"
