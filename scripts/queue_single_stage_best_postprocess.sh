#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/DATA_ROOT/}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/bdetr/bin/python}"
TRAIN_EXP="${TRAIN_EXP:-mcln_epoch71_parent_geometry_single_stage_e1_e100_b18x4}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${DATA_ROOT%/}/output/single_stage_best_postprocess/scanrefer/${TRAIN_EXP}/1785907694}"
FINAL_EPOCH="${FINAL_EPOCH:-100}"
EXPECTED_SAMPLE_COUNT="${EXPECTED_SAMPLE_COUNT:-9508}"
POST_ROOT="${POST_ROOT:-${ROOT_DIR}/experiment_output/single_stage_best_postprocess/1785907694}"
MIN_FREE_KIB="${MIN_FREE_KIB:-7340032}"
QUEUE_LOG="${QUEUE_LOG:-${POST_ROOT}/queue.log}"
LOCK_FILE="${LOCK_FILE:-${POST_ROOT}/queue.lock}"
MASTER_PORT="${MASTER_PORT:-4591}"

BASE_TRAIN="${POST_ROOT}/candidate_train"
BASE_VAL="${POST_ROOT}/candidate_val"
PARENT_ARTIFACT="${POST_ROOT}/parent_reranker.pth"
AUDIT_DIR="${POST_ROOT}/mask_geometry_audit"
GEOMETRY_TRAIN="${POST_ROOT}/geometry_train"
GEOMETRY_VAL="${POST_ROOT}/geometry_val"
GEOMETRY_ARTIFACT="${POST_ROOT}/geometry_reranker.pth"
EVAL_ROOT="${POST_ROOT}/official_eval"

for value in "${FINAL_EPOCH}" "${EXPECTED_SAMPLE_COUNT}" \
             "${MIN_FREE_KIB}" "${MASTER_PORT}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "queue integer settings must be positive" >&2
    exit 2
  fi
done
for path in "${PYTHON_BIN}" "${TRAIN_RUN_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required path is missing: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${POST_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another single-stage postprocess queue owns ${LOCK_FILE}" >&2
  exit 3
fi
exec > >(tee -a "${QUEUE_LOG}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

echo "[$(timestamp)] waiting for single-stage experiment ${TRAIN_EXP}"
train_pid="$(pgrep -o -f "train_dist_mod.py.*--exp ${TRAIN_EXP}" || true)"
if [[ -n "${train_pid}" ]]; then
  echo "[$(timestamp)] bound to process ${train_pid}; no log polling"
  tail --pid="${train_pid}" -f /dev/null
fi

final_receipt="${TRAIN_RUN_DIR}/eval_metrics_epoch_${FINAL_EPOCH}.json"
final_checkpoint="${TRAIN_RUN_DIR}/ckpt_epoch_last.pth"
"${PYTHON_BIN}" scripts/audit_training_completion.py \
  --metrics "${final_receipt}" \
  --checkpoint "${final_checkpoint}" \
  --expected-epoch "${FINAL_EPOCH}" \
  --expected-sample-count "${EXPECTED_SAMPLE_COUNT}" \
  --output "${POST_ROOT}/training_completion.json"

best_checkpoint="${TRAIN_RUN_DIR}/ckpt_best_rec_acc025.pth"
if [[ ! -f "${best_checkpoint}" ]]; then
  echo "REC@0.25 best checkpoint is missing" >&2
  exit 4
fi
"${PYTHON_BIN}" - "${best_checkpoint}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
config = checkpoint.get("config")
if config is None:
    raise ValueError("best checkpoint has no config")
inputs = {
    name: bool(getattr(config, name, False))
    for name in ("butd", "butd_gt", "butd_cls")
}
if inputs != {"butd": False, "butd_gt": False, "butd_cls": False}:
    raise ValueError("best checkpoint is not single-stage: {}".format(inputs))
print("single-stage best checkpoint epoch={}".format(checkpoint.get("epoch")))
PY
sha256sum "${best_checkpoint}" | tee "${POST_ROOT}/best_checkpoint.sha256"

free_kib="$(df -Pk "${POST_ROOT}" | awk 'NR == 2 {print $4}')"
if [[ ! "${free_kib}" =~ ^[1-9][0-9]*$ \
      || "${free_kib}" -lt "${MIN_FREE_KIB}" ]]; then
  echo "postprocess filesystem has insufficient free space: ${free_kib:-unknown} KiB" >&2
  exit 4
fi

echo "[$(timestamp)] extracting train/val candidate caches in parallel"
candidate_pids=()
for split in train val; do
  gpu=0
  output="${BASE_TRAIN}"
  if [[ "${split}" == "val" ]]; then
    gpu=1
    output="${BASE_VAL}"
  fi
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
      scripts/cache_scanrefer_rec_candidates.py \
      --split "${split}" --data-root "${DATA_ROOT}" \
      --checkpoint "${best_checkpoint}" --output-dir "${output}" \
      --batch-size 24 --num-workers 4 --shard-size 256 \
      --max-candidates 16 --device cuda:0
  ) >"${POST_ROOT}/candidate_${split}.log" 2>&1 &
  candidate_pids+=("$!")
done
failed=0
for pid in "${candidate_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "candidate cache extraction failed" >&2
  exit 5
fi

echo "[$(timestamp)] training parent reranker and mask audit in parallel"
independent_pids=()
(
  CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" scripts/train_rec_reranker.py \
    --train-cache "${BASE_TRAIN}" --output "${PARENT_ARTIFACT}" \
    --seed 0 --hidden-dim 256 --dropout 0.1 --lr 1e-3 \
    --weight-decay 1e-4 --batch-size 256 --max-epochs 100 \
    --patience 10 --device cuda:0
) >"${POST_ROOT}/parent_train.log" 2>&1 &
independent_pids+=("$!")
(
  CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" scripts/audit_scanrefer_mask_geometry.py \
    --data-root "${DATA_ROOT}" --checkpoint "${best_checkpoint}" \
    --train-cache "${BASE_TRAIN}" --output-dir "${AUDIT_DIR}" \
    --scene-count 64 --expressions-per-scene 4 --selection-seed 0 \
    --batch-size 12 --num-workers 2 --device cuda:0
) >"${POST_ROOT}/mask_geometry_audit.log" 2>&1 &
independent_pids+=("$!")
failed=0
for pid in "${independent_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "parent reranker or mask geometry audit failed" >&2
  exit 6
fi

echo "[$(timestamp)] extracting train/val geometry caches in parallel"
geometry_pids=()
for split in train val; do
  gpu=0
  base="${BASE_TRAIN}"
  output="${GEOMETRY_TRAIN}"
  if [[ "${split}" == "val" ]]; then
    gpu=1
    base="${BASE_VAL}"
    output="${GEOMETRY_VAL}"
  fi
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
      scripts/cache_scanrefer_rec_mask_geometry.py \
      --split "${split}" --data-root "${DATA_ROOT}" \
      --checkpoint "${best_checkpoint}" --base-cache "${base}" \
      --output-dir "${output}" \
      --audit-provenance "${AUDIT_DIR}/selection.json" \
      --portable-provenance --audit-train-cache "${BASE_TRAIN}" \
      --batch-size 12 --num-workers 2 --shard-size 252 \
      --device cuda:0
  ) >"${POST_ROOT}/geometry_${split}.log" 2>&1 &
  geometry_pids+=("$!")
done
failed=0
for pid in "${geometry_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "geometry cache extraction failed" >&2
  exit 7
fi

echo "[$(timestamp)] training geometry reranker"
CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" scripts/train_rec_geometry_reranker.py \
  --base-cache "${BASE_TRAIN}" --geometry-cache "${GEOMETRY_TRAIN}" \
  --parent-artifact "${PARENT_ARTIFACT}" --output "${GEOMETRY_ARTIFACT}" \
  --split-seed 0 --model-seed 0 --hidden-dim 256 --dropout 0.1 \
  --lr 1e-3 --weight-decay 1e-4 --batch-size 256 \
  --max-epochs 100 --patience 10 --device cuda:0 --verbose \
  >"${POST_ROOT}/geometry_train.log" 2>&1

echo "[$(timestamp)] running contract-bound official single-stage parent+geometry evaluation"
mkdir -p "${EVAL_ROOT}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node 1 --master_port "${MASTER_PORT}" \
  train_dist_mod.py --num_decoder_layers 6 --num_target 256 \
  --model MCLN --use_color --self_attend --detect_intermediate --joint_det \
  --use_soft_token_loss --use_contrastive_align \
  --use_source_choice_selector --eval_use_selector_choice_scores \
  --source_choice_selector_sources default,default_rank_blend_contrastive010 \
  --source_choice_selector_hidden_dim 288 \
  --skip_missing_superpoints --dataset scanrefer --test_dataset scanrefer \
  --data_root "${DATA_ROOT}" --batch_size 12 --num_workers 4 \
  --dataloader_prefetch_factor 1 --print_freq 100 \
  --checkpoint_path "${best_checkpoint}" \
  --rec_reranker_checkpoint "${PARENT_ARTIFACT}" \
  --rec_geometry_reranker_checkpoint "${GEOMETRY_ARTIFACT}" \
  --eval_use_rec_reranker_scores --eval_use_rec_geometry_reranker_scores \
  --expected_eval_sample_count "${EXPECTED_SAMPLE_COUNT}" \
  --log_dir "${EVAL_ROOT}" --exp single_stage_parent_geometry_official --eval \
  >"${POST_ROOT}/official_eval.log" 2>&1

"${PYTHON_BIN}" - "${EVAL_ROOT}" "${EXPECTED_SAMPLE_COUNT}" \
  "${POST_ROOT}/official_eval_subgroup_audit.json" <<'PY'
import glob
import json
import os
import sys

from scripts.audit_source_moe_candidate_oracle import metrics_from_receipt

eval_root, expected_count, output_path = sys.argv[1:]
expected_count = int(expected_count)
paths = sorted(
    glob.glob(os.path.join(eval_root, "eval_metrics_epoch_*.json")),
    key=os.path.getmtime,
)
if not paths:
    raise SystemExit("official evaluation produced no metrics receipt")
receipt_path = paths[-1]
with open(receipt_path, "r", encoding="utf-8") as handle:
    receipt = json.load(handle)
metrics = metrics_from_receipt(
    receipt,
    expected_sample_count=expected_count,
    require_position_subgroups=True,
)
result = {
    "schema": "mcln-official-eval-subgroup-audit-v1",
    "receipt": receipt_path,
    "passed": True,
    "metrics": metrics,
}
temporary = output_path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output_path)
print(json.dumps(result, indent=2, sort_keys=True))
PY

echo "[$(timestamp)] single-stage parent+geometry pipeline complete"

# The joint branch is train-only gated.  It is deliberately appended after the
# protected parent+geometry receipt so a rejected mask policy leaves a complete
# baseline result and does not break the downstream architecture queue.
JOINT_TRAIN="${POST_ROOT}/joint_mask_train"
JOINT_ARTIFACT="${POST_ROOT}/joint_box_mask_adapter.pth"
JOINT_EVAL_ROOT="${POST_ROOT}/joint_official_eval"
JOINT_TRAIN_LOG="${POST_ROOT}/joint_box_mask_train.log"

echo "[$(timestamp)] extracting complete train-only joint Mask policy cache"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" \
  scripts/cache_scanrefer_joint_box_mask.py \
  --split train --data-root "${DATA_ROOT}" \
  --checkpoint "${best_checkpoint}" \
  --base-cache "${BASE_TRAIN}" --geometry-cache "${GEOMETRY_TRAIN}" \
  --output-dir "${JOINT_TRAIN}" --batch-size 24 --num-workers 4 \
  --shard-size 256 --device cuda:0 \
  >"${POST_ROOT}/joint_mask_cache.log" 2>&1

echo "[$(timestamp)] training query-consistent joint Box/Mask adapter"
joint_selected=0
set +e
CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" \
  scripts/train_scanrefer_joint_box_mask.py \
  --base-cache "${BASE_TRAIN}" --geometry-cache "${GEOMETRY_TRAIN}" \
  --joint-cache "${JOINT_TRAIN}" \
  --parent-checkpoint "${PARENT_ARTIFACT}" \
  --geometry-checkpoint "${GEOMETRY_ARTIFACT}" \
  --output "${JOINT_ARTIFACT}" --device cuda:0 \
  --epochs 30 --train-batch-size 256 --runtime-batch-size 512 \
  --hidden-dim 128 --dropout 0.1 --lr 1e-3 --weight-decay 1e-4 \
  >"${JOINT_TRAIN_LOG}" 2>&1
joint_status=$?
set -e
if (( joint_status == 0 )); then
  joint_selected=1
  chmod 0444 "${JOINT_ARTIFACT}" "${JOINT_ARTIFACT}.receipt.json"
elif (( joint_status == 2 )) \
    && "${PYTHON_BIN}" - "${JOINT_ARTIFACT}.receipt.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    receipt = json.load(handle)
if receipt.get("selection") != "baseline" or receipt.get("deployable") is not False:
    raise SystemExit(1)
PY
then
  echo "[$(timestamp)] joint Mask train gate retained parent+geometry baseline"
else
  echo "joint Box/Mask training failed without a valid baseline receipt" >&2
  exit 8
fi

if (( joint_selected == 1 )); then
  echo "[$(timestamp)] running official single-stage joint Box/Mask evaluation"
  mkdir -p "${JOINT_EVAL_ROOT}"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node 1 --master_port "$((MASTER_PORT + 1))" \
    train_dist_mod.py --num_decoder_layers 6 --num_target 256 \
    --model MCLN --use_color --self_attend --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --use_source_choice_selector --eval_use_selector_choice_scores \
    --source_choice_selector_sources default,default_rank_blend_contrastive010 \
    --source_choice_selector_hidden_dim 288 \
    --skip_missing_superpoints --dataset scanrefer --test_dataset scanrefer \
    --data_root "${DATA_ROOT}" --batch_size 12 --num_workers 4 \
    --dataloader_prefetch_factor 1 --print_freq 100 \
    --checkpoint_path "${best_checkpoint}" \
    --rec_reranker_checkpoint "${PARENT_ARTIFACT}" \
    --rec_geometry_reranker_checkpoint "${GEOMETRY_ARTIFACT}" \
    --rec_joint_box_mask_checkpoint "${JOINT_ARTIFACT}" \
    --eval_use_rec_reranker_scores --eval_use_rec_geometry_reranker_scores \
    --eval_use_rec_joint_box_mask \
    --expected_eval_sample_count "${EXPECTED_SAMPLE_COUNT}" \
    --log_dir "${JOINT_EVAL_ROOT}" \
    --exp single_stage_joint_box_mask_official --eval \
    >"${POST_ROOT}/joint_official_eval.log" 2>&1

  "${PYTHON_BIN}" - "${JOINT_EVAL_ROOT}" "${EXPECTED_SAMPLE_COUNT}" \
    "${POST_ROOT}/joint_official_eval_subgroup_audit.json" <<'PY'
import glob
import json
import os
import sys

from scripts.audit_source_moe_candidate_oracle import metrics_from_receipt

eval_root, expected_count, output_path = sys.argv[1:]
paths = sorted(
    glob.glob(os.path.join(eval_root, "eval_metrics_epoch_*.json")),
    key=os.path.getmtime,
)
if not paths:
    raise SystemExit("joint official evaluation produced no metrics receipt")
with open(paths[-1], "r", encoding="utf-8") as handle:
    receipt = json.load(handle)
metrics = metrics_from_receipt(
    receipt,
    expected_sample_count=int(expected_count),
    require_position_subgroups=True,
)
result = {
    "schema": "mcln-joint-box-mask-official-subgroup-audit-v1",
    "receipt": paths[-1],
    "passed": True,
    "inference_uses_ground_truth": False,
    "metrics": metrics,
}
temporary = output_path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output_path)
print(json.dumps(result, indent=2, sort_keys=True))
PY
fi

echo "[$(timestamp)] single-stage joint Box/Mask branch complete selected=${joint_selected}"
