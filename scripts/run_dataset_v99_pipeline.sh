#!/usr/bin/env bash
# Shared implementation sourced by the three public dataset launchers.

set -euo pipefail

: "${ROOT_DIR:?wrapper must define ROOT_DIR}"
: "${DATA_ROOT:?wrapper must define DATA_ROOT}"
: "${PYTHON_BIN:?wrapper must define PYTHON_BIN}"
: "${SOURCE_CHECKPOINT:?wrapper must define SOURCE_CHECKPOINT}"
: "${SOURCE_SHA256:?wrapper must define SOURCE_SHA256}"
: "${DATASET:?wrapper must define DATASET}"
: "${OUTPUT_ROOT:?wrapper must define OUTPUT_ROOT}"
: "${EXP:?wrapper must define EXP}"
: "${BATCH_SIZE:?wrapper must define BATCH_SIZE}"
: "${MAX_EPOCH:?wrapper must define MAX_EPOCH}"
: "${EXPECTED_EVAL_SAMPLE_COUNT:?wrapper must define EXPECTED_EVAL_SAMPLE_COUNT}"
: "${MASTER_PORT:?wrapper must define MASTER_PORT}"
: "${MIN_FREE_GB:?wrapper must define MIN_FREE_GB}"
if [[ -z "${BACKBONE_JOINT_TRAINING+x}" ]]; then
  BACKBONE_JOINT_TRAINING=0
fi
if [[ -z "${INFERENCE_USES_GROUND_TRUTH+x}" ]]; then
  INFERENCE_USES_GROUND_TRUTH=0
fi
if [[ -z "${USE_BACKBONE_INITIALIZATION+x}" ]]; then
  USE_BACKBONE_INITIALIZATION=1
fi
if [[ -z "${TASK_CHECKPOINT_TRANSFER+x}" ]]; then
  TASK_CHECKPOINT_TRANSFER=1
fi
if [[ -z "${BACKBONE_AUGMENT_DET+x}" ]]; then
  BACKBONE_AUGMENT_DET=1
fi
if ! declare -p DATASET_LR_ARGS >/dev/null 2>&1; then
  echo "wrapper must define DATASET_LR_ARGS array" >&2
  exit 2
fi
if ! declare -p BACKBONE_EXTRA_ARGS >/dev/null 2>&1; then
  BACKBONE_EXTRA_ARGS=()
fi
if ! declare -p CHECKPOINT_RETENTION_METRICS >/dev/null 2>&1; then
  CHECKPOINT_RETENTION_METRICS=(
    rec_acc025 rec_acc050 mask_acc025 mask_acc050 mask_miou
  )
fi
if ((${#CHECKPOINT_RETENTION_METRICS[@]} == 0)); then
  echo "CHECKPOINT_RETENTION_METRICS must not be empty" >&2
  exit 2
fi
for metric_name in "${CHECKPOINT_RETENTION_METRICS[@]}"; do
  case "${metric_name}" in
    rec_acc025|rec_acc050|mask_acc025|mask_acc050|mask_miou) ;;
    *) echo "unsupported checkpoint retention metric: ${metric_name}" >&2; exit 2 ;;
  esac
done

cd "${ROOT_DIR}"
MODE="${MODE:-all}"
CHECKPOINT="${CHECKPOINT:-}"
BACKBONE_RESUME_CHECKPOINT="${BACKBONE_RESUME_CHECKPOINT:-}"
BACKBONE_RESUME_SHA256="${BACKBONE_RESUME_SHA256:-}"
BACKBONE_RESUME_EPOCH="${BACKBONE_RESUME_EPOCH:-}"
VALIDATE_BACKBONE_RESUME="${VALIDATE_BACKBONE_RESUME:-1}"
CLEAN_RECONSTRUCTIBLE_CACHES="${CLEAN_RECONSTRUCTIBLE_CACHES:-1}"
PRUNE_NONBEST_BACKBONE_WEIGHTS="${PRUNE_NONBEST_BACKBONE_WEIGHTS:-1}"
readonly V97_SOURCE="${ROOT_DIR}/experiment_output/historical_e71_geometry/v97_contextual_listwise_hierarchical_trainonly_v1.json"
readonly V97_SOURCE_SHA256="ca04b4cbd1804b92d676d815b79bfcacdaab3e8745742177bd94283cedda7f8d"

if (($# != 0)); then
  echo "usage: MODE=preflight|backbone|v99|all [CHECKPOINT=/path] $0" >&2
  exit 2
fi
case "${MODE}" in
  preflight|backbone|v99|all) ;;
  *) echo "MODE must be preflight, backbone, v99, or all" >&2; exit 2 ;;
esac
case "${CLEAN_RECONSTRUCTIBLE_CACHES}:${PRUNE_NONBEST_BACKBONE_WEIGHTS}" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "cleanup switches must be 0 or 1" >&2; exit 2 ;;
esac
case "${VALIDATE_BACKBONE_RESUME}" in
  0|1) ;;
  *) echo "VALIDATE_BACKBONE_RESUME must be 0 or 1" >&2; exit 2 ;;
esac
if [[ -n "${BACKBONE_RESUME_CHECKPOINT}" ]]; then
  [[ -n "${BACKBONE_RESUME_SHA256}" ]] || {
    echo "BACKBONE_RESUME_SHA256 is required for an exact resume" >&2
    exit 2
  }
  [[ "${BACKBONE_RESUME_EPOCH}" =~ ^[1-9][0-9]*$ ]] || {
    echo "BACKBONE_RESUME_EPOCH must be a positive integer" >&2
    exit 2
  }
fi
case "${BACKBONE_JOINT_TRAINING}:${INFERENCE_USES_GROUND_TRUTH}:${USE_BACKBONE_INITIALIZATION}:${TASK_CHECKPOINT_TRANSFER}:${BACKBONE_AUGMENT_DET}" in
  [01]:[01]:[01]:[01]:[01]) ;;
  *) echo "training/provenance switches must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "${TASK_CHECKPOINT_TRANSFER}" == "1"
      && "${USE_BACKBONE_INITIALIZATION}" != "1" ]]; then
  echo "task checkpoint transfer requires checkpoint initialization" >&2
  exit 2
fi
profile_args=" ${BACKBONE_EXTRA_ARGS[*]} "
if [[ "${BACKBONE_JOINT_TRAINING}" == "1" ]]; then
  [[ "${profile_args}" == *" --joint_det "* ]] || {
    echo "joint-training contract requires --joint_det" >&2; exit 2;
  }
else
  [[ "${profile_args}" != *" --joint_det "* ]] || {
    echo "dataset-only contract rejected --joint_det" >&2; exit 2;
  }
fi
if [[ "${INFERENCE_USES_GROUND_TRUTH}" == "1" ]]; then
  [[ "${profile_args}" == *" --butd_cls "* ]] || {
    echo "ground-truth proposal contract requires --butd_cls" >&2; exit 2;
  }
  [[ "${profile_args}" != *" --butd_gt "* ]] || {
    echo "baseline-compatible contract rejects perfect GT classes" >&2; exit 2;
  }
else
  [[ "${profile_args}" != *" --butd_cls "* ]] || {
    echo "non-GT inference contract rejected --butd_cls" >&2; exit 2;
  }
fi

mkdir -p "${OUTPUT_ROOT}/launch"
timestamp="$(date '+%Y%m%d_%H%M%S')"
launch_log="${OUTPUT_ROOT}/launch/${EXP}_${MODE}_${timestamp}.log"
exec > >(tee -a "${launch_log}") 2>&1

require_sha256() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; exit 3; }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA-256 changed: ${actual}" >&2
    exit 3
  }
}

require_sha256 "${SOURCE_CHECKPOINT}" "${SOURCE_SHA256}" "initialization checkpoint"
if [[ -n "${BACKBONE_RESUME_CHECKPOINT}" ]]; then
  require_sha256 \
    "${BACKBONE_RESUME_CHECKPOINT}" \
    "${BACKBONE_RESUME_SHA256}" \
    "backbone resume checkpoint"
fi
require_sha256 "${V97_SOURCE}" "${V97_SOURCE_SHA256}" "V97 method-lineage source"
free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
if [[ "${BACKBONE_JOINT_TRAINING}" == "1" ]]; then
  dataset_only="false"
else
  dataset_only="true"
fi
if [[ -n "${BACKBONE_RESUME_CHECKPOINT}" ]]; then
  initialization="exact_training_resume"
elif [[ "${USE_BACKBONE_INITIALIZATION}" == "0" ]]; then
  initialization="fully_random"
elif [[ "${TASK_CHECKPOINT_TRANSFER}" == "1" ]]; then
  initialization="checkpoint_transfer"
else
  initialization="groupfree_detector_pretraining_no_task_transfer"
fi
echo "dataset=${DATASET} dataset_only=${dataset_only} joint_training=$([[ "${BACKBONE_JOINT_TRAINING}" == "1" ]] && echo true || echo false)"
echo "inference_uses_ground_truth=$([[ "${INFERENCE_USES_GROUND_TRUTH}" == "1" ]] && echo true || echo false)"
echo "initialization=${initialization}"
echo "v99_contract=16_queries_x_7_variants_parent_geometry_contextual_pareto"
echo "checkpoint_retention_metrics=$(IFS=,; echo "${CHECKPOINT_RETENTION_METRICS[*]}")"
echo "source_checkpoint=${SOURCE_CHECKPOINT}"
echo "source_sha256=${SOURCE_SHA256}"
if [[ -n "${BACKBONE_RESUME_CHECKPOINT}" ]]; then
  echo "backbone_resume_checkpoint=${BACKBONE_RESUME_CHECKPOINT}"
  echo "backbone_resume_sha256=${BACKBONE_RESUME_SHA256}"
  echo "backbone_resume_epoch=${BACKBONE_RESUME_EPOCH}"
fi
echo "gpu0_memory_used_mib=${gpu_used} free_disk_gib=${free_gb}"
echo "launch_log=${launch_log}"
if [[ "${MODE}" == "preflight" ]]; then
  exit 0
fi
if ((gpu_used >= 500)); then
  echo "GPU0 is busy (${gpu_used} MiB)" >&2
  exit 4
fi
if ((free_gb < MIN_FREE_GB)); then
  echo "need at least ${MIN_FREE_GB} GiB free under DATA_ROOT" >&2
  exit 5
fi

lock_file="${DATA_ROOT%/}/output/network_v99/single_gpu.lock"
mkdir -p "$(dirname "${lock_file}")"
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "another V99 job owns ${lock_file}" >&2
  exit 6
fi

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/pointnet2:${PYTHONPATH:-}"

DISCOVERED_CHECKPOINT=""
BACKBONE_RUN_DIR=""
run_backbone() {
  BACKBONE_RUN_DIR="${OUTPUT_ROOT}/backbone/${EXP}_${timestamp}"
  mkdir -p "${BACKBONE_RUN_DIR}"
  if declare -F start_backbone_guard >/dev/null 2>&1; then
    start_backbone_guard "${BACKBONE_RUN_DIR}"
  fi
  local -a initialization_args=()
  local -a augmentation_args=()
  if [[ -n "${BACKBONE_RESUME_CHECKPOINT}" ]]; then
    initialization_args=(
      --checkpoint_path "${BACKBONE_RESUME_CHECKPOINT}"
    )
  elif [[ "${USE_BACKBONE_INITIALIZATION}" == "1" ]]; then
    initialization_args=(
      --checkpoint_path "${SOURCE_CHECKPOINT}"
      --checkpoint_start_epoch 1
    )
    if [[ "${TASK_CHECKPOINT_TRANSFER}" == "1" ]]; then
      initialization_args+=(--reduce_lr)
    else
      initialization_args+=(--model_only_initialization)
    fi
  fi
  if [[ "${BACKBONE_AUGMENT_DET}" == "1" ]]; then
    augmentation_args=(--augment_det)
  fi
  local -a train_args=(
    train_dist_mod.py
    --num_decoder_layers 6 --num_target 256
    --use_color --weight_decay 0.0005
    --data_root "${DATA_ROOT}"
    --val_freq 1 --batch_size "${BATCH_SIZE}"
    --num_workers 4 --dataloader_prefetch_factor 2 --persistent_train_workers
    --save_freq 1 --print_freq 500
    "${DATASET_LR_ARGS[@]}"
    --dataset "${DATASET}" --test_dataset "${DATASET}"
    "${BACKBONE_EXTRA_ARGS[@]}"
    --detect_intermediate
    --use_soft_token_loss --use_contrastive_align
    --log_dir "${BACKBONE_RUN_DIR}"
    --pp_checkpoint "${DATA_ROOT%/}/gf_detector_l6o256.pth"
    --self_attend --skip_missing_superpoints
    "${augmentation_args[@]}"
    "${initialization_args[@]}"
    --start_epoch 1 --max_epoch "${MAX_EPOCH}"
    --model MCLN --exp "${EXP}"
    --use_source_choice_selector --eval_use_selector_choice_scores
    --source_choice_selector_sources default,default_rank_blend_contrastive010
    --source_choice_selector_default_source default
    --source_choice_selector_hidden_dim 288
    --source_choice_selector_lr 1.25e-4
    --source_choice_selector_loss_weight 0.5
    --source_choice_selector_choice_target precision_gain_default_sourcewise_focal_bce
    --source_choice_selector_min_iou_gap 0.03
    --expected_eval_sample_count "${EXPECTED_EVAL_SAMPLE_COUNT}"
    --checkpoint_metric_retention
    --checkpoint_retention_metrics "${CHECKPOINT_RETENTION_METRICS[@]}"
  )
  local joined=" ${train_args[*]} "
  if [[ "${BACKBONE_JOINT_TRAINING}" == "1" ]]; then
    [[ "${joined}" == *" --joint_det "* ]] || {
      echo "joint-training contract requires --joint_det" >&2
      exit 7
    }
  else
    [[ "${joined}" != *" --joint_det "* ]] || {
      echo "dataset-only contract rejected --joint_det" >&2
      exit 7
    }
  fi
  if [[ "${INFERENCE_USES_GROUND_TRUTH}" == "1" ]]; then
    [[ "${joined}" == *" --butd_cls "* ]] || {
      echo "ground-truth proposal contract requires --butd_cls" >&2
      exit 7
    }
    [[ "${joined}" != *" --butd_gt "* ]] || {
      echo "baseline-compatible contract rejects perfect GT classes" >&2
      exit 7
    }
  fi
  if [[ -n "${BACKBONE_RESUME_CHECKPOINT}"
        && "${VALIDATE_BACKBONE_RESUME}" == "1" ]]; then
    local -a eval_args=(
      "${train_args[@]}"
      --eval --checkpoint_start_epoch "${BACKBONE_RESUME_EPOCH}"
    )
    "${PYTHON_BIN}" -m torch.distributed.launch \
      --nproc_per_node 1 --master_port "${MASTER_PORT}" \
      "${eval_args[@]}"
  fi
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node 1 --master_port "${MASTER_PORT}" \
    "${train_args[@]}"
  mapfile -t candidates < <(
    find "${BACKBONE_RUN_DIR}" -type f -name ckpt_best_rec_acc025.pth -print
  )
  if ((${#candidates[@]} != 1)); then
    echo "expected exactly one REC@0.25-best checkpoint, found ${#candidates[@]}" >&2
    exit 8
  fi
  DISCOVERED_CHECKPOINT="${candidates[0]}"
  chmod 0444 "${DISCOVERED_CHECKPOINT}"
}

run_if_missing() {
  local output="$1"
  shift
  if [[ -e "${output}" ]]; then
    echo "resume: keeping existing ${output}"
  else
    "$@"
  fi
}

run_v99() {
  local checkpoint="$1" checkpoint_sha v99_root
  [[ -f "${checkpoint}" ]] || {
    echo "dataset backbone checkpoint is missing: ${checkpoint}" >&2
    exit 8
  }
  chmod 0444 "${checkpoint}"
  checkpoint_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
  v99_root="${OUTPUT_ROOT}/v99/${checkpoint_sha}"
  local train_cache="${v99_root}/candidate_cache/train"
  local val_cache="${v99_root}/candidate_cache/val"
  local audit_dir="${v99_root}/geometry_audit"
  local train_geometry="${v99_root}/geometry_cache/train"
  local val_geometry="${v99_root}/geometry_cache/val"
  local panel_preflight="${v99_root}/geometry_panel_preflight.json"
  local parent_artifact="${v99_root}/artifacts/parent_reranker.pth"
  local geometry_artifact="${v99_root}/artifacts/geometry_reranker.pth"
  local oof_result="${v99_root}/artifacts/v99_oof.json"
  local hierarchy_artifact="${v99_root}/artifacts/v99_contextual_pareto.pth"
  local artifact_receipt="${v99_root}/artifacts/v99_contextual_pareto_receipt.json"
  local official_dir="${v99_root}/official"
  local pipeline_receipt="${v99_root}/pipeline_receipt.json"
  mkdir -p "${v99_root}/artifacts" "${official_dir}"
  local -a provenance_args=()
  local -a geometry_input_args=()
  local -a finalizer_args=()
  if [[ "${BACKBONE_JOINT_TRAINING}" == "1" ]]; then
    provenance_args+=(--backbone-joint-training)
  fi
  if [[ "${INFERENCE_USES_GROUND_TRUTH}" == "1" ]]; then
    provenance_args+=(--inference-uses-ground-truth)
    geometry_input_args+=(--allow-butd-cls)
  fi
  finalizer_args=("${provenance_args[@]}")
  if [[ "${TASK_CHECKPOINT_TRANSFER}" == "0" ]]; then
    finalizer_args+=(--no-task-checkpoint-transfer)
  fi

  "${PYTHON_BIN}" scripts/cache_scanrefer_rec_candidates.py \
    --dataset "${DATASET}" --split train --data-root "${DATA_ROOT}" \
    --checkpoint "${checkpoint}" --output-dir "${train_cache}" \
    --batch-size 12 --num-workers 4 --shard-size 252 \
    --max-candidates 16 --device cuda:0
  "${PYTHON_BIN}" scripts/preflight_dataset_v99_geometry_panel.py \
    --dataset "${DATASET}" --checkpoint "${checkpoint}" \
    --train-cache "${train_cache}" --scene-count 64 \
    --expressions-per-scene 4 --selection-seed 0 \
    --output "${panel_preflight}"
  "${PYTHON_BIN}" scripts/cache_scanrefer_rec_candidates.py \
    --dataset "${DATASET}" --split val --data-root "${DATA_ROOT}" \
    --checkpoint "${checkpoint}" --output-dir "${val_cache}" \
    --batch-size 12 --num-workers 4 --shard-size 252 \
    --max-candidates 16 --device cuda:0

  run_if_missing "${parent_artifact}" \
    "${PYTHON_BIN}" scripts/train_rec_reranker.py \
      --train-cache "${train_cache}" --output "${parent_artifact}" \
      --seed 0 --hidden-dim 256 --dropout 0.1 --device cuda:0
  chmod 0444 "${parent_artifact}"

  run_if_missing "${audit_dir}/selection.json" \
    "${PYTHON_BIN}" scripts/audit_scanrefer_mask_geometry.py \
      --dataset "${DATASET}" --data-root "${DATA_ROOT}" \
      --checkpoint "${checkpoint}" --train-cache "${train_cache}" \
      --output-dir "${audit_dir}" --scene-count 64 \
      --expressions-per-scene 4 --batch-size 12 --num-workers 2 \
      --device cuda:0

  "${PYTHON_BIN}" scripts/cache_scanrefer_rec_mask_geometry.py \
    --dataset "${DATASET}" --portable-provenance \
    "${geometry_input_args[@]}" \
    --split train --data-root "${DATA_ROOT}" \
    --checkpoint "${checkpoint}" --base-cache "${train_cache}" \
    --audit-train-cache "${train_cache}" \
    --audit-provenance "${audit_dir}/selection.json" \
    --output-dir "${train_geometry}" --batch-size 12 \
    --num-workers 2 --shard-size 252 --device cuda:0
  "${PYTHON_BIN}" scripts/cache_scanrefer_rec_mask_geometry.py \
    --dataset "${DATASET}" --portable-provenance \
    "${geometry_input_args[@]}" \
    --split val --data-root "${DATA_ROOT}" \
    --checkpoint "${checkpoint}" --base-cache "${val_cache}" \
    --audit-train-cache "${train_cache}" \
    --audit-provenance "${audit_dir}/selection.json" \
    --output-dir "${val_geometry}" --batch-size 12 \
    --num-workers 2 --shard-size 252 --device cuda:0

  run_if_missing "${geometry_artifact}" \
    "${PYTHON_BIN}" scripts/train_rec_geometry_reranker.py \
      --base-cache "${train_cache}" --geometry-cache "${train_geometry}" \
      --parent-artifact "${parent_artifact}" \
      --output "${geometry_artifact}" --split-seed 0 --model-seed 0 \
      --hidden-dim 256 --dropout 0.1 --device cuda:0 --verbose
  chmod 0444 "${geometry_artifact}"

  run_if_missing "${oof_result}" \
    "${PYTHON_BIN}" scripts/run_v99_pareto_contextual_hierarchical.py \
      --portable-dataset-contract --dataset "${DATASET}" \
      "${provenance_args[@]}" \
      --backbone-checkpoint "${checkpoint}" --v97-source "${V97_SOURCE}" \
      --base-cache "${train_cache}" --geometry-cache "${train_geometry}" \
      --parent-artifact "${parent_artifact}" \
      --geometry-artifact "${geometry_artifact}" \
      --output "${oof_result}" --device cuda:0

  if [[ -e "${hierarchy_artifact}" || -e "${artifact_receipt}" ]]; then
    [[ -f "${hierarchy_artifact}" && -f "${artifact_receipt}" ]] || {
      echo "partial V99 artifact publication detected" >&2
      exit 9
    }
  else
    "${PYTHON_BIN}" scripts/build_v99_pareto_contextual_artifact.py \
      --portable-dataset-contract --dataset "${DATASET}" \
      --backbone-checkpoint "${checkpoint}" \
      --v99-result "${oof_result}" \
      --v99-script "${ROOT_DIR}/scripts/run_v99_pareto_contextual_hierarchical.py" \
      --base-cache "${train_cache}" --geometry-cache "${train_geometry}" \
      --parent-artifact "${parent_artifact}" \
      --geometry-artifact "${geometry_artifact}" \
      --artifact-output "${hierarchy_artifact}" \
      --receipt-output "${artifact_receipt}" --device cuda:0
  fi
  chmod 0444 "${hierarchy_artifact}" "${artifact_receipt}"

  "${PYTHON_BIN}" scripts/run_dataset_v99_official.py \
    --dataset "${DATASET}" \
    "${provenance_args[@]}" \
    --expected-sample-count "${EXPECTED_EVAL_SAMPLE_COUNT}" \
    --python-bin "${PYTHON_BIN}" --project-root "${ROOT_DIR}" \
    --data-root "${DATA_ROOT}" --master-port "$((MASTER_PORT + 1))" \
    --experiment "${EXP}_v99_official" \
    --backbone-checkpoint "${checkpoint}" \
    --parent-artifact "${parent_artifact}" \
    --geometry-artifact "${geometry_artifact}" \
    --hierarchical-artifact "${hierarchy_artifact}" \
    --output-dir "${official_dir}"
  local official_result="${official_dir}/official_result.json"
  local eval_receipt
  eval_receipt="$("${PYTHON_BIN}" -c \
    'import json, pathlib, sys; p=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["eval_receipt"]["path"]; print(pathlib.Path(p).resolve())' \
    "${official_result}")"
  [[ -f "${eval_receipt}" ]] || {
    echo "official result points to a missing eval receipt: ${eval_receipt}" >&2
    exit 10
  }

  if [[ ! -e "${pipeline_receipt}" ]]; then
    "${PYTHON_BIN}" scripts/finalize_dataset_v99_pipeline.py \
      --dataset "${DATASET}" \
      "${finalizer_args[@]}" \
      --expected-sample-count "${EXPECTED_EVAL_SAMPLE_COUNT}" \
      --initialization-checkpoint "${SOURCE_CHECKPOINT}" \
      --expected-initialization-sha256 "${SOURCE_SHA256}" \
      --backbone-checkpoint "${checkpoint}" \
      --parent-artifact "${parent_artifact}" \
      --geometry-artifact "${geometry_artifact}" \
      --hierarchical-artifact "${hierarchy_artifact}" \
      --audit-panel-preflight "${panel_preflight}" \
      --oof-result "${oof_result}" \
      --artifact-receipt "${artifact_receipt}" \
      --eval-receipt "${eval_receipt}" \
      --official-result "${official_result}" \
      --output "${pipeline_receipt}"
  fi

  if [[ "${CLEAN_RECONSTRUCTIBLE_CACHES}" == "1" ]]; then
    local cleanup_receipt="${v99_root}/cleanup_receipt.json"
    if [[ ! -e "${cleanup_receipt}" ]]; then
      local -a cleanup_args=(
        --pipeline-root "${v99_root}"
        --pipeline-receipt "${pipeline_receipt}"
        --output "${cleanup_receipt}"
      )
      if [[ "${PRUNE_NONBEST_BACKBONE_WEIGHTS}" == "1"
            && -n "${BACKBONE_RUN_DIR}" ]]; then
        cleanup_args+=(
          --backbone-run-dir "${BACKBONE_RUN_DIR}"
          --keep-checkpoint "${checkpoint}"
        )
      fi
      "${PYTHON_BIN}" scripts/cleanup_dataset_v99_intermediates.py \
        "${cleanup_args[@]}"
    fi
  fi
  echo "pipeline_receipt=${pipeline_receipt}"
}

if [[ "${MODE}" == "backbone" || "${MODE}" == "all" ]]; then
  run_backbone
  CHECKPOINT="${DISCOVERED_CHECKPOINT}"
  echo "selected_backbone=${CHECKPOINT}"
  echo "selected_backbone_sha256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
fi
if [[ "${MODE}" == "v99" || "${MODE}" == "all" ]]; then
  if [[ -z "${CHECKPOINT}" ]]; then
    echo "MODE=${MODE} requires CHECKPOINT=/absolute/path" >&2
    exit 2
  fi
  run_v99 "${CHECKPOINT}"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] completed ${DATASET} ${MODE}"
