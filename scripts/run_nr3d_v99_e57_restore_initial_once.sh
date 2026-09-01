#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly SOURCE_CHECKPOINT="${DATA_ROOT%/}/gf_detector_l6o256.pth"
readonly SOURCE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly DATASET="nr3d"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v99_baseline_gt/nr3d"
readonly E57_CHECKPOINT="${OUTPUT_ROOT}/control/official_rec_monitor/official_best_rec025_epoch_57_0p56500823.pth"
readonly E57_SHA256="fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
readonly E57_EPOCH=57
readonly CLAIM_ROOT="${OUTPUT_ROOT}/control/e57_restore_initial_once_claim"
readonly EXP="nr3d_mcln_joint_butdcls_v99_e57_restore_initial_once_e58_e62_b16a1"
readonly BATCH_SIZE=16
readonly MAX_EPOCH=62
readonly EXPECTED_EVAL_SAMPLE_COUNT=7899
readonly MASTER_PORT=5317
readonly MIN_FREE_GB=7
readonly BACKBONE_JOINT_TRAINING=1
readonly INFERENCE_USES_GROUND_TRUTH=1
readonly USE_BACKBONE_INITIALIZATION=1
readonly TASK_CHECKPOINT_TRANSFER=0
readonly BACKBONE_AUGMENT_DET=0

readonly REQUIRED_PIPELINE_SHA256="264eabacb8c034ad51f4fc30ce33ef990408a19e68400069a74c575f58da31a9"
readonly REQUIRED_TRAIN_ENTRY_SHA256="8f78cca50174423d0c4ab0b3c76a1fa6f22bbd1b179bd547013243ad199996f1"
readonly REQUIRED_MCLN_SHA256="3b9a2b88e9fa36c6f94b1595ae3812818ca5fdeb4b1db64c68f467912c04c9b1"
readonly REQUIRED_SELECTOR_SHA256="61211cea91de4c3f3c11e44bc5ff035711307f3381d5a4069d4ab7842bee17dc"
readonly REQUIRED_TRAINING_GROUPS_SHA256="0298531a3adefd2f010cccc65a0724cf9f0521374446cfe7a9081dfacdd437ce"
readonly REQUIRED_LOSSES_SHA256="cb0ba618ea5a126eb41503691a0c2853aceb3a803bd3fae557178b0e81a29816"
readonly REQUIRED_DATASET_SHA256="800bac2caf9b7a319bdc200f60386000e4e374a559d9581113a8eb57d525f9f0"
readonly REQUIRED_EVALUATOR_SHA256="0173b31a7a818f872c210b01a4e5d17601c4e5f10ec8d97f78c7e537fa44e062"

CHECKPOINT_RETENTION_METRICS=(rec_acc025)
DATASET_LR_ARGS=(
  --lr_backbone 1e-3
  --lr 1e-4
  --lr_decay_epochs 150
  --warmup-epoch -1
)
BACKBONE_EXTRA_ARGS=(
  --print_freq 20
  --joint_det
  --butd_cls
  --restore_e57_lr_to_initial
  --e57_lr_restore_claim "${CLAIM_ROOT}/claim.json"
)

export BACKBONE_RESUME_CHECKPOINT="${E57_CHECKPOINT}"
export BACKBONE_RESUME_SHA256="${E57_SHA256}"
export BACKBONE_RESUME_EPOCH="${E57_EPOCH}"
export VALIDATE_BACKBONE_RESUME=0
export MODE="${MODE:-preflight}"
case "${MODE}" in
  preflight|backbone) ;;
  *) echo "MODE must be preflight or backbone" >&2; exit 2 ;;
esac

: "${REVIEWED_LAUNCHER_SHA256:?set the independently reviewed launcher SHA-256}"
: "${REVIEWED_MAIN_UTILS_SHA256:?set the independently reviewed main_utils.py SHA-256}"
if [[ ! "${REVIEWED_LAUNCHER_SHA256}" =~ ^[0-9a-f]{64}$
      || ! "${REVIEWED_MAIN_UTILS_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "reviewed SHA-256 values must be lowercase 64-hex digests" >&2
  exit 2
fi

require_sha256() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || {
    echo "missing ${label}: ${path}" >&2
    exit 3
  }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA-256 changed: ${actual} != ${expected}" >&2
    exit 3
  }
}

require_sha256 "${BASH_SOURCE[0]}" \
  "${REVIEWED_LAUNCHER_SHA256}" "reviewed E57 LR-restore launcher"
require_sha256 "${ROOT_DIR}/main_utils.py" \
  "${REVIEWED_MAIN_UTILS_SHA256}" "E57 LR-restore implementation"
require_sha256 "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh" \
  "${REQUIRED_PIPELINE_SHA256}" "shared V99 launcher"
require_sha256 "${ROOT_DIR}/train_dist_mod.py" \
  "${REQUIRED_TRAIN_ENTRY_SHA256}" "training entry point"
require_sha256 "${ROOT_DIR}/models/mcln.py" \
  "${REQUIRED_MCLN_SHA256}" "MCLN network"
require_sha256 "${ROOT_DIR}/models/source_choice_selector.py" \
  "${REQUIRED_SELECTOR_SHA256}" "V99 selector"
require_sha256 "${ROOT_DIR}/models/mcln_training_groups.py" \
  "${REQUIRED_TRAINING_GROUPS_SHA256}" "optimizer grouping"
require_sha256 "${ROOT_DIR}/models/losses.py" \
  "${REQUIRED_LOSSES_SHA256}" "training losses"
require_sha256 "${ROOT_DIR}/src/joint_det_dataset.py" \
  "${REQUIRED_DATASET_SHA256}" "joint dataset"
require_sha256 "${ROOT_DIR}/src/grounding_evaluator.py" \
  "${REQUIRED_EVALUATOR_SHA256}" "grounding evaluator"
require_sha256 "${SOURCE_CHECKPOINT}" "${SOURCE_SHA256}" \
  "GroupFree initialization checkpoint"
require_sha256 "${E57_CHECKPOINT}" "${E57_SHA256}" \
  "protected Nr3D E57 checkpoint"

[[ -d "$(dirname "${CLAIM_ROOT}")" ]] || {
  echo "claim parent is missing: $(dirname "${CLAIM_ROOT}")" >&2
  exit 3
}
if [[ -e "${CLAIM_ROOT}" || -L "${CLAIM_ROOT}" ]]; then
  echo "the one-shot E57 LR-restore attempt is already consumed: ${CLAIM_ROOT}" >&2
  exit 3
fi

"${PYTHON_BIN}" - "${E57_CHECKPOINT}" <<'PY'
import math
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
config = vars(checkpoint["config"])
optimizer = checkpoint["optimizer"]
scheduler = checkpoint["scheduler"]
expected_current = [1e-5, 1e-4, 1e-5, 1.25e-5]
expected_initial = [1e-4, 1e-3, 1e-4, 1.25e-4]
current = [float(group["lr"]) for group in optimizer["param_groups"]]
initial = [float(group["initial_lr"]) for group in optimizer["param_groups"]]
if checkpoint.get("epoch") != 57:
    raise SystemExit("protected checkpoint is not E57")
if len(optimizer["param_groups"]) != 4 or len(optimizer["state"]) != 716:
    raise SystemExit("protected E57 optimizer topology mismatch")
if current != expected_current or initial != expected_initial:
    raise SystemExit("protected E57 learning-rate provenance mismatch")
if [float(value) for value in scheduler["base_lrs"]] != expected_initial:
    raise SystemExit("protected E57 scheduler base-LR mismatch")
if [float(value) for value in scheduler["_last_lr"]] != expected_current:
    raise SystemExit("protected E57 scheduler current-LR mismatch")
if int(scheduler["last_epoch"]) != 159942:
    raise SystemExit("protected E57 scheduler progress mismatch")
if int(scheduler["_step_count"]) != 159943:
    raise SystemExit("protected E57 scheduler step-count mismatch")
if dict(scheduler["milestones"]) != {423706: 1}:
    raise SystemExit("protected E57 scheduler milestone mismatch")
required = {
    "dataset": ["nr3d"],
    "test_dataset": "nr3d",
    "batch_size": 16,
    "lr_scheduler": "step",
    "warmup_epoch": -1,
    "lr_decay_epochs": [150],
    "lr": 1e-4,
    "lr_backbone": 1e-3,
    "text_encoder_lr": 1e-5,
    "weight_decay": 5e-4,
    "augment_det": False,
    "joint_det": True,
    "butd_cls": True,
    "butd_gt": False,
    "use_color": True,
    "num_decoder_layers": 6,
    "num_target": 256,
    "model": "MCLN",
    "mask_head_lr_multiplier": 1.0,
    "use_source_choice_selector": True,
    "eval_use_selector_choice_scores": True,
    "source_choice_selector_sources": (
        "default,default_rank_blend_contrastive010"
    ),
    "source_choice_selector_default_source": "default",
    "source_choice_selector_hidden_dim": 288,
    "source_choice_selector_lr": 1.25e-4,
    "source_choice_selector_loss_weight": 0.5,
    "source_choice_selector_choice_target": (
        "precision_gain_default_sourcewise_focal_bce"
    ),
    "source_choice_selector_min_iou_gap": 0.03,
}
for name, expected in required.items():
    observed = config.get(name)
    if type(observed) is not type(expected) or observed != expected:
        raise SystemExit(
            "protected E57 config mismatch for {}: {!r} != {!r}".format(
                name, observed, expected
            )
        )
if config.get("gradient_accumulation_steps", 1) != 1:
    raise SystemExit("protected E57 accumulation mismatch")
if config.get("drop_incomplete_accumulation_group", False):
    raise SystemExit("protected E57 unexpectedly drops accumulation tail")
print(
    "protected_e57_proof=epoch57 groups4 states716 "
    "scheduler159942/159943 milestone423706 current_lrs={} initial_lrs={}"
    .format(current, initial)
)
PY

free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
gpu_used="$(nvidia-smi --query-gpu=memory.used \
  --format=csv,noheader,nounits -i 0 | tr -d ' ')"
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
exec 8>"${lock_file}"
if ! flock -n 8; then
  echo "another V99 job owns ${lock_file}" >&2
  exit 6
fi
flock -u 8
exec 8>&-

readonly LAUNCHER_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
echo "mode=${MODE} launcher_sha256=${LAUNCHER_SHA256}"
echo "gpu0_memory_used_mib=${gpu_used} free_disk_gib=${free_gb}"
echo "one_shot_claim=${CLAIM_ROOT} status=available"

start_backbone_guard() {
  local backbone_run_dir="$1"
  "${PYTHON_BIN}" - \
    "${CLAIM_ROOT}" \
    "${backbone_run_dir}" \
    "${BASH_SOURCE[0]}" \
    "${LAUNCHER_SHA256}" \
    "${REVIEWED_MAIN_UTILS_SHA256}" \
    "${REQUIRED_PIPELINE_SHA256}" \
    "${E57_CHECKPOINT}" \
    "${E57_SHA256}" \
    "${EXP}" <<'PY'
import json
import os
import sys
import time

(
    claim_root,
    run_dir,
    launcher,
    launcher_sha256,
    main_utils_sha256,
    pipeline_sha256,
    checkpoint,
    checkpoint_sha256,
    experiment,
) = sys.argv[1:]
parent = os.path.dirname(claim_root)
if not os.path.isdir(parent):
    raise SystemExit("one-shot claim parent disappeared")
if os.path.lexists(claim_root):
    raise SystemExit("one-shot E57 LR-restore attempt was already consumed")
os.mkdir(claim_root, 0o700)
parent_fd = os.open(parent, os.O_RDONLY)
try:
    os.fsync(parent_fd)
finally:
    os.close(parent_fd)
payload = {
    "schema": "mcln-e57-lr-restore-one-shot-claim-v1",
    "claimed_unix_time": time.time(),
    "pid": os.getpid(),
    "experiment": experiment,
    "backbone_run_dir": os.path.realpath(run_dir),
    "launcher": os.path.realpath(launcher),
    "launcher_sha256": launcher_sha256,
    "main_utils_sha256": main_utils_sha256,
    "pipeline_sha256": pipeline_sha256,
    "checkpoint": os.path.realpath(checkpoint),
    "checkpoint_sha256": checkpoint_sha256,
    "one_shot_consumed": True,
    "training_contract": {
        "augment_det": False,
        "batch_size": 16,
        "butd": False,
        "butd_cls": True,
        "butd_gt": False,
        "checkpoint_start_epoch": None,
        "checkpoint_metric_retention": True,
        "checkpoint_retention_metrics": ["rec_acc025"],
        "data_root": "/root/autodl-tmp/DATA_ROOT/",
        "dataloader_prefetch_factor": 2,
        "dataset": ["nr3d"],
        "detect_intermediate": True,
        "drop_incomplete_accumulation_group": False,
        "eval": False,
        "eval_use_selector_choice_scores": True,
        "expected_eval_sample_count": 7899,
        "frozen": False,
        "gradient_accumulation_steps": 1,
        "joint_det": True,
        "lr": 0.0001,
        "lr_backbone": 0.001,
        "lr_decay_epochs": [150],
        "lr_decay_rate": 0.1,
        "lr_scheduler": "step",
        "mask_head_lr_multiplier": 1.0,
        "max_epoch": 62,
        "max_train_batches": 0,
        "model": "MCLN",
        "model_only_initialization": False,
        "num_decoder_layers": 6,
        "num_target": 256,
        "num_workers": 4,
        "optimizer": "adamW",
        "persistent_train_workers": True,
        "pp_checkpoint": "/root/autodl-tmp/DATA_ROOT/gf_detector_l6o256.pth",
        "print_freq": 20,
        "reduce_lr": False,
        "restore_e57_lr_to_initial": True,
        "resume_lr_scale": 1.0,
        "save_freq": 1,
        "self_attend": True,
        "skip_missing_superpoints": True,
        "small_lr": False,
        "source_choice_selector_choice_target": "precision_gain_default_sourcewise_focal_bce",
        "source_choice_selector_default_source": "default",
        "source_choice_selector_hidden_dim": 288,
        "source_choice_selector_loss_weight": 0.5,
        "source_choice_selector_lr": 0.000125,
        "source_choice_selector_min_iou_gap": 0.03,
        "source_choice_selector_sources": "default,default_rank_blend_contrastive010",
        "start_epoch": 1,
        "test_dataset": "nr3d",
        "text_encoder_lr": 0.00001,
        "use_color": True,
        "use_contrastive_align": True,
        "use_height": False,
        "use_multiview": False,
        "use_soft_token_loss": True,
        "use_source_choice_selector": True,
        "val_freq": 1,
        "warmup_epoch": -1,
        "weight_decay": 0.0005,
        "wo_obj_name": "None",
    },
}
temporary = os.path.join(claim_root, "claim.json.tmp.{}".format(os.getpid()))
claim_path = os.path.join(claim_root, "claim.json")
with open(temporary, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(temporary, 0o444)
os.replace(temporary, claim_path)
claim_fd = os.open(claim_root, os.O_RDONLY)
try:
    os.fsync(claim_fd)
finally:
    os.close(claim_fd)
print("one_shot_claim_committed={}".format(claim_path))
PY
}

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
