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
readonly PARENT_LEAF="${OUTPUT_ROOT}/backbone/nr3d_mcln_joint_butdcls_v99_e57_restore_initial_once_e58_e62_b16a1_20260829_170634/nr3d/nr3d_mcln_joint_butdcls_v99_e57_restore_initial_once_e58_e62_b16a1/1787994403"
readonly REQUIRED_RESUME_CHECKPOINT="${PARENT_LEAF}/ckpt_epoch_59.pth"
readonly REQUIRED_RESUME_SHA256="279c23c8987b4a851c8e1bf6ef5b4296edf8800f55f8827c2dee42e731acf589"
readonly REQUIRED_RESUME_EPOCH=59
readonly E58_RECEIPT="${PARENT_LEAF}/eval_metrics_epoch_58.json"
readonly E58_RECEIPT_SHA256="6bb744a5cd3f3b8e559ba167fbf93e0d809c9d6a758053ddcd7c57b466fa39ed"
readonly E59_RECEIPT="${PARENT_LEAF}/eval_metrics_epoch_59.json"
readonly E59_RECEIPT_SHA256="7b61c922b6c4617defe8793a5ea35a69177a94877c6bdb31eeb64a0b333c32b9"
readonly PROTECTED_E57="${OUTPUT_ROOT}/control/official_rec_monitor/official_best_rec025_epoch_57_0p56500823.pth"
readonly PROTECTED_E57_SHA256="fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
readonly REQUIRED_MAIN_UTILS_SHA256="4742cdbc5fb9a25c88224eaa5096eeb592d468a38b869a7309be743c9cea6808"
readonly REQUIRED_PIPELINE_SHA256="264eabacb8c034ad51f4fc30ce33ef990408a19e68400069a74c575f58da31a9"
readonly EXP="nr3d_mcln_joint_butdcls_v99_e59_lr10_e60_e62_b16a1"
readonly TRAIN_SCREEN_NAME="mcln_nr3d_e59_lr10_resume"
readonly BATCH_SIZE=16
readonly MAX_EPOCH=62
readonly EXPECTED_EVAL_SAMPLE_COUNT=7899
readonly MASTER_PORT=5323
readonly MIN_FREE_GB=7
readonly BACKBONE_JOINT_TRAINING=1
readonly INFERENCE_USES_GROUND_TRUTH=1
readonly USE_BACKBONE_INITIALIZATION=1
readonly TASK_CHECKPOINT_TRANSFER=0
readonly BACKBONE_AUGMENT_DET=0

CHECKPOINT_RETENTION_METRICS=(rec_acc025)
DATASET_LR_ARGS=(
  --lr_backbone 1e-3 --lr 1e-4
  --lr_decay_epochs 150
  --warmup-epoch -1
)
BACKBONE_EXTRA_ARGS=(
  --print_freq 20
  --joint_det
  --butd_cls
  --resume_lr_scale 0.1
)

for variable_name in \
    BACKBONE_RESUME_CHECKPOINT BACKBONE_RESUME_SHA256 BACKBONE_RESUME_EPOCH; do
  if [[ -n "${!variable_name:-}" ]]; then
    case "${variable_name}" in
      BACKBONE_RESUME_CHECKPOINT) required_value="${REQUIRED_RESUME_CHECKPOINT}" ;;
      BACKBONE_RESUME_SHA256) required_value="${REQUIRED_RESUME_SHA256}" ;;
      BACKBONE_RESUME_EPOCH) required_value="${REQUIRED_RESUME_EPOCH}" ;;
    esac
    [[ "${!variable_name}" == "${required_value}" ]] || {
      echo "${variable_name} conflicts with the pinned E59 resume" >&2
      exit 2
    }
  fi
done
export BACKBONE_RESUME_CHECKPOINT="${REQUIRED_RESUME_CHECKPOINT}"
export BACKBONE_RESUME_SHA256="${REQUIRED_RESUME_SHA256}"
export BACKBONE_RESUME_EPOCH="${REQUIRED_RESUME_EPOCH}"
export VALIDATE_BACKBONE_RESUME=0
export MODE="${MODE:-backbone}"
case "${MODE}" in
  preflight|backbone) ;;
  *) echo "MODE must be preflight or backbone" >&2; exit 2 ;;
esac

require_sha() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; exit 3; }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA-256 changed: ${actual}" >&2
    exit 3
  }
}

require_sha "${ROOT_DIR}/main_utils.py" "${REQUIRED_MAIN_UTILS_SHA256}" "resume-LR implementation"
require_sha "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh" "${REQUIRED_PIPELINE_SHA256}" "shared launcher"
require_sha "${SOURCE_CHECKPOINT}" "${SOURCE_SHA256}" "GroupFree checkpoint"
require_sha "${PROTECTED_E57}" "${PROTECTED_E57_SHA256}" "protected E57 checkpoint"
require_sha "${E58_RECEIPT}" "${E58_RECEIPT_SHA256}" "E58 formal receipt"
require_sha "${E59_RECEIPT}" "${E59_RECEIPT_SHA256}" "E59 formal receipt"
require_sha "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" "E59 resume checkpoint"

PROOF_E58_RECEIPT="${E58_RECEIPT}" \
PROOF_E59_RECEIPT="${E59_RECEIPT}" \
PROOF_CHECKPOINT="${REQUIRED_RESUME_CHECKPOINT}" \
PROOF_CHECKPOINT_SHA256="${REQUIRED_RESUME_SHA256}" \
"${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import math
import os

import torch

sample_count = 7899
baseline_hits = 4463
expected = ((58, 4262, 3513), (59, 4283, 3513))
for (epoch, expected_hits025, expected_hits050), env_name in zip(
        expected, ("PROOF_E58_RECEIPT", "PROOF_E59_RECEIPT")):
    with open(os.environ[env_name], "rb") as handle:
        raw = handle.read()
    payload = json.loads(raw.decode("utf-8"))
    metric = payload["position_subgroups"]["multiple"]
    hits025 = int(metric["hits025"])
    hits050 = int(metric["hits050"])
    if payload.get("schema") != "mcln-retrain-metrics-v1":
        raise SystemExit("receipt schema mismatch at E{}".format(epoch))
    if int(payload["sample_count"]) != sample_count:
        raise SystemExit("top-level sample count mismatch at E{}".format(epoch))
    if int(metric["sample_count"]) != sample_count:
        raise SystemExit("formal multiple sample count mismatch at E{}".format(epoch))
    if hits025 != expected_hits025 or hits050 != expected_hits050:
        raise SystemExit("formal REC hits mismatch at E{}".format(epoch))
    if hits025 > baseline_hits:
        raise SystemExit("E{} refreshed the protected E57 baseline".format(epoch))
    if not math.isclose(
            float(metric["acc025"]), hits025 / sample_count,
            rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("REC@0.25 hits/accuracy mismatch at E{}".format(epoch))
    if not math.isclose(
            float(metric["acc050"]), hits050 / sample_count,
            rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("REC@0.50 hits/accuracy mismatch at E{}".format(epoch))

checkpoint_path = os.environ["PROOF_CHECKPOINT"]
with open(checkpoint_path, "rb") as handle:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    if digest.hexdigest() != os.environ["PROOF_CHECKPOINT_SHA256"]:
        raise SystemExit("E59 checkpoint SHA mismatch")
    handle.seek(0)
    checkpoint = torch.load(handle, map_location="cpu")
    handle.seek(0)
    digest_after = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest_after.update(chunk)
    if digest_after.hexdigest() != digest.hexdigest():
        raise SystemExit("E59 checkpoint changed while loading")

config = checkpoint.get("config")
config = vars(config) if hasattr(config, "__dict__") else config
optimizer = checkpoint.get("optimizer", {})
scheduler = checkpoint.get("scheduler", {})
groups = optimizer.get("param_groups", [])
expected_lrs = [1e-4, 1e-3, 1e-4, 1.25e-4]
observed_lrs = [float(group["lr"]) for group in groups]
observed_initial_lrs = [float(group["initial_lr"]) for group in groups]
required_config = {
    "dataset": ["nr3d"],
    "test_dataset": "nr3d",
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "drop_incomplete_accumulation_group": False,
    "joint_det": True,
    "butd_cls": True,
    "butd_gt": False,
    "augment_det": False,
    "lr_scheduler": "step",
    "warmup_epoch": -1,
    "use_source_choice_selector": True,
    "eval_use_selector_choice_scores": True,
    "source_choice_selector_sources": "default,default_rank_blend_contrastive010",
    "source_choice_selector_default_source": "default",
    "source_choice_selector_hidden_dim": 288,
    "source_choice_selector_lr": 1.25e-4,
    "source_choice_selector_loss_weight": 0.5,
    "source_choice_selector_choice_target": "precision_gain_default_sourcewise_focal_bce",
    "source_choice_selector_min_iou_gap": 0.03,
    "resume_lr_scale_lineage": 1.0,
    "e57_lr_restore_lineage": True,
}
if int(checkpoint.get("epoch", -1)) != 59:
    raise SystemExit("resume checkpoint is not E59")
if not isinstance(config, dict):
    raise SystemExit("resume checkpoint config is missing")
for name, value in required_config.items():
    if config.get(name) != value:
        raise SystemExit(
            "checkpoint config mismatch {}: {!r} != {!r}".format(
                name, config.get(name), value
            )
        )
if len(groups) != 4 or len(optimizer.get("state", {})) != 716:
    raise SystemExit("E59 optimizer topology mismatch")
if observed_lrs != expected_lrs or observed_initial_lrs != expected_lrs:
    raise SystemExit("E59 optimizer learning-rate provenance mismatch")
if scheduler.get("base_lrs") != expected_lrs:
    raise SystemExit("E59 scheduler base LR mismatch")
if scheduler.get("_last_lr") != expected_lrs:
    raise SystemExit("E59 scheduler current LR mismatch")
if int(scheduler.get("last_epoch", -1)) != 165554:
    raise SystemExit("E59 scheduler progress mismatch")
if int(scheduler.get("_step_count", -1)) != 165555:
    raise SystemExit("E59 scheduler step-count mismatch")
if dict(scheduler.get("milestones", {})) != {423706: 1}:
    raise SystemExit("E59 scheduler milestone mismatch")
print(
    "plateau_proof=E58:4262,E59:4283,baseline:4463,patience:2 "
    "resume=E59 optimizer_states=716 lr_scale=0.1"
)
PY

if pgrep -f 'train_dist_mod.py.*nr3d_mcln_joint_butdcls_v99_e57_restore_initial_once_e58_e62_b16a1' >/dev/null; then
  echo "stopped high-LR parent experiment is still running" >&2
  exit 4
fi
pgrep -f '/root/mcln_official_rec_monitor.py nr3d ' >/dev/null || {
  echo "Nr3D official REC monitor is not running" >&2
  exit 4
}
free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
((gpu_used < 500)) || { echo "GPU0 is busy (${gpu_used} MiB)" >&2; exit 4; }
((free_gb >= MIN_FREE_GB)) || { echo "need at least ${MIN_FREE_GB} GiB free" >&2; exit 5; }
lock_file="${DATA_ROOT%/}/output/network_v99/single_gpu.lock"
mkdir -p "$(dirname "${lock_file}")"
exec 8>"${lock_file}"
flock -n 8 || { echo "another V99 job owns ${lock_file}" >&2; exit 6; }
flock -u 8
exec 8>&-
if [[ "${MODE}" == "backbone" ]]; then
  current_screen_name="${STY##*.}"
  [[ -n "${STY:-}" && "${current_screen_name}" == "${TRAIN_SCREEN_NAME}" ]] || {
    echo "formal launch must run inside screen ${TRAIN_SCREEN_NAME}" >&2
    exit 7
  }
fi

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
