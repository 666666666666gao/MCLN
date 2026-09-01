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
readonly BASELINE_LEAF="${OUTPUT_ROOT}/backbone/nr3d_mcln_joint_butdcls_v99_e62_continue_e63_e65_b16a1_20260830_004448/nr3d/nr3d_mcln_joint_butdcls_v99_e62_continue_e63_e65_b16a1/1788021897"
readonly PARENT_LEAF="${OUTPUT_ROOT}/backbone/nr3d_mcln_joint_butdcls_v99_e65_continue_e66_e68_b16a1_20260830_061342/nr3d/nr3d_mcln_joint_butdcls_v99_e65_continue_e66_e68_b16a1/1788041631"
readonly E65_RECEIPT="${BASELINE_LEAF}/eval_metrics_epoch_65.json"
readonly E65_RECEIPT_SHA256="3bcbd1be37d74bb46378501c58e25433f04548b4b5cb0c948339912b289c5b93"
readonly E66_RECEIPT="${PARENT_LEAF}/eval_metrics_epoch_66.json"
readonly E66_RECEIPT_SHA256="f42511c757f35f87d443ffcec1e3db61382416ab225d5308386db8a796592fc5"
readonly E67_RECEIPT="${PARENT_LEAF}/eval_metrics_epoch_67.json"
readonly E67_RECEIPT_SHA256="186cdbda2904f85b3c22b1b0621078e3f4b2dad54f64c5000c2b23e45d0020aa"
readonly PLATEAU_STATE="${OUTPUT_ROOT}/control/e65_continue_e66_e68_patience2/guard_state.json"
readonly PLATEAU_STATE_SHA256="72eac9a4060310991fdf78c0ae10eec39b9b1adbdbd41ca6008c35f441c83fc2"
readonly REQUIRED_RESUME_CHECKPOINT="${PARENT_LEAF}/ckpt_epoch_67.pth"
readonly REQUIRED_RESUME_SHA256="10a619813f5b825c25daaa62cdef874b98ca808ad6d2b9d62f373954f320cbfd"
readonly REQUIRED_RESUME_EPOCH=67
readonly PROTECTED_E57="${OUTPUT_ROOT}/control/official_rec_monitor/official_best_rec025_epoch_57_0p56500823.pth"
readonly PROTECTED_E57_SHA256="fe1e2047b3c4d5ed0aae3569418abff5a65f5608edcb7f34d01ffab1ee1f6655"
readonly REQUIRED_MAIN_UTILS_SHA256="180078267ab1fb2f74bbbfeb5e8fb9ae409ecd69b0bfa05d4aed6af0bd703eab"
readonly REQUIRED_PIPELINE_SHA256="264eabacb8c034ad51f4fc30ce33ef990408a19e68400069a74c575f58da31a9"
readonly GUARD_SCRIPT="/root/mcln_nr3d_e68_e70_patience_guard.py"
readonly GUARD_SCRIPT_SHA256="d84532ed6efed71c0c929df90c88464e034e6c26164a301babad2937c007b30a"
readonly EXP="nr3d_mcln_joint_butdcls_v99_e67_plateau_lr10_e68_e70_b16a1"
readonly TRAIN_SCREEN_NAME="mcln_nr3d_e67_lr10"
readonly BATCH_SIZE=16
readonly MAX_EPOCH=70
readonly EXPECTED_EVAL_SAMPLE_COUNT=7899
readonly MASTER_PORT=5326
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
  --resume_lr_scale_expected_lineage 0.1
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
      echo "${variable_name} conflicts with the pinned E67 resume" >&2
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

require_sha "${ROOT_DIR}/main_utils.py" "${REQUIRED_MAIN_UTILS_SHA256}" "resume implementation"
require_sha "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh" "${REQUIRED_PIPELINE_SHA256}" "shared launcher"
require_sha "${GUARD_SCRIPT}" "${GUARD_SCRIPT_SHA256}" "patience guard"
require_sha "${SOURCE_CHECKPOINT}" "${SOURCE_SHA256}" "GroupFree checkpoint"
require_sha "${PROTECTED_E57}" "${PROTECTED_E57_SHA256}" "protected E57 checkpoint"
require_sha "${E65_RECEIPT}" "${E65_RECEIPT_SHA256}" "E65 formal receipt"
require_sha "${E66_RECEIPT}" "${E66_RECEIPT_SHA256}" "E66 formal receipt"
require_sha "${E67_RECEIPT}" "${E67_RECEIPT_SHA256}" "E67 formal receipt"
require_sha "${PLATEAU_STATE}" "${PLATEAU_STATE_SHA256}" "patience2 decision"
require_sha "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" "E67 resume checkpoint"

PROOF_E65_RECEIPT="${E65_RECEIPT}" \
PROOF_E66_RECEIPT="${E66_RECEIPT}" \
PROOF_E67_RECEIPT="${E67_RECEIPT}" \
PROOF_PLATEAU_STATE="${PLATEAU_STATE}" \
PROOF_CHECKPOINT="${REQUIRED_RESUME_CHECKPOINT}" \
PROOF_CHECKPOINT_SHA256="${REQUIRED_RESUME_SHA256}" \
"${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import math
import os

import torch

sample_count = 7899
branch_best = 4405
protected_hits = 4463
target_hits = 4724
expected = ((65, 4405, 3697), (66, 4382, 3731), (67, 4400, 3719))
for (epoch, expected_hits025, expected_hits050), env_name in zip(
        expected,
        ("PROOF_E65_RECEIPT", "PROOF_E66_RECEIPT", "PROOF_E67_RECEIPT")):
    with open(os.environ[env_name], "rb") as handle:
        payload = json.loads(handle.read().decode("utf-8"))
    metric = payload["position_subgroups"]["multiple"]
    hits025 = int(metric["hits025"])
    hits050 = int(metric["hits050"])
    if payload.get("schema") != "mcln-retrain-metrics-v1":
        raise SystemExit("receipt schema mismatch at E{}".format(epoch))
    if int(payload.get("sample_count", -1)) != sample_count:
        raise SystemExit("top-level sample count mismatch at E{}".format(epoch))
    if int(metric.get("sample_count", -1)) != sample_count:
        raise SystemExit("formal multiple count mismatch at E{}".format(epoch))
    if (hits025, hits050) != (expected_hits025, expected_hits050):
        raise SystemExit("formal REC hits mismatch at E{}".format(epoch))
    if not math.isclose(float(metric["acc025"]), hits025 / sample_count,
                        rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("REC@0.25 hits/accuracy mismatch at E{}".format(epoch))
    if not math.isclose(float(metric["acc050"]), hits050 / sample_count,
                        rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("REC@0.50 hits/accuracy mismatch at E{}".format(epoch))
if expected[0][1] != branch_best:
    raise SystemExit("E65 branch baseline changed")
if any(row[1] > branch_best for row in expected[1:]):
    raise SystemExit("E66/E67 no longer prove a patience2 plateau")
if branch_best >= protected_hits or protected_hits >= target_hits:
    raise SystemExit("registered REC thresholds changed")

with open(os.environ["PROOF_PLATEAU_STATE"], "rb") as handle:
    state = json.loads(handle.read().decode("utf-8"))
if (
        state.get("schema") != "mcln-nr3d-e66-e68-patience-decision-v1"
        or state.get("status") != "complete"
        or state.get("terminal_reason") != "patience2_plateau"
        or int(state.get("baseline_hits025", -1)) != branch_best
        or int(state.get("best_hits025", -1)) != branch_best
        or int(state.get("consecutive_non_improvements", -1)) != 2
        or [int(row["epoch"]) for row in state.get("rows", [])] != [66, 67]
        or [int(row["hits025"]) for row in state.get("rows", [])] != [4382, 4400]):
    raise SystemExit("formal guard state does not prove the E66/E67 plateau")

checkpoint_path = os.environ["PROOF_CHECKPOINT"]
with open(checkpoint_path, "rb") as handle:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    if digest.hexdigest() != os.environ["PROOF_CHECKPOINT_SHA256"]:
        raise SystemExit("E67 checkpoint SHA mismatch")
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
        raise SystemExit("E67 checkpoint changed while loading")

config = checkpoint.get("config")
config = vars(config) if hasattr(config, "__dict__") else config
optimizer = checkpoint.get("optimizer", {})
scheduler = checkpoint.get("scheduler", {})
groups = optimizer.get("param_groups", [])
expected_current_lrs = [1e-5, 1e-4, 1e-5, 1.25e-5]
expected_initial_lrs = [1e-4, 1e-3, 1e-4, 1.25e-4]
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
    "resume_lr_scale_lineage": 0.1,
    "e57_lr_restore_lineage": True,
    "checkpoint_metric_retention": True,
    "checkpoint_retention_metrics": ["rec_acc025"],
}
if int(checkpoint.get("epoch", -1)) != 67:
    raise SystemExit("resume checkpoint is not E67")
if not isinstance(config, dict):
    raise SystemExit("resume checkpoint config is missing")
for name, value in required_config.items():
    if config.get(name) != value:
        raise SystemExit("checkpoint config mismatch {}: {!r} != {!r}".format(
            name, config.get(name), value))
if len(groups) != 4 or len(optimizer.get("state", {})) != 716:
    raise SystemExit("E67 optimizer topology mismatch")
if [float(group["lr"]) for group in groups] != expected_current_lrs:
    raise SystemExit("E67 current LR mismatch")
if [float(group["initial_lr"]) for group in groups] != expected_initial_lrs:
    raise SystemExit("E67 initial LR mismatch")
if scheduler.get("base_lrs") != expected_initial_lrs:
    raise SystemExit("E67 scheduler base LR mismatch")
if scheduler.get("_last_lr") != expected_current_lrs:
    raise SystemExit("E67 scheduler current LR mismatch")
if int(scheduler.get("last_epoch", -1)) != 188002:
    raise SystemExit("E67 scheduler progress mismatch")
if int(scheduler.get("_step_count", -1)) != 188003:
    raise SystemExit("E67 scheduler step-count mismatch")
if dict(scheduler.get("milestones", {})) != {423706: 1}:
    raise SystemExit("E67 scheduler milestone mismatch")
print("plateau_proof=E65:4405,E66:4382,E67:4400 patience:2")
print("resume_proof=E67 lineage:0.1 current_lr:{} next_scale:0.1".format(
    expected_current_lrs))
PY

pgrep -f '/root/mcln_official_rec_monitor.py nr3d ' >/dev/null || {
  echo "Nr3D official REC monitor is not running" >&2
  exit 4
}
if pgrep -f '[t]rain_dist_mod.py.*nr3d_mcln_joint_butdcls' >/dev/null; then
  echo "another Nr3D training process is still running" >&2
  exit 4
fi
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
