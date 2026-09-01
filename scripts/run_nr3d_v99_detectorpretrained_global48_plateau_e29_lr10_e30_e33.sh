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
readonly PARENT_RUN="${OUTPUT_ROOT}/backbone/nr3d_mcln_joint_butdcls_v99_detectorpretrained_no_taskresume_global48_e1_e240_b16a3_w4p2_20260826_225839/nr3d/nr3d_mcln_joint_butdcls_v99_detectorpretrained_no_taskresume_global48_e1_e240_b16a3_w4p2/1787756323"
readonly REQUIRED_RESUME_CHECKPOINT="${PARENT_RUN}/ckpt_epoch_29.pth"
readonly REQUIRED_RESUME_SHA256="edf06345cd10b02322412c8565520c849067bf32f75862423cfd501169bceab9"
readonly REQUIRED_RESUME_EPOCH=29
readonly REQUIRED_MAIN_UTILS_SHA256="555b122fa44ab91d73113094a5220f768e2b39130d389f69a5d26262c1cd7f21"
readonly REQUIRED_PIPELINE_SHA256="264eabacb8c034ad51f4fc30ce33ef990408a19e68400069a74c575f58da31a9"
readonly EXP="nr3d_mcln_joint_butdcls_v99_detectorpretrained_global48_plateau_e29_lr10_e30_e33_b16a3_w4p2"
readonly BATCH_SIZE=16
readonly MAX_EPOCH=33
readonly EXPECTED_EVAL_SAMPLE_COUNT=7899
readonly MASTER_PORT=5299
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
  --warmup-epoch 0
)
BACKBONE_EXTRA_ARGS=(
  --print_freq 20
  --joint_det
  --butd_cls
  --gradient_accumulation_steps 3
  --drop_incomplete_accumulation_group
  --resume_lr_scale 0.1
)

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
require_sha "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" "E29 checkpoint"

readonly E27_RECEIPT="${PARENT_RUN}/eval_metrics_epoch_27.json"
readonly E28_RECEIPT="${PARENT_RUN}/eval_metrics_epoch_28.json"
readonly E29_RECEIPT="${PARENT_RUN}/eval_metrics_epoch_29.json"
require_sha "${E27_RECEIPT}" "f8da22e0f63096fd82e15ecb519b4ad19a82d77a0ebf60ac1b7bd5972754fc1d" "E27 receipt"
require_sha "${E28_RECEIPT}" "0fd51e5ae93cdd43199df248f389c6bbc82f83b1d14ea9d5eba3aba8a76adb20" "E28 receipt"
require_sha "${E29_RECEIPT}" "f71e423c4a2e378dd25388ba702cbb34d5bf1fd1280dea93e1f5d7fb14e3e7eb" "E29 receipt"

"${PYTHON_BIN}" - "${E27_RECEIPT}" "${E28_RECEIPT}" "${E29_RECEIPT}" <<'PY'
import json
import math
import sys

expected = ((27, 3991), (28, 3959), (29, 3931))
sample_count = 7899
observed = []
for (epoch, expected_hits), path in zip(expected, sys.argv[1:]):
    with open(path) as handle:
        payload = json.load(handle)
    metric = payload["position_subgroups"]["multiple"]
    hits = int(metric["hits025"])
    count = int(metric["sample_count"])
    acc = float(metric["acc025"])
    if payload.get("schema") != "mcln-retrain-metrics-v1":
        raise SystemExit("receipt schema mismatch at E{}".format(epoch))
    if count != sample_count or int(payload["sample_count"]) != sample_count:
        raise SystemExit("sample-count mismatch at E{}".format(epoch))
    if hits != expected_hits:
        raise SystemExit("hits mismatch at E{}".format(epoch))
    if not math.isclose(acc, hits / sample_count, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("hits/accuracy mismatch at E{}".format(epoch))
    observed.append(hits)
baseline = observed[0]
if not all(hits <= baseline for hits in observed[1:]):
    raise SystemExit("E28-E29 do not prove a two-epoch plateau after E27")
print("plateau_proof=E27:{} E28:{} E29:{} patience=2".format(*observed))
PY

"${PYTHON_BIN}" - "${REQUIRED_RESUME_CHECKPOINT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
raw_config = checkpoint["config"]
config = raw_config if isinstance(raw_config, dict) else vars(raw_config)
optimizer = checkpoint["optimizer"]
scheduler = checkpoint["scheduler"]
expected_lrs = [1e-4, 1e-3, 1e-4, 1.25e-4]
observed_lrs = [float(group["lr"]) for group in optimizer["param_groups"]]
if int(checkpoint["epoch"]) != 29:
    raise SystemExit("resume checkpoint epoch mismatch")
if len(optimizer["state"]) != 716 or len(observed_lrs) != 4:
    raise SystemExit("optimizer state is incomplete")
if observed_lrs != expected_lrs:
    raise SystemExit("unexpected pre-decay learning rates")
required = {
    "test_dataset": "nr3d",
    "batch_size": 16,
    "gradient_accumulation_steps": 3,
    "drop_incomplete_accumulation_group": True,
    "lr_scheduler": "step",
    "warmup_epoch": 0,
    "joint_det": True,
    "butd_cls": True,
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
}
for name, value in required.items():
    if config.get(name) != value:
        raise SystemExit("checkpoint config mismatch: {}".format(name))
if config.get("dataset") not in ("nr3d", ["nr3d"]):
    raise SystemExit("checkpoint dataset mismatch")
if int(scheduler["last_epoch"]) != 29 * 935:
    raise SystemExit("scheduler progress mismatch")
if dict(scheduler.get("milestones", {})) != {150 * 935: 1}:
    raise SystemExit("scheduler milestone mismatch")
if [float(value) for value in scheduler["_last_lr"]] != expected_lrs:
    raise SystemExit("scheduler current-LR mismatch")
print("checkpoint_proof=epoch29 optimizer_states716 scheduler_steps{} current_lrs={}".format(
    scheduler["last_epoch"], observed_lrs
))
PY

free_kb="$(df -Pk "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
echo "preflight_gpu0_memory_used_mib=${gpu_used}"
echo "preflight_free_disk_gib=${free_gb}"
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

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
