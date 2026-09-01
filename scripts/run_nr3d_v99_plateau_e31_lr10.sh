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
readonly PARENT_RUN="${OUTPUT_ROOT}/backbone/nr3d_mcln_joint_butdcls_v99_detectorpretrained_no_taskresume_global48_e1_e240_b16a3_w4p2_20260823_131051/nr3d/nr3d_mcln_joint_butdcls_v99_detectorpretrained_no_taskresume_global48_e1_e240_b16a3_w4p2/1787461856"
readonly REQUIRED_RESUME_CHECKPOINT="${PARENT_RUN}/ckpt_epoch_31.pth"
readonly REQUIRED_RESUME_SHA256="51b2ba1a23924762fa581460c38d7778fe027f2c289686d3c43fa45bc8bf713e"
readonly REQUIRED_RESUME_EPOCH=31
readonly REQUIRED_MAIN_UTILS_SHA256="f0ff9c2bcde8d39e516092b63580fbdd494c9bc48a487d0c561f5ede8bdfe4b9"
readonly REQUIRED_PIPELINE_SHA256="264eabacb8c034ad51f4fc30ce33ef990408a19e68400069a74c575f58da31a9"
readonly EXP="nr3d_mcln_joint_butdcls_v99_plateau_e31_lr10_e32_e36_b16a3_w4p2"
readonly BATCH_SIZE=16
readonly MAX_EPOCH=36
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
require_sha "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" "E31 checkpoint"

readonly E28_RECEIPT="${PARENT_RUN}/eval_metrics_epoch_28.json"
readonly E29_RECEIPT="${PARENT_RUN}/eval_metrics_epoch_29.json"
readonly E30_RECEIPT="${PARENT_RUN}/eval_metrics_epoch_30.json"
readonly E31_RECEIPT="${PARENT_RUN}/eval_metrics_epoch_31.json"
require_sha "${E28_RECEIPT}" "0119b2ee5807815d83b2620e768796c7a73d172c066f856a4998a73efc9cc63e" "E28 receipt"
require_sha "${E29_RECEIPT}" "5b3c0612fcf36a4d7cab788b2072726c7e98dff485905178757ab10956a99813" "E29 receipt"
require_sha "${E30_RECEIPT}" "ce8cf972f7cefad0fff1f7e4f153d6b8c31c9cccaaf41241ee49f7b1753cfc64" "E30 receipt"
require_sha "${E31_RECEIPT}" "66441b59546e9b6caf7454bda4fd3e723b02e57407118d78db715e8cad277ca2" "E31 receipt"

"${PYTHON_BIN}" - "${E28_RECEIPT}" "${E29_RECEIPT}" "${E30_RECEIPT}" "${E31_RECEIPT}" <<'PY'
import json
import math
import sys

expected = ((28, 4160), (29, 4083), (30, 4105), (31, 4115))
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
if any(hits > baseline for hits in observed[1:]):
    raise SystemExit("E29-E31 do not prove a three-epoch plateau")
print("plateau_proof=E28:{} E29:{} E30:{} E31:{} patience=3".format(*observed))
PY

"${PYTHON_BIN}" - "${REQUIRED_RESUME_CHECKPOINT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
config = vars(checkpoint["config"])
optimizer = checkpoint["optimizer"]
scheduler = checkpoint["scheduler"]
expected_lrs = [1e-4, 1e-3, 1e-4, 1.25e-4]
observed_lrs = [float(group["lr"]) for group in optimizer["param_groups"]]
if int(checkpoint["epoch"]) != 31:
    raise SystemExit("resume checkpoint epoch mismatch")
if len(optimizer["state"]) != 716 or len(observed_lrs) != 4:
    raise SystemExit("optimizer state is incomplete")
if observed_lrs != expected_lrs:
    raise SystemExit("unexpected pre-decay learning rates")
required = {
    "batch_size": 16,
    "gradient_accumulation_steps": 3,
    "drop_incomplete_accumulation_group": True,
    "lr_scheduler": "step",
    "warmup_epoch": 0,
    "joint_det": True,
    "butd_cls": True,
    "use_source_choice_selector": True,
    "resume_lr_scale_lineage": 1.0,
}
for name, value in required.items():
    if config.get(name) != value:
        raise SystemExit("checkpoint config mismatch: {}".format(name))
if int(scheduler["last_epoch"]) != 31 * 935:
    raise SystemExit("scheduler progress mismatch")
print("checkpoint_proof=epoch31 optimizer_states716 scheduler_steps{} current_lrs={}".format(
    scheduler["last_epoch"], observed_lrs
))
PY

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
