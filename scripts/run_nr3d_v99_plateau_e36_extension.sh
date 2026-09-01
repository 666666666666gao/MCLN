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
readonly PARENT_RUN="${OUTPUT_ROOT}/backbone/nr3d_mcln_joint_butdcls_v99_plateau_e31_lr10_e32_e36_b16a3_w4p2_20260826_020312/nr3d/nr3d_mcln_joint_butdcls_v99_plateau_e31_lr10_e32_e36_b16a3_w4p2/1787681000"
readonly REQUIRED_RESUME_CHECKPOINT="${PARENT_RUN}/ckpt_epoch_36.pth"
readonly REQUIRED_RESUME_SHA256="943ebef66971777592f49ef1eb4594518f5ec61fd30dfb924e086cc29f243e01"
readonly REQUIRED_RESUME_EPOCH=36
readonly REQUIRED_E36_RECEIPT="${PARENT_RUN}/eval_metrics_epoch_36.json"
readonly REQUIRED_E36_RECEIPT_SHA256="079dda745c7974abd56edfcab70c1d7582b0443e997aed2c4801eb09e71497ea"
readonly REQUIRED_E35_RECEIPT="${PARENT_RUN}/eval_metrics_epoch_35.json"
readonly REQUIRED_E35_RECEIPT_SHA256="8223d67d4b729f806b48757f4846cc3de7ca54576c1e9c7e34a9de38745cf5f4"
readonly REQUIRED_MAIN_UTILS_SHA256="f0ff9c2bcde8d39e516092b63580fbdd494c9bc48a487d0c561f5ede8bdfe4b9"
readonly REQUIRED_PIPELINE_SHA256="264eabacb8c034ad51f4fc30ce33ef990408a19e68400069a74c575f58da31a9"
readonly EXP="nr3d_mcln_joint_butdcls_v99_plateau_e36_extension_e37_e45_b16a3_w4p2"
readonly BATCH_SIZE=16
readonly MAX_EPOCH=45
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
  --resume_lr_scale 1.0
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

if [[ "${REQUIRED_RESUME_SHA256}" == PENDING_* ||
      "${REQUIRED_E36_RECEIPT_SHA256}" == PENDING_* ]]; then
  echo "E36 checkpoint/receipt SHA-256 is not pinned; refusing launch" >&2
  exit 2
fi

require_sha() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; exit 3; }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA-256 changed: ${actual}" >&2
    exit 3
  }
}

require_sha "${ROOT_DIR}/main_utils.py" "${REQUIRED_MAIN_UTILS_SHA256}" "main_utils.py"
require_sha "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh" "${REQUIRED_PIPELINE_SHA256}" "shared pipeline"
require_sha "${REQUIRED_E35_RECEIPT}" "${REQUIRED_E35_RECEIPT_SHA256}" "E35 receipt"
require_sha "${REQUIRED_E36_RECEIPT}" "${REQUIRED_E36_RECEIPT_SHA256}" "E36 receipt"
require_sha "${REQUIRED_RESUME_CHECKPOINT}" "${REQUIRED_RESUME_SHA256}" "E36 checkpoint"

"${PYTHON_BIN}" - "${REQUIRED_E35_RECEIPT}" "${REQUIRED_E36_RECEIPT}" "${REQUIRED_RESUME_CHECKPOINT}" <<'PY'
import json
import math
import sys
import torch

e35_path, e36_path, checkpoint_path = sys.argv[1:]
for epoch, path in ((35, e35_path), (36, e36_path)):
    with open(path) as handle:
        payload = json.load(handle)
    metric = payload.get("position_subgroups", {}).get("multiple", {})
    if payload.get("schema") != "mcln-retrain-metrics-v1":
        raise SystemExit("E{} receipt schema mismatch".format(epoch))
    if payload.get("sample_count") != 7899 or metric.get("sample_count") != 7899:
        raise SystemExit("E{} sample-count mismatch".format(epoch))
    hits = int(metric.get("hits025"))
    if not math.isclose(float(metric.get("acc025")), hits / 7899.0,
                        rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("E{} hits/accuracy mismatch".format(epoch))
    if epoch == 35 and hits != 4269:
        raise SystemExit("E35 hits mismatch")
    if epoch == 36 and hits >= 4724:
        raise SystemExit("Nr3D strict target is already achieved; refusing extension")

checkpoint = torch.load(checkpoint_path, map_location="cpu")
config = vars(checkpoint["config"])
optimizer = checkpoint["optimizer"]
scheduler = checkpoint["scheduler"]
current_lrs = [float(group["lr"]) for group in optimizer["param_groups"]]
expected_current_lrs = [1e-5, 1e-4, 1e-5, 1.25e-5]
if int(checkpoint["epoch"]) != 36:
    raise SystemExit("checkpoint epoch mismatch")
if len(optimizer["state"]) != 716 or len(current_lrs) != 4:
    raise SystemExit("optimizer state is incomplete")
if current_lrs != expected_current_lrs:
    raise SystemExit("E36 current LR mismatch")
required = {
    "batch_size": 16,
    "gradient_accumulation_steps": 3,
    "drop_incomplete_accumulation_group": True,
    "lr_scheduler": "step",
    "warmup_epoch": 0,
    "joint_det": True,
    "butd_cls": True,
    "use_source_choice_selector": True,
    "resume_lr_scale_lineage": 0.1,
}
for name, value in required.items():
    if config.get(name) != value:
        raise SystemExit("checkpoint config mismatch: {}={!r}".format(name, config.get(name)))
if int(scheduler["last_epoch"]) != 36 * 935:
    raise SystemExit("scheduler progress mismatch")
if [float(value) for value in scheduler["_last_lr"]] != expected_current_lrs:
    raise SystemExit("scheduler current LR mismatch")
print("NR3D_E36_EXTENSION_PREFLIGHT=PASS epoch=36 current_lrs={} scheduler_steps={}".format(
    current_lrs, scheduler["last_epoch"]))
PY

# shellcheck source=run_dataset_v99_pipeline.sh
source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
