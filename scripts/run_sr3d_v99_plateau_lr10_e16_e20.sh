#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
readonly DATA_ROOT="/root/autodl-tmp/DATA_ROOT/"
readonly PYTHON_BIN="/root/miniconda3/envs/bdetr/bin/python"
readonly SOURCE_CHECKPOINT="${DATA_ROOT%/}/gf_detector_l6o256.pth"
readonly SOURCE_SHA256="9ff3e25070bf48a0b70240e098a89be6bfd26a92af863e71698a0737fc6e54f2"
readonly DATASET="sr3d"
readonly OUTPUT_ROOT="${DATA_ROOT%/}/output/network_v99_baseline_gt/sr3d"
readonly EXP="sr3d_mcln_joint_butdcls_v99_plateau_lr10_e16_e20_b12a2_w4p2"
readonly BASE_RUN="${OUTPUT_ROOT}/backbone/sr3d_mcln_joint_butdcls_v99_detectorpretrained_no_taskresume_global24_e1_e140_b12a2_w4p2_20260823_173119/sr3d/sr3d_mcln_joint_butdcls_v99_detectorpretrained_no_taskresume_global24_e1_e140_b12a2_w4p2/1787477483"
BACKBONE_RESUME_CHECKPOINT="${BASE_RUN}/ckpt_epoch_15.pth"
BACKBONE_RESUME_SHA256="5c5921939b2476925d4d3afa2ac9eaa2cc706e28bb967c5fa1db9075b15999f4"
BACKBONE_RESUME_EPOCH=15
readonly E12_RECEIPT="${BASE_RUN}/eval_metrics_epoch_12.json"
readonly E13_RECEIPT="${BASE_RUN}/eval_metrics_epoch_13.json"
readonly E14_RECEIPT="${BASE_RUN}/eval_metrics_epoch_14.json"
readonly E15_RECEIPT="${BASE_RUN}/eval_metrics_epoch_15.json"
readonly E12_SHA256="c3a958cfd2667c12a4e50ea9d511e506d2a8ddb6eb97b54e8124dbed658fda8c"
readonly E13_SHA256="8116e975b7514cdf687250c32771a20221cd34dfd310957496aac6b1c2c5ddb7"
readonly E14_SHA256="3e7a3cb3a54c7ce2ed0c1a1a872aa4ab177a627320e5f918b8683c0c2fd36811"
readonly E15_SHA256="2b436a0c90a0443ecb4b95596035725ade29c90c9172622373e9f48ffe4a0d93"
readonly EXPECTED_MAIN_UTILS_SHA256="6592fa938680240cd75dd181cc1e63cc0624714d78c7d92bbba8ae6f8622a850"
readonly EXPECTED_TEST_SHA256="4645376bcb66c924d57df0681a1c1d98849d58492000b381af4f79ebbdf27ad5"
readonly EXPECTED_PIPELINE_SHA256="d6a11499ff250b1c3c8b7a338e0e09556423742bb4592f10063dcb4f81db6038"
readonly BATCH_SIZE=12
readonly MAX_EPOCH=20
readonly EXPECTED_EVAL_SAMPLE_COUNT=17726
readonly MASTER_PORT=5499
readonly MIN_FREE_GB=7
readonly EXPECTED_TRAIN_DATASET_SIZE=77836
readonly EXPECTED_TRAIN_LOADER_BATCH_COUNT=6486
readonly EXPECTED_EFFECTIVE_TRAIN_BATCH_COUNT=6486
readonly EXPECTED_DROPPED_TRAIN_BATCH_COUNT=0
readonly EXPECTED_OPTIMIZER_STEPS_PER_EPOCH=3243
readonly RESUME_LR_SCALE=0.1
readonly BACKBONE_JOINT_TRAINING=1
readonly INFERENCE_USES_GROUND_TRUTH=1
readonly USE_BACKBONE_INITIALIZATION=0
readonly TASK_CHECKPOINT_TRANSFER=0
readonly BACKBONE_AUGMENT_DET=0
VALIDATE_BACKBONE_RESUME=0

CHECKPOINT_RETENTION_METRICS=(rec_acc025)
DATASET_LR_ARGS=(
  --lr_backbone 1e-3 --lr 1e-4
  --lr_decay_epochs 30 40
  --warmup-epoch 0
)
BACKBONE_EXTRA_ARGS=(
  --print_freq 20
  --joint_det
  --butd_cls
  --gradient_accumulation_steps 2
  --drop_incomplete_accumulation_group
  --expected_train_dataset_size "${EXPECTED_TRAIN_DATASET_SIZE}"
  --expected_train_loader_batch_count "${EXPECTED_TRAIN_LOADER_BATCH_COUNT}"
  --expected_effective_train_batch_count "${EXPECTED_EFFECTIVE_TRAIN_BATCH_COUNT}"
  --expected_dropped_train_batch_count "${EXPECTED_DROPPED_TRAIN_BATCH_COUNT}"
  --expected_optimizer_steps_per_epoch "${EXPECTED_OPTIMIZER_STEPS_PER_EPOCH}"
  --resume_lr_scale "${RESUME_LR_SCALE}"
)

export BACKBONE_RESUME_CHECKPOINT BACKBONE_RESUME_SHA256 BACKBONE_RESUME_EPOCH
export VALIDATE_BACKBONE_RESUME
export MODE="${MODE:-backbone}"
case "${MODE}" in
  preflight|backbone) ;;
  *) echo "MODE must be preflight or backbone" >&2; exit 2 ;;
esac

require_sha256() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" ]] || { echo "missing ${label}: ${path}" >&2; exit 3; }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA-256 changed: ${actual}" >&2
    exit 3
  }
}
require_sha256 "${ROOT_DIR}/main_utils.py" "${EXPECTED_MAIN_UTILS_SHA256}" "main_utils.py"
require_sha256 "${ROOT_DIR}/tests/test_main_utils_source_choice_checkpoint.py" "${EXPECTED_TEST_SHA256}" "LR-scale tests"
require_sha256 "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh" "${EXPECTED_PIPELINE_SHA256}" "shared pipeline"
require_sha256 "${BACKBONE_RESUME_CHECKPOINT}" "${BACKBONE_RESUME_SHA256}" "E15 checkpoint"
require_sha256 "${E12_RECEIPT}" "${E12_SHA256}" "E12 receipt"
require_sha256 "${E13_RECEIPT}" "${E13_SHA256}" "E13 receipt"
require_sha256 "${E14_RECEIPT}" "${E14_SHA256}" "E14 receipt"
require_sha256 "${E15_RECEIPT}" "${E15_SHA256}" "E15 receipt"

"${PYTHON_BIN}" - \
  "${ROOT_DIR}" "${BACKBONE_RESUME_CHECKPOINT}" \
  "${E12_RECEIPT}" "${E13_RECEIPT}" "${E14_RECEIPT}" "${E15_RECEIPT}" <<'PY'
import json
import math
import sys
from collections import Counter
import torch

root, checkpoint_path = sys.argv[1:3]
receipt_paths = sys.argv[3:]
sys.path.insert(0, root)
import main_utils

expected_hits = [11680, 11621, 11580, 11522]
for epoch, path, hits in zip(range(12, 16), receipt_paths, expected_hits):
    with open(path) as handle:
        receipt = json.load(handle)
    multiple = receipt.get("position_subgroups", {}).get("multiple", {})
    if receipt.get("schema") != "mcln-retrain-metrics-v1":
        raise SystemExit("E{} receipt schema mismatch".format(epoch))
    if receipt.get("sample_count") != 17726:
        raise SystemExit("E{} top-level sample count mismatch".format(epoch))
    if multiple.get("sample_count") != 17726:
        raise SystemExit("E{} multiple sample count mismatch".format(epoch))
    if multiple.get("hits025") != hits:
        raise SystemExit("E{} hits025 mismatch".format(epoch))
    if not math.isclose(
            float(multiple.get("acc025")), float(hits) / 17726.0,
            rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("E{} hits/accuracy mismatch".format(epoch))
if not all(a > b for a, b in zip(expected_hits, expected_hits[1:])):
    raise SystemExit("formal E12-E15 plateau proof is not strictly declining")

checkpoint = torch.load(checkpoint_path, map_location="cpu")
config = checkpoint.get("config")
get = ((lambda name, default=None: config.get(name, default))
       if isinstance(config, dict)
       else (lambda name, default=None: getattr(config, name, default)))
if checkpoint.get("epoch") != 15:
    raise SystemExit("checkpoint epoch mismatch")
expected_config = {
    "dataset": ["sr3d"], "test_dataset": "sr3d",
    "joint_det": True, "butd_cls": True, "batch_size": 12,
    "gradient_accumulation_steps": 2, "lr_scheduler": "step",
    "warmup_epoch": 0, "lr_decay_epochs": [30, 40],
}
for name, expected in expected_config.items():
    if get(name) != expected:
        raise SystemExit("{} mismatch: {!r}".format(name, get(name)))
if get("resume_lr_scale", 1.0) not in (None, 1.0):
    raise SystemExit("checkpoint already records a manual LR scale")
if get("resume_lr_scale_lineage", 1.0) not in (None, 1.0):
    raise SystemExit("checkpoint already records LR-scale lineage")

optimizer = checkpoint.get("optimizer", {})
groups = optimizer.get("param_groups", [])
if len(groups) != 4 or len(optimizer.get("state", {})) != 716:
    raise SystemExit("E15 AdamW state contract mismatch")
expected_lrs = [1e-4, 1e-3, 1e-4, 1.25e-4]
if [group.get("lr") for group in groups] != expected_lrs:
    raise SystemExit("E15 current LR mismatch")
if [group.get("initial_lr") for group in groups] != expected_lrs:
    raise SystemExit("E15 initial LR mismatch")

scheduler = checkpoint.get("scheduler", {})
if scheduler.get("last_epoch") != 48645:
    raise SystemExit("E15 scheduler last_epoch mismatch")
if scheduler.get("_step_count") != 48646:
    raise SystemExit("E15 scheduler step count mismatch")
if scheduler.get("_last_lr") != expected_lrs:
    raise SystemExit("E15 scheduler current LR mismatch")
if scheduler.get("base_lrs") != expected_lrs:
    raise SystemExit("E15 scheduler base LR mismatch")
if scheduler.get("milestones") != Counter({97290: 1, 129720: 1}):
    raise SystemExit("E15 scheduler milestones mismatch")

plan = main_utils._gradient_accumulation_plan(
    loader_batch_count=6486, max_train_batches=0,
    accumulation_steps=2, drop_incomplete_accumulation_group=True,
)
expected_plan = {
    "requested_batch_count": 6486, "effective_batch_count": 6486,
    "dropped_batch_count": 0, "optimizer_step_count": 3243,
}
if plan != expected_plan:
    raise SystemExit("global24 accumulation plan mismatch")
target_lrs = [1e-5, 1e-4, 1e-5, 1.25e-5]
if [value * 0.1 for value in expected_lrs] != target_lrs:
    raise SystemExit("target LR proof mismatch")
print("SR3D_E15_PLATEAU_LR10_PREFLIGHT=PASS hits={} target_lrs={} milestones={}".format(
    expected_hits, target_lrs, sorted(scheduler["milestones"].elements())
))
PY

source "${ROOT_DIR}/scripts/run_dataset_v99_pipeline.sh"
